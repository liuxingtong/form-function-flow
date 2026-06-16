"""
1) Export site.dxf + district.dxf to GeoJSON per layer (local CAD coordinates).
2) Fit a 2D similarity transform (rotation + uniform scale + translation) from CAD to UTM
   using iterative closest-point refinement against data/SITE.json projected to UTM
   (EPSG:32651), then export WGS84 GeoJSON.
3) Write WGS84 GeoJSON copies under .../crs84/ for site + district exports.

Requires: ezdxf, numpy, pyproj, shapely
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import ezdxf
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point as ShyPoint
from shapely.ops import nearest_points

from export_dxf_layers_to_geojson import export_dxf_by_layer


def _as_closed_vertex_ring(coords: list[list[float]]) -> np.ndarray:
    """coords: GeoJSON-style ring [[lon,lat],...] possibly closed."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Expected Nx2+ coordinate array")
    if len(arr) >= 2 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    if len(arr) < 3:
        raise ValueError("Need at least 3 unique ring vertices")
    return arr[:, :2]


def _dxf_site_ring(dxf_path: Path, layer: str = "SITE") -> np.ndarray:
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    for e in msp:
        if str(e.dxf.layer).upper() != layer.upper():
            continue
        if e.dxftype() != "POLYLINE":
            continue
        verts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in e.vertices]
        if len(verts) < 3:
            continue
        return np.asarray(verts, dtype=float)
    raise RuntimeError(f"No closed POLYLINE on layer {layer!r} in {dxf_path}")


def _signed_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _maybe_reverse_ring(pts: np.ndarray, *, target_positive: bool) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    a = _signed_area(pts)
    if a == 0:
        return pts
    if (a > 0) != target_positive:
        return pts[::-1].copy()
    return pts


def _resample_closed_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """Evenly-spaced samples along closed polyline (including closing segment)."""
    if n < 3:
        raise ValueError("n must be >= 3")
    pts = np.asarray(pts, dtype=float)
    ring = np.vstack([pts, pts[0:1]])
    seg_len = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 0:
        raise ValueError("Degenerate perimeter")
    targets = np.linspace(0.0, total, n, endpoint=False)

    out = np.zeros((n, 2), dtype=float)
    j = 0
    for i, t in enumerate(targets):
        while j + 1 < len(cum) and cum[j + 1] < t - 1e-12:
            j += 1
        if j + 1 >= len(cum):
            j = len(cum) - 2
        a, b = cum[j], cum[j + 1]
        u = 0.0 if b - a < 1e-15 else (t - a) / (b - a)
        out[i] = (1.0 - u) * ring[j] + u * ring[j + 1]
    return out


    """
    src, dst: (N,2) row vectors.
    Returns (s, R, t_row) such that dst ~= s * src @ R.T + t_row
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src and dst must be (N,2)")
    n = src.shape[0]
    EA = np.mean(src, axis=0)
    EB = np.mean(dst, axis=0)
    VarA = np.mean(np.sum((src - EA) ** 2, axis=1))
    if VarA < 1e-18:
        raise ValueError("Degenerate variance in source points")
    UP = src - EA
    VP = dst - EB
    H = UP.T @ VP / n
    U, D, Vt = np.linalg.svd(H)
    d = float(np.linalg.det(U) * np.linalg.det(Vt.T))
    S = np.eye(2)
    if d < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / VarA)
    t_row = EB - EA @ R.T * s
    return s, R, t_row


def _closed_perimeter(pts: np.ndarray) -> float:
    ring = np.vstack([pts, pts[0:1]])
    return float(np.sum(np.linalg.norm(np.diff(ring, axis=0), axis=1)))


def _icp_similarity_cad_to_utm(
    P: np.ndarray,
    ref_ls: LineString,
    *,
    max_iter: int = 40,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    """
    ICP in UTM: repeatedly assign each transformed CAD sample to its closest point on the
    reference SITE polyline, then re-estimate a similarity transform with Umeyama.
    """
    P = np.asarray(P, dtype=float)
    mu_p = np.mean(P, axis=0)
    mu_q = np.array(ref_ls.centroid.coords[0], dtype=float)
    Lp = _closed_perimeter(P)
    Lq = float(ref_ls.length)
    if Lp < 1e-9 or Lq < 1e-9:
        raise ValueError("Degenerate SITE boundary lengths")
    s = float(Lq / Lp)
    R = np.eye(2, dtype=float)
    t_row = mu_q - s * (mu_p @ R.T)

    for _ in range(max_iter):
        X = s * (P @ R.T) + t_row
        Y = np.zeros_like(X)
        for i in range(X.shape[0]):
            p_on_ref, _ = nearest_points(ref_ls, ShyPoint(float(X[i, 0]), float(X[i, 1])))
            Y[i] = np.array(p_on_ref.coords[0], dtype=float)
        s2, R2, t2 = umeyama_similarity_rows(P, Y)
        delta = float(abs(s2 - s) + float(np.linalg.norm(R2 - R)) + float(np.linalg.norm(t2 - t_row)))
        s, R, t_row = s2, R2, t2
        if delta < 1e-7:
            break

    Xf = s * (P @ R.T) + t_row
    Yf = np.zeros_like(Xf)
    for i in range(Xf.shape[0]):
        p_on_ref, _ = nearest_points(ref_ls, ShyPoint(float(Xf[i, 0]), float(Xf[i, 1])))
        Yf[i] = np.array(p_on_ref.coords[0], dtype=float)
    dists = np.array([ShyPoint(float(xy[0]), float(xy[1])).distance(ref_ls) for xy in Xf], dtype=float)
    stats = {
        "mean_distance_m_point_to_ref_polyline": float(np.mean(dists)),
        "max_distance_m_point_to_ref_polyline": float(np.max(dists)),
        "rms_residual_m_to_icp_targets": float(np.sqrt(np.mean(np.sum((Xf - Yf) ** 2, axis=1)))),
    }
    return s, R, t_row, stats


def _transform_xy_row(
    xy: Iterable[float],
    s: float,
    R: np.ndarray,
    t_row: np.ndarray,
    inv: Transformer,
) -> tuple[float, float]:
    p = np.array([[float(xy[0]), float(xy[1])]], dtype=float)
    east, north = (p @ R.T * s + t_row).reshape(2).tolist()
    lon, lat = inv.transform(east, north)
    return float(lon), float(lat)


def transform_geojson_geometry(
    geom: dict[str, Any],
    s: float,
    R: np.ndarray,
    t_row: np.ndarray,
    inv: Transformer,
) -> dict[str, Any]:
    t = geom.get("type")
    if t == "Point":
        c = geom["coordinates"]
        lon, lat = _transform_xy_row(c, s, R, t_row, inv)
        return {"type": "Point", "coordinates": [lon, lat]}
    if t == "LineString":
        coords = geom["coordinates"]
        return {
            "type": "LineString",
            "coordinates": [_transform_xy_row(xy, s, R, t_row, inv) for xy in coords],
        }
    if t == "Polygon":
        rings = geom["coordinates"]
        return {
            "type": "Polygon",
            "coordinates": [
                [_transform_xy_row(xy, s, R, t_row, inv) for xy in ring] for ring in rings
            ],
        }
    if t == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [_transform_xy_row(xy, s, R, t_row, inv) for xy in ring]
                    for ring in poly
                ]
                for poly in geom["coordinates"]
            ],
        }
    raise ValueError(f"Unsupported geometry type: {t!r}")


def _transform_fc(fc: dict[str, Any], s: float, R: np.ndarray, t_row: np.ndarray, inv: Transformer) -> dict[str, Any]:
    out = dict(fc)
    out.pop("crs", None)
    feats = []
    for f in fc.get("features", []):
        nf = dict(f)
        g = nf.get("geometry")
        if g is None:
            feats.append(nf)
            continue
        nf["geometry"] = transform_geojson_geometry(g, s, R, t_row, inv)
        feats.append(nf)
    out["features"] = feats
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Align CAD GeoJSON exports to data/SITE.json (WGS84).")
    ap.add_argument("--site-dxf", type=Path, default=Path(r"F:\Aworks\2026studio\shanghaistation\site.dxf"))
    ap.add_argument("--district-dxf", type=Path, default=Path(r"F:\Aworks\2026studio\shanghaistation\district.dxf"))
    ap.add_argument("--site-json", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "SITE.json")
    ap.add_argument("--repo-data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--utm-epsg", type=int, default=32651)
    args = ap.parse_args()

    repo_data: Path = args.repo_data
    site_local = repo_data / "site_dxf" / "local"
    site_crs84 = repo_data / "site_dxf" / "crs84"
    district_local = repo_data / "district_dxf" / "local"
    district_crs84 = repo_data / "district_dxf" / "crs84"
    meta_path = repo_data / "cad_alignment" / "transform_meta.json"

    export_dxf_by_layer(args.site_dxf, site_local)
    export_dxf_by_layer(args.district_dxf, district_local)

    site_json = json.loads(args.site_json.read_text(encoding="utf-8"))
    ref_ring_ll = _as_closed_vertex_ring(site_json["features"][0]["geometry"]["coordinates"])
    src_ring = _dxf_site_ring(args.site_dxf, layer="SITE")

    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{args.utm_epsg}", always_xy=True)
    inv = Transformer.from_crs(f"EPSG:{args.utm_epsg}", "EPSG:4326", always_xy=True)

    ref_east, ref_north = fwd.transform(ref_ring_ll[:, 0], ref_ring_ll[:, 1])
    ref_vert_utm = np.column_stack([ref_east, ref_north])
    target_positive = _signed_area(ref_vert_utm) > 0
    src_ring = _maybe_reverse_ring(src_ring, target_positive=target_positive)

    ref_ls = LineString(np.vstack([ref_vert_utm, ref_vert_utm[0:1]]))
    P = _resample_closed_polyline(src_ring, args.samples)
    s, R, t_row, icp_stats = _icp_similarity_cad_to_utm(P, ref_ls, max_iter=60)

    V = np.asarray(src_ring, dtype=float)
    Xv = s * (V @ R.T) + t_row
    vert_dists = np.array([ShyPoint(float(xy[0]), float(xy[1])).distance(ref_ls) for xy in Xv], dtype=float)
    vertex_stats = {
        "mean_distance_m_dxf_vertices_to_ref_polyline": float(np.mean(vert_dists)),
        "max_distance_m_dxf_vertices_to_ref_polyline": float(np.max(vert_dists)),
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "site_dxf": str(args.site_dxf.resolve()),
        "district_dxf": str(args.district_dxf.resolve()),
        "site_json": str(args.site_json.resolve()),
        "utm_epsg": int(args.utm_epsg),
        "similarity": {
            "scale": s,
            "R": R.tolist(),
            "t_row_utm": t_row.tolist(),
            "convention": "utm_row = s * cad_row @ R.T + t_row; lonlat = inv_epsg4326(utm_row)",
        },
        "fit": {
            "samples": int(args.samples),
            "icp_on_resampled_cad_points": icp_stats,
            "dxf_polyline_vertices": vertex_stats,
        },
        "outputs": {
            "site_local": str(site_local),
            "site_crs84": str(site_crs84),
            "district_local": str(district_local),
            "district_crs84": str(district_crs84),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    site_crs84.mkdir(parents=True, exist_ok=True)
    district_crs84.mkdir(parents=True, exist_ok=True)

    for path in sorted(site_local.glob("*.geojson")):
        fc = json.loads(path.read_text(encoding="utf-8"))
        out = _transform_fc(fc, s, R, t_row, inv)
        (site_crs84 / path.name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for path in sorted(district_local.glob("*.geojson")):
        fc = json.loads(path.read_text(encoding="utf-8"))
        out = _transform_fc(fc, s, R, t_row, inv)
        (district_crs84 / path.name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Remove legacy flat exports under data/district_dxf/ (now lives in local/ + crs84/).
    district_root = repo_data / "district_dxf"
    for name in ("Z_CBD.geojson", "Z_OFC.geojson", "Z_TOD.geojson", "export_meta.json"):
        legacy = district_root / name
        if legacy.is_file():
            legacy.unlink()

    print(meta_path)
    print("icp", icp_stats)
    print("vertices", vertex_stats)


if __name__ == "__main__":
    main()
