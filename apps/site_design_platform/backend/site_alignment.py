"""Align Rhino local coordinates to WGS84 SITE boundary via ring-matched affine transform."""
from __future__ import annotations

import math
from typing import Any


def first_ring(geojson_obj: dict | None) -> list[list[float]] | None:
    if not isinstance(geojson_obj, dict):
        return None
    if geojson_obj.get("type") == "Polygon":
        coords = geojson_obj.get("coordinates", [])
        return coords[0] if coords else None
    if geojson_obj.get("type") == "Feature":
        return first_ring(geojson_obj.get("geometry", {}))
    if geojson_obj.get("type") == "FeatureCollection":
        for f in geojson_obj.get("features", []):
            r = first_ring(f)
            if r:
                return r
    return None


def _open_ring(ring: list[list[float]]) -> list[list[float]]:
    if len(ring) < 2:
        return list(ring)
    if ring[0][0] == ring[-1][0] and ring[0][1] == ring[-1][1]:
        return ring[:-1]
    return list(ring)


def _signed_area(ring: list[list[float]]) -> float:
    s = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % len(ring)]
        s += x1 * y2 - x2 * y1
    return s * 0.5


def _canonical_ring(ring: list[list[float]]) -> list[list[float]]:
    """Consistent start vertex and CCW winding for index-based correspondence."""
    pts = _open_ring(ring)
    if len(pts) < 3:
        return pts
    if _signed_area(pts) < 0:
        pts = list(reversed(pts))
    start = min(range(len(pts)), key=lambda i: (pts[i][0] + pts[i][1], pts[i][0], pts[i][1]))
    return pts[start:] + pts[:start]


def _solve_affine_3x3(src_pts: list[list[float]], dst_pts: list[list[float]]) -> list[list[float]] | None:
    """Least-squares affine: [u,v] = M @ [x,y,1]. Returns 2x3 matrix."""
    n = min(len(src_pts), len(dst_pts))
    if n < 3:
        return None
    ata = [[0.0] * 3 for _ in range(3)]
    bu = [0.0, 0.0, 0.0]
    bv = [0.0, 0.0, 0.0]
    for i in range(n):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        row = [x, y, 1.0]
        for r in range(3):
            for c in range(3):
                ata[r][c] += row[r] * row[c]
            bu[r] += row[r] * u
            bv[r] += row[r] * v
    try:
        au = _solve_3x3(ata, bu)
        av = _solve_3x3(ata, bv)
    except ValueError:
        return None
    if au is None or av is None:
        return None
    return [au, av]


def _solve_3x3(a: list[list[float]], b: list[float]) -> list[float] | None:
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        for j in range(4):
            m[col][j] /= div
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col]
            for j in range(4):
                m[r][j] -= factor * m[col][j]
    return [m[i][3] for i in range(3)]


def _affine_rms(src_pts: list[list[float]], dst_pts: list[list[float]], m: list[list[float]]) -> float:
    err = 0.0
    n = min(len(src_pts), len(dst_pts))
    for i in range(n):
        x, y = src_pts[i]
        u = m[0][0] * x + m[0][1] * y + m[0][2]
        v = m[1][0] * x + m[1][1] * y + m[1][2]
        err += (u - dst_pts[i][0]) ** 2 + (v - dst_pts[i][1]) ** 2
    return math.sqrt(err / max(1, n))


def fit_ring_affine(src_ring: list[list[float]], dst_ring: list[list[float]]) -> list[list[float]] | None:
    """Find best cyclic-shift affine mapping from src ring vertices to dst ring."""
    src = _canonical_ring(src_ring)
    dst = _canonical_ring(dst_ring)
    if len(src) < 3 or len(dst) < 3:
        return None

    best_m: list[list[float]] | None = None
    best_err = float("inf")

    # Equal vertex count: try cyclic shifts on source ring.
    if len(src) == len(dst):
        n = len(src)
        for shift in range(n):
            shifted = src[shift:] + src[:shift]
            m = _solve_affine_3x3(shifted, dst)
            if m is None:
                continue
            err = _affine_rms(shifted, dst, m)
            if err < best_err:
                best_err = err
                best_m = m
    else:
        # Different counts: resample both to the same arc-length parameterization.
        count = min(len(src), len(dst), 64)
        src_rs = _resample_ring(src, count)
        dst_rs = _resample_ring(dst, count)
        best_m = _solve_affine_3x3(src_rs, dst_rs)

    return best_m


def _resample_ring(ring: list[list[float]], count: int) -> list[list[float]]:
    pts = _open_ring(ring)
    if len(pts) < 2:
        return pts
    closed = pts + [pts[0]]
    seg_lens = []
    total = 0.0
    for i in range(len(closed) - 1):
        l = math.hypot(closed[i + 1][0] - closed[i][0], closed[i + 1][1] - closed[i][1])
        seg_lens.append(l)
        total += l
    if total <= 0:
        return pts[:count]
    out: list[list[float]] = []
    for k in range(count):
        target = total * k / count
        acc = 0.0
        for i, l in enumerate(seg_lens):
            if acc + l >= target or i == len(seg_lens) - 1:
                t = 0.0 if l <= 0 else (target - acc) / l
                x = closed[i][0] + t * (closed[i + 1][0] - closed[i][0])
                y = closed[i][1] + t * (closed[i + 1][1] - closed[i][1])
                out.append([x, y])
                break
            acc += l
    return out


def apply_affine_point(x: float, y: float, m: list[list[float]]) -> tuple[float, float]:
    return (
        m[0][0] * x + m[0][1] * y + m[0][2],
        m[1][0] * x + m[1][1] * y + m[1][2],
    )


def transform_geometry_affine(g: dict, m: list[list[float]]) -> dict:
    gt = g.get("type")
    out: dict[str, Any] = {"type": gt}
    if gt == "Polygon":
        out["coordinates"] = [
            [list(apply_affine_point(p[0], p[1], m)) for p in ring]
            for ring in g.get("coordinates", [])
        ]
    elif gt == "MultiPolygon":
        out["coordinates"] = [
            [[list(apply_affine_point(p[0], p[1], m)) for p in ring] for ring in poly]
            for poly in g.get("coordinates", [])
        ]
    elif gt == "LineString":
        out["coordinates"] = [list(apply_affine_point(p[0], p[1], m)) for p in g.get("coordinates", [])]
    elif gt == "MultiLineString":
        out["coordinates"] = [
            [list(apply_affine_point(p[0], p[1], m)) for p in ln]
            for ln in g.get("coordinates", [])
        ]
    else:
        return g
    return out


def _transform_payload_geometries(payload: dict, m: list[list[float]]) -> None:
    for b in payload.get("blocks", []):
        g = b.get("geometry")
        if isinstance(g, dict):
            b["geometry"] = transform_geometry_affine(g, m)
    for key in ("rhino_parcels", "rhino_original_buildings", "rhino_walking", "rhino_ground"):
        fc = payload.get(key, {})
        if isinstance(fc, dict):
            for f in fc.get("features", []):
                g = f.get("geometry")
                if isinstance(g, dict):
                    f["geometry"] = transform_geometry_affine(g, m)
    outline = payload.get("rhino_site_outline")
    if isinstance(outline, dict):
        payload["rhino_site_outline"] = transform_geometry_affine(outline, m)


def align_payload_to_site(payload: dict, site_geojson: dict) -> bool:
    """Transform all payload geometries from Rhino local space to SITE WGS84."""
    src_ring = first_ring(payload.get("rhino_site_outline"))
    dst_ring = first_ring(site_geojson)
    if not src_ring or not dst_ring or len(src_ring) < 4 or len(dst_ring) < 4:
        return False
    m = fit_ring_affine(src_ring, dst_ring)
    if m is None:
        return False
    _transform_payload_geometries(payload, m)
    return True
