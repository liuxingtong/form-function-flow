"""
Convert input.3dm → rhino_live.json and POST to /api/site-design/rhino/sync.

Usage:
    python apps/site_design_platform/tools/import_3dm.py [path/to/input.3dm]
    python apps/site_design_platform/tools/import_3dm.py --dry-run
    python apps/site_design_platform/tools/import_3dm.py --write-only

Options:
    --dry-run     Print stats and exit without writing/POSTing
    --write-only  Write rhino_live.json directly, skip HTTP POST (coords stay in Rhino local space)
    --host        Server host (default: 127.0.0.1)
    --port        Server port (default: 8088)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

try:
    import rhino3dm
except ImportError:
    sys.exit("rhino3dm not installed. Run: pip install rhino3dm")


# Layer names that map directly to frontend functionType values
BLOCK_LAYERS = {
    "CENTER_OFFICE",
    "SMALL_OFFICE",
    "APARTMENT",
    "HOTEL",
    "RESIDENTIAL",
    "CENTER_COMMERCIAL",
    "LEISURE_COMMERCIAL",
    "PUBLIC",
    "GROUND",
    "GREEN",
}

SITE_LAYER = "SITE"

# Road/path surfaces — extracted into rhino_ground / rhino_walking, not blocks
GROUND_LAYERS = {"HIGHWAY"}    # → rhino_ground (gray flat fill)
WALKING_LAYERS = {"WALKWAY"}   # → rhino_walking (white dashed lines)


# ── Geometry extraction ──────────────────────────────────────────────────────

def _angle_from_centroid(pt: list[float], cx: float, cy: float) -> float:
    return math.atan2(pt[1] - cy, pt[0] - cx)


def _sort_ring(pts: list[list[float]]) -> list[list[float]]:
    """Sort XY points by angle from centroid to form a convex polygon ring."""
    if len(pts) < 3:
        return pts
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    pts_sorted = sorted(pts, key=lambda p: _angle_from_centroid(p, cx, cy))
    pts_sorted.append(pts_sorted[0])  # close ring
    return pts_sorted


def _dedup_xy(pts: list[list[float]], tol: float = 0.01) -> list[list[float]]:
    out: list[list[float]] = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol for q in out):
            out.append(p)
    return out


def _polyline_points(curve) -> list[list[float]] | None:
    """Extract XY from a PolylineCurve using its indexed Point() API."""
    try:
        n = curve.PointCount
        pts = []
        for i in range(n):
            pt = curve.Point(i)
            pts.append([round(pt.X, 4), round(pt.Y, 4)])
        # remove closing duplicate if present
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        return pts if len(pts) >= 3 else None
    except Exception:
        return None


def _nurbs_sample(curve, n: int = 64) -> list[list[float]] | None:
    """Sample n evenly-spaced points along any curve via ToNurbsCurve()."""
    try:
        nc = curve.ToNurbsCurve()
        if nc is None:
            return None
        # rhino3dm Interval subscript: [0]=t0, [1]=t1
        dom = nc.Domain
        t0, t1 = dom[0], dom[1]
        pts = []
        for i in range(n):
            t = t0 + (t1 - t0) * i / n
            pt = nc.PointAt(t)
            if pt is not None:
                pts.append([round(pt.X, 4), round(pt.Y, 4)])
        return pts if len(pts) >= 3 else None
    except Exception:
        return None


def _brep_bottom_ring(brep) -> list[list[float]] | None:
    """Extract building footprint from a box Brep.

    For box buildings: uses only bottom-face vertices (min Z slice).
    Falls back to XY projection of all vertices for inclined/non-box shapes
    (e.g. road panels where only 2 vertices sit at min Z).
    """
    bb = brep.GetBoundingBox()
    if bb is None:
        return None
    min_z = bb.Min.Z
    height_span = bb.Max.Z - bb.Min.Z
    tol = max(0.3, height_span * 0.05)

    bottom: list[list[float]] = []
    all_xy: list[list[float]] = []
    for i in range(len(brep.Vertices)):
        v = brep.Vertices[i].Location
        xy = [round(v.X, 4), round(v.Y, 4)]
        all_xy.append(xy)
        if abs(v.Z - min_z) <= tol:
            bottom.append(xy)

    bottom = _dedup_xy(bottom)
    if len(bottom) >= 3:
        return _sort_ring(bottom)

    # Fallback: project all vertices to XY (handles slanted panels)
    all_xy = _dedup_xy(all_xy)
    return _sort_ring(all_xy) if len(all_xy) >= 3 else None


def _extrusion_bottom_ring(extrusion) -> list[list[float]] | None:
    """Extract bottom-face footprint from an Extrusion via its mesh."""
    try:
        bb = extrusion.GetBoundingBox()
        min_z = bb.Min.Z if bb else None
        mesh = extrusion.GetMesh(rhino3dm.MeshType.Default)
        if mesh is None:
            return None
        tol = 0.5 if min_z is not None else None
        pts: list[list[float]] = []
        for i in range(len(mesh.Vertices)):
            v = mesh.Vertices[i]
            if tol is None or (min_z is not None and abs(v.Z - min_z) <= tol):
                pts.append([round(v.X, 4), round(v.Y, 4)])
        pts = _dedup_xy(pts)
        return _sort_ring(pts) if len(pts) >= 3 else None
    except Exception:
        return None


def _extract_ring(obj) -> list[list[float]] | None:
    """Dispatch footprint extraction based on geometry type."""
    t = obj.ObjectType
    if t == rhino3dm.ObjectType.Brep:
        return _brep_bottom_ring(obj)
    if t == rhino3dm.ObjectType.Extrusion:
        return _extrusion_bottom_ring(obj)
    # Curve types: try PolylineCurve direct points first, then NurbsCurve sampling
    if hasattr(obj, "PointCount"):
        return _polyline_points(obj)
    return _nurbs_sample(obj)


def _bbox_heights(obj) -> tuple[float, float]:
    """Return (height, base) from bounding box Z extents."""
    try:
        bb = obj.GetBoundingBox()
        if bb is None:
            return 0.0, 0.0
        h = max(0.0, round(bb.Max.Z - bb.Min.Z, 2))
        base = max(0.0, round(bb.Min.Z, 2))
        return h, base
    except Exception:
        return 0.0, 0.0


# ── Main conversion ───────────────────────────────────────────────────────────

def _layer_map(model) -> dict[int, str]:
    return {m.Index: m.Name.upper().strip() for m in model.Layers}


def convert(model_path: Path) -> dict:
    """Read .3dm and return a rhino_live.json-compatible payload dict."""
    model = rhino3dm.File3dm.Read(str(model_path))
    if model is None:
        sys.exit(f"Failed to read: {model_path}")

    layers = _layer_map(model)

    blocks: list[dict] = []
    ground_features: list[dict] = []
    walking_features: list[dict] = []
    site_ring: list[list[float]] | None = None
    counts: dict[str, int] = {}
    skipped = 0
    errors = 0

    for i in range(len(model.Objects)):
        obj_ref = model.Objects[i]
        obj = obj_ref.Geometry
        layer_name = layers.get(obj_ref.Attributes.LayerIndex, "")

        # ── SITE boundary ──────────────────────────────────────────────────
        if layer_name == SITE_LAYER:
            if site_ring is None:
                ring = _polyline_points(obj) if hasattr(obj, "PointCount") else _nurbs_sample(obj, 128)
                if ring:
                    site_ring = ring
                    site_ring.append(site_ring[0])  # close
            continue

        # ── Road surfaces → rhino_ground ───────────────────────────────────
        if layer_name in GROUND_LAYERS:
            ring = _extract_ring(obj)
            if ring and len(ring) >= 4:
                ground_features.append({
                    "type": "Feature",
                    "properties": {"id": f"{layer_name}_{i}", "layer": layer_name},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                })
            else:
                errors += 1
            continue

        # ── Pedestrian paths → rhino_walking ───────────────────────────────
        if layer_name in WALKING_LAYERS:
            ring = _extract_ring(obj)
            if ring and len(ring) >= 4:
                walking_features.append({
                    "type": "Feature",
                    "properties": {"id": f"{layer_name}_{i}", "layer": layer_name},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                })
            else:
                errors += 1
            continue

        if layer_name not in BLOCK_LAYERS:
            skipped += 1
            continue

        # ── Extract building footprint ─────────────────────────────────────
        ring = _extract_ring(obj)
        if ring is None or len(ring) < 4:
            errors += 1
            continue

        height, base = _bbox_heights(obj)
        blocks.append({
            "id": f"{layer_name}_{i}",
            "function": layer_name,
            "height": height,
            "base": base,
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
        counts[layer_name] = counts.get(layer_name, 0) + 1

    print(f"Extracted {len(blocks)} blocks  |  ground={len(ground_features)}  walking={len(walking_features)}  errors={errors}  skipped={skipped}")
    print(f"  Site outline: {'found (%d pts)' % len(site_ring) if site_ring else 'NOT FOUND'}")
    for fn, cnt in sorted(counts.items()):
        print(f"    {fn}: {cnt}")

    return {
        "blocks": blocks,
        "rhino_site_outline": (
            {"type": "Polygon", "coordinates": [site_ring]} if site_ring else None
        ),
        "rhino_parcels":            {"type": "FeatureCollection", "features": []},
        "rhino_original_buildings": {"type": "FeatureCollection", "features": []},
        "rhino_walking":            {"type": "FeatureCollection", "features": walking_features},
        "rhino_ground":             {"type": "FeatureCollection", "features": ground_features},
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    ap = argparse.ArgumentParser(description="Import .3dm building blocks into rhino_live.json")
    ap.add_argument("input", nargs="?", default=str(repo_root / "input.3dm"),
                    help="Path to .3dm file (default: <repo>/input.3dm)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats only, do not write or POST")
    ap.add_argument("--write-only", action="store_true",
                    help="Write rhino_live.json directly (skips server coordinate alignment)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    args = ap.parse_args()

    model_path = Path(args.input)
    if not model_path.exists():
        sys.exit(f"File not found: {model_path}")

    print(f"Reading {model_path} ...")
    payload = convert(model_path)

    if args.dry_run:
        print("Dry run — done.")
        return

    if args.write_only:
        out = repo_root / "data" / "site_design_platform" / "scenarios" / "rhino_live.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Written: {out}")
        print("NOTE: coords remain in Rhino local space (no server alignment).")
        return

    # POST to server — server aligns Rhino coords → WGS84 via SITE.geojson
    url = f"http://{args.host}:{args.port}/api/site-design/rhino/sync"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
        data=body,
    )
    print(f"POSTing {len(payload['blocks'])} blocks → {url} ...")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        aligned = result.get("aligned_to_site", False)
        print(f"Server: {result}")
        print(f"Coordinate alignment to SITE.geojson: {'SUCCESS' if aligned else 'SKIPPED'}")
    except urllib.error.URLError as e:
        print(f"POST failed: {e}")
        print("Start the server first:  python apps/site_design_platform/start.py")
        print("Or skip the server with: --write-only")
        sys.exit(1)


if __name__ == "__main__":
    main()
