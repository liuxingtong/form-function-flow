"""
Export DXF modelspace entities to one GeoJSON FeatureCollection per layer.

Requires: ezdxf
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import ezdxf


def _xy(loc: Any) -> tuple[float, float]:
    return (float(loc.x), float(loc.y))


def _polyline_to_geometry(e: Any) -> dict[str, Any] | None:
    verts = [_xy(v.dxf.location) for v in e.vertices]
    if len(verts) < 2:
        return None
    closed = bool(e.is_closed)
    if closed and len(verts) >= 3:
        ring = list(verts)
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        if len(ring) < 4:
            return {"type": "LineString", "coordinates": verts}
        return {"type": "Polygon", "coordinates": [ring]}
    return {"type": "LineString", "coordinates": verts}


def _entity_to_feature(e: Any) -> dict[str, Any] | None:
    layer = str(e.dxf.layer)
    dxftype = e.dxftype()
    geom: dict[str, Any] | None = None

    if dxftype == "POLYLINE":
        geom = _polyline_to_geometry(e)
    elif dxftype == "LWPOLYLINE":
        pts = [(float(x), float(y)) for x, y, *_ in e.get_points("xy")]
        if len(pts) < 2:
            return None
        closed = bool(e.closed)
        if closed and len(pts) >= 3:
            ring = list(pts)
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            if len(ring) < 4:
                geom = {"type": "LineString", "coordinates": pts}
            else:
                geom = {"type": "Polygon", "coordinates": [ring]}
        else:
            geom = {"type": "LineString", "coordinates": pts}
    elif dxftype == "LINE":
        start = e.dxf.start
        end = e.dxf.end
        geom = {
            "type": "LineString",
            "coordinates": [_xy(start), _xy(end)],
        }
    elif dxftype == "POINT":
        geom = {"type": "Point", "coordinates": _xy(e.dxf.location)}
    else:
        return None

    if geom is None:
        return None

    props: dict[str, Any] = {
        "layer": layer,
        "dxftype": dxftype,
        "handle": str(e.dxf.handle),
    }
    if hasattr(e.dxf, "color"):
        props["color"] = int(e.dxf.color)
    return {"type": "Feature", "geometry": geom, "properties": props}


def _safe_filename(layer: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", layer, flags=re.UNICODE).strip("_")
    return s or "layer"


def export_dxf_by_layer(dxf_path: Path, out_dir: Path) -> list[Path]:
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped_by_type: dict[str, int] = defaultdict(int)

    for e in msp:
        feat = _entity_to_feature(e)
        if feat is None:
            skipped_by_type[e.dxftype()] += 1
            continue
        by_layer[str(e.dxf.layer)].append(feat)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for layer, features in sorted(by_layer.items(), key=lambda x: x[0]):
        fc: dict[str, Any] = {
            "type": "FeatureCollection",
            "name": layer,
            "features": features,
        }
        stem = _safe_filename(layer)
        out_path = out_dir / f"{stem}.geojson"
        out_path.write_text(
            json.dumps(fc, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(out_path)

    meta = {
        "source_dxf": str(dxf_path.resolve()),
        "layers": {k: len(v) for k, v in by_layer.items()},
        "skipped_by_dxftype": dict(skipped_by_type),
    }
    (out_dir / "export_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Export DXF modelspace to GeoJSON per layer.")
    p.add_argument(
        "--dxf",
        type=Path,
        default=Path(r"F:\Aworks\2026studio\shanghaistation\district.dxf"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "district_dxf",
    )
    args = p.parse_args()
    paths = export_dxf_by_layer(args.dxf, args.out_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
