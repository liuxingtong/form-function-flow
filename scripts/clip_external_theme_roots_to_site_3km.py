"""
Clip vector layers from external theme folders (e.g. sibling ``all/04-交通数据``)
into ``data/site_3km/<theme>/...`` as GeoJSON, using the same 3 km metric buffer
as ``export_deprecated_to_site_3km_geojson.py``.

Does **not** delete existing ``site_3km`` content; only writes/overwrites matching
output paths.

Run from repo root (adjust paths if your ``all`` folder differs):

  python scripts/clip_external_theme_roots_to_site_3km.py ^
    --src-root "F:/Aworks/2026studio/shanghaistation/all/04-交通数据" ^
    --src-root "F:/Aworks/2026studio/shanghaistation/all/16-城市环路"

Requires: geopandas, pandas, shapely
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from clip_data_to_site import clip_gdf, load_clip_mask  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

DEFAULT_SRC_ROOTS = (
    Path(r"F:/Aworks/2026studio/shanghaistation/all/04-交通数据"),
    Path(r"F:/Aworks/2026studio/shanghaistation/all/16-城市环路"),
)

VECTOR_SUFFIX = frozenset({".shp", ".geojson"})


def write_geojson_fc(gdf: gpd.GeoDataFrame, out_path: Path, stem_name: str, source_rel: str) -> None:
    gdf2 = gdf.copy()
    gdf2["__source_file__"] = source_rel.replace("\\", "/")
    for col in gdf2.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_datetime64_any_dtype(gdf2[col]):
            gdf2[col] = gdf2[col].astype(str)
    fc = json.loads(gdf2.to_json())
    fc["name"] = stem_name
    fc["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, separators=(",", ":"))


def read_shapefile(path: Path) -> gpd.GeoDataFrame | None:
    last_err: Exception | None = None
    for enc in ("utf-8", "gb18030", "gbk", "latin1"):
        try:
            return gpd.read_file(path, encoding=enc)
        except Exception as e:
            last_err = e
    print(f"    SKIP shp read {path.name}: {last_err}", file=sys.stderr)
    return None


def load_mask_wgs84(repo: Path, site_json: Path | None, buffer_m: float) -> gpd.GeoDataFrame:
    site_3km = repo / "data" / "site_3km"
    buf_path = site_3km / "SITE.buffer_3km.geojson"
    if buf_path.is_file():
        g = gpd.read_file(buf_path)
        if g.crs is None:
            g = g.set_crs("EPSG:4326")
        return g.to_crs("EPSG:4326")

    candidates = [p for p in (site_json, site_3km / "SITE.json", repo / "data" / "SITE.json") if p and p.is_file()]
    if not candidates:
        raise FileNotFoundError("No SITE.buffer_3km.geojson and no SITE.json found.")
    return load_clip_mask(candidates[0], buffer_m)


def ensure_site_buffer_geojson(mask: gpd.GeoDataFrame, site_3km: Path) -> None:
    site_3km.mkdir(parents=True, exist_ok=True)
    out = site_3km / "SITE.buffer_3km.geojson"
    if out.is_file():
        return
    buf_fc = json.loads(mask.to_json())
    buf_fc.setdefault("name", "SITE_buffer_3km")
    buf_fc["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
    with out.open("w", encoding="utf-8") as f:
        json.dump(buf_fc, f, ensure_ascii=False, separators=(",", ":"))


def iter_vector_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VECTOR_SUFFIX:
            continue
        out.append(p)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Clip external all/04 交通 + 16 环路 into data/site_3km.")
    ap.add_argument(
        "--src-root",
        type=Path,
        action="append",
        default=[],
        help="Theme folder (basename becomes output subfolder under site_3km). Repeatable.",
    )
    ap.add_argument("--out-root", type=Path, default=REPO / "data" / "site_3km", help="Default data/site_3km")
    ap.add_argument("--site-json", type=Path, default=None, help="SITE.json for buffer if buffer geojson missing")
    ap.add_argument("--buffer-m", type=float, default=3000.0)
    ap.add_argument("--max-file-mb", type=float, default=250.0, help="Skip vectors larger than this")
    ap.add_argument(
        "--skip-name-substrings",
        nargs="*",
        default=["路网合集"],
        help="Skip source stems containing any of these substrings (default: 路网合集)",
    )
    args = ap.parse_args()

    roots = [Path(p).resolve() for p in (args.src_root or list(DEFAULT_SRC_ROOTS))]
    out_root: Path = args.out_root.resolve()
    site_3km = out_root

    for r in roots:
        if not r.is_dir():
            print("Missing src-root:", r, file=sys.stderr)
            return 1

    mask = load_mask_wgs84(REPO, args.site_json, args.buffer_m)
    ensure_site_buffer_geojson(mask, site_3km)

    max_bytes = int(args.max_file_mb * 1024 * 1024)
    skips = list(args.skip_name_substrings)

    manifest: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "buffer_m": args.buffer_m,
        "src_roots": [str(r) for r in roots],
        "out_root": str(out_root),
        "results": {"ok": [], "skip": [], "fail": []},
    }

    for src_root in roots:
        theme = src_root.name
        for src in iter_vector_files(src_root):
            rel = src.relative_to(src_root)
            rel_posix = f"{theme}/{rel.as_posix()}"
            if "csv" in src.stem.lower():
                manifest["results"]["skip"].append({"source": rel_posix, "detail": "csv artifact"})
                continue
            if any(s in src.stem for s in skips):
                manifest["results"]["skip"].append({"source": rel_posix, "detail": "skip-name-substrings"})
                continue
            try:
                sz = src.stat().st_size
            except OSError as e:
                manifest["results"]["fail"].append({"source": rel_posix, "detail": str(e)})
                continue
            if sz > max_bytes:
                manifest["results"]["skip"].append({"source": rel_posix, "detail": f"file>{args.max_file_mb}MB"})
                continue

            out_path = out_root / theme / rel.parent / f"{src.stem}.geojson"

            if src.suffix.lower() == ".shp":
                gdf = read_shapefile(src)
            else:
                try:
                    gdf = gpd.read_file(src)
                except Exception as e:
                    manifest["results"]["fail"].append({"source": rel_posix, "detail": str(e)})
                    continue

            if gdf is None:
                manifest["results"]["skip"].append({"source": rel_posix, "detail": "unreadable"})
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")

            n0 = len(gdf)
            try:
                clipped = clip_gdf(gdf, mask)
            except Exception as e:
                manifest["results"]["fail"].append({"source": rel_posix, "detail": f"clip:{e}"})
                continue

            if clipped.empty:
                manifest["results"]["skip"].append({"source": rel_posix, "detail": f"empty clip ({n0})"})
                continue

            try:
                write_geojson_fc(clipped, out_path, src.stem, rel_posix)
            except Exception as e:
                manifest["results"]["fail"].append({"source": rel_posix, "detail": str(e)})
                continue

            manifest["results"]["ok"].append({"source": rel_posix, "detail": f"{n0}->{len(clipped)}"})
            print(rel_posix, f"{n0}->{len(clipped)}")

    for k in ("ok", "skip", "fail"):
        manifest["counts"] = manifest.get("counts", {})
        manifest["counts"][k] = len(manifest["results"][k])

    meta_path = site_3km / "clip_external_transport_manifest.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("Wrote manifest:", meta_path)
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    return 0 if manifest["counts"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
