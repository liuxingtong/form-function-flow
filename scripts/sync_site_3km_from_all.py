"""
Clip vector layers from sibling ``all/`` theme folders into ``data/site_3km/<theme>/...``
as GeoJSON, using ``SITE.buffer_3km.geojson`` or SITE.json + 3 km buffer.

Targets function-layer inputs (``output/function/数据包/analysis.py``):
  02-POI&AOI, 08-房价, 10-AI, 11-商圈/点评类, 13-公服, 04-交通, 16-环路, 09-土地利用备份,
  plus folders containing ``fanwei_meituan_data.shp`` (美团) and ``*二手房*.csv`` 所在楼盘目录.

Skips ``_deprecated``、路径含 ``历史版本`` 的文件。大文件可用 ``--max-file-mb`` 放宽。

  python scripts/sync_site_3km_from_all.py ^
    --all-root "F:/Aworks/2026studio/shanghaistation/all"

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

DEFAULT_ALL_ROOT = Path(r"F:/Aworks/2026studio/shanghaistation/all")

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


def path_has_skip_parts(p: Path, root: Path) -> bool:
    try:
        rel = p.relative_to(root)
    except ValueError:
        return True
    for part in rel.parts:
        if part == "_deprecated" or "历史版本" in part:
            return True
    return False


def iter_vector_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VECTOR_SUFFIX:
            continue
        if path_has_skip_parts(p, root):
            continue
        out.append(p)
    return sorted(out)


def discover_theme_roots(all_root: Path) -> list[Path]:
    """按编号主题 + 美团/楼盘目录构造裁剪根目录列表。"""
    roots: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        k = str(p.resolve())
        if k not in seen and p.is_dir():
            seen.add(k)
            roots.append(p)

    for p in sorted(all_root.iterdir()):
        if not p.is_dir() or p.name == "_deprecated":
            continue
        if p.name.startswith(
            (
                "02-POI",
                "08-",
                "10-",
                "11-",
                "13-",
                "04-",
                "16-",
                "09-",
                "05-",
                "14-",
            )
        ):
            add(p)

    for shp in all_root.rglob("fanwei_meituan_data.shp"):
        if path_has_skip_parts(shp, all_root):
            continue
        add(shp.parent)

    for csv in all_root.rglob("*二手房*.csv"):
        if path_has_skip_parts(csv, all_root):
            continue
        add(csv.parent)

    return sorted(roots, key=lambda x: str(x))


def main() -> int:
    ap = argparse.ArgumentParser(description="Clip all/ themes into data/site_3km for function + flow inputs.")
    ap.add_argument("--all-root", type=Path, default=DEFAULT_ALL_ROOT, help="Sibling all/ folder")
    ap.add_argument("--out-root", type=Path, default=REPO / "data" / "site_3km")
    ap.add_argument("--site-json", type=Path, default=None)
    ap.add_argument("--buffer-m", type=float, default=3000.0)
    ap.add_argument("--max-file-mb", type=float, default=600.0)
    ap.add_argument(
        "--skip-name-substrings",
        nargs="*",
        default=["路网合集"],
        help="Skip source stems containing any of these substrings",
    )
    args = ap.parse_args()

    all_root: Path = args.all_root.resolve()
    if not all_root.is_dir():
        print("Missing --all-root:", all_root, file=sys.stderr)
        return 1

    out_root: Path = args.out_root.resolve()
    site_3km = out_root
    site_3km.mkdir(parents=True, exist_ok=True)

    roots = discover_theme_roots(all_root)
    if not roots:
        print("No theme roots discovered under", all_root, file=sys.stderr)
        return 1

    mask = load_mask_wgs84(REPO, args.site_json, args.buffer_m)
    ensure_site_buffer_geojson(mask, site_3km)

    max_bytes = int(args.max_file_mb * 1024 * 1024)
    skips = list(args.skip_name_substrings)

    manifest: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "buffer_m": args.buffer_m,
        "all_root": str(all_root),
        "theme_roots": [str(r) for r in roots],
        "out_root": str(out_root),
        "results": {"ok": [], "skip": [], "fail": []},
    }

    for src_root in roots:
        theme = src_root.name
        for src in iter_vector_files(src_root):
            rel = src.relative_to(src_root)
            rel_posix = f"{theme}/{rel.as_posix()}"
            if src.suffix.lower() == ".csv":
                manifest["results"]["skip"].append({"source": rel_posix, "detail": "not a vector suffix"})
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

    meta_path = site_3km / "sync_from_all_manifest.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("Wrote manifest:", meta_path)
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    return 0 if manifest["counts"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
