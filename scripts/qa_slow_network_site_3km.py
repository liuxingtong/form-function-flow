"""
Plan A：评估「慢行相关」路网在 data/site_3km 裁切后的几何覆盖（线长、要素数）。

默认扫描：
  data/site_3km/04-交通数据/1-交通路网/2-分类图层/

对文件名含关键字的分组统计（米制长度，GeoJSON WGS84 下 geometry.to_crs(32651)）。

用法（仓库根目录）：
  python scripts/qa_slow_network_site_3km.py
  python scripts/qa_slow_network_site_3km.py --write-json data/site_3km/qa/slow_network_coverage_summary.json

裁切更新请重跑：
  python scripts/clip_external_theme_roots_to_site_3km.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LAYER_DIR = REPO / "data" / "site_3km" / "04-交通数据" / "1-交通路网" / "2-分类图层"

GROUP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("行人专用层", ("行人",)),
    ("自行车道", ("自行车",)),
    ("轮渡渡口", ("轮渡", "渡口", "人渡口")),
    ("运动场跑道", ("运动场", "跑道")),
    ("其它道路（可能含可步行段）", ("其它道路", "其它")),
    ("主干快速路（对照）", ("快速路", "主干道", "公路", "省道", "市区一级")),
)


def _match_group(stem: str) -> str | None:
    s = stem.lower()
    for name, keys in GROUP_RULES:
        if any(k.lower() in s for k in keys):
            return name
    return None


def line_length_m(gdf: gpd.GeoDataFrame) -> float:
    if gdf.empty:
        return 0.0
    gg = gdf
    if gg.crs is None:
        gg = gg.set_crs(4326)
    gg = gg.to_crs(32651)
    return float(gg.geometry.length.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description="慢行相关路网裁切后覆盖 QA")
    ap.add_argument("--layer-dir", type=Path, default=DEFAULT_LAYER_DIR)
    ap.add_argument("--write-json", type=Path, default=None)
    ns = ap.parse_args()
    root = ns.layer_dir
    if not root.is_dir():
        print(f"目录不存在: {root}", file=sys.stderr)
        return 1

    per_file: list[dict] = []
    by_group: dict[str, dict[str, float]] = defaultdict(lambda: {"n_features": 0.0, "length_m": 0.0})

    for p in sorted(root.glob("*.geojson")):
        try:
            g = gpd.read_file(p)
        except Exception as e:
            print(f"SKIP {p.name}: {e}", file=sys.stderr)
            continue
        lm = line_length_m(g)
        grp = _match_group(p.stem) or "未分组（其它文件名）"
        by_group[grp]["n_features"] += float(len(g))
        by_group[grp]["length_m"] += lm
        per_file.append({"file": p.name, "group": grp, "n": len(g), "length_m": round(lm, 2)})

    summary = {
        "layer_dir": str(root.relative_to(REPO)),
        "by_group": {k: {"n_features": int(v["n_features"]), "length_m": round(v["length_m"], 2)} for k, v in sorted(by_group.items())},
        "files": sorted(per_file, key=lambda x: -x["length_m"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if ns.write_json is not None:
        ns.write_json.parent.mkdir(parents=True, exist_ok=True)
        ns.write_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {ns.write_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
