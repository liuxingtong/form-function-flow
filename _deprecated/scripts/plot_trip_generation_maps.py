#!/usr/bin/env python3
"""从 trip_generation.csv + 01_units 重绘出行生成四面板图（无需重算 OD）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from synthetic_flow_od_gravity import _load_units, plot_trip_generation_maps  # noqa: E402
from site_map_overlay import resolve_site_json_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=Path, default=REPO / "output/function/数据包/01_units.gpkg")
    ap.add_argument("--trip-gen", type=Path, default=REPO / "output/synthetic_flow/trip_generation.csv")
    ap.add_argument("--out", type=Path, default=REPO / "output/synthetic_flow/trip_generation_maps.png")
    ap.add_argument("--site-json", type=Path, default=None)
    ns = ap.parse_args()
    tg = pd.read_csv(ns.trip_gen, encoding="utf-8-sig")
    need = ("prior_production", "prior_attraction", "trip_production", "trip_attraction")
    miss = [c for c in need if c not in tg.columns]
    if miss:
        raise SystemExit(f"{ns.trip_gen} 缺少列 {miss}；请用新版 synthetic_flow_od_gravity.py 重算出行生成表。")
    u = _load_units(ns.units)
    sp = ns.site_json if ns.site_json is not None and ns.site_json.is_file() else resolve_site_json_path()
    plot_trip_generation_maps(u, tg, ns.out, site_path=sp)
    print(f"Wrote {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
