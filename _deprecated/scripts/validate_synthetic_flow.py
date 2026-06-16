#!/usr/bin/env python3
"""
轻量校验合成出行产出（不要求 scipy）：边流量 vs conductance Spearman、站域边份额、AON 丢失流量等。

用法（仓库根目录）：
  python scripts/synthetic_flow_od_gravity.py
  python scripts/validate_synthetic_flow.py --out-dir output/synthetic_flow
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="校验 synthetic_flow 目录下的 CSV/JSON")
    ap.add_argument("--out-dir", type=Path, default=REPO / "output/synthetic_flow")
    ap.add_argument("--edges", type=Path, default=REPO / "output/function/数据包/02_edges.csv")
    ap.add_argument("--units", type=Path, default=REPO / "output/function/数据包/01_units.gpkg")
    ns = ap.parse_args()
    out_dir = Path(ns.out_dir)
    report: dict = {"out_dir": str(out_dir)}

    meta_p = out_dir / "synthetic_od_meta.json"
    if meta_p.is_file():
        report["meta_snippet"] = json.loads(meta_p.read_text(encoding="utf-8"))

    aon_p = out_dir / "synthetic_edge_flow_aon_ped.csv"
    if not aon_p.is_file():
        report["status"] = "skip_edge_checks"
        report["note"] = "缺少 synthetic_edge_flow_aon_ped.csv（需带 edges 运行合成脚本且勿使用 --no-four-step-extras）"
        (out_dir / "validate_synthetic_flow.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    edges = pd.read_csv(ns.edges)
    flow = pd.read_csv(aon_p, encoding="utf-8-sig")

    def _uk(a: pd.Series, b: pd.Series) -> pd.Series:
        sa, sb = a.astype(str), b.astype(str)
        return np.where(sa <= sb, sa + "|" + sb, sb + "|" + sa)

    flow = flow.copy()
    edges = edges.copy()
    flow["_uk"] = _uk(flow["source_id"], flow["target_id"])
    edges["_uk"] = _uk(edges["source_id"], edges["target_id"])
    edges_u = edges.drop_duplicates(subset=["_uk"], keep="first")
    m = flow.merge(edges_u.drop(columns=["source_id", "target_id"]), on="_uk", how="inner")
    report["aon_merge_hit_frac"] = float(len(m) / max(len(flow), 1))

    if len(m) > 10 and "edge_conductance" in m.columns:
        report["spearman_flow_vs_conductance"] = float(
            pd.to_numeric(m["flow_ped_aon"], errors="coerce").corr(
                pd.to_numeric(m["edge_conductance"], errors="coerce"), method="spearman"
            )
        )

    if "cross_arterial" in m.columns:
        ca = pd.to_numeric(m["cross_arterial"], errors="coerce").fillna(0) > 0
        f = pd.to_numeric(m["flow_ped_aon"], errors="coerce").fillna(0.0)
        mu_ca = float(f[ca].mean()) if ca.any() else 0.0
        mu_nc = float(f[~ca].mean()) if (~ca).any() else 0.0
        report["mean_flow_ped_aon_cross_arterial"] = mu_ca
        report["mean_flow_ped_aon_no_cross_arterial"] = mu_nc

    if ns.units.is_file():
        try:
            u = gpd.read_file(ns.units, layer="units")
        except Exception:
            u = gpd.read_file(ns.units)
        if "dist_to_station" in u.columns:
            du = pd.to_numeric(u["dist_to_station"], errors="coerce")
            near = set(u.loc[du <= du.quantile(0.25), "unit_id"].astype(str))
            es = m["source_id"].astype(str)
            et = m["target_id"].astype(str)
            mask_near = es.isin(near) | et.isin(near)
            fn = pd.to_numeric(m.loc[mask_near, "flow_ped_aon"], errors="coerce").fillna(0.0)
            ff = pd.to_numeric(m.loc[~mask_near, "flow_ped_aon"], errors="coerce").fillna(0.0)
            report["mean_flow_ped_aon_near_station_quartile"] = float(fn.mean()) if len(fn) else 0.0
            report["mean_flow_ped_aon_far_station"] = float(ff.mean()) if len(ff) else 0.0

    report["status"] = "ok"
    out_json = out_dir / "validate_synthetic_flow.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
