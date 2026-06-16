#!/usr/bin/env python3
"""
分时段 OD 长表（如 ``synthetic_od_by_period_long.csv``）制图：八时段各一张 **Top-K 期望线网**
（origin→destination 质心连线，颜色/粗细按 flow）。

默认按 ``time_slice_constants.T_IDS`` 顺序排布 4×2 子图。

用法（仓库根目录）：
  python scripts/plot_synthetic_od_period_network.py ^
    --units output/function/数据包/01_units.gpkg ^
    --od-csv output/synthetic_flow_worldpop/synthetic_od_by_period_long.csv ^
    --out-png output/synthetic_flow_worldpop/od_network_eight_periods.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_flow_modality_networks import configure_cn_font  # noqa: E402
from site_map_overlay import plot_site_boundary, resolve_site_json_path  # noqa: E402
from time_slice_constants import T_IDS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CRS_M = "EPSG:32651"


def _unit_centroids_wgs84(units: gpd.GeoDataFrame) -> dict[str, tuple[float, float]]:
    um = units.copy()
    if um.crs is None:
        um = um.set_crs(4326)
    um = um.to_crs(CRS_M)
    um["geometry"] = um.geometry.centroid
    uwgs = um.to_crs(4326)
    out: dict[str, tuple[float, float]] = {}
    for uid, geom in zip(uwgs["unit_id"].astype(str), uwgs.geometry):
        if geom is None or geom.is_empty:
            continue
        out[str(uid)] = (float(geom.x), float(geom.y))
    return out


def main() -> int:
    configure_cn_font()
    ap = argparse.ArgumentParser(description="八时段（或任意 t_id）OD 期望线网络总览")
    ap.add_argument("--units", type=Path, default=REPO / "output" / "function" / "数据包" / "01_units.gpkg")
    ap.add_argument(
        "--od-csv",
        type=Path,
        default=REPO / "output" / "synthetic_flow_worldpop" / "synthetic_od_by_period_long.csv",
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=REPO / "output" / "synthetic_flow_worldpop" / "od_network_eight_periods.png",
    )
    ap.add_argument("--site-json", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--top-per-period", type=int, default=2500, help="每时段保留的 OD 边数上限（按 flow 降序）")
    ap.add_argument(
        "--min-dist-m",
        type=float,
        default=0.0,
        help="仅保留 dist_m≥该值的 OD（与 plot_synthetic_od_gravity_maps 一致；默认不过滤）",
    )
    ap.add_argument("--percentile-cap", type=float, default=98.0, help="期望线色标 flow 上限分位")
    ap.add_argument("--log-scale", action="store_true", help="期望线色标用对数")
    ns = ap.parse_args()

    od_path = Path(ns.od_csv)
    if not od_path.is_file():
        raise SystemExit(f"Missing {od_path}")

    site_path = Path(ns.site_json) if ns.site_json is not None and Path(ns.site_json).is_file() else resolve_site_json_path()

    try:
        units = gpd.read_file(ns.units, layer="units")
    except Exception:
        units = gpd.read_file(ns.units)
    if units.crs is None:
        units = units.set_crs(4326)
    if "unit_id" not in units.columns:
        raise SystemExit("units 缺少 unit_id")
    units_wgs = units.to_crs(4326)
    cents = _unit_centroids_wgs84(units)
    bb = units_wgs.total_bounds
    pad = 0.002

    df = pd.read_csv(od_path, encoding="utf-8-sig")
    for c in ("origin_id", "destination_id", "flow", "t_id"):
        if c not in df.columns:
            raise SystemExit(f"od-csv 缺少列 {c}")
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce").fillna(0.0)

    t_order = [t for t in T_IDS if t in set(df["t_id"].astype(str).unique())]
    if not t_order:
        t_order = sorted(df["t_id"].astype(str).unique().tolist())

    n_panels = len(t_order)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig_w = 13.5
    fig_h = max(3.4 * nrows, 8.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=int(ns.dpi), layout="constrained")
    axes_flat = np.atleast_1d(axes).ravel()

    top_k = max(1, int(ns.top_per_period))
    md = float(ns.min_dist_m)
    pct = float(ns.percentile_cap)

    for ax, t_id in zip(axes_flat, t_order):
        sub = df[df["t_id"].astype(str).eq(str(t_id))].copy()
        sub = sub[sub["origin_id"].astype(str) != sub["destination_id"].astype(str)]
        if md > 0.0 and "dist_m" in sub.columns:
            sub["dist_m"] = pd.to_numeric(sub["dist_m"], errors="coerce").fillna(0.0)
            sub = sub[sub["dist_m"] >= md]
        sub = sub.nlargest(min(top_k, len(sub)), "flow", keep="first")

        units_wgs.plot(ax=ax, color="#f0f0f0", edgecolor="#d4d4d4", linewidth=0.05)
        plot_site_boundary(ax, units_wgs.crs, site_path)

        segs: list[list[tuple[float, float]]] = []
        vals: list[float] = []
        for r in sub.itertuples(index=False):
            o, d = str(r.origin_id), str(r.destination_id)
            po, pd_ = cents.get(o), cents.get(d)
            if po is None or pd_ is None:
                continue
            segs.append([po, pd_])
            vals.append(float(r.flow))

        if segs:
            v = np.asarray(vals, dtype=float)
            pos = v[v > 0]
            if pos.size:
                lo = float(np.min(pos))
                hi = float(np.percentile(pos, min(100.0, pct)))
                if hi <= lo:
                    hi = float(np.max(pos))
            else:
                lo, hi = 0.0, 1.0
            if bool(ns.log_scale) and lo > 0:
                norm: LogNorm | Normalize = LogNorm(vmin=max(lo, 1e-18), vmax=max(hi, lo * 1.01))
            else:
                norm = Normalize(vmin=0.0, vmax=max(hi, 1e-18), clip=True)
            lw = 0.35 + 1.15 * np.sqrt(np.clip((v - lo) / max(hi - lo, 1e-18), 0.0, 1.0))
            lc = LineCollection(
                segs,
                cmap="inferno",
                array=v,
                norm=norm,
                linewidths=lw,
                alpha=0.5,
                zorder=2,
            )
            ax.add_collection(lc)

        tot = float(sub["flow"].sum()) if len(sub) else 0.0
        ax.set_title(f"{t_id}  ·  Top-{len(segs)} OD  ·  Σflow≈{tot:.4g}", fontsize=10)
        ax.set_xlim(float(bb[0] - pad), float(bb[2] + pad))
        ax.set_ylim(float(bb[1] - pad), float(bb[3] + pad))
        ax.set_aspect("equal")
        ax.axis("off")

    for j in range(len(t_order), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "合成 OD 期望线网（分时段；按 flow 取 Top-K；线宽随 flow 略变）",
        fontsize=12,
    )
    out_png = Path(ns.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(ns.dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
