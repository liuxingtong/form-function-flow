#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from shapely.ops import unary_union

from site_map_overlay import plot_site_boundary, resolve_site_json_path
from urban_global_state_gmm import _site_polygon_projected


REPO = Path(__file__).resolve().parents[1]
SITE_WEAVE_DIR = REPO / "output" / "site_weave_diffusion"
FIG_DIR = SITE_WEAVE_DIR / "figures"
UNITS_PATH = REPO / "output" / "function" / "数据包" / "01_units.gpkg"
EDGES_PATH = REPO / "output" / "function" / "数据包" / "02_edges.csv"
HOUSING_GRID_PATH = REPO / "data" / "site_3km" / "08-房价数据" / "网格房价" / "geojson" / "上海房价数据.geojson"


def _configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _load_site_units(units_gdf: gpd.GeoDataFrame, site_path: Path) -> set[str]:
    ug = units_gdf.copy()
    if ug.crs is None:
        ug = ug.set_crs(4326)
    ug_m = ug.to_crs(32651)
    site_poly, _ = _site_polygon_projected(site_path)
    if site_poly is None or site_poly.is_empty:
        return set()
    inside = ug_m.geometry.centroid.within(site_poly)
    return set(ug.loc[inside.fillna(False).to_numpy(), "unit_id"].astype(str))


def _row_norm_adjacency(
    unit_ids: list[str],
    edges_path: Path,
    site_units: set[str],
    *,
    edge_beta: float,
    symmetrize: bool,
) -> np.ndarray:
    edges = pd.read_csv(edges_path)
    e = edges[["source_id", "target_id", "edge_weight_norm"]].copy()
    e["source_id"] = e["source_id"].astype(str)
    e["target_id"] = e["target_id"].astype(str)
    if symmetrize:
        rev = pd.DataFrame(
            {
                "source_id": e["target_id"],
                "target_id": e["source_id"],
                "edge_weight_norm": e["edge_weight_norm"],
            }
        )
        e = pd.concat([e, rev], ignore_index=True)
        e = e.groupby(["source_id", "target_id"], as_index=False)["edge_weight_norm"].sum()

    mask = e["source_id"].isin(site_units) | e["target_id"].isin(site_units)
    e.loc[mask, "edge_weight_norm"] = e.loc[mask, "edge_weight_norm"].astype(float) * float(edge_beta)

    idx = {u: i for i, u in enumerate(unit_ids)}
    n = len(unit_ids)
    w = np.zeros((n, n), dtype=float)
    for _, r in e.iterrows():
        s = r["source_id"]
        t = r["target_id"]
        if s not in idx or t not in idx:
            continue
        w[idx[s], idx[t]] += float(r["edge_weight_norm"])
    rs = w.sum(axis=1, keepdims=True)
    rs[rs <= 1e-12] = 1.0
    w = w / rs
    return w


def plot_form_barrier_story(meta: dict, site_path: Path) -> None:
    df = pd.read_csv(SITE_WEAVE_DIR / "urban_state_input_after_site_drivers.csv")
    units = gpd.read_file(UNITS_PATH)
    units["unit_id"] = units["unit_id"].astype(str)

    anchor_tid = str(meta.get("anchor_tid", "WE_NT"))
    dfa = df.loc[df["t_id"].astype(str) == anchor_tid].copy()
    if dfa.empty:
        pref_order = ["WD_AM", "WD_PM", "WD_EVE", "WD_NT", "WE_AM", "WE_MD", "WE_EVE", "WE_NT", "WE_PM"]
        avail = [t for t in pref_order if t in set(df["t_id"].astype(str))]
        if not avail:
            raise ValueError("urban_state_input_after_site_drivers.csv 中没有可用 t_id")
        anchor_tid = avail[-1]
        dfa = df.loc[df["t_id"].astype(str) == anchor_tid].copy()
    dfa["unit_id"] = dfa["unit_id"].astype(str)
    dfa = dfa.drop_duplicates("unit_id")

    site_units = _load_site_units(units, site_path)
    uid = sorted(set(units["unit_id"]).intersection(set(dfa["unit_id"])))
    n = len(uid)
    if n == 0:
        raise ValueError("units 与 urban_state_input_after_site_drivers.csv 的 unit_id 无交集")

    site_mask = np.array([u in site_units for u in uid], dtype=bool)
    if not site_mask.any():
        raise ValueError("SITE 单元未匹配到当前锚点时段样本")
    if (~site_mask).sum() == 0:
        raise ValueError("当前样本全部落在 SITE 内，无法形成扩散对照")
    base_barrier = np.zeros(n, dtype=float)
    dfx = dfa.set_index("unit_id")
    for i, u in enumerate(uid):
        base_barrier[i] = float(dfx.at[u, "barrier_index"])

    drv = meta.get("driver_intervention", {})
    barrier_relief = float(drv.get("barrier_relief", 0.06))
    conductance_boost = float(drv.get("conductance_boost", 0.08))
    green_delta = float(drv.get("green_delta", 0.015))
    acc_boost = float(drv.get("accessibility_boost", 0.0))
    poi_boost = float(drv.get("poi_density_boost", 0.0))
    diff = meta.get("diffusion", {})
    steps = int(diff.get("steps", 8))
    alpha = float(diff.get("alpha", 0.38))
    edges_meta = meta.get("edges", {})
    edge_beta = float(edges_meta.get("beta", 1.12))
    symmetrize = bool(edges_meta.get("symmetrize", True))

    w = _row_norm_adjacency(uid, EDGES_PATH, site_units, edge_beta=edge_beta, symmetrize=symmetrize)

    x = base_barrier.copy()
    step_pick = [0, min(2, steps), min(4, steps), steps]
    states: dict[int, np.ndarray] = {}
    curve = []

    for h in range(steps + 1):
        progress = h / max(steps, 1)
        x[site_mask] = base_barrier[site_mask] * (1.0 - barrier_relief * progress)
        states[h] = x.copy()
        curve.append(
            {
                "step": h,
                "site_barrier_mean": float(x[site_mask].mean()),
                "outside_barrier_mean": float(x[~site_mask].mean()),
                "barrier_drop_vs_base_site_pct": float(
                    100.0 * (x[site_mask].mean() - base_barrier[site_mask].mean()) / (abs(base_barrier[site_mask].mean()) + 1e-9)
                ),
            }
        )
        neigh = w @ x
        x = (1.0 - alpha) * x + alpha * neigh

    pd.DataFrame(curve).to_csv(SITE_WEAVE_DIR / "form_barrier_diffusion_curve.csv", index=False)

    map_df = units[units["unit_id"].isin(uid)].copy()
    vals = np.concatenate([states[h] for h in step_pick])
    vmin, vmax = float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98))
    vmin = min(vmin, float(vals.min()))
    vmax = max(vmax, float(vals.max()))

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    axes = axes.ravel()
    for ax, h in zip(axes, step_pick):
        vv = pd.DataFrame({"unit_id": uid, "barrier_sim": states[h]})
        mg = map_df.merge(vv, on="unit_id", how="left")
        mg.plot(
            column="barrier_sim",
            ax=ax,
            cmap="magma_r",
            vmin=vmin,
            vmax=vmax,
            edgecolor="0.25",
            linewidth=0.08,
            legend=False,
            missing_kwds={"color": "#ededed"},
        )
        plot_site_boundary(ax, mg.crs, site_path)
        ax.set_title(f"Form 阻隔扩散步 {h} / {steps} · 锚点 {anchor_tid}")
        ax.axis("off")

    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="magma_r")
    sm.set_array([])
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.915, 0.22, 0.018, 0.56])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("模拟阻隔度（barrier index diffusion）")

    note = (
        "SITE 织补区驱动变化（用于触发阻隔下降）\n"
        f"- barrier_index: -{barrier_relief * 100:.1f}%\n"
        f"- edge_conductance_mean: +{conductance_boost * 100:.1f}%\n"
        f"- green_blue_ratio: +{green_delta:.3f}\n"
        f"- accessibility_index: +{acc_boost * 100:.1f}%\n"
        f"- poi_density: +{poi_boost * 100:.1f}%"
    )
    fig.text(
        0.02,
        0.03,
        note,
        fontsize=11,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#999999"),
    )

    fig.suptitle("织补扩散（Form）: 阻隔性指标逐步减弱", fontsize=17, y=0.98)
    fig.tight_layout(rect=[0, 0.07, 0.9, 0.96])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "weave_form_barrier_diffusion_story.png", dpi=180)
    plt.close(fig)

    # 曲线图（附加输出）
    cd = pd.DataFrame(curve)
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(cd["step"], cd["site_barrier_mean"], "o-", color="#d1495b", label="SITE 内均值")
    ax2.plot(cd["step"], cd["outside_barrier_mean"], "o-", color="#2a9d8f", label="SITE 外均值")
    ax2.set_xlabel("扩散步")
    ax2.set_ylabel("模拟阻隔度")
    ax2.set_title("阻隔度均值随扩散步变化")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "weave_form_barrier_diffusion_curve.png", dpi=180)
    plt.close(fig2)


def _knn_row_norm_weights(xy: np.ndarray, k: int = 6) -> np.ndarray:
    n = xy.shape[0]
    d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    order = np.argpartition(d2, kth=min(k, n - 1), axis=1)[:, :k]
    w = np.zeros((n, n), dtype=float)
    for i in range(n):
        nn = order[i]
        dd = np.sqrt(d2[i, nn])
        ww = 1.0 / np.maximum(dd, 1.0)
        ww = ww / np.maximum(ww.sum(), 1e-12)
        w[i, nn] = ww
    return w


def plot_housing_story(site_path: Path) -> None:
    g = gpd.read_file(HOUSING_GRID_PATH)
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326).copy()
    g = g[g["avgprice"].notna()].copy()
    g["avgprice"] = pd.to_numeric(g["avgprice"], errors="coerce")
    g = g[g["avgprice"].notna()].copy()

    site_gdf = gpd.read_file(site_path)
    if site_gdf.crs is None:
        site_gdf = site_gdf.set_crs(4326)
    site_center = unary_union(site_gdf.to_crs(4326).geometry).centroid

    gm = g.to_crs(32651)
    cc = gm.geometry.centroid
    site_m = site_gdf.to_crs(32651)
    sp = unary_union(site_m.geometry).centroid

    north = cc.y.to_numpy() >= float(sp.y)
    south = ~north
    base = g["avgprice"].to_numpy(dtype=float)

    # 初始态：人为强化“北弱南强”对比，再让扩散把北部带起来。
    p = base.copy()
    p[north] *= 0.76
    p[south] *= 1.10

    xy = np.column_stack([cc.x.to_numpy(), cc.y.to_numpy()])
    w = _knn_row_norm_weights(xy, k=6)

    dist = np.sqrt((xy[:, 0] - sp.x) ** 2 + (xy[:, 1] - sp.y) ** 2)
    kernel = np.exp(-((dist / 1400.0) ** 2))
    north_kernel = kernel * north.astype(float)
    target = base * (1.0 + 0.55 * north_kernel)

    steps = 8
    alpha = 0.42
    gamma = 0.95
    pick = [0, 2, 4, 8]
    states: dict[int, np.ndarray] = {}
    curve = []
    for h in range(steps + 1):
        states[h] = p.copy()
        north_mean = float(p[north].mean())
        south_mean = float(p[south].mean())
        curve.append(
            {
                "step": h,
                "north_mean_price": north_mean,
                "south_mean_price": south_mean,
                "north_to_south_ratio": north_mean / max(south_mean, 1e-9),
            }
        )
        neigh = w @ p
        p = (1.0 - alpha) * p + alpha * neigh
        p = p + gamma * north_kernel * (target - p)

    pd.DataFrame(curve).to_csv(SITE_WEAVE_DIR / "housing_north_south_evolution_curve.csv", index=False)

    vals = np.concatenate([states[h] for h in pick])
    vmin, vmax = float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98))
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    axes = axes.ravel()
    for ax, h in zip(axes, pick):
        gg = g.copy()
        gg["price_sim"] = states[h]
        gg.plot(
            column="price_sim",
            ax=ax,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolor="#f2f2f2",
            linewidth=0.12,
            legend=False,
        )
        plot_site_boundary(ax, gg.crs, site_path)
        ax.axhline(float(site_center.y), color="#ffffff", lw=1.1, ls=(0, (3, 3)))
        ax.text(0.02, 0.95, "北", transform=ax.transAxes, color="white", fontsize=11, fontweight="bold")
        ax.text(0.02, 0.03, "南", transform=ax.transAxes, color="white", fontsize=11, fontweight="bold")
        ax.set_title(f"房价演变步 {h} / {steps}")
        ax.axis("off")

    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
    sm.set_array([])
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.915, 0.22, 0.018, 0.56])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("模拟房价（元/㎡）")

    cd = pd.DataFrame(curve)
    ratio0 = float(cd.loc[cd["step"] == 0, "north_to_south_ratio"].iloc[0])
    ratio8 = float(cd.loc[cd["step"] == steps, "north_to_south_ratio"].iloc[0])
    txt = (
        "扩散逻辑：\n"
        "- 初始态强化“北弱南强”（北-24%，南+10%）\n"
        "- kNN 邻域价格扩散（k=6）\n"
        "- SITE 北向织补增益（靠近 site 北侧增益更高）\n"
        f"- 北/南均价比：{ratio0:.3f} → {ratio8:.3f}"
    )
    fig.text(
        0.02,
        0.03,
        txt,
        fontsize=11,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#999999"),
    )
    fig.suptitle("织补扩散（房价）: 北部由弱转升的演变", fontsize=17, y=0.98)
    fig.tight_layout(rect=[0, 0.08, 0.9, 0.96])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "weave_housing_price_north_south_evolution.png", dpi=180)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cd["step"], cd["north_mean_price"], "o-", color="#3a86ff", label="北部均价")
    ax.plot(cd["step"], cd["south_mean_price"], "o-", color="#ff006e", label="南部均价")
    ax2 = ax.twinx()
    ax2.plot(cd["step"], cd["north_to_south_ratio"], "s--", color="#8338ec", label="北/南比")
    ax.set_xlabel("扩散步")
    ax.set_ylabel("均价（元/㎡）")
    ax2.set_ylabel("北/南均价比")
    ax.set_title("北南房价均值演变曲线")
    ax.grid(alpha=0.25)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "weave_housing_price_north_south_curve.png", dpi=180)
    plt.close(fig2)


def main() -> int:
    _configure_fonts()
    site_path = resolve_site_json_path()
    if site_path is None:
        raise FileNotFoundError("未找到 SITE.json")
    meta_path = SITE_WEAVE_DIR / "site_weave_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    plot_form_barrier_story(meta, site_path)
    plot_housing_story(site_path)
    print("Wrote story figures to:", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
