#!/usr/bin/env python3
"""
SITE「织补」反事实：单元驱动因子微调（frozen GMM）+ 相关边权重放大 + 锚点时段多步概率扩散。

保留与 urban_global_state_gmm 相同的 feat_cols（含各类驱动因子与 site_z）。
不修改形/功/流离散标签列，仅可选微调 SITE 内连续驱动；默认小幅度。

输出：global_gmm_meta 风格的扩散参数 JSON、逐步 CSV、TV 与主导类地图序列。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from site_map_overlay import plot_site_boundary, resolve_site_json_path  # noqa: E402
from time_slice_constants import DEFAULT_ANCHOR_TID, T_ORDER  # noqa: E402
from urban_global_state_gmm import (  # noqa: E402
    RNG,
    _build_merge,
    _site_polygon_projected,
    _unit_site_zone_features,
)

GMM_FEAT_SUFFIX = (
    "feat_cols: p_M1..7, p_F1..7, p_R1..7, barrier_index, accessibility_index, "
    "poi_density, stay_proxy, green_blue_ratio, building_coverage, dist_to_station, "
    "edge_conductance_mean, site_z_iface_residential, site_z_stack_barrier, site_z_expanse_disused"
)


def _configure_plot_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def units_inside_site(units_path: Path, site_json: Path | None) -> set[str]:
    u = gpd.read_file(units_path)
    if "unit_id" not in u.columns:
        raise ValueError("units 缺少 unit_id")
    if u.crs is None:
        u = u.set_crs(4326)
    u_m = u.to_crs(32651)
    poly, _ = _site_polygon_projected(site_json if site_json and Path(site_json).is_file() else None)
    if poly is None or poly.is_empty:
        raise ValueError("无法解析 SITE 几何，请提供 --site-json")
    inside = u_m.geometry.centroid.within(poly)
    return set(u.loc[inside.fillna(False).to_numpy(), "unit_id"].astype(str))


def symmetrize_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """无向化：每条有向边添加反向，同对权重相加后用于后续归一。"""
    need = {"source_id", "target_id", "edge_weight_norm"}
    if not need.issubset(edges.columns):
        raise ValueError(f"edges 需要列 {need}")
    e = edges[list(need)].copy()
    rev = pd.DataFrame(
        {
            "source_id": e["target_id"],
            "target_id": e["source_id"],
            "edge_weight_norm": e["edge_weight_norm"],
        }
    )
    both = pd.concat([e, rev], ignore_index=True)
    both = both.groupby(["source_id", "target_id"], as_index=False)["edge_weight_norm"].sum()
    return both


def boost_edges_site(
    edges: pd.DataFrame,
    site_units: set[str],
    beta: float,
) -> pd.DataFrame:
    """对与 SITE 相交的边（端点任一端 ∈ SITE）乘以 beta，再按 source 对出边归一化。"""
    df = edges.copy()
    if not {"source_id", "target_id", "edge_weight_norm"}.issubset(df.columns):
        raise ValueError("edges 需要 source_id, target_id, edge_weight_norm")
    su = df["source_id"].astype(str)
    tu = df["target_id"].astype(str)
    mask = su.isin(site_units) | tu.isin(site_units)
    w = df["edge_weight_norm"].astype(float).copy()
    w.loc[mask] *= float(beta)
    df["edge_weight_norm"] = w
    sums = df.groupby("source_id")["edge_weight_norm"].transform("sum")
    sums = sums.replace(0.0, np.nan)
    df["edge_weight_norm"] = (df["edge_weight_norm"] / sums).fillna(0.0)
    return df


def neighbor_out_aggregate(
    edges: pd.DataFrame,
    p: np.ndarray,
    uid_index: dict[str, int],
    n: int,
    k: int,
) -> np.ndarray:
    """与 urban_global_state_gmm._neighbor_prob_matrix 单时段块一致：source 聚合 target 的 p。"""
    neigh = np.zeros((n, k))
    for _, row in edges.iterrows():
        s = str(row["source_id"])
        t = str(row["target_id"])
        w = float(row["edge_weight_norm"])
        if s not in uid_index or t not in uid_index:
            continue
        si, ti_ = uid_index[s], uid_index[t]
        neigh[si] += w * p[ti_]
    for i in range(n):
        ssum = neigh[i].sum()
        if ssum < 1e-12:
            neigh[i] = np.ones(k) / k
        else:
            neigh[i] /= ssum
    return neigh


def fit_global_gmm(X: np.ndarray, n_components: int = 7):
    """与 urban_global_state_gmm.main 中拟合逻辑对齐（diag/full，失败则降维）。"""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_comp = int(np.clip(n_components, 6, 7))
    gmm: GaussianMixture | None = None
    for cov in ("diag", "full"):
        try:
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=cov,
                random_state=RNG,
                n_init=8,
                max_iter=400,
                reg_covar=1e-5,
            )
            gmm.fit(Xs)
            break
        except ValueError:
            continue
    if gmm is None:
        n_comp = 6
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type="diag",
            random_state=RNG,
            n_init=10,
            max_iter=500,
            reg_covar=1e-4,
        )
        gmm.fit(Xs)
    return gmm, scaler, Xs.shape[1]


def apply_site_driver_intervention(
    merged: pd.DataFrame,
    site_units: set[str],
    *,
    barrier_relief: float,
    conductance_boost: float,
    green_delta: float,
    accessibility_boost: float,
    poi_density_boost: float,
) -> pd.DataFrame:
    """
    仅在 unit_id ∈ site_units 的行上微调连续驱动（不改 p_M/p_F/p_R）。
    barrier_relief: 阻隔指数乘以 (1-relief)，relief∈[0,1)
    conductance_boost: edge_conductance_mean 乘以 (1+boost)
    green_delta: green_blue_ratio 加上 delta 后 clip [0,1]
    accessibility_boost / poi_density_boost: 乘以 (1+boost)，保守默认可设 0
    """
    out = merged.copy()
    su = out["unit_id"].astype(str).isin(site_units)
    if not su.any():
        raise ValueError("SITE 内没有匹配到任何 unit_id，请检查 SITE 与 units 坐标系")

    if "barrier_index" in out.columns:
        b = pd.to_numeric(out.loc[su, "barrier_index"], errors="coerce").fillna(0.0)
        out.loc[su, "barrier_index"] = (b * (1.0 - float(barrier_relief))).clip(lower=0.0)

    if "edge_conductance_mean" in out.columns:
        ec = pd.to_numeric(out.loc[su, "edge_conductance_mean"], errors="coerce").fillna(0.0)
        out.loc[su, "edge_conductance_mean"] = ec * (1.0 + float(conductance_boost))

    if "green_blue_ratio" in out.columns and abs(green_delta) > 1e-12:
        g = pd.to_numeric(out.loc[su, "green_blue_ratio"], errors="coerce").fillna(0.0)
        out.loc[su, "green_blue_ratio"] = (g + float(green_delta)).clip(0.0, 1.0)

    if "accessibility_index" in out.columns and abs(accessibility_boost) > 1e-12:
        a = pd.to_numeric(out.loc[su, "accessibility_index"], errors="coerce").fillna(0.0)
        out.loc[su, "accessibility_index"] = a * (1.0 + float(accessibility_boost))

    if "poi_density" in out.columns and abs(poi_density_boost) > 1e-12:
        p = pd.to_numeric(out.loc[su, "poi_density"], errors="coerce").fillna(0.0)
        out.loc[su, "poi_density"] = p * (1.0 + float(poi_density_boost))

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="SITE 织补：驱动微调 + 边权 + 多步扩散（frozen GMM）")
    ap.add_argument("--morph", type=Path, default=Path("data/morph_state.csv"))
    ap.add_argument("--func", type=Path, default=Path("output/function/func_state.csv"))
    ap.add_argument("--mob", type=Path, default=Path("data/mob_state.csv"))
    ap.add_argument("--units", type=Path, default=Path("output/function/数据包/01_units.gpkg"))
    ap.add_argument("--edges", type=Path, default=Path("output/function/数据包/02_edges.csv"))
    ap.add_argument("--site-json", type=Path, default=None, help="默认与 urban_global_state_gmm 一致")
    ap.add_argument("--out-dir", type=Path, default=Path("output/site_weave_diffusion"))
    ap.add_argument("--anchor-tid", type=str, default=DEFAULT_ANCHOR_TID)
    ap.add_argument("--n-components", type=int, default=7)
    ap.add_argument("--barrier-relief", type=float, default=0.06, help="SITE 内 barrier_index 乘以 (1-relief)")
    ap.add_argument("--conductance-boost", type=float, default=0.08, help="SITE 内 edge_conductance_mean 乘以 (1+boost)")
    ap.add_argument("--green-delta", type=float, default=0.015, help="SITE 内 green_blue_ratio +delta，clip 0–1")
    ap.add_argument("--accessibility-boost", type=float, default=0.0, help="SITE 内 accessibility 乘以 (1+boost)，默认 0 极保守")
    ap.add_argument("--poi-density-boost", type=float, default=0.0, help="SITE 内 poi_density 乘以 (1+boost)，默认 0")
    ap.add_argument("--edge-beta", type=float, default=1.12, help="触及 SITE 的边 edge_weight_norm 放大系数")
    ap.add_argument("--no-symmetrize-edges", action="store_true", help="不对边做无向加倍（默认加倍以利于扩散）")
    ap.add_argument("--diffusion-steps", type=int, default=8)
    ap.add_argument("--diffusion-alpha", type=float, default=0.38, help="每步 p←(1-a)*p + a*neigh")
    args = ap.parse_args()

    site_path = args.site_json if args.site_json is not None and args.site_json.is_file() else resolve_site_json_path()
    if site_path is None:
        print("ERROR: 未找到 SITE.json", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_fonts()

    site_units = units_inside_site(args.units, Path(site_path))
    if not site_units:
        print("ERROR: SITE 多边形内未包含任何单元质心，请检查 SITE.json 与 01_units", file=sys.stderr)
        return 1
    merged = _build_merge(args.morph, args.func, args.mob, args.units)
    units_gdf = gpd.read_file(args.units)
    zone_df = _unit_site_zone_features(units_gdf, Path(site_path))
    merged = merged.merge(zone_df, on="unit_id", how="left")
    for zc in ("site_z_iface_residential", "site_z_stack_barrier", "site_z_expanse_disused"):
        merged[zc] = merged[zc].fillna(0.0)
    merged["_t_ord"] = merged["t_id"].map({t: i for i, t in enumerate(T_ORDER)})
    merged = merged.sort_values(["_t_ord", "unit_id"]).drop(columns=["_t_ord"]).reset_index(drop=True)

    p_m = [f"p_M{i}" for i in range(1, 8)]
    p_f = [f"p_F{i}" for i in range(1, 8)]
    p_r = [f"p_R{i}" for i in range(1, 8)]
    drv_cols = [
        "barrier_index",
        "accessibility_index",
        "poi_density",
        "stay_proxy",
        "green_blue_ratio",
        "building_coverage",
        "dist_to_station",
        "edge_conductance_mean",
    ]
    zcols = ["site_z_iface_residential", "site_z_stack_barrier", "site_z_expanse_disused"]
    feat_cols = p_m + p_f + p_r + drv_cols + zcols

    X_base = np.nan_to_num(merged[feat_cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    gmm, scaler, _ = fit_global_gmm(X_base, int(np.clip(args.n_components, 6, 7)))

    merged_iv = apply_site_driver_intervention(
        merged,
        site_units,
        barrier_relief=args.barrier_relief,
        conductance_boost=args.conductance_boost,
        green_delta=args.green_delta,
        accessibility_boost=args.accessibility_boost,
        poi_density_boost=args.poi_density_boost,
    )
    merged_iv.to_csv(out_dir / "urban_state_input_after_site_drivers.csv", index=False)
    X_iv = np.nan_to_num(merged_iv[feat_cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    Xs_base = scaler.transform(X_base)
    Xs_iv = scaler.transform(X_iv)
    p_base = gmm.predict_proba(Xs_base)
    p_iv = gmm.predict_proba(Xs_iv)
    k = p_base.shape[1]

    unit_ids = sorted(merged["unit_id"].unique().tolist())
    uid_index = {u: i for i, u in enumerate(unit_ids)}
    n = len(unit_ids)
    t_index = {t: i for i, t in enumerate(T_ORDER)}
    if args.anchor_tid not in t_index:
        raise ValueError(f"anchor-tid 必须是 {T_ORDER} 之一")

    uid_arr = merged["unit_id"].astype(str).values
    tid_arr = merged["t_id"].astype(str).values

    def row_idx_anchor(uid: str) -> int:
        hit = np.where((uid_arr == uid) & (tid_arr == args.anchor_tid))[0]
        if len(hit) == 1:
            return int(hit[0])
        hit2 = np.where(uid_arr == uid)[0]
        if len(hit2) == 0:
            raise ValueError(f"unit_id={uid} 在 merged 中不存在")
        best_ord = -1
        best_i = int(hit2[0])
        for i in hit2.flat:
            t = str(tid_arr[int(i)])
            if t in t_index:
                o = t_index[t]
                if o > best_ord:
                    best_ord = o
                    best_i = int(i)
        return best_i

    p_ref = np.zeros((n, k))
    p_curr = np.zeros((n, k))
    for ui, uid in enumerate(unit_ids):
        ri = row_idx_anchor(uid)
        p_ref[ui] = p_base[ri]
        p_curr[ui] = p_base[ri]
    # SITE 内用干预后分布，外保持基线（锚点切片）
    for ui, uid in enumerate(unit_ids):
        if uid in site_units:
            ri = row_idx_anchor(uid)
            p_curr[ui] = p_iv[ri]

    edges_raw = pd.read_csv(args.edges)
    if not args.no_symmetrize_edges:
        edges_work = symmetrize_edges(edges_raw)
    else:
        edges_work = edges_raw[["source_id", "target_id", "edge_weight_norm"]].copy()

    edges_boosted = boost_edges_site(edges_work, site_units, args.edge_beta)

    meta = {
        "method": "site_weave_diffusion",
        "gmm_random_state": RNG,
        "n_components": int(k),
        "feature_columns": feat_cols,
        "feature_note": GMM_FEAT_SUFFIX,
        "site_json": str(site_path),
        "site_unit_count": len(site_units),
        "anchor_tid": args.anchor_tid,
        "anchor_row_fallback": "若某 unit 无锚点时段行，则用该 unit 在 T_ORDER 中最晚可用时段所在行",
        "driver_intervention": {
            "barrier_relief": args.barrier_relief,
            "conductance_boost": args.conductance_boost,
            "green_delta": args.green_delta,
            "accessibility_boost": args.accessibility_boost,
            "poi_density_boost": args.poi_density_boost,
            "scope": "SITE unit rows only; columns as listed; p_M/p_F/p_R untouched",
        },
        "edges": {
            "symmetrize": not args.no_symmetrize_edges,
            "beta": args.edge_beta,
            "mask": "source_id or target_id in SITE units",
        },
        "diffusion": {
            "steps": int(args.diffusion_steps),
            "alpha": float(args.diffusion_alpha),
            "mix": "p^{h+1} = (1-alpha)*p^h + alpha*Neigh(p^h); Neigh uses boosted edge_weight_norm",
        },
    }
    (out_dir / "site_weave_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # —— 逐步扩散 ——
    alpha = float(args.diffusion_alpha)
    steps = int(args.diffusion_steps)
    tv_series: list[float] = []

    def tv_l1(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
        return 0.5 * np.abs(pa - pb).sum(axis=1)

    for h in range(steps + 1):
        neigh = neighbor_out_aggregate(edges_boosted, p_curr, uid_index, n, k)
        tv = tv_l1(p_curr, p_ref)
        tv_series.append(float(tv.mean()))
        step_df = pd.DataFrame(
            {
                "unit_id": unit_ids,
                "step": h,
                "tv_from_baseline_anchor": tv,
                "argmax_component": np.argmax(p_curr, axis=1),
            }
        )
        for j in range(k):
            step_df[f"p_G{j + 1}"] = p_curr[:, j]
        step_df.to_csv(out_dir / f"p_anchor_{args.anchor_tid}_step_{h:02d}.csv", index=False)

        if h < steps:
            p_curr = (1.0 - alpha) * p_curr + alpha * neigh
            row_sums = p_curr.sum(axis=1, keepdims=True)
            row_sums[row_sums < 1e-15] = 1.0
            p_curr = p_curr / row_sums

    pd.DataFrame({"step": range(steps + 1), "mean_tv_vs_baseline": tv_series}).to_csv(
        out_dir / "diffusion_tv_curve.csv", index=False
    )

    # —— 图：TV 曲线 + 若干步地图 ——
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(steps + 1), tv_series, "o-", color="steelblue")
    ax.set_xlabel("扩散步")
    ax.set_ylabel("锚点时段 mean TV（相对基线）")
    ax.set_title(f"织补反事实 · {args.anchor_tid} · TV 均值随步变化")
    fig.savefig(fig_dir / "diffusion_mean_tv_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    plot_steps = sorted(set([0, min(2, steps), min(4, steps), steps]))
    lbl_codes = [f"G{i + 1}" for i in range(k)]
    cmap = plt.colormaps["tab10"].resampled(max(10, k))
    cat_to_c = {lbl_codes[i]: mcolors.to_hex(cmap(i)) for i in range(k)}

    for h in plot_steps:
        path_csv = out_dir / f"p_anchor_{args.anchor_tid}_step_{h:02d}.csv"
        sdf = pd.read_csv(path_csv)
        mg = units_gdf.merge(sdf[["unit_id", "tv_from_baseline_anchor"]], on="unit_id", how="left")
        fig, ax = plt.subplots(figsize=(10, 10))
        vmax = float(np.nanpercentile(mg["tv_from_baseline_anchor"].to_numpy(), 98)) if mg["tv_from_baseline_anchor"].notna().any() else 1.0
        vmax = max(vmax, 1e-6)
        mg.plot(
            column="tv_from_baseline_anchor",
            ax=ax,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            legend=True,
            legend_kwds={"shrink": 0.55, "label": "TV vs baseline"},
            missing_kwds={"color": "#e8e8e8"},
            edgecolor="0.18",
            linewidth=0.08,
        )
        plot_site_boundary(ax, mg.crs, Path(site_path))
        ax.set_title(f"织补扩散 · 步 {h} · {args.anchor_tid} · TV（相对基线锚点分布）")
        ax.axis("off")
        fig.savefig(fig_dir / f"map_tv_step_{h:02d}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

        mg2 = units_gdf.merge(sdf[["unit_id", "argmax_component"]], on="unit_id", how="left")
        mg2["lbl"] = mg2["argmax_component"].map(lambda i: lbl_codes[int(i)] if pd.notna(i) else "NA")
        cats = list(pd.unique(mg2["lbl"].dropna()))
        cols = mg2["lbl"].map(lambda x: cat_to_c.get(x, "#cccccc"))
        fig, ax = plt.subplots(figsize=(10, 10))
        mg2.plot(color=cols.fillna("#e8e8e8"), ax=ax, edgecolor="0.2", linewidth=0.1)
        site_ok = plot_site_boundary(ax, mg2.crs, Path(site_path))
        ps = [Rectangle((0, 0), 1, 1, fc=cat_to_c[c]) for c in lbl_codes if c in cats]
        leg = [c for c in lbl_codes if c in cats]
        if site_ok:
            ps.append(Line2D([0], [0], color="#d90429", lw=2.2, linestyle=(0, (5, 3))))
            leg.append("SITE")
        ax.legend(ps, leg, loc="lower left", fontsize=7)
        ax.set_title(f"织补扩散 · 步 {h} · 主导综合类 G")
        ax.axis("off")
        fig.savefig(fig_dir / f"map_argmax_step_{h:02d}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    print("Wrote:", out_dir / "site_weave_meta.json")
    print("Figures:", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
