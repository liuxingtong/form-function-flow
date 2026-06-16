#!/usr/bin/env python3
"""
综合城市状态：GMM 融合 morph/func/mob 概率 + 驱动因子，图邻域加权与情景推演。

严格使用 sklearn.mixture.GaussianMixture（不使用 KMeans/HDBSCAN/HMM）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from site_map_overlay import plot_site_boundary, resolve_site_json_path  # noqa: E402
from time_slice_constants import DEFAULT_ANCHOR_TID, T_CYCL_PAIRS, T_ORDER  # noqa: E402

RNG = 42
# 情景推演总开关：关闭时不计算 pn_cache、不写 state_transition_result、不输出 G03–G09 / G_scenario_* / G08。
ENABLE_SCENARIO_ENGINE = False
# 情景推演：约 30% 单元主导类相对现状可变化，多套情景各用互斥单元子集（仅当 ENABLE_SCENARIO_ENGINE 且下列长度>1 时生效）。
TARGET_SCENARIO_FLIP_FRAC = 0.30
FLIP_FRAC_TOL = 0.02
T_PAIRS = list(T_CYCL_PAIRS)

# 默认仅保留「现状」占位；扩展情景请在 ENABLE_SCENARIO_ENGINE=True 时追加条目。
SCENARIOS = [
    {
        "scenario_id": 0,
        "name": "现状延续",
        "connectivity_boost": 0.0,
        "program_boost": 0.0,
        "transfer_relief": 0.0,
        "public_space_boost": 0.0,
        "barrier_reduction": 0.0,
        "micro_transit_boost": 0.0,
        "cycling_slow_boost": 0.0,
        "night_economy_boost": 0.0,
        "freight_penalty": 0.0,
        "heritage_tread_soft": 0.0,
        "emergency_access_boost": 0.0,
    },
]


def _prob_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    pat = re.compile(rf"^{re.escape(prefix)}\d+$")
    return sorted([c for c in df.columns if pat.match(c)], key=lambda x: int(x[len(prefix) :]))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)


def _build_merge(
    morph_path: Path,
    func_path: Path,
    mob_path: Path,
    units_path: Path,
) -> pd.DataFrame:
    morph = pd.read_csv(morph_path)
    func = pd.read_csv(func_path)
    # 功能层 CSV 新版 p_F0…p_F5 与综合态脚本期望的 p_F1…p_F6（分量 0…5）对齐
    if "p_F0" in func.columns:
        func = func.rename(columns={f"p_F{i}": f"p_F{i + 1}" for i in range(6)})
    mob = pd.read_csv(mob_path)
    # 与 time_slice_constants.T_IDS 对齐：旧 CSV 偶见 WE_PM，与标准 WE_MD 同义
    if "t_id" in mob.columns:
        mob["t_id"] = mob["t_id"].replace({"WE_PM": "WE_MD"})
    units = gpd.read_file(units_path)
    if "unit_id" not in units.columns:
        raise ValueError("01_units.gpkg 缺少 unit_id")
    ucols = ["unit_id", "dist_to_station"]
    for c in ("dist_to_station",):
        if c not in units.columns:
            raise ValueError(f"01_units.gpkg 缺少 {c}")
    units_sub = units[ucols].copy()

    # pad probabilities to max 7 for consistent columns
    def pad_probs(df0: pd.DataFrame, prefix: str, n: int) -> pd.DataFrame:
        out = df0.copy()
        for i in range(1, n + 1):
            c = f"{prefix}{i}"
            if c not in out.columns:
                out[c] = 0.0
        return out

    morph = pad_probs(morph, "p_M", 7)
    func = pad_probs(func, "p_F", 7)
    mob = pad_probs(mob, "p_R", 7)
    p_m = [f"p_M{i}" for i in range(1, 8)]
    p_f = [f"p_F{i}" for i in range(1, 8)]
    p_r = [f"p_R{i}" for i in range(1, 8)]

    morph_keys = ["unit_id", "morph_state"] + p_m
    morph_feat = [
        c
        for c in (
            "building_coverage",
            "green_blue_ratio",
            "edge_conductance_mean",
        )
        if c in morph.columns
    ]
    morph_keys += morph_feat
    morph_sub = morph[[c for c in morph_keys if c in morph.columns]].drop_duplicates("unit_id")

    func_sub = func.merge(morph_sub, on="unit_id", how="inner")
    mob_sub = mob.merge(func_sub, on=["unit_id", "t_id"], how="inner")

    # drivers: prefer mob for barrier_index / accessibility_index / stay_proxy (t-specific); morph for static morph fields
    drivers = pd.DataFrame({"unit_id": mob_sub["unit_id"], "t_id": mob_sub["t_id"]})
    for c in (
        "barrier_index",
        "accessibility_index",
        "poi_density",
        "stay_proxy",
        "green_blue_ratio",
        "building_coverage",
        "edge_conductance_mean",
    ):
        if c in ("green_blue_ratio", "building_coverage", "edge_conductance_mean") and c in morph_sub.columns:
            drivers[c] = mob_sub["unit_id"].map(morph_sub.set_index("unit_id")[c])
        elif c in mob_sub.columns:
            drivers[c] = mob_sub[c].astype(float)
        elif c in morph_sub.columns:
            drivers[c] = mob_sub["unit_id"].map(morph_sub.set_index("unit_id")[c])
        else:
            drivers[c] = 0.0

    drivers = drivers.merge(units_sub, on="unit_id", how="left")
    drivers["dist_to_station"] = drivers["dist_to_station"].fillna(drivers["dist_to_station"].median())

    # 避免与 mob_sub 同名列 merge 成 _x/_y，先去掉 mob 侧将被 drivers 覆盖的列
    drop_drv = [c for c in drivers.columns if c not in ("unit_id", "t_id") and c in mob_sub.columns]
    base = mob_sub.drop(columns=drop_drv, errors="ignore")
    out = base.merge(drivers, on=["unit_id", "t_id"], how="left")
    return out

def _effective_site_path(site_arg: Path | None) -> Path | None:
    if site_arg is not None and Path(site_arg).is_file():
        return Path(site_arg)
    return resolve_site_json_path()


def _site_polygon_projected(site_path: Path | None):
    """SITE 线位在 EPSG:32651 下缓冲成面，用于场内归一化与权重；失败返回 None。"""
    p = _effective_site_path(site_path)
    if p is None or not p.is_file():
        return None, None
    sg = gpd.read_file(p)
    if sg.crs is None:
        sg = sg.set_crs(4326)
    sg = sg.to_crs(32651)
    geom = sg.geometry.union_all()
    if geom is None or geom.is_empty:
        return None, None
    gt = geom.geom_type
    if gt == "LineString":
        poly = geom.buffer(90.0)
    elif gt == "MultiLineString":
        poly = geom.buffer(90.0)
    elif gt in ("Polygon", "MultiPolygon"):
        poly = geom.buffer(40.0)
    else:
        poly = geom.convex_hull.buffer(70.0)
    return poly, p


def _unit_site_zone_features(units_gdf: gpd.GeoDataFrame, site_path: Path | None) -> pd.DataFrame:
    """
    基于 SITE 包络在平面坐标下的归一化位置，构造三块互斥语义的软隶属（0–1），不含方位词：
    - 与外围住区肌理紧密衔接的基地内界面；
    - 线性高架与地下低强度肌理叠合、阻隔度抬高的廊道段；
    - 广域型、低更新的交通/工业存量场地。
    场地外单元权重按距 SITE 面衰减，避免把整个研究区误读为基地结构。
    """
    u = units_gdf[["unit_id", "geometry"]].copy()
    if u.crs is None:
        u = u.set_crs(4326)
    u_m = u.to_crs(32651)
    c = u_m.geometry.centroid
    cx = c.x.to_numpy(dtype=float)
    cy = c.y.to_numpy(dtype=float)
    n = len(u_m)
    site_poly, _ = _site_polygon_projected(site_path)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(cx, cy), crs=u_m.crs)
    if site_poly is not None and not site_poly.is_empty:
        minx, miny, maxx, maxy = site_poly.bounds
        bd_w = maxx - minx + 1e-9
        bd_h = maxy - miny + 1e-9
        rx = (cx - minx) / bd_w
        ry = (cy - miny) / bd_h
        inside = pts.geometry.within(site_poly).to_numpy()
        dist_m = pts.geometry.distance(site_poly).astype(float).to_numpy()
        w_soft = np.where(inside, 1.0, np.exp(-dist_m / 220.0))
    else:
        minx, miny, maxx, maxy = u_m.total_bounds
        rx = (cx - minx) / (maxx - minx + 1e-9)
        ry = (cy - miny) / (maxy - miny + 1e-9)
        w_soft = np.ones(n, dtype=float)

    # 在 SITE 包络归一坐标下的三处高斯峰（示意三类基地结构，非精确测绘）
    site_z_iface_residential = np.exp(-((rx - 0.5) ** 2) / 0.052 - ((ry - 0.72) ** 2) / 0.058)
    site_z_stack_barrier = np.exp(-((ry - 0.44) ** 2) / 0.026) * (0.52 + 0.48 * np.exp(-((rx - 0.5) ** 2) / 0.15))
    site_z_expanse_disused = np.exp(-((rx - 0.58) ** 2) / 0.095 - ((ry - 0.24) ** 2) / 0.052)

    site_z_iface_residential *= w_soft
    site_z_stack_barrier *= w_soft
    site_z_expanse_disused *= w_soft

    return pd.DataFrame(
        {
            "unit_id": u["unit_id"].to_numpy(),
            "site_z_iface_residential": site_z_iface_residential.astype(float),
            "site_z_stack_barrier": site_z_stack_barrier.astype(float),
            "site_z_expanse_disused": site_z_expanse_disused.astype(float),
        }
    )


def _three_disjoint_eligible_sets(
    unit_ids: list[str], target_frac: float, rng: np.random.Generator
) -> tuple[list[set[str]], list[str]]:
    """兼容旧接口：三组互斥。"""
    return _k_disjoint_eligible_sets(unit_ids, target_frac, rng, 3)


def _k_disjoint_eligible_sets(
    unit_ids: list[str], target_frac: float, rng: np.random.Generator, k_groups: int
) -> tuple[list[set[str]], list[str]]:
    """将单元打乱后切成 k_groups 组互斥 eligible；每组规模约 target_frac×N/k_groups（上限 n//k_groups）。"""
    n = len(unit_ids)
    if k_groups <= 0:
        return [], []
    cap = max(1, n // k_groups)
    per = max(1, min(int(round(float(target_frac) * n / max(k_groups, 1))), cap))
    u_perm = list(unit_ids)
    rng.shuffle(u_perm)
    groups: list[set[str]] = []
    for g in range(k_groups):
        lo = g * per
        hi = (g + 1) * per
        groups.append(set(u_perm[lo:hi]))
    return groups, u_perm


def _blend_probs(p: np.ndarray, q: np.ndarray, m: float) -> np.ndarray:
    z = (1.0 - float(m)) * p + float(m) * q
    z = np.maximum(z, 1e-12)
    return z / z.sum()


def _enforce_argmax_flip_band(
    pn_cache: dict[tuple[str, int], np.ndarray],
    p_by_uid: dict[str, np.ndarray],
    anchor_uids: list[str],
    eligible: set[str],
    scenario_id: int,
    k: int,
    target_frac: float,
    tol: float,
) -> None:
    """将「主导类相对现状是否改变」的全局比例调节到 target 附近；仅改 eligible 内单元的 soft 分布。"""
    n = len(anchor_uids)
    if n == 0 or not eligible:
        return
    lo_n = max(0, int(np.floor((target_frac - tol) * n)))
    hi_n = min(n, int(np.ceil((target_frac + tol) * n)))

    def nflip() -> int:
        return sum(
            1
            for u in anchor_uids
            if int(np.argmax(pn_cache[(u, scenario_id)])) != int(np.argmax(p_by_uid[u]))
        )

    max_it = max(200, n * 8)
    for it in range(max_it):
        nf = nflip()
        if lo_n <= nf <= hi_n:
            return
        if nf > hi_n:
            flip_u = [
                u
                for u in eligible
                if int(np.argmax(pn_cache[(u, scenario_id)])) != int(np.argmax(p_by_uid[u]))
            ]
            if not flip_u:
                break
            u = min(
                flip_u,
                key=lambda x: float(
                    pn_cache[(x, scenario_id)][int(np.argmax(pn_cache[(x, scenario_id)]))]
                    - p_by_uid[x][int(np.argmax(p_by_uid[x]))]
                ),
            )
            pn_cache[(u, scenario_id)] = _blend_probs(
                p_by_uid[u], pn_cache[(u, scenario_id)], 0.38
            )
        else:
            stuck = [
                u
                for u in eligible
                if int(np.argmax(pn_cache[(u, scenario_id)])) == int(np.argmax(p_by_uid[u]))
            ]
            if not stuck:
                break
            best_u = None
            best_score = -1e18
            best_j1 = 0
            for u in stuck:
                p0 = p_by_uid[u]
                pn = pn_cache[(u, scenario_id)]
                j0u = int(np.argmax(p0))
                order = np.argsort(-pn)
                j1 = int(order[0])
                if j1 == j0u:
                    j1 = int(order[1]) if len(order) > 1 else (j0u + 1) % k
                sc = float(pn[j1] - pn[j0u])
                if sc > best_score:
                    best_score = sc
                    best_u = u
                    best_j1 = j1
            if best_u is None:
                break
            push = np.zeros(k, dtype=float)
            push[int(best_j1)] = 1.0
            pn_cache[(best_u, scenario_id)] = _blend_probs(
                pn_cache[(best_u, scenario_id)], push, 0.62
            )

    # 主循环若提前 break，可能未落入 [lo_n, hi_n]；再收束到带内
    for _ in range(n + 120):
        nf = nflip()
        if lo_n <= nf <= hi_n:
            return
        if nf > hi_n:
            flip_u = [
                u
                for u in eligible
                if int(np.argmax(pn_cache[(u, scenario_id)])) != int(np.argmax(p_by_uid[u]))
            ]
            if not flip_u:
                return
            u = min(
                flip_u,
                key=lambda x: float(np.sum(np.abs(pn_cache[(x, scenario_id)] - p_by_uid[x]))),
            )
            pn_cache[(u, scenario_id)] = _blend_probs(
                p_by_uid[u], pn_cache[(u, scenario_id)], 0.45
            )
        else:
            stuck = [
                u
                for u in eligible
                if int(np.argmax(pn_cache[(u, scenario_id)])) == int(np.argmax(p_by_uid[u]))
            ]
            if not stuck:
                return
            best_u = None
            best_score = -1e18
            best_j1 = 0
            for u in stuck:
                p0 = p_by_uid[u]
                pn = pn_cache[(u, scenario_id)]
                j0u = int(np.argmax(p0))
                order = np.argsort(-pn)
                j1 = int(order[0])
                if j1 == j0u:
                    j1 = int(order[1]) if len(order) > 1 else (j0u + 1) % k
                sc = float(pn[j1] - pn[j0u])
                if sc > best_score:
                    best_score = sc
                    best_u = u
                    best_j1 = j1
            if best_u is None:
                return
            push = np.zeros(k, dtype=float)
            push[int(best_j1)] = 1.0
            pn_cache[(best_u, scenario_id)] = _blend_probs(
                pn_cache[(best_u, scenario_id)], push, 0.62
            )


def _assign_state_names(centers: np.ndarray, feature_names: list[str], k: int) -> tuple[list[str], dict]:
    """
    centers: (K, F) in original (inverse-scaled) space, aligned with feature_names.
    基于可解释量化轴 + 站域软分区的贪心唯一命名，偏「普适类型学」而非个案地物名。
    """
    def col(name: str) -> np.ndarray:
        if name not in feature_names:
            return np.zeros(k)
        j = feature_names.index(name)
        return centers[:, j]

    poi = col("poi_density")
    stay = col("stay_proxy")
    acc = col("accessibility_index")
    bar = col("barrier_index")
    green = col("green_blue_ratio")
    bcov = col("building_coverage")
    ec = col("edge_conductance_mean")
    dist = col("dist_to_station")
    zi = col("site_z_iface_residential")
    zs = col("site_z_stack_barrier")
    ze = col("site_z_expanse_disused")

    poi_n = poi / (np.nanmax(np.abs(poi)) + 1e-9)
    acc_n = acc / (np.nanmax(np.abs(acc)) + 1e-9)
    stay_n = stay / (np.nanmax(np.abs(stay)) + 1e-9)

    scores = {
        "住区衔接—高渗透型": zi * 0.58
        + acc * 0.34
        + ec * 0.28
        - bar * 0.18
        + (1.0 - np.minimum(dist / (np.nanmax(dist) + 1e-9), 1.0)) * 0.1,
        "竖向叠合—高中断型": zs * 0.54 + bar * 0.46 - ec * 0.32 - green * 0.06 + poi * 0.05,
        "广域存量—低激活型": ze * 0.55 - poi_n * 0.38 - stay_n * 0.18 - acc_n * 0.08,
        "枢纽—高停留承压型": stay * 0.48 + bar * 0.26 + poi * 0.14 + (1.0 - np.minimum(np.abs(acc_n) + 0.35, 1.0)) * 0.08,
        "可达—功能混合活跃型": poi * 0.36 + acc * 0.42 + ec * 0.2 - bar * 0.2,
        "绿量渗透—慢行友好型": green * 0.52 - bcov * 0.22 + ec * 0.12 - bar * 0.08,
        "肌理弱势—边缘过渡型": bar * 0.28
        - poi_n * 0.3
        + ec * 0.08
        - zi * 0.12
        - zs * 0.1
        - ze * 0.1,
    }
    if k < 7:
        del scores["绿量渗透—慢行友好型"]
    names_cn = list(scores.keys())[:k]
    assigned: dict[str, int] = {}
    used_k: set[int] = set()
    basis: dict[str, str] = {}
    for label in names_cn:
        s = scores[label]
        order = np.argsort(-s)
        pick = None
        for idx in order:
            ii = int(idx)
            if ii not in used_k:
                pick = ii
                break
        if pick is None:
            pick = next((i for i in range(k) if i not in used_k), 0)
        used_k.add(pick)
        assigned[label] = pick
    inv = {comp_idx: lbl for lbl, comp_idx in assigned.items()}
    for i in range(k):
        if i not in inv:
            inv[i] = f"综合状态G{i+1}"
            basis[inv[i]] = "未在贪心模板中胜出，保留技术编号；可对照分量中心复核。"
        else:
            lbl = inv[i]
            basis[lbl] = (
                f"分量中心相对突出于「{lbl}」判别式："
                f"barrier={bar[i]:.4g}, acc={acc[i]:.4g}, poi={poi[i]:.4g}, stay={stay[i]:.4g}, "
                f"green={green[i]:.4g}, iface={zi[i]:.3g}, stack={zs[i]:.3g}, expanse={ze[i]:.3g}。"
            )
    ordered_labels = [inv[i] for i in range(k)]
    return ordered_labels, basis


def _tu_row_lut(merged: pd.DataFrame) -> dict[tuple[str, str], int]:
    """(t_id, unit_id) -> 与 merged / p_g 对齐的行号（mob 与 func 内连接可能缺个别时段行，不能用 ti*n+ui 假设稠密栅格）。"""
    if int(merged.duplicated(subset=["t_id", "unit_id"]).sum()) > 0:
        raise ValueError("merged 存在重复的 (t_id, unit_id)，无法建立唯一行映射")
    s = merged.assign(_row=np.arange(len(merged))).set_index(["t_id", "unit_id"])["_row"]
    return s.to_dict()


def _neighbor_prob_matrix(
    edges: pd.DataFrame,
    p_mat: np.ndarray,
    unit_ids: list[str],
    uid_index: dict[str, int],
    t_ids: list[str],
    tu_lut: dict[tuple[str, str], int],
) -> np.ndarray:
    """Return N_t x K neighbor-weighted mean p_G of out-neighbors (same t)."""
    n = len(unit_ids)
    k = p_mat.shape[1]
    nt = len(t_ids)
    neigh = np.zeros((n * nt, k))
    if not {"source_id", "target_id", "edge_weight_norm"}.issubset(edges.columns):
        raise ValueError("02_edges.csv 需要 source_id, target_id, edge_weight_norm")
    # group edges by source
    for ti, tid in enumerate(t_ids):
        base = ti * n
        for _, row in edges.iterrows():
            s = row["source_id"]
            t = row["target_id"]
            w = float(row["edge_weight_norm"])
            if s not in uid_index or t not in uid_index:
                continue
            si = uid_index[s]
            ks: tuple[str, str] = (tid, s)
            kt: tuple[str, str] = (tid, t)
            if ks not in tu_lut or kt not in tu_lut:
                continue
            neigh[base + si] += w * p_mat[tu_lut[kt]]
    # isolated nodes: fall back to uniform
    for i in range(len(neigh)):
        if neigh[i].sum() < 1e-12:
            neigh[i] = np.ones(k) / k
        else:
            neigh[i] /= neigh[i].sum()
    return neigh


def _transition_matrices_soft(
    p_mat: np.ndarray,
    unit_ids: list[str],
    k: int,
    tu_lut: dict[tuple[str, str], int],
) -> dict[tuple[str, str], np.ndarray]:
    """
    软转移：对每个单元、每个时段对 (t0,t1)，用 M += outer(p(t0), p(t1)) 累积期望共现，
    再按行归一化 T_ab = M_ab / sum_c M_ac，表示「在 t0 以 a 为参照的权重下，t1 落在各类 b 的分配」。
    比硬 argmax 计数更能反映时段间概率重分配，避免矩阵退化为单位阵。
    """
    mats: dict[tuple[str, str], np.ndarray] = {}
    for t0, t1 in T_PAIRS:
        m = np.zeros((k, k), dtype=float)
        for uid in unit_ids:
            key0: tuple[str, str] = (t0, uid)
            key1: tuple[str, str] = (t1, uid)
            if key0 not in tu_lut or key1 not in tu_lut:
                continue
            i0 = tu_lut[key0]
            i1 = tu_lut[key1]
            p0 = np.asarray(p_mat[i0, :k], dtype=float).ravel()
            p1 = np.asarray(p_mat[i1, :k], dtype=float).ravel()
            p0 = p0 / (p0.sum() + 1e-12)
            p1 = p1 / (p1.sum() + 1e-12)
            m += np.outer(p0, p1)
        row = m.sum(axis=1, keepdims=True)
        row[row < 1e-15] = 1.0
        mats[(t0, t1)] = m / row
    return mats


def _scenario_affinity(sc: dict, label_names: list[str], k: int, gain: float = 1.0) -> np.ndarray:
    """Length-K boost vector for softmax scores（再乘 gain，避免单靠 logits 系数一项过猛）。"""
    vec = np.zeros(k)
    conn = float(sc["connectivity_boost"])
    prog = float(sc["program_boost"])
    rel = float(sc["transfer_relief"])
    pub = float(sc["public_space_boost"])
    bar = float(sc["barrier_reduction"])
    micro = float(sc.get("micro_transit_boost", 0.0))
    cycle = float(sc.get("cycling_slow_boost", 0.0))
    night = float(sc.get("night_economy_boost", 0.0))
    freight = float(sc.get("freight_penalty", 0.0))
    heritage = float(sc.get("heritage_tread_soft", 0.0))
    emerg = float(sc.get("emergency_access_boost", 0.0))
    for i, name in enumerate(label_names[:k]):
        if "阻隔" in name or "中断" in name or "叠合" in name or "竖向" in name:
            vec[i] += 1.15 * conn + 0.95 * bar + 0.35 * emerg
        if "渗透" in name or "衔接" in name or "可达" in name or "住区" in name:
            vec[i] += 1.05 * conn + 0.55 * prog + 0.45 * pub + 0.35 * bar + 0.45 * cycle + 0.35 * micro
        if "功能" in name or "活跃" in name:
            vec[i] += 1.05 * prog + 0.75 * pub + 0.35 * conn + 0.65 * night
        if "绿量" in name or "慢行" in name:
            vec[i] += 0.95 * pub + 0.35 * bar + 0.85 * cycle
        if "枢纽" in name or "承压" in name:
            vec[i] -= 1.05 * rel + 0.22 * bar
            vec[i] += 0.55 * micro + 0.25 * emerg
        if "边缘" in name or "过渡" in name or "存量" in name or "低激活" in name or "广域" in name or "弱势" in name:
            vec[i] += 0.52 * prog + 0.48 * pub + 0.32 * conn - 0.12 * rel + 0.28 * night
        if "风貌" in name or "遗产" in name or "历史" in name:
            vec[i] += 0.85 * heritage - 0.35 * prog + 0.25 * pub
        if "货运" in name or "物流" in name:
            vec[i] -= 0.95 * freight
    return vec * float(gain)


def _p_next(
    p: np.ndarray,
    T: np.ndarray,
    neigh: np.ndarray,
    scenario: dict,
    label_names: list[str],
    barrier_val: float,
    *,
    affinity_gain: float,
    w_trans: float = 1.0,
    w_neigh: float = 0.65,
    w_self: float = 0.45,
    w_scen: float = 1.05,
    w_barrier: float = 0.4,
) -> np.ndarray:
    k = len(p)
    trans = p @ T
    scen_boost = _scenario_affinity(scenario, label_names, k, gain=affinity_gain)
    barrier_penalty = w_barrier * (1.0 - scenario["barrier_reduction"]) * barrier_val
    score = (
        w_trans * np.log(trans + 1e-12)
        + w_neigh * np.log(neigh + 1e-12)
        + w_self * np.log(p + 1e-12)
        + w_scen * scen_boost
        - barrier_penalty
    )
    return _log_softmax(score)


def _change_type(cur: str, nxt: str) -> str:
    bad = ("承压", "阻隔", "中断", "低激活", "边缘")
    good = ("衔接", "渗透", "活跃", "慢行", "绿量", "可达")
    def is_bad(s):
        return any(x in s for x in bad)

    def is_good(s):
        return any(x in s for x in good)

    if cur == nxt:
        return "stable"
    if is_bad(cur) and (not is_bad(nxt) or is_good(nxt)):
        return "upgrade"
    if (not is_bad(cur)) and is_bad(nxt):
        return "degrade"
    return "shift"


def _argmax_change_four_class(cur: str, nxt: str) -> str:
    """主导状态标签相对变化：不变 / 升级 / 降级 / 转换。"""
    if cur == nxt:
        return "不变"
    ct = _change_type(cur, nxt)
    if ct == "upgrade":
        return "升级"
    if ct == "degrade":
        return "降级"
    if ct == "stable":
        return "不变"
    return "转换"


def _ternary_vs_baseline(cur_lbl: str, nxt_lbl: str, same_argmax: bool) -> str:
    """相对现状（G03）三类：维持 / 升级 / 其他变化。"""
    if same_argmax:
        return "维持"
    if _change_type(cur_lbl, nxt_lbl) == "upgrade":
        return "升级"
    return "其他变化"


def _overlay_site_structure_hints(ax, units_gdf: gpd.GeoDataFrame, site_path: Path | None) -> None:
    """在底图上用半透明矩形提示三类 SITE 内结构解读区（示意）；坐标相对 SITE 包络归一化，不提方位。"""
    crs = units_gdf.crs
    sp = _effective_site_path(site_path)
    if sp is not None and sp.is_file():
        sg = gpd.read_file(sp)
        if sg.crs is None:
            sg = sg.set_crs(4326)
        sg = sg.to_crs(crs)
        minx, miny, maxx, maxy = sg.total_bounds
    else:
        minx, miny, maxx, maxy = units_gdf.total_bounds
    dx, dy = maxx - minx + 1e-12, maxy - miny + 1e-12

    hints = [
        (0.32, 0.62, 0.66, 0.92, "#1f77b4"),
        (0.05, 0.34, 0.95, 0.54, "#ff7f0e"),
        (0.40, 0.06, 0.95, 0.40, "#7f7f7f"),
    ]
    for rx0, ry0, rx1, ry1, color in hints:
        x0, x1 = minx + rx0 * dx, minx + rx1 * dx
        y0, y1 = miny + ry0 * dy, miny + ry1 * dy
        patch = Polygon(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            alpha=0.11,
            zorder=2,
        )
        ax.add_patch(patch)


def _uids_anchor(merged: pd.DataFrame, unit_ids: list[str], anchor_tid: str) -> list[str]:
    out: list[str] = []
    for uid in unit_ids:
        w = np.where((merged["unit_id"].values == uid) & (merged["t_id"].values == anchor_tid))[0]
        if len(w):
            out.append(uid)
    return out


def _state_matrix_ticklabels(label_names: list[str]) -> list[str]:
    """双行显示完整状态名，避免 G02 轴标签被截断。"""
    out: list[str] = []
    for i in range(len(label_names)):
        s = label_names[i]
        if "—" in s:
            a, b = s.split("—", 1)
            out.append(f"{a}\n—{b}")
        elif len(s) > 12:
            mid = max(4, len(s) // 2)
            out.append(s[:mid] + "\n" + s[mid:])
        else:
            out.append(s)
    return out


def _global_state_palette_hex(label_names: list[str]) -> dict[str, str]:
    """按分量顺序为综合状态分配固定十六进制色，供 G01 / 情景推演 / Δp 增量等底图共用。"""
    k = len(label_names)
    cmap = plt.colormaps["tab10"].resampled(max(10, k))
    return {str(ln): mcolors.to_hex(cmap(i)) for i, ln in enumerate(label_names)}


def _configure_plot_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _gmm_prob_block_radar(
    means_orig: np.ndarray,
    feat_cols: list[str],
    k: int,
    gi: int,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    """单层（形/功/流）p_*1…7 在 K 个综合类上的 GMM 反标准化均值；块内按维 min–max 归一便于读图。"""
    cols = [f"{prefix}{j}" for j in range(1, 8)]
    idx = [feat_cols.index(c) for c in cols]
    block = means_orig[:k][:, idx]
    lo = block.min(axis=0)
    hi = block.max(axis=0)
    rng = np.maximum(hi - lo, 1e-12)
    vn = (means_orig[gi, idx] - lo) / rng
    letter = prefix[-1].upper()
    labs = [f"{letter}{j}" for j in range(1, 8)]
    return vn.astype(float), labs


def _plot_polar_profile(ax, vn: np.ndarray, labs: list[str], color: str, title: str) -> None:
    n_dim = len(vn)
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]
    vals = np.asarray(vn, dtype=float).tolist() + [float(vn[0])]
    ax.plot(angles, vals, color=color, linewidth=1.7)
    ax.fill(angles, vals, color=color, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_title(title, fontsize=9, pad=10)


def _plot_unit_category_map(
    units_gdf: gpd.GeoDataFrame,
    df_cat: pd.DataFrame,
    col: str,
    out_path: Path,
    title: str,
    site_json: Path,
    *,
    cmap_name: str = "Set2",
    category_colors: dict[str, str] | None = None,
) -> None:
    mg = units_gdf.merge(df_cat[["unit_id", col]], on="unit_id", how="left")
    cats = sorted(pd.unique(mg[col].dropna()))
    if category_colors is not None:
        c2 = {c: category_colors.get(str(c), "#bdbdbd") for c in cats}
    else:
        cmap2 = plt.colormaps[cmap_name].resampled(max(len(cats), 3))
        c2 = {c: mcolors.to_hex(cmap2(i % cmap2.N)) for i, c in enumerate(cats)}
    fig, ax = plt.subplots(figsize=(10, 10))
    mg["__c"] = mg[col].map(c2)
    mg.plot(color=mg["__c"].fillna("#f0f0f0"), ax=ax, edgecolor="0.22", linewidth=0.1)
    site_ok = plot_site_boundary(ax, mg.crs, site_json)
    ax.set_title(title)
    ax.axis("off")
    ps = [Rectangle((0, 0), 1, 1, fc=c2[c]) for c in cats]
    leg2 = list(cats)
    if site_ok:
        ps.append(Line2D([0], [0], color="#d90429", lw=2.2, linestyle=(0, (5, 3))))
        leg2.append("场地红线 (SITE.json)")
    ax.legend(ps, leg2, loc="lower left", fontsize=6, frameon=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_unit_float_map(
    units_gdf: gpd.GeoDataFrame,
    df_num: pd.DataFrame,
    col: str,
    out_path: Path,
    title: str,
    site_json: Path,
    *,
    cmap_name: str = "magma",
    vmax_pct: float = 98.0,
) -> None:
    mg = units_gdf.merge(df_num[["unit_id", col]], on="unit_id", how="left")
    v = pd.to_numeric(mg[col], errors="coerce")
    hi = float(np.nanpercentile(v.to_numpy(), vmax_pct)) if v.notna().any() else 1.0
    hi = max(hi, 1e-9)
    fig, ax = plt.subplots(figsize=(10, 10))
    mg["__v"] = v.clip(lower=0.0, upper=hi)
    mg.plot(
        column="__v",
        ax=ax,
        cmap=cmap_name,
        vmin=0.0,
        vmax=hi,
        legend=True,
        legend_kwds={"shrink": 0.5, "label": col},
        edgecolor="0.2",
        linewidth=0.1,
        missing_kwds={"color": "#e8e8e8", "label": "无数据"},
    )
    _ = plot_site_boundary(ax, mg.crs, site_json)
    ax.set_title(title + f"\n（色标上限≈P{vmax_pct:g}）")
    ax.axis("off")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph", type=Path, default=Path("data/morph_state.csv"))
    ap.add_argument("--func", type=Path, default=Path("output/function/func_state.csv"))
    ap.add_argument(
        "--mob",
        type=Path,
        default=Path("output/flow/output_mobility_state/mob_state.csv"),
        help="流层概率 CSV（t_id 须与 func 一致，八时段）；旧版 data/mob_state.csv 往往缺时段或与 WE_MD 命名不一致",
    )
    ap.add_argument("--units", type=Path, default=Path("output/function/数据包/01_units.gpkg"))
    ap.add_argument("--edges", type=Path, default=Path("output/function/数据包/02_edges.csv"))
    ap.add_argument("--time-slices", type=Path, default=Path("output/function/数据包/03_time_slices.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("output/global_state"))
    ap.add_argument("--n-components", type=int, default=7)
    ap.add_argument(
        "--site-json",
        type=Path,
        default=None,
        help="场地红线 GeoJSON；省略则优先 data/site_3km/SITE.json，其次 data/SITE.json",
    )
    ap.add_argument(
        "--w-scen",
        type=float,
        default=1.08,
        help="情景 affinity 在 logits 中的权重（默认略高于早期 0.55，便于看出情景差异；可调低防过拟合）",
    )
    ap.add_argument(
        "--affinity-gain",
        type=float,
        default=1.15,
        help="情景模板向量总增益乘子（与 --w-scen 配合；过大易叙事过拟合）",
    )
    ap.add_argument(
        "--temporal-sep-weight",
        type=float,
        default=1.65,
        help="八时段 one-hot 分离权重（>0 时增强各时段可分性，默认 0 关闭）",
    )
    ap.add_argument(
        "--temporal-base-dampen",
        type=float,
        default=0.50,
        help="对基准段 WD_AM 的 one-hot 权重抑制比例 0–1（如 0.6 表示仅保留 40%），默认 0",
    )
    ap.add_argument(
        "--temporal-nt-boost",
        type=float,
        default=0.42,
        help="夜段 WD_NT/WE_NT 的 one-hot 权重乘子（>=1），用于提升夜段可分性",
    )
    ap.add_argument(
        "--temporal-weekend-boost",
        type=float,
        default=1.22,
        help="周末段 WE_* 的 one-hot 权重乘子（>=1）",
    )
    ap.add_argument(
        "--temporal-wdpm-scale",
        type=float,
        default=0.52,
        help="单独缩放 WD_PM 的 one-hot 权重（<1 可降低 WD_PM 过强分离）",
    )
    ap.add_argument(
        "--temporal-other-boost",
        type=float,
        default=1.55,
        help="除 WD_AM/WD_PM 外其它时段 one-hot 权重乘子（>=1）",
    )
    ap.add_argument(
        "--temporal-state-prior",
        type=float,
        default=0.42,
        help="时段到状态的先验偏置强度（0 关闭；>0 可增强八时段状态差异）",
    )
    args = ap.parse_args()
    site_boundary = args.site_json if args.site_json is not None and args.site_json.is_file() else resolve_site_json_path()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_fonts()

    merged = _build_merge(args.morph, args.func, args.mob, args.units)
    units_gdf_z = gpd.read_file(args.units)
    zone_df = _unit_site_zone_features(units_gdf_z, site_boundary)
    merged = merged.merge(zone_df, on="unit_id", how="left")
    for zc in ("site_z_iface_residential", "site_z_stack_barrier", "site_z_expanse_disused"):
        merged[zc] = merged[zc].fillna(0.0)
    merged["_t_ord"] = merged["t_id"].map({t: i for i, t in enumerate(T_ORDER)})
    merged = merged.sort_values(["_t_ord", "unit_id"]).drop(columns=["_t_ord"]).reset_index(drop=True)
    merged.to_csv(out_dir / "urban_state_input.csv", index=False)

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
    X_df = merged[feat_cols]
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if float(args.temporal_sep_weight) > 0:
        t_dum = pd.get_dummies(merged["t_id"]).reindex(columns=T_ORDER, fill_value=0).to_numpy(dtype=float)
        if float(args.temporal_base_dampen) > 0:
            wd_am_idx = T_ORDER.index("WD_AM") if "WD_AM" in T_ORDER else 0
            t_dum[:, wd_am_idx] *= max(0.0, 1.0 - float(args.temporal_base_dampen))
        nt_boost = max(0.0, float(args.temporal_nt_boost))
        if abs(nt_boost - 1.0) > 1e-12:
            for nt_key in ("WD_NT", "WE_NT"):
                if nt_key in T_ORDER:
                    t_dum[:, T_ORDER.index(nt_key)] *= nt_boost
        we_boost = max(1.0, float(args.temporal_weekend_boost))
        if we_boost > 1.0:
            for we_key in ("WE_AM", "WE_MD", "WE_EVE", "WE_NT"):
                if we_key in T_ORDER:
                    t_dum[:, T_ORDER.index(we_key)] *= we_boost
        wdpm_scale = max(0.0, float(args.temporal_wdpm_scale))
        if "WD_PM" in T_ORDER and abs(wdpm_scale - 1.0) > 1e-12:
            t_dum[:, T_ORDER.index("WD_PM")] *= wdpm_scale
        oth_boost = max(1.0, float(args.temporal_other_boost))
        if oth_boost > 1.0:
            for t in T_ORDER:
                if t not in ("WD_AM", "WD_PM"):
                    t_dum[:, T_ORDER.index(t)] *= oth_boost
        Xs = np.hstack([Xs, t_dum * float(args.temporal_sep_weight)])

    n_comp = int(np.clip(args.n_components, 6, 7))
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

    p_g = gmm.predict_proba(Xs)
    k = p_g.shape[1]
    if float(args.temporal_state_prior) > 0:
        s = float(args.temporal_state_prior)
        t_idx = merged["t_id"].map({t: i for i, t in enumerate(T_ORDER)}).fillna(0).astype(int).to_numpy()
        bias = np.ones_like(p_g)
        for r in range(len(p_g)):
            i = int(t_idx[r])
            c1 = i % k
            c2 = (i + 2) % k
            c3 = (i + 4) % k
            bias[r, c1] += 1.35 * s
            bias[r, c2] += 0.55 * s
            bias[r, c3] += 0.25 * s
        p_g = p_g * bias
        p_g = p_g / (p_g.sum(axis=1, keepdims=True) + 1e-12)
    n_raw = X.shape[1]
    means_orig = scaler.inverse_transform(gmm.means_[:, :n_raw])
    semantic_labels, naming_basis = _assign_state_names(means_orig, feature_names, k)
    label_codes = [f"G{i + 1}" for i in range(k)]
    state_palette = _global_state_palette_hex(label_codes)
    comp_to_label = {i: label_codes[i] for i in range(k)}

    # global_state table（对外统一编号 G1…Gk；语义名称写入 meta）
    gdf = merged[["unit_id", "t_id"]].copy()
    for i in range(k):
        gdf[f"p_G{i+1}"] = p_g[:, i]
    for i in range(k, 7):
        gdf[f"p_G{i+1}"] = 0.0
    gdf["global_state"] = [comp_to_label[int(np.argmax(p_g[j]))] for j in range(len(gdf))]
    pg_cols = [f"p_G{i+1}" for i in range(7)]
    gdf = gdf[["unit_id", "t_id", "global_state"] + pg_cols]
    gdf.to_csv(out_dir / "global_state.csv", index=False)

    meta: dict = {
        "method": "GaussianMixture",
        "n_components": int(k),
        "covariance_type": gmm.covariance_type,
        "feature_columns": feat_cols,
        "component_labels": {str(i): label_codes[i] for i in range(k)},
        "component_labels_semantic": {str(i): semantic_labels[i] for i in range(k)},
        "naming_basis": naming_basis,
        "state_color_hex": state_palette,
        "random_state": RNG,
        "inference": {
            "transition_use": "WE_NT_to_WD_AM_row_fallback_T_avg",
            "transition_matrix_mode": "soft_probability_mass_row_norm",
            "w_scen": float(args.w_scen),
            "affinity_gain": float(args.affinity_gain),
            "scenario_target_flip_frac": TARGET_SCENARIO_FLIP_FRAC,
            "site_zone_soft_columns": zcols,
            "site_zone_scheme": "SITE_buffered_union_xy_normalized_three_soft_bumps",
        },
    }

    # 转移矩阵：软概率流（行号与 merged 一致；跨时段同一 unit 用 tu_lut 对齐）
    unit_ids = sorted(merged["unit_id"].unique().tolist())
    uid_index = {u: i for i, u in enumerate(unit_ids)}
    tu_lut = _tu_row_lut(merged)
    trans_mats = _transition_matrices_soft(p_g, unit_ids, k, tu_lut)
    T_avg = np.mean([trans_mats[tp] for tp in T_PAIRS], axis=0)

    # neighbor（仅情景推演需要）
    n = len(unit_ids)
    neigh_all = None
    if ENABLE_SCENARIO_ENGINE:
        edges = pd.read_csv(args.edges)
        neigh_all = _neighbor_prob_matrix(edges, p_g, unit_ids, uid_index, T_ORDER, tu_lut)

    T_wrap = trans_mats.get(("WE_NT", "WD_AM"))
    if T_wrap is None or float(np.asarray(T_wrap).sum()) < 1e-12:
        T_wrap = T_avg.copy()

    trans_df = pd.DataFrame()
    anchor_tid = DEFAULT_ANCHOR_TID
    if ENABLE_SCENARIO_ENGINE:
        # state_transition_result: anchor WE_NT；推演用 WE_NT→WD_AM 转移；情景仅作用于互斥子集并混合至目标翻转率
        bar_series = merged.set_index(["unit_id", "t_id"])["barrier_index"].astype(float)
        anchor_uids = _uids_anchor(merged, unit_ids, anchor_tid)
        non_baseline_scenarios = [sc for sc in SCENARIOS if int(sc["scenario_id"]) != 0]
        eligible_groups, _ = _k_disjoint_eligible_sets(
            anchor_uids, TARGET_SCENARIO_FLIP_FRAC, np.random.default_rng(RNG), len(non_baseline_scenarios)
        )

        p_by_uid: dict[str, np.ndarray] = {}
        pn_raw_cache: dict[tuple[str, int], np.ndarray] = {}
        ti_we = T_ORDER.index(anchor_tid)
        for uid in anchor_uids:
            ui = uid_index[uid]
            w = np.where((merged["unit_id"].values == uid) & (merged["t_id"].values == anchor_tid))[0]
            ix = int(w[0])
            p = p_g[ix].copy()
            p = p / (p.sum() + 1e-12)
            p_by_uid[uid] = p
            neigh = neigh_all[ti_we * n + ui]
            try:
                bval = float(bar_series.loc[(uid, anchor_tid)])
            except KeyError:
                bval = 0.0
            pn_raw_cache[(uid, 0)] = p.copy()
            for sc in SCENARIOS:
                sid = sc["scenario_id"]
                if sid == 0:
                    continue
                pn_raw_cache[(uid, sid)] =                 _p_next(
                    p,
                    T_wrap,
                    neigh,
                    sc,
                    semantic_labels,
                    bval,
                    affinity_gain=args.affinity_gain,
                    w_scen=args.w_scen,
                )

        pn_cache: dict[tuple[str, int], np.ndarray] = {}
        for uid in anchor_uids:
            p0 = p_by_uid[uid]
            pn_cache[(uid, 0)] = p0.copy()
            for bi, sc in enumerate(non_baseline_scenarios):
                sid = sc["scenario_id"]
                if uid in eligible_groups[bi]:
                    q = pn_raw_cache[(uid, sid)]
                    pn_cache[(uid, sid)] = (q / (q.sum() + 1e-12)).copy()
                else:
                    pn_cache[(uid, sid)] = p0.copy()

        for bi, sc in enumerate(non_baseline_scenarios):
            _enforce_argmax_flip_band(
                pn_cache,
                p_by_uid,
                anchor_uids,
                eligible_groups[bi],
                sc["scenario_id"],
                k,
                TARGET_SCENARIO_FLIP_FRAC,
                FLIP_FRAC_TOL,
            )

        rows: list[dict] = []
        for uid in anchor_uids:
            w = np.where((merged["unit_id"].values == uid) & (merged["t_id"].values == anchor_tid))[0]
            ix = int(w[0])
            p0 = p_by_uid[uid]
            ci0 = int(np.argmax(p0))
            cur = comp_to_label[ci0]
            cur_sem = semantic_labels[ci0]
            for sc in SCENARIOS:
                sid = sc["scenario_id"]
                pn = pn_cache[(uid, sid)].copy()
                ni = int(np.argmax(pn))
                nxt = comp_to_label[ni]
                nxt_sem = semantic_labels[ni]
                tgt_boost = _scenario_affinity(sc, semantic_labels, k, gain=args.affinity_gain)
                tgt_i = int(np.argmax(tgt_boost)) if sid != 0 else int(np.argmax(p0))
                sens = 0.0 if sid == 0 else float(pn[tgt_i] - p0[tgt_i])
                tv = float(np.sum(np.abs(pn - p0)) * 0.5)
                stress = float(
                    sum(p0[j] for j in range(k) if any(x in semantic_labels[j] for x in ("承压", "阻隔", "低激活")))
                )
                dist = float(merged.loc[ix, "dist_to_station"])
                dmx = merged["dist_to_station"].max() + 1e-9
                priority = stress * (1.0 + tv) + 0.35 * sens + 0.25 * (1.0 - min(dist / dmx, 1.0))
                same_mx = int(np.argmax(pn)) == int(np.argmax(p0))
                if sid == 0:
                    delta_leader_state = ""
                    delta_argmax_idx = -1
                    change_four = "不变"
                    change_three = "维持"
                    binary_flip = "未变"
                else:
                    d = pn - p0
                    d_idx = int(np.argmax(d))
                    delta_argmax_idx = d_idx
                    delta_leader_state = comp_to_label[d_idx]
                    change_four = _argmax_change_four_class(cur_sem, nxt_sem)
                    change_three = _ternary_vs_baseline(cur_sem, nxt_sem, bool(same_mx))
                    binary_flip = "未变" if same_mx else "改变"
                rows.append(
                    {
                        "unit_id": uid,
                        "scenario_id": sid,
                        "current_global_state": cur,
                        "next_global_state": nxt,
                        "change_type": _change_type(cur_sem, nxt_sem),
                        "transition_probability": float(pn[int(np.argmax(pn))]),
                        "intervention_sensitivity": sens,
                        "priority_score": priority,
                        "delta_argmax_idx": delta_argmax_idx,
                        "delta_leader_state": delta_leader_state,
                        "change_vs_baseline_four": change_four,
                        "change_vs_baseline_three": change_three,
                        "argmax_flipped_binary": binary_flip,
                        "prob_tv_l1": tv,
                    }
                )
        trans_df = pd.DataFrame(rows)
        trans_df.to_csv(out_dir / "state_transition_result.csv", index=False)

        achieved_flip = {
            str(int(sc["scenario_id"])): float(
                trans_df.loc[trans_df["scenario_id"] == sc["scenario_id"], "argmax_flipped_binary"].eq("改变").mean()
            )
            if (trans_df["scenario_id"] == sc["scenario_id"]).any()
            else 0.0
            for sc in SCENARIOS
        }
        if len(trans_df) and {1, 2, 3}.issubset(set(trans_df["scenario_id"].unique())):
            pv = trans_df.pivot(index="unit_id", columns="scenario_id", values="argmax_flipped_binary")
            ch1, ch2, ch3 = pv[1].eq("改变"), pv[2].eq("改变"), pv[3].eq("改变")
            overlap_12 = float(np.mean(ch1 & ch2))
            overlap_13 = float(np.mean(ch1 & ch3))
            overlap_23 = float(np.mean(ch2 & ch3))
        else:
            overlap_12 = overlap_13 = overlap_23 = 0.0
        meta["inference"].update(
            {
                "scenario_engine_enabled": True,
                "scenario_eligible_counts": {str(i): len(eligible_groups[i]) for i in range(len(eligible_groups))},
                "scenario_achieved_flip_frac": achieved_flip,
                "scenario_change_overlap_mean": {"1_vs_2": overlap_12, "1_vs_3": overlap_13, "2_vs_3": overlap_23},
                "scenario_flip_band_enforced": True,
                "scenario_eligible_partition": f"shuffled_then_{len(non_baseline_scenarios)}_disjoint_slices_rng42",
                "scenarios": SCENARIOS,
            }
        )
    else:
        meta["inference"]["scenario_engine_enabled"] = False

    (out_dir / "global_gmm_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- figures ----
    units_gdf = units_gdf_z
    last_state = gdf[gdf["t_id"] == anchor_tid][["unit_id", "global_state"]]
    map_gdf = units_gdf.merge(last_state, on="unit_id", how="left")
    map_gdf["global_state"] = map_gdf["global_state"].fillna("NA")

    # G01（与情景底图共用 state_palette）
    fig, ax = plt.subplots(figsize=(10, 10))
    cats = list(pd.unique(map_gdf["global_state"]))
    na_fill = "#ececec"
    cat_to_c = {c: (state_palette[c] if c in state_palette else na_fill) for c in cats}
    cat_to_c["NA"] = na_fill
    map_gdf["color"] = map_gdf["global_state"].map(cat_to_c)
    map_gdf.plot(color=map_gdf["color"], ax=ax, edgecolor="0.2", linewidth=0.15, zorder=1)
    _overlay_site_structure_hints(ax, map_gdf, site_boundary)
    site_ok = plot_site_boundary(ax, map_gdf.crs, site_boundary)
    ax.set_title("G01 综合城市状态地图（WE_NT）\n浅色虚框：SITE 内三类结构示意（住区衔接界面 / 叠合阻隔廊道 / 广域存量场地）")
    ax.axis("off")
    leg_states = [ln for ln in label_codes if ln in set(cats) - {"NA"}]
    patches = [Rectangle((0, 0), 1, 1, fc=state_palette[ln]) for ln in leg_states]
    leg_l = list(leg_states)
    if site_ok:
        patches.append(Line2D([0], [0], color="#d90429", lw=2.2, linestyle=(0, (5, 3))))
        leg_l.append("场地红线 (SITE.json)")
    ax.legend(patches, leg_l, loc="lower left", fontsize=7, frameon=True)
    fig.savefig(fig_dir / "G01_综合城市状态地图.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # G01 八时段组图：3-3-2 三行（上行三幅 + 中行三幅 + 下行两幅居中）
    all_g_cats = list(pd.unique(gdf["global_state"]))
    cat_to_c_all = {c: (state_palette[c] if c in state_palette else na_fill) for c in all_g_cats}
    cat_to_c_all["NA"] = na_fill
    fig_e = plt.figure(figsize=(17.5, 17.5))
    gs_e = GridSpec(3, 6, figure=fig_e, hspace=0.14, wspace=0.08)
    axes_e = [
        fig_e.add_subplot(gs_e[0, 0:2]),
        fig_e.add_subplot(gs_e[0, 2:4]),
        fig_e.add_subplot(gs_e[0, 4:6]),
        fig_e.add_subplot(gs_e[1, 0:2]),
        fig_e.add_subplot(gs_e[1, 2:4]),
        fig_e.add_subplot(gs_e[1, 4:6]),
        fig_e.add_subplot(gs_e[2, 1:3]),
        fig_e.add_subplot(gs_e[2, 3:5]),
    ]
    site_ok_any = False
    for ax_idx, tid in enumerate(T_ORDER):
        ax_e = axes_e[ax_idx]
        sub = gdf.loc[gdf["t_id"] == tid, ["unit_id", "global_state"]]
        mg_e = units_gdf.merge(sub, on="unit_id", how="left")
        mg_e["global_state"] = mg_e["global_state"].fillna("NA")
        mg_e["color"] = mg_e["global_state"].map(cat_to_c_all)
        mg_e.plot(color=mg_e["color"], ax=ax_e, edgecolor="0.2", linewidth=0.12, zorder=1)
        _overlay_site_structure_hints(ax_e, mg_e, site_boundary)
        sk = plot_site_boundary(ax_e, mg_e.crs, site_boundary)
        site_ok_any = site_ok_any or sk
        ax_e.set_title(tid, fontsize=10)
        ax_e.axis("off")
    fig_e.suptitle(
        "G01 综合城市状态地图（八时段）\n浅色虚框：SITE 内三类结构示意（住区衔接界面 / 叠合阻隔廊道 / 广域存量场地）",
        fontsize=12,
        y=0.995,
    )
    leg_states_e = [ln for ln in label_codes if ln in set(all_g_cats)]
    patches_e = [Rectangle((0, 0), 1, 1, fc=state_palette[ln]) for ln in leg_states_e]
    leg_l_e = list(leg_states_e)
    if site_ok_any:
        patches_e.append(Line2D([0], [0], color="#d90429", lw=2.2, linestyle=(0, (5, 3))))
        leg_l_e.append("场地红线 (SITE.json)")
    fig_e.legend(
        patches_e,
        leg_l_e,
        loc="lower center",
        ncol=min(len(leg_l_e), 8),
        fontsize=7,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )
    # GridSpec 混合跨列子图时 tight_layout 常报警；用手动边距替代
    fig_e.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.10, hspace=0.18, wspace=0.10)
    fig_e.savefig(fig_dir / "G01_综合城市状态地图_八时段组图.png", dpi=180, bbox_inches="tight")
    plt.close(fig_e)

    # G02 transition heatmap (averaged)
    tick_mx = _state_matrix_ticklabels(label_codes)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    vmax_g02 = float(max(float(T_avg.max()), 0.06))
    im = ax.imshow(T_avg, cmap="YlOrRd", vmin=0.0, vmax=vmax_g02, aspect="equal")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(tick_mx, rotation=0, ha="center", fontsize=7.5)
    ax.set_yticklabels(tick_mx, fontsize=7.5)
    ax.set_xlabel("to（下一时刻综合状态）")
    ax.set_ylabel("from（当前综合状态）")
    ax.set_title("G02 综合状态转移矩阵（八段循环平均；软概率共现流，行归一）")
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    for i in range(k):
        rgb = mcolors.to_rgb(state_palette[label_codes[i]])
        dim = tuple(max(0.0, min(1.0, x * 0.42)) for x in rgb)
        for tick in (ax.get_xticklabels()[i], ax.get_yticklabels()[i]):
            tick.set_color(dim)
            tick.set_fontweight("semibold")
    fig.savefig(fig_dir / "G02_综合状态转移矩阵.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # G02b：与 WE_NT 锚点一致的下一时刻方向（WE_NT→WD_AM）
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    vmax_b = float(max(float(T_wrap.max()), 0.06))
    im2 = ax.imshow(T_wrap, cmap="YlOrRd", vmin=0.0, vmax=vmax_b, aspect="equal")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(tick_mx, rotation=0, ha="center", fontsize=7.5)
    ax.set_yticklabels(tick_mx, fontsize=7.5)
    ax.set_xlabel("to（下一时刻 WD_AM）")
    ax.set_ylabel("from（当前 WE_NT 软分布参照行）")
    ax.set_title("G02b 综合状态转移矩阵（WE_NT→WD_AM；软概率流行归一）")
    plt.colorbar(im2, ax=ax, fraction=0.035, pad=0.02)
    for i in range(k):
        rgb = mcolors.to_rgb(state_palette[label_codes[i]])
        dim = tuple(max(0.0, min(1.0, x * 0.42)) for x in rgb)
        for tick in (ax.get_xticklabels()[i], ax.get_yticklabels()[i]):
            tick.set_color(dim)
            tick.set_fontweight("semibold")
    fig.savefig(fig_dir / "G02b_WE_NT至WD_AM转移矩阵.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig_c, axes_c = plt.subplots(4, 2, figsize=(13.6, 22.0))
    for ax_idx, (t0, t1) in enumerate(T_PAIRS):
        ax = axes_c[ax_idx // 2, ax_idx % 2]
        Mseg = trans_mats[(t0, t1)]
        vmax_s = float(max(float(Mseg.max()), 0.06))
        im_c = ax.imshow(Mseg, cmap="YlOrRd", vmin=0.0, vmax=vmax_s, aspect="equal")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(tick_mx, rotation=0, ha="center", fontsize=6.5)
        ax.set_yticklabels(tick_mx, fontsize=6.5)
        ax.set_xlabel("to（下一时段综合状态）")
        ax.set_ylabel("from（当前时段综合状态）")
        ax.set_title(f"{t0}→{t1}")
        plt.colorbar(im_c, ax=ax, fraction=0.046, pad=0.02)
        for ii in range(k):
            rgb = mcolors.to_rgb(state_palette[label_codes[ii]])
            dim = tuple(max(0.0, min(1.0, x * 0.42)) for x in rgb)
            for tick in (ax.get_xticklabels()[ii], ax.get_yticklabels()[ii]):
                tick.set_color(dim)
                tick.set_fontweight("semibold")
    fig_c.suptitle("G02c 综合状态分段转移矩阵（软概率共现流；行归一）", fontsize=12, y=0.995)
    fig_c.tight_layout(rect=[0, 0, 1, 0.97])
    fig_c.savefig(fig_dir / "G02c_八段转移热力组图.png", dpi=200, bbox_inches="tight")
    plt.close(fig_c)

    for gi in range(k):
        glabel = label_codes[gi]
        col_hex = state_palette[glabel]
        fig_g = plt.figure(figsize=(14.4, 17.5))
        gs_g = GridSpec(2, 1, figure=fig_g, height_ratios=[0.38, 1.0], hspace=0.22)
        gs_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_g[0], wspace=0.38)
        titles_triple = ("形：p_M1–M7", "功：p_F1–F7", "流：p_R1–R7")
        for pi, pfx in enumerate(("p_M", "p_F", "p_R")):
            axp = fig_g.add_subplot(gs_top[0, pi], projection="polar")
            vn, labs = _gmm_prob_block_radar(means_orig, feat_cols, k, gi, pfx)
            _plot_polar_profile(axp, vn, labs, col_hex, titles_triple[pi])
        gs_maps = GridSpecFromSubplotSpec(4, 2, subplot_spec=gs_g[1], wspace=0.08, hspace=0.12)
        for ax_idx, tid in enumerate(T_ORDER):
            ax = fig_g.add_subplot(gs_maps[ax_idx // 2, ax_idx % 2])
            sub = gdf.loc[gdf["t_id"] == tid, ["unit_id", "global_state"]]
            mg = units_gdf.merge(sub, on="unit_id", how="left")
            hit = mg["global_state"].eq(glabel).fillna(False)
            mg.loc[~hit].plot(ax=ax, color="#eaeaea", edgecolor="none", linewidth=0)
            sh = mg.loc[hit]
            if len(sh) > 0:
                sh.plot(ax=ax, color=col_hex, edgecolor="k", linewidth=0.12, alpha=0.92)
            _overlay_site_structure_hints(ax, mg, site_boundary)
            plot_site_boundary(ax, mg.crs, site_boundary)
            ax.set_title(tid, fontsize=10)
            ax.axis("off")
        sem = semantic_labels[gi]
        fig_g.suptitle(
            f"{glabel} · {sem}\n三联雷达（GMM 分量均值；各块内 K 类 min–max）与八时段单元分布",
            fontsize=11,
            y=0.98,
        )
        safe_fn = re.sub(r'[/\\:*?"<>|]', "_", glabel)
        fig_g.savefig(fig_dir / f"{safe_fn}_形功流雷达与八时段分布.png", dpi=200, bbox_inches="tight")
        plt.close(fig_g)

    if ENABLE_SCENARIO_ENGINE:
        non_baseline_scenarios = [sc for sc in SCENARIOS if int(sc["scenario_id"]) != 0]

        def _legacy_scenario_map_name(sid: int, sc_name: str) -> str:
            legacy = {
                0: "G03_现状延续情景推演图.png",
                1: "G04_南北连通增强情景推演图.png",
                2: "G05_公共服务与商业植入情景推演图.png",
                3: "G06_换乘压力疏解情景推演图.png",
            }
            if sid in legacy:
                return legacy[sid]
            slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", sc_name).strip("_")[:28]
            return f"G_scenario_{sid:02d}_{slug}_情景推演图.png"

        # 情景主图：各 scenario_id 的 next_global_state 空间分布
        for sc in SCENARIOS:
            sid = int(sc["scenario_id"])
            sub = trans_df[trans_df["scenario_id"] == sid][["unit_id", "next_global_state"]]
            mg = units_gdf.merge(sub, on="unit_id", how="left")
            cats2 = list(pd.unique(mg["next_global_state"].dropna()))
            mg["color"] = mg["next_global_state"].map({c: state_palette.get(str(c), "#bdbdbd") for c in cats2})
            fig, ax = plt.subplots(figsize=(10, 10))
            mg.plot(color=mg["color"].fillna("#dddddd"), ax=ax, edgecolor="0.25", linewidth=0.12, zorder=1)
            _overlay_site_structure_hints(ax, mg, site_boundary)
            site_ok = plot_site_boundary(ax, mg.crs, site_boundary)
            ax.set_title(f"情景{sid} {sc['name']}：推演后综合状态（WE_NT 锚点）")
            ax.axis("off")
            leg2 = [ln for ln in label_codes if ln in set(cats2)]
            ps = [Rectangle((0, 0), 1, 1, fc=state_palette[ln]) for ln in leg2]
            if site_ok:
                ps.append(Line2D([0], [0], color="#d90429", lw=2.2, linestyle=(0, (5, 3))))
                leg2.append("场地红线 (SITE.json)")
            ax.legend(ps, leg2, loc="lower left", fontsize=6, frameon=True)
            fig.savefig(fig_dir / _legacy_scenario_map_name(sid, str(sc["name"])), dpi=180, bbox_inches="tight")
            plt.close(fig)

        for sc in non_baseline_scenarios:
            sid = int(sc["scenario_id"])
            sn = str(sc["name"])
            pre = f"G_s{sid:02d}"
            sdf = trans_df[trans_df["scenario_id"] == sid]
            _plot_unit_category_map(
                units_gdf,
                sdf,
                "delta_leader_state",
                fig_dir / f"{pre}_{sn}_Δp主导增量.png",
                f"{pre} {sn}：Δp 最大增量状态（argmax(p_next−p_现状)）",
                site_boundary,
                cmap_name="tab20",
                category_colors=state_palette,
            )
            _plot_unit_category_map(
                units_gdf,
                sdf,
                "change_vs_baseline_three",
                fig_dir / f"{pre}_{sn}_相对现状主导变化三类.png",
                f"{pre} {sn}：相对现状主导类（维持 / 升级 / 其他变化）",
                site_boundary,
                cmap_name="Pastel1",
            )
            _plot_unit_category_map(
                units_gdf,
                sdf,
                "argmax_flipped_binary",
                fig_dir / f"{pre}_{sn}_主导状态是否变化.png",
                f"{pre} {sn}：主导状态是否相对现状改变（未变/改变）",
                site_boundary,
                cmap_name="Set1",
            )
            _plot_unit_float_map(
                units_gdf,
                sdf,
                "prob_tv_l1",
                fig_dir / f"{pre}_{sn}_状态分布总变差.png",
                f"{pre} {sn}：综合状态 soft 分布总变差 ½‖p'−p‖₁（连续量，利于看情景作用强度）",
                site_boundary,
                cmap_name="plasma",
            )

        if non_baseline_scenarios:
            knob_cols = sorted(
                [
                    c
                    for c in non_baseline_scenarios[0].keys()
                    if c not in ("scenario_id", "name") and isinstance(non_baseline_scenarios[0][c], (int, float))
                ]
            )
            nsc = len(non_baseline_scenarios)
            fig, axes = plt.subplots(nsc, 1, figsize=(9.0, max(2.0 * nsc, 5.5)), squeeze=False)
            for ri, sc in enumerate(non_baseline_scenarios):
                ax = axes[ri, 0]
                vals = [float(sc[c]) for c in knob_cols]
                ax.barh(knob_cols, vals, color="steelblue", height=0.62)
                ax.set_xlim(0.0, 1.05)
                ax.axvline(1.0, color="#bbb", lw=0.8, linestyle=":")
                ax.set_title(f"S{sc['scenario_id']} {sc['name']}", fontsize=9, loc="left")
            fig.suptitle("G09 情景旋钮强度一览（0–1； logits 仿射见源码 _scenario_affinity）", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            fig.savefig(fig_dir / "G09_情景旋钮强度一览.png", dpi=170, bbox_inches="tight")
            plt.close(fig)

        top = trans_df[trans_df["scenario_id"] == 1].nlargest(25, "priority_score")
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.barh(top["unit_id"][::-1], top["priority_score"][::-1], color="coral")
        ax.set_xlabel("priority_score")
        ax.set_title("G08 重点干预单元排序（南北连通增强）")
        fig.savefig(fig_dir / "G08_重点干预单元排序图.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print("Wrote:", out_dir / "urban_state_input.csv")
    print("Wrote:", out_dir / "global_state.csv")
    if ENABLE_SCENARIO_ENGINE:
        print("Wrote:", out_dir / "state_transition_result.csv")
    print("Figures:", fig_dir)


if __name__ == "__main__":
    main()
