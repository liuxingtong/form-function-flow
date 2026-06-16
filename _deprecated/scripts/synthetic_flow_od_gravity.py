#!/usr/bin/env python3
"""
合成出行（轻量四阶段骨架）：

1) 出行生成：单元 production / attraction（人口·POI·站核核；可选 Mob 列增强吸引）；写出 trip_generation.csv。
2) 出行分布：产约束重力得种子 OD，再 **Furness 双约束** 使行、列边际分别对齐产生/吸引先验（欧氏阻抗）。
3) 方式划分：轻量 Logit 式 softmax（步行/自行车/公交轨道/小汽车 proxy），用 Mob + 距离 + 阻隔 + conductance。
4) 交通分配：**基于路径的 Frank–Wolfe 型迭代**（与常见讲义一致）：每轮在当前阻抗下求全有全无辅助流 aux，
   边流更新 ``x <- x + (1/k)(aux - x)``（``k`` 为从 1 递增的全局步号，支持跨时段链式热启动时接续步号）。
   可选 ``--assignment-scheme aon_replace`` 恢复旧版「每轮整网替换辅助流」。
   多方式负载列与 ``plot_flow_modality_networks``（N01–N04）语义对齐；**拓扑为地块邻接网**，与 flow 图纸中的 GeoJSON 矢量路网非同一张图（见 synthetic_od_meta）。

不对真实观测标定；输出 CSV/JSON 供仿真或 validate_synthetic_flow.py 校验。

示例：
  python scripts/synthetic_flow_od_gravity.py \\
    --units output/function/数据包/01_units.gpkg \\
    --edges output/function/数据包/02_edges.csv \\
    --out-dir output/synthetic_flow
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from site_map_overlay import plot_site_boundary, resolve_site_json_path  # noqa: E402
from time_slice_constants import T_IDS, T_IDS_WEEKDAY, T_IDS_WEEKEND  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# 与 output/function/数据包/03_time_slices.csv、MetroFlow 校准一致（工作日 4 + 周末 4）
PERIOD_CHAIN_ORDER: tuple[tuple[str, str], ...] = tuple(
    [("weekday", t) for t in T_IDS_WEEKDAY] + [("weekend", t) for t in T_IDS_WEEKEND]
)

# 与 plot_flow_modality_networks 出图编号对齐（慢行拆步行/自行车两分配）
FLOW_MODAL_ASSIGN_KEYS: tuple[str, ...] = (
    "N01_pedestrian",
    "N01_bike",
    "N02_fast_auto",
    "N03_slow_auto",
    "N04_transit_proxy",
)

MODAL_OD_FLOW_KEYS: dict[str, str] = {
    "N01_pedestrian": "flow_walk",
    "N01_bike": "flow_bike",
    "N02_fast_auto": "flow_N02_fast_auto",
    "N03_slow_auto": "flow_N03_slow_auto",
    "N04_transit_proxy": "flow_transit",
}

MOD_TO_ALLOW_COL: dict[str, str] = {m: f"allow_{m}" for m in FLOW_MODAL_ASSIGN_KEYS}

# 各时段对 Logit 效用乘子（再归一化占比）；突出 AM 轨道、晚间小汽车等弱先验
MODE_PERIOD_UTIL_SCALE: dict[str, dict[str, float]] = {
    "WD_AM": {"walk": 1.02, "bike": 0.92, "transit": 1.18, "auto": 0.95},
    "WD_PM": {"walk": 1.0, "bike": 1.05, "transit": 1.0, "auto": 1.0},
    "WD_EVE": {"walk": 0.98, "bike": 1.08, "transit": 0.98, "auto": 1.05},
    "WD_NT": {"walk": 0.95, "bike": 1.0, "transit": 0.92, "auto": 1.12},
    "WE_AM": {"walk": 1.04, "bike": 0.94, "transit": 1.08, "auto": 0.98},
    "WE_MD": {"walk": 1.02, "bike": 1.06, "transit": 1.0, "auto": 1.02},
    "WE_EVE": {"walk": 0.97, "bike": 1.1, "transit": 0.96, "auto": 1.06},
    "WE_NT": {"walk": 0.93, "bike": 1.02, "transit": 0.9, "auto": 1.14},
}

# 与 MetroFlow / build_site_units_and_edges 默认一致（WGS84）
DEFAULT_STATION_LON = 121.451257271
DEFAULT_STATION_LAT = 31.249149419


def _load_units(path: Path) -> gpd.GeoDataFrame:
    try:
        u = gpd.read_file(path, layer="units")
    except Exception:
        u = gpd.read_file(path)
    if "unit_id" not in u.columns:
        raise ValueError("units 缺少 unit_id")
    if u.crs is None:
        u = u.set_crs(4326)
    return u


def _centroids_xy_m(u: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    um = u.to_crs(32651)
    if {"centroid_x", "centroid_y"}.issubset(u.columns):
        xs = pd.to_numeric(u["centroid_x"], errors="coerce").to_numpy(dtype=float)
        ys = pd.to_numeric(u["centroid_y"], errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(xs) | ~np.isfinite(ys)
        if bad.any():
            c = um.geometry.centroid
            xs = np.where(bad, c.x.to_numpy(), xs)
            ys = np.where(bad, c.y.to_numpy(), ys)
    else:
        c = um.geometry.centroid
        xs = c.x.to_numpy(dtype=float)
        ys = c.y.to_numpy(dtype=float)
    return xs, ys, um.index.to_numpy()


def _station_kernel(dist_m: np.ndarray, sigma_m: float, weight: float) -> np.ndarray:
    return 1.0 + float(weight) * np.exp(-np.clip(dist_m, 0.0, None) / max(float(sigma_m), 1.0))


def _column_or_ones(u: gpd.GeoDataFrame, name: str | None) -> np.ndarray:
    if not name or name not in u.columns:
        return np.ones(len(u), dtype=float)
    v = pd.to_numeric(u[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.maximum(v, 0.0)


def gravity_row_normalized(
    O: np.ndarray,
    A: np.ndarray,
    dist_m: np.ndarray,
    *,
    beta: float,
    d_floor_m: float,
) -> np.ndarray:
    """生产约束重力：行 i 之和 ≈ O_i；对角为 0。"""
    n = len(O)
    d = np.maximum(dist_m, float(d_floor_m))
    np.fill_diagonal(d, np.inf)
    F = A / (np.power(d, float(beta)))
    np.fill_diagonal(F, 0.0)
    row_sum = F.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-15] = 1.0
    G = O[:, None] * F / row_sum
    np.fill_diagonal(G, 0.0)
    return G


def furness_balance_od(
    seed: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    *,
    max_iter: int = 400,
    tol: float = 1e-9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Furness / IPF：在 ``seed`` 非负、**对角线为 0**（无区内出行）的矩阵上交替按行、列缩放，
    使行和逼近 ``row_target``、列和逼近 ``col_target``（双约束重力）。

    若行、列总量在浮点下不一致，将 ``col_target`` 按行总量比例缩放。对极小或全零格加 ``floor`` 以保证可除。
    """
    M = np.asarray(seed, dtype=np.float64).copy()
    n = int(M.shape[0])
    if M.ndim != 2 or M.shape[1] != n:
        raise ValueError("furness_balance_od expects a square matrix")
    if n < 2:
        Z = np.zeros((n, n), dtype=np.float64)
        return Z, {"furness_skipped": True, "reason": "n_lt_2"}

    r = np.maximum(np.asarray(row_target, dtype=np.float64).reshape(n), 0.0)
    c = np.maximum(np.asarray(col_target, dtype=np.float64).reshape(n), 0.0)
    rs = float(r.sum())
    cs = float(c.sum())
    if rs < 1e-18:
        return np.zeros((n, n), dtype=np.float64), {"furness_skipped": True, "reason": "zero_row_total"}
    if cs < 1e-18:
        c = np.ones(n, dtype=np.float64) * (rs / max(n, 1))
    else:
        c = c * (rs / cs)

    np.fill_diagonal(M, 0.0)
    floor = max(1e-20, 1e-16 * rs / max(n * (n - 1), 1))
    off = ~np.eye(n, dtype=bool)
    M[off] = np.where(M[off] < floor, floor, M[off])
    np.fill_diagonal(M, 0.0)

    meta: dict[str, Any] = {
        "furness_max_iter": int(max_iter),
        "furness_tol": float(tol),
        "furness_converged": False,
    }
    scale_ref = max(float(r.max()), float(c.max()), 1.0)
    it_final = 0
    for it in range(int(max_iter)):
        it_final = it + 1
        rs_cur = M.sum(axis=1)
        rs_cur = np.maximum(rs_cur, floor * 0.01 * n)
        M *= (r / rs_cur)[:, None]
        np.fill_diagonal(M, 0.0)
        cs_cur = M.sum(axis=0)
        cs_cur = np.maximum(cs_cur, floor * 0.01 * n)
        M *= (c / cs_cur)[None, :]
        np.fill_diagonal(M, 0.0)
        err_r = float(np.max(np.abs(M.sum(axis=1) - r)))
        err_c = float(np.max(np.abs(M.sum(axis=0) - c)))
        if err_r <= tol * scale_ref and err_c <= tol * scale_ref:
            meta["furness_converged"] = True
            break

    meta["furness_iters"] = it_final
    meta["furness_row_max_err"] = float(np.max(np.abs(M.sum(axis=1) - r)))
    meta["furness_col_max_err"] = float(np.max(np.abs(M.sum(axis=0) - c)))
    meta["furness_total_err"] = float(abs(M.sum() - rs))
    return M, meta


def build_adjacency(edges: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """无向多重边合并为权重和。"""
    wcol = "edge_conductance" if "edge_conductance" in edges.columns else None
    tcol = "walk_time_min" if "walk_time_min" in edges.columns else None
    acc: dict[tuple[str, str], float] = {}
    for row in edges.itertuples(index=False):
        a, b = str(getattr(row, "source_id")), str(getattr(row, "target_id"))
        if a > b:
            a, b = b, a
        w = 1.0
        if wcol:
            w *= float(getattr(row, wcol)) + 1e-9
        if tcol:
            w /= float(getattr(row, tcol)) + 1e-6
        acc[(a, b)] = acc.get((a, b), 0.0) + w
    adj: dict[str, list[tuple[str, float]]] = {}
    for (a, b), w in acc.items():
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    return adj


def dijkstra_tree(adj: dict[str, list[tuple[str, float]]], start: str) -> tuple[dict[str, float], dict[str, str | None]]:
    inf = 1e300
    dist: dict[str, float] = {}
    parent: dict[str, str | None] = {}
    dist[start] = 0.0
    parent[start] = None
    pq: list[tuple[float, str]] = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, inf):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, inf):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent


def path_directed_edges(parent: dict[str, str | None], s: str, t: str) -> list[tuple[str, str]]:
    if t not in parent:
        return []
    nodes: list[str] = []
    cur: str | None = t
    while cur is not None:
        nodes.append(cur)
        cur = parent[cur]
    nodes.reverse()
    if not nodes or nodes[0] != s:
        return []
    return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


def enrich_units_for_modal(u: gpd.GeoDataFrame, mob_csv: Path | None) -> pd.DataFrame:
    """合并 Mob 时间均值列，缺列填中位数/0。"""
    df = pd.DataFrame({"unit_id": u["unit_id"].astype(str)})
    for c in ("dist_to_station", "edge_conductance_mean"):
        if c in u.columns:
            df[c] = pd.to_numeric(u[c], errors="coerce")
    if mob_csv is not None and Path(mob_csv).is_file():
        mob = pd.read_csv(mob_csv, encoding="utf-8-sig")
        cols = [
            c
            for c in (
                "transit_facility_density",
                "station_attraction",
                "barrier_index",
                "accessibility_index",
            )
            if c in mob.columns
        ]
        if cols:
            g = mob.groupby("unit_id", as_index=False)[cols].mean()
            g["unit_id"] = g["unit_id"].astype(str)
            df = df.merge(g, on="unit_id", how="left")
    for c in ("dist_to_station", "edge_conductance_mean", "transit_facility_density", "station_attraction", "barrier_index", "accessibility_index"):
        if c not in df.columns:
            df[c] = 0.0 if c != "dist_to_station" else 800.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            fill = float(df[c].median()) if np.isfinite(df[c].median()) else (800.0 if c == "dist_to_station" else 0.0)
            df[c] = df[c].fillna(fill)
    if "edge_conductance_mean" in df.columns:
        df["edge_conductance_mean"] = df["edge_conductance_mean"].clip(0.0, None)
    return df


def modal_shares_softmax(
    d_m: float,
    row_o: pd.Series,
    row_d: pd.Series,
) -> tuple[float, float, float, float]:
    """步行 / 自行车 / 公交轨道 / 小汽车 占比（轻量 proxy，无标定）。"""
    ds_o = float(row_o.get("dist_to_station", 800.0))
    ds_d = float(row_d.get("dist_to_station", 800.0))
    tr = float(row_o.get("transit_facility_density", 0.0)) + float(row_d.get("transit_facility_density", 0.0))
    st = float(row_o.get("station_attraction", 0.0)) + float(row_d.get("station_attraction", 0.0))
    bar = (float(row_o.get("barrier_index", 0.0)) + float(row_d.get("barrier_index", 0.0))) * 0.5
    acc = (float(row_o.get("accessibility_index", 0.0)) + float(row_d.get("accessibility_index", 0.0))) * 0.5
    ec = (float(row_o.get("edge_conductance_mean", 0.0)) + float(row_d.get("edge_conductance_mean", 0.0))) * 0.5

    u_walk = -d_m / 320.0 + 0.95 * ec - 0.42 * bar + 0.55 * acc + 0.12 * np.exp(-ds_o / 2200.0) + 0.12 * np.exp(-ds_d / 2200.0)
    u_bike = -d_m / 750.0 + 0.55 * ec - 0.28 * bar + 0.08 * acc
    u_tr = (
        -d_m / 2200.0
        + 0.85 * np.log1p(tr + 1e-6)
        + 0.45 * np.log1p(st + 1e-6)
        + 1.05 * np.exp(-ds_o / 1300.0)
        + 1.05 * np.exp(-ds_d / 1300.0)
        - 0.18 * bar
    )
    u_car = -d_m / 4500.0 - 0.22 * bar - 0.12 * acc + 0.08 * (1.0 - min(ec, 1.2))

    u = np.array([u_walk, u_bike, u_tr, u_car], dtype=float)
    u -= float(np.max(u))
    p = np.exp(u)
    p /= float(np.sum(p)) + 1e-15
    return float(p[0]), float(p[1]), float(p[2]), float(p[3])


def modal_shares_softmax_period(
    d_m: float,
    row_o: pd.Series,
    row_d: pd.Series,
    t_id: str,
) -> tuple[float, float, float, float]:
    """在分时段（t_id）尺度下重算方式划分（效用乘子后再 softmax）。"""
    ds_o = float(row_o.get("dist_to_station", 800.0))
    ds_d = float(row_d.get("dist_to_station", 800.0))
    tr = float(row_o.get("transit_facility_density", 0.0)) + float(row_d.get("transit_facility_density", 0.0))
    st = float(row_o.get("station_attraction", 0.0)) + float(row_d.get("station_attraction", 0.0))
    bar = (float(row_o.get("barrier_index", 0.0)) + float(row_d.get("barrier_index", 0.0))) * 0.5
    acc = (float(row_o.get("accessibility_index", 0.0)) + float(row_d.get("accessibility_index", 0.0))) * 0.5
    ec = (float(row_o.get("edge_conductance_mean", 0.0)) + float(row_d.get("edge_conductance_mean", 0.0))) * 0.5

    u_walk = -d_m / 320.0 + 0.95 * ec - 0.42 * bar + 0.55 * acc + 0.12 * np.exp(-ds_o / 2200.0) + 0.12 * np.exp(-ds_d / 2200.0)
    u_bike = -d_m / 750.0 + 0.55 * ec - 0.28 * bar + 0.08 * acc
    u_tr = (
        -d_m / 2200.0
        + 0.85 * np.log1p(tr + 1e-6)
        + 0.45 * np.log1p(st + 1e-6)
        + 1.05 * np.exp(-ds_o / 1300.0)
        + 1.05 * np.exp(-ds_d / 1300.0)
        - 0.18 * bar
    )
    u_car = -d_m / 4500.0 - 0.22 * bar - 0.12 * acc + 0.08 * (1.0 - min(ec, 1.2))

    scales = MODE_PERIOD_UTIL_SCALE.get(t_id, MODE_PERIOD_UTIL_SCALE["WD_PM"])
    u = np.array(
        [
            u_walk + np.log(max(scales["walk"], 1e-9)),
            u_bike + np.log(max(scales["bike"], 1e-9)),
            u_tr + np.log(max(scales["transit"], 1e-9)),
            u_car + np.log(max(scales["auto"], 1e-9)),
        ],
        dtype=float,
    )
    u -= float(np.max(u))
    p = np.exp(u)
    p /= float(np.sum(p)) + 1e-15
    return float(p[0]), float(p[1]), float(p[2]), float(p[3])


def auto_highway_fraction(dist_m: float, half_ms: float = 1400.0) -> tuple[float, float]:
    """将小汽车 OD 拆成 N02（快速路主干 proxy）与 N03（其它机动车道路 proxy）份额。"""
    d = max(float(dist_m), 1.0)
    f_fast = float(np.clip(d / (d + half_ms), 0.0, 1.0))
    return f_fast, 1.0 - f_fast


def load_period_curve_mass(path: Path | None, *, day_key: str) -> dict[str, float]:
    """读取``curve_mass_share``（归一化质量和≈1）；工作日/周末各 4 窗。"""
    want = T_IDS_WEEKDAY if day_key == "weekday" else T_IDS_WEEKEND
    default = {t: 1.0 / len(want) for t in want}
    if path is None or not path.is_file():
        return default
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        blk = blob.get("flow_proxy_period_weights_blended") or blob.get("empirical_flow_proxy_period_weights")
        if not isinstance(blk, dict):
            return default
        day = blk.get(day_key) or blk.get("weekday")
        if not isinstance(day, dict):
            return default
        out: dict[str, float] = {}
        fallback = 1.0 / len(want)
        for t in want:
            cell = day.get(t)
            if isinstance(cell, dict) and "curve_mass_share" in cell:
                out[t] = float(cell["curve_mass_share"])
            else:
                out[t] = fallback
        s = sum(out.values())
        if s > 1e-12:
            for t in want:
                out[t] /= s
        return out
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return default


def _edge_uk(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def build_ped_assignment_network(edges: pd.DataFrame) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """无向键 → 基准步行时间（并联边取 min）、容量 proxy（conductance 求和）。"""
    pair_base, pair_cap, _ = build_modal_assignment_networks(edges)["N01_pedestrian"]
    return pair_base, pair_cap


def build_modal_assignment_networks(
    edges: pd.DataFrame,
) -> dict[str, tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], float]]:
    """各类交通方式可用的边表 → 各 modality 的 (pair_base, pair_cap, delay_alpha_scale)；支持 ``allow_*`` 列按方式过滤。"""
    if "walk_time_min" in edges.columns:
        tcol = "walk_time_min"
    elif "edge_cost" in edges.columns:
        tcol = "edge_cost"
    else:
        tcol = None
    has_wd = "walk_dist_m" in edges.columns
    has_cond = "edge_conductance" in edges.columns
    has_bar = "barrier_cost" in edges.columns
    has_ca = "cross_arterial" in edges.columns
    allow_cols_present = {m: MOD_TO_ALLOW_COL[m] in edges.columns for m in FLOW_MODAL_ASSIGN_KEYS}

    acc: dict[str, dict[tuple[str, str], dict[str, float]]] = {
        m: {} for m in FLOW_MODAL_ASSIGN_KEYS
    }
    delay_scales = {
        "N01_pedestrian": 1.0,
        "N01_bike": 0.74,
        "N02_fast_auto": 0.52,
        "N03_slow_auto": 0.9,
        "N04_transit_proxy": 0.36,
    }

    for row in edges.itertuples(index=False):
        a, b = str(getattr(row, "source_id")), str(getattr(row, "target_id"))
        uk = _edge_uk(a, b)
        if tcol:
            wt = max(float(getattr(row, tcol)), 1e-6)
        else:
            wt = 1.0
        wd_m = float(getattr(row, "walk_dist_m")) if has_wd else wt * 75.0
        wd_m = max(wd_m, 1e-6)
        cond = float(getattr(row, "edge_conductance")) if has_cond else 1.0
        cond = max(cond, 1e-9)
        bar = float(getattr(row, "barrier_cost")) if has_bar else 0.0
        ca = float(getattr(row, "cross_arterial")) if has_ca else 0.0

        bases = {
            "N01_pedestrian": wt,
            "N01_bike": wt * 0.38,
            "N02_fast_auto": max((wd_m / 1000.0) / 52.0 * 60.0 / max(1.0, 1.0 + 0.42 * min(cond, 1.85)), 1e-6),
            "N03_slow_auto": max((wd_m / 1000.0) / 26.0 * 60.0 + 0.07 * bar + 0.06 * ca, 1e-6),
            "N04_transit_proxy": max(wt * 0.44, 1e-6),
        }
        for m in FLOW_MODAL_ASSIGN_KEYS:
            if allow_cols_present[m]:
                try:
                    av = int(getattr(row, MOD_TO_ALLOW_COL[m]))
                except (TypeError, ValueError):
                    av = 1
                if av == 0:
                    continue
            base = bases[m]
            if uk not in acc[m]:
                acc[m][uk] = {"base": base, "cap": cond}
            else:
                acc[m][uk]["base"] = min(acc[m][uk]["base"], base)
                acc[m][uk]["cap"] += cond

    out: dict[str, tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], float]] = {}
    for m in FLOW_MODAL_ASSIGN_KEYS:
        pb = {k: float(v["base"]) for k, v in acc[m].items()}
        pc = {k: float(v["cap"]) for k, v in acc[m].items()}
        out[m] = (pb, pc, float(delay_scales[m]))
    return out


def _adj_from_pairs(
    pair_base: dict[tuple[str, str], float],
    pair_cap: dict[tuple[str, str], float],
    pair_flow: dict[tuple[str, str], float],
    *,
    delay_alpha: float,
    delay_power: float,
) -> dict[str, list[tuple[str, float]]]:
    """当前迭代下的步行阻抗图（对称）。"""
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for uk, base in pair_base.items():
        u, v = uk
        cf = max(pair_flow.get(uk, 0.0), 0.0)
        cap = pair_cap[uk]
        c = base * (1.0 + float(delay_alpha) * cf / (cap + 1e-9)) ** float(delay_power)
        adj[u].append((v, c))
        adj[v].append((u, c))
    return adj


def _directed_to_pair_flow(
    directed: dict[tuple[str, str], float],
    pair_keys: set[tuple[str, str]],
) -> dict[tuple[str, str], float]:
    """无向键上的总流量（双向相加），用于 BPR 类延迟。"""
    pair_flow = {uk: 0.0 for uk in pair_keys}
    for (a, b), f in directed.items():
        if f <= 0:
            continue
        uk = _edge_uk(a, b)
        if uk in pair_flow:
            pair_flow[uk] += float(f)
    return pair_flow


def assign_modal_iterative_delay(
    pair_base: dict[tuple[str, str], float],
    pair_cap: dict[tuple[str, str], float],
    od_rows: list[dict],
    *,
    flow_key: str,
    out_flow_column: str,
    lost_stat_key: str,
    max_origins: int,
    n_iters: int,
    delay_alpha: float,
    delay_power: float,
    delay_alpha_scale: float = 1.0,
    scheme: str = "frank_wolfe",
    fw_step_base: int = 0,
    warm_start_directed: dict[tuple[str, str], float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    基于路径的迭代分配。

    - ``frank_wolfe``（默认）：每轮求辅助全有全无流 aux，再按讲义 ``x <- x + (1/k)(aux-x)`` 更新；
      全局步号 ``k = fw_step_base + 本轮内序号``，便于分时段链式热启动。
    - ``aon_replace``：每轮用 aux 完全替换用于阻抗的边流（旧实现）。
    """
    eff_alpha = float(delay_alpha) * float(delay_alpha_scale)
    pair_keys = set(pair_base.keys())

    origin_totals: dict[str, float] = defaultdict(float)
    od_by_o: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in od_rows:
        o, d = str(r["origin_id"]), str(r["destination_id"])
        f = float(r.get(flow_key, 0.0))
        if f <= 0 or o == d:
            continue
        origin_totals[o] += f
        od_by_o[o].append((d, f))

    ranked_o = sorted(origin_totals.keys(), key=lambda x: -origin_totals[x])
    if max_origins > 0:
        ranked_o = ranked_o[: int(max_origins)]

    n_it = max(1, int(n_iters))
    sch = str(scheme).lower().strip()
    lost_flow = 0.0
    n_unreach = 0

    if sch == "aon_replace":
        pair_flow: dict[tuple[str, str], float] = {uk: 0.0 for uk in pair_keys}
        directed_final: dict[tuple[str, str], float] = defaultdict(float)
        for _it in range(n_it):
            adj = _adj_from_pairs(pair_base, pair_cap, pair_flow, delay_alpha=eff_alpha, delay_power=delay_power)
            iter_directed: dict[tuple[str, str], float] = defaultdict(float)
            iter_lost = 0.0
            iter_bad = 0
            for o in ranked_o:
                if o not in adj:
                    iter_lost += sum(fw for _, fw in od_by_o[o])
                    iter_bad += len(od_by_o[o])
                    continue
                _, parent = dijkstra_tree(adj, o)
                for dest, fw in od_by_o[o]:
                    if fw <= 0:
                        continue
                    pe = path_directed_edges(parent, o, dest)
                    if not pe:
                        iter_lost += fw
                        iter_bad += 1
                        continue
                    for a, b in pe:
                        iter_directed[(a, b)] += fw
            directed_final = iter_directed
            lost_flow = iter_lost
            n_unreach = iter_bad
            pair_flow = {uk: 0.0 for uk in pair_keys}
            for (a, b), fw in iter_directed.items():
                if fw <= 0:
                    continue
                uk = _edge_uk(a, b)
                pair_flow[uk] = pair_flow.get(uk, 0.0) + fw
    else:
        directed_current: dict[tuple[str, str], float] = defaultdict(float)
        if warm_start_directed:
            for e, fv in warm_start_directed.items():
                if fv > 0:
                    directed_current[tuple((str(e[0]), str(e[1])))] = float(fv)

        directed_final = directed_current
        base_step = max(0, int(fw_step_base))

        for local_k in range(1, n_it + 1):
            global_k = base_step + local_k
            alpha = 1.0 / float(max(global_k, 1))
            pair_flow = _directed_to_pair_flow(directed_current, pair_keys)
            adj = _adj_from_pairs(pair_base, pair_cap, pair_flow, delay_alpha=eff_alpha, delay_power=delay_power)
            aux_directed: dict[tuple[str, str], float] = defaultdict(float)
            iter_lost = 0.0
            iter_bad = 0
            for o in ranked_o:
                if o not in adj:
                    iter_lost += sum(fw for _, fw in od_by_o[o])
                    iter_bad += len(od_by_o[o])
                    continue
                _, parent = dijkstra_tree(adj, o)
                for dest, fw in od_by_o[o]:
                    if fw <= 0:
                        continue
                    pe = path_directed_edges(parent, o, dest)
                    if not pe:
                        iter_lost += fw
                        iter_bad += 1
                        continue
                    for a, b in pe:
                        aux_directed[(a, b)] += fw
            lost_flow = iter_lost
            n_unreach = iter_bad

            keys = set(directed_current.keys()) | set(aux_directed.keys())
            new_dc: dict[tuple[str, str], float] = defaultdict(float)
            for e in keys:
                old = float(directed_current.get(e, 0.0))
                aux = float(aux_directed.get(e, 0.0))
                nv = old + alpha * (aux - old)
                if nv > 1e-18:
                    new_dc[e] = nv
            directed_current = new_dc
            directed_final = directed_current

    rows = [
        {"source_id": a, "target_id": b, out_flow_column: float(v)}
        for (a, b), v in directed_final.items()
        if v > 1e-15
    ]
    if not rows:
        df_out = pd.DataFrame(columns=["source_id", "target_id", out_flow_column])
    else:
        df_out = pd.DataFrame(rows)
    stats = {
        lost_stat_key: float(lost_flow),
        f"{lost_stat_key}__unassigned_od_pairs": float(n_unreach),
        f"{out_flow_column}__assignment_iters": int(n_it),
        f"{out_flow_column}__assignment_scheme": sch,
        f"{out_flow_column}__fw_step_base": int(fw_step_base),
        f"{out_flow_column}__fw_step_end": int(fw_step_base + n_it),
        f"{out_flow_column}__delay_alpha_effective": float(eff_alpha),
        f"{out_flow_column}__assignment_origins_used": float(len(ranked_o)),
    }
    return df_out, stats


def assign_pedestrian_iterative_delay(
    pair_base: dict[tuple[str, str], float],
    pair_cap: dict[tuple[str, str], float],
    od_rows: list[dict],
    *,
    max_origins: int,
    n_iters: int,
    delay_alpha: float,
    delay_power: float,
    scheme: str = "frank_wolfe",
    fw_step_base: int = 0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """兼容旧接口：仅加载 flow_walk → flow_ped_aon。"""
    df, st = assign_modal_iterative_delay(
        pair_base,
        pair_cap,
        od_rows,
        flow_key="flow_walk",
        out_flow_column="flow_ped_aon",
        lost_stat_key="lost_walk_flow_aon",
        max_origins=max_origins,
        n_iters=n_iters,
        delay_alpha=delay_alpha,
        delay_power=delay_power,
        delay_alpha_scale=1.0,
        scheme=scheme,
        fw_step_base=int(fw_step_base),
    )
    legacy = {
        "lost_walk_flow_aon": float(st.get("lost_walk_flow_aon", 0.0)),
        "aon_unassigned_od_pairs": float(st.get("lost_walk_flow_aon__unassigned_od_pairs", 0.0)),
        "assignment_iters": int(max(1, n_iters)),
        "assignment_delay_alpha": float(delay_alpha),
        "assignment_delay_power": float(delay_power),
        "assignment_origins_used": float(st.get("flow_ped_aon__assignment_origins_used", 0.0)),
    }
    return df, legacy


def plot_trip_generation_maps(
    units: gpd.GeoDataFrame,
    tg: pd.DataFrame,
    out_path: Path,
    *,
    site_path: Path | None = None,
) -> None:
    """四阶段第 1 步可视化：先验产生/吸引（重力输入）与 OD 矩阵行列和（实现边际）。"""
    from matplotlib import font_manager

    preferred = ("Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS")
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in avail:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False

    cols_needed = ("prior_production", "prior_attraction", "trip_production", "trip_attraction")
    for c in cols_needed:
        if c not in tg.columns:
            raise ValueError(f"trip_generation 缺少列 {c}")
    mg = units.merge(tg[list(("unit_id",) + cols_needed)], on="unit_id", how="left")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 12))
    titles = (
        "先验出行产生量（人口×站核核等，×总出行）",
        "先验出行吸引量（POI×站核核等，×总出行）",
        "实现产生量（OD 行和）",
        "实现吸引量（OD 列和）",
    )
    cmaps = ("Blues", "Oranges", "Blues", "Oranges")
    for ax, col, title, cm_name in zip(axes.ravel(), cols_needed, titles, cmaps, strict=True):
        v = pd.to_numeric(mg[col], errors="coerce")
        hi = float(np.nanpercentile(v.to_numpy(), 98)) if v.notna().any() else 1.0
        hi = max(hi, 1e-12)
        mg.plot(
            column=col,
            ax=ax,
            cmap=cm_name,
            vmin=0.0,
            vmax=hi,
            legend=True,
            linewidth=0.06,
            edgecolor="0.25",
            legend_kwds={"shrink": 0.62, "label": col},
            missing_kwds={"color": "#e8e8e8"},
        )
        plot_site_boundary(ax, mg.crs, site_path)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("出行生成（第 1 阶段）：先验质量 vs 重力 OD 边际", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def allocate_edge_flows(
    adj: dict[str, list[tuple[str, float]]],
    od: pd.DataFrame,
    *,
    symmetrize: bool = True,
) -> pd.DataFrame:
    """按起点净流出将流量一步分摊到邻居边（试探性路径近似）。"""
    out_flow = od.groupby("origin_id")["flow"].sum()
    edge_map: dict[tuple[str, str], float] = {}

    for oid, tot in out_flow.items():
        o = str(oid)
        nbrs = adj.get(o)
        if not nbrs or float(tot) <= 0:
            continue
        ws = np.array([w for _, w in nbrs], dtype=float)
        s = float(ws.sum())
        if s < 1e-15:
            continue
        shares = ws / s
        for (nb, _w), share in zip(nbrs, shares.tolist(), strict=False):
            f = float(tot) * float(share)
            a, b = (o, str(nb)) if o <= str(nb) else (str(nb), o)
            edge_map[(a, b)] = edge_map.get((a, b), 0.0) + f

    rows = [{"source_id": a, "target_id": b, "synthetic_edge_flow": v} for (a, b), v in edge_map.items()]
    df = pd.DataFrame(rows)
    if symmetrize and len(df):
        df2 = df.rename(columns={"source_id": "target_id", "target_id": "source_id"})
        df = pd.concat([df, df2], ignore_index=True)
    return df


def build_period_modal_od_rows(
    od_modal_slice: list[dict],
    attr_ix: pd.DataFrame,
    def_row: pd.Series,
    wm_weekday: dict[str, float],
    wm_weekend: dict[str, float],
) -> list[dict]:
    """全日 OD × (weekday|weekend) × 各日类型四时段；质量权重来自 MetroFlow 混合校准。"""
    rows: list[dict] = []
    for row in od_modal_slice:
        o, d = str(row["origin_id"]), str(row["destination_id"])
        dm = float(row["dist_m"])
        base_f = float(row["flow"])
        try:
            ro = attr_ix.loc[o]
            rd = attr_ix.loc[d]
            if isinstance(ro, pd.DataFrame):
                ro = ro.iloc[0]
            if isinstance(rd, pd.DataFrame):
                rd = rd.iloc[0]
        except KeyError:
            ro = rd = def_row
        for day_type, wm in (("weekday", wm_weekday), ("weekend", wm_weekend)):
            t_ids = T_IDS_WEEKDAY if day_type == "weekday" else T_IDS_WEEKEND
            dmass = 1.0 / len(t_ids)
            for t_id in t_ids:
                mass = float(wm.get(t_id, dmass))
                fp = base_f * mass
                sw, sb, st, sa = modal_shares_softmax_period(dm, ro, rd, t_id)
                ff, fs = auto_highway_fraction(dm)
                rows.append(
                    {
                        "day_type": day_type,
                        "t_id": t_id,
                        "origin_id": o,
                        "destination_id": d,
                        "dist_m": dm,
                        "period_mass_share": mass,
                        "flow": fp,
                        "share_walk": sw,
                        "share_bike": sb,
                        "share_transit": st,
                        "share_auto": sa,
                        "share_auto_fast": sa * ff,
                        "share_auto_slow": sa * fs,
                        "flow_walk": fp * sw,
                        "flow_bike": fp * sb,
                        "flow_transit": fp * st,
                        "flow_auto": fp * sa,
                        "flow_N02_fast_auto": fp * sa * ff,
                        "flow_N03_slow_auto": fp * sa * fs,
                    }
                )
    return rows


def group_period_od_by_slice(per_rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    g: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in per_rows:
        g[(str(r["day_type"]), str(r["t_id"]))].append(r)
    return g


def directed_flow_dict_from_edge_df(df: pd.DataFrame, flow_col: str) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    if df is None or len(df) == 0 or flow_col not in df.columns:
        return out
    for row in df.itertuples(index=False):
        v = float(getattr(row, flow_col))
        if v <= 1e-18:
            continue
        out[(str(getattr(row, "source_id")), str(getattr(row, "target_id")))] = v
    return out


def run_period_chain_assignment(
    nets: dict[str, tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], float]],
    per_rows: list[dict],
    *,
    max_origins: int,
    n_iters: int,
    delay_alpha: float,
    delay_power: float,
    scheme: str,
    period_chain_order: tuple[tuple[str, str], ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """
    按 PERIOD_CHAIN_ORDER 顺序：各交通方式在相邻时段之间 **热启动**（上一时段均衡边流作 warm_start），
    Frank–Wolfe 全局步号 ``fw_step_base`` 接续，步长 ``1/k`` 不与首轮 ``aux`` 硬替换冲突。
    """
    grouped = group_period_od_by_slice(per_rows)
    chain = period_chain_order if period_chain_order is not None else PERIOD_CHAIN_ORDER
    long_rows: list[dict] = []
    warm: dict[str, dict[tuple[str, str], float]] = {m: {} for m in FLOW_MODAL_ASSIGN_KEYS}
    fw_off: dict[str, int] = {m: 0 for m in FLOW_MODAL_ASSIGN_KEYS}
    slices_used = 0

    for day_type, t_id in chain:
        slice_od = grouped.get((day_type, t_id), [])
        if not slice_od:
            continue
        slices_used += 1
        for mod in FLOW_MODAL_ASSIGN_KEYS:
            pb, pc, dscale = nets[mod]
            fk = MODAL_OD_FLOW_KEYS[mod]
            col = f"flow_{mod}_aon"
            df_m, _st_m = assign_modal_iterative_delay(
                pb,
                pc,
                slice_od,
                flow_key=fk,
                out_flow_column=col,
                lost_stat_key=f"lost_{mod}",
                max_origins=max_origins,
                n_iters=n_iters,
                delay_alpha=delay_alpha,
                delay_power=delay_power,
                delay_alpha_scale=float(dscale),
                scheme=scheme,
                fw_step_base=int(fw_off[mod]),
                warm_start_directed=warm[mod] if warm[mod] else None,
            )
            warm[mod] = directed_flow_dict_from_edge_df(df_m, col)
            fw_off[mod] += max(1, int(n_iters))

            for row in df_m.itertuples(index=False):
                fv = float(getattr(row, col))
                if fv <= 1e-18:
                    continue
                long_rows.append(
                    {
                        "day_type": day_type,
                        "t_id": t_id,
                        "modality": mod,
                        "source_id": str(getattr(row, "source_id")),
                        "target_id": str(getattr(row, "target_id")),
                        "flow_aon": fv,
                    }
                )

    summary: dict[str, float | int | bool] = {
        "period_edge_assignment_long_rows": int(len(long_rows)),
        "period_chain_slices_with_od": int(slices_used),
        "period_chain_fw_step_end_by_mod": {m: int(fw_off[m]) for m in FLOW_MODAL_ASSIGN_KEYS},
    }
    return pd.DataFrame(long_rows), summary


def main() -> int:
    ap = argparse.ArgumentParser(description="重力型合成 OD + 站核核 + 边流量近似")
    ap.add_argument("--units", type=Path, default=REPO / "output/function/数据包/01_units.gpkg")
    ap.add_argument("--edges", type=Path, default=REPO / "output/function/数据包/02_edges.csv")
    ap.add_argument(
        "--assignment-edges-csv",
        type=Path,
        default=None,
        help="交通分配专用边表（如 build_flow_road_assignment_edges.py 从 GeoJSON 生成）；省略则用 --edges（parcel 邻接）",
    )
    ap.add_argument("--out-dir", type=Path, default=REPO / "output/synthetic_flow")
    ap.add_argument("--total-trips", type=float, default=1.0, help="归一化总出行量（默认 1）")
    ap.add_argument("--beta", type=float, default=2.0, help="距离衰减指数")
    ap.add_argument("--d-floor-m", type=float, default=25.0, help="距离下限，避免奇异")
    ap.add_argument("--station-sigma-m", type=float, default=1650.0)
    ap.add_argument("--station-weight", type=float, default=1.15, help="站核 exp 核强度")
    ap.add_argument("--transfer-fraction", type=float, default=0.12, help="Plan C：换乘/到发类 OD 占比（独立一层再混合）")
    ap.add_argument("--station-band-m", type=float, default=400.0, help="距站小于该值的单元参与换乘层出发")
    ap.add_argument("--col-production", type=str, default="population_density", help="生成侧列名；缺列则用 1")
    ap.add_argument("--col-attraction", type=str, default="poi_density", help="吸引侧列名；缺列则用 1")
    ap.add_argument("--long-format-max-rows", type=int, default=400_000, help="long CSV 最多行数（按流量截断）")
    ap.add_argument("--no-edge-flows", action="store_true")
    ap.add_argument(
        "--mob-csv",
        type=Path,
        default=REPO / "output/flow/output_mobility_state/mob_state.csv",
        help="Mob 长表；用于方式划分 proxy（不存在则仅用单元可用列）",
    )
    ap.add_argument(
        "--no-four-step-extras",
        action="store_true",
        help="不写出方式划分列与步行网络分配（仅 OD + 一步邻居边流）",
    )
    ap.add_argument(
        "--aon-max-origins",
        type=int,
        default=0,
        help="分配阶段最多扫描的起点数；0 表示不截断（全部有步行 OD 的起点）",
    )
    ap.add_argument(
        "--assignment-iters",
        type=int,
        default=5,
        help="交通分配迭代次数（每轮按当前拥堵阻抗重算最短路并加载流量）",
    )
    ap.add_argument(
        "--assignment-delay-alpha",
        type=float,
        default=0.22,
        help="阻抗反馈强度：cost *= (1 + alpha * flow_on_edge / conductance)^power",
    )
    ap.add_argument(
        "--assignment-delay-power",
        type=float,
        default=2.0,
        help="拥堵非线性指数（通常取 2）",
    )
    ap.add_argument(
        "--no-trip-gen-plots",
        action="store_true",
        help="不写出出行生成四面板地图 trip_generation_maps.png",
    )
    ap.add_argument(
        "--site-json",
        type=Path,
        default=None,
        help="出行生成地图叠加 SITE（GeoJSON）；省略则 data/site_3km/SITE.json 回退",
    )
    ap.add_argument(
        "--metroflow-calibration-json",
        type=Path,
        default=REPO / "data/site_3km/metroflow/time_slice_calibration.json",
        help="各日型四时段 curve_mass_share（weekday/weekend）；用于 synthetic_od_modal_by_period_long.csv",
    )
    ap.add_argument(
        "--no-period-od",
        action="store_true",
        help="不写分日类型×各四时段的长表 synthetic_od_modal_by_period_long.csv",
    )
    ap.add_argument(
        "--period-od-max-rows",
        type=int,
        default=150_000,
        help="分时段 OD 表最多保留的 OD 条数（按全日流量降序截断，避免爆盘）",
    )
    ap.add_argument(
        "--no-multimodal-edge-assignment",
        action="store_true",
        help="仅做 N01 步行边分配（与旧版一致）；默认对五种 flow 模态各做一次 parcel 网分配",
    )
    ap.add_argument(
        "--assignment-scheme",
        choices=("frank_wolfe", "aon_replace"),
        default="frank_wolfe",
        help="frank_wolfe：基于路径的 FW 更新 x←x+(1/k)(aux−x)；aon_replace：每轮阻抗仅用整网 aux（旧）",
    )
    ap.add_argument(
        "--assign-edges-per-period",
        action="store_true",
        help="在 synthetic_od_modal_by_period_long 对应的 OD 上按时段链式分配（热启动），写出 synthetic_edge_flow_period_long.csv",
    )
    ns = ap.parse_args()

    if ns.assign_edges_per_period and (ns.no_four_step_extras or ns.no_period_od):
        print(
            "错误：--assign-edges-per-period 需启用分时段 OD（不要使用 --no-four-step-extras 或 --no-period-od）",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    u = _load_units(ns.units)
    uid = u["unit_id"].astype(str).tolist()
    xs, ys, _ = _centroids_xy_m(u)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_m = np.sqrt(dx * dx + dy * dy)

    d_sta = None
    if "dist_to_station" in u.columns:
        raw = pd.to_numeric(u["dist_to_station"], errors="coerce").to_numpy(dtype=float)
        med = float(np.nanmedian(raw[np.isfinite(raw)])) if np.isfinite(raw).any() else 500.0
        d_sta = np.nan_to_num(raw, nan=med)
    else:
        st = gpd.GeoSeries(gpd.points_from_xy([DEFAULT_STATION_LON], [DEFAULT_STATION_LAT]), crs=4326).to_crs(32651)
        sx, sy = float(st.geometry.x.iloc[0]), float(st.geometry.y.iloc[0])
        d_sta = np.hypot(xs - sx, ys - sy)

    K = _station_kernel(d_sta, ns.station_sigma_m, ns.station_weight)
    P = _column_or_ones(u, ns.col_production) * K
    A_attr = _column_or_ones(u, ns.col_attraction) * K
    P = P / (P.sum() + 1e-12)
    A_attr = A_attr / (A_attr.sum() + 1e-12)

    Od_base = gravity_row_normalized(P, A_attr, dist_m, beta=ns.beta, d_floor_m=ns.d_floor_m)
    mask_xfer_o = d_sta <= float(ns.station_band_m)
    Ox = np.where(mask_xfer_o, P, 0.0)
    sxo = Ox.sum()
    if sxo < 1e-12:
        Ox = np.ones_like(P) / len(P)
    else:
        Ox = Ox / sxo
    Od_xfer = gravity_row_normalized(Ox, A_attr, dist_m, beta=ns.beta * 0.92, d_floor_m=ns.d_floor_m)

    tf = float(np.clip(ns.transfer_fraction, 0.0, 0.95))
    tt = float(ns.total_trips)
    seed = ((1.0 - tf) * Od_base + tf * Od_xfer) * tt
    OD, _furn_meta = furness_balance_od(seed, P * tt, A_attr * tt)

    od_long = []
    n = len(uid)
    flat = OD.ravel()
    idx = np.argsort(-flat)
    max_rows = int(ns.long_format_max_rows)
    for k in idx:
        if flat[k] <= 0:
            break
        if len(od_long) >= max_rows:
            break
        i, j = divmod(int(k), n)
        if i == j:
            continue
        od_long.append({"origin_id": uid[i], "destination_id": uid[j], "flow": float(flat[k]), "dist_m": float(dist_m[i, j])})

    mob_path = Path(ns.mob_csv) if ns.mob_csv is not None else None
    if mob_path is not None and not mob_path.is_file():
        mob_path = None

    row_mass = OD.sum(axis=1)
    col_mass = OD.sum(axis=0)
    tt = float(ns.total_trips)
    tg_df = pd.DataFrame(
        {
            "unit_id": uid,
            "prior_production": P * tt,
            "prior_attraction": A_attr * tt,
            "trip_production": row_mass,
            "trip_attraction": col_mass,
        }
    )
    tg_df.to_csv(out_dir / "trip_generation.csv", index=False, encoding="utf-8-sig")

    if not ns.no_trip_gen_plots:
        sp = ns.site_json if ns.site_json is not None and ns.site_json.is_file() else resolve_site_json_path()
        plot_trip_generation_maps(u, tg_df, out_dir / "trip_generation_maps.png", site_path=sp)

    extras = not bool(ns.no_four_step_extras)
    od_modal: list[dict] = []
    per_rows: list[dict] | None = None
    if extras:
        attr_tbl = enrich_units_for_modal(u, mob_path)
        attr_ix = attr_tbl.set_index("unit_id")
        def_row = pd.Series({c: 0.0 for c in attr_ix.columns}, dtype=float)
        def_row["dist_to_station"] = 800.0
        for row in od_long:
            o, d = str(row["origin_id"]), str(row["destination_id"])
            dm = float(row["dist_m"])
            try:
                ro = attr_ix.loc[o]
                rd = attr_ix.loc[d]
                if isinstance(ro, pd.DataFrame):
                    ro = ro.iloc[0]
                if isinstance(rd, pd.DataFrame):
                    rd = rd.iloc[0]
            except KeyError:
                ro = rd = def_row
            sw, sb, st, sa = modal_shares_softmax(dm, ro, rd)
            fv = float(row["flow"])
            ff, fs = auto_highway_fraction(dm)
            od_modal.append(
                {
                    **row,
                    "share_walk": sw,
                    "share_bike": sb,
                    "share_transit": st,
                    "share_auto": sa,
                    "share_auto_fast": sa * ff,
                    "share_auto_slow": sa * fs,
                    "flow_walk": fv * sw,
                    "flow_bike": fv * sb,
                    "flow_transit": fv * st,
                    "flow_auto": fv * sa,
                    "flow_N02_fast_auto": fv * sa * ff,
                    "flow_N03_slow_auto": fv * sa * fs,
                }
            )
        pd.DataFrame(od_modal).to_csv(out_dir / "synthetic_od_modal_long.csv", index=False, encoding="utf-8-sig")
    else:
        od_modal = []
        for r in od_long:
            fv = float(r["flow"])
            od_modal.append(
                {
                    **r,
                    "share_walk": 1.0,
                    "share_bike": 0.0,
                    "share_transit": 0.0,
                    "share_auto": 0.0,
                    "share_auto_fast": 0.0,
                    "share_auto_slow": 0.0,
                    "flow_walk": fv,
                    "flow_bike": 0.0,
                    "flow_transit": 0.0,
                    "flow_auto": 0.0,
                    "flow_N02_fast_auto": 0.0,
                    "flow_N03_slow_auto": 0.0,
                }
            )

    if extras and not ns.no_period_od:
        wm_wd = load_period_curve_mass(Path(ns.metroflow_calibration_json), day_key="weekday")
        wm_we = load_period_curve_mass(Path(ns.metroflow_calibration_json), day_key="weekend")
        cap = max(0, int(ns.period_od_max_rows))
        sl = sorted(od_modal, key=lambda r: -float(r["flow"]))[:cap]
        per_rows = build_period_modal_od_rows(sl, attr_ix, def_row, wm_wd, wm_we)
        pd.DataFrame(per_rows).to_csv(
            out_dir / "synthetic_od_modal_by_period_long.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(od_long).to_csv(out_dir / "synthetic_od_long.csv", index=False, encoding="utf-8-sig")

    meta = {
        "method": (
            "four_step_lite: trip_gen + gravity_od + modal_softmax + "
            "path_based_assignment (Frank–Wolfe step 1/k by default; modalities N01–N04 proxy)"
        ),
        "plan_notes": (
            "交通分配默认采用讲义中的路径型 Frank–Wolfe：每轮基于当前阻抗求全有全无辅助流 aux，"
            "边流更新 x←x+(1/k)(aux−x)，全局步号 k 在分时段链式分配时可接续。"
            "Parcel 邻接 ``--edges`` 仍可用于一步邻居 synthetic_edge_flow；交通分配可用 ``--assignment-edges-csv`` 挂接 flow GeoJSON 拓扑。"
            "若未指定 ``--assignment-edges-csv``，分配拓扑为 parcel ``02_edges``；若指定则为 GeoJSON 离散路网 + 质心接驳边。"
            "分时段 OD：MetroFlow×POI 混合 curve_mass_share × 各日型四时段方式效用微调。"
        ),
        "inputs": {
            "units": str(ns.units),
            "edges_parcel": str(ns.edges),
            "assignment_edges_csv": str(ns.assignment_edges_csv) if ns.assignment_edges_csv else None,
            "mob_csv_used": str(mob_path) if mob_path else None,
            "metroflow_calibration_json": str(ns.metroflow_calibration_json),
        },
        "parameters": {
            "total_trips": float(ns.total_trips),
            "beta": float(ns.beta),
            "d_floor_m": float(ns.d_floor_m),
            "station_sigma_m": float(ns.station_sigma_m),
            "station_weight": float(ns.station_weight),
            "transfer_fraction": tf,
            "station_band_m": float(ns.station_band_m),
            "col_production": ns.col_production,
            "col_attraction": ns.col_attraction,
            "period_od_max_rows": int(ns.period_od_max_rows),
            "no_period_od": bool(ns.no_period_od),
            "no_multimodal_edge_assignment": bool(ns.no_multimodal_edge_assignment),
            "assignment_scheme": ns.assignment_scheme,
            "assign_edges_per_period": bool(ns.assign_edges_per_period),
        },
        "flow_modal_assign_keys": list(FLOW_MODAL_ASSIGN_KEYS),
        "modal_od_flow_columns": MODAL_OD_FLOW_KEYS,
        "time_slices_T_IDS": list(T_IDS),
        "arrays_shape": [n, n],
        "od_sum": float(OD.sum()),
        "od_long_rows_written": len(od_long),
        "station_lon_lat_default": [DEFAULT_STATION_LON, DEFAULT_STATION_LAT],
        "four_step_extras": extras,
        "aon_max_origins": int(ns.aon_max_origins),
        "assignment_iters": int(ns.assignment_iters),
        "assignment_delay_alpha": float(ns.assignment_delay_alpha),
        "assignment_delay_power": float(ns.assignment_delay_power),
    }
    (out_dir / "synthetic_od_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not ns.no_edge_flows and ns.edges.is_file():
        edges = pd.read_csv(ns.edges)
        edges_assign = edges
        if ns.assignment_edges_csv is not None:
            pae = Path(ns.assignment_edges_csv)
            if pae.is_file():
                edges_assign = pd.read_csv(pae, encoding="utf-8-sig")
                meta["assignment_edges_used"] = str(pae.resolve())
            else:
                meta["assignment_edges_csv_missing"] = str(pae)
        adj = build_adjacency(edges)
        ef = allocate_edge_flows(adj, pd.DataFrame(od_long), symmetrize=True)
        ef.to_csv(out_dir / "synthetic_edge_flow.csv", index=False, encoding="utf-8-sig")
        meta["edge_flow_rows"] = int(len(ef))
        if extras:
            mo = int(ns.aon_max_origins)
            sch = str(ns.assignment_scheme)
            nets = build_modal_assignment_networks(edges_assign)
            if ns.no_multimodal_edge_assignment:
                pair_base, pair_cap, _ = nets["N01_pedestrian"]
                aon_df, aon_stats = assign_pedestrian_iterative_delay(
                    pair_base,
                    pair_cap,
                    od_modal,
                    max_origins=mo,
                    n_iters=int(ns.assignment_iters),
                    delay_alpha=float(ns.assignment_delay_alpha),
                    delay_power=float(ns.assignment_delay_power),
                    scheme=sch,
                )
                meta.update(aon_stats)
                if len(aon_df):
                    aon_df.to_csv(out_dir / "synthetic_edge_flow_aon_ped.csv", index=False, encoding="utf-8-sig")
                    meta["edge_flow_aon_ped_rows"] = int(len(aon_df))
            else:
                merged: pd.DataFrame | None = None
                for mod in FLOW_MODAL_ASSIGN_KEYS:
                    pb, pc, dscale = nets[mod]
                    fk = MODAL_OD_FLOW_KEYS[mod]
                    col = f"flow_{mod}_aon"
                    df_m, st_m = assign_modal_iterative_delay(
                        pb,
                        pc,
                        od_modal,
                        flow_key=fk,
                        out_flow_column=col,
                        lost_stat_key=f"lost_{mod}",
                        max_origins=mo,
                        n_iters=int(ns.assignment_iters),
                        delay_alpha=float(ns.assignment_delay_alpha),
                        delay_power=float(ns.assignment_delay_power),
                        delay_alpha_scale=float(dscale),
                        scheme=sch,
                    )
                    meta.update(st_m)
                    merged = df_m if merged is None else merged.merge(df_m, on=["source_id", "target_id"], how="outer")
                if merged is not None and len(merged.columns) > 2:
                    fill_cols = [c for c in merged.columns if c.startswith("flow_")]
                    merged[fill_cols] = merged[fill_cols].fillna(0.0)
                    multimodal_path = out_dir / "synthetic_edge_flow_aon_multimodal.csv"
                    merged.to_csv(multimodal_path, index=False, encoding="utf-8-sig")
                    meta["edge_flow_aon_multimodal_rows"] = int(len(merged))
                    ped_col = "flow_N01_pedestrian_aon"
                    if ped_col in merged.columns:
                        merged[["source_id", "target_id", ped_col]].rename(columns={ped_col: "flow_ped_aon"}).to_csv(
                            out_dir / "synthetic_edge_flow_aon_ped.csv",
                            index=False,
                            encoding="utf-8-sig",
                        )
                        meta["edge_flow_aon_ped_rows"] = int(len(merged))
            if ns.assign_edges_per_period and per_rows is not None:
                df_period_long, period_summary = run_period_chain_assignment(
                    nets,
                    per_rows,
                    max_origins=mo,
                    n_iters=int(ns.assignment_iters),
                    delay_alpha=float(ns.assignment_delay_alpha),
                    delay_power=float(ns.assignment_delay_power),
                    scheme=sch,
                )
                meta["period_edge_assignment"] = period_summary
                if len(df_period_long) > 0:
                    df_period_long.to_csv(
                        out_dir / "synthetic_edge_flow_period_long.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
        (out_dir / "synthetic_od_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {out_dir / 'synthetic_od_long.csv'} and meta ({len(od_long)} long rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
