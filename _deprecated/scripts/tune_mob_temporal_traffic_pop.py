"""
结合两类先验调制 mob_state 长表中的 traffic_intensity、population_density、stay_proxy：

1) MetroFlow（经 `time_slice_calibration.json` 的 `flow_proxy_period_weights_blended`）
   — 用真实缓冲池化得到的四窗 **mass / inflow / outflow** 相对形状（2017 节律，作形状先验）。
2) PPT 职住/OD + LBS 描述
   — 早：北向源、站核吸附；午：核心略抬；晚：东向扩散、出站型增强；夜间 Q4：站核条带 + 外围活力，
     人口 proxy 模拟 LBS「热点集中、外围傍晚抬、深夜略收」的 caricature。

仍重算 congestion_proxy；**动态** GMM 在 8+2+2+2 维上拟合；另用单元四窗 **FEAT_GMM 均值** 拟合 **静态** GMM（7 类、`diag`）。
匈牙利对齐动态簇与静态簇后，按单元内 **交通/人口相对偏离** 排序：约 **80% 单元 1 个时段**、**20% 单元 2 个时段** 采用动态标签，其余三（二）格沿用静态标签 → 全样本约 **30%** 格点与静态基底不同，且变化时段随单元而异。

用法:
  python scripts/tune_mob_temporal_traffic_pop.py
  python scripts/tune_mob_temporal_traffic_pop.py --weekend-blended   # 用 blended 的 weekend 权重
  python scripts/plot_mobility_state_pack.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import zlib
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from time_slice_constants import T_IDS, T_IDS_WEEKDAY, T_IDS_WEEKEND

REPO = Path(__file__).resolve().parents[1]


def expand_legacy_four_to_eight(df: pd.DataFrame) -> pd.DataFrame:
    """
    旧版四窗长表（WD_AM / WD_PM / WD_EVE / WE_PM）→ 规范八窗 ``T_IDS``。
    缺失窗格由同源窗复制（与 ``generate_worldpop_four_step_flow.MOB_T_SLICE_FALLBACK`` 语义一致；
    WE_PM 视作周末日间复合窗，用于 WE_MD / WE_NT 种子）。
    """
    present = set(df["t_id"].astype(str).unique())
    need = set(T_IDS)
    if need.issubset(present):
        return df
    legacy = {"WD_AM", "WD_PM", "WD_EVE", "WE_PM"}
    if present != legacy:
        raise ValueError(
            "expand_legacy_four_to_eight: 需要恰好四窗 "
            f"{sorted(legacy)}，实际为 {sorted(present)}"
        )
    src_for: dict[str, str] = {
        "WD_AM": "WD_AM",
        "WD_PM": "WD_PM",
        "WD_EVE": "WD_EVE",
        "WD_NT": "WD_EVE",
        "WE_AM": "WD_AM",
        "WE_MD": "WE_PM",
        "WE_EVE": "WD_EVE",
        "WE_NT": "WE_PM",
    }
    rows: list[pd.Series] = []
    for _, grp in df.groupby("unit_id", sort=False):
        by_t = {str(r["t_id"]): r for _, r in grp.iterrows()}
        for tid in T_IDS:
            r = by_t[src_for[tid]].copy()
            r["t_id"] = tid
            rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


DEFAULT_MOB = REPO / "output" / "flow" / "output_mobility_state" / "mob_state.csv"
DEFAULT_UNITS = REPO / "output" / "function" / "数据包" / "01_units.gpkg"
DEFAULT_SYNTH = REPO / "data" / "site_3km" / "poi_temporal_synthesis.json"
DEFAULT_CAL = REPO / "data" / "site_3km" / "metroflow" / "time_slice_calibration.json"

FEAT_GMM = [
    "road_centrality",
    "accessibility_index",
    "population_density",
    "traffic_intensity",
    "congestion_proxy",
    "barrier_index",
    "bottleneck_index",
    "stay_proxy",
]


def load_poi_mass_only(path: Path) -> dict[str, dict[str, float]]:
    """仅 POI 质量占比 → mass≈rhythm，win/wout=1（工作日/周末各 4 窗）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    swd = data.get("slices_weekday") or data.get("slices") or []
    swe = data.get("slices_weekend") or data.get("slices") or []

    def _mass_map(rows: list, key: str) -> dict[str, float]:
        return {str(s["t_id"]): float(s[key]) for s in rows if key in s}

    ms_wd = _mass_map(swd, "mass_share_weekday_in_slice")
    if not ms_wd and swd:
        ms_wd = _mass_map(swd, "mass_share_weekend_in_slice")
    ms_we = _mass_map(swe, "mass_share_weekend_in_slice")
    if not ms_we and swe:
        ms_we = _mass_map(swe, "mass_share_weekday_in_slice")
    if not ms_we:
        ms_we = ms_wd.copy()
    m0w = float(np.mean(list(ms_wd.values()))) if ms_wd else 1.0
    m0e = float(np.mean(list(ms_we.values()))) if ms_we else 1.0
    out: dict[str, dict[str, float]] = {}
    for t in T_IDS_WEEKDAY:
        out[t] = {"mass": ms_wd.get(t, 0.25) / m0w, "win": 1.0, "wout": 1.0}
    for t in T_IDS_WEEKEND:
        out[t] = {"mass": ms_we.get(t, 0.25) / m0e, "win": 1.0, "wout": 1.0}
    return out


def load_blended_flow(path: Path, weekend: bool) -> dict[str, dict[str, float]]:
    """`flow_proxy_period_weights_blended`：MetroFlow 与 POI 曲线凸混后的分日类型四窗形状。"""
    j = json.loads(path.read_text(encoding="utf-8"))
    key = "weekend" if weekend else "weekday"
    wd = j["flow_proxy_period_weights_blended"][key]
    want = T_IDS_WEEKEND if weekend else T_IDS_WEEKDAY
    mass_raw = {t: float(wd[t]["curve_mass_share"]) for t in want}
    win_raw = {t: float(wd[t]["period_inflow_weight"]) for t in want}
    wout_raw = {t: float(wd[t]["period_outflow_weight"]) for t in want}
    mm = float(np.mean(list(mass_raw.values())))
    im = float(np.mean(list(win_raw.values())))
    om = float(np.mean(list(wout_raw.values())))
    return {
        t: {
            "mass": mass_raw[t] / mm,
            "win": win_raw[t] / im,
            "wout": wout_raw[t] / om,
        }
        for t in want
    }


def station_proxy_xy(units: gpd.GeoDataFrame) -> tuple[float, float]:
    row = units.loc[units["dist_to_station"].astype(float).idxmin()]
    return float(row["centroid_x"]), float(row["centroid_y"])


def attach_spatial(df: pd.DataFrame, units: gpd.GeoDataFrame) -> pd.DataFrame:
    u = units[["unit_id", "dist_to_station", "centroid_x", "centroid_y"]].copy()
    out = df.merge(u, on="unit_id", how="left")
    sx, sy = station_proxy_xy(u)
    out["_sx"], out["_sy"] = sx, sy
    out["north_hat"] = np.tanh((out["centroid_y"].astype(float) - sy) / 0.012)
    out["east_hat"] = np.tanh((out["centroid_x"].astype(float) - sx) / 0.015)
    out["K"] = np.exp(-out["dist_to_station"].astype(float).clip(lower=0) / 1650.0)
    return out


def od_spatial_traffic_mult(row: pd.Series) -> float:
    """PPT 职住/OD：方向 caricature；午间略压站核，避免与晚高峰一样「满屏 R4/R6」。"""
    tid = str(row["t_id"])
    K, n, e = float(row["K"]), float(row["north_hat"]), float(row["east_hat"])
    if tid == "WD_AM":
        return (0.48 + 1.05 * K) * (1.0 + 0.62 * max(0.0, n))
    if tid == "WD_PM":
        return 0.62 + 0.36 * K
    if tid == "WD_EVE":
        return (0.52 + 0.88 * K) * (1.0 + 0.52 * max(0.0, e))
    if tid == "WD_NT":
        return (0.5 + 0.62 * K) * (1.0 + 0.28 * max(0.0, -n) + 0.26 * max(0.0, n))
    if tid == "WE_AM":
        return (0.5 + 1.02 * K) * (1.0 + 0.45 * max(0.0, n))
    if tid == "WE_MD":
        return 0.64 + 0.38 * K
    if tid == "WE_EVE":
        return (0.54 + 0.86 * K) * (1.0 + 0.48 * max(0.0, e))
    if tid == "WE_NT":
        return (0.48 + 0.6 * K) * (1.0 + 0.32 * max(0.0, -n) + 0.28 * max(0.0, n))
    return 1.0


# 午间交通再略压一档；傍晚/夜间与早峰拉开
PPT_TIME_SCALES_TRAFFIC = {
    "WD_AM": 1.12,
    "WD_PM": 0.82,
    "WD_EVE": 1.14,
    "WD_NT": 0.9,
    "WE_AM": 1.08,
    "WE_MD": 0.85,
    "WE_EVE": 1.12,
    "WE_NT": 0.92,
}
PPT_TIME_SCALES_POP = {
    "WD_AM": 1.06,
    "WD_PM": 1.05,
    "WD_EVE": 1.1,
    "WD_NT": 0.95,
    "WE_AM": 1.04,
    "WE_MD": 1.06,
    "WE_EVE": 1.08,
    "WE_NT": 0.96,
}


def metro_gate_on_station(K: np.ndarray, win: np.ndarray, wout: np.ndarray, tid: np.ndarray) -> np.ndarray:
    """Metro 分向 × 站核距离：略增强非站域的晚高峰出站感（与 PPT 外扩一致）。"""
    out = np.ones(len(K), dtype=float)
    for i, t in enumerate(tid):
        wi, wo, k = float(win[i]), float(wout[i]), float(K[i])
        if t == "WD_AM":
            out[i] *= 1.0 + 0.72 * k * max(0.0, wi - 1.0) + 0.15 * k
        elif t == "WD_PM":
            out[i] *= 0.88 * (1.0 + 0.22 * k * (abs(wi - 1.0) + abs(wo - 1.0)) * 0.5)
        elif t == "WD_NT":
            out[i] *= 1.0 + 0.52 * k * max(0.0, wo - 1.0) + 0.32 * (1.0 - k) * wo
        elif t in ("WE_AM", "WE_MD"):
            out[i] *= 0.9 * (1.0 + 0.2 * k * (abs(wi - 1.0) + abs(wo - 1.0)) * 0.5)
        elif t == "WE_EVE":
            out[i] *= 1.0 + 0.58 * k * max(0.0, wo - 1.0) + 0.35 * (1.0 - k) * max(0.0, wo - 1.0)
        elif t == "WE_NT":
            out[i] *= 1.0 + 0.5 * k * max(0.0, wo - 1.0) + 0.32 * (1.0 - k) * wo
    return out


def lbs_population_mult(row: pd.Series, mass: float) -> float:
    """
    PPT LBS：热点在站域锐化；午间略抬；傍晚外围抬；夜间站核仍高、整体略收。
    mass：已含 Metro+POI 分日类型四窗相对强度。
    """
    tid = str(row["t_id"])
    K, n, e = float(row["K"]), float(row["north_hat"]), float(row["east_hat"])
    core = 0.28 + 1.12 * (K**0.82)
    if tid == "WD_AM":
        core *= 0.92 + 0.28 * max(0.0, n)
    if tid == "WD_PM":
        core *= 0.98
    if tid == "WD_EVE":
        core *= 1.05 + 0.12 * max(0.0, e)
    if tid == "WD_NT":
        core *= 1.02 + 0.08 * max(0.0, -n)
    if tid == "WE_AM":
        core *= 0.95 + 0.22 * max(0.0, n)
    if tid == "WE_MD":
        core *= 1.0
    if tid == "WE_EVE":
        core *= 1.08 + 0.14 * max(0.0, e)
    if tid == "WE_NT":
        core *= 1.04 + 0.06 * max(0.0, -n)
    fringe = 1.0
    if tid in ("WD_EVE", "WD_NT", "WE_EVE", "WE_NT"):
        fringe = 1.0 + 0.62 * (1.0 - K) * (0.45 + 0.65 * max(0.0, n))
    night = 1.0
    if tid in ("WD_NT", "WE_NT"):
        night = 0.88 + 0.42 * K + 0.18 * (1.0 - K)
    return mass * core * fringe * night


def lbs_stay_mult(row: pd.Series, mass: float) -> float:
    """停留/界面活力 proxy：午后站核略抬（略弱于前版，减轻 WD_PM 满屏暖色/高轨道类）。"""
    tid = str(row["t_id"])
    K = float(row["K"])
    if tid == "WD_PM":
        return mass * (0.64 + 0.68 * (K**0.98))
    if tid == "WE_MD":
        return mass * (0.62 + 0.72 * (K**1.0))
    if tid in ("WD_NT", "WE_NT"):
        return mass * (0.52 + 1.18 * (K**1.15) + 0.22 * (1.0 - K))
    if tid == "WE_EVE":
        return mass * (0.56 + 1.02 * (K**1.05))
    if tid == "WD_EVE":
        return mass * (0.58 + 0.95 * (K**1.0))
    if tid == "WE_AM":
        return mass * (0.56 + 0.88 * (K**0.95))
    return mass * (0.55 + 0.85 * (K**0.95))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weekend-blended",
        action="store_true",
        help="（兼容占位）曾用于仅取周末 blended；现已合并工作日+周末四窗权重，此选项无实质效果。",
    )
    ap.add_argument("--poi-mass-only", action="store_true", help="不用 Metro 校准，仅用 POI 分日类型四窗 mass")
    args = ap.parse_args()

    mob_path = DEFAULT_MOB
    units_path = DEFAULT_UNITS
    synth_path = DEFAULT_SYNTH
    cal_path = DEFAULT_CAL

    if not mob_path.is_file():
        raise SystemExit(f"缺少 {mob_path}")
    if not units_path.is_file():
        raise SystemExit(f"缺少 {units_path}")

    if args.poi_mass_only or not cal_path.is_file():
        flow_w = (
            load_poi_mass_only(synth_path)
            if synth_path.is_file()
            else {t: {"mass": 1.0, "win": 1.0, "wout": 1.0} for t in T_IDS}
        )
        if args.poi_mass_only:
            print("Flow weights: POI mass only")
        else:
            print(f"Missing {cal_path.name}, fallback POI mass only")
    else:
        flow_w = {
            **load_blended_flow(cal_path, weekend=False),
            **load_blended_flow(cal_path, weekend=True),
        }
        print(f"Flow weights: blended weekday+weekend from {cal_path.name}")

    backup = mob_path.with_name(f"mob_state_before_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    shutil.copy2(mob_path, backup)
    print(f"Backed up -> {backup.name}")

    df = pd.read_csv(mob_path)
    n_rows_in = len(df)
    df = expand_legacy_four_to_eight(df)
    if len(df) != n_rows_in:
        print(f"Expanded legacy 4-slice rows {n_rows_in} -> {len(df)} (canonical 8 per unit).")
    prob_cols = [c for c in df.columns if c.startswith("p_R")]
    if len(prob_cols) != 7:
        raise SystemExit("期望 7 列 p_R*")

    try:
        units = gpd.read_file(units_path, layer="units")
    except Exception:
        units = gpd.read_file(units_path)

    df["traffic_intensity"] = df.groupby("unit_id")["traffic_intensity"].transform("first")
    df["population_density"] = df.groupby("unit_id")["population_density"].transform("first")
    df["stay_proxy"] = df.groupby("unit_id")["stay_proxy"].transform("first")

    df = attach_spatial(df, units)
    tid = df["t_id"].astype(str).to_numpy()
    mass = np.array([flow_w[t]["mass"] for t in tid], dtype=float)
    win = np.array([flow_w[t]["win"] for t in tid], dtype=float)
    wout = np.array([flow_w[t]["wout"] for t in tid], dtype=float)
    K = df["K"].astype(float).to_numpy()

    tr_od = df.apply(od_spatial_traffic_mult, axis=1).astype(float).to_numpy()
    tr_metro = metro_gate_on_station(K, win, wout, tid)
    df["traffic_intensity"] = df["traffic_intensity"].astype(float) * mass * tr_od * tr_metro
    ts_tr = df["t_id"].map(PPT_TIME_SCALES_TRAFFIC).astype(float).to_numpy()
    df["traffic_intensity"] *= ts_tr

    pop_row = [lbs_population_mult(df.iloc[i], flow_w[str(df.iloc[i]["t_id"])]["mass"]) for i in range(len(df))]
    df["population_density"] = df["population_density"].astype(float) * np.array(pop_row, dtype=float)
    ts_pop = df["t_id"].map(PPT_TIME_SCALES_POP).astype(float).to_numpy()
    df["population_density"] *= ts_pop

    st_base = np.maximum(df["stay_proxy"].astype(float).to_numpy(), 1e-5)
    st_row = np.array(
        [lbs_stay_mult(df.iloc[i], flow_w[str(df.iloc[i]["t_id"])]["mass"]) for i in range(len(df))],
        dtype=float,
    )
    df["stay_proxy"] = st_row * st_base

    mask_eve = df["t_id"].isin(["WD_EVE", "WD_NT", "WE_EVE", "WE_NT"])
    df.loc[mask_eve, "traffic_intensity"] *= 1.0 + 0.22 * np.tanh(np.log1p(df.loc[mask_eve, "population_density"].astype(float)) / 8.0)

    df["traffic_intensity"] = df["traffic_intensity"].clip(lower=0.0)
    df["population_density"] = df["population_density"].clip(lower=0.0)
    df["stay_proxy"] = df["stay_proxy"].clip(lower=0.0)

    rc = df["road_centrality"].astype(float).values
    bi = df["barrier_index"].astype(float).values
    ti = df["traffic_intensity"].astype(float).values
    df["congestion_proxy"] = rc * ti * bi

    df[FEAT_GMM] = df[FEAT_GMM].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Xbase = df[FEAT_GMM].astype(float)
    tr_rel = df.groupby("unit_id")["traffic_intensity"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    ).astype(float)
    pop_rel = df.groupby("unit_id")["population_density"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    ).astype(float)
    tr_gz = df.groupby("t_id")["traffic_intensity"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-6)).astype(float)
    pop_gz = df.groupby("t_id")["population_density"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-6)).astype(float)
    tr_gz = tr_gz.clip(-4.0, 4.0)
    pop_gz = pop_gz.clip(-4.0, 4.0)
    score_dev = (tr_gz.abs() + pop_gz.abs()).to_numpy(dtype=float)

    # ---- 静态基底：单元内 FEAT_GMM 八窗均值 → GMM（与行级同一 log1p+标准化空间可比）----
    u_mean = df.groupby("unit_id", sort=False)[FEAT_GMM].mean().reset_index()
    X_u = np.log1p(np.maximum(u_mean[FEAT_GMM].to_numpy(dtype=float), 0.0))
    scaler_b = StandardScaler()
    X_u_s = scaler_b.fit_transform(X_u)
    X_u_s = np.nan_to_num(X_u_s, nan=0.0, posinf=0.0, neginf=0.0)
    gmm_b = GaussianMixture(
        n_components=7,
        covariance_type="diag",
        random_state=43,
        n_init=10,
        max_iter=350,
        reg_covar=1e-4,
    )
    gmm_b.fit(X_u_s)
    b_unit = gmm_b.predict(X_u_s)
    b_map = dict(zip(u_mean["unit_id"].astype(str), b_unit.astype(int)))

    X8_row = np.log1p(np.maximum(df[FEAT_GMM].to_numpy(dtype=float), 0.0))
    X8_row_s = scaler_b.transform(X8_row)
    X8_row_s = np.nan_to_num(X8_row_s, nan=0.0, posinf=0.0, neginf=0.0)

    _pidx = {t: float(i) for i, t in enumerate(T_IDS)}
    pidx = df["t_id"].map(_pidx).astype(float).to_numpy()
    denom = max(len(T_IDS) - 1, 1)
    pt_sin = np.sin(pidx * (np.pi / float(denom))).reshape(-1, 1)
    pt_cos = np.cos(pidx * (np.pi / float(denom))).reshape(-1, 1)
    Xraw = np.hstack(
        [
            Xbase.to_numpy(),
            tr_rel.to_numpy().reshape(-1, 1),
            pop_rel.to_numpy().reshape(-1, 1),
            tr_gz.to_numpy().reshape(-1, 1),
            pop_gz.to_numpy().reshape(-1, 1),
            pt_sin,
            pt_cos,
        ]
    )
    Xlog = np.log1p(np.maximum(Xraw, 0.0))
    Xlog = np.nan_to_num(Xlog, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = StandardScaler().fit_transform(Xlog)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    gmm = GaussianMixture(
        n_components=7,
        covariance_type="diag",
        random_state=42,
        n_init=10,
        max_iter=350,
        reg_covar=1e-4,
    )
    gmm.fit(Xs)
    d_pred = gmm.predict(Xs)
    proba = gmm.predict_proba(Xs)

    # 动态簇 → 静态簇语义：按 8 维行向量质心距离匈牙利对齐
    stat_c = np.zeros((7, X8_row_s.shape[1]), dtype=float)
    dyn_c = np.zeros((7, X8_row_s.shape[1]), dtype=float)
    bu = np.array([b_map[str(u)] for u in df["unit_id"].astype(str)], dtype=int)
    for k in range(7):
        mk = bu == k
        if mk.any():
            stat_c[k] = X8_row_s[mk].mean(axis=0)
        else:
            stat_c[k] = X8_row_s.mean(axis=0)
    for j in range(7):
        mj = d_pred == j
        if mj.any():
            dyn_c[j] = X8_row_s[mj].mean(axis=0)
        else:
            dyn_c[j] = X8_row_s.mean(axis=0)
    cost = np.zeros((7, 7), dtype=float)
    for j in range(7):
        for k in range(7):
            cost[j, k] = float(np.linalg.norm(dyn_c[j] - stat_c[k]))
    r_i, c_i = linear_sum_assignment(cost)
    dyn_to_stat = np.arange(7, dtype=int)
    for a in range(len(r_i)):
        dyn_to_stat[r_i[a]] = int(c_i[a])
    d_aligned = dyn_to_stat[d_pred]
    proba_stat = np.zeros_like(proba)
    for j in range(7):
        proba_stat[:, dyn_to_stat[j]] += proba[:, j]

    order_map = {t: i for i, t in enumerate(T_IDS)}
    # 抑制单时段（尤其 WD_PM）垄断动态替换；鼓励晚间与周末段也承担一部分变化。
    t_balance_w = {
        "WD_AM": 0.24,
        "WD_PM": 1.08,
        "WD_EVE": 1.22,
        "WD_NT": 1.18,
        "WE_AM": 1.26,
        "WE_MD": 1.18,
        "WE_EVE": 1.20,
        "WE_NT": 1.24,
    }
    wd_non_am = ("WD_PM", "WD_EVE", "WD_NT")
    we_all = ("WE_AM", "WE_MD", "WE_EVE", "WE_NT")
    mob_arr = np.zeros(len(df), dtype=int)
    for uid, row_idx in df.groupby("unit_id", sort=False).groups.items():
        idx = np.array(sorted(row_idx, key=lambda p: order_map[str(df.at[p, "t_id"])]), dtype=int)
        b0 = int(b_map[str(uid)])
        cand = [int(p) for p in idx if str(df.at[p, "t_id"]) != "WD_AM"]
        if not cand:
            for p in idx:
                mob_arr[p] = b0 + 1
            continue
        # 按时段平衡权重重排，减少单时段垄断并提升周末段可见变化。
        sc_pairs = sorted(
            (
                (
                    float(score_dev[p]) * float(t_balance_w.get(str(df.at[p, "t_id"]), 1.0)),
                    p,
                )
                for p in cand
            ),
            reverse=True,
        )
        # 提高动态替换配额：多数单元 4 个时段，部分单元 5 个时段。
        n_take = 5 if (zlib.crc32(str(uid).encode("utf-8")) % 5 == 0) else 4
        n_take = min(n_take, len(sc_pairs))

        # 先按 hash 在工作日(非AM)与周末时段做均衡抽样，保证各时段都有机会出现变化。
        uid_hash = zlib.crc32(str(uid).encode("utf-8"))
        pref_wd = [wd_non_am[(uid_hash + k) % len(wd_non_am)] for k in range(len(wd_non_am))]
        pref_we = [we_all[(uid_hash // 7 + k) % len(we_all)] for k in range(len(we_all))]

        by_tid: dict[str, list[int]] = {}
        score_by_p: dict[int, float] = {}
        for sc, p in sc_pairs:
            tid_p = str(df.at[p, "t_id"])
            by_tid.setdefault(tid_p, []).append(int(p))
            score_by_p[int(p)] = float(sc)

        change_idx: set[int] = set()
        for t in pref_wd:
            arr_t = by_tid.get(t, [])
            if arr_t:
                change_idx.add(int(arr_t[0]))
                break
        for t in pref_we:
            if len(change_idx) >= n_take:
                break
            arr_t = by_tid.get(t, [])
            if arr_t:
                change_idx.add(int(arr_t[0]))
                break

        # 尽量再补一个周末次优时段，拉开周末内部差异。
        for t in pref_we[1:]:
            if len(change_idx) >= n_take:
                break
            arr_t = by_tid.get(t, [])
            if arr_t:
                change_idx.add(int(arr_t[0]))
                break

        # 再补一个工作日次优时段，拉开工作日内部差异。
        for t in pref_wd[1:]:
            if len(change_idx) >= n_take:
                break
            arr_t = by_tid.get(t, [])
            if arr_t:
                change_idx.add(int(arr_t[0]))
                break

        # 其余按综合分数补齐。
        for _, p in sc_pairs:
            if len(change_idx) >= n_take:
                break
            change_idx.add(int(p))

        for p in idx:
            if p in change_idx:
                # 变化时段优先选“非基底”的最高概率语义簇，避免变化槽位回落基底。
                ord_k = np.argsort(proba_stat[p])[::-1]
                pick = None
                for kk in ord_k:
                    kki = int(kk)
                    if kki != b0:
                        pick = kki
                        break
                if pick is None:
                    pick = int(d_aligned[p]) if int(d_aligned[p]) != b0 else b0
                mob_arr[p] = int(pick) + 1
            else:
                mob_arr[p] = b0 + 1

    df["mob_state"] = [f"R{int(x)}" for x in mob_arr]
    # 让 p_R 与最终选中的 mob_state 一致但保持软概率：避免 global 层仍只看到“旧 proba”而时段不变。
    p_soft = proba_stat.copy()
    p_soft = np.where(p_soft.sum(axis=1, keepdims=True) > 1e-12, p_soft / (p_soft.sum(axis=1, keepdims=True) + 1e-12), 1.0 / 7.0)
    for r in range(len(df)):
        sel = int(mob_arr[r] - 1)
        base = 0.35 * p_soft[r]
        base[sel] += 0.65
        s = float(base.sum()) + 1e-12
        p_soft[r] = base / s
    for i in range(7):
        df[f"p_R{i + 1}"] = p_soft[:, i]

    drop_cols = ("_sx", "_sy", "north_hat", "east_hat", "K", "dist_to_station", "centroid_x", "centroid_y")
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df.to_csv(mob_path, index=False, encoding="utf-8-sig")

    n_unit = df["unit_id"].nunique()
    b_slot = np.array([b_map[str(u)] + 1 for u in df["unit_id"].astype(str)], dtype=int)
    b_codes = np.array([f"R{int(x)}" for x in b_slot], dtype=object)
    frac_diff = float((df["mob_state"].to_numpy() != b_codes).mean())
    print(f"Slots differing from unit static base: {frac_diff:.1%} (target ~30%)")
    chg = sum(1 for _, g in df.groupby("unit_id") if g["mob_state"].nunique() > 1)
    print(f"Units with varying mob_state across 8 slices: {chg} / {n_unit}")
    for c in ("traffic_intensity", "population_density", "stay_proxy"):
        v = sum(1 for _, g in df.groupby("unit_id") if g[c].nunique() > 1)
        print(f"  {c}: units with cross-t variation {v} / {n_unit}")
    print(f"Wrote {mob_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
