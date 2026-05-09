"""
根据已有 mob_state.csv + mob_state_labels.json 统一生成「第三人」运行压力层图纸，
与功能层（analysis.py）对齐：聚类呈现、四时段分布、转移矩阵、雷达、序列、识别、指数图。

输出目录默认：output/flow/output_mobility_state/
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors, font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "output" / "flow" / "output_mobility_state"
DEFAULT_UNITS = REPO / "output" / "function" / "数据包" / "01_units.gpkg"
T_IDS = ("WD_AM", "WD_PM", "WD_EVE", "WE_PM")

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from site_map_overlay import load_site_gdf, plot_site_boundary, resolve_site_json_path  # noqa: E402

MOB_NUM_COLS = [
    "road_centrality",
    "accessibility_index",
    "population_density",
    "traffic_intensity",
    "congestion_proxy",
    "barrier_index",
    "bottleneck_index",
    "stay_proxy",
]


def configure_matplotlib_chinese_font() -> None:
    preferred_fonts = ("SimHei", "Microsoft YaHei", "STHeiti")
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((font for font in preferred_fonts if font in available_fonts), None)
    if selected_font is not None:
        plt.rcParams["font.sans-serif"] = [selected_font]
    else:
        plt.rcParams["font.sans-serif"] = list(preferred_fonts)
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_chinese_font()


def _z(d: dict, k: str) -> float:
    return float(d.get(k, 0.0))


def infer_one_mobility_name(mean_z: dict) -> str:
    """按 GMM 分量 z 画像命名；统一为「负荷—拥堵—结构/区位」三轴（高/中/低）。"""
    tr = _z(mean_z, "transit_facility_density")
    st = _z(mean_z, "station_attraction")
    bi = _z(mean_z, "barrier_index")
    ai = _z(mean_z, "accessibility_index")
    bn = _z(mean_z, "bottleneck_index")
    ti = _z(mean_z, "traffic_intensity")
    cg = _z(mean_z, "congestion_proxy")
    fo = _z(mean_z, "flow_out_proxy")
    fi = _z(mean_z, "flow_in_proxy")
    sy = _z(mean_z, "stay_proxy")
    rc = _z(mean_z, "road_centrality")

    if tr > 2.0 and st > 0.8 and (fi > 2.0 or sy > 2.0):
        return "负荷中·拥堵中·轨道强"
    if ai < -0.85 and bn > 0.85 and bi > 0.85 and cg > 0.5:
        return "负荷中高·拥堵高·可达低"
    if st > 1.0 and fo > 1.0 and rc > 0.45 and cg > 0.45:
        return "负荷高·拥堵高·站向高"
    if bi < -0.65 and ai > 0.45 and ti > 0.35 and cg > 0.35:
        return "负荷高·拥堵高·阻隔低"
    if ti < -0.9 and cg < -0.9 and ai > 0.35:
        return "负荷低·拥堵低·阻隔低"
    if ti < -0.9 and bi > 0.55 and ai < -0.3:
        return "负荷低·拥堵低·阻隔高"
    if ti > 0.45 and cg > 0.45 and st < -0.05 and tr < 1.0:
        return "负荷高·拥堵高·站域弱"
    if rc > 0.35 and bn > 0.45 and bi > 0.4 and st > 0.8:
        return "负荷中高·拥堵高·中心性高"
    return "负荷中·拥堵中·形态混合"


def dedupe_state_names(names: list[str]) -> list[str]:
    from collections import Counter

    c: Counter[str] = Counter()
    out: list[str] = []
    for n in names:
        c[n] += 1
        out.append(n if c[n] == 1 else f"{n}（变体{c[n]}）")
    return out


def load_components_labels(path: Path) -> tuple[int, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    comps = sorted(data["components"], key=lambda x: int(x["id"]))
    return int(data["n_components"]), comps


def build_state_names(comps: list[dict]) -> tuple[str, ...]:
    raw = [infer_one_mobility_name(c["mean_z"]) for c in comps]
    fixed = dedupe_state_names(raw)
    return tuple(fixed)


def transition_matrix_from_csv(df: pd.DataFrame, k: int) -> np.ndarray:
    order_map = {t: i for i, t in enumerate(T_IDS)}
    M = np.zeros((k, k), dtype=np.int64)
    for _, grp in df.groupby("unit_id", sort=False):
        g = grp.sort_values("t_id", key=lambda s: s.map(lambda x: order_map[str(x)]))
        labs = (g["mob_state"].astype(int).to_numpy() - 1).tolist()
        for i in range(len(T_IDS) - 1):
            a, b = labs[i], labs[i + 1]
            if 0 <= a < k and 0 <= b < k:
                M[a, b] += 1
    return M


def plot_cluster_pca(df: pd.DataFrame, labels_0: np.ndarray, state_names: tuple[str, ...], path: Path) -> None:
    Xs = df[MOB_NUM_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(Xs)
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(Xs)
    k = len(state_names)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.5, 6.2))
    cmap = ListedColormap(cm.tab10.colors[: max(k, 3)])
    for j in range(k):
        m = labels_0 == j
        if not m.any():
            continue
        ax0.scatter(
            xy[m, 0],
            xy[m, 1],
            s=5,
            alpha=0.25,
            color=cmap(j),
            label=f"R{j + 1} {state_names[j]}",
            rasterized=True,
        )
    var = pca.explained_variance_ratio_
    ax0.set_xlabel(f"PC1 ({var[0] * 100:.1f}% 方差)")
    ax0.set_ylabel(f"PC2 ({var[1] * 100:.1f}% 方差)")
    ax0.set_title("运行压力 GMM 聚类结果（PCA，unit×时段）")
    ax0.legend(loc="best", fontsize=6, ncol=2, framealpha=0.92)
    counts = np.bincount(labels_0, minlength=k)
    ylabs = [f"R{i + 1} {state_names[i]}" for i in range(k)]
    ax1.barh(ylabs, counts[:k], color=[cmap(i) for i in range(k)], edgecolor="#333", linewidth=0.4)
    ax1.set_xlabel("样本数")
    ax1.set_title("各运行状态样本量")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _mob_state_to_id0_series(df: pd.DataFrame) -> pd.Series:
    """mob_state 转为 0..k-1；支持已从概率列重建的整数标签。"""
    return pd.to_numeric(df["mob_state"], errors="coerce").astype(int) - 1


def _pick_r01_vary_indices(
    ti: int,
    mat: np.ndarray,
    consensus: np.ndarray,
    n_vary: int,
    rng: np.random.Generator,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """
    抽取本时段「展示真实识别状态」的单元索引：优先选 mat[,ti]≠consensus 的地块（视觉上必有差异），
    不足再补齐随机单元。exclude 用于从候选池中剔除（例如避免 WD_AM 与 WE_PM 两块图共用同一批变动地块）。
    """
    n = consensus.shape[0]
    base = np.arange(n, dtype=int)
    if exclude is not None and len(exclude) > 0:
        base = base[~np.isin(base, exclude)]
    if len(base) == 0 or n_vary <= 0:
        return np.array([], dtype=int)
    # 与共识不一致优先（更有信息量）；其次其余单元
    mismatch = base[mat[base, ti] != consensus[base]]
    same = base[mat[base, ti] == consensus[base]]
    mismatch = rng.permutation(mismatch)
    same = rng.permutation(same)
    merged = np.concatenate([mismatch, same])
    if len(merged) < n_vary:
        return merged.astype(int)
    return merged[:n_vary].astype(int)


def build_r01_blended_mob_id0(
    df: pd.DataFrame,
    *,
    stable_frac: float,
    seed: int,
) -> pd.DataFrame:
    """
    R01 专用：约 stable_frac 单元四时段均显示众数共识状态；
    约 (1-stable_frac) 单元在该子图显示该时段真实识别状态。
    抽样优先「该时段≠共识」单元；WD_AM 与 WE_PM 的变动地块互斥，减轻晨型与晚间图过于相似。
    """
    if not (0.0 < stable_frac < 1.0):
        raise ValueError("stable_frac must be in (0, 1)")
    work = df.copy()
    work["mob_id0"] = _mob_state_to_id0_series(work)
    wide = work.pivot(index="unit_id", columns="t_id", values="mob_id0")
    for tid in T_IDS:
        if tid not in wide.columns:
            raise ValueError(f"pivot 缺少时段列 {tid}")
    wide = wide.sort_index()
    unit_ids = list(wide.index)
    mat = wide[list(T_IDS)].to_numpy(dtype=int)
    n = len(unit_ids)
    consensus = np.empty(n, dtype=int)
    for i in range(n):
        row = mat[i]
        vals, cnts = np.unique(row, return_counts=True)
        consensus[i] = int(vals[int(np.argmax(cnts))])

    n_vary = int(round(n * (1.0 - stable_frac)))
    n_vary = max(0, min(n, n_vary))
    # 两时段各抽 30% 且互斥，需 2*n_vary<=n
    if n_vary * 2 > n:
        n_vary = n // 2

    rows: list[dict] = []
    idx_vary_am: np.ndarray | None = None
    for ti, tid in enumerate(T_IDS):
        rng = np.random.default_rng(seed + ti * 1_000_003)
        exclude: np.ndarray | None = None
        # 左上 WD_AM 与 右下 WE_PM：变动集合强制不交叠，增强两图差异
        if ti == 3 and idx_vary_am is not None and len(idx_vary_am) > 0:
            exclude = idx_vary_am
        if n_vary <= 0:
            idx_vary = np.array([], dtype=int)
        else:
            idx_vary = _pick_r01_vary_indices(ti, mat, consensus, n_vary, rng, exclude=exclude)
        if ti == 0:
            idx_vary_am = idx_vary
        mask = np.zeros(n, dtype=bool)
        mask[idx_vary] = True
        disp = consensus.copy()
        disp[mask] = mat[mask, ti]
        for j, uid in enumerate(unit_ids):
            rows.append({"unit_id": uid, "t_id": tid, "mob_id0": int(disp[j])})
    return pd.DataFrame(rows)


def plot_four_maps(
    units: gpd.GeoDataFrame,
    df: pd.DataFrame,
    state_names: tuple[str, ...],
    path: Path,
    site_path: Path | None = None,
    *,
    blend_consensus: bool = True,
    stable_frac: float = 0.7,
    blend_seed: int = 42,
) -> None:
    u = units.copy()
    k = len(state_names)
    if blend_consensus:
        sub = build_r01_blended_mob_id0(df, stable_frac=stable_frac, seed=blend_seed)
    else:
        sub = df.copy()
        sub["mob_id0"] = _mob_state_to_id0_series(sub)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    cmap = ListedColormap(cm.tab10.colors[:k])
    norm = colors.BoundaryNorm(np.arange(-0.5, k + 0.5, 1), cmap.N)
    for ax, tid in zip(axes.ravel(), T_IDS):
        s = sub.loc[sub["t_id"] == tid, ["unit_id", "mob_id0"]]
        mg = u.merge(s, on="unit_id", how="left")
        mg.plot(column="mob_id0", ax=ax, cmap=cmap, norm=norm, linewidth=0.1, edgecolor="k", legend=False)
        plot_site_boundary(ax, u.crs, site_path)
        ax.set_title(tid)
        ax.axis("off")
    labs = [f"R{i + 1} {state_names[i]}" for i in range(k)]
    patches = [Patch(facecolor=cmap(i), edgecolor="#333", linewidth=0.6, label=labs[i]) for i in range(k)]
    leg_handles: list = list(patches)
    leg_labels = list(labs)
    if load_site_gdf(site_path) is not None:
        leg_handles.append(Line2D([0], [0], color="#d90429", lw=2.2, linestyle=(0, (5, 3))))
        leg_labels.append("场地红线 (SITE.json)")
    fig.legend(
        handles=leg_handles,
        labels=leg_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        fontsize=9,
        title="运行压力状态类型",
        title_fontsize=10,
        frameon=True,
    )
    if blend_consensus:
        fig.suptitle(
            "四时段运行压力状态分布图\n"
            "（约 {:.0%} 单元四时段同显共识；约 {:.0%} 优先展示「该时段≠共识」；"
            "WD_AM 与 WE_PM 变动地块互不重叠）".format(stable_frac, 1.0 - stable_frac),
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0.1, 1, 0.91])
    else:
        fig.suptitle("四时段运行压力状态分布图", fontsize=14)
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_transition_matrix(M: np.ndarray, state_names: tuple[str, ...], path: Path) -> None:
    k = M.shape[0]
    row_sum = M.sum(axis=1, keepdims=True)
    P = np.divide(M, row_sum, out=np.zeros_like(M, dtype=float), where=row_sum != 0)
    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    im = ax.imshow(P, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    labs = [f"R{i + 1} {state_names[i]}" for i in range(k)]
    ax.set_xticklabels(labs, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlabel("转入状态")
    ax.set_ylabel("转出状态")
    for i in range(k):
        for j in range(k):
            val = P[i, j]
            tc = "white" if val >= 0.55 else "#1f1f1f"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=tc, fontsize=9)
            if i != j and val > 0.05:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#d62728", linewidth=2.0))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="转移概率（行归一化）")
    ax.set_title("运行状态转移矩阵（行归一化概率）", fontsize=15, pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def transition_matrix_segment(df: pd.DataFrame, k: int, t_from: str, t_to: str) -> np.ndarray:
    """单元层面硬转移计数：t_from → t_to（单跳）。"""
    M = np.zeros((k, k), dtype=np.int64)
    for _, grp in df.groupby("unit_id", sort=False):
        g = grp.set_index("t_id")
        if t_from not in g.index or t_to not in g.index:
            continue
        a = int(pd.to_numeric(g.loc[t_from, "mob_state"], errors="raise")) - 1
        b = int(pd.to_numeric(g.loc[t_to, "mob_state"], errors="raise")) - 1
        if 0 <= a < k and 0 <= b < k:
            M[a, b] += 1
    return M


def plot_transition_segments(Ms: list[np.ndarray], titles: list[str], state_names: tuple[str, ...], path: Path) -> None:
    """三联热力图：相邻时段单跳转移（硬标签），便于与闭环矩阵对照。"""
    k = len(state_names)
    fig, axes = plt.subplots(1, len(Ms), figsize=(6.5 * len(Ms), 7.2))
    if len(Ms) == 1:
        axes = np.array([axes])
    for ax, M, tt in zip(axes.flat, Ms, titles, strict=True):
        rs = M.sum(axis=1, keepdims=True)
        P = np.divide(M, rs, out=np.zeros_like(M, dtype=float), where=rs != 0)
        vmax = max(float(P.max()), 0.08)
        im = ax.imshow(P, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=min(vmax, 1.0))
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        labs = [f"R{i + 1}" for i in range(k)]
        ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labs, fontsize=8)
        ax.set_xlabel("to")
        ax.set_ylabel("from")
        ax.set_title(tt)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("运行状态分段转移（相邻时段单跳；行归一）", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_slot_state_mix(df: pd.DataFrame, k: int, state_names: tuple[str, ...], path: Path) -> None:
    """堆叠面积图：各时段状态占比（硬标签），突出节律。"""
    order_map = {t: i for i, t in enumerate(T_IDS)}
    rows = []
    for t in T_IDS:
        sub = df.loc[df["t_id"] == t, "mob_state"].astype(int) - 1
        cnt = np.bincount(sub.clip(min=0, max=k - 1), minlength=k).astype(float)
        s = cnt.sum()
        if s > 0:
            cnt /= s
        rows.append(cnt)
    Z = np.stack(rows, axis=0)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = [cm.tab10(i % 10) for i in range(k)]
    ax.stackplot(range(len(T_IDS)), Z.T, labels=[f"R{i + 1}" for i in range(k)], colors=colors, alpha=0.88)
    ax.set_xticks(range(len(T_IDS)))
    ax.set_xticklabels(list(T_IDS))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("占比")
    ax.set_title("四时段运行状态构成（硬标签占比）")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, ncol=1)
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radar_from_json(comps: list[dict], state_names: tuple[str, ...], path: Path) -> None:
    keys_full = [
        "road_centrality",
        "accessibility_index",
        "station_attraction",
        "transit_facility_density",
        "barrier_index",
        "bottleneck_index",
        "traffic_intensity",
        "congestion_proxy",
        "flow_in_proxy",
        "flow_out_proxy",
        "stay_proxy",
    ]
    k = len(comps)
    V = np.zeros((k, len(keys_full)))
    for i, c in enumerate(comps):
        mz = c["mean_z"]
        for j, kk in enumerate(keys_full):
            V[i, j] = float(mz.get(kk, 0.0))
    per_idx = []
    for i in range(k):
        row = np.abs(V[i])
        mx = float(np.max(row)) if row.size else 0.0
        thr = max(1e-9, 0.08 * mx) if mx > 1e-9 else 1e-9
        idx = np.where(row >= thr)[0]
        if idx.size < 4:
            idx = np.argsort(-row)[: min(4, len(keys_full))]
            idx = np.sort(idx)
        if idx.size > 18:
            sel = idx[np.argsort(-row[idx])[:18]]
            idx = np.sort(sel)
        per_idx.append(idx)

    cmap = cm.tab10
    fig, axes = plt.subplots(1, k, figsize=(max(4.0 * k, 8.0), 4.8), subplot_kw=dict(polar=True), squeeze=False)
    for i in range(k):
        ax = axes[0, i]
        idx = per_idx[i]
        cols = [keys_full[j] for j in idx]
        Vi = V[i, idx]
        lo = Vi.min()
        hi = Vi.max()
        rng = max(hi - lo, 1e-9)
        Vn = (Vi - lo) / rng
        n_dim = len(cols)
        angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
        angles += angles[:1]
        vals = Vn.tolist() + [Vn[0]]
        ax.plot(angles, vals, color=cmap(i % 10), linewidth=1.6)
        ax.fill(angles, vals, color=cmap(i % 10), alpha=0.12)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace("_", "\n")[:11] for c in cols], fontsize=6)
        ax.set_title(f"R{i + 1}", fontsize=10, pad=12)
    fig.suptitle("运行状态原型雷达（各 R 仅保留非近似零维；组内 min–max）", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _transition_count_seq(seq: tuple[int, ...]) -> int:
    return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])


def _hamming_seq(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _greedy_diverse_patterns(sorted_candidates: list[tuple[int, ...]], n: int) -> list[tuple[int, ...]]:
    """在已排序候选上贪心：先取最优一条，之后每次选与已选集合「最小汉明距离」最大的形态（尽量拉开差异）。"""
    if n <= 0 or not sorted_candidates:
        return []
    picked: list[tuple[int, ...]] = [sorted_candidates[0]]
    pool = list(dict.fromkeys(sorted_candidates[1:]))  # 保序去重
    while len(picked) < n and pool:
        nxt = max(pool, key=lambda p: min(_hamming_seq(p, q) for q in picked))
        picked.append(nxt)
        pool.remove(nxt)
    return picked


def plot_typical_sequences(
    df: pd.DataFrame, state_names: tuple[str, ...], path: Path, n_units: int = 10
) -> None:
    """每种跨时段状态序列形态最多 1 个代表单元；贪心选取彼此差异大的形态；纵轴为真实 R 索引（无垂直偏移）。"""
    order_map = {t: i for i, t in enumerate(T_IDS)}
    k = len(state_names)
    patterns: dict[tuple[int, ...], list[str]] = {}
    for uid, grp in df.groupby("unit_id"):
        g = grp.sort_values("t_id", key=lambda s: s.map(lambda x: order_map[str(x)]))
        states_t = tuple((g["mob_state"].astype(int).to_numpy() - 1).tolist())
        if len(states_t) != len(T_IDS):
            continue
        patterns.setdefault(states_t, []).append(str(uid))

    def pattern_rank_key(p: tuple[int, ...]) -> tuple:
        tc = _transition_count_seq(p)
        return (-tc, -len(set(p)), -max(p) + min(p))

    with_chg = sorted([p for p in patterns if _transition_count_seq(p) > 0], key=pattern_rank_key)
    without_chg = sorted([p for p in patterns if _transition_count_seq(p) == 0], key=pattern_rank_key)
    # 先「有跨时段变化」的形态，再补「全程不变」等；在此顺序上做贪心拉开差异
    combined_candidates = with_chg + without_chg
    chosen_patterns = _greedy_diverse_patterns(combined_candidates, min(n_units, len(combined_candidates)))

    pick: list[tuple[str, list[int]]] = []
    for p in chosen_patterns:
        uid = patterns[p][0]
        pick.append((uid, list(p)))

    markers = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "d", "p")
    linestyles = ("-", "--", "-.", ":")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for i, (uid, states) in enumerate(pick):
        xs = np.arange(len(T_IDS))
        ys = np.array(states, dtype=float)
        ax.plot(
            xs,
            ys,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1.8,
            markersize=6,
            alpha=0.92,
            label=str(uid)[:14],
        )
    ax.set_xticks(range(4))
    ax.set_xticklabels(list(T_IDS))
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"R{i + 1}" for i in range(k)])
    ax.set_ylim(-0.35, k - 1 + 0.35)
    ax.set_ylabel("运行压力状态类别")
    ax.legend(fontsize=7, ncol=3, loc="upper left", framealpha=0.92)
    ax.set_title("典型单元运行状态序列图（多种形态 · 无垂直偏移）")
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_highlight(
    units: gpd.GeoDataFrame,
    df: pd.DataFrame,
    state_idx0: int,
    state_names: tuple[str, ...],
    path: Path,
    site_path: Path | None = None,
) -> None:
    hit = df.loc[df["mob_state"].astype(int) - 1 == state_idx0, "unit_id"].unique()
    u = units.copy()
    u["hit"] = u["unit_id"].isin(hit).astype(int)
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    u[u["hit"] == 0].plot(ax=ax, color="#eaeaea", edgecolor="none", linewidth=0)
    u[u["hit"] == 1].plot(ax=ax, color="#d62728", edgecolor="k", linewidth=0.1)
    plot_site_boundary(ax, u.crs, site_path)
    ax.set_title(f"R{state_idx0 + 1} {state_names[state_idx0]}（任一时段出现即标红）")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_centrality_barrier_maps(units: gpd.GeoDataFrame, df: pd.DataFrame, path: Path, site_path: Path | None = None) -> None:
    g = df.groupby("unit_id", as_index=False)[["road_centrality", "barrier_index"]].mean()
    u = units.merge(g, on="unit_id", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    u.plot(column="road_centrality", ax=axes[0], legend=True, cmap="viridis", linewidth=0.1, edgecolor="k")
    plot_site_boundary(axes[0], u.crs, site_path)
    axes[0].set_title("路网中心性 proxy（四时段均值）")
    axes[0].axis("off")
    u.plot(column="barrier_index", ax=axes[1], legend=True, cmap="magma", linewidth=0.1, edgecolor="k")
    plot_site_boundary(axes[1], u.crs, site_path)
    axes[1].set_title("阻隔指数（四时段均值）")
    axes[1].axis("off")
    fig.suptitle("路网中心性与阻隔指数图", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pick_highlight_indices(state_names: tuple[str, ...]) -> tuple[int, int]:
    """可达低（瓶颈阻隔）类、轨道强类的 0-based 索引。"""
    bn = next((i for i, n in enumerate(state_names) if "可达低" in n), None)
    hub = next((i for i, n in enumerate(state_names) if "轨道强" in n), None)
    k = len(state_names)
    if bn is None:
        bn = min(2, k - 1)
    if hub is None:
        hub = min(3, k - 1)
    return bn, hub


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--site-json",
        type=Path,
        default=None,
        help="场地红线 GeoJSON；省略则优先 data/site_3km/SITE.json，其次 data/SITE.json",
    )
    ap.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--no-r01-blend",
        action="store_true",
        help="R01 四时段图不使用 70%% 共识 + 各时段 30%% 抽样展示（恢复原始逐时段识别结果）",
    )
    ap.add_argument(
        "--r01-stable-frac",
        type=float,
        default=0.7,
        metavar="P",
        help="R01 共识单元占比（默认 0.7）；其余各时段独立抽样展示该时段真实状态",
    )
    ap.add_argument("--r01-seed", type=int, default=42, help="R01 各时段抽样随机种子")
    ap.add_argument(
        "--r04-n-units",
        type=int,
        default=10,
        metavar="N",
        help="R04 展示的互异运行状态序列条数（默认 10；贪心拉开形态差异）",
    )
    ns = ap.parse_args()
    out_dir = ns.out_dir
    units_path = ns.units
    site_json = ns.site_json if ns.site_json is not None and ns.site_json.is_file() else resolve_site_json_path()

    mob_csv = out_dir / "mob_state.csv"
    labels_json = out_dir / "mob_state_labels.json"

    if not mob_csv.is_file():
        raise SystemExit(f"缺少 {mob_csv}")
    if not units_path.is_file():
        raise SystemExit(f"缺少 {units_path}")
    if not labels_json.is_file():
        raise SystemExit(f"缺少 {labels_json}")

    k, comps = load_components_labels(labels_json)
    state_names = build_state_names(comps)

    for c in comps:
        cid = int(c["id"])
        c["zh_name"] = state_names[cid - 1]
    data_full = json.loads(labels_json.read_text(encoding="utf-8"))
    data_full["components"] = comps
    data_full["state_names_ordered"] = list(state_names)
    labels_json.write_text(json.dumps(data_full, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.read_csv(mob_csv, encoding="utf-8-sig")
    prob_cols = [f"p_R{i + 1}" for i in range(k)]
    if all(c in df.columns for c in prob_cols):
        id0 = df[prob_cols].to_numpy().argmax(axis=1)
        df = df.copy()
        df["mob_state"] = (id0 + 1).astype(int)
    else:
        df["mob_state"] = pd.to_numeric(df["mob_state"], errors="raise")
        id0 = df["mob_state"].to_numpy() - 1

    try:
        units = gpd.read_file(units_path, layer="units")
    except Exception:
        units = gpd.read_file(units_path)

    fig_r00 = out_dir / "R00_运行压力聚类结果呈现图.png"
    fig_r01 = out_dir / "R01_四时段运行压力状态分布图.png"
    fig_r02 = out_dir / "R02_运行状态转移矩阵.png"
    fig_r03 = out_dir / "R03_运行状态原型雷达图.png"
    fig_r04 = out_dir / "R04_典型单元运行状态序列图.png"
    fig_r05 = out_dir / "R05_可达低型识别图.png"
    fig_r06 = out_dir / "R06_轨道强型识别图.png"
    fig_r07 = out_dir / "R07_路网中心性与阻隔指数图.png"

    print("R00 PCA …")
    plot_cluster_pca(df, id0, state_names, fig_r00)
    print("R01 maps …")
    plot_four_maps(
        units,
        df,
        state_names,
        fig_r01,
        site_path=site_json,
        blend_consensus=not ns.no_r01_blend,
        stable_frac=float(ns.r01_stable_frac),
        blend_seed=int(ns.r01_seed),
    )
    M = transition_matrix_from_csv(df, k)
    pd.DataFrame(M, index=[f"R{i+1}" for i in range(k)], columns=[f"R{j+1}" for j in range(k)]).to_csv(
        out_dir / "transition_matrix.csv", encoding="utf-8-sig"
    )
    print("R02 transition …")
    plot_transition_matrix(M, state_names, fig_r02)
    seg_pairs = [("WD_AM", "WD_PM"), ("WD_PM", "WD_EVE"), ("WD_EVE", "WE_PM")]
    Ms = [transition_matrix_segment(df, k, a, b) for a, b in seg_pairs]
    titles = [f"{a}→{b}" for a, b in seg_pairs]
    plot_transition_segments(Ms, titles, state_names, out_dir / "R02b_分段转移热力组图.png")
    plot_slot_state_mix(df, k, state_names, out_dir / "R02c_四时段状态构成堆叠图.png")
    print("R03 radar …")
    plot_radar_from_json(comps, state_names, fig_r03)
    print("R04 sequences …")
    plot_typical_sequences(df, state_names, fig_r04, n_units=max(1, int(ns.r04_n_units)))

    idx_bn, idx_hub = pick_highlight_indices(state_names)
    print("R05–R07 highlights / indices …")
    plot_highlight(units, df, idx_bn, state_names, fig_r05, site_path=site_json)
    plot_highlight(units, df, idx_hub, state_names, fig_r06, site_path=site_json)
    plot_centrality_barrier_maps(units, df, fig_r07, site_path=site_json)

    name_by_int = {i + 1: state_names[i] for i in range(k)}
    df_out = df.copy()
    df_out["mob_state"] = df_out["mob_state"].map(lambda x: name_by_int[int(x)])
    df_out.to_csv(out_dir / "mob_state.csv", index=False, encoding="utf-8-sig")

    print("Done. Wrote figures R00–R07, R02b–R02c, updated mob_state.csv / mob_state_labels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
