"""
基于已有 morph_state.csv + morph_gmm_meta.json 重绘「第一人」形态层图纸（与既有命名一致）。

默认输出：
  output/form/figures/M01_morph_state_map.png
  output/form/figures/M02_morph_prototype_radar.png
  output/form/figures/M03_barrier_permeability.png
  output/form/figures/M04_morph_interpretation.png
并同步写入 output/form/output_morph/figures/（若目录存在或可创建）。

用法（仓库根目录）：
  python scripts/plot_morph_state_pack.py
  python scripts/plot_morph_state_pack.py --morph path/to/morph_state.csv --out-dir path/to/figures
"""

from __future__ import annotations

import argparse
import json
import re
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
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MORPH_CSV = REPO / "output/form/morph_state.csv"
DEFAULT_META = REPO / "output/form/morph_gmm_meta.json"
DEFAULT_UNITS = REPO / "output/function/数据包/01_units.gpkg"

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from site_map_overlay import load_site_gdf, plot_site_boundary, resolve_site_json_path  # noqa: E402

_M_RE = re.compile(r"^M\s*(\d+)")

# 形态层数值字段 → M04 解释图横轴中文（与 morph_gmm_meta feature_names 对齐）
FEATURE_LABEL_ZH: dict[str, str] = {
    "building_coverage": "建筑覆盖率",
    "building_density": "建筑密度",
    "avg_height": "平均建筑高度",
    "max_height": "最大建筑高度",
    "volume_intensity": "体量强度",
    "block_compactness": "街坊紧凑度",
    "building_fragmentation": "建筑碎片度",
    "road_density": "道路密度",
    "intersection_density": "交叉口密度",
    "green_blue_ratio": "蓝绿空间占比",
    "dem_mean": "地形高程均值",
    "dem_slope": "地形坡度",
    "heritage_ratio": "风貌保护区占比",
    "landuse_mix": "土地利用混合度",
    "edge_conductance_mean": "边界传导均值",
    "edge_conductance_std": "边界传导标准差",
    "barrier_index": "阻隔指数",
    "permeability_index": "渗透性指数",
}


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


def prob_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.match(r"^p_M\d+$", str(c))]


def morph_id0_from_row(row: pd.Series, p_cols: list[str]) -> int:
    s = str(row.get("morph_state", "")).strip()
    m = _M_RE.match(s)
    if m:
        return int(m.group(1)) - 1
    if p_cols:
        vals = [float(row[c]) for c in sorted(p_cols, key=lambda x: int(x.replace("p_M", "")))]
        return int(np.argmax(vals))
    return 0


def load_state_names(df: pd.DataFrame, k: int) -> tuple[str, ...]:
    """仅用于内部占位；图例统一用数字编码（见 plot_m01_map 的 labs）。"""
    return tuple(str(j) for j in range(k))


def _radar_active_dims(
    V: np.ndarray,
    state_row: int,
    *,
    eps_rel: float = 0.08,
    abs_floor: float = 1e-12,
    min_axes: int = 4,
    max_axes: int = 18,
) -> np.ndarray:
    """
    选取「该状态在维度上有分量」的轴：去掉近似全 0 的维度；不足 min_axes 时按 |值| 取 top。
    V: (k, F) 已为该类内特征均值（原始量纲，非极坐标归一化前）。
    """
    row = np.abs(V[state_row].astype(float))
    mx = float(np.nanmax(row)) if row.size else 0.0
    thr = max(abs_floor, eps_rel * mx) if mx > abs_floor else abs_floor
    mask = row >= thr
    idx = np.where(mask)[0]
    if idx.size < min_axes:
        order = np.argsort(-row)
        idx = np.sort(order[: min(min_axes, len(row))])
    if idx.size > max_axes:
        sub = np.argsort(-row[idx])[:max_axes]
        idx = np.sort(idx[sub])
    return idx


def plot_m01_map(
    units: gpd.GeoDataFrame,
    df: pd.DataFrame,
    state_names: tuple[str, ...],
    path: Path,
    site_path: Path | None,
) -> None:
    u = units.copy()
    sub = df[["unit_id", "_morph_id0"]].drop_duplicates("unit_id")
    k = len(state_names)
    mg = u.merge(sub, on="unit_id", how="left")
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    cmap = ListedColormap(cm.tab10.colors[: max(k, 3)])
    norm = colors.BoundaryNorm(np.arange(-0.5, k + 0.5, 1), cmap.N)
    mg.plot(column="_morph_id0", ax=ax, cmap=cmap, norm=norm, linewidth=0.1, edgecolor="k", legend=False)
    plot_site_boundary(ax, u.crs, site_path)
    ax.set_title("空间可供性状态地图（morph_state）")
    ax.axis("off")
    labs = [str(i) for i in range(k)]
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
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=8,
        title="形态状态编号",
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_m02_radar(df: pd.DataFrame, feature_cols: list[str], state_names: tuple[str, ...], path: Path) -> None:
    k = len(state_names)
    V = np.zeros((k, len(feature_cols)))
    for j in range(k):
        sub = df.loc[df["_morph_id0"] == j, feature_cols]
        if len(sub):
            V[j] = sub.mean(axis=0).to_numpy(dtype=float)
        else:
            V[j] = 0.0
    cmap = cm.tab10
    fig, axes = plt.subplots(
        1,
        k,
        figsize=(max(4.2 * k, 8.0), 4.6),
        subplot_kw=dict(polar=True),
        squeeze=False,
    )
    for i in range(k):
        ax = axes[0, i]
        idx = _radar_active_dims(V, i)
        cols_i = [feature_cols[j] for j in idx]
        Vi = V[i, idx]
        lo = Vi.min()
        hi = Vi.max()
        rng = max(hi - lo, 1e-9)
        Vn = (Vi - lo) / rng
        n_dim = len(cols_i)
        angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
        angles += angles[:1]
        vals = Vn.tolist() + [Vn[0]]
        ax.plot(angles, vals, color=cmap(i % 10), linewidth=1.6)
        ax.fill(angles, vals, color=cmap(i % 10), alpha=0.12)
        short_labs = [c.replace("_", "\n")[:12] for c in cols_i]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_labs, fontsize=6)
        ax.set_title(f"形态类 {i}", fontsize=9, pad=12)
    fig.suptitle("空间可供性状态原型雷达图（每类仅展示非近似零维；组内 min–max）", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_m03_barrier_perm(units: gpd.GeoDataFrame, df: pd.DataFrame, path: Path, site_path: Path | None) -> None:
    sub = df[["unit_id", "barrier_index", "permeability_index"]].drop_duplicates("unit_id")
    u = units.merge(sub, on="unit_id", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5))
    u.plot(column="barrier_index", ax=axes[0], legend=True, cmap="magma", linewidth=0.1, edgecolor="k")
    plot_site_boundary(axes[0], u.crs, site_path)
    axes[0].set_title("阻隔指数")
    axes[0].axis("off")
    u.plot(column="permeability_index", ax=axes[1], legend=True, cmap="viridis", linewidth=0.1, edgecolor="k")
    plot_site_boundary(axes[1], u.crs, site_path)
    axes[1].set_title("渗透性指数")
    axes[1].axis("off")
    fig.suptitle("阻隔指数与渗透性指数", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_m04_heatmap(df: pd.DataFrame, feature_cols: list[str], state_names: tuple[str, ...], path: Path) -> None:
    k = len(state_names)
    M = np.zeros((k, len(feature_cols)))
    for j in range(k):
        sub = df.loc[df["_morph_id0"] == j, feature_cols]
        M[j] = sub.mean(axis=0).to_numpy(dtype=float) if len(sub) else 0.0
    col_mean = M.mean(axis=0)
    col_std = M.std(axis=0).clip(min=1e-9)
    Z = (M - col_mean) / col_std
    fig, ax = plt.subplots(figsize=(max(10.0, len(feature_cols) * 0.55), max(6.0, k * 0.65)))
    im = ax.imshow(Z, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_yticks(range(k))
    ax.set_yticklabels([str(i) for i in range(k)], fontsize=9)
    xlabs = [FEATURE_LABEL_ZH.get(c, c.replace("_", "\n")) for c in feature_cols]
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(xlabs, rotation=45, ha="right", fontsize=8)
    ax.set_title("空间可供性状态解释图（各类别特征均值相对全体类别的 z-score）")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label="z-score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def default_output_dirs(extra: Path | None) -> list[Path]:
    primary = REPO / "output/form/figures"
    alt = REPO / "output/form/output_morph/figures"
    out = [primary]
    if extra is not None:
        out.append(extra)
    else:
        out.append(alt)
    # de-dupe
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description="重绘形态层 M01–M04")
    ap.add_argument("--morph", type=Path, default=DEFAULT_MORPH_CSV)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    ap.add_argument(
        "--site-json",
        type=Path,
        default=None,
        help="场地红线；默认优先 data/site_3km/SITE.json，其次 data/SITE.json",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="仅写此目录（不设则写 figures + output_morph/figures）")
    ns = ap.parse_args()

    morph_path = ns.morph
    if not morph_path.is_file():
        alt = REPO / "data/morph_state.csv"
        if alt.is_file():
            morph_path = alt
        else:
            raise SystemExit(f"未找到 morph_state.csv：{ns.morph}")

    df = pd.read_csv(morph_path)
    _pc = prob_cols(df)
    df["_morph_id0"] = df.apply(lambda r: morph_id0_from_row(r, _pc), axis=1)
    k = int(df["_morph_id0"].max()) + 1
    state_names = load_state_names(df, k)

    num_exclude = {"unit_id", "morph_state", "_morph_id0", *_pc}
    feature_cols = [
        c for c in df.columns if c not in num_exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise SystemExit("morph_state.csv 中无可用的数值特征列用于雷达/解释图")

    if ns.meta.is_file():
        meta = json.loads(ns.meta.read_text(encoding="utf-8"))
        preferred = meta.get("feature_names") or []
        feature_cols = [c for c in preferred if c in df.columns] or feature_cols

    try:
        units = gpd.read_file(ns.units, layer="units")
    except Exception:
        units = gpd.read_file(ns.units)

    out_dirs = [ns.out_dir] if ns.out_dir is not None else default_output_dirs(None)
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    if ns.site_json is not None and ns.site_json.is_file():
        site = ns.site_json
    else:
        site = resolve_site_json_path()

    names = ("M01_morph_state_map.png", "M02_morph_prototype_radar.png", "M03_barrier_permeability.png", "M04_morph_interpretation.png")
    for d in out_dirs:
        print(f"Writing to {d} …")
        plot_m01_map(units, df, state_names, d / names[0], site)
        plot_m02_radar(df, feature_cols, state_names, d / names[1])
        plot_m03_barrier_perm(units, df, d / names[2], site)
        plot_m04_heatmap(df, feature_cols, state_names, d / names[3])

    print("Done. M01–M04 written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
