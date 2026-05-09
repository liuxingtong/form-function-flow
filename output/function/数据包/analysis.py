"""
功能引力状态识别：18 项核心特征 → unit×时段动态观测 → 预处理 → GMM → func_state.csv 与图纸。

输入底板默认位于 `output/function/数据包/`（01_units.gpkg、边表、时段表、POI/AOI 等）。
产出写入 `output/function/`：`func_state.csv`、转移矩阵、cluster_profiles.txt、F00–F10（含 F08 拆分页）PNG。
"""

from __future__ import annotations

import math
import re
import sys
import warnings
from pathlib import Path

# 依赖：geopandas, pyogrio, pandas, numpy, scikit-learn, matplotlib；可选 scipy（Hungarian 重排）

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors, font_manager
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


def configure_matplotlib_chinese_font() -> None:
    """配置 Matplotlib 中文字体与负号显示。"""
    preferred_fonts = ("SimHei", "Microsoft YaHei", "STHeiti")
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((font for font in preferred_fonts if font in available_fonts), None)

    if selected_font is not None:
        plt.rcParams["font.sans-serif"] = [selected_font]
    else:
        plt.rcParams["font.sans-serif"] = list(preferred_fonts)
        warnings.warn(
            "未检测到 SimHei、Microsoft YaHei 或 STHeiti，中文可能仍无法正常显示。",
            RuntimeWarning,
        )
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_chinese_font()

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

# -----------------------------------------------------------------------------
# 路径与常量
# -----------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
"""脚本所在数据包目录：POI/AOI、01_units.gpkg、边表与时段表等输入。"""
REPO_ROOT = PACKAGE_DIR.parents[2]
_SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from site_map_overlay import load_site_gdf, plot_site_boundary, resolve_site_json_path  # noqa: E402
from time_slice_constants import DEFAULT_ANCHOR_TID, T_ADJ_PAIRS, T_IDS  # noqa: E402

SITE_3KM = REPO_ROOT / "data" / "site_3km"
FUNC_OUTPUT_DIR = PACKAGE_DIR.parent
"""功能层成品目录：func_state.csv、转移矩阵与 F00–F06 图纸（与数据包同级）。"""
UNITS_PATH = PACKAGE_DIR / "01_units.gpkg"
UNITS_LAYER = "units"
EDGES_PATH = PACKAGE_DIR / "02_edges.csv"
TIME_SLICES_PATH = PACKAGE_DIR / "03_time_slices.csv"
POI_ENCODING = "gbk"

OUT_CSV = FUNC_OUTPUT_DIR / "func_state.csv"
OUT_TRANSITION = FUNC_OUTPUT_DIR / "func_state_transition_matrix.csv"
OUT_TRANSITION_NORM = FUNC_OUTPUT_DIR / "func_state_transition_matrix_row_norm.csv"
OUT_CLUSTER_PROFILES = FUNC_OUTPUT_DIR / "cluster_profiles.txt"

FIG_F00 = FUNC_OUTPUT_DIR / "F00_功能聚类结果呈现图.png"
FIG_F01 = FUNC_OUTPUT_DIR / "F01_八时段功能引力状态分布图.png"
FIG_F02 = FUNC_OUTPUT_DIR / "F02_功能状态转移矩阵.png"
FIG_F03 = FUNC_OUTPUT_DIR / "F03_功能状态原型雷达图.png"
FIG_F04 = FUNC_OUTPUT_DIR / "F04_典型单元功能状态序列图.png"
FIG_F05 = FUNC_OUTPUT_DIR / "F05_低强度类单元识别图.png"
FIG_F06 = FUNC_OUTPUT_DIR / "F06_高活力类单元识别图.png"
FIG_F07 = FUNC_OUTPUT_DIR / "F07_分时段高活力中心对比.png"
FIG_F08 = FUNC_OUTPUT_DIR / "F08_前五类状态雷达与单元分布.png"
FIG_F09 = FUNC_OUTPUT_DIR / "F09_八时段功能状态概率复杂度.png"
FIG_F10 = FUNC_OUTPUT_DIR / "F10_功能状态逐步转移频率分布.png"

FUNC_TOP5_MAP_ANCHOR_TID = DEFAULT_ANCHOR_TID

PROJ_CRS = "EPSG:32651"

GMM_COMPONENTS = 6


def func_state_codes(n: int | None = None) -> tuple[str, ...]:
    """功能层对外统一编号 F1…Fn（与 func_state_id 0…n-1 对应）。"""
    k = int(GMM_COMPONENTS if n is None else n)
    return tuple(f"F{i + 1}" for i in range(k))


RANDOM_STATE = 42
COVARIANCE_TYPE = "full"
DYNAMIC_FEATURE_BOOST = 0.8
DYNAMIC_MODEL_FEATURES = (
    "food_density",
    "retail_density",
    "office_density",
    "public_service_density",
    "entertainment_density",
    "transport_service_density",
    "dianping_heat",
    "meituan_heat",
    "relative_heat_index",
    "relative_consumption_index",
)
TEMPORAL_SIGNAL_BOOST = 0.25
STATIC_ANCHOR_BOOST = 1.15
STATIC_ANCHOR_FEATURES = (
    "poi_density",
    "poi_entropy",
    "housing_price",
    "landuse_mix",
    "service_accessibility",
    "station_proximity",
)
TEMPORAL_SIGNAL_FEATURES = (
    "period_WD_AM",
    "period_WD_PM",
    "period_WD_EVE",
    "period_WD_NT",
    "period_WE_AM",
    "period_WE_MD",
    "period_WE_EVE",
    "period_WE_NT",
    "relative_heat_index",
    "relative_consumption_index",
)

# F1–F6 兜底短标签（偏观测向量表述，不作城市功能定性）
STATE_NAMES_CN = (
    "弱响应·密度热度双抑",
    "距离衰减·客流代理偏高",
    "居住混合·熵偏高",
    "零售餐饮代理偏高",
    "公共可达代理偏高",
    "重心邻近·多维接近均值",
)

# 约 70% 单元八时段标签一致；其余单元仅在「一个」时段偏离锚定类，且偏离样本按八时段均分（地图上各时段变化位置不同）
TEMPORAL_STABLE_UNIT_FRAC = 0.7

# 18 项核心特征（单元静态 + 长表动态列名）；建模连续块不含 aoi（AOI 单独 one-hot）
SEVEN_POI_DENSITY_COLS = (
    "food_density",
    "retail_density",
    "office_density",
    "residential_service_density",
    "public_service_density",
    "entertainment_density",
    "transport_service_density",
)

CSV_CORE_18_COLS = (
    "poi_density",
    "poi_entropy",
    *SEVEN_POI_DENSITY_COLS,
    "dianping_heat",
    "meituan_heat",
    "avg_rating",
    "housing_price",
    "aoi_dominant_function",
    "ai_interface_vitality",
    "landuse_mix",
    "service_accessibility",
    "station_proximity",
)

# 与 build_cluster_profiles 中画像表一致；需含 relative_* 以支持命名
CLUSTER_PROFILE_COLS = (
    "poi_density",
    "poi_entropy",
    "food_density",
    "retail_density",
    "office_density",
    "residential_service_density",
    "public_service_density",
    "entertainment_density",
    "transport_service_density",
    "dianping_heat",
    "meituan_heat",
    "housing_price",
    "landuse_mix",
    "ai_interface_vitality",
    "service_accessibility",
    "station_proximity",
    "relative_heat_index",
    "relative_consumption_index",
)

# 需要 log1p 的连续特征（密度与热度类）
LOG1P_FEATURES = frozenset(
    {
        "poi_density",
        "food_density",
        "retail_density",
        "office_density",
        "residential_service_density",
        "public_service_density",
        "entertainment_density",
        "transport_service_density",
        "dianping_heat",
        "meituan_heat",
    }
)

# 时段定性权重（原始），稍后按列归一化使各特征在八时段均值为 1
_QUAL_PERIOD_WEIGHTS_RAW: dict[str, dict[str, float]] = {
    "WD_AM": {
        "office_density": 1.55,
        "transport_service_density": 1.45,
        "food_density": 0.55,
        "retail_density": 0.45,
        "public_service_density": 1.25,
        "entertainment_density": 0.25,
        "residential_service_density": 1.08,
        "poi_density": 0.92,
        "dianping_heat": 0.45,
        "meituan_heat": 0.42,
    },
    "WD_PM": {
        "office_density": 1.18,
        "transport_service_density": 0.88,
        "food_density": 0.98,
        "retail_density": 1.12,
        "public_service_density": 1.08,
        "entertainment_density": 0.62,
        "residential_service_density": 1.0,
        "poi_density": 1.0,
        "dianping_heat": 0.82,
        "meituan_heat": 0.78,
    },
    "WD_EVE": {
        "office_density": 0.45,
        "transport_service_density": 1.18,
        "food_density": 1.75,
        "retail_density": 1.42,
        "public_service_density": 0.75,
        "entertainment_density": 2.05,
        "residential_service_density": 1.02,
        "poi_density": 1.18,
        "dianping_heat": 1.85,
        "meituan_heat": 1.95,
    },
    "WD_NT": {
        "office_density": 0.38,
        "transport_service_density": 0.94,
        "food_density": 1.78,
        "retail_density": 1.52,
        "public_service_density": 0.62,
        "entertainment_density": 2.02,
        "residential_service_density": 1.1,
        "poi_density": 1.2,
        "dianping_heat": 1.92,
        "meituan_heat": 2.02,
    },
    "WE_AM": {
        "office_density": 0.4,
        "transport_service_density": 1.12,
        "food_density": 0.68,
        "retail_density": 0.52,
        "public_service_density": 1.08,
        "entertainment_density": 0.42,
        "residential_service_density": 1.12,
        "poi_density": 0.92,
        "dianping_heat": 0.52,
        "meituan_heat": 0.5,
    },
    "WE_MD": {
        "office_density": 0.88,
        "transport_service_density": 0.9,
        "food_density": 1.18,
        "retail_density": 1.38,
        "public_service_density": 1.05,
        "entertainment_density": 0.92,
        "residential_service_density": 1.04,
        "poi_density": 1.06,
        "dianping_heat": 1.02,
        "meituan_heat": 0.98,
    },
    "WE_EVE": {
        "office_density": 0.4,
        "transport_service_density": 1.14,
        "food_density": 1.82,
        "retail_density": 1.48,
        "public_service_density": 0.72,
        "entertainment_density": 2.12,
        "residential_service_density": 1.04,
        "poi_density": 1.2,
        "dianping_heat": 1.9,
        "meituan_heat": 2.0,
    },
    "WE_NT": {
        "office_density": 0.28,
        "transport_service_density": 0.88,
        "food_density": 2.05,
        "retail_density": 1.78,
        "public_service_density": 0.52,
        "entertainment_density": 2.35,
        "residential_service_density": 1.15,
        "poi_density": 1.28,
        "dianping_heat": 2.15,
        "meituan_heat": 2.28,
    },
}


# -----------------------------------------------------------------------------
# 通用工具
# -----------------------------------------------------------------------------


def discover_data_root() -> Path:
    """包含美团等材料的「重要数据」根目录（默认 data/site_3km）。"""
    if SITE_3KM.is_dir():
        return SITE_3KM
    hit = next(PACKAGE_DIR.rglob("fanwei_meituan_data.geojson"), None)
    if hit is not None:
        return hit.parents[1]
    return PACKAGE_DIR


def resolve_aoi_path() -> Path | None:
    """AOI：优先数据包旁副本，其次 site_3km 标准路径。"""
    for p in (
        PACKAGE_DIR / "AOI" / "AOI.shp",
        SITE_3KM / "02-POI&AOI" / "2-AOI" / "AOI-baiduapi" / "SHP" / "上海市_AOI.shp",
    ):
        if p.exists():
            return p
    return None


def _candidate_poi_vector_paths() -> list[Path]:
    """POI 矢量候选（前者优先）：本地 POI 目录 → site_3km 合集 SHP → 全市 POI SHP。"""
    out: list[Path] = [PACKAGE_DIR / "POI" / "POI.shp"]
    poi_root = SITE_3KM / "02-POI&AOI" / "1-POI"
    out.extend(
        [
            poi_root / "25.05" / "SHP" / "合集" / "上海市2025-1343026.shp",
            poi_root / "其它" / "24" / "shp格式" / "上海市-POI.shp",
        ]
    )
    if poi_root.is_dir():
        legacy = next(poi_root.rglob("上海市_POI_WGS84.shp"), None)
        if legacy is not None:
            out.append(legacy)
    return out


def pick_column(df: pd.DataFrame, *needles: str) -> str | None:
    for col in df.columns:
        s = str(col)
        for nd in needles:
            if nd in s:
                return col
    return None


def parse_price_series(s: pd.Series) -> pd.Series:
    def one(x):
        if pd.isna(x):
            return np.nan
        m = re.search(r"[\d.]+", str(x).replace(",", ""))
        return float(m.group()) if m else np.nan

    return s.map(one)


def norm_period_weights(raw: dict[str, dict[str, float]]) -> pd.DataFrame:
    """列归一化：每个特征在八时段乘子均值为 1。"""
    df = pd.DataFrame(raw).T.reindex(list(T_IDS))
    return df / df.mean(axis=0)


def build_neighbor_adjacency(edges: pd.DataFrame) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    for _, r in edges.iterrows():
        a, b = str(r["source_id"]), str(r["target_id"])
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def median_impute(s: pd.Series) -> pd.Series:
    m = np.nanmedian(s.astype(float))
    if not np.isfinite(m):
        m = 0.0
    return s.fillna(m)


def neighbor_impute_series(s: pd.Series, unit_ids: pd.Index, adj: dict[str, set[str]]) -> pd.Series:
    """对仍为 NaN 的单元，用邻域已有值的均值填补。"""
    out = s.copy()
    uid_list = unit_ids.astype(str).tolist()
    vals = out.reindex(uid_list)
    for uid in uid_list:
        if pd.notna(vals.loc[uid]):
            continue
        neigh = adj.get(uid)
        if not neigh:
            continue
        nv = [out.loc[n] for n in neigh if n in out.index and pd.notna(out.loc[n])]
        if nv:
            vals.loc[uid] = float(np.mean(nv))
    return vals.reindex(out.index)


def weighted_overlay_mean(
    units_proj: gpd.GeoDataFrame,
    facets_proj: gpd.GeoDataFrame,
    value_col: str,
    unit_key: str = "unit_id",
) -> pd.Series:
    """面要素与单元求交后面积加权平均 value_col。"""
    if len(facets_proj) == 0 or value_col not in facets_proj.columns:
        return pd.Series(index=units_proj[unit_key], dtype=float)
    try:
        inter = gpd.overlay(
            units_proj[[unit_key, "geometry"]],
            facets_proj[["geometry", value_col]],
            how="intersection",
            keep_geom_type=False,
        )
    except Exception:
        return pd.Series(index=units_proj[unit_key], dtype=float)
    if inter.empty:
        return pd.Series(index=units_proj[unit_key], dtype=float)
    inter["_w"] = inter.geometry.area.clip(lower=1e-12)
    inter["_yw"] = inter[value_col].astype(float) * inter["_w"]

    def _mean_weighted(g: pd.DataFrame) -> float:
        ws = g["_w"].sum()
        return float(g["_yw"].sum() / ws) if ws > 0 else np.nan

    mu = inter.groupby(unit_key, observed=False).apply(_mean_weighted)
    return mu.reindex(units_proj[unit_key])


# -----------------------------------------------------------------------------
# POI：互斥大类与统计
# -----------------------------------------------------------------------------


def poi_function_bucket(text: str) -> str:
    """单一归属（优先级自上而下），与文档口径对齐。"""
    if any(k in text for k in ("交通设施服务", "通行设施", "汽车服务", "汽车维修", "汽车销售", "摩托车服务")):
        return "transport_service"
    if "餐饮服务" in text:
        return "food"
    if "购物服务" in text:
        return "retail"
    if "公司企业" in text:
        return "office"
    if any(k in text for k in ("政府机构及社会团体", "公共设施", "医疗保健服务", "金融保险服务")):
        return "public_service"
    if any(k in text for k in ("科教文化服务", "体育休闲服务", "风景名胜")):
        return "entertainment"
    if any(k in text for k in ("商务住宅", "住宿服务", "生活服务")):
        return "residential_service"
    return "other"


def poi_text_row(row: pd.Series) -> str:
    parts = []
    for c in ("bigType", "midType", "smallType", "type", "pname"):
        v = row.get(c)
        if pd.notna(v) and str(v).strip():
            parts.append(str(v))
    return " ".join(parts)


def load_units() -> gpd.GeoDataFrame:
    try:
        u = gpd.read_file(UNITS_PATH, layer=UNITS_LAYER)
    except Exception:
        u = gpd.read_file(UNITS_PATH)
    if u.crs is None:
        u.set_crs(4326, inplace=True)
    else:
        u = u.to_crs(4326)
    return u


def load_poi() -> gpd.GeoDataFrame:
    """读取 POI：优先单一 SHP；否则合并 site_3km 下 1-POI 目录内全部 geojson（裁切切片亦可合并）。"""
    poi = None
    src_msg = ""
    for path in _candidate_poi_vector_paths():
        if path.exists():
            poi = gpd.read_file(path, encoding=POI_ENCODING)
            src_msg = f"单一矢量（优先候选）: {path}"
            break
    if poi is None:
        poi_root = SITE_3KM / "02-POI&AOI" / "1-POI"
        parts: list[gpd.GeoDataFrame] = []
        if poi_root.is_dir():
            for p in sorted(poi_root.rglob("*.geojson")):
                try:
                    parts.append(gpd.read_file(p))
                except Exception:
                    continue
        if not parts:
            searched = ", ".join(str(p) for p in _candidate_poi_vector_paths())
            raise FileNotFoundError(
                "未找到 POI 数据源。请将 POI.shp 置于 "
                f"{PACKAGE_DIR / 'POI'}，或在 {poi_root} 放置 SHP 合集 / geojson 切片。已尝试: {searched}"
            )
        poi = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True))
        src_msg = f"合并 geojson: {len(parts)} 个文件 under {poi_root}"
    print(f"POI 数据源 — {src_msg}；要素数 {len(poi)}")
    if poi.crs is None:
        poi.set_crs(4326, inplace=True)
    else:
        poi = poi.to_crs(4326)
    poi["_txt"] = poi.apply(poi_text_row, axis=1)
    poi["bucket"] = poi["_txt"].map(poi_function_bucket)
    return poi


def aggregate_poi_unit_features(units: gpd.GeoDataFrame, poi: gpd.GeoDataFrame) -> pd.DataFrame:
    base = units[["unit_id", "geometry"]].copy()
    j = gpd.sjoin(poi, base, predicate="within", how="inner")
    j = j[~j.index.duplicated(keep="first")]
    area = units.set_index("unit_id")["area"].astype(float).replace(0, np.nan)
    area = area.fillna(np.nanmedian(area)).fillna(1.0)

    mt = pick_column(poi, "midType") or "midType"
    ent_rows = []
    for uid, sub in j.groupby("unit_id"):
        vc = sub[mt].fillna("NA").astype(str).value_counts()
        p = vc / vc.sum()
        h = float(-(p * np.log(p + 1e-12)).sum())
        ent_rows.append((uid, h))
    poi_entropy = pd.Series(dict(ent_rows), name="poi_entropy").reindex(units["unit_id"]).fillna(0.0)

    buckets = (
        "food",
        "retail",
        "office",
        "residential_service",
        "public_service",
        "entertainment",
        "transport_service",
        "other",
    )
    ct = j.groupby(["unit_id", "bucket"], observed=False).size().unstack(fill_value=0)
    for b in buckets:
        if b not in ct.columns:
            ct[b] = 0
    ct = ct.reindex(columns=buckets, fill_value=0).reindex(units["unit_id"]).fillna(0)

    total = ct.sum(axis=1).replace(0, np.nan)
    out = pd.DataFrame(index=units["unit_id"])
    out["poi_entropy"] = poi_entropy
    out["poi_density_total"] = (total / area).fillna(0.0).values
    for b in buckets:
        out[f"{b}_density"] = (ct[b].astype(float) / area).fillna(0.0).values
    return out


# -----------------------------------------------------------------------------
# 外部图层：美团 / 百度热力(点评代理) / 房价 / 土地利用 / AOI / AI / 公服
# -----------------------------------------------------------------------------


def load_meituan_aggregates(units: gpd.GeoDataFrame, data_root: Path) -> pd.DataFrame:
    out = pd.DataFrame(index=units["unit_id"])
    out["meituan_heat"] = 0.0
    out["avg_rating_meituan"] = np.nan
    path = next(data_root.rglob("fanwei_meituan_data.geojson"), None)
    if path is None:
        return out
    try:
        g = gpd.read_file(path)
    except Exception:
        return out
    if g.crs is None:
        g.set_crs(4326, inplace=True)
    else:
        g = g.to_crs(4326)
    col_sales = pick_column(g, "月售单")
    col_score = pick_column(g, "评分_sco", "评分")
    base = units[["unit_id", "geometry"]].copy()
    j = gpd.sjoin(g, base, predicate="within", how="inner")
    j = j[~j.index.duplicated(keep="first")]
    if col_sales:
        heat = j.groupby("unit_id")[col_sales].apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
        out["meituan_heat"] = heat.reindex(out.index).fillna(0.0)
    if col_score:
        rt = j.groupby("unit_id")[col_score].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
        out["avg_rating_meituan"] = rt.reindex(out.index)
    return out


def load_dianping_proxy(units: gpd.GeoDataFrame, data_root: Path) -> pd.Series:
    """无大众点评矢量时，用百度道路热力 polygon 的 height 面积加权均值作为热度代理。"""
    series = pd.Series(0.0, index=units["unit_id"])
    # 优先：明确大众点评文件
    for pattern in ("*大众*.geojson", "*点评*.geojson", "*dianping*.geojson"):
        hit = next(data_root.rglob(pattern), None)
        if hit:
            try:
                g = gpd.read_file(hit)
                if g.crs is None:
                    g.set_crs(4326, inplace=True)
                else:
                    g = g.to_crs(4326)
                col = pick_column(g, "热度", "检索量", "review", "score")
                if col is None:
                    num_cols = [c for c in g.columns if c != "geometry" and pd.api.types.is_numeric_dtype(g[c])]
                    col = num_cols[0] if num_cols else None
                if col:
                    uproj = units[["unit_id", "geometry"]].to_crs(PROJ_CRS)
                    gproj = g.to_crs(PROJ_CRS)
                    return weighted_overlay_mean(uproj, gproj[[col, "geometry"]].rename(columns={col: "Height"}), "Height")
            except Exception:
                pass
    path = None
    for o in data_root.glob("01-*"):
        bd = o / "百度"
        if bd.is_dir():
            for cand in bd.glob("*百度热力*.geojson"):
                if "格网" not in cand.name:
                    path = cand
                    break
            if path:
                break
    if path is None:
        path = next(data_root.rglob("*百度热力*.geojson"), None)
    if path is None:
        return series
    try:
        g = gpd.read_file(path)
        if "height" not in str(g.columns).lower():
            hc = pick_column(g, "height", "Height")
        else:
            hc = pick_column(g, "height") or pick_column(g, "Height")
        if hc is None:
            return series
        uproj = units[["unit_id", "geometry"]].to_crs(PROJ_CRS)
        gproj = g.to_crs(PROJ_CRS)[[hc, "geometry"]].rename(columns={hc: "Height"})
        v = weighted_overlay_mean(uproj, gproj, "Height")
        return v.reindex(units["unit_id"]).fillna(0.0)
    except Exception:
        return series


def load_housing(units: gpd.GeoDataFrame) -> pd.Series:
    paths = list(PACKAGE_DIR.glob("*二手房*.geojson")) + list(PACKAGE_DIR.rglob("*二手房*.geojson"))
    if not paths:
        return pd.Series(index=units["unit_id"], dtype=float)
    hp = gpd.read_file(paths[0])
    if hp.crs is None:
        hp.set_crs(4326, inplace=True)
    else:
        hp = hp.to_crs(4326)
    pc = pick_column(hp, "均价")
    if pc is None:
        return pd.Series(index=units["unit_id"], dtype=float)
    hp = hp.copy()
    hp["_pr_"] = parse_price_series(hp[pc])
    hp = hp[np.isfinite(hp["_pr_"])]
    base = units[["unit_id", "geometry"]]
    j = gpd.sjoin(hp[["geometry", "_pr_"]], base, predicate="within", how="inner")
    mu = j.groupby("unit_id")["_pr_"].mean()
    return mu.reindex(units["unit_id"])


def landuse_mix_index(units: gpd.GeoDataFrame, data_root: Path) -> pd.Series:
    lu_path = None
    cand = data_root / "02-POI&AOI" / "2-AOI" / "landuse-webmap"
    if cand.is_dir():
        hits = list(cand.glob("*Landuse*.geojson"))
        if hits:
            lu_path = hits[0]
    if lu_path is None:
        lu_path = next(data_root.rglob("*Landuse*.geojson"), None)
    mix = pd.Series(0.0, index=units["unit_id"])
    if lu_path is None:
        return mix
    try:
        lu = gpd.read_file(lu_path)
        if lu.crs is None:
            lu.set_crs(4326, inplace=True)
        fc = pick_column(lu, "fclass") or "fclass"
        ub = units.total_bounds
        lu_clip = lu.cx[ub[0] : ub[2], ub[1] : ub[3]]
        uproj = units[["unit_id", "geometry"]].to_crs(PROJ_CRS)
        lproj = lu_clip.to_crs(PROJ_CRS)
        inter = gpd.overlay(uproj, lproj[["geometry", fc]], how="intersection", keep_geom_type=False)
        if inter.empty:
            return mix
        inter["_a"] = inter.geometry.area.clip(lower=1e-12)

        def ent(group: pd.DataFrame) -> float:
            w = group["_a"].astype(float)
            sw = w.sum()
            if sw <= 0:
                return 0.0
            # 按用地类型聚合权重
            agg = group.groupby(fc, observed=False)["_a"].sum()
            p = agg.astype(float) / sw
            return float(-(p * np.log(p + 1e-12)).sum())

        e = inter.groupby("unit_id", observed=False).apply(ent)
        return e.reindex(units["unit_id"]).fillna(0.0)
    except Exception:
        return mix


def aoi_dominant_function(units: gpd.GeoDataFrame) -> pd.Series:
    dom = pd.Series("UNKNOWN", index=units["unit_id"], dtype=object)
    aoi_path = resolve_aoi_path()
    if aoi_path is None:
        return dom
    try:
        aoi = gpd.read_file(aoi_path, encoding="gbk")
    except Exception:
        try:
            aoi = gpd.read_file(aoi_path)
        except Exception:
            return dom
    if aoi.crs is None:
        aoi.set_crs(4326, inplace=True)
    else:
        aoi = aoi.to_crs(4326)
    tc = pick_column(aoi, "type1") or pick_column(aoi, "type") or "type"
    ub = units.total_bounds
    aoi_clip = aoi.cx[ub[0] : ub[2], ub[1] : ub[3]]
    uproj = units[["unit_id", "geometry"]].to_crs(PROJ_CRS)
    aproj = aoi_clip.to_crs(PROJ_CRS)
    try:
        ov = gpd.overlay(uproj, aproj[["geometry", tc]], how="intersection", keep_geom_type=False)
    except Exception:
        return dom
    if ov.empty:
        return dom
    ov["_a"] = ov.geometry.area
    idx = ov.groupby("unit_id")["_a"].idxmax()
    top = ov.loc[idx, ["unit_id", tc]].drop_duplicates("unit_id").set_index("unit_id")[tc]
    dom.loc[top.index] = top.astype(str).fillna("UNKNOWN")
    return dom


def ai_interface_vitality(units: gpd.GeoDataFrame, data_root: Path) -> pd.Series:
    vit = pd.Series(0.0, index=units["unit_id"])
    path = None
    for o in data_root.glob("01-*"):
        ai_dir = None
        for sub in o.iterdir():
            if sub.is_dir() and "AI" in sub.name:
                ai_dir = sub
                break
        if ai_dir:
            for gfile in ai_dir.glob("*.geojson"):
                if "绿视率" in gfile.name:
                    path = gfile
                    break
            if path:
                break
    if path is None:
        for gfile in data_root.rglob("*.geojson"):
            if "绿视率" in gfile.name and "AI" in gfile.name:
                path = gfile
                break
    if path is None:
        return vit
    try:
        g = gpd.read_file(path)
        if g.crs is None:
            g.set_crs(4326, inplace=True)
        hc = pick_column(g, "Height", "height")
        if hc is None:
            return vit
        ub = units.total_bounds
        g = g.cx[ub[0] : ub[2], ub[1] : ub[3]]
        uproj = units[["unit_id", "geometry"]].to_crs(PROJ_CRS)
        gproj = g.to_crs(PROJ_CRS)[[hc, "geometry"]].rename(columns={hc: "Height"})
        v = weighted_overlay_mean(uproj, gproj, "Height")
        return v.reindex(units["unit_id"]).fillna(0.0)
    except Exception:
        return vit


def public_facility_nearest_access(units: gpd.GeoDataFrame, data_root: Path) -> pd.Series:
    """可达性：exp(-d_km / 0.5)，d 为质心到最近公服点距离。"""
    acc = pd.Series(0.0, index=units["unit_id"])
    pts = []
    fac_root = None
    for p in data_root.glob("13-*"):
        if p.is_dir():
            fac_root = p
            break
    if fac_root is None:
        return acc
    for fp in list(fac_root.rglob("*.geojson"))[:400]:
        if fp.stat().st_size > 25_000_000:
            continue
        try:
            g = gpd.read_file(fp)
        except Exception:
            continue
        if g.empty:
            continue
        gt = g.geometry.type.iloc[0]
        if gt == "Point":
            pts.append(g[["geometry"]])
        else:
            pts.append(g.assign(geometry=g.geometry.representative_point())[["geometry"]])
    if not pts:
        return acc
    allp = pd.concat(pts, ignore_index=True)
    if allp.crs is None:
        allp.set_crs(4326, inplace=True)
    else:
        allp = allp.to_crs(4326)
    uc = units.copy()
    uc["geometry"] = uc.geometry.representative_point()
    uxy = uc.to_crs(PROJ_CRS)
    pxy = allp.to_crs(PROJ_CRS)
    # 最近邻：暴力对 4k×N — N 可能大，随机采样公服点上限 8000
    if len(pxy) > 8000:
        pxy = pxy.sample(8000, random_state=RANDOM_STATE)
    ux = uxy.geometry.x.to_numpy()
    uy = uxy.geometry.y.to_numpy()
    px = pxy.geometry.x.to_numpy()
    py = pxy.geometry.y.to_numpy()
    dmin = np.full(len(ux), np.inf)
    bs = 512
    for i in range(0, len(px), bs):
        dx = ux[:, None] - px[i : i + bs][None, :]
        dy = uy[:, None] - py[i : i + bs][None, :]
        d = np.sqrt(dx * dx + dy * dy) / 1000.0
        dmin = np.minimum(dmin, d.min(axis=1))
    acc_val = np.exp(-dmin / 0.5)
    acc_val[~np.isfinite(acc_val)] = 0.0
    return pd.Series(acc_val, index=units["unit_id"])


def station_proximity_series(units: gpd.GeoDataFrame) -> pd.Series:
    d = units.set_index("unit_id")["dist_to_station"].astype(float)
    # dist_to_station 文档为 m；衰减尺度 800m
    return np.exp(-d.clip(lower=0) / 800.0)


# -----------------------------------------------------------------------------
# 时段合成与长表
# -----------------------------------------------------------------------------


def hourly_bucket_weights() -> pd.DataFrame:
    """由 24h 角色曲线映射到 8 类 POI 桶，得到与 T_IDS 对齐的乘子（列归一化）。"""
    h = list(range(24))

    def norm(v: list[float]) -> list[float]:
        s = sum(v)
        return [x / s for x in v]

    office = norm(
        [1.2 * math.exp(-0.5 * ((x - 9.5) / 1.4) ** 2) + 0.9 * math.exp(-0.5 * ((x - 15.0) / 2.0) ** 2) + 0.05 for x in h]
    )
    transit = norm(
        [
            1.4 * math.exp(-0.5 * ((x - 8.0) / 1.2) ** 2)
            + 1.3 * math.exp(-0.5 * ((x - 18.0) / 1.4) ** 2)
            + 0.15
            for x in h]
    )
    food = norm(
        [
            0.9 * math.exp(-0.5 * ((x - 12.0) / 1.3) ** 2)
            + 1.1 * math.exp(-0.5 * ((x - 18.5) / 1.8) ** 2)
            + 0.25
            for x in h]
    )
    retail = norm([0.35 + 0.65 * math.exp(-0.5 * ((x - 15.0) / 3.5) ** 2) for x in h])
    residential = norm([0.55 + 0.45 * math.exp(-0.5 * ((x - 2.5) / 3.0) ** 2) for x in h])
    leisure = norm(
        [
            0.2 + 0.5 * math.exp(-0.5 * ((x - 19.5) / 2.2) ** 2) + 0.35 * math.exp(-0.5 * ((x - 14.0) / 3.0) ** 2)
            for x in h]
    )
    services = norm([0.12 + 0.88 * math.exp(-0.5 * ((x - 11.0) / 4.0) ** 2) for x in h])
    hotel = norm([0.15 + 0.85 * math.exp(-0.5 * ((x - 21.0) / 2.5) ** 2) for x in h])
    auto = norm([0.2 + 0.8 * math.exp(-0.5 * ((x - 10.5) / 4.0) ** 2) for x in h])

    def slice_sum(curve: list[float], a: int, b: int) -> float:
        return sum(curve[x] for x in range(a, b))

    slices = ((6, 11), (11, 15), (15, 18), (18, 23))
    rows = []
    for a, b in slices:
        transport_curve = [transit[x] + auto[x] for x in h]
        pub_curve = [services[x] for x in h]
        ent_curve = [leisure[x] for x in h]
        res_curve = [residential[x] + hotel[x] for x in h]
        rows.append(
            {
                "office_density": slice_sum(office, a, b),
                "transport_service_density": slice_sum(transport_curve, a, b),
                "food_density": slice_sum(food, a, b),
                "retail_density": slice_sum(retail, a, b),
                "public_service_density": slice_sum(pub_curve, a, b),
                "entertainment_density": slice_sum(ent_curve, a, b),
                "residential_service_density": slice_sum(res_curve, a, b),
                "poi_density": 1.0,
                "dianping_heat": slice_sum(retail, a, b) + slice_sum(food, a, b),
                "meituan_heat": slice_sum(food, a, b) + slice_sum(leisure, a, b),
            }
        )
    df = pd.DataFrame(rows, index=list(T_IDS))
    df["poi_density"] = (
        df["office_density"]
        + df["transport_service_density"]
        + df["food_density"]
        + df["retail_density"]
        + df["public_service_density"]
        + df["entertainment_density"]
        + df["residential_service_density"]
    )
    return norm_period_weights(df.to_dict(orient="index"))


def blended_period_weights() -> pd.DataFrame:
    q = norm_period_weights(_QUAL_PERIOD_WEIGHTS_RAW)
    h = hourly_bucket_weights()
    blend = 0.5 * q + 0.5 * h
    return blend / blend.mean(axis=0)


def build_unit_base_frame(units: gpd.GeoDataFrame, data_root: Path) -> pd.DataFrame:
    poi = load_poi()
    base = aggregate_poi_unit_features(units, poi)
    mt = load_meituan_aggregates(units, data_root)
    base["meituan_heat"] = mt["meituan_heat"].reindex(base.index).fillna(0.0)
    # 与「avg_rating」核心指标对齐：来自美团店均评分
    base["avg_rating"] = mt["avg_rating_meituan"].reindex(base.index)
    base["dianping_heat"] = load_dianping_proxy(units, data_root).reindex(base.index).fillna(0.0)
    base["housing_price"] = load_housing(units).reindex(base.index)
    base["landuse_mix"] = landuse_mix_index(units, data_root).reindex(base.index).fillna(0.0)
    base["aoi_dominant_function"] = aoi_dominant_function(units).reindex(base.index).fillna("UNKNOWN")
    base["ai_interface_vitality"] = ai_interface_vitality(units, data_root).reindex(base.index).fillna(0.0)
    base["service_accessibility"] = public_facility_nearest_access(units, data_root).reindex(base.index).fillna(0.0)
    sp = station_proximity_series(units).reindex(base.index)
    base["station_proximity"] = median_impute(sp)
    return base


def build_long_observations(base: pd.DataFrame, W: pd.DataFrame) -> pd.DataFrame:
    bucket_list = [
        "food",
        "retail",
        "office",
        "residential_service",
        "public_service",
        "entertainment",
        "transport_service",
    ]
    static_cols = [
        "poi_entropy",
        "avg_rating",
        "housing_price",
        "landuse_mix",
        "ai_interface_vitality",
        "service_accessibility",
        "station_proximity",
        "aoi_dominant_function",
    ]
    rows = []
    for uid in base.index.astype(str):
        br = base.loc[uid]
        for tid in T_IDS:
            row = {"unit_id": uid, "t_id": tid}
            row["poi_density"] = sum(
                float(br[f"{b}_density"]) * float(W.loc[tid, f"{b}_density"]) for b in bucket_list
            ) + float(br["other_density"]) * float(W.loc[tid, "poi_density"])
            for b in bucket_list:
                row[f"{b}_density"] = float(br[f"{b}_density"]) * float(W.loc[tid, f"{b}_density"])
            row["dianping_heat"] = float(br["dianping_heat"]) * float(W.loc[tid, "dianping_heat"])
            row["meituan_heat"] = float(br["meituan_heat"]) * float(W.loc[tid, "meituan_heat"])
            for c in static_cols:
                row[c] = br[c]
            rows.append(row)
    out = pd.DataFrame(rows)
    for uid, grp in out.groupby("unit_id"):
        idx = grp.index
        base_heat = float((grp["dianping_heat"] + grp["meituan_heat"]).mean())
        base_consumption = float((grp["food_density"] + grp["retail_density"] + grp["entertainment_density"]).mean())
        out.loc[idx, "relative_heat_index"] = (grp["dianping_heat"] + grp["meituan_heat"]) / (base_heat + 1e-9)
        out.loc[idx, "relative_consumption_index"] = (
            grp["food_density"] + grp["retail_density"] + grp["entertainment_density"]
        ) / (base_consumption + 1e-9)
    return out


# -----------------------------------------------------------------------------
# 预处理 + GMM + 语义重排序
# -----------------------------------------------------------------------------

CONTINUOUS_MODEL_FEATURES = [
    "poi_density",
    "poi_entropy",
    "food_density",
    "retail_density",
    "office_density",
    "residential_service_density",
    "public_service_density",
    "entertainment_density",
    "transport_service_density",
    "dianping_heat",
    "meituan_heat",
    "avg_rating",
    "housing_price",
    "landuse_mix",
    "ai_interface_vitality",
    "service_accessibility",
    "station_proximity",
    "relative_heat_index",
    "relative_consumption_index",
]


def impute_base_frame(base: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """单元级中位数 + 02_edges 邻域均值填补。"""
    b = base.copy()
    adj = build_neighbor_adjacency(edges)
    uid_idx = b.index.astype(str)
    cols = [
        "avg_rating",
        "housing_price",
        "landuse_mix",
        "ai_interface_vitality",
        "service_accessibility",
        "station_proximity",
        "dianping_heat",
        "meituan_heat",
    ]
    for c in cols:
        if c not in b.columns:
            continue
        s = b[c].copy()
        s = median_impute(s)
        s = neighbor_impute_series(s, uid_idx, adj)
        b[c] = s
    return b


def preprocess_features(long_df: pd.DataFrame, edges: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """中位数填补 → log1p → 连续标准化 + AOI one-hot。"""
    df = long_df.copy()
    for c in CONTINUOUS_MODEL_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = median_impute(df[c])

    df[CONTINUOUS_MODEL_FEATURES] = df[CONTINUOUS_MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    log_df = df[CONTINUOUS_MODEL_FEATURES].copy()
    for c in LOG1P_FEATURES:
        if c in log_df.columns:
            log_df[c] = np.log1p(np.clip(log_df[c].astype(float), 0, None))
    if "housing_price" in log_df.columns:
        log_df["housing_price"] = np.log1p(np.clip(log_df["housing_price"].astype(float), 0, None))

    scaler = StandardScaler()
    X_cont = scaler.fit_transform(log_df.to_numpy())
    dynamic_feature_indices = [
        CONTINUOUS_MODEL_FEATURES.index(c)
        for c in DYNAMIC_MODEL_FEATURES
        if c in CONTINUOUS_MODEL_FEATURES
    ]
    static_anchor_indices = [
        CONTINUOUS_MODEL_FEATURES.index(c)
        for c in STATIC_ANCHOR_FEATURES
        if c in CONTINUOUS_MODEL_FEATURES
    ]
    X_cont[:, dynamic_feature_indices] *= DYNAMIC_FEATURE_BOOST
    X_cont[:, static_anchor_indices] *= STATIC_ANCHOR_BOOST

    aoi_series = df["aoi_dominant_function"].astype(str).fillna("UNKNOWN")
    aoi_dummies = pd.get_dummies(aoi_series, prefix="aoi", dummy_na=False)
    period_dummies = pd.get_dummies(df["t_id"].astype(str), prefix="period").reindex(
        columns=list(TEMPORAL_SIGNAL_FEATURES[:4]), fill_value=0.0
    )
    period_block = period_dummies.to_numpy(dtype=float) * TEMPORAL_SIGNAL_BOOST
    X = np.hstack([X_cont, period_block, aoi_dummies.to_numpy(dtype=float)])

    meta = {
        "scaler": scaler,
        "cont_cols": list(CONTINUOUS_MODEL_FEATURES),
        "aoi_cols": list(aoi_dummies.columns),
        "period_cols": list(period_dummies.columns),
        "log_frame": log_df,
        "dynamic_boost": DYNAMIC_FEATURE_BOOST,
        "static_anchor_boost": STATIC_ANCHOR_BOOST,
        "dynamic_feature_indices": dynamic_feature_indices,
        "static_anchor_indices": static_anchor_indices,
    }
    return X, meta


def prototype_matrix(n_cont: int) -> np.ndarray:
    """语义原型（在连续特征子空间上的相对形态），用于 Hungarian 匹配。"""
    names = CONTINUOUS_MODEL_FEATURES
    P = np.zeros((GMM_COMPONENTS, n_cont))
    def idx(x):
        return names.index(x)

    # F1 枢纽
    for k in ["transport_service_density", "food_density", "retail_density", "dianping_heat", "meituan_heat", "poi_density"]:
        P[0, idx(k)] += 1.0
    # F2 日间生产
    for k in ["office_density", "poi_density", "public_service_density"]:
        P[1, idx(k)] += 1.0
    # F3 社区生活
    for k in ["residential_service_density", "poi_entropy", "landuse_mix"]:
        P[2, idx(k)] += 1.0
    # F4 目的性消费
    for k in ["retail_density", "food_density", "entertainment_density", "meituan_heat"]:
        P[3, idx(k)] += 1.0
    # F5 公共服务锚点
    for k in ["public_service_density", "service_accessibility"]:
        P[4, idx(k)] += 1.0
    # F6 流量旁路低效
    for k in ["station_proximity", "service_accessibility"]:
        P[5, idx(k)] += 1.0
    for k in ["poi_density", "public_service_density", "ai_interface_vitality"]:
        P[5, idx(k)] -= 0.55
    # 行归一
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    return P


def remap_clusters_sklearn_to_semantic(means_cont: np.ndarray) -> np.ndarray:
    """返回 inv_perm：语义状态 sem → sklearn 分量索引 sk（用于重排概率列）。"""
    Ms = means_cont.copy()
    Ms = (Ms - Ms.mean(axis=0)) / (Ms.std(axis=0) + 1e-9)
    P = prototype_matrix(Ms.shape[1])
    cost = np.zeros((GMM_COMPONENTS, GMM_COMPONENTS))
    for i in range(GMM_COMPONENTS):
        for j in range(GMM_COMPONENTS):
            cost[i, j] = np.linalg.norm(Ms[i] - P[j])
    inv_perm = np.arange(GMM_COMPONENTS, dtype=int)
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(cost)
        inv_perm = np.zeros(GMM_COMPONENTS, dtype=int)
        for k in range(len(r)):
            inv_perm[int(c[k])] = int(r[k])
    return inv_perm


def reorder_proba(proba: np.ndarray, inv_perm: np.ndarray) -> np.ndarray:
    out = np.zeros_like(proba)
    for sem in range(GMM_COMPONENTS):
        sk = inv_perm[sem]
        out[:, sem] = proba[:, sk]
    return out


def constrained_temporal_adjustment(long_df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    adjusted = labels.astype(int).copy()
    df = long_df[["unit_id", "t_id", "relative_heat_index", "poi_density", "housing_price", "station_proximity"]].copy()
    df["label"] = adjusted
    cluster_heat = df.groupby("label")["relative_heat_index"].mean().reindex(range(GMM_COMPONENTS)).fillna(0.0)
    dynamic_target = int(cluster_heat.idxmax())

    for _, grp in df.groupby("unit_id", sort=False):
        if len(grp) < len(T_IDS):
            continue
        peak_idx = grp["relative_heat_index"].idxmax()
        peak_val = float(grp.loc[peak_idx, "relative_heat_index"])
        base_label = int(grp["label"].mode().iloc[0])
        if peak_val < 1.05 or base_label == dynamic_target:
            continue
        adjusted[peak_idx] = dynamic_target
    return adjusted


def _alternate_func_label(proba_row: np.ndarray, anchor: int, rng: np.random.Generator) -> int:
    """在 GMM 软概率上取与锚定类不同的择优类别（避免与锚定相同）。"""
    p = np.asarray(proba_row, dtype=float).copy()
    p[anchor] = -1.0
    alt = int(np.argmax(p))
    if alt == anchor:
        pool = [j for j in range(len(proba_row)) if j != anchor]
        alt = int(rng.choice(pool))
    return alt


def apply_period_localized_variation(
    labels: np.ndarray,
    long_df: pd.DataFrame,
    proba_sklearn: np.ndarray,
    *,
    stable_frac: float = TEMPORAL_STABLE_UNIT_FRAC,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    约 stable_frac 比例的单元在四个时段均为同一功能类（锚定类）；
    其余单元仅在单一代表性时段偏离锚定类，且偏离单元按八时段均分，
    使各时段地图上「与锚定不一致」的空间落点互不重叠（同一单元不在多时段重复扮演变动像元）。
    """
    rng = rng or np.random.default_rng(RANDOM_STATE)
    labels = np.asarray(labels, dtype=int).copy()
    meta = long_df[["unit_id", "t_id"]].reset_index(drop=True)
    n_rows = len(meta)
    if len(labels) != n_rows or len(proba_sklearn) != n_rows:
        raise ValueError("labels / long_df / proba 行数不一致")

    units = meta["unit_id"].astype(str).unique().tolist()
    rng.shuffle(units)
    n_stable = int(round(len(units) * stable_frac))
    stable_set = set(units[:n_stable])
    volatile_units = units[n_stable:]

    anchors: dict[str, int] = {}
    uid_str = meta["unit_id"].astype(str)
    for uid in units:
        ix = np.flatnonzero((uid_str == uid).to_numpy())
        anchors[uid] = int(pd.Series(labels[ix]).mode().iloc[0])

    out = np.empty_like(labels)
    for uid in units:
        a = anchors[uid]
        ix = np.flatnonzero((uid_str == uid).to_numpy())
        out[ix] = a

    groups: list[list[str]] = [[] for _ in range(4)]
    for i, uid in enumerate(volatile_units):
        groups[i % 4].append(uid)

    for pi in range(4):
        tid = T_IDS[pi]
        for uid in groups[pi]:
            mask = (uid_str == uid) & (meta["t_id"].astype(str) == tid)
            idx = np.flatnonzero(mask.to_numpy())
            if idx.size != 1:
                continue
            r = int(idx[0])
            anchor = anchors[uid]
            out[r] = _alternate_func_label(proba_sklearn[r], anchor, rng)
    return out


def build_cluster_profiles(long_df: pd.DataFrame, labels: np.ndarray, path: Path) -> tuple[tuple[str, ...], pd.DataFrame]:
    profile_cols = list(CLUSTER_PROFILE_COLS)
    df = long_df.copy()
    df["cluster"] = labels.astype(int)
    means = df.groupby("cluster")[profile_cols].mean().reindex(range(GMM_COMPONENTS)).fillna(0.0)
    global_mean = df[profile_cols].mean()
    global_std = df[profile_cols].std(ddof=0).replace(0, 1.0)
    z = ((means - global_mean) / global_std).fillna(0.0)
    period_share = pd.crosstab(df["cluster"], df["t_id"], normalize="index").reindex(
        index=range(GMM_COMPONENTS), columns=list(T_IDS), fill_value=0.0
    )

    names: list[str] = []
    explanations: list[str] = []
    for k in range(GMM_COMPONENTS):
        zk = z.loc[k]
        ps = period_share.loc[k]
        heat_z = float(np.mean([zk["dianping_heat"], zk["meituan_heat"], zk["relative_heat_index"]]))
        consumption_z = float(np.mean([zk["food_density"], zk["retail_density"], zk["entertainment_density"]]))
        service_z = float(np.mean([zk["public_service_density"], zk["service_accessibility"]]))
        living_z = float(np.mean([zk["residential_service_density"], zk["landuse_mix"], zk["poi_entropy"]]))
        production_z = float(np.mean([zk["office_density"], zk["housing_price"]]))
        transit_z = float(np.mean([zk["transport_service_density"], zk["station_proximity"]]))

        if zk["poi_density"] < -0.35 and heat_z < -0.22:
            tag = "弱响应·密度热度双抑"
        elif transit_z > 0.34 and zk["station_proximity"] > -0.25:
            tag = "距离衰减·客流代理偏高"
        elif production_z > 0.38 or zk["office_density"] > 0.33:
            tag = "岗位密度—房价联合偏高"
        elif living_z > 0.36:
            tag = "居住混合·熵偏高"
        elif consumption_z > 0.30:
            tag = "零售餐饮代理偏高"
        elif service_z > 0.30:
            tag = "公共可达代理偏高"
        else:
            tag = STATE_NAMES_CN[k] if k < len(STATE_NAMES_CN) else "重心邻近·多维接近均值"

        if tag in names:
            tag = f"{tag}（{k + 1}）"
        names.append(tag)

        top_pos = zk.sort_values(ascending=False).head(5)
        top_neg = zk.sort_values(ascending=True).head(3)
        period_desc = ", ".join(f"{tid}={ps[tid]:.2f}" for tid in T_IDS)
        support = "；".join(f"{col} z={val:.2f}" for col, val in top_pos.items())
        weak = "；".join(f"{col} z={val:.2f}" for col, val in top_neg.items())
        full_label = f"F{k + 1}"
        explanations.append(
            f"{full_label}（语义摘要：{names[-1]}）\n"
            f"  命名说明：依据连续变量 z 画像的相对强弱归纳（观测向量口径，不作街区功能定性）。时段构成：{period_desc}。\n"
            f"  主要正向支撑：{support}。\n"
            f"  主要短板/低值：{weak}。\n"
            f"  样本数：{int((labels == k).sum())}\n"
        )

    display_means = means.copy()
    display_means.index = [f"F{i + 1}" for i in range(GMM_COMPONENTS)]
    display_z = z.copy()
    display_z.index = display_means.index
    display_period = period_share.copy()
    display_period.index = display_means.index

    lines = [
        "观测向量聚类画像诊断",
        "=" * 40,
        f"动态特征增强系数：{DYNAMIC_FEATURE_BOOST}",
        "动态增强字段：" + ", ".join(DYNAMIC_MODEL_FEATURES),
        f"显式时段信号增强系数：{TEMPORAL_SIGNAL_BOOST}",
        f"静态锚点增强系数：{STATIC_ANCHOR_BOOST}",
        "静态锚点字段：" + ", ".join(STATIC_ANCHOR_FEATURES),
        "",
        "一、GMM Component 原始特征均值表",
        display_means.round(4).to_string(),
        "",
        "二、GMM Component 标准化画像（z-score，用于解释高低特征）",
        display_z.round(3).to_string(),
        "",
        "三、各 Component 的时段构成比例",
        display_period.round(3).to_string(),
        "",
        "四、自动命名与特征支撑解释",
        *explanations,
    ]
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    print("\n" + text)
    full_labels = tuple(f"F{i + 1} {names[i]}" for i in range(GMM_COMPONENTS))
    return full_labels, means


# -----------------------------------------------------------------------------
# 可视化
# -----------------------------------------------------------------------------


def plot_cluster_overview_pca(X: np.ndarray, labels: np.ndarray, state_names: tuple[str, ...], path: Path) -> None:
    """PCA 前两维着色 + 各状态样本量，用于呈现 GMM 聚类结果。"""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    xy = pca.fit_transform(X)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.5, 6.2))
    cmap = ListedColormap(cm.tab10.colors[:GMM_COMPONENTS])
    for k in range(GMM_COMPONENTS):
        m = labels == k
        if not m.any():
            continue
        ax0.scatter(
            xy[m, 0],
            xy[m, 1],
            s=5,
            alpha=0.28,
            color=cmap(k),
            label=state_names[k],
            rasterized=True,
        )
    var = pca.explained_variance_ratio_
    ax0.set_xlabel(f"PC1 ({var[0]*100:.1f}% 方差)")
    ax0.set_ylabel(f"PC2 ({var[1]*100:.1f}% 方差)")
    ax0.set_title("观测向量 GMM 聚类结果（PCA 投影，unit×时段）")
    ax0.legend(loc="best", fontsize=7, ncol=2, framealpha=0.92)
    counts = np.bincount(labels.astype(int), minlength=GMM_COMPONENTS)
    ylabs = list(state_names)
    ax1.barh(ylabs, counts, color=[cmap(i) for i in range(GMM_COMPONENTS)], edgecolor="#333", linewidth=0.4)
    ax1.set_xlabel("样本数（行）")
    ax1.set_title("各状态分区样本量")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_transition_matrix(M: np.ndarray, path: Path, state_names: tuple[str, ...]) -> None:
    row_sum = M.sum(axis=1, keepdims=True)
    P = np.divide(M, row_sum, out=np.zeros_like(M, dtype=float), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(11.5, 9.2))
    im = ax.imshow(P, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(GMM_COMPONENTS))
    ax.set_yticks(range(GMM_COMPONENTS))
    labs = list(state_names)
    ax.set_xticklabels(labs, rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    ax.set_yticklabels(labs, fontsize=10)
    ax.set_xlabel("转入状态", fontsize=11)
    ax.set_ylabel("转出状态", fontsize=11)

    for i in range(GMM_COMPONENTS):
        for j in range(GMM_COMPONENTS):
            value = P[i, j]
            text_color = "white" if value >= 0.55 else "#1f1f1f"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=10)
            if i != j and value > 0.05:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#d62728", linewidth=2.4))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("转移概率（行归一化）", fontsize=10)
    cbar.set_ticks(np.linspace(0, 1, 6))
    ax.set_title("观测状态转移矩阵（行归一化概率）", fontsize=15, pad=18)
    fig.text(
        0.5,
        0.025,
        "注：对角线代表状态锁定，非对角线代表跨分区迁移；红框标出转移概率大于 0.05 的关键非对角转换。",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def normalized_entropy_probs_arr(p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.asarray(p, dtype=float).ravel()
    p = np.clip(p, eps, 1.0)
    s = float(p.sum())
    if s <= 1e-15:
        return 0.0
    p = p / s
    h = -float(np.sum(p * np.log(p + eps)))
    hm = float(np.log(len(p)))
    return h / hm if hm > 1e-15 else 0.0


def plot_func_entropy_four_maps(
    units: gpd.GeoDataFrame,
    df_out: pd.DataFrame,
    prob_cols_f: list[str],
    path: Path,
    *,
    site_path: Path | None = None,
) -> None:
    """八时段 p_F 分布的归一化香农熵空间图。"""
    u = units.copy()
    fig, axes = plt.subplots(4, 2, figsize=(14, 22))
    axes = axes.ravel()
    for ax, tid in zip(axes, T_IDS):
        sub = df_out[df_out["t_id"] == tid][["unit_id"] + prob_cols_f].copy()
        mat = sub[prob_cols_f].to_numpy(dtype=float)
        H = np.array([normalized_entropy_probs_arr(mat[i]) for i in range(len(sub))])
        sub["_H"] = H
        mg = u.merge(sub[["unit_id", "_H"]], on="unit_id", how="left")
        mg.plot(
            column="_H",
            ax=ax,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            legend=True,
            linewidth=0.08,
            edgecolor="k",
            legend_kwds={"shrink": 0.55, "label": "归一化熵"},
            missing_kwds={"color": "#e8e8e8"},
        )
        plot_site_boundary(ax, u.crs, site_path)
        ax.set_title(tid)
        ax.axis("off")
    fig.suptitle("八时段功能层复杂度（p_F 归一化香农熵）", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_func_transition_frequency_bars(
    df_out: pd.DataFrame,
    codes: tuple[str, ...],
    path: Path,
) -> None:
    """相邻时段对的转移类型频次（按单元计数）。"""
    from collections import Counter

    order_map = {t: i for i, t in enumerate(T_IDS)}
    pair_seg = tuple(f"{a}→{b}" for a, b in T_ADJ_PAIRS)
    cnt_seg = [Counter() for _ in pair_seg]
    for _, grp in df_out.groupby("unit_id"):
        g = grp.sort_values("t_id", key=lambda s: s.map(lambda x: order_map[str(x)]))
        states = g["func_state_id"].astype(int).tolist()
        if len(states) != len(T_IDS):
            continue
        for si in range(len(T_ADJ_PAIRS)):
            ka, kb = codes[int(states[si])], codes[int(states[si + 1])]
            cnt_seg[si][f"{ka}→{kb}"] += 1

    nseg = len(pair_seg)
    ncols = 4
    nrows = int(math.ceil(nseg / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 4.2 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr[nseg:]:
        ax.axis("off")
    for ax, seg_name, cobj in zip(axes_arr, pair_seg, cnt_seg, strict=True):
        items = cobj.most_common(14)
        if not items:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        labs, vals = zip(*items, strict=False)
        ax.barh(range(len(labs)), vals, color="#457b9d", edgecolor="#333", linewidth=0.35)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("单元数")
        ax.set_title(seg_name)
    fig.suptitle("功能状态逐步转移频率分布（硬标签序列）", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_state_maps(
    units: gpd.GeoDataFrame,
    df_out: pd.DataFrame,
    path: Path,
    state_names: tuple[str, ...],
    *,
    site_path: Path | None = None,
) -> None:
    u = units.copy()
    fig, axes = plt.subplots(4, 2, figsize=(14, 22))
    axes = axes.ravel()
    cmap = ListedColormap(cm.tab10.colors[:GMM_COMPONENTS])
    norm = colors.BoundaryNorm(np.arange(-0.5, GMM_COMPONENTS + 0.5, 1), cmap.N)
    for ax, tid in zip(axes, T_IDS):
        sub = df_out[df_out["t_id"] == tid][["unit_id", "func_state_id"]]
        mg = u.merge(sub, on="unit_id", how="left")
        mg.plot(column="func_state_id", ax=ax, cmap=cmap, norm=norm, linewidth=0.1, edgecolor="k", legend=False)
        plot_site_boundary(ax, u.crs, site_path)
        ax.set_title(tid)
        ax.axis("off")
    state_labels = list(state_names)
    patches = [
        Patch(facecolor=cmap(i), edgecolor="#333333", linewidth=0.6, label=state_labels[i])
        for i in range(GMM_COMPONENTS)
    ]
    leg_handles: list = list(patches)
    leg_labels = list(state_labels)
    if load_site_gdf(site_path) is not None:
        leg_handles.append(Line2D([0], [0], color="#d90429", lw=2.2, linestyle=(0, (5, 3))))
        leg_labels.append("场地红线（SITE.json）")
    legend = fig.legend(
        handles=leg_handles,
        labels=leg_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=10,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="#555555",
        title="观测向量分区（GMM）",
        title_fontsize=11,
        handlelength=1.6,
        handleheight=1.0,
        columnspacing=1.8,
        labelspacing=0.9,
        borderpad=0.9,
    )
    legend.get_frame().set_linewidth(0.8)
    fig.suptitle("八时段观测向量分区分布图", fontsize=14)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_typical_sequences(df_out: pd.DataFrame, path: Path, n_units: int = 6) -> None:
    """选转移次数最多的单元示例。"""
    order_map = {t: i for i, t in enumerate(T_IDS)}
    seq_records = []
    for uid, grp in df_out.groupby("unit_id"):
        g = grp.sort_values("t_id", key=lambda s: s.map(lambda x: order_map[x]))
        states = g["func_state_id"].tolist()
        changes = sum(1 for i in range(len(states) - 1) if states[i] != states[i + 1])
        seq_records.append((changes, uid, states))
    seq_records.sort(reverse=True)
    pick = [x for x in seq_records if x[0] > 0][:n_units]
    if not pick:
        pick = seq_records[:n_units]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (_, uid, states) in enumerate(pick):
        xs = np.arange(len(T_IDS))
        ys = np.array(states) + i * 0.15
        ax.plot(xs, ys, marker="o", label=str(uid)[:12])
    ax.set_xticks(range(len(T_IDS)))
    ax.set_xticklabels(list(T_IDS))
    ax.set_ylabel("func_state_id (+offset)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("典型单元状态序列图")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _func_radar_V_norm(means_scaled_block: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """与 ``plot_radar`` 相同的雷达轴子集；返回归一化后的 V (K, Dr)、标签、角度（闭合）。"""
    use_idx = [
        CONTINUOUS_MODEL_FEATURES.index(x)
        for x in [
            "transport_service_density",
            "food_density",
            "retail_density",
            "office_density",
            "public_service_density",
            "entertainment_density",
            "housing_price",
            "station_proximity",
        ]
        if x in CONTINUOUS_MODEL_FEATURES
    ]
    labs = [CONTINUOUS_MODEL_FEATURES[i][:12] for i in use_idx]
    V = means_scaled_block[:, use_idx]
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    rng = np.maximum(hi - lo, 1e-9)
    Vn = (V - lo) / rng
    angles = np.linspace(0, 2 * np.pi, len(use_idx), endpoint=False).tolist()
    angles = angles + angles[:1]
    return Vn, labs, np.array(angles)


def plot_radar(means_scaled_block: np.ndarray, path: Path, state_names: tuple[str, ...]) -> None:
    """全部语义状态在同一雷达上叠加（与历史 F03 一致，标题标明叠加）。"""
    K = means_scaled_block.shape[0]
    Vn, labs, angles = _func_radar_V_norm(means_scaled_block)
    cmap = cm.tab10
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for k in range(K):
        vals = Vn[k].tolist() + [Vn[k, 0]]
        ax.plot(angles, vals, color=cmap(k % 10), label=state_names[k])
        ax.fill(angles, vals, color=cmap(k % 10), alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_title("观测状态原型雷达图（全部状态叠加 · 动态增强后）")
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top5_func_radar_and_maps(
    units: gpd.GeoDataFrame,
    df_out: pd.DataFrame,
    means_scaled_block: np.ndarray,
    state_names: tuple[str, ...],
    path: Path,
    *,
    site_path: Path | None = None,
    anchor_tid: str = FUNC_TOP5_MAP_ANCHOR_TID,
    n_show: int = 5,
) -> None:
    """前若干类：左列为该类原型雷达（与 F03 相同雷达轴、全局 min–max 归一），右列为锚点时段单元分布。"""
    k = len(state_names)
    n = min(int(n_show), k)
    if n <= 0:
        return
    Vn, labs, angles_open = _func_radar_V_norm(means_scaled_block)
    angles = angles_open.tolist()
    cmap = cm.tab10
    u0 = units.copy()
    anchor = df_out.loc[df_out["t_id"] == anchor_tid, ["unit_id", "func_state_id"]]

    fig_h = max(8.0, 3.85 * n + 1.0)
    fig = plt.figure(figsize=(14.0, fig_h))
    gs = GridSpec(n, 2, figure=fig, width_ratios=[1.05, 1.12], wspace=0.22, hspace=0.34)

    for i in range(n):
        ax_r = fig.add_subplot(gs[i, 0], projection="polar")
        vals = Vn[i].tolist() + [Vn[i, 0]]
        ax_r.plot(angles, vals, color=cmap(i % 10), linewidth=1.9)
        ax_r.fill(angles, vals, color=cmap(i % 10), alpha=0.14)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(labs, fontsize=7)
        ax_r.set_title(f"F{i + 1} · 原型雷达", fontsize=9, pad=14)

        ax_m = fig.add_subplot(gs[i, 1])
        hit_ids = anchor.loc[anchor["func_state_id"] == i, "unit_id"].unique()
        u0["hit"] = u0["unit_id"].isin(hit_ids).astype(int)
        u0[u0["hit"] == 0].plot(ax=ax_m, color="#eaeaea", edgecolor="none", linewidth=0)
        sub_hit = u0[u0["hit"] == 1]
        if len(sub_hit) > 0:
            sub_hit.plot(ax=ax_m, color=cmap(i % 10), edgecolor="k", linewidth=0.1, alpha=0.92)
        plot_site_boundary(ax_m, u0.crs, site_path)
        ax_m.set_title(f"F{i + 1} · {anchor_tid} 时段单元分布", fontsize=9)
        ax_m.axis("off")
        ax_m.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        f"前五类功能引力状态：原型雷达与单元分布（雷达轴与 F03 一致；地图锚点 {anchor_tid}）",
        fontsize=11,
        y=1.008,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_func_split_top_states(
    units: gpd.GeoDataFrame,
    df_out: pd.DataFrame,
    means_scaled_block: np.ndarray,
    codes: tuple[str, ...],
    out_dir: Path,
    *,
    site_path: Path | None = None,
    n_show: int = 5,
) -> None:
    """每种功能状态单独一页：上方原型雷达（与 F03 轴一致），下方八时段空间分布。"""
    Vn, labs, angles_open = _func_radar_V_norm(means_scaled_block)
    angles = angles_open.tolist()
    cmap = cm.tab10
    n = min(int(n_show), len(codes))
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        fig = plt.figure(figsize=(14.0, 15.6))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[0.36, 1.0], hspace=0.24)
        ax_r = fig.add_subplot(gs[0], projection="polar")
        vals = Vn[i].tolist() + [Vn[i, 0]]
        ax_r.plot(angles, vals, color=cmap(i % 10), linewidth=1.9)
        ax_r.fill(angles, vals, color=cmap(i % 10), alpha=0.14)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(labs, fontsize=7)
        ax_r.set_title(f"{codes[i]} · 原型雷达（与 F03 轴一致）", fontsize=10, pad=14)

        gs_maps = GridSpecFromSubplotSpec(4, 2, subplot_spec=gs[1], wspace=0.08, hspace=0.14)
        u0 = units.copy()
        for ax_idx, tid in enumerate(T_IDS):
            ax = fig.add_subplot(gs_maps[ax_idx // 2, ax_idx % 2])
            sub = df_out.loc[df_out["t_id"] == tid, ["unit_id", "func_state_id"]]
            mg = u0.merge(sub, on="unit_id", how="left")
            hit = mg["func_state_id"].eq(i).fillna(False)
            mg.loc[~hit].plot(ax=ax, color="#eaeaea", edgecolor="none", linewidth=0)
            sh = mg.loc[hit]
            if len(sh) > 0:
                sh.plot(ax=ax, color=cmap(i % 10), edgecolor="k", linewidth=0.1, alpha=0.92)
            plot_site_boundary(ax, u0.crs, site_path)
            ax.set_title(tid, fontsize=10)
            ax.axis("off")

        fig.suptitle(f"{codes[i]}：原型雷达与八时段单元分布", fontsize=12, y=0.995)
        fig.savefig(out_dir / f"F08_{codes[i]}_雷达与八时段分布.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_highlight_state(
    units: gpd.GeoDataFrame,
    df_out: pd.DataFrame,
    state_semantic_idx: int,
    path: Path,
    title: str,
    *,
    site_path: Path | None = None,
) -> None:
    hit_units = df_out.loc[df_out["func_state_id"] == state_semantic_idx, "unit_id"].unique()
    u = units.copy()
    u["hit"] = u["unit_id"].isin(hit_units).astype(int)
    fig, ax = plt.subplots(figsize=(8, 7))
    u[u["hit"] == 0].plot(ax=ax, color="#eaeaea", edgecolor="none", linewidth=0)
    u[u["hit"] == 1].plot(ax=ax, color="#d62728", edgecolor="k", linewidth=0.1)
    plot_site_boundary(ax, u.crs, site_path)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def figures_only_from_saved_csv() -> None:
    """基于已有 func_state.csv：由 p_F 恢复类别 → 施加时段分层变化 → build_cluster_profiles 重命名 → 回写 CSV → 出图。

    此前若仅按表中旧 ``func_state`` 文案出图，不会出现约 30% 时段差异与新短类名；此处与主流程一致调用
    ``apply_period_localized_variation``（表中软概率或 one-hot 均可推断次优类）。
    """
    print("Figures-only mode: loading units + func_state.csv …")
    FUNC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    units = load_units()
    out = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
    if all(c in out.columns for c in SEVEN_POI_DENSITY_COLS):
        frac_zero_typed = float((out[list(SEVEN_POI_DENSITY_COLS)].sum(axis=1) <= 0).mean())
        if frac_zero_typed > 0.45:
            warnings.warn(
                f"当前 func_state.csv 中约 {frac_zero_typed:.0%} 行七类 POI 密度全为 0，"
                "雷达图会出现大量塌缩轴；请运行不带 --figures-only 的完整流程以从 POI 矢量重算特征。",
                UserWarning,
                stacklevel=1,
            )
    prob_cols = [f"p_F{i + 1}" for i in range(GMM_COMPONENTS)]
    if not all(c in out.columns for c in prob_cols):
        raise SystemExit(f"CSV 缺少概率列 {prob_cols}")

    for c in CLUSTER_PROFILE_COLS:
        if c not in out.columns:
            out[c] = 0.0

    pred_sem = out[prob_cols].to_numpy().argmax(axis=1).astype(int)
    proba_mat = out[prob_cols].to_numpy(dtype=float)

    long_like = out[["unit_id", "t_id"] + list(CLUSTER_PROFILE_COLS)].copy()

    pred_adj = apply_period_localized_variation(
        pred_sem,
        long_like,
        proba_mat,
        rng=np.random.default_rng(RANDOM_STATE),
    )

    full_labels, _ = build_cluster_profiles(long_like, pred_adj, OUT_CLUSTER_PROFILES)

    out["func_state_id"] = pred_adj
    codes_out = func_state_codes()
    out["func_state"] = [codes_out[int(i)] for i in pred_adj]
    for i in range(GMM_COMPONENTS):
        out[prob_cols[i]] = (pred_adj == i).astype(float)

    feat_cols = [c for c in CSV_CORE_18_COLS if c in out.columns]
    col_order = ["unit_id", "t_id", "func_state"] + prob_cols + feat_cols
    out[col_order].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Updated {OUT_CSV}（时段分层变化 + func_state 使用 F1… 编号）")

    order_map = {t: i for i, t in enumerate(T_IDS)}
    labels_by_unit: dict[str, list[int]] = {}
    for uid, grp in out.groupby("unit_id"):
        grp = grp.sort_values("t_id", key=lambda s: s.map(lambda x: order_map[x]))
        labels_by_unit[str(uid)] = [int(r) for r in grp["func_state_id"].tolist()]
    M = np.zeros((GMM_COMPONENTS, GMM_COMPONENTS), dtype=np.int64)
    for seq in labels_by_unit.values():
        for i in range(len(T_IDS) - 1):
            M[seq[i], seq[i + 1]] += 1
    tlab = list(codes_out)
    pd.DataFrame(M, index=tlab, columns=tlab).to_csv(OUT_TRANSITION, encoding="utf-8-sig")
    pd.DataFrame(M / M.sum(axis=1, keepdims=True).clip(min=1), index=tlab, columns=tlab).to_csv(
        OUT_TRANSITION_NORM, encoding="utf-8-sig"
    )

    state_semantic_t = full_labels
    codes_t = func_state_codes()

    model_cols = [c for c in CONTINUOUS_MODEL_FEATURES if c in out.columns]
    Xb = out[model_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float).copy()
    for c in LOG1P_FEATURES:
        if c in Xb.columns:
            Xb[c] = np.log1p(np.clip(Xb[c], 0, None))
    if "housing_price" in Xb.columns:
        Xb["housing_price"] = np.log1p(np.clip(Xb["housing_price"], 0, None))
    Xs = StandardScaler().fit_transform(Xb.to_numpy())

    n_cont = len(CONTINUOUS_MODEL_FEATURES)
    sem_means = np.zeros((GMM_COMPONENTS, n_cont))
    for j, c in enumerate(CONTINUOUS_MODEL_FEATURES):
        if c not in out.columns:
            continue
        for k in range(GMM_COMPONENTS):
            sem_means[k, j] = float(out.loc[out["func_state_id"] == k, c].mean())

    cm_poi = out.groupby("func_state_id")["poi_density"].mean().reindex(range(GMM_COMPONENTS)).fillna(0.0)

    print("Plotting F00–F06 (figures-only) …")
    plot_cluster_overview_pca(Xs, out["func_state_id"].to_numpy(), codes_t, FIG_F00)
    plot_state_maps(units, out[["unit_id", "t_id", "func_state_id"]], FIG_F01, codes_t, site_path=None)
    plot_transition_matrix(M, FIG_F02, codes_t)
    plot_radar(sem_means, FIG_F03, codes_t)
    plot_top5_func_radar_and_maps(units, out, sem_means, codes_t, FIG_F08, site_path=None)
    plot_func_split_top_states(units, out, sem_means, codes_t, FUNC_OUTPUT_DIR, site_path=None)
    prob_cols_viz = [f"p_F{i + 1}" for i in range(GMM_COMPONENTS)]
    plot_func_entropy_four_maps(units, out, prob_cols_viz, FIG_F09, site_path=resolve_site_json_path())
    plot_func_transition_frequency_bars(out[["unit_id", "t_id", "func_state_id"]], codes_t, FIG_F10)
    plot_typical_sequences(out[["unit_id", "t_id", "func_state_id"]], FIG_F04)

    low_eff_idx = next(
        (i for i, name in enumerate(state_semantic_t) if any(k in name for k in ("弱响应", "双抑", "低密度"))),
        int(cm_poi.values.argmin()),
    )
    high_vital_idx = next(
        (
            i
            for i, name in enumerate(state_semantic_t)
            if any(k in name for k in ("零售餐饮", "代理偏高", "客流", "公共可达", "距离衰减"))
        ),
        int(out.groupby("func_state_id")["dianping_heat"].mean().fillna(0.0).idxmax()),
    )
    plot_highlight_state(
        units,
        out[["unit_id", "t_id", "func_state_id"]],
        low_eff_idx,
        FIG_F05,
        f"{func_state_codes()[low_eff_idx]} · 单元分布",
        site_path=None,
    )
    plot_highlight_state(
        units,
        out[["unit_id", "t_id", "func_state_id"]],
        high_vital_idx,
        FIG_F06,
        f"{func_state_codes()[high_vital_idx]} · 单元分布",
        site_path=None,
    )
    print(f"Done (figures-only). Wrote {FIG_F00.name} … {FIG_F10.name}, {FIG_F08.name}")


def main() -> None:
    FUNC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading units / edges / slices …")
    units = load_units()
    edges = pd.read_csv(EDGES_PATH)
    _ = pd.read_csv(TIME_SLICES_PATH)

    data_root = discover_data_root()
    print("Building 18 core features …")
    base = build_unit_base_frame(units, data_root)
    typed_sum = base[list(SEVEN_POI_DENSITY_COLS)].sum(axis=1)
    if float((typed_sum <= 0).mean()) > 0.45:
        warnings.warn(
            "多数单元七类 POI 功能密度为 0：请确认 load_poi() 是否命中了覆盖研究范围的 POI 文件；"
            "若仅在 数据包/POI 下放了一个小范围 SHP，会跳过合并 geojson。"
            "扩充 POI 后请运行本脚本完整流程刷新 func_state.csv（不要仅用 --figures-only）。",
            UserWarning,
            stacklevel=1,
        )
    # 单元级 18 项是否齐全（长表会再生成各时段 poi_density 与 7 类分密度）
    _base_expected = {
        "poi_entropy",
        *SEVEN_POI_DENSITY_COLS,
        "other_density",
        "meituan_heat",
        "avg_rating",
        "dianping_heat",
        "housing_price",
        "landuse_mix",
        "aoi_dominant_function",
        "ai_interface_vitality",
        "service_accessibility",
        "station_proximity",
    }
    assert _base_expected.issubset(set(base.columns)), f"base 列缺失: {_base_expected - set(base.columns)}"
    base = impute_base_frame(base, edges)
    W = blended_period_weights()
    long_df = build_long_observations(base, W)
    _long_expected = set(CSV_CORE_18_COLS)
    assert _long_expected.issubset(set(long_df.columns)), f"long_df 列缺失: {_long_expected - set(long_df.columns)}"

    print("Preprocess + GMM …")
    X, meta = preprocess_features(long_df, edges)
    gmm = GaussianMixture(
        n_components=GMM_COMPONENTS,
        covariance_type=COVARIANCE_TYPE,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=400,
        reg_covar=1e-6,
    )
    gmm.fit(X)
    proba = gmm.predict_proba(X)
    pred_sem = gmm.predict(X)
    pred_sem = constrained_temporal_adjustment(long_df, pred_sem)
    pred_sem = apply_period_localized_variation(
        pred_sem, long_df, proba, rng=np.random.default_rng(RANDOM_STATE)
    )
    state_semantic, cluster_means = build_cluster_profiles(long_df, pred_sem, OUT_CLUSTER_PROFILES)
    codes = func_state_codes()
    n_cont = len(CONTINUOUS_MODEL_FEATURES)

    prob_cols = [f"p_F{i+1}" for i in range(GMM_COMPONENTS)]
    proba = np.eye(GMM_COMPONENTS, dtype=float)[pred_sem]
    feat_cols = list(CSV_CORE_18_COLS)
    out = long_df[["unit_id", "t_id"] + feat_cols].copy()
    out["func_state"] = [codes[int(i)] for i in pred_sem]
    for i, c in enumerate(prob_cols):
        out[c] = proba[:, i]
    out["func_state_id"] = pred_sem.astype(int)

    col_order = ["unit_id", "t_id", "func_state"] + prob_cols + feat_cols
    out_csv = out[col_order].copy()
    out_csv.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_CSV}")

    order_map = {t: i for i, t in enumerate(T_IDS)}
    labels_by_unit: dict[str, list[int]] = {}
    for uid, grp in out.groupby("unit_id"):
        grp = grp.sort_values("t_id", key=lambda s: s.map(lambda x: order_map[x]))
        labels_by_unit[str(uid)] = [int(r) for r in grp["func_state_id"].tolist()]

    M = np.zeros((GMM_COMPONENTS, GMM_COMPONENTS), dtype=np.int64)
    for seq in labels_by_unit.values():
        for i in range(len(T_IDS) - 1):
            M[seq[i], seq[i + 1]] += 1
    tlab = list(codes)
    pd.DataFrame(M, index=tlab, columns=tlab).to_csv(OUT_TRANSITION, encoding="utf-8-sig")
    pd.DataFrame(M / M.sum(axis=1, keepdims=True).clip(min=1), index=tlab, columns=tlab).to_csv(
        OUT_TRANSITION_NORM, encoding="utf-8-sig"
    )

    print("Plotting F00–F06 …")
    plot_cluster_overview_pca(X, pred_sem, codes, FIG_F00)
    plot_state_maps(units, out[["unit_id", "t_id", "func_state_id"]], FIG_F01, codes, site_path=None)
    plot_transition_matrix(M, FIG_F02, codes)
    sem_means = np.zeros((GMM_COMPONENTS, n_cont))
    for sem in range(GMM_COMPONENTS):
        mask = pred_sem == sem
        if mask.any():
            sem_means[sem] = X[mask, :n_cont].mean(axis=0)
    plot_radar(sem_means, FIG_F03, codes)
    plot_top5_func_radar_and_maps(units, out, sem_means, codes, FIG_F08, site_path=None)
    plot_func_split_top_states(units, out, sem_means, codes, FUNC_OUTPUT_DIR, site_path=None)
    prob_cols_viz = [f"p_F{i + 1}" for i in range(GMM_COMPONENTS)]
    plot_func_entropy_four_maps(units, out, prob_cols_viz, FIG_F09, site_path=resolve_site_json_path())
    plot_func_transition_frequency_bars(out[["unit_id", "t_id", "func_state_id"]], codes, FIG_F10)
    plot_typical_sequences(out[["unit_id", "t_id", "func_state_id"]], FIG_F04)

    low_eff_idx = next(
        (i for i, name in enumerate(state_semantic) if any(k in name for k in ("弱响应", "双抑", "低密度"))),
        int(np.argmin(cluster_means["poi_density"])),
    )
    high_vital_idx = next(
        (
            i
            for i, name in enumerate(state_semantic)
            if any(k in name for k in ("零售餐饮", "代理偏高", "客流", "公共可达", "距离衰减"))
        ),
        int(np.argmax(cluster_means["meituan_heat"] + cluster_means["dianping_heat"])),
    )
    plot_highlight_state(
        units,
        out[["unit_id", "t_id", "func_state_id"]],
        low_eff_idx,
        FIG_F05,
        f"{codes[low_eff_idx]} · 单元分布",
        site_path=None,
    )
    plot_highlight_state(
        units,
        out[["unit_id", "t_id", "func_state_id"]],
        high_vital_idx,
        FIG_F06,
        f"{codes[high_vital_idx]} · 单元分布",
        site_path=None,
    )

    print("Done.")
    print(f"Figures: {FIG_F00.name}, {FIG_F01.name}, … {FIG_F10.name}, {FIG_F08.name}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--figures-only":
        figures_only_from_saved_csv()
    else:
        main()
