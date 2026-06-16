#!/usr/bin/env python3
"""
WorldPop-driven four-step synthetic flow pipeline.

Outputs:
  - clipped WorldPop raster and unit zonal statistics
  - unit_id x t_id trip-generation features
  - period-specific gravity OD (Furness doubly constrained to production / attraction priors)
  - period-specific modal OD aligned to N01/N02/N03/N04
  - optional modal assignment edge flows using flow GeoJSON-derived networks
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import unary_union

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.mask import mask as rio_mask
except ImportError as exc:  # pragma: no cover - reported at runtime
    rasterio = None
    rasterize = None
    rio_mask = None
    _RASTERIO_IMPORT_ERROR = exc
else:
    _RASTERIO_IMPORT_ERROR = None

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from synthetic_flow_od_gravity import (  # noqa: E402
    FLOW_MODAL_ASSIGN_KEYS,
    MODAL_OD_FLOW_KEYS,
    _centroids_xy_m,
    _station_kernel,
    auto_highway_fraction,
    build_modal_assignment_networks,
    furness_balance_od,
    gravity_row_normalized,
    load_period_curve_mass,
    modal_shares_softmax_period,
    run_period_chain_assignment,
)
from site_map_overlay import plot_site_boundary, resolve_site_json_path  # noqa: E402
from time_slice_constants import T_IDS, T_IDS_WEEKDAY, T_IDS_WEEKEND  # noqa: E402
from landcover_poi_proxy import GLC_POI_ZERO_EPS, aggregate_glc_landuse_by_unit  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CRS_M = "EPSG:32651"

# GLC / ESA 地类代理：见 ``landcover_poi_proxy``。POI 分量为 0 时用 glc_lu_* 参与加权（与 func 密度同量级粗匹配后再 ``_scale01``）。

PRODUCTION_PERIOD_FACTOR = {
    "WD_AM": 1.18,
    "WD_PM": 0.88,
    "WD_EVE": 1.08,
    "WD_NT": 0.62,
    "WE_AM": 0.95,
    "WE_MD": 1.02,
    "WE_EVE": 1.05,
    "WE_NT": 0.68,
}

# mob_state 若只含部分 t_id，用已有切片行回填缺失时段（避免 merge 后全 0 再 min-max 失真）
MOB_T_SLICE_FALLBACK: dict[str, str] = {
    "WD_NT": "WD_EVE",
    "WE_AM": "WD_AM",
    "WE_MD": "WD_PM",
    "WE_EVE": "WD_EVE",
    "WE_NT": "WE_MD",
}

# 吸引侧 trip_attraction：与产生侧共用同一套 POI 分量与公式，仅 ``ATTRACTION_WEIGHTS`` 不同。
# 逻辑：O/D 只靠两套权重区分——例：工作日早晨 D 偏办公/站域/交通，O 偏居住服务；傍晚 O 偏办公离开、D 偏居住到达；周末午间 D 偏零售餐饮文娱。
ATTRACTION_WEIGHTS = {
    "WD_AM": {
        "office_density": 2.55,
        "transport_service_density": 1.85,
        "station_proximity": 1.78,
        "public_service_density": 0.52,
        "food_density": 0.20,
        "retail_density": 0.20,
        "residential_service_density": 0.03,
        "heat": 0.12,
    },
    "WD_PM": {
        "office_density": 0.72,
        "transport_service_density": 0.68,
        "station_proximity": 0.68,
        "public_service_density": 0.82,
        "food_density": 1.55,
        "retail_density": 0.88,
        "residential_service_density": 0.22,
        "heat": 0.88,
    },
    "WD_EVE": {
        "office_density": 0.10,
        "transport_service_density": 0.95,
        "station_proximity": 0.88,
        "public_service_density": 0.22,
        "food_density": 1.32,
        "retail_density": 1.15,
        "residential_service_density": 1.05,
        "heat": 1.08,
    },
    "WD_NT": {
        "office_density": 0.05,
        "transport_service_density": 0.58,
        "station_proximity": 0.72,
        "public_service_density": 0.05,
        "food_density": 0.92,
        "retail_density": 0.55,
        "residential_service_density": 0.88,
        "heat": 1.05,
    },
    "WE_AM": {
        "office_density": 0.05,
        "transport_service_density": 0.92,
        "station_proximity": 0.88,
        "public_service_density": 0.72,
        "food_density": 0.78,
        "retail_density": 0.85,
        "residential_service_density": 0.38,
        "entertainment_density": 0.22,
        "heat": 0.72,
    },
    "WE_MD": {
        "office_density": 0.05,
        "transport_service_density": 0.75,
        "station_proximity": 0.78,
        "public_service_density": 0.88,
        "food_density": 1.38,
        "retail_density": 1.58,
        "residential_service_density": 0.42,
        "entertainment_density": 0.38,
        "heat": 1.28,
    },
    "WE_EVE": {
        "office_density": 0.05,
        "transport_service_density": 0.88,
        "station_proximity": 0.90,
        "public_service_density": 0.48,
        "food_density": 1.35,
        "retail_density": 1.28,
        "residential_service_density": 0.55,
        "entertainment_density": 0.72,
        "heat": 1.48,
    },
    "WE_NT": {
        "office_density": 0.05,
        "transport_service_density": 0.55,
        "station_proximity": 0.65,
        "public_service_density": 0.05,
        "food_density": 1.02,
        "retail_density": 0.58,
        "residential_service_density": 0.82,
        "entertainment_density": 0.45,
        "heat": 1.12,
    },
}

ATTRACTION_DEFAULT = copy.deepcopy(ATTRACTION_WEIGHTS["WD_PM"])

# 产生侧 trip_production：``PRODUCTION_POI_WEIGHTS`` 与吸引侧 ``ATTRACTION_WEIGHTS`` 独立设定，用 POI 类型权重表达 O 的时段角色（与 D 对照）。
PRODUCTION_POI_WEIGHTS: dict[str, dict[str, float]] = {
    "WD_AM": {
        # 居住类 POI 常为 0：出行分量里由 GLC 地类代理补全（见 ``_func_flow_components``）；站域权重保持小。
        "residential_service_density": 2.85,
        "public_service_density": 0.72,
        "office_density": 0.03,
        "transport_service_density": 0.12,
        "station_proximity": 0.05,
        "food_density": 0.10,
        "retail_density": 0.05,
        "heat": 0.03,
    },
    "WD_PM": {
        "residential_service_density": 0.16,
        "food_density": 1.52,
        "retail_density": 0.92,
        "public_service_density": 1.18,
        "office_density": 1.58,
        "transport_service_density": 0.52,
        "station_proximity": 0.24,
        "heat": 0.68,
    },
    "WD_EVE": {
        "residential_service_density": 0.04,
        "food_density": 1.35,
        "retail_density": 1.18,
        "public_service_density": 0.12,
        "office_density": 2.05,
        "transport_service_density": 1.02,
        "station_proximity": 0.78,
        "heat": 1.02,
    },
    "WD_NT": {
        "residential_service_density": 0.42,
        "public_service_density": 0.05,
        "office_density": 0.10,
        "food_density": 1.28,
        "retail_density": 0.92,
        "transport_service_density": 0.78,
        "station_proximity": 0.68,
        "heat": 1.18,
    },
    "WE_AM": {
        "residential_service_density": 1.62,
        "public_service_density": 0.58,
        "retail_density": 0.52,
        "food_density": 0.58,
        "office_density": 0.05,
        "transport_service_density": 0.35,
        "station_proximity": 0.12,
        "entertainment_density": 0.18,
        "heat": 0.32,
    },
    "WE_MD": {
        "residential_service_density": 0.28,
        "retail_density": 1.88,
        "food_density": 1.68,
        "public_service_density": 0.92,
        "office_density": 0.05,
        "transport_service_density": 0.72,
        "station_proximity": 0.58,
        "entertainment_density": 0.42,
        "heat": 1.38,
    },
    "WE_EVE": {
        "residential_service_density": 0.22,
        "food_density": 1.92,
        "retail_density": 1.68,
        "public_service_density": 0.26,
        "office_density": 0.05,
        "transport_service_density": 0.92,
        "station_proximity": 0.75,
        "entertainment_density": 0.62,
        "heat": 1.72,
    },
    "WE_NT": {
        "residential_service_density": 0.52,
        "public_service_density": 0.05,
        "office_density": 0.05,
        "food_density": 1.42,
        "retail_density": 0.98,
        "transport_service_density": 0.82,
        "station_proximity": 0.68,
        "entertainment_density": 0.48,
        "heat": 1.32,
    },
}

PRODUCTION_POI_DEFAULT = copy.deepcopy(PRODUCTION_POI_WEIGHTS["WE_MD"])

# 产生/吸引共用：POI×可达 活动臂与人口臂（``pop_base_origin``）的凸组合权重；越大越偏人口底座。
# 人口臂混合权重相对上一版再 ×0.1（进一步弱化人口基底）。
ORIGIN_POP_BASE_SHARE: dict[str, float] = {
    "WD_AM": 0.0088,
    "WD_PM": 0.0034,
    "WD_EVE": 0.001,
    "WD_NT": 0.0008,
    "WE_AM": 0.0058,
    "WE_MD": 0.0026,
    "WE_EVE": 0.0008,
    "WE_NT": 0.0006,
}

# 若为 False：产生/吸引两侧均不混入人口臂（等价于人口混合权重为 0）；``ORIGIN_POP_BASE_SHARE`` 与混合式子仍保留，改回 True 即可恢复。
ENABLE_GENERATION_POP_BASE_MIX: bool = False


def _json_default(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, Path):
        return str(v)
    return str(v)


def resolve_time_slices_path(cli_path: Path | None) -> Path | None:
    if cli_path is not None and Path(cli_path).is_file():
        return Path(cli_path)
    for p in (
        REPO / "data" / "site_3km" / "03_time_slices.csv",
        REPO / "output" / "function" / "数据包" / "03_time_slices.csv",
    ):
        if p.is_file():
            return p
    return None


def load_time_slice_catalog(csv_path: Path | None) -> tuple[list[str], dict[str, str], pd.DataFrame, tuple[tuple[str, str], ...]]:
    """从 03_time_slices.csv 读取八时段（或任意数量）t_id 与 day_type，并生成分配链顺序。"""
    path = resolve_time_slices_path(csv_path)
    if path is None:
        t_ids = list(T_IDS)
        dt_map = {**{t: "weekday" for t in T_IDS_WEEKDAY}, **{t: "weekend" for t in T_IDS_WEEKEND}}
        chain = tuple((dt_map[t], t) for t in t_ids)
        return t_ids, dt_map, pd.DataFrame(), chain

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "t_id" not in df.columns or "day_type" not in df.columns:
        raise ValueError(f"{path} 需包含列 t_id、day_type")

    df = df.copy()
    df["t_id"] = df["t_id"].astype(str).str.strip()
    df["day_type"] = df["day_type"].astype(str).str.strip().str.lower()
    d_rank = df["day_type"].map({"weekday": 0, "weekend": 1})
    if d_rank.isna().any():
        bad = sorted(df.loc[d_rank.isna(), "day_type"].astype(str).unique().tolist())
        raise ValueError(f"day_type 需为 weekday 或 weekend，发现: {bad}")
    if "hour_range_inclusive_start" in df.columns:
        df["_h"] = pd.to_numeric(df["hour_range_inclusive_start"], errors="coerce").fillna(0)
        df = df.assign(_d=d_rank).sort_values(["_d", "_h"], kind="mergesort").drop(columns=["_d", "_h"])
    else:
        df = df.assign(_d=d_rank).sort_values("_d", kind="mergesort").drop(columns=["_d"])

    if df["t_id"].duplicated().any():
        dup = df.loc[df["t_id"].duplicated(keep=False), "t_id"].astype(str).unique().tolist()
        raise ValueError(f"t_id 在表中重复: {dup}")

    t_ids = df["t_id"].tolist()
    day_type_map = dict(zip(df["t_id"], df["day_type"], strict=False))
    chain = tuple((str(r["day_type"]), str(r["t_id"])) for r in df.to_dict("records"))
    return t_ids, day_type_map, df.reset_index(drop=True), chain


def _load_units(path: Path) -> gpd.GeoDataFrame:
    try:
        u = gpd.read_file(path, layer="units")
    except Exception:
        u = gpd.read_file(path)
    if "unit_id" not in u.columns:
        raise ValueError(f"{path} 缺少 unit_id")
    if u.crs is None:
        u = u.set_crs(4326)
    return u


def _load_site_polygon(site_path: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(site_path)
    if g.crs is None:
        g = g.set_crs(4326)
    else:
        g = g.to_crs(4326)
    geoms = []
    for geom in g.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
            geoms.append(Polygon(coords))
        else:
            geoms.append(geom)
    if not geoms:
        raise ValueError(f"SITE 文件为空: {site_path}")
    poly = unary_union(geoms)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return gpd.GeoDataFrame({"name": ["site"]}, geometry=[poly], crs="EPSG:4326")


def _site_buffer_3km(site_path: Path, buffer_m: float) -> gpd.GeoDataFrame:
    site = _load_site_polygon(site_path)
    sm = site.to_crs(CRS_M)
    geom = sm.geometry.iloc[0].buffer(float(buffer_m))
    return gpd.GeoDataFrame({"name": [f"site_buffer_{int(buffer_m)}m"]}, geometry=[geom], crs=CRS_M)


def _find_worldpop_tif(raw: Path) -> Path:
    if raw.is_file():
        return raw
    if not raw.exists():
        raise FileNotFoundError(f"WorldPop 路径不存在: {raw}")
    candidates = sorted(
        [p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff"} and not p.name.lower().endswith(".ovr")]
    )
    if not candidates:
        raise FileNotFoundError(f"WorldPop 目录中未找到 tif: {raw}")
    preferred = [p for p in candidates if "ppp" in p.name.lower() and "unadj" in p.name.lower()]
    return preferred[0] if preferred else candidates[0]


def clip_worldpop(worldpop_path: Path, site_path: Path, out_dir: Path, buffer_m: float) -> tuple[Path, Path, dict[str, Any]]:
    if rasterio is None or rio_mask is None:
        raise RuntimeError(f"缺少 rasterio，无法裁剪 WorldPop: {_RASTERIO_IMPORT_ERROR}")
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer_gdf_m = _site_buffer_3km(site_path, buffer_m)
    buffer_geojson = out_dir / "SITE_buffer_3km.geojson"
    buffer_gdf_m.to_crs(4326).to_file(buffer_geojson, driver="GeoJSON", encoding="utf-8")

    out_tif = out_dir / "worldpop_3km.tif"
    with rasterio.open(worldpop_path) as src:
        mask_geom = buffer_gdf_m.to_crs(src.crs).geometry.iloc[0]
        data, transform = rio_mask(src, [mapping(mask_geom)], crop=True, all_touched=True)
        meta = src.meta.copy()
        meta.update({"driver": "GTiff", "height": int(data.shape[1]), "width": int(data.shape[2]), "transform": transform})
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(data)
        arr = data[0]
        nodata = src.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata
        valid &= arr > 0
        stats = {
            "source": str(worldpop_path),
            "output_tif": str(out_tif),
            "buffer_geojson": str(buffer_geojson),
            "buffer_m": float(buffer_m),
            "src_crs": str(src.crs),
            "src_nodata": nodata,
            "raster_interpretation": "WorldPop chn_ppp_* values are treated as persons per pixel; sums are population estimates.",
            "clipped_shape": [int(data.shape[1]), int(data.shape[2])],
            "clipped_positive_pixel_count": int(valid.sum()),
            "clipped_population_sum": float(arr[valid].sum()) if valid.any() else 0.0,
            "clipped_pixel_mean_positive": float(arr[valid].mean()) if valid.any() else 0.0,
        }
    manifest = out_dir / "worldpop_3km_manifest.json"
    manifest.write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return out_tif, buffer_geojson, stats


def zonal_worldpop(units: gpd.GeoDataFrame, raster_path: Path, out_csv: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rasterio is None or rasterize is None:
        raise RuntimeError(f"缺少 rasterio，无法做 zonal statistics: {_RASTERIO_IMPORT_ERROR}")
    units_m = units.to_crs(CRS_M)
    area_by_id = dict(zip(units["unit_id"].astype(str), units_m.geometry.area, strict=False))
    with rasterio.open(raster_path) as src:
        ur = units.to_crs(src.crs)
        nodata = src.nodata
        uid = ur["unit_id"].astype(str).tolist()
        shapes = ((geom, i + 1) for i, geom in enumerate(ur.geometry) if geom is not None and not geom.is_empty)
        zones = rasterize(
            shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="int32",
            all_touched=False,
        )
        arr = src.read(1).astype("float64", copy=False)
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata
        valid &= arr > 0
        z = zones[valid].ravel()
        v = arr[valid].ravel()
        keep = z > 0
        z = z[keep]
        v = v[keep]
        sums = np.bincount(z, weights=v, minlength=len(uid) + 1)
        counts = np.bincount(z, minlength=len(uid) + 1)

    rows: list[dict[str, Any]] = []
    for i, unit_id in enumerate(uid, start=1):
        pop_sum = float(sums[i]) if i < len(sums) else 0.0
        pix_count = int(counts[i]) if i < len(counts) else 0
        pop_mean = pop_sum / pix_count if pix_count else 0.0
        area_m2 = float(area_by_id.get(unit_id, 0.0))
        rows.append(
            {
                "unit_id": unit_id,
                "worldpop_sum": pop_sum,
                "worldpop_mean_pixel": pop_mean,
                "worldpop_pixel_count": pix_count,
                "unit_area_m2": area_m2,
                "worldpop_density_per_km2": pop_sum / max(area_m2 / 1_000_000.0, 1e-9),
            }
        )
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    stats = {
        "rows": int(len(df)),
        "units_with_positive_worldpop": int((df["worldpop_sum"] > 0).sum()),
        "worldpop_sum_by_units": float(df["worldpop_sum"].sum()),
        "worldpop_unit_max": float(df["worldpop_sum"].max()) if len(df) else 0.0,
    }
    return df, stats


def _read_csv_optional(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _scale01(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    hi = float(v.quantile(0.95)) if len(v) else 0.0
    if not np.isfinite(hi) or hi <= 1e-12:
        hi = float(v.max()) if len(v) else 0.0
    if hi <= 1e-12:
        return pd.Series(0.0, index=s.index, dtype=float)
    return (v / hi).clip(0.0, 1.5)


def _scale01_by_t_id(df: pd.DataFrame, values: pd.Series) -> pd.Series:
    """在各 ``t_id`` 内分别做 ``_scale01``（p95 按时段），突出分时段 POI 截面差异。"""
    if "t_id" not in df.columns:
        return _scale01(values)
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    tmp = pd.DataFrame({"t_id": df["t_id"].astype(str).to_numpy(), "_v": v.to_numpy(dtype=float)}, index=df.index)
    return tmp.groupby("t_id", sort=False)["_v"].transform(_scale01)


def _aggregate_static(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty or "unit_id" not in df.columns:
        return pd.DataFrame(columns=["unit_id"] + cols)
    use = ["unit_id"] + [c for c in cols if c in df.columns]
    out = df[use].copy()
    out["unit_id"] = out["unit_id"].astype(str)
    for c in use[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.groupby("unit_id", as_index=False).mean(numeric_only=True)


def _impute_mob_slice_rows(
    features: pd.DataFrame,
    mob_t: pd.DataFrame,
    t_ids: list[str],
    value_cols: list[str],
) -> dict[str, Any]:
    """用 ``MOB_T_SLICE_FALLBACK`` 把 mob_state 中缺失的 ``t_id`` 行用已有切片数值填齐。"""
    if mob_t.empty or not value_cols:
        return {"mob_slice_fallback_applied": {}, "mob_present_slices": []}
    present = sorted(mob_t["t_id"].astype(str).unique().tolist())
    applied: dict[str, str] = {}
    for tgt, src in MOB_T_SLICE_FALLBACK.items():
        if tgt not in t_ids or tgt in present:
            continue
        if src not in present:
            continue
        mask = features["t_id"].astype(str).eq(tgt)
        if not bool(mask.any()):
            continue
        src_df = mob_t[mob_t["t_id"].astype(str).eq(src)].set_index("unit_id")
        if src_df.empty:
            continue
        uidx = features.loc[mask, "unit_id"].astype(str)
        for col in value_cols:
            if col not in src_df.columns or col not in features.columns:
                continue
            mapped = uidx.map(src_df[col])
            features.loc[mask, col] = pd.to_numeric(mapped, errors="coerce").to_numpy()
        applied[str(tgt)] = str(src)
    return {"mob_slice_fallback_applied": applied, "mob_present_slices": present}


def _impute_func_slice_rows(
    features: pd.DataFrame,
    func_t: pd.DataFrame,
    t_ids: list[str],
    value_cols: list[str],
) -> dict[str, Any]:
    """func_state 若只含部分 ``t_id``，用 ``MOB_T_SLICE_FALLBACK`` 同源映射回填（与 mob 对齐）。"""
    if func_t.empty or not value_cols:
        return {"func_slice_fallback_applied": {}, "func_present_slices": []}
    present = sorted(func_t["t_id"].astype(str).unique().tolist())
    applied: dict[str, str] = {}
    for tgt, src in MOB_T_SLICE_FALLBACK.items():
        if tgt not in t_ids or tgt in present:
            continue
        if src not in present:
            continue
        mask = features["t_id"].astype(str).eq(tgt)
        if not bool(mask.any()):
            continue
        src_df = func_t[func_t["t_id"].astype(str).eq(src)].set_index("unit_id")
        if src_df.empty:
            continue
        uidx = features.loc[mask, "unit_id"].astype(str)
        for col in value_cols:
            if col not in src_df.columns or col not in features.columns:
                continue
            mapped = uidx.map(src_df[col])
            features.loc[mask, col] = pd.to_numeric(mapped, errors="coerce").to_numpy()
        applied[str(tgt)] = str(src)
    return {"func_slice_fallback_applied": applied, "func_present_slices": present}


def _mob_activity_composite_score(df: pd.DataFrame) -> np.ndarray:
    """
    使用 **mob_state 的 p_R1–p_R7**（在源数据里随 ``t_id`` 有真实结构变化，且与八时段表对齐后经
    ``MOB_T_SLICE_FALLBACK`` 回填），构造「相对该时段**全局典型运行状态**的偏离」：

    对每个 ``t_id`` 取全样本 ``p_R`` 均值的 argmax 为典型状态 ``k*(t)``，再令
    ``off_modal = 1 - p_{i,t,k*(t)}``；辅以少量归一化熵。避免主要依赖 traffic/pop 等全城近似
    同步涨跌的量，否则归一化产生量 ``p`` 的空间向量仍会高度相关。
    """
    p_cols = [f"p_R{k}" for k in range(1, 8) if f"p_R{k}" in df.columns]
    if len(p_cols) < 3:
        lt = np.log1p(np.maximum(_safe_num(df, "traffic_intensity", 0.0).to_numpy(dtype=float), 0.0))
        lp = np.log1p(np.maximum(_safe_num(df, "population_density", 0.0).to_numpy(dtype=float), 0.0))
        return (lt - lp + 5.0).astype(float)

    mus = df.groupby("t_id", sort=False)[p_cols].mean()
    mode_for_t = {str(t): int(np.argmax(mus.loc[t].to_numpy(dtype=float))) for t in mus.index.astype(str)}
    row_mode = df["t_id"].astype(str).map(mode_for_t).fillna(0).astype(int).to_numpy()
    P = np.stack([pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float) for c in p_cols], axis=1)
    srow = P.sum(axis=1, keepdims=True)
    P = np.where(srow > 1e-12, P / srow, 1.0 / float(len(p_cols)))
    on_mode = P[np.arange(P.shape[0], dtype=int), np.clip(row_mode, 0, P.shape[1] - 1)]
    off_modal = np.clip(1.0 - on_mode, 0.0, 1.0)
    ent = -(P * np.log(P + 1e-15)).sum(axis=1)
    ent_norm = np.clip(ent / max(np.log(float(P.shape[1])), 1e-9), 0.0, 1.0)
    return (0.82 * off_modal + 0.18 * ent_norm).astype(float)


def build_feature_table(
    units: gpd.GeoDataFrame,
    worldpop_df: pd.DataFrame,
    func_csv: Path,
    mob_csv: Path,
    morph_csv: Path,
    out_csv: Path,
    t_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = pd.DataFrame(
        {
            "unit_id": units["unit_id"].astype(str),
            "dist_to_station": pd.to_numeric(units.get("dist_to_station", 800.0), errors="coerce").fillna(800.0),
            "area": pd.to_numeric(units.get("area", 0.0), errors="coerce").fillna(0.0),
        }
    )
    wp = worldpop_df.copy()
    wp["unit_id"] = wp["unit_id"].astype(str)
    base = base.merge(wp, on="unit_id", how="left")
    for c in ("worldpop_sum", "worldpop_mean_pixel", "worldpop_density_per_km2"):
        base[c] = pd.to_numeric(base.get(c, 0.0), errors="coerce").fillna(0.0)

    morph = _aggregate_static(
        _read_csv_optional(morph_csv),
        ["building_coverage", "avg_height", "road_density", "barrier_index", "permeability_index", "edge_conductance_mean"],
    )
    if not morph.empty:
        base = base.merge(morph, on="unit_id", how="left")

    glc_tab, glc_meta = aggregate_glc_landuse_by_unit(units)
    base = base.merge(glc_tab, on="unit_id", how="left")

    func = _read_csv_optional(func_csv)
    if func.empty:
        func = pd.DataFrame({"unit_id": np.repeat(base["unit_id"].to_numpy(), len(t_ids)), "t_id": t_ids * len(base)})
    else:
        func["unit_id"] = func["unit_id"].astype(str)
        func["t_id"] = func["t_id"].astype(str)

    # Fill missing unit/t_id rows with each unit's mean function profile.
    grid = pd.MultiIndex.from_product([base["unit_id"].astype(str), t_ids], names=["unit_id", "t_id"]).to_frame(index=False)
    func_cols = [
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
        "housing_price",
        "station_proximity",
        "service_accessibility",
    ]
    keep = ["unit_id", "t_id"] + [c for c in func_cols if c in func.columns]
    func_t = func[keep].copy()
    for c in keep[2:]:
        func_t[c] = pd.to_numeric(func_t[c], errors="coerce")
    func_unit_mean = func_t.groupby("unit_id", as_index=False).mean(numeric_only=True)
    features = grid.merge(func_t, on=["unit_id", "t_id"], how="left")
    func_value_cols = [c for c in func_t.columns if c not in ("unit_id", "t_id")]
    func_impute_report = _impute_func_slice_rows(features, func_t, t_ids, func_value_cols)
    features = features.merge(func_unit_mean, on="unit_id", how="left", suffixes=("", "__unit_mean"))
    for c in func_cols:
        if c not in features.columns:
            features[c] = 0.0
        m = f"{c}__unit_mean"
        if m in features.columns:
            features[c] = pd.to_numeric(features[c], errors="coerce").fillna(pd.to_numeric(features[m], errors="coerce"))
            features = features.drop(columns=[m])
        features[c] = pd.to_numeric(features[c], errors="coerce").fillna(0.0)

    features = features.merge(base, on="unit_id", how="left")

    mob = _read_csv_optional(mob_csv)
    if not mob.empty:
        mob["unit_id"] = mob["unit_id"].astype(str)
        mob["t_id"] = mob["t_id"].astype(str)
        mob_cols = [
            "road_centrality",
            "accessibility_index",
            "population_density",
            "traffic_intensity",
            "congestion_proxy",
            "barrier_index",
            "bottleneck_index",
            "stay_proxy",
            "p_R1",
            "p_R2",
            "p_R3",
            "p_R4",
            "p_R5",
            "p_R6",
            "p_R7",
        ]
        keep_m = ["unit_id", "t_id"] + [c for c in mob_cols if c in mob.columns]
        mob_t = mob[keep_m].copy()
        for c in keep_m[2:]:
            mob_t[c] = pd.to_numeric(mob_t[c], errors="coerce")
        features = features.merge(mob_t, on=["unit_id", "t_id"], how="left", suffixes=("", "__mob"))
        if "barrier_index__mob" in features.columns:
            features["barrier_index"] = features["barrier_index__mob"].fillna(features.get("barrier_index", 0.0))
            features = features.drop(columns=["barrier_index__mob"])
        mob_value_cols = [c for c in mob_t.columns if c not in ("unit_id", "t_id")]
        impute_report = _impute_mob_slice_rows(features, mob_t, t_ids, mob_value_cols)
    else:
        impute_report = {"mob_slice_fallback_applied": {}, "mob_present_slices": []}

    if "accessibility_index" not in features.columns:
        features["accessibility_index"] = _safe_num(features, "service_accessibility", 0.0)
    if "transit_facility_density" not in features.columns:
        features["transit_facility_density"] = _safe_num(features, "transport_service_density", 0.0)
    if "station_attraction" not in features.columns:
        features["station_attraction"] = _safe_num(features, "station_proximity", 0.0) + _safe_num(features, "transport_service_density", 0.0)
    if "edge_conductance_mean" not in features.columns:
        features["edge_conductance_mean"] = 0.0

    for c in features.columns:
        if c not in {"unit_id", "t_id"}:
            features[c] = pd.to_numeric(features[c], errors="coerce").fillna(0.0)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_csv, index=False, encoding="utf-8-sig")
    stats = {
        "rows": int(len(features)),
        "units": int(features["unit_id"].nunique()),
        "t_ids": t_ids.copy(),
        "worldpop_positive_units": int((base["worldpop_sum"] > 0).sum()),
        "func_csv": str(func_csv) if func_csv.is_file() else None,
        "mob_csv": str(mob_csv) if mob_csv.is_file() else None,
        "morph_csv": str(morph_csv) if morph_csv.is_file() else None,
        "mob_time_impute": impute_report if mob_csv.is_file() and not mob.empty else {},
        "func_time_impute": func_impute_report,
        "glc_land_cover": glc_meta,
    }
    return features, stats


def _production_diurnal_rel(
    v: np.ndarray,
    unit_ids: np.ndarray,
    *,
    mode: str,
    clip_lo: float,
    clip_hi: float,
    power: float,
    softmax_temp: float,
) -> np.ndarray:
    """Return per-row multiplier ``rel`` (positive), same length as ``v``."""
    n = len(v)
    rel = np.ones(n, dtype=float)
    mode = str(mode)
    uid = pd.Series(unit_ids).astype(str)
    for u in uid.unique():
        m = uid.eq(u).to_numpy()
        vv = np.asarray(v[m], dtype=float)
        if vv.size == 0:
            continue
        if mode == "relative_mean":
            mu = float(np.mean(vv))
            r = vv / max(mu, 1e-12)
        elif mode == "power_mean":
            mu = float(np.mean(vv))
            r = (vv / max(mu, 1e-12)) ** float(power)
        elif mode == "minmax_unit":
            lo = float(np.min(vv))
            hi = float(np.max(vv))
            span = max(hi - lo, 1e-15)
            if hi - lo <= 1e-14:
                r = np.ones_like(vv, dtype=float)
            else:
                r = float(clip_lo) + (vv - lo) / span * (float(clip_hi) - float(clip_lo))
        elif mode == "softmax_unit":
            t = max(float(softmax_temp), 1e-6)
            s = vv / t
            s = s - float(np.max(s))
            w = np.exp(s)
            sw = float(np.sum(w))
            if sw <= 1e-30 or not np.isfinite(sw):
                r = np.ones_like(vv, dtype=float)
            else:
                r = float(vv.size) * (w / sw)
        else:
            raise ValueError(f"unknown production_diurnal mode: {mode!r}")
        rel[m] = np.clip(r, float(clip_lo), float(clip_hi))
    return rel


def apply_production_diurnal_factor(
    df: pd.DataFrame,
    col: str | None,
    *,
    mode: str = "poi_composite",
    clip_lo: float = 0.25,
    clip_hi: float = 6.5,
    power: float = 1.15,
    softmax_temp: float = 0.28,
    poi_period_scale: bool = True,
    poi_unit_mean_weight: float = 0.32,
    origin_poi_mix: float = 0.55,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    将 **unit×t_id** 分时信号乘到 ``production_raw``。

    ``poi_composite``（默认）：``PRODUCTION_POI_WEIGHTS`` 加权和；POI 分量可选 **按 ``t_id`` 内**
    ``_scale01``（``poi_period_scale``）。乘子分母为
    ``w * mean_t(score|unit) + (1-w) * median(score|t_id)``，``w`` 即 ``poi_unit_mean_weight``，
    用于减弱「仅除以单元内八段均值」带来的乘子塌缩。最终 ``production_raw`` 会在
    ``人口底座 * rel`` 与 ``时段中位人口底座 * rel`` 两个 origin 容量之间混合，避免办公/商业
    活动地永远被居住人口底座压住。

    ``mob_composite``：mob_state 的 **p_R** 偏离典型状态 + 熵（见 ``_mob_activity_composite_score``）。

    其它 ``mode`` 仍用单列 ``col`` 与 ``_production_diurnal_rel`` 构造乘子。
    """
    mode = str(mode)
    meta: dict[str, Any] = {
        "enabled": False,
        "column": col,
        "mode": mode,
        "clip": [clip_lo, clip_hi],
        "power": float(power),
        "softmax_temp": float(softmax_temp),
    }

    if mode == "poi_composite":
        out = df.copy()
        score = _production_poi_activity_score(out, period_scale=bool(poi_period_scale))
        out["production_poi_activity_score"] = score.astype(float)
        u_mean = (
            pd.Series(score, index=out.index)
            .groupby(out["unit_id"].astype(str), sort=False)
            .transform("mean")
            .to_numpy(dtype=float)
        )
        m_period = (
            pd.Series(score, index=out.index)
            .groupby(out["t_id"].astype(str), sort=False)
            .transform("median")
            .to_numpy(dtype=float)
        )
        wu = float(np.clip(poi_unit_mean_weight, 0.0, 1.0))
        den = wu * u_mean + (1.0 - wu) * np.maximum(m_period, 1e-12)
        rel = score / den
        if abs(float(power) - 1.0) > 1e-9:
            rel = np.power(np.maximum(rel, 1e-12), float(power))
        rel = np.clip(rel, float(clip_lo), float(clip_hi))
        meta["poi_composite"] = {
            "weights_table": "PRODUCTION_POI_WEIGHTS",
            "t_ids": sorted(PRODUCTION_POI_WEIGHTS.keys()),
            "poi_period_scale": bool(poi_period_scale),
            "poi_unit_mean_weight": wu,
            "denominator": "w*mean_score_by_unit + (1-w)*median_score_by_t_id",
        }
        meta["score_std_global"] = float(np.nanstd(score))
        meta["rel_std_global"] = float(np.nanstd(rel))
        meta["rel_p05"] = float(np.nanpercentile(rel, 5))
        meta["rel_p95"] = float(np.nanpercentile(rel, 95))
        out["production_diurnal_factor"] = rel.astype(float)
        pr = pd.to_numeric(out["production_raw"], errors="coerce").fillna(0.0)
        pr_med = pr.groupby(out["t_id"].astype(str), sort=False).transform(
            lambda s: float(s[s > 0].median()) if bool((s > 0).any()) else float(s.median())
        ).to_numpy(dtype=float)
        mix = float(np.clip(origin_poi_mix, 0.0, 1.0))
        population_supported = pr.to_numpy(dtype=float) * rel
        activity_place_capacity = np.maximum(pr_med, 1e-12) * rel
        out["production_raw"] = ((1.0 - mix) * population_supported + mix * activity_place_capacity).astype(float)
        meta["enabled"] = True
        meta["poi_composite"]["origin_poi_mix"] = mix
        meta["poi_composite"]["production_raw_formula"] = "(1-mix)*(population_base*rel) + mix*(period_median_population_base*rel)"
        return out, meta

    if mode == "mob_composite":
        out = df.copy()
        score = _mob_activity_composite_score(out)
        out["mob_production_activity_score"] = score.astype(float)
        u_mean = (
            pd.Series(score, index=out.index)
            .groupby(out["unit_id"].astype(str), sort=False)
            .transform("mean")
            .to_numpy(dtype=float)
        )
        rel = score / np.maximum(u_mean, 1e-12)
        if abs(float(power) - 1.0) > 1e-9:
            rel = np.power(np.maximum(rel, 1e-12), float(power))
        rel = np.clip(rel, float(clip_lo), float(clip_hi))
        meta["mob_composite"] = {
            "definition": "off_modal_from_typical_p_R_plus_entropy",
            "off_modal_weight": 0.82,
            "entropy_norm_weight": 0.18,
        }
        meta["score_std_global"] = float(np.nanstd(score))
        meta["rel_std_global"] = float(np.nanstd(rel))
        meta["rel_p05"] = float(np.nanpercentile(rel, 5))
        meta["rel_p95"] = float(np.nanpercentile(rel, 95))
        out["production_diurnal_factor"] = rel.astype(float)
        pr = pd.to_numeric(out["production_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        out["production_raw"] = (pr * rel).astype(float)
        meta["enabled"] = True
        return out, meta

    if not col or col not in df.columns:
        meta["reason"] = "column_missing_or_disabled"
        return df, meta

    out = df.copy()
    v = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    unit_ids = out["unit_id"].astype(str).to_numpy()
    rel = _production_diurnal_rel(
        v,
        unit_ids,
        mode=str(mode),
        clip_lo=float(clip_lo),
        clip_hi=float(clip_hi),
        power=float(power),
        softmax_temp=float(softmax_temp),
    )
    meta["rel_std_global"] = float(np.nanstd(rel))
    meta["rel_p05"] = float(np.nanpercentile(rel, 5))
    meta["rel_p95"] = float(np.nanpercentile(rel, 95))
    out["production_diurnal_factor"] = rel
    pr = pd.to_numeric(out["production_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["production_raw"] = (pr * rel).astype(float)
    meta["enabled"] = True
    return out, meta


def _func_flow_components(df: pd.DataFrame, *, poi_period_scale: bool = False) -> tuple[dict[str, pd.Series], pd.Series]:
    """POI 分量：全局 ``_scale01``。某类 POI 在单元内为 0 时用 GLC 地类代理列补全再缩放（见 ``aggregate_glc_landuse_by_unit``）。"""
    heat = np.log1p(_safe_num(df, "dianping_heat", 0.0).clip(lower=0.0) + _safe_num(df, "meituan_heat", 0.0).clip(lower=0.0))
    heat_s = pd.Series(heat, index=df.index, copy=False)

    def sc(raw: pd.Series) -> pd.Series:
        if poi_period_scale:
            return _scale01_by_t_id(df, raw)
        return _scale01(raw)

    def nz_poi(poi_col: str, glc_col: str) -> pd.Series:
        p = _safe_num(df, poi_col, 0.0).to_numpy(dtype=float)
        g = _safe_num(df, glc_col, 0.0).to_numpy(dtype=float)
        v = np.where(p > GLC_POI_ZERO_EPS, p, g)
        return pd.Series(v, index=df.index, dtype=float)

    station_raw = _safe_num(df, "station_proximity", 0.0) + np.exp(-_safe_num(df, "dist_to_station", 800.0) / 900.0)
    comps: dict[str, pd.Series] = {
        "office_density": sc(nz_poi("office_density", "glc_lu_workplace_proxy")),
        "retail_density": sc(nz_poi("retail_density", "glc_lu_retail_proxy")),
        "food_density": sc(nz_poi("food_density", "glc_lu_food_proxy")),
        "public_service_density": sc(_safe_num(df, "public_service_density", 0.0)),
        "transport_service_density": sc(nz_poi("transport_service_density", "glc_lu_workplace_proxy")),
        "residential_service_density": sc(nz_poi("residential_service_density", "glc_lu_residential_proxy")),
        "entertainment_density": sc(nz_poi("entertainment_density", "glc_lu_leisure_proxy")),
        "station_proximity": sc(station_raw),
        "heat": sc(heat_s),
    }
    return comps, heat_s


def _poi_weighted_series(
    df: pd.DataFrame,
    components: dict[str, pd.Series],
    weights_by_tid: dict[str, dict[str, float]],
    default_weights: dict[str, float],
) -> pd.Series:
    """``0.04 + Σ w_k·component_k``，按行 ``t_id`` 选用对应权重表。"""
    score = pd.Series(0.04, index=df.index, dtype=float)
    for tid in sorted(df["t_id"].unique()):
        wmap = weights_by_tid.get(str(tid), default_weights)
        mask = df["t_id"].eq(tid)
        local = pd.Series(0.04, index=df.index, dtype=float)
        for name, weight in wmap.items():
            if name not in components:
                continue
            local = local + float(weight) * components[name]
        score.loc[mask] = local.loc[mask]
    return score


def _production_poi_activity_score(df: pd.DataFrame, *, period_scale: bool = True) -> np.ndarray:
    """分时段 POI 加权和（``PRODUCTION_POI_WEIGHTS``）；``period_scale`` 为真时各分量按 ``t_id`` 内 ``_scale01``。"""
    components, _ = _func_flow_components(df, poi_period_scale=period_scale)
    return _poi_weighted_series(df, components, PRODUCTION_POI_WEIGHTS, PRODUCTION_POI_DEFAULT).to_numpy(dtype=float)


def _rescale_like_by_t_id(df: pd.DataFrame, source: pd.Series, reference: pd.Series) -> np.ndarray:
    """
    按 ``t_id`` 将 ``source`` 重标定到 ``reference`` 的中位数尺度，避免两类基底量纲不一致。
    """
    tids = df["t_id"].astype(str)
    out = pd.Series(0.0, index=df.index, dtype=float)
    s = pd.to_numeric(source, errors="coerce").fillna(0.0)
    r = pd.to_numeric(reference, errors="coerce").fillna(0.0)
    for tid in tids.unique():
        m = tids.eq(str(tid))
        ss = s.loc[m]
        rr = r.loc[m]
        s_med = float(ss[ss > 0].median()) if bool((ss > 0).any()) else float(ss.median())
        r_med = float(rr[rr > 0].median()) if bool((rr > 0).any()) else float(rr.median())
        k = r_med / max(s_med, 1e-12)
        out.loc[m] = ss * k
    return out.to_numpy(dtype=float)


def add_generation_scores(features: pd.DataFrame, station_sigma_m: float, station_weight: float, station_kernel_strength: float) -> pd.DataFrame:
    df = features.copy()
    worldpop = _safe_num(df, "worldpop_sum", 0.0)
    if float(worldpop.sum()) <= 1e-12:
        area = _safe_num(df, "area", 1.0).clip(lower=1.0)
        worldpop = area / max(float(area.sum()), 1e-12)
    pop_floor = max(float(worldpop[worldpop > 0].median()) if (worldpop > 0).any() else 1.0, 1e-6) * 0.002

    dist = _safe_num(df, "dist_to_station", 800.0).to_numpy(dtype=float)
    k = _station_kernel(dist, station_sigma_m, station_weight)
    station_factor = 1.0 + float(station_kernel_strength) * (k - 1.0)

    morph_factor = (
        0.82
        + 0.22 * _scale01(_safe_num(df, "building_coverage", 0.0))
        + 0.14 * _scale01(_safe_num(df, "avg_height", 0.0))
        + 0.10 * _scale01(_safe_num(df, "edge_conductance_mean", 0.0))
        - 0.08 * _scale01(_safe_num(df, "barrier_index", 0.0))
    ).clip(0.55, 1.35)

    df["home_activity_period_factor"] = df["t_id"].map(PRODUCTION_PERIOD_FACTOR).fillna(1.0).astype(float)
    df["station_kernel_factor"] = station_factor
    df["morphology_sanity_factor"] = morph_factor

    pop_base = (worldpop + pop_floor) * df["station_kernel_factor"] * df["morphology_sanity_factor"]
    pop_base_origin = pop_base * df["home_activity_period_factor"]

    # 产生/吸引：同一 POI 分量与 access_factor，仅 ``PRODUCTION_POI_WEIGHTS`` / ``ATTRACTION_WEIGHTS`` 不同以区分 O/D；
    # 再与 ``pop_base_origin`` 做中位尺度对齐（``ENABLE_GENERATION_POP_BASE_MIX`` 关闭时不混入人口臂）。
    components, heat_raw = _func_flow_components(df, poi_period_scale=False)
    df["heat"] = heat_raw
    prod_sum = _poi_weighted_series(df, components, PRODUCTION_POI_WEIGHTS, PRODUCTION_POI_DEFAULT)
    attr_sum = _poi_weighted_series(df, components, ATTRACTION_WEIGHTS, ATTRACTION_DEFAULT)
    access_factor = 0.82 + 0.38 * _scale01(_safe_num(df, "accessibility_index", 0.0))
    af = access_factor.clip(0.75, 1.3)
    prod_poi = (prod_sum * af).clip(lower=1e-12)
    attr_poi = (attr_sum * af).clip(lower=1e-12)
    prod_scaled = _rescale_like_by_t_id(df, prod_poi, pop_base_origin)
    attr_scaled = _rescale_like_by_t_id(df, attr_poi, pop_base_origin)
    if ENABLE_GENERATION_POP_BASE_MIX:
        origin_pop_share = (
            df["t_id"].astype(str).map(ORIGIN_POP_BASE_SHARE).fillna(0.5).clip(0.0, 1.0).to_numpy(dtype=float)
        )
    else:
        origin_pop_share = np.zeros(len(df), dtype=float)
    pop_ref = pop_base_origin.to_numpy(dtype=float)
    df["production_raw"] = origin_pop_share * pop_ref + (1.0 - origin_pop_share) * prod_scaled
    df["attraction_raw"] = origin_pop_share * pop_ref + (1.0 - origin_pop_share) * attr_scaled
    df["attraction_raw"] = pd.to_numeric(df["attraction_raw"], errors="coerce").fillna(1e-9).clip(lower=1e-9)
    df["production_raw"] = pd.to_numeric(df["production_raw"], errors="coerce").fillna(1e-9).clip(lower=1e-9)

    comp_record = {str(t): dict(ATTRACTION_WEIGHTS.get(str(t), ATTRACTION_DEFAULT)) for t in sorted(df["t_id"].unique())}
    df.attrs["attraction_weights"] = comp_record
    df.attrs["production_poi_weights"] = {
        str(t): dict(PRODUCTION_POI_WEIGHTS.get(str(t), PRODUCTION_POI_DEFAULT)) for t in sorted(df["t_id"].unique())
    }
    df.attrs["trip_generation_od_weights_note"] = (
        "trip_production uses PRODUCTION_POI_WEIGHTS; trip_attraction uses ATTRACTION_WEIGHTS "
        "(same components, different tables per t_id). Where a POI density is at or below "
        f"{GLC_POI_ZERO_EPS}, trip-generation scoring substitutes GLC land-cover proxies merged from "
        "data/site_3km/09-开源土地利用/开源土地利用/GLC (not WorldPop)."
    )
    df.attrs["origin_pop_base_share"] = {str(t): float(v) for t, v in ORIGIN_POP_BASE_SHARE.items()}
    df.attrs["generation_pop_base_mix_enabled"] = bool(ENABLE_GENERATION_POP_BASE_MIX)
    return df


def _top_od_rows(od: np.ndarray, uid: list[str], dist_m: np.ndarray, cap: int) -> list[dict[str, Any]]:
    flat = od.ravel()
    n = od.shape[0]
    positive = int((flat > 0).sum())
    if positive == 0:
        return []
    k = min(int(cap), positive)
    if k <= 0:
        return []
    if k < flat.size:
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]
    else:
        idx = np.argsort(-flat)
    rows = []
    for ix in idx:
        val = float(flat[int(ix)])
        if val <= 0:
            break
        i, j = divmod(int(ix), n)
        if i == j:
            continue
        rows.append({"origin_id": uid[i], "destination_id": uid[j], "flow": val, "dist_m": float(dist_m[i, j])})
    return rows


def generate_period_od(
    units: gpd.GeoDataFrame,
    features_scored: pd.DataFrame,
    out_dir: Path,
    *,
    t_ids: list[str],
    day_type_by_tid: dict[str, str],
    total_trips: float,
    beta: float,
    d_floor_m: float,
    transfer_fraction: float,
    station_band_m: float,
    metroflow_calibration_json: Path,
    period_od_max_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """分时段：产约束重力得 ``seed``，再 **Furness 双约束** 使行和 ``= p·T``、列和 ``= a·T``（``p``、``a`` 为 ``production_raw`` / ``attraction_raw`` 归一化向量）。"""
    uid = units["unit_id"].astype(str).tolist()
    uid_index = {u: i for i, u in enumerate(uid)}
    xs, ys, _ = _centroids_xy_m(units)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_m = np.sqrt(dx * dx + dy * dy)
    d_sta = pd.to_numeric(units.get("dist_to_station", 800.0), errors="coerce").fillna(800.0).to_numpy(dtype=float)

    wm_wd = load_period_curve_mass(metroflow_calibration_json, day_key="weekday")
    wm_we = load_period_curve_mass(metroflow_calibration_json, day_key="weekend")
    period_weights = {**wm_wd, **wm_we}

    raw = np.array([float(period_weights.get(t, 0.0)) for t in t_ids], dtype=float)
    if raw.sum() < 1e-12:
        raw = np.ones(len(t_ids), dtype=float) / max(len(t_ids), 1)
    else:
        bad = raw <= 0
        if bad.any():
            good_sum = float(raw[~bad].sum())
            n_bad = int(bad.sum())
            if good_sum < 1e-12:
                raw = np.ones(len(t_ids), dtype=float) / max(len(t_ids), 1)
            else:
                rem = max(0.0, 1.0 - good_sum)
                raw[bad] = rem / max(n_bad, 1)
        raw = raw / max(float(raw.sum()), 1e-12)
    mass_by_t = {t_ids[i]: float(raw[i]) for i in range(len(t_ids))}

    gen_rows: list[dict[str, Any]] = []
    od_rows: list[dict[str, Any]] = []
    cap_per_period = max(1, int(period_od_max_rows))
    tf = float(np.clip(transfer_fraction, 0.0, 0.95))

    f_by_t = {str(t): g.copy() for t, g in features_scored.groupby("t_id", sort=False)}
    furness_by_tid: dict[str, Any] = {}
    for t_id in t_ids:
        ft = f_by_t.get(t_id)
        if ft is None or ft.empty:
            continue
        ft = ft.copy()
        ft["__ix"] = ft["unit_id"].map(uid_index)
        ft = ft.dropna(subset=["__ix"]).sort_values("__ix")
        prod_raw = _safe_num(ft, "production_raw", 0.0).to_numpy(dtype=float)
        attr_raw = _safe_num(ft, "attraction_raw", 0.0).to_numpy(dtype=float)
        p = prod_raw / max(float(prod_raw.sum()), 1e-12)
        a = attr_raw / max(float(attr_raw.sum()), 1e-12)
        od_base = gravity_row_normalized(p, a, dist_m, beta=beta, d_floor_m=d_floor_m)

        ox = np.where(d_sta <= float(station_band_m), p, 0.0)
        if float(ox.sum()) <= 1e-12:
            ox = p.copy()
        else:
            ox = ox / float(ox.sum())
        od_xfer = gravity_row_normalized(ox, a, dist_m, beta=float(beta) * 0.92, d_floor_m=d_floor_m)
        period_total = float(total_trips) * float(mass_by_t.get(t_id, 1.0 / max(len(t_ids), 1)))
        seed = ((1.0 - tf) * od_base + tf * od_xfer) * period_total
        row_tgt = p * period_total
        col_tgt = a * period_total
        od, f_meta = furness_balance_od(seed, row_tgt, col_tgt)
        furness_by_tid[str(t_id)] = f_meta

        row_mass = od.sum(axis=1)
        col_mass = od.sum(axis=0)
        for i, unit_id in enumerate(uid):
            gen_rows.append(
                {
                    "unit_id": unit_id,
                    "day_type": day_type_by_tid.get(t_id, "weekday"),
                    "t_id": t_id,
                    "period_mass_share": float(mass_by_t.get(t_id, 0.0)),
                    "prior_production": float(p[i] * period_total),
                    "prior_attraction": float(a[i] * period_total),
                    "trip_production": float(row_mass[i]),
                    "trip_attraction": float(col_mass[i]),
                    "production_raw": float(prod_raw[i]),
                    "attraction_raw": float(attr_raw[i]),
                }
            )

        for r in _top_od_rows(od, uid, dist_m, cap_per_period):
            r["day_type"] = day_type_by_tid.get(t_id, "weekday")
            r["t_id"] = t_id
            od_rows.append(r)

    gen = pd.DataFrame(gen_rows)
    od_long = pd.DataFrame(od_rows)
    gen.to_csv(out_dir / "trip_generation_by_period.csv", index=False, encoding="utf-8-sig")
    od_long.to_csv(out_dir / "synthetic_od_by_period_long.csv", index=False, encoding="utf-8-sig")
    full_period_sums = gen.groupby("t_id")["trip_production"].sum().to_dict() if len(gen) else {}
    retained_period_sums = od_long.groupby("t_id")["flow"].sum().to_dict() if len(od_long) else {}
    retained_share = {
        t: float(retained_period_sums.get(t, 0.0)) / max(float(full_period_sums.get(t, 0.0)), 1e-12)
        for t in sorted(set(full_period_sums) | set(retained_period_sums))
    }
    stats = {
        "period_od_rows": int(len(od_long)),
        "period_generation_rows": int(len(gen)),
        "period_od_max_rows_per_t_id": int(cap_per_period),
        "full_period_trip_generation_sums": full_period_sums,
        "retained_top_od_flow_sums": retained_period_sums,
        "retained_top_od_share": retained_share,
        "period_weights_raw_merged": period_weights,
        "period_mass_by_t_id_normalized": mass_by_t,
        "furness_doubly_constrained_by_t_id": furness_by_tid,
    }
    return gen, od_long, stats


def build_modal_period_od(features: pd.DataFrame, od_period: pd.DataFrame, out_csv: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if od_period.empty:
        out = pd.DataFrame()
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        return out, {"modal_od_rows": 0}
    f_ix = features.set_index(["unit_id", "t_id"], drop=False)
    fallback = pd.Series(dtype=float)
    rows: list[dict[str, Any]] = []
    for row in od_period.itertuples(index=False):
        o, d, t_id = str(row.origin_id), str(row.destination_id), str(row.t_id)
        ro = f_ix.loc[(o, t_id)] if (o, t_id) in f_ix.index else fallback
        rd = f_ix.loc[(d, t_id)] if (d, t_id) in f_ix.index else fallback
        if isinstance(ro, pd.DataFrame):
            ro = ro.iloc[0]
        if isinstance(rd, pd.DataFrame):
            rd = rd.iloc[0]
        dm = float(row.dist_m)
        sw, sb, st, sa = modal_shares_softmax_period(dm, ro, rd, t_id)
        ff, fs = auto_highway_fraction(dm)
        fv = float(row.flow)
        rows.append(
            {
                "day_type": str(row.day_type),
                "t_id": t_id,
                "origin_id": o,
                "destination_id": d,
                "dist_m": dm,
                "flow": fv,
                "share_walk": sw,
                "share_bike": sb,
                "share_N01_pedestrian": sw + sb,
                "share_transit": st,
                "share_N04_transit_proxy": st,
                "share_auto": sa,
                "share_auto_fast": sa * ff,
                "share_auto_slow": sa * fs,
                "flow_walk": fv * sw,
                "flow_bike": fv * sb,
                "flow_N01_pedestrian": fv * (sw + sb),
                "flow_transit": fv * st,
                "flow_N04_transit_proxy": fv * st,
                "flow_auto": fv * sa,
                "flow_N02_fast_auto": fv * sa * ff,
                "flow_N03_slow_auto": fv * sa * fs,
            }
        )
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    stats = {
        "modal_od_rows": int(len(out)),
        "modal_flow_sums": {
            c: float(out[c].sum())
            for c in ["flow_N01_pedestrian", "flow_N02_fast_auto", "flow_N03_slow_auto", "flow_N04_transit_proxy"]
            if c in out.columns
        },
    }
    return out, stats


def maybe_build_assignment_edges(ns: argparse.Namespace, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    out_csv = Path(ns.assignment_edges_csv) if ns.assignment_edges_csv else out_dir / "flow_road_assignment_edges.csv"
    if out_csv.is_file() and not ns.force_rebuild_assignment_edges:
        return out_csv, {"assignment_edges_csv": str(out_csv), "rebuilt": False}
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "build_flow_road_assignment_edges.py"),
        "--units",
        str(ns.units),
        "--data-root",
        str(ns.data_root),
        "--site-buffer-m",
        str(ns.site_buffer_m),
        "--out-csv",
        str(out_csv),
    ]
    if ns.site_json:
        cmd.extend(["--site-json", str(ns.site_json)])
    subprocess.run(cmd, cwd=REPO, check=True)
    return out_csv, {"assignment_edges_csv": str(out_csv), "rebuilt": True, "command": cmd}


def assign_period_edges(
    modal_od: pd.DataFrame,
    assignment_edges_csv: Path,
    out_dir: Path,
    *,
    max_origins: int,
    n_iters: int,
    delay_alpha: float,
    delay_power: float,
    scheme: str,
    period_chain_order: tuple[tuple[str, str], ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if modal_od.empty:
        out = pd.DataFrame(columns=["day_type", "t_id", "modality", "source_id", "target_id", "flow_aon"])
        out.to_csv(out_dir / "synthetic_edge_flow_period_long.csv", index=False, encoding="utf-8-sig")
        return out, {"period_edge_assignment_long_rows": 0}
    edges = pd.read_csv(assignment_edges_csv, encoding="utf-8-sig")
    nets = build_modal_assignment_networks(edges)
    per_rows = modal_od.to_dict("records")
    long_df, stats = run_period_chain_assignment(
        nets,
        per_rows,
        max_origins=int(max_origins),
        n_iters=int(n_iters),
        delay_alpha=float(delay_alpha),
        delay_power=float(delay_power),
        scheme=str(scheme),
        period_chain_order=period_chain_order,
    )
    long_df.to_csv(out_dir / "synthetic_edge_flow_period_long.csv", index=False, encoding="utf-8-sig")
    if not long_df.empty:
        four = long_df.copy()
        four["modality"] = four["modality"].replace({"N01_bike": "N01_pedestrian"})
        four = four.groupby(["day_type", "t_id", "modality", "source_id", "target_id"], as_index=False)["flow_aon"].sum()
    else:
        four = long_df.copy()
    four.to_csv(out_dir / "synthetic_edge_flow_period_long_4class.csv", index=False, encoding="utf-8-sig")
    stats = {
        **stats,
        "assignment_edges_csv": str(assignment_edges_csv),
        "assignment_edge_rows": int(len(edges)),
        "four_class_edge_rows": int(len(four)),
        "assignment_modalities": sorted(long_df["modality"].unique().tolist()) if not long_df.empty else [],
    }
    return long_df, stats


def plot_generation_quick(
    units: gpd.GeoDataFrame,
    gen: pd.DataFrame,
    out_path: Path,
    slices_catalog: pd.DataFrame,
    site_path: Path | None = None,
) -> None:
    if gen.empty:
        return
    from matplotlib import font_manager

    preferred = ("Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS")
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in avail:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False

    if len(slices_catalog) and "t_id" in slices_catalog.columns:
        t_order = slices_catalog["t_id"].astype(str).tolist()
    else:
        t_order = []
        for dt in ("weekday", "weekend"):
            part = gen.loc[gen["day_type"].astype(str).str.lower().eq(dt), "t_id"].astype(str)
            for t in part.unique():
                if t not in t_order:
                    t_order.append(t)
        for t in sorted(gen["t_id"].unique()):
            if t not in t_order:
                t_order.append(str(t))
    t_order = [t for t in t_order if t in set(gen["t_id"].astype(str))]

    tname: dict[str, str] = {}
    if len(slices_catalog) and "t_name" in slices_catalog.columns and "t_id" in slices_catalog.columns:
        tname = dict(zip(slices_catalog["t_id"].astype(str), slices_catalog["t_name"].astype(str), strict=False))

    prod_all = pd.to_numeric(gen["trip_production"], errors="coerce").to_numpy(dtype=float)
    attr_all = pd.to_numeric(gen["trip_attraction"], errors="coerce").to_numpy(dtype=float)
    vmax_p = float(np.nanpercentile(prod_all, 98)) if np.isfinite(prod_all).any() else 1.0
    vmax_a = float(np.nanpercentile(attr_all, 98)) if np.isfinite(attr_all).any() else 1.0
    vmax_p = max(vmax_p, 1e-12)
    vmax_a = max(vmax_a, 1e-12)

    n = len(t_order)
    fig_h = max(2.15 * n, 6.0)
    fig, axes = plt.subplots(n, 2, figsize=(14.5, fig_h))
    if n == 1:
        axes = np.asarray([axes])
    sp = site_path if site_path is not None and Path(site_path).is_file() else resolve_site_json_path()
    for r, t_id in enumerate(t_order):
        sub = gen[gen["t_id"].astype(str).eq(str(t_id))][["unit_id", "trip_production", "trip_attraction"]]
        mg = units.merge(sub, on="unit_id", how="left")
        title_base = tname.get(str(t_id), str(t_id))
        for c, col in enumerate(["trip_production", "trip_attraction"]):
            ax = axes[r, c]
            vmax = vmax_p if c == 0 else vmax_a
            mg.plot(
                column=col,
                ax=ax,
                cmap="Blues" if c == 0 else "YlOrRd",
                linewidth=0.03,
                edgecolor="0.55",
                vmin=0.0,
                vmax=vmax,
                legend=False,
                missing_kwds={"color": "#e8e8e8"},
            )
            plot_site_boundary(ax, mg.crs, sp)
            ax.set_title(f"{title_base}\n{col}", fontsize=10)
            ax.axis("off")
    fig.suptitle(
        "出行生成：八时段 trip_production / trip_attraction（Furness 双约束对齐 p·T 与 a·T；色标跨时段统一 p98；O/D 先验来自两套分时段 POI 权重）",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="WorldPop + POI/AOI four-step period flow generation")
    ap.add_argument("--worldpop", type=Path, default=REPO.parent / "all" / "15-Worldpop人口密度")
    ap.add_argument("--site-json", type=Path, default=REPO / "data" / "site_3km" / "SITE.json")
    ap.add_argument("--units", type=Path, default=REPO / "output" / "function" / "数据包" / "01_units.gpkg")
    ap.add_argument("--func-csv", type=Path, default=REPO / "output" / "function" / "数据包" / "func_state.csv")
    ap.add_argument("--mob-csv", type=Path, default=REPO / "output" / "flow" / "output_mobility_state" / "mob_state.csv")
    ap.add_argument("--morph-csv", type=Path, default=REPO / "output" / "form" / "morph_state.csv")
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "site_3km")
    ap.add_argument("--out-dir", type=Path, default=REPO / "output" / "synthetic_flow_worldpop")
    ap.add_argument("--demography-dir", type=Path, default=REPO / "data" / "site_3km" / "13_demography")
    ap.add_argument("--metroflow-calibration-json", type=Path, default=REPO / "data" / "site_3km" / "metroflow" / "time_slice_calibration.json")
    ap.add_argument("--assignment-edges-csv", type=Path, default=None)
    ap.add_argument("--force-rebuild-assignment-edges", action="store_true")
    ap.add_argument("--skip-assignment", action="store_true")
    ap.add_argument("--buffer-m", type=float, default=3000.0)
    ap.add_argument("--site-buffer-m", type=float, default=3000.0)
    ap.add_argument("--total-trips", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--d-floor-m", type=float, default=25.0)
    ap.add_argument("--station-sigma-m", type=float, default=1650.0)
    ap.add_argument("--station-weight", type=float, default=1.15)
    ap.add_argument("--station-kernel-strength", type=float, default=0.35)
    ap.add_argument("--transfer-fraction", type=float, default=0.12)
    ap.add_argument("--station-band-m", type=float, default=400.0)
    ap.add_argument("--period-od-max-rows", type=int, default=20000, help="Top OD rows retained per t_id")
    ap.add_argument("--aon-max-origins", type=int, default=0)
    ap.add_argument("--assignment-iters", type=int, default=5)
    ap.add_argument("--assignment-delay-alpha", type=float, default=0.22)
    ap.add_argument("--assignment-delay-power", type=float, default=2.0)
    ap.add_argument("--assignment-scheme", choices=("frank_wolfe", "aon_replace"), default="frank_wolfe")
    ap.add_argument(
        "--time-slices-csv",
        type=Path,
        default=None,
        help="时段定义表（默认自动查找 data/site_3km/03_time_slices.csv 或 output/function/数据包/03_time_slices.csv）；需含 t_id、day_type",
    )
    ap.add_argument(
        "--production-diurnal-col",
        type=str,
        default="population_density",
        help="单列模式（非 poi_composite / mob_composite）下使用的列名。",
    )
    ap.add_argument(
        "--production-diurnal-mode",
        choices=("poi_composite", "mob_composite", "relative_mean", "power_mean", "minmax_unit", "softmax_unit"),
        default="poi_composite",
        help="poi_composite=分时段 POI 加权和（PRODUCTION_POI_WEIGHTS，默认）；mob_composite=mob p_R 偏离+熵；其它为单列构造。",
    )
    ap.add_argument("--production-diurnal-clip-lo", type=float, default=0.25, help="乘子下限")
    ap.add_argument("--production-diurnal-clip-hi", type=float, default=6.5, help="乘子上限")
    ap.add_argument(
        "--production-diurnal-power",
        type=float,
        default=1.15,
        help="poi/mob_composite：对 (score/分母) 的指数；分母见 --production-poi-unit-mean-weight；单列 power_mean 亦用此指数。",
    )
    ap.add_argument(
        "--no-production-poi-period-scale",
        action="store_true",
        help="poi_composite：不按 t_id 内做 _scale01（与吸引侧相同的全局 POI 缩放）。",
    )
    ap.add_argument(
        "--production-poi-unit-mean-weight",
        type=float,
        default=0.32,
        help="poi_composite 乘子分母：w·mean(score|unit)+(1-w)·median(score|t_id) 中的 w；越小越弱化单元内八段归一。",
    )
    ap.add_argument(
        "--production-origin-poi-mix",
        type=float,
        default=0.55,
        help="poi_composite 下 production_raw 中活动地 POI 容量的混合比例；越大越弱化 WorldPop 居住人口底座。",
    )
    ap.add_argument("--production-diurnal-softmax-temp", type=float, default=0.28, help="softmax_unit 的温度，越小高峰时段越突出")
    ap.add_argument(
        "--no-production-diurnal",
        action="store_true",
        help="关闭产生量分时乘子（恢复仅 WorldPop+站核+形态+标量时段因子）。",
    )
    ns = ap.parse_args()

    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ns.site_json.is_file() and (REPO / "data" / "SITE.json").is_file():
        ns.site_json = REPO / "data" / "SITE.json"

    t_ids, day_type_map, slices_df, chain_order = load_time_slice_catalog(ns.time_slices_csv)
    ts_resolved = resolve_time_slices_path(ns.time_slices_csv)

    meta: dict[str, Any] = {
        "method": "worldpop_four_step_period_generation_modal_assignment",
        "inputs": {
            "worldpop": str(ns.worldpop),
            "site_json": str(ns.site_json),
            "units": str(ns.units),
            "func_csv": str(ns.func_csv),
            "mob_csv": str(ns.mob_csv),
            "morph_csv": str(ns.morph_csv),
            "metroflow_calibration_json": str(ns.metroflow_calibration_json),
            "time_slices_csv_resolved": str(ts_resolved) if ts_resolved else None,
            "time_slices_csv_cli": str(ns.time_slices_csv) if ns.time_slices_csv else None,
        },
        "parameters": vars(ns).copy(),
        "time_slices": {
            "t_ids": list(t_ids),
            "day_type_by_t_id": day_type_map,
            "assignment_period_chain": [list(x) for x in chain_order],
            "catalog_rows": slices_df.to_dict("records") if len(slices_df) else [],
        },
    }

    units = _load_units(ns.units)
    worldpop_tif = _find_worldpop_tif(Path(ns.worldpop))
    clipped_tif, buffer_geojson, clip_stats = clip_worldpop(worldpop_tif, ns.site_json, ns.demography_dir, float(ns.buffer_m))
    worldpop_by_unit, zonal_stats = zonal_worldpop(units, clipped_tif, ns.demography_dir / "worldpop_by_unit.csv")
    meta["worldpop_clip"] = clip_stats
    meta["worldpop_zonal"] = zonal_stats

    feature_csv = out_dir / "trip_generation_unit_features_by_period.csv"
    features, feature_stats = build_feature_table(
        units, worldpop_by_unit, ns.func_csv, ns.mob_csv, ns.morph_csv, feature_csv, t_ids
    )
    features = add_generation_scores(features, ns.station_sigma_m, ns.station_weight, ns.station_kernel_strength)
    if ns.no_production_diurnal:
        meta["production_diurnal"] = {"enabled": False, "reason": "no_production_diurnal"}
        features["production_diurnal_factor"] = 1.0
    elif str(ns.production_diurnal_mode) == "poi_composite":
        features, diurnal_meta = apply_production_diurnal_factor(
            features,
            None,
            mode="poi_composite",
            clip_lo=float(ns.production_diurnal_clip_lo),
            clip_hi=float(ns.production_diurnal_clip_hi),
            power=float(ns.production_diurnal_power),
            softmax_temp=float(ns.production_diurnal_softmax_temp),
            poi_period_scale=not bool(ns.no_production_poi_period_scale),
            poi_unit_mean_weight=float(ns.production_poi_unit_mean_weight),
            origin_poi_mix=float(ns.production_origin_poi_mix),
        )
        meta["production_diurnal"] = diurnal_meta
    elif str(ns.production_diurnal_mode) == "mob_composite":
        features, diurnal_meta = apply_production_diurnal_factor(
            features,
            None,
            mode="mob_composite",
            clip_lo=float(ns.production_diurnal_clip_lo),
            clip_hi=float(ns.production_diurnal_clip_hi),
            power=float(ns.production_diurnal_power),
            softmax_temp=float(ns.production_diurnal_softmax_temp),
        )
        meta["production_diurnal"] = diurnal_meta
    else:
        diurnal_col = str(ns.production_diurnal_col or "").strip() or None
        if diurnal_col and diurnal_col not in features.columns:
            meta["production_diurnal"] = {"enabled": False, "column": diurnal_col, "reason": "column_not_in_features_table"}
            diurnal_col = None
        if diurnal_col:
            features, diurnal_meta = apply_production_diurnal_factor(
                features,
                diurnal_col,
                mode=str(ns.production_diurnal_mode),
                clip_lo=float(ns.production_diurnal_clip_lo),
                clip_hi=float(ns.production_diurnal_clip_hi),
                power=float(ns.production_diurnal_power),
                softmax_temp=float(ns.production_diurnal_softmax_temp),
            )
            meta["production_diurnal"] = diurnal_meta
        else:
            meta["production_diurnal"] = {"enabled": False, "reason": "column_missing_or_disabled"}
            features["production_diurnal_factor"] = 1.0
    features.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    meta["feature_table"] = feature_stats
    meta["attraction_weights"] = features.attrs.get("attraction_weights", ATTRACTION_WEIGHTS)
    meta["trip_generation_od_weights_note"] = features.attrs.get("trip_generation_od_weights_note", "")

    gen, od_period, od_stats = generate_period_od(
        units,
        features,
        out_dir,
        t_ids=t_ids,
        day_type_by_tid=day_type_map,
        total_trips=float(ns.total_trips),
        beta=float(ns.beta),
        d_floor_m=float(ns.d_floor_m),
        transfer_fraction=float(ns.transfer_fraction),
        station_band_m=float(ns.station_band_m),
        metroflow_calibration_json=ns.metroflow_calibration_json,
        period_od_max_rows=int(ns.period_od_max_rows),
    )
    meta["period_generation"] = od_stats

    modal_od, modal_stats = build_modal_period_od(features, od_period, out_dir / "synthetic_od_modal_by_period_long.csv")
    meta["modal_od"] = modal_stats

    try:
        plot_generation_quick(units, gen, out_dir / "trip_generation_maps.png", slices_df, ns.site_json)
        meta["trip_generation_maps"] = str(out_dir / "trip_generation_maps.png")
    except Exception as exc:
        meta["trip_generation_maps_error"] = str(exc)

    assignment_meta: dict[str, Any] = {"skipped": bool(ns.skip_assignment)}
    if not ns.skip_assignment:
        assignment_edges_csv, build_meta = maybe_build_assignment_edges(ns, out_dir)
        _edge_df, assign_stats = assign_period_edges(
            modal_od,
            assignment_edges_csv,
            out_dir,
            max_origins=int(ns.aon_max_origins),
            n_iters=int(ns.assignment_iters),
            delay_alpha=float(ns.assignment_delay_alpha),
            delay_power=float(ns.assignment_delay_power),
            scheme=str(ns.assignment_scheme),
            period_chain_order=chain_order,
        )
        assignment_meta = {**build_meta, **assign_stats, "skipped": False}
    meta["assignment"] = assignment_meta

    meta["outputs"] = {
        "worldpop_3km_tif": str(clipped_tif),
        "site_buffer_3km_geojson": str(buffer_geojson),
        "worldpop_by_unit_csv": str(ns.demography_dir / "worldpop_by_unit.csv"),
        "unit_features_by_period_csv": str(feature_csv),
        "trip_generation_by_period_csv": str(out_dir / "trip_generation_by_period.csv"),
        "synthetic_od_by_period_long_csv": str(out_dir / "synthetic_od_by_period_long.csv"),
        "synthetic_od_modal_by_period_long_csv": str(out_dir / "synthetic_od_modal_by_period_long.csv"),
        "synthetic_edge_flow_period_long_csv": str(out_dir / "synthetic_edge_flow_period_long.csv"),
        "synthetic_edge_flow_period_long_4class_csv": str(out_dir / "synthetic_edge_flow_period_long_4class.csv"),
    }
    (out_dir / "method_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "outputs": meta["outputs"], "assignment": assignment_meta}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
