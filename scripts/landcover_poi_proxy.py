"""
GLC 点栅格与 ESA WorldCover 多边形：按单元汇总地类代理，供功能层 / 合成流在 POI 缺失时补全密度口径。

- GLC：``data/site_3km/09-开源土地利用/开源土地利用/GLC/*.geojson``，属性 ``z``（80 不透水、60 水体、30 草、20 林）。
- 回退 / 补缺：``…/ESA/SHP/上海市_开源土地利用-ESA10m.geojson``，属性 ``gridcode``（50 建成区等）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

GLC_POI_ZERO_EPS = 1e-12
GLC_LU_PROXY_SCALE = 0.00042

PROJ_CRS_M = "EPSG:32651"

_LAND_TAB_COLS = (
    "glc_impervious_share",
    "glc_forest_share",
    "glc_grass_share",
    "glc_water_share",
    "glc_samples_per_km2",
    "glc_lu_residential_proxy",
    "glc_lu_workplace_proxy",
    "glc_lu_commerce_proxy",
    "glc_lu_retail_proxy",
    "glc_lu_food_proxy",
    "glc_lu_leisure_proxy",
)


def resolve_glc_landcover_geojson_path() -> Path | None:
    root = REPO / "data" / "site_3km" / "09-开源土地利用" / "开源土地利用" / "GLC"
    if not root.is_dir():
        return None
    c2020 = sorted(root.glob("*2020*.geojson"), key=lambda p: len(p.name))
    if c2020:
        return c2020[0]
    any_g = sorted(root.glob("*.geojson"))
    return any_g[0] if any_g else None


def resolve_esa_landcover_geojson_path() -> Path | None:
    p = (
        REPO
        / "data"
        / "site_3km"
        / "09-开源土地利用"
        / "开源土地利用"
        / "ESA"
        / "SHP"
        / "上海市_开源土地利用-ESA10m.geojson"
    )
    return p if p.is_file() else None


def _empty_land_table(units: gpd.GeoDataFrame) -> pd.DataFrame:
    uid = units["unit_id"].astype(str)
    out = pd.DataFrame({"unit_id": uid})
    for c in _LAND_TAB_COLS:
        out[c] = 0.0
    return out


def _proxy_from_counts(
    n80: int,
    n60: int,
    n30: int,
    n20: int,
    n: int,
    spd_u: float,
    p95_spd: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    nk = n80 + n60 + n30 + n20
    denom = float(max(n, 1)) if nk < max(1, n // 2) else float(max(nk, 1))
    p80, p60, p30, p20 = n80 / denom, n60 / denom, n30 / denom, n20 / denom
    S = float(GLC_LU_PROXY_SCALE)
    nat = min(1.0, p20 + p30 + p60)
    lu_res = p80 * (0.18 + 0.82 * nat) * S
    lu_wrk = p80 * (0.32 + 0.68 * min(spd_u / max(p95_spd, 1e-9), 2.8)) * S * 1.15
    lu_com = p80 * (0.22 + 0.55 * p60 + 0.42 * (p20 + p30)) * S * 1.05
    lu_ret = lu_com * (0.85 + 0.65 * p60 + 0.25 * (p20 + p30))
    lu_food = lu_com * (0.9 + 0.4 * min(1.0, p20 + p30) + 0.35 * p60)
    lu_lei = min(1.0, p20 + p30 + 0.45 * p60) * S * 2.0
    return p80, p60, p30, p20, lu_res, lu_wrk, lu_com, lu_ret, lu_food, lu_lei


def _aggregate_glc_points_by_unit(units: gpd.GeoDataFrame, path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {"glc_geojson": str(path), "glc_points_joined": 0, "glc_units_with_points": 0}
    out = _empty_land_table(units)
    try:
        pts = gpd.read_file(path)
    except Exception as exc:  # pragma: no cover
        meta["glc_read_error"] = str(exc)
        return out, meta

    if pts.crs is None:
        pts = pts.set_crs(4326)
    if "z" not in pts.columns:
        meta["glc_missing_z_column"] = True
        return out, meta

    pts = pts.copy()
    pts["z"] = pd.to_numeric(pts["z"], errors="coerce")
    pts = pts[pts["geometry"].notna()].copy()
    if len(pts) == 0:
        return out, meta

    u = units.copy()
    if u.crs is None:
        u = u.set_crs(4326)
    u["_uid"] = u["unit_id"].astype(str)

    join = gpd.sjoin(pts[["geometry", "z"]], u[["_uid", "geometry", "area"]], predicate="within", how="inner")
    meta["glc_points_joined"] = int(len(join))
    if join.empty:
        return out, meta

    area_by = u.set_index("_uid")["area"].astype(float).clip(lower=1.0)
    per_unit: dict[str, tuple[int, int, int, int, int]] = {}
    spd_by: dict[str, float] = {}
    for uu, grp in join.groupby(join["_uid"].astype(str), sort=False):
        uus = str(uu)
        z = grp["z"].to_numpy(dtype=float)
        n = int(len(z))
        if n <= 0:
            continue
        n80 = int(np.sum(np.isclose(z, 80.0)))
        n60 = int(np.sum(np.isclose(z, 60.0)))
        n30 = int(np.sum(np.isclose(z, 30.0)))
        n20 = int(np.sum(np.isclose(z, 20.0)))
        am = float(area_by.get(uus, float(area_by.median())))
        area_km2 = max(am / 1_000_000.0, 1e-9)
        spd_by[uus] = float(n) / area_km2
        per_unit[uus] = (n80, n60, n30, n20, n)

    if not spd_by:
        return out, meta

    p95_spd = float(np.nanpercentile(np.asarray(list(spd_by.values()), dtype=float), 95)) or 1.0
    row_ix = dict(zip(out["unit_id"].astype(str), out.index.astype(int)))

    for uus, (n80, n60, n30, n20, n) in per_unit.items():
        spd_u = spd_by.get(uus, 0.0)
        p80, p60, p30, p20, lu_res, lu_wrk, lu_com, lu_ret, lu_food, lu_lei = _proxy_from_counts(
            n80, n60, n30, n20, n, spd_u, p95_spd
        )
        ix = row_ix.get(uus)
        if ix is None:
            continue
        out.loc[ix, "glc_impervious_share"] = float(p80)
        out.loc[ix, "glc_forest_share"] = float(p20)
        out.loc[ix, "glc_grass_share"] = float(p30)
        out.loc[ix, "glc_water_share"] = float(p60)
        out.loc[ix, "glc_samples_per_km2"] = float(spd_u)
        out.loc[ix, "glc_lu_residential_proxy"] = float(lu_res)
        out.loc[ix, "glc_lu_workplace_proxy"] = float(lu_wrk)
        out.loc[ix, "glc_lu_commerce_proxy"] = float(lu_com)
        out.loc[ix, "glc_lu_retail_proxy"] = float(lu_ret)
        out.loc[ix, "glc_lu_food_proxy"] = float(lu_food)
        out.loc[ix, "glc_lu_leisure_proxy"] = float(lu_lei)

    meta["glc_units_with_points"] = int((out["glc_samples_per_km2"] > 0).sum())
    return out, meta


def _esa_code_bucket(code: Any) -> str:
    try:
        c = int(round(float(code)))
    except (TypeError, ValueError):
        return "other"
    if c == 50:
        return "built"
    if c == 80:
        return "water"
    if c in (30, 40, 90):
        return "grass"
    if c in (10, 20):
        return "forest"
    return "other"


def _aggregate_esa_polygons_by_unit(units: gpd.GeoDataFrame, path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """ESA WorldCover 面与单元求交，用地类面积占比近似 GLC 的 80/60/30/20 结构，再套同一套代理公式。"""
    meta: dict[str, Any] = {"esa_geojson": str(path), "esa_overlay_rows": 0, "esa_units_covered": 0}
    out = _empty_land_table(units)
    try:
        lu = gpd.read_file(path)
    except Exception as exc:  # pragma: no cover
        meta["esa_read_error"] = str(exc)
        return out, meta

    zcol = "gridcode" if "gridcode" in lu.columns else ("Code" if "Code" in lu.columns else None)
    if zcol is None:
        meta["esa_missing_code_column"] = True
        return out, meta

    u = units[["unit_id", "geometry", "area"]].copy()
    if u.crs is None:
        u = u.set_crs(4326)
    u = u.to_crs(PROJ_CRS_M)
    lu = lu.to_crs(PROJ_CRS_M)
    lu = lu[["geometry", zcol]].copy()
    lu[zcol] = pd.to_numeric(lu[zcol], errors="coerce")

    try:
        inter = gpd.overlay(u, lu, how="intersection", keep_geom_type=False)
    except Exception as exc:
        meta["esa_overlay_error"] = str(exc)
        return out, meta

    if inter.empty:
        return out, meta

    inter["_w"] = inter.geometry.area.clip(lower=0.0)
    inter = inter[inter["_w"] > 0].copy()
    inter["_cat"] = inter[zcol].map(_esa_code_bucket)
    meta["esa_overlay_rows"] = int(len(inter))

    area_by = u.set_index(u["unit_id"].astype(str))["area"].astype(float).clip(lower=1.0)
    row_ix = dict(zip(out["unit_id"].astype(str), out.index.astype(int)))

    spd_by: dict[str, float] = {}
    shares: dict[str, tuple[float, float, float, float]] = {}
    for uus, grp in inter.groupby(inter["unit_id"].astype(str), sort=False):
        uus = str(uus)
        wsum = grp.groupby("_cat", observed=False)["_w"].sum()
        total_w = float(wsum.sum())
        if total_w <= 0:
            continue
        sh_built = float(wsum.get("built", 0.0)) / total_w
        sh_water = float(wsum.get("water", 0.0)) / total_w
        sh_grass = float(wsum.get("grass", 0.0)) / total_w
        sh_forest = float(wsum.get("forest", 0.0)) / total_w
        am = float(area_by.get(uus, float(area_by.median())))
        area_km2 = max(am / 1_000_000.0, 1e-9)
        spd_by[uus] = float(len(grp)) / area_km2
        shares[uus] = (sh_built, sh_water, sh_grass, sh_forest)

    if not spd_by:
        return out, meta

    p95_spd = float(np.nanpercentile(np.asarray(list(spd_by.values()), dtype=float), 95)) or 1.0
    n_syn = 1000
    for uus, (sh80, sh60, sh30, sh20) in shares.items():
        n80 = max(int(round(sh80 * n_syn)), 0)
        n60 = max(int(round(sh60 * n_syn)), 0)
        n30 = max(int(round(sh30 * n_syn)), 0)
        n20 = max(int(round(sh20 * n_syn)), 0)
        spd_u = spd_by.get(uus, 0.0)
        p80, p60, p30, p20, lu_res, lu_wrk, lu_com, lu_ret, lu_food, lu_lei = _proxy_from_counts(
            n80, n60, n30, n20, n_syn, spd_u, p95_spd
        )
        ix = row_ix.get(uus)
        if ix is None:
            continue
        out.loc[ix, "glc_impervious_share"] = float(p80)
        out.loc[ix, "glc_forest_share"] = float(p20)
        out.loc[ix, "glc_grass_share"] = float(p30)
        out.loc[ix, "glc_water_share"] = float(p60)
        out.loc[ix, "glc_samples_per_km2"] = float(spd_u)
        out.loc[ix, "glc_lu_residential_proxy"] = float(lu_res)
        out.loc[ix, "glc_lu_workplace_proxy"] = float(lu_wrk)
        out.loc[ix, "glc_lu_commerce_proxy"] = float(lu_com)
        out.loc[ix, "glc_lu_retail_proxy"] = float(lu_ret)
        out.loc[ix, "glc_lu_food_proxy"] = float(lu_food)
        out.loc[ix, "glc_lu_leisure_proxy"] = float(lu_lei)

    meta["esa_units_covered"] = int((out["glc_samples_per_km2"] > 0).sum())
    return out, meta


def aggregate_glc_landuse_by_unit(units: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    按单元输出 glc_* 代理列。优先 GLC 点；无点或缺测单元用 ESA 面叠加补缺。
    """
    meta: dict[str, Any] = {"landcover_source": "none"}
    out = _empty_land_table(units)
    glc_path = resolve_glc_landcover_geojson_path()
    if glc_path is not None and glc_path.is_file():
        out, gmeta = _aggregate_glc_points_by_unit(units, glc_path)
        meta.update(gmeta)
        if int(meta.get("glc_points_joined", 0)) > 0:
            meta["landcover_source"] = "GLC points"

    gap = out["glc_samples_per_km2"] <= 0
    esa_path = resolve_esa_landcover_geojson_path()
    if esa_path is not None and esa_path.is_file() and (bool(gap.any()) or int(meta.get("glc_points_joined", 0)) == 0):
        esa_out, emeta = _aggregate_esa_polygons_by_unit(units, esa_path)
        meta.update(emeta)
        fill = gap
        if int(meta.get("glc_points_joined", 0)) == 0:
            out = esa_out
            meta["landcover_source"] = "ESA WorldCover polygons"
        else:
            for c in _LAND_TAB_COLS:
                out.loc[fill, c] = esa_out.loc[fill, c].to_numpy()
            meta["landcover_source"] = "GLC points + ESA gap-fill"

    return out, meta
