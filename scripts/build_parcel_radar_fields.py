from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data' / 'site_3km'
OUT_DIR = REPO / 'output' / 'radar_fields'
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNITS_PATH = DATA / '01_units.gpkg'
MORPH_RAW_PATH = REPO / 'data' / 'morph_state.csv'
URBAN_STITCHING = Path(r'F:\Aworks\2026studio\shanghaistation\urban-stitching')
NETWORK_DIR = URBAN_STITCHING / 'analysis' / 'out' / 'networks'

FORM_AXES = [
    'building_coverage',
    'avg_height',
    'road_density',
    'green_blue_ratio',
    'heritage_ratio',
    'barrier_index',
    'permeability_index',
    'edge_conductance_mean',
]
FUNCTION_AXES = [
    'poi_food_density',
    'poi_retail_density',
    'poi_office_density',
    'poi_public_service_density',
    'poi_leisure_density',
    'poi_transport_density',
    'poi_life_service_density',
    'landuse_residential_ratio',
    'landuse_commercial_ratio',
    'landuse_public_ratio',
    'landuse_green_ratio',
    'vitality_food_weighted_density',
    'vitality_retail_weighted_density',
    'vitality_leisure_weighted_density',
    'vitality_life_service_weighted_density',
    'meituan_vitality_proxy',
]
FLOW_AXES = [
    'walk_length_km_per_km2',
    'slow_road_length_km_per_km2',
    'fast_road_length_km_per_km2',
    'all_road_length_km_per_km2',
    'walk_length_share',
    'slow_road_length_share',
    'fast_road_length_share',
]


POI_BIGTYPE_MAP = {
    'food': {'餐饮服务'},
    'retail': {'购物服务'},
    'office': {'公司企业'},
    'public_service': {'公共设施', '医疗保健服务', '政府机构及社会团体', '科教文化服务'},
    'leisure': {'体育休闲服务', '风景名胜', '住宿服务'},
    'transport': {'交通设施服务', '通行设施'},
    'life_service': {'生活服务', '金融保险服务'},
}
LANDUSE_FCLASS_MAP = {
    'landuse_residential_ratio': {'residential'},
    'landuse_commercial_ratio': {'commercial', 'retail', 'mixed', 'office'},
    'landuse_public_ratio': {'civic', 'education', 'medical', 'government', 'religious', 'transport'},
    'landuse_green_ratio': {'park', 'forest', 'grass', 'recreation_ground'},
}
MEITUAN_MAIN_MAP = {
    'food': {'美食'},
    'retail': {'本地购物'},
    'leisure': {'休闲娱乐', '运动健身', '旅游'},
    'life_service': {'生活服务', '爱车', '结婚', '亲子', '宠物', '医疗健康', '教育培训'},
}
MEITUAN_SUBSTR_MAP = {
    'food': ['餐', '饮', '咖啡', '甜品', '小吃'],
    'retail': ['购物', '服饰', '超市', '商场', '便利'],
    'leisure': ['娱乐', '影院', '健身', '景点', '玩'],
    'life_service': ['生活服务', '租车', '丽人', '培训', '家政', '维修', '宠物', '健康'],
}


def _read_units() -> gpd.GeoDataFrame:
    units = gpd.read_file(UNITS_PATH)
    if units.crs is None:
        units = units.set_crs(4326)
    elif units.crs.to_epsg() != 4326:
        units = units.to_crs(4326)
    units['area_m2'] = pd.to_numeric(units['area'], errors='coerce').fillna(0.0)
    units['area_m2'] = units['area_m2'].clip(lower=1.0)
    units['area_ha'] = units['area_m2'] / 10000.0
    units['area_km2'] = units['area_m2'] / 1_000_000.0
    return units[['unit_id', 'area_m2', 'area_ha', 'area_km2', 'geometry']].copy()


def _units_polygon_parts(units: gpd.GeoDataFrame, crs_epsg: int = 32651) -> gpd.GeoDataFrame:
    up = units[['unit_id', 'area_m2', 'area_km2', 'geometry']].copy()
    up = up[up.geometry.notna()].copy()
    up = up.explode(index_parts=False).reset_index(drop=True)
    up = up[up.geometry.geom_type == 'Polygon'].copy()
    if up.crs is None:
        up = up.set_crs(4326)
    if up.crs.to_epsg() != crs_epsg:
        up = up.to_crs(crs_epsg)
    return up


def _parse_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(',', '', regex=False).str.strip()
    s = s.replace({'None': np.nan, 'nan': np.nan, '无': np.nan, '': np.nan, 'null': np.nan})
    s = s.str.extract(r'([-+]?[0-9]*\.?[0-9]+)', expand=False)
    return pd.to_numeric(s, errors='coerce')


def _find_first(pattern: str) -> Path:
    match = next(DATA.rglob(pattern), None)
    if match is None:
        raise FileNotFoundError(pattern)
    return match


def build_form_fields(units: gpd.GeoDataFrame) -> pd.DataFrame:
    morph = pd.read_csv(MORPH_RAW_PATH)
    keep = ['unit_id'] + [c for c in FORM_AXES if c in morph.columns]
    out = units[['unit_id']].merge(morph[keep], on='unit_id', how='left')
    return out


def _categorize_meituan(row: pd.Series) -> str | None:
    main = str(row.get('主分类_', '') or '').strip()
    sub = str(row.get('分类_cat', '') or '').strip()
    for label, vals in MEITUAN_MAIN_MAP.items():
        if main in vals:
            return label
    for label, parts in MEITUAN_SUBSTR_MAP.items():
        if any(p in sub or p in main for p in parts):
            return label
    return None


def _vitality_score(df: pd.DataFrame) -> pd.Series:
    rating = _parse_numeric(df.get('评分_sco', pd.Series(index=df.index, dtype=object))).fillna(0.0)
    sales = _parse_numeric(df.get('月售单', pd.Series(index=df.index, dtype=object))).fillna(0.0)
    spend = _parse_numeric(df.get('人均消', pd.Series(index=df.index, dtype=object))).fillna(0.0)
    rating_norm = (rating / 5.0).clip(lower=0.0, upper=1.2)
    sales_norm = np.log1p(sales) / np.log(1000.0)
    spend_norm = np.log1p(spend) / np.log(1000.0)
    score = 0.55 * rating_norm + 0.35 * sales_norm + 0.10 * spend_norm
    return (1.0 + score.clip(lower=0.0)).astype(float)


def _build_landuse_fields(units: gpd.GeoDataFrame) -> pd.DataFrame:
    landuse_path = _find_first('*Landuse*.geojson')
    lu = gpd.read_file(landuse_path)
    if lu.crs is None:
        lu = lu.set_crs(4326)
    elif lu.crs.to_epsg() != 4326:
        lu = lu.to_crs(4326)
    lu = lu[['fclass', 'geometry']].dropna(subset=['geometry']).copy()
    lu = lu.explode(index_parts=False).reset_index(drop=True)
    lu = lu[lu.geometry.geom_type == 'Polygon'].copy().to_crs(32651)
    units_m = _units_polygon_parts(units, 32651)[['unit_id', 'area_m2', 'geometry']]

    rows: list[pd.Series] = []
    for out_col, fclasses in LANDUSE_FCLASS_MAP.items():
        sub = lu[lu['fclass'].isin(fclasses)].copy()
        if sub.empty:
            rows.append(pd.Series(dtype=float, name=out_col))
            continue
        inter = gpd.overlay(units_m[['unit_id', 'geometry']], sub[['geometry']], how='intersection', keep_geom_type=False)
        if inter.empty:
            rows.append(pd.Series(dtype=float, name=out_col))
            continue
        inter['part_area_m2'] = inter.geometry.area
        ser = inter.groupby('unit_id')['part_area_m2'].sum().rename(out_col)
        rows.append(ser)
    out = units[['unit_id', 'area_m2']].copy()
    for ser in rows:
        out = out.merge(ser, left_on='unit_id', right_index=True, how='left')
    for col in LANDUSE_FCLASS_MAP:
        out[col] = out[col].fillna(0.0) / out['area_m2']
    return out[['unit_id'] + list(LANDUSE_FCLASS_MAP.keys())]


def build_function_fields(units: gpd.GeoDataFrame) -> pd.DataFrame:
    poi_path = _find_first('Poidata-2025-*.geojson')
    poi = gpd.read_file(poi_path)
    if poi.crs is None:
        poi = poi.set_crs(4326)
    elif poi.crs.to_epsg() != 4326:
        poi = poi.to_crs(4326)
    poi = poi[['bigType', 'geometry']].dropna(subset=['geometry']).copy()
    poi_join = gpd.sjoin(poi, units[['unit_id', 'area_ha', 'geometry']], predicate='within', how='inner')

    poi_wide = units[['unit_id', 'area_ha']].copy()
    for label, values in POI_BIGTYPE_MAP.items():
        cnt = poi_join[poi_join['bigType'].isin(values)].groupby('unit_id').size().rename(f'poi_{label}_count')
        poi_wide = poi_wide.merge(cnt, left_on='unit_id', right_index=True, how='left')
        poi_wide[f'poi_{label}_count'] = poi_wide[f'poi_{label}_count'].fillna(0.0)
        poi_wide[f'poi_{label}_density'] = poi_wide[f'poi_{label}_count'] / poi_wide['area_ha']

    meituan_path = next(DATA.rglob('*meituan_NEW.geojson'), None) or next(DATA.rglob('*meituan_data.geojson'), None)
    if meituan_path is None:
        raise FileNotFoundError('meituan geojson')
    mt = gpd.read_file(meituan_path)
    if mt.crs is None:
        mt = mt.set_crs(4326)
    elif mt.crs.to_epsg() != 4326:
        mt = mt.to_crs(4326)
    mt = mt.dropna(subset=['geometry']).copy()
    mt['meituan_bucket'] = mt.apply(_categorize_meituan, axis=1)
    mt['vitality_score'] = _vitality_score(mt)
    mt_join = gpd.sjoin(mt, units[['unit_id', 'area_ha', 'geometry']], predicate='within', how='inner')
    mt_join['rating_num'] = _parse_numeric(mt_join['评分_sco']) if '评分_sco' in mt_join.columns else np.nan
    mt_agg = mt_join.groupby('unit_id').agg(
        meituan_vitality_proxy=('vitality_score', 'sum'),
        meituan_rating_mean=('rating_num', 'mean'),
    )
    mt_agg = units[['unit_id', 'area_ha']].merge(mt_agg, left_on='unit_id', right_index=True, how='left')
    mt_agg[['meituan_vitality_proxy', 'meituan_rating_mean']] = mt_agg[['meituan_vitality_proxy', 'meituan_rating_mean']].fillna(0.0)
    mt_agg['meituan_vitality_proxy'] = mt_agg['meituan_vitality_proxy'] / mt_agg['area_ha']

    vitality_fields = units[['unit_id', 'area_ha']].copy()
    for bucket in ['food', 'retail', 'leisure', 'life_service']:
        ser = mt_join.loc[mt_join['meituan_bucket'] == bucket].groupby('unit_id')['vitality_score'].sum().rename(f'vitality_{bucket}_weighted_density')
        vitality_fields = vitality_fields.merge(ser, left_on='unit_id', right_index=True, how='left')
        vitality_fields[f'vitality_{bucket}_weighted_density'] = vitality_fields[f'vitality_{bucket}_weighted_density'].fillna(0.0) / vitality_fields['area_ha']

    landuse_df = _build_landuse_fields(units)

    out = units[['unit_id']].merge(
        poi_wide[['unit_id'] + [f'poi_{k}_density' for k in POI_BIGTYPE_MAP]],
        on='unit_id',
        how='left',
    ).merge(
        landuse_df,
        on='unit_id',
        how='left',
    ).merge(
        vitality_fields[['unit_id'] + [f'vitality_{b}_weighted_density' for b in ['food', 'retail', 'leisure', 'life_service']]],
        on='unit_id',
        how='left',
    ).merge(
        mt_agg[['unit_id', 'meituan_vitality_proxy', 'meituan_rating_mean']],
        on='unit_id',
        how='left',
    )
    num_cols = [c for c in out.columns if c != 'unit_id']
    out[num_cols] = out[num_cols].fillna(0.0)
    return out


def _network_density_for_units(units: gpd.GeoDataFrame, network_path: Path, out_col: str) -> pd.DataFrame:
    lines = gpd.read_file(network_path)
    if lines.crs is None:
        lines = lines.set_crs(4326)
    elif lines.crs.to_epsg() != 4326:
        lines = lines.to_crs(4326)
    lines = lines[['geometry']].dropna(subset=['geometry']).copy()
    lines = lines.to_crs(32651)
    units_m = _units_polygon_parts(units, 32651)[['unit_id', 'area_km2', 'geometry']]
    inter = gpd.overlay(units_m[['unit_id', 'geometry']], lines, how='intersection', keep_geom_type=False)
    out = units[['unit_id', 'area_km2']].copy()
    if inter.empty:
        out[out_col] = 0.0
        return out[['unit_id', out_col]]
    inter['length_km'] = inter.geometry.length / 1000.0
    ser = inter.groupby('unit_id')['length_km'].sum().rename(out_col)
    out = out.merge(ser, left_on='unit_id', right_index=True, how='left')
    out[out_col] = out[out_col].fillna(0.0) / out['area_km2']
    return out[['unit_id', out_col]]


def build_flow_fields(units: gpd.GeoDataFrame) -> pd.DataFrame:
    out = units[['unit_id']].copy()
    nets = {
        'walk_length_km_per_km2': NETWORK_DIR / 'walk_clipped.geojson',
        'slow_road_length_km_per_km2': NETWORK_DIR / 'slow_clipped.geojson',
        'fast_road_length_km_per_km2': NETWORK_DIR / 'express_clipped.geojson',
    }
    for col, path in nets.items():
        out = out.merge(_network_density_for_units(units, path, col), on='unit_id', how='left')
    out['all_road_length_km_per_km2'] = (
        out['walk_length_km_per_km2'] + out['slow_road_length_km_per_km2'] + out['fast_road_length_km_per_km2']
    )
    denom = out['all_road_length_km_per_km2'].replace(0.0, np.nan)
    out['walk_length_share'] = out['walk_length_km_per_km2'] / denom
    out['slow_road_length_share'] = out['slow_road_length_km_per_km2'] / denom
    out['fast_road_length_share'] = out['fast_road_length_km_per_km2'] / denom
    num_cols = [c for c in out.columns if c != 'unit_id']
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def write_outputs(form_df: pd.DataFrame, function_df: pd.DataFrame, flow_df: pd.DataFrame) -> None:
    combined = form_df.merge(function_df, on='unit_id', how='outer').merge(flow_df, on='unit_id', how='outer')
    form_path = OUT_DIR / 'form_radar_fields.csv'
    function_path = OUT_DIR / 'function_radar_fields.csv'
    flow_path = OUT_DIR / 'flow_radar_fields.csv'
    combined_path = OUT_DIR / 'parcel_radar_fields.csv'
    form_df.to_csv(form_path, index=False, encoding='utf-8-sig')
    function_df.to_csv(function_path, index=False, encoding='utf-8-sig')
    flow_df.to_csv(flow_path, index=False, encoding='utf-8-sig')
    combined.to_csv(combined_path, index=False, encoding='utf-8-sig')

    meta = {
        'principle': 'No GMM, no clustering, no time-slice proxy. Only raw repository data or direct summary statistics from source data.',
        'outputs': {
            'form': str(form_path.relative_to(REPO)),
            'function': str(function_path.relative_to(REPO)),
            'flow': str(flow_path.relative_to(REPO)),
            'combined': str(combined_path.relative_to(REPO)),
        },
        'axes': {'form': FORM_AXES, 'function': FUNCTION_AXES, 'flow': FLOW_AXES},
        'sources': {
            'form': ['data/morph_state.csv'],
            'function': [
                'data/site_3km/02-POI&AOI/.../Poidata-2025-上海市.geojson',
                'data/site_3km/02-POI&AOI/2-AOI/landuse-webmap/上海市_Landuse.geojson',
                'data/site_3km/美团/*.geojson',
                'data/site_3km/01_units.gpkg',
            ],
            'flow': [
                'F:/Aworks/2026studio/shanghaistation/urban-stitching/analysis/out/networks/walk_clipped.geojson',
                'F:/Aworks/2026studio/shanghaistation/urban-stitching/analysis/out/networks/slow_clipped.geojson',
                'F:/Aworks/2026studio/shanghaistation/urban-stitching/analysis/out/networks/express_clipped.geojson',
                'data/site_3km/01_units.gpkg',
            ],
        },
        'notes': {
            'function': 'POI provides functional skeleton; landuse provides parcel program base; Meituan is mapped to consumption/service buckets and used as vitality weighting instead of a parallel duplicate count layer.',
            'flow': 'Flow layer here is parcel-normalized network structure density from the already prepared urban-stitching walk/slow/express networks, not station-flow or synthetic assignment results.',
        },
    }
    (OUT_DIR / 'radar_fields_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    units = _read_units()
    form_df = build_form_fields(units)
    function_df = build_function_fields(units)
    flow_df = build_flow_fields(units)
    write_outputs(form_df, function_df, flow_df)
    print('Wrote:', OUT_DIR)


if __name__ == '__main__':
    main()
