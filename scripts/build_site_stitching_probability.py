#!/usr/bin/env python3
from __future__ import annotations

import heapq
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

REPO = Path(__file__).resolve().parents[1]
RADAR_PATH = REPO / 'output' / 'radar_fields' / 'parcel_radar_fields.csv'
UNITS_PATH = REPO / 'data' / 'site_3km' / '01_units.gpkg'
EDGES_PATH = REPO / 'data' / 'site_3km' / '02_edges.csv'
SITE_PATH = REPO / 'data' / 'SITE.json'
OUT_DIR = REPO / 'output' / 'stitching_field'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECTED_EPSG = 32651
SITE_BUFFER_M = 1200.0

FUNCTION_GROUPS = {
    'life_service': {
        'label_zh': '生活服务型',
        'supply': {
            'poi_food_density': 0.24,
            'poi_retail_density': 0.24,
            'poi_life_service_density': 0.20,
            'vitality_food_weighted_density': 0.10,
            'vitality_retail_weighted_density': 0.10,
            'vitality_life_service_weighted_density': 0.07,
            'landuse_commercial_ratio': 0.05,
        },
        'need': {
            'landuse_residential_ratio': 0.45,
            'poi_office_density': 0.25,
            'poi_public_service_density': 0.15,
            'poi_retail_density': 0.15,
        },
        'fit': {
            'walk_length_share': 0.22,
            'walk_length_km_per_km2': 0.18,
            'permeability_index': 0.20,
            'edge_conductance_mean': 0.15,
            'meituan_vitality_proxy': 0.15,
            'building_coverage': 0.10,
        },
    },
    'public_activity': {
        'label_zh': '公共活动型',
        'supply': {
            'poi_public_service_density': 0.34,
            'poi_leisure_density': 0.26,
            'landuse_public_ratio': 0.16,
            'landuse_green_ratio': 0.16,
            'vitality_leisure_weighted_density': 0.08,
        },
        'need': {
            'landuse_residential_ratio': 0.30,
            'poi_office_density': 0.20,
            'poi_retail_density': 0.20,
            'poi_food_density': 0.15,
            'poi_public_service_density': 0.15,
        },
        'fit': {
            'walk_length_share': 0.15,
            'slow_road_length_share': 0.12,
            'permeability_index': 0.18,
            'edge_conductance_mean': 0.18,
            'landuse_green_ratio': 0.17,
            'poi_public_service_density': 0.10,
            'poi_leisure_density': 0.10,
        },
    },
    'community_living': {
        'label_zh': '社区栖居型',
        'supply': {
            'landuse_residential_ratio': 0.46,
            'poi_public_service_density': 0.20,
            'poi_life_service_density': 0.14,
            'landuse_green_ratio': 0.10,
            'vitality_life_service_weighted_density': 0.10,
        },
        'need': {
            'landuse_residential_ratio': 0.60,
            'poi_life_service_density': 0.15,
            'poi_public_service_density': 0.15,
            'landuse_green_ratio': 0.10,
        },
        'fit': {
            'landuse_residential_ratio': 0.24,
            'landuse_green_ratio': 0.18,
            'slow_road_length_share': 0.18,
            'walk_length_share': 0.14,
            'permeability_index': 0.12,
            'edge_conductance_mean': 0.08,
            'barrier_index': -0.06,
        },
    },
    'productive_mix': {
        'label_zh': '复合生产型',
        'supply': {
            'poi_office_density': 0.42,
            'landuse_commercial_ratio': 0.26,
            'poi_retail_density': 0.12,
            'poi_public_service_density': 0.10,
            'vitality_retail_weighted_density': 0.10,
        },
        'need': {
            'poi_office_density': 0.50,
            'landuse_commercial_ratio': 0.25,
            'poi_retail_density': 0.15,
            'poi_public_service_density': 0.10,
        },
        'fit': {
            'fast_length_share': 0.20,
            'slow_road_length_share': 0.18,
            'poi_office_density': 0.18,
            'avg_height': 0.16,
            'building_coverage': 0.14,
            'edge_conductance_mean': 0.08,
            'landuse_commercial_ratio': 0.06,
        },
    },
}

NETWORK_STITCH = {
    'walk': {
        'label_zh': '步行网络',
        'fields': {
            'walk_length_km_per_km2': 0.32,
            'walk_length_share': 0.26,
            'permeability_index': 0.18,
            'edge_conductance_mean': 0.16,
            'barrier_index': -0.08,
        },
    },
    'slow': {
        'label_zh': '慢速网络',
        'fields': {
            'slow_road_length_km_per_km2': 0.30,
            'slow_road_length_share': 0.24,
            'road_density': 0.14,
            'permeability_index': 0.16,
            'edge_conductance_mean': 0.10,
            'barrier_index': -0.06,
        },
    },
    'fast': {
        'label_zh': '快速路网络',
        'fields': {
            'fast_road_length_km_per_km2': 0.34,
            'fast_road_length_share': 0.26,
            'avg_height': 0.14,
            'building_coverage': 0.12,
            'edge_conductance_mean': 0.08,
            'barrier_index': -0.06,
        },
    },
}

NETWORK_FUNCTION_WEIGHT = {
    'walk': {'life_service': 1.00, 'public_activity': 0.95, 'community_living': 0.62, 'productive_mix': 0.36},
    'slow': {'life_service': 0.78, 'public_activity': 0.86, 'community_living': 1.00, 'productive_mix': 0.72},
    'fast': {'life_service': 0.28, 'public_activity': 0.52, 'community_living': 0.22, 'productive_mix': 1.00},
}

BLOCK_NAMES = {
    'NW': 'block_nw',
    'NE': 'block_ne',
    'SW': 'block_sw',
    'SE': 'block_se',
}


def _load_site_polygon(path: Path) -> gpd.GeoDataFrame:
    site = gpd.read_file(path)
    if site.crs is None:
        site = site.set_crs(4326)
    elif site.crs.to_epsg() != 4326:
        site = site.to_crs(4326)
    geom = site.geometry.iloc[0]
    if geom.geom_type == 'LineString':
        coords = list(geom.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        geom = Polygon(coords)
    elif geom.geom_type != 'Polygon':
        geom = geom.convex_hull
    return gpd.GeoDataFrame({'site_id': ['site']}, geometry=[geom], crs=4326)


def _read_units() -> gpd.GeoDataFrame:
    units = gpd.read_file(UNITS_PATH)
    if units.crs is None:
        units = units.set_crs(4326)
    elif units.crs.to_epsg() != 4326:
        units = units.to_crs(4326)
    units['unit_id'] = units['unit_id'].astype(str)
    units['area_m2'] = pd.to_numeric(units['area'], errors='coerce').fillna(0.0).clip(lower=1.0)
    units['area_ha'] = units['area_m2'] / 10000.0
    return units[['unit_id', 'area_m2', 'area_ha', 'geometry']].copy()


def _load_parcel_fields() -> pd.DataFrame:
    df = pd.read_csv(RADAR_PATH)
    df['unit_id'] = df['unit_id'].astype(str)
    if 'slow_road_length_share' not in df.columns and 'slow_length_share' in df.columns:
        df['slow_road_length_share'] = df['slow_length_share']
    if 'fast_road_length_share' not in df.columns and 'fast_length_share' in df.columns:
        df['fast_road_length_share'] = df['fast_length_share']
    return df


def _robust_scale(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    q05 = s.quantile(0.05)
    q95 = s.quantile(0.95)
    if not np.isfinite(q05) or not np.isfinite(q95) or math.isclose(float(q05), float(q95)):
        mn = s.min()
        mx = s.max()
        if math.isclose(float(mn), float(mx)):
            return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
        return ((s - mn) / (mx - mn)).clip(0.0, 1.0)
    return ((s - q05) / (q95 - q05)).clip(0.0, 1.0)


def _weighted_score(df: pd.DataFrame, weights: dict[str, float], scaled_cache: dict[str, pd.Series]) -> pd.Series:
    score = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    denom = sum(abs(v) for v in weights.values()) or 1.0
    for field, weight in weights.items():
        if field not in scaled_cache:
            scaled_cache[field] = _robust_scale(df[field]) if field in df.columns else pd.Series(0.0, index=df.index)
        base = scaled_cache[field]
        if weight >= 0:
            score = score + base * weight
        else:
            score = score + (1.0 - base) * abs(weight)
    return (score / denom).clip(0.0, 1.0)


def _build_context_units(units: gpd.GeoDataFrame, site_poly: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    units_m = units.to_crs(PROJECTED_EPSG)
    site_m = site_poly.to_crs(PROJECTED_EPSG)
    site_geom = site_m.geometry.iloc[0]
    site_mask = units_m.geometry.intersects(site_geom)
    context_mask = units_m.geometry.intersects(site_geom.buffer(SITE_BUFFER_M)) & (~site_mask)
    site_units = units.loc[site_mask.to_numpy()].copy()
    context_units = units.loc[context_mask.to_numpy()].copy()
    return site_units, context_units


def _compute_function_gap_fields(all_units: gpd.GeoDataFrame, site_units: gpd.GeoDataFrame, context_units: gpd.GeoDataFrame) -> pd.DataFrame:
    context_df = all_units[all_units['unit_id'].isin(context_units['unit_id'])].copy()
    site_df = all_units[all_units['unit_id'].isin(site_units['unit_id'])].copy()
    scaled_cache_context: dict[str, pd.Series] = {}
    for func_key, spec in FUNCTION_GROUPS.items():
        context_df[f'{func_key}_supply_score'] = _weighted_score(context_df, spec['supply'], scaled_cache_context)
        context_df[f'{func_key}_need_score'] = _weighted_score(context_df, spec['need'], scaled_cache_context)
        context_df[f'{func_key}_gap_local'] = (context_df[f'{func_key}_need_score'] * (1.0 - context_df[f'{func_key}_supply_score'])).clip(0.0, 1.0)

    out = site_df[['unit_id']].copy()
    if context_df.empty or site_df.empty:
        for func_key in FUNCTION_GROUPS:
            out[f'gap_{func_key}'] = 0.0
        return out

    site_cent = site_units[['unit_id', 'geometry']].to_crs(PROJECTED_EPSG).copy()
    site_xy = np.array([[geom.centroid.x, geom.centroid.y] for geom in site_cent.geometry])
    context_cent = context_units[['unit_id', 'geometry']].to_crs(PROJECTED_EPSG).copy()
    context_xy = np.array([[geom.centroid.x, geom.centroid.y] for geom in context_cent.geometry])

    dx = site_xy[:, 0][:, None] - context_xy[:, 0][None, :]
    dy = site_xy[:, 1][:, None] - context_xy[:, 1][None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    weights = np.clip(1.0 - (dist / SITE_BUFFER_M), 0.0, 1.0)
    weights = np.where(weights > 0, weights, 0.0)
    weights = np.where(weights < 0.05, 0.05, weights)

    for func_key in FUNCTION_GROUPS:
        vals = context_df.set_index('unit_id').loc[context_cent['unit_id'], f'{func_key}_gap_local'].fillna(0.0).to_numpy(dtype=float)
        numerator = (weights * vals[None, :]).sum(axis=1)
        denom = weights.sum(axis=1)
        denom = np.where(denom <= 1e-9, 1.0, denom)
        out[f'gap_{func_key}'] = numerator / denom
        fallback = float(context_df[f'{func_key}_gap_local'].mean()) if not context_df.empty else 0.0
        out[f'gap_{func_key}'] = out[f'gap_{func_key}'].fillna(fallback).clip(0.0, 1.0)
    return out


def _compute_site_probabilities(site_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaled_cache_site: dict[str, pd.Series] = {}
    out = site_df[['unit_id', 'area_m2']].copy()
    for net_key, net_spec in NETWORK_STITCH.items():
        out[f'stitch_{net_key}'] = _weighted_score(site_df, net_spec['fields'], scaled_cache_site)
    for func_key, func_spec in FUNCTION_GROUPS.items():
        out[f'fit_{func_key}'] = _weighted_score(site_df, func_spec['fit'], scaled_cache_site)

    records: list[dict[str, float | str]] = []
    for idx, row in out.iterrows():
        raw_scores: dict[tuple[str, str], float] = {}
        total_raw = 0.0
        for net_key in NETWORK_STITCH:
            for func_key in FUNCTION_GROUPS:
                gap_val = float(site_df.at[idx, f'gap_{func_key}'])
                stitch_val = float(out.at[idx, f'stitch_{net_key}'])
                fit_val = float(out.at[idx, f'fit_{func_key}'])
                compat = NETWORK_FUNCTION_WEIGHT[net_key][func_key]
                raw = max(gap_val, 0.0) * max(stitch_val, 0.0) * max(fit_val, 0.0) * compat
                raw_scores[(net_key, func_key)] = raw
                total_raw += raw
        total_raw = total_raw if total_raw > 0 else 1.0
        for net_key in NETWORK_STITCH:
            net_total = sum(raw_scores[(net_key, func_key)] for func_key in FUNCTION_GROUPS)
            records.append({
                'unit_id': row['unit_id'],
                'network_layer': net_key,
                'function_type': '__network_total__',
                'probability': net_total / total_raw,
                'gap_score': np.nan,
                'stitch_score': float(out.at[idx, f'stitch_{net_key}']),
                'fit_score': np.nan,
                'compatibility_weight': np.nan,
                'raw_score': net_total,
            })
            for func_key in FUNCTION_GROUPS:
                raw = raw_scores[(net_key, func_key)]
                records.append({
                    'unit_id': row['unit_id'],
                    'network_layer': net_key,
                    'function_type': func_key,
                    'probability': raw / total_raw,
                    'gap_score': float(site_df.at[idx, f'gap_{func_key}']),
                    'stitch_score': float(out.at[idx, f'stitch_{net_key}']),
                    'fit_score': float(out.at[idx, f'fit_{func_key}']),
                    'compatibility_weight': NETWORK_FUNCTION_WEIGHT[net_key][func_key],
                    'raw_score': raw,
                })
    long_df = pd.DataFrame.from_records(records)

    wide = out[['unit_id', 'area_m2']].copy()
    for net_key in NETWORK_STITCH:
        sub = long_df[(long_df['network_layer'] == net_key) & (long_df['function_type'] == '__network_total__')][['unit_id', 'probability']]
        wide = wide.merge(sub.rename(columns={'probability': f'P_{net_key}_total'}), on='unit_id', how='left')
        for func_key in FUNCTION_GROUPS:
            subf = long_df[(long_df['network_layer'] == net_key) & (long_df['function_type'] == func_key)][['unit_id', 'probability']]
            wide = wide.merge(subf.rename(columns={'probability': f'P_{net_key}_{func_key}'}), on='unit_id', how='left')
    for func_key in FUNCTION_GROUPS:
        wide = wide.merge(site_df[['unit_id', f'gap_{func_key}']], on='unit_id', how='left')
    return long_df, wide


def _build_adjacency(site_units: gpd.GeoDataFrame) -> dict[str, set[str]]:
    g = site_units[['unit_id', 'geometry']].copy().reset_index(drop=True)
    adj: dict[str, set[str]] = {uid: set() for uid in g['unit_id']}
    for i in range(len(g)):
        uid_i = str(g.at[i, 'unit_id'])
        geom_i = g.at[i, 'geometry']
        for j in range(i + 1, len(g)):
            uid_j = str(g.at[j, 'unit_id'])
            geom_j = g.at[j, 'geometry']
            if geom_i.touches(geom_j) or geom_i.intersects(geom_j):
                adj[uid_i].add(uid_j)
                adj[uid_j].add(uid_i)
    return adj


def _connected_components(nodes: list[str], adjacency: dict[str, set[str]]) -> list[list[str]]:
    remaining = set(nodes)
    comps: list[list[str]] = []
    while remaining:
        start = next(iter(remaining))
        q = deque([start])
        seen = {start}
        remaining.remove(start)
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in adjacency.get(cur, set()):
                if nb in remaining and nb not in seen:
                    seen.add(nb)
                    remaining.remove(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


def _assign_four_blocks(site_units: gpd.GeoDataFrame) -> pd.DataFrame:
    site_m = site_units.to_crs(PROJECTED_EPSG).copy()
    site_m['centroid'] = site_m.geometry.centroid
    xy = np.array([[p.x, p.y] for p in site_m['centroid']])
    center = xy.mean(axis=0)
    xy0 = xy - center
    cov = np.cov(xy0.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order]
    rot = xy0 @ basis
    x_med = float(np.median(rot[:, 0]))
    y_med = float(np.median(rot[:, 1]))

    block_df = pd.DataFrame({'unit_id': site_m['unit_id'].astype(str), 'rot_x': rot[:, 0], 'rot_y': rot[:, 1]})
    block_df['initial_block_code'] = [
        ('N' if y > y_med else 'S') + ('E' if x > x_med else 'W')
        for x, y in zip(block_df['rot_x'], block_df['rot_y'])
    ]

    centers: dict[str, tuple[float, float]] = {}
    for code in ['NW', 'NE', 'SW', 'SE']:
        sub = block_df[block_df['initial_block_code'] == code]
        if sub.empty:
            centers[code] = (
                float(block_df['rot_x'].max() if code.endswith('E') else block_df['rot_x'].min()),
                float(block_df['rot_y'].max() if code.startswith('N') else block_df['rot_y'].min()),
            )
        else:
            centers[code] = (float(sub['rot_x'].mean()), float(sub['rot_y'].mean()))

    assigned_seeds: set[str] = set()
    seeds: dict[str, str] = {}
    for code in ['NW', 'NE', 'SW', 'SE']:
        cx, cy = centers[code]
        cand = block_df.copy()
        cand['seed_dist'] = (cand['rot_x'] - cx) ** 2 + (cand['rot_y'] - cy) ** 2
        cand = cand.sort_values(['seed_dist', 'initial_block_code'])
        for uid in cand['unit_id']:
            if uid not in assigned_seeds:
                seeds[code] = str(uid)
                assigned_seeds.add(str(uid))
                break

    site_ids = set(block_df['unit_id'])
    edges = pd.read_csv(EDGES_PATH, usecols=['source_id', 'target_id', 'shared_length_m', 'centroid_dist_m'])
    edges['source_id'] = edges['source_id'].astype(str)
    edges['target_id'] = edges['target_id'].astype(str)
    edges = edges[edges['source_id'].isin(site_ids) & edges['target_id'].isin(site_ids)].copy()
    edges = edges[(edges['shared_length_m'].fillna(0.0) > 0.0) | (edges['centroid_dist_m'].fillna(9999.0) <= 180.0)].copy()
    adjacency: dict[str, set[str]] = {uid: set() for uid in site_ids}
    for _, row in edges.iterrows():
        s = row['source_id']
        t = row['target_id']
        if s == t:
            continue
        adjacency[s].add(t)
        adjacency[t].add(s)

    coord_map = block_df.set_index('unit_id')[['rot_x', 'rot_y']].to_dict('index')
    max_span = float(np.ptp(block_df['rot_x']) + np.ptp(block_df['rot_y']) + 1e-6)
    assigned: dict[str, str] = {}
    pq: list[tuple[float, int, str, str]] = []
    for code, seed_uid in seeds.items():
        heapq.heappush(pq, (0.0, 0, code, seed_uid))

    while pq:
        score, depth, code, uid = heapq.heappop(pq)
        if uid in assigned:
            continue
        assigned[uid] = code
        for nb in adjacency.get(uid, set()):
            if nb in assigned:
                continue
            cx, cy = centers[code]
            nbx = coord_map[nb]['rot_x']
            nby = coord_map[nb]['rot_y']
            dist_score = (((nbx - cx) ** 2 + (nby - cy) ** 2) ** 0.5) / max_span
            heapq.heappush(pq, (depth + 1 + dist_score, depth + 1, code, nb))

    for uid in block_df['unit_id']:
        if uid in assigned:
            continue
        ux = coord_map[uid]['rot_x']
        uy = coord_map[uid]['rot_y']
        best_code = min(centers, key=lambda code: (ux - centers[code][0]) ** 2 + (uy - centers[code][1]) ** 2)
        assigned[uid] = best_code

    block_df['block_code'] = block_df['unit_id'].map(assigned)
    block_df['site_block_id'] = block_df['block_code'].map(BLOCK_NAMES)
    return block_df[['unit_id', 'site_block_id', 'block_code']]


def _summarize_blocks(site_units: gpd.GeoDataFrame, wide_df: pd.DataFrame, block_df: pd.DataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    merged = site_units[['unit_id', 'area_m2', 'geometry']].merge(block_df, on='unit_id', how='left').merge(wide_df, on=['unit_id', 'area_m2'], how='left')
    prob_cols = [c for c in wide_df.columns if c.startswith('P_') or c.startswith('gap_')]
    rows = []
    for block_id, grp in merged.groupby('site_block_id'):
        weights = grp['area_m2'].fillna(1.0).to_numpy()
        rec = {'site_block_id': block_id, 'unit_count': int(len(grp)), 'block_area_m2': float(grp['area_m2'].sum())}
        for col in prob_cols:
            rec[col] = float(np.average(grp[col].fillna(0.0), weights=weights))
        rows.append(rec)
    block_summary = pd.DataFrame(rows).sort_values('site_block_id').reset_index(drop=True)
    block_gdf = merged.dissolve(by='site_block_id', aggfunc='first').reset_index()[['site_block_id', 'geometry']]
    block_gdf = block_gdf.merge(block_summary, on='site_block_id', how='left')
    return block_gdf, block_summary


def _build_meta() -> dict:
    return {
        'principle': 'Keep probabilistic guidance. No hard assignment of a single function type to each site block.',
        'site': {'source': str(SITE_PATH.relative_to(REPO)), 'site_buffer_m_for_gap_context': SITE_BUFFER_M},
        'function_groups': {
            key: {
                'label_zh': spec['label_zh'],
                'supply_fields': spec['supply'],
                'need_fields': spec['need'],
                'fit_fields': spec['fit'],
            }
            for key, spec in FUNCTION_GROUPS.items()
        },
        'network_layers': {
            key: {
                'label_zh': spec['label_zh'],
                'stitch_fields': spec['fields'],
                'function_compatibility': NETWORK_FUNCTION_WEIGHT[key],
            }
            for key, spec in NETWORK_STITCH.items()
        },
        'outputs': {
            'unit_long': 'output/stitching_field/site_unit_probability_long.csv',
            'unit_wide': 'output/stitching_field/site_unit_probability_wide.csv',
            'site_units_geojson': 'output/stitching_field/site_units_with_probability.geojson',
            'site_blocks_geojson': 'output/stitching_field/site_blocks_4.geojson',
            'site_block_probability': 'output/stitching_field/site_block_probability.csv',
        },
        'notes': [
            'Function gap fields are computed by function category only, not split by network layer.',
            'Network layers only act as site-internal stitching carriers: walk, slow, fast.',
            'The four blocks are a design coordination partition, not a hard functional zoning partition.',
            'Landuse in SITE is sparse and mostly residential/park/industrial, so function evidence is POI-led with landuse as program-base correction.',
        ],
    }


def main() -> None:
    site_poly = _load_site_polygon(SITE_PATH)
    units = _read_units()
    radar = _load_parcel_fields()
    all_units = units.merge(radar, on='unit_id', how='left')
    numeric_cols = [c for c in all_units.columns if c not in {'unit_id', 'geometry'}]
    all_units[numeric_cols] = all_units[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    site_units, context_units = _build_context_units(units, site_poly)
    gap_df = _compute_function_gap_fields(all_units, site_units, context_units)
    site_df = all_units[all_units['unit_id'].isin(site_units['unit_id'])].merge(gap_df, on='unit_id', how='left')
    long_df, wide_df = _compute_site_probabilities(site_df)
    block_df = _assign_four_blocks(site_units)
    block_gdf, block_summary = _summarize_blocks(site_units, wide_df, block_df)

    site_units_out = site_units.merge(block_df, on='unit_id', how='left').merge(wide_df, on=['unit_id', 'area_m2'], how='left')
    site_units_out = site_units_out.merge(
        long_df[long_df['function_type'] == '__network_total__'][['unit_id', 'network_layer', 'probability']].pivot(index='unit_id', columns='network_layer', values='probability').reset_index().rename(columns={
            'walk': 'P_walk_total_check',
            'slow': 'P_slow_total_check',
            'fast': 'P_fast_total_check',
        }),
        on='unit_id',
        how='left',
    )

    long_df.to_csv(OUT_DIR / 'site_unit_probability_long.csv', index=False, encoding='utf-8-sig')
    wide_df.merge(block_df, on='unit_id', how='left').to_csv(OUT_DIR / 'site_unit_probability_wide.csv', index=False, encoding='utf-8-sig')
    block_summary.to_csv(OUT_DIR / 'site_block_probability.csv', index=False, encoding='utf-8-sig')
    site_units_out.to_file(OUT_DIR / 'site_units_with_probability.geojson', driver='GeoJSON', encoding='utf-8')
    block_gdf.to_file(OUT_DIR / 'site_blocks_4.geojson', driver='GeoJSON', encoding='utf-8')
    (OUT_DIR / 'site_stitching_meta.json').write_text(json.dumps(_build_meta(), ensure_ascii=False, indent=2), encoding='utf-8')
    print('Wrote:', OUT_DIR)


if __name__ == '__main__':
    main()
