#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / 'output' / 'stitching_field'
UNITS_PATH = DATA_DIR / 'site_units_with_probability.geojson'
BLOCKS_PATH = DATA_DIR / 'site_blocks_4.geojson'
META_PATH = DATA_DIR / 'site_stitching_meta.json'
FIG_DIR = DATA_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

NETWORKS = ['walk', 'slow', 'fast']
FUNCTIONS = [
    ('life_service', 'Life Service'),
    ('public_activity', 'Public Activity'),
    ('community_living', 'Community Living'),
    ('productive_mix', 'Productive Mix'),
]
NETWORK_LABELS = {'walk': 'Walking Layer', 'slow': 'Slow Mobility Layer', 'fast': 'Fast Road Layer'}
NETWORK_COLORS = {'walk': 'YlOrRd', 'slow': 'YlGnBu', 'fast': 'PuRd'}
BLOCK_NAME_MAP = {'block_nw': 'A', 'block_ne': 'B', 'block_sw': 'C', 'block_se': 'D'}


def _configure_fonts() -> None:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = '#f5f1e8'
    plt.rcParams['axes.facecolor'] = '#fcfaf5'


def _load_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    units = gpd.read_file(UNITS_PATH)
    blocks = gpd.read_file(BLOCKS_PATH)
    if units.crs is None:
        units = units.set_crs(4326)
    if blocks.crs is None:
        blocks = blocks.set_crs(4326)
    meta = json.loads(META_PATH.read_text(encoding='utf-8')) if META_PATH.exists() else {}
    return units, blocks, meta


def _norm_bounds(values: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vmax = float(s.quantile(0.95)) if len(s) else 1.0
    vmax = max(vmax, float(s.max()) if len(s) else 1.0, 1e-6)
    return 0.0, vmax


def _add_block_labels(ax: plt.Axes, blocks: gpd.GeoDataFrame) -> None:
    for _, row in blocks.iterrows():
        geom = row.geometry
        pt = geom.representative_point()
        label = BLOCK_NAME_MAP.get(str(row['site_block_id']), str(row['site_block_id']))
        ax.text(
            pt.x,
            pt.y,
            label,
            ha='center',
            va='center',
            fontsize=16,
            weight='bold',
            color='#111111',
            bbox={'boxstyle': 'circle,pad=0.3', 'facecolor': 'white', 'edgecolor': '#111111', 'linewidth': 1.1, 'alpha': 0.95},
            zorder=6,
        )


def _style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=13, loc='left', pad=8, color='#1f1f1f', weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_panel(ax: plt.Axes, units: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, field: str, title: str, cmap: str) -> None:
    vmin, vmax = _norm_bounds(units[field])
    units.plot(
        column=field,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0.45,
        edgecolor='#f8f5ed',
        legend=False,
        missing_kwds={'color': '#dddddd'},
        zorder=2,
    )
    blocks.boundary.plot(ax=ax, color='#111111', linewidth=1.8, zorder=5)
    _add_block_labels(ax, blocks)
    _style_axis(ax, title)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin, vmax)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.036, pad=0.015)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.6)


def _network_figure(units: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, network: str, meta: dict) -> Path:
    cmap = NETWORK_COLORS[network]
    fig = plt.figure(figsize=(16, 12), facecolor='#f5f1e8')
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 1.0], hspace=0.16, wspace=0.08)

    ax_main = fig.add_subplot(gs[0, :])
    _plot_panel(ax_main, units, blocks, f'P_{network}_total', f'{NETWORK_LABELS[network]}: Total Stitching Probability', cmap)

    func_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]
    for ax, (func_key, func_label) in zip(func_axes, FUNCTIONS):
        _plot_panel(ax, units, blocks, f'P_{network}_{func_key}', f'{func_label}', cmap)

    fig.text(0.04, 0.965, f'{NETWORK_LABELS[network]} Stitching Probability Field', fontsize=22, weight='bold', color='#171717')
    fig.text(0.04, 0.938, 'Top panel shows the network-layer guidance cloud; four panels below show function-specific probability components over the same four design blocks.', fontsize=11, color='#444444')

    notes = [
        'Blocks A-D follow merged parcel units and remain probabilistic rather than hard-zoned.',
        'Function gaps come from raw POI / landuse / Meituan evidence; network layers only modify how stitching is carried inside SITE.',
    ]
    if meta.get('notes'):
        notes.extend(meta['notes'][:2])
    fig.text(0.04, 0.02, '\n'.join(f'- {line}' for line in notes[:3]), fontsize=9.5, color='#4a4a4a')

    out_path = FIG_DIR / f'{network}_stitching_probability_field.png'
    fig.savefig(out_path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _summary_figure(units: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#f5f1e8')
    for ax, network in zip(axes, NETWORKS):
        _plot_panel(ax, units, blocks, f'P_{network}_total', NETWORK_LABELS[network], NETWORK_COLORS[network])
    fig.text(0.03, 0.95, 'SITE Stitching Probability Overview', fontsize=22, weight='bold', color='#171717')
    fig.text(0.03, 0.91, 'Three network-layer probability clouds over the same four parcel-based design blocks.', fontsize=11, color='#444444')
    out_path = FIG_DIR / 'site_stitching_probability_overview.png'
    fig.savefig(out_path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    _configure_fonts()
    units, blocks, meta = _load_data()
    outputs = [_summary_figure(units, blocks)]
    for network in NETWORKS:
        outputs.append(_network_figure(units, blocks, network, meta))
    manifest = {'figures': [str(path.relative_to(REPO)) for path in outputs]}
    (FIG_DIR / 'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Wrote figures to', FIG_DIR)


if __name__ == '__main__':
    main()
