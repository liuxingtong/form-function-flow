"""
在 geopandas 地图轴上叠加场地红线（GeoJSON LineString，CRS84≈WGS84）。

默认优先 ``data/site_3km/SITE.json``，不存在时回退 ``data/SITE.json``。
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

REPO = Path(__file__).resolve().parents[1]
_SITE_3KM_JSON = REPO / "data" / "site_3km" / "SITE.json"
_SITE_FALLBACK_JSON = REPO / "data" / "SITE.json"


def resolve_site_json_path() -> Path | None:
    """优先 3km 底图中的 SITE，其次仓库根 data/SITE.json。"""
    if _SITE_3KM_JSON.is_file():
        return _SITE_3KM_JSON
    if _SITE_FALLBACK_JSON.is_file():
        return _SITE_FALLBACK_JSON
    return None


DEFAULT_SITE_JSON = resolve_site_json_path() or _SITE_3KM_JSON


def load_site_gdf(site_path: Path | None = None) -> gpd.GeoDataFrame | None:
    if site_path is not None:
        p = Path(site_path)
        if not p.is_file():
            return None
        candidates = [p]
    else:
        candidates = [p for p in (_SITE_3KM_JSON, _SITE_FALLBACK_JSON) if p.is_file()]
    if not candidates:
        return None
    p = candidates[0]
    g = gpd.read_file(p)
    if g.crs is None:
        g = g.set_crs(4326)
    else:
        g = g.to_crs(4326)
    return g


def plot_site_boundary(
    ax,
    match_crs,
    site_path: Path | None = None,
    *,
    edgecolor: str = "#d90429",
    linewidth: float = 2.2,
    linestyle: tuple = (0, (5, 3)),
    zorder: float = 25,
) -> bool:
    """将 SITE 红线画在 ax 上（坐标系与 match_crs 对齐）。无文件时返回 False。"""
    g = load_site_gdf(site_path)
    if g is None or g.empty:
        return False
    if match_crs is not None:
        g = g.to_crs(match_crs)
    g.plot(
        ax=ax,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    return True
