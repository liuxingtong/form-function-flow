#!/usr/bin/env python3
"""
N01 慢行与人行：步行矢量默认 **裁切到基地（SITE）周边 ``--site-buffer-m`` 米**（默认 3000），
从 ``all`` 目录裁剪补充时 **不含专用自行车道**（文件名含「自行车」）；上海道路 OSM ``fclass`` 亦不纳入 ``cycleway``。
制图时 **官方与补充合并为同一步行路网图层**（单色）；**不使用地块邻接 proxy**。

覆盖率仍为 **质心至合并步行线网 ≤ R m**（地块仍为全套单元）。

用法：
  python scripts/plot_N01_pedestrian_enriched.py --units output/function/数据包/01_units.gpkg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_flow_modality_networks import (  # noqa: E402
    DEFAULT_MAX_FILE_MB,
    MODALITY_RULES,
    _assign_modality,
    _collect_geojsons,
    _explode_lines,
    configure_cn_font,
    is_dedicated_bicycle_geojson_path,
)
from site_map_overlay import load_site_gdf, plot_site_boundary, resolve_site_json_path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CRS_M = "EPSG:32651"

configure_cn_font()

N01_MOD_NAME = "慢行与人行（原始矢量）"

DEFAULT_SUPPLEMENT_ROOTS: tuple[str, ...] = (
    r"f:\Aworks\2026studio\shanghaistation\all\04-交通数据",
    r"f:\Aworks\2026studio\shanghaistation\all\16-城市环路",
    r"f:\Aworks\2026studio\shanghaistation\all\上海道路",
)

INCLUDE_SHP_STEM_KEYS: tuple[str, ...] = (
    "行人",
    "步行",
    "慢行",
    "绿道",
    "轮渡",
    "渡口",
    "跑道",
    "人行道",
)
EXCLUDE_SHP_STEM_KEYS: tuple[str, ...] = (
    "铁路",
    "地铁线",
    "地铁",
    "轨道",
    "高速",
    "快速路合集",
)

WALK_FCLASS_CORE: frozenset[str] = frozenset(
    {"footway", "path", "pedestrian", "steps", "living_street", "bridleway"}
)
WALK_FCLASS_EXTENDED: frozenset[str] = frozenset(
    WALK_FCLASS_CORE
    | {
        "track",
        "service",
        "residential",
        "unclassified",
        "tertiary",
        "tertiary_link",
        "secondary",
        "secondary_link",
    }
)


def _load_units(path: Path) -> gpd.GeoDataFrame:
    try:
        u = gpd.read_file(path, layer="units")
    except Exception:
        u = gpd.read_file(path)
    if u.crs is None:
        u = u.set_crs(4326)
    return u


def _bbox4326_pad_polygon(clip_poly_m: BaseGeometry, pad_deg: float) -> tuple[float, float, float, float]:
    g4326 = gpd.GeoDataFrame(geometry=[clip_poly_m], crs=CRS_M).to_crs(4326)
    minx, miny, maxx, maxy = g4326.total_bounds
    p = float(pad_deg)
    return (minx - p, miny - p, maxx + p, maxy + p)


def site_buffer_polygons(
    units: gpd.GeoDataFrame,
    site_path: Path | None,
    *,
    radius_m: float,
) -> tuple[BaseGeometry, BaseGeometry]:
    """(缓冲区 Polygon EPSG:32651, 同日缓冲区 transform 到 units.crs)。"""
    r = float(radius_m)
    site = load_site_gdf(site_path)
    if site is not None and not site.empty:
        sm = site.to_crs(CRS_M)
        geom = sm.geometry
        core = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
    else:
        um = units.to_crs(CRS_M)
        geom = um.geometry
        core = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
        try:
            core = core.centroid
        except Exception:
            pass
    buf_m = core.buffer(r)
    buf_ucrs = gpd.GeoDataFrame(geometry=[buf_m], crs=CRS_M).to_crs(units.crs).geometry.iloc[0]
    return buf_m, buf_ucrs


def _clip_exploded_lines(gdf: gpd.GeoDataFrame, clip_poly_m: BaseGeometry, *, units_crs) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=units_crs)
    gm = gdf.to_crs(CRS_M)
    gm = gm[gm.intersects(clip_poly_m)].copy()
    if gm.empty:
        return gpd.GeoDataFrame(geometry=[], crs=units_crs)
    gt = gm.geometry.geom_type
    Llines = gm[gt.isin(["LineString", "MultiLineString", "GeometryCollection"])].copy()
    if Llines.empty:
        return gpd.GeoDataFrame(geometry=[], crs=units_crs)
    exp = _explode_lines(Llines)
    return exp.to_crs(units_crs) if len(exp) else gpd.GeoDataFrame(geometry=[], crs=units_crs)


def _collect_n01_frames(data_root: Path, *, max_file_mb: float) -> tuple[list[gpd.GeoDataFrame], list[Path]]:
    roots = [data_root / "04-交通数据", data_root / "metroflow"]
    roots = [r.resolve() for r in roots if r.is_dir()]
    paths = _collect_geojsons(roots)
    buckets: dict[str, list[Path]] = {name: [] for name, _ in MODALITY_RULES}
    for p in paths:
        mod = _assign_modality(p)
        if mod:
            buckets[mod].append(p)
    n01_paths = [p for p in buckets.get(N01_MOD_NAME, []) if not is_dedicated_bicycle_geojson_path(p)]
    max_bytes = int(max(1.0, max_file_mb) * 1024 * 1024)
    frames: list[gpd.GeoDataFrame] = []
    used: list[Path] = []
    for p in n01_paths:
        if p.stat().st_size > max_bytes:
            continue
        try:
            g = gpd.read_file(p)
        except Exception:
            continue
        if g.empty:
            continue
        if g.crs is None:
            g = g.set_crs(4326)
        frames.append(g)
        used.append(p)
    return frames, used


def _official_exploded_lines(
    units: gpd.GeoDataFrame,
    frames: list[gpd.GeoDataFrame],
    clip_poly_m: BaseGeometry,
    clip_poly_ucrs: BaseGeometry,
) -> gpd.GeoDataFrame:
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=units.crs)
    merged = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(merged, crs=frames[0].crs)
    if gdf.crs != units.crs:
        gdf = gdf.to_crs(units.crs)
    try:
        gdf = gdf[gdf.intersects(clip_poly_ucrs)].copy()
    except Exception:
        pass
    gt = gdf.geometry.geom_type
    Llines = gdf[gt.isin(["LineString", "MultiLineString", "GeometryCollection"])].copy()
    exp = _explode_lines(Llines) if len(Llines) else gpd.GeoDataFrame(geometry=[], crs=units.crs)
    return _clip_exploded_lines(exp, clip_poly_m, units_crs=units.crs) if len(exp) else exp


def _stem_include_shp(stem: str) -> bool:
    if "自行车" in stem:
        return False
    if any(k in stem for k in EXCLUDE_SHP_STEM_KEYS):
        return False
    return any(k in stem for k in INCLUDE_SHP_STEM_KEYS)


def _is_ring_shapefile(stem: str) -> bool:
    return stem in ("内环", "中环", "外环", "郊环") or ("环路" in stem and len(stem) <= 12)


def _gdf_to_linestrings_for_rings(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """环路数据常为 Polygon 场面；取其外环为线以便 explode / 绘图。"""
    rows: list = []
    crs = g.crs
    for geom in g.geometry:
        if geom is None or geom.is_empty:
            continue
        gt = geom.geom_type
        if gt == "LineString":
            rows.append(LineString(geom.coords))
        elif gt == "MultiLineString":
            for seg in geom.geoms:
                rows.append(LineString(seg.coords))
        elif gt == "Polygon":
            rows.append(LineString(geom.exterior.coords))
        elif gt == "MultiPolygon":
            for p in geom.geoms:
                rows.append(LineString(p.exterior.coords))
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    return gpd.GeoDataFrame(geometry=rows, crs=crs)


def _append_ring_lines_from_geojson_dir(
    ring_dir: Path,
    *,
    bbox: tuple[float, float, float, float],
    clip_poly_m: BaseGeometry,
    units_crs,
    max_bytes: int,
    notes: list[str],
    ring_parts: list[gpd.GeoDataFrame],
    note_prefix: str,
    accept_any_geojson: bool,
) -> None:
    if not ring_dir.is_dir():
        return
    for gj in sorted(ring_dir.glob("*.geojson")):
        if gj.stat().st_size > max_bytes:
            continue
        if not accept_any_geojson and not _is_ring_shapefile(gj.stem):
            continue
        try:
            g = gpd.read_file(gj, bbox=bbox)
        except Exception:
            try:
                g = gpd.read_file(gj)
            except Exception:
                continue
        if g.empty or "geometry" not in g.columns:
            continue
        if g.crs is None:
            g = g.set_crs(4326)
        line_g = _gdf_to_linestrings_for_rings(g)
        if line_g.empty:
            continue
        exp = _explode_lines(line_g)
        ring_parts.append(_clip_exploded_lines(exp, clip_poly_m, units_crs=units_crs))
        notes.append(f"{note_prefix}: {gj.name} ({len(exp)} seg)")


def _load_supplement_layers(
    supplement_roots: list[Path],
    units: gpd.GeoDataFrame,
    *,
    clip_poly_m: BaseGeometry,
    extended_fclass: bool,
    max_file_mb: float,
    bbox_pad_deg: float,
    ring_geojson_fallback_dir: Path | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[str]]:
    """返回 (步行补充线, 环路参考线, 日志摘要行)。"""
    walk_parts: list[gpd.GeoDataFrame] = []
    ring_parts: list[gpd.GeoDataFrame] = []
    notes: list[str] = []
    max_bytes = int(max(1.0, max_file_mb) * 1024 * 1024)
    bbox = _bbox4326_pad_polygon(clip_poly_m, bbox_pad_deg)
    ucrs = units.crs
    fclasses = WALK_FCLASS_EXTENDED if extended_fclass else WALK_FCLASS_CORE

    for root in supplement_roots:
        root = Path(root)
        if not root.is_dir():
            notes.append(f"跳过（不存在）: {root}")
            continue
        notes.append(f"扫描: {root}")

        is_ring_dir = ("16-" in root.name) or ("城市环路" in root.name) or ("环路" in root.name and "交通" not in root.name)
        is_shanghai_road_dir = "上海道路" in root.name or root.name.endswith("上海道路")

        shps = sorted(root.rglob("*.shp"))
        for shp in shps:
            if shp.stat().st_size > max_bytes:
                continue
            stem = shp.stem

            if is_ring_dir and _is_ring_shapefile(stem):
                try:
                    g = gpd.read_file(shp, bbox=bbox)
                except Exception:
                    try:
                        g = gpd.read_file(shp)
                    except Exception:
                        continue
                if g.empty or "geometry" not in g.columns:
                    continue
                if g.crs is None:
                    g = g.set_crs(4326)
                line_g = _gdf_to_linestrings_for_rings(g)
                if line_g.empty:
                    continue
                exp = _explode_lines(line_g)
                ring_parts.append(_clip_exploded_lines(exp, clip_poly_m, units_crs=ucrs))
                notes.append(f"环路 shp: {shp.name} ({len(exp)} seg)")
                continue

            if is_shanghai_road_dir and ("road" in stem.lower()):
                try:
                    g = gpd.read_file(shp, bbox=bbox)
                except Exception:
                    try:
                        g = gpd.read_file(shp)
                    except Exception:
                        continue
                if g.empty or "fclass" not in g.columns:
                    notes.append(f"上海道路跳过（无 fclass）: {shp.name}")
                    continue
                if g.crs is None:
                    g = g.set_crs(4326)
                mask = g["fclass"].astype(str).str.lower().isin(fclasses)
                g2 = g.loc[mask].copy()
                if g2.empty:
                    notes.append(f"上海道路 {shp.name}: fclass 筛选后为空（试 --extended-road-fclass）")
                    continue
                gt = g2.geometry.geom_type
                Llines = g2[gt.isin(["LineString", "MultiLineString"])].copy()
                if Llines.empty:
                    continue
                exp = _explode_lines(Llines)
                walk_parts.append(_clip_exploded_lines(exp, clip_poly_m, units_crs=ucrs))
                notes.append(f"上海道路 {shp.name}: {len(exp)} 段（extended={extended_fclass}）")
                continue

            if _stem_include_shp(stem):
                try:
                    g = gpd.read_file(shp, bbox=bbox)
                except Exception:
                    try:
                        g = gpd.read_file(shp)
                    except Exception:
                        continue
                if g.empty:
                    continue
                if g.crs is None:
                    g = g.set_crs(4326)
                gt = g.geometry.geom_type
                Llines = g[gt.isin(["LineString", "MultiLineString", "GeometryCollection"])].copy()
                if Llines.empty:
                    continue
                exp = _explode_lines(Llines)
                walk_parts.append(_clip_exploded_lines(exp, clip_poly_m, units_crs=ucrs))
                notes.append(f"慢行 shp: {shp.name} → {len(exp)} 段")

        if is_ring_dir:
            _append_ring_lines_from_geojson_dir(
                root,
                bbox=bbox,
                clip_poly_m=clip_poly_m,
                units_crs=ucrs,
                max_bytes=max_bytes,
                notes=notes,
                ring_parts=ring_parts,
                note_prefix="环路 geojson",
                accept_any_geojson=True,
            )

        for gj in root.rglob("*.geojson"):
            if gj.stat().st_size > max_bytes:
                continue
            if is_dedicated_bicycle_geojson_path(gj):
                continue
            if not any(k in gj.stem for k in INCLUDE_SHP_STEM_KEYS):
                continue
            if _assign_modality(gj) != N01_MOD_NAME:
                continue
            try:
                g = gpd.read_file(gj, bbox=bbox)
            except Exception:
                continue
            if g.empty:
                continue
            if g.crs is None:
                g = g.set_crs(4326)
            gt = g.geometry.geom_type
            Llines = g[gt.isin(["LineString", "MultiLineString", "GeometryCollection"])].copy()
            if Llines.empty:
                continue
            exp = _explode_lines(Llines)
            walk_parts.append(_clip_exploded_lines(exp, clip_poly_m, units_crs=ucrs))
            notes.append(f"geojson: {gj.name}")

    if ring_geojson_fallback_dir is not None and not ring_parts:
        fb = Path(ring_geojson_fallback_dir)
        notes.append(f"环路未自 all 载入 → 回退 {fb}")
        _append_ring_lines_from_geojson_dir(
            fb,
            bbox=bbox,
            clip_poly_m=clip_poly_m,
            units_crs=ucrs,
            max_bytes=max_bytes,
            notes=notes,
            ring_parts=ring_parts,
            note_prefix="环路 geojson(回退)",
            accept_any_geojson=True,
        )

    def _conc(parts: list[gpd.GeoDataFrame], crs) -> gpd.GeoDataFrame:
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            return gpd.GeoDataFrame(geometry=[], crs=crs)
        out = pd.concat(parts, ignore_index=True)
        return gpd.GeoDataFrame(out, crs=crs)

    walk_g = _conc(walk_parts, ucrs)
    ring_g = _conc(ring_parts, ucrs)
    return walk_g, ring_g, notes


def _centroid_coverage_frac(
    units: gpd.GeoDataFrame,
    official_m: gpd.GeoDataFrame,
    supplement_m: gpd.GeoDataFrame,
    *,
    radius_m: float,
    include_ring_m: gpd.GeoDataFrame | None,
    rings_in_coverage: bool,
) -> tuple[float, float, float]:
    um = units.to_crs(CRS_M)
    pts = um.copy()
    pts["geometry"] = pts.geometry.centroid
    pts_m = pts[["geometry"]]

    def geoms_list(gdf: gpd.GeoDataFrame) -> list:
        if gdf is None or gdf.empty:
            return []
        gg = gdf.to_crs(CRS_M)
        return [g for g in gg.geometry if g is not None and not g.is_empty]

    g_o = geoms_list(official_m)
    g_s = geoms_list(supplement_m)
    g_r = geoms_list(include_ring_m) if rings_in_coverage and include_ring_m is not None else []

    def frac(geoms: list) -> float:
        if not geoms or len(pts_m) == 0:
            return 0.0
        U = unary_union(geoms)
        if U.is_empty:
            return 0.0
        d = pts_m.geometry.distance(U)
        return float((d <= float(radius_m)).mean())

    combined = g_o + g_s + g_r
    return frac(g_o), frac(combined), frac(g_s)


WALK_NET_COLOR = "#334155"


def _line_segments_for_plot(
    gdf: gpd.GeoDataFrame,
    *,
    match_crs,
    clip_ucrs: BaseGeometry,
) -> list[np.ndarray]:
    segs: list[np.ndarray] = []
    if gdf.empty:
        return segs
    G = gdf.copy()
    if G.crs != match_crs:
        G = G.to_crs(match_crs)
    try:
        G = G[G.intersects(clip_ucrs)].copy()
    except Exception:
        pass
    for geom in G.geometry:
        if geom is None or geom.is_empty:
            continue
        c = np.array(geom.coords)
        if len(c) >= 2:
            segs.append(c)
    return segs


def plot_enriched(
    *,
    units: gpd.GeoDataFrame,
    official_exp: gpd.GeoDataFrame,
    supplement_exp: gpd.GeoDataFrame,
    rings_exp: gpd.GeoDataFrame,
    site_path: Path | None,
    out_path: Path,
    meta_lines: dict,
    official_sources: list[str],
    supplement_notes: list[str],
    clip_poly_ucrs: BaseGeometry,
    plot_margin_frac: float,
    walk_color: str = WALK_NET_COLOR,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 10.5))
    u = units.copy()
    mf = max(float(plot_margin_frac), 0.0)

    try:
        u_vis = u[u.intersects(clip_poly_ucrs)].copy()
    except Exception:
        u_vis = u
    if u_vis.empty:
        u_vis = u
    u_vis.plot(ax=ax, facecolor="#f4f4f4", edgecolor="#bbbbbb", linewidth=0.15, zorder=0)

    all_segs: list[np.ndarray] = []
    for exp in (rings_exp, supplement_exp, official_exp):
        all_segs.extend(_line_segments_for_plot(exp, match_crs=u.crs, clip_ucrs=clip_poly_ucrs))
    if all_segs:
        ax.add_collection(
            LineCollection(all_segs, colors=walk_color, linewidths=0.95, alpha=0.88, zorder=2)
        )

    plot_site_boundary(ax, u.crs, site_path)

    minx, miny, maxx, maxy = clip_poly_ucrs.bounds
    dx = (maxx - minx) * mf
    dy = (maxy - miny) * mf
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)
    ax.set_aspect("equal")
    ax.axis("off")

    buf_m = float(meta_lines.get("site_buffer_m", 3000.0))
    cov_txt = (
        f"质心覆盖率（≤{meta_lines['coverage_radius_m']:.0f}m）：site官方 {meta_lines['coverage_official_only']*100:.1f}% → "
        f"合并(官方+all补充{rings_note(meta_lines)}) {meta_lines['coverage_combined']*100:.1f}% "
        f"（目标≥{meta_lines['target_frac']*100:.0f}%）\n"
        f"补充-only：{meta_lines['coverage_supplement_only']*100:.1f}% | extended_fclass={meta_lines['extended_road_fclass']}"
    )
    osrc = "\n".join(official_sources[:8])
    sup_note = "\n".join(supplement_notes[:14])
    if len(supplement_notes) > 14:
        sup_note += f"\n… 共 {len(supplement_notes)} 条摘要"

    ax.set_title(
        f"{N01_MOD_NAME} · 制图裁切：SITE 周边 {buf_m:.0f} m（单色步行路网）\n{cov_txt}\n"
        f"site 节选：\n{osrc}\n— all 补充摘要 —\n{sup_note}",
        fontsize=9,
        loc="left",
    )

    leg = [
        Line2D([0], [0], color=walk_color, lw=2.2, alpha=0.88, label="步行路网"),
        Line2D([0], [0], color="#999999", lw=1, label="地块单元"),
        Line2D([0], [0], color="#d90429", lw=2.2, linestyle=(0, (5, 3)), label="SITE 边界"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=175, bbox_inches="tight")
    plt.close(fig)


def rings_note(meta_lines: dict) -> str:
    return "+环路" if meta_lines.get("rings_in_coverage") else ""


def write_walk_assignment_segments_csv(
    *,
    official_exp: gpd.GeoDataFrame,
    supplement_exp: gpd.GeoDataFrame,
    rings_exp: gpd.GeoDataFrame,
    out_csv: Path,
) -> int:
    """写出可与 ``build_flow_road_assignment_edges --enriched-n01-segments-csv`` 合并的线段表（无接驳边）。"""
    from build_flow_road_assignment_edges import _segment_rows_from_lines

    parts: list[gpd.GeoDataFrame] = []
    for g in (official_exp, supplement_exp, rings_exp):
        if g is None or g.empty:
            continue
        parts.append(g.to_crs(CRS_M))
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not parts:
        pd.DataFrame().to_csv(out_csv, index=False, encoding="utf-8-sig")
        return 0
    merged = pd.concat(parts, ignore_index=True)
    lines_m = gpd.GeoDataFrame(merged, crs=CRS_M)
    rows = _segment_rows_from_lines(lines_m, N01_MOD_NAME)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="N01：site GeoJSON + all 目录步行矢量补充（无地块 proxy）")
    ap.add_argument("--units", type=Path, default=REPO / "output/function/数据包/01_units.gpkg")
    ap.add_argument("--data-root", type=Path, default=REPO / "data/site_3km")
    ap.add_argument(
        "--supplement-root",
        type=Path,
        action="append",
        default=None,
        help="补充数据根目录，可多次指定；省略则用默认三条 all 路径",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "output/flow/flow_modality_networks/N01_pedestrian_raw_layers_enriched.png",
    )
    ap.add_argument("--site-json", type=Path, default=None)
    ap.add_argument(
        "--site-buffer-m",
        type=float,
        default=3000.0,
        help="以 SITE 几何为中心缓冲半径 [m]，矢量与制图均裁切到此范围（无 SITE 时用单元 union 质心）",
    )
    ap.add_argument(
        "--plot-margin-frac",
        type=float,
        default=0.04,
        help="图幅在缓冲外包框基础上的留白比例（相对边长）",
    )
    ap.add_argument("--coverage-radius-m", type=float, default=70.0)
    ap.add_argument("--target-coverage", type=float, default=0.70)
    ap.add_argument("--max-file-mb", type=float, default=DEFAULT_MAX_FILE_MB)
    ap.add_argument("--bbox-pad-deg", type=float, default=0.012, help="read_file bbox 外扩 [deg] ~1.3km")
    ap.add_argument(
        "--extended-road-fclass",
        action="store_true",
        help="上海道路 shp：除 core 步行类外，纳入 residential/service 等（更易达标）",
    )
    ap.add_argument(
        "--rings-in-coverage",
        action="store_true",
        help="城市环路几何计入覆盖率（默认仅制图）",
    )
    ap.add_argument(
        "--no-ring-fallback",
        action="store_true",
        help="禁用环路 geojson 回退：不在 all 环路目录产出线段时使用 --data-root/16-城市环路",
    )
    ap.add_argument("--no-supplement", action="store_true", help="仅绘 site GeoJSON，不读 all")
    ap.add_argument(
        "--emit-walk-assignment-segments-csv",
        type=Path,
        default=None,
        help="写出慢行线段 CSV，供 build_flow_road_assignment_edges --enriched-n01-segments-csv 合并（与图中步行网一致）",
    )
    ns = ap.parse_args()

    roots = list(ns.supplement_root) if ns.supplement_root else [Path(p) for p in DEFAULT_SUPPLEMENT_ROOTS]

    units = _load_units(ns.units)
    site_path = ns.site_json if ns.site_json is not None and ns.site_json.is_file() else resolve_site_json_path()
    clip_poly_m, clip_poly_ucrs = site_buffer_polygons(units, site_path, radius_m=float(ns.site_buffer_m))

    frames, src_paths = _collect_n01_frames(ns.data_root, max_file_mb=float(ns.max_file_mb))
    official_exp = _official_exploded_lines(units, frames, clip_poly_m, clip_poly_ucrs)
    official_m = official_exp.to_crs(CRS_M) if len(official_exp) else gpd.GeoDataFrame(geometry=[], crs=CRS_M)

    supplement_exp = gpd.GeoDataFrame(geometry=[], crs=units.crs)
    rings_exp = gpd.GeoDataFrame(geometry=[], crs=units.crs)
    supplement_notes: list[str] = []

    if not ns.no_supplement:
        supplement_exp, rings_exp, supplement_notes = _load_supplement_layers(
            roots,
            units,
            clip_poly_m=clip_poly_m,
            extended_fclass=bool(ns.extended_road_fclass),
            max_file_mb=float(ns.max_file_mb),
            bbox_pad_deg=float(ns.bbox_pad_deg),
            ring_geojson_fallback_dir=(
                None
                if ns.no_supplement or ns.no_ring_fallback
                else Path(ns.data_root) / "16-城市环路"
            ),
        )

    sup_m = supplement_exp.to_crs(CRS_M) if len(supplement_exp) else gpd.GeoDataFrame(geometry=[], crs=CRS_M)
    ring_m = rings_exp.to_crs(CRS_M) if len(rings_exp) else gpd.GeoDataFrame(geometry=[], crs=CRS_M)

    cov_o, cov_c, cov_s = _centroid_coverage_frac(
        units,
        official_m,
        sup_m,
        radius_m=float(ns.coverage_radius_m),
        include_ring_m=ring_m,
        rings_in_coverage=bool(ns.rings_in_coverage),
    )

    meta_lines = {
        "coverage_radius_m": float(ns.coverage_radius_m),
        "target_frac": float(ns.target_coverage),
        "coverage_official_only": cov_o,
        "coverage_combined": cov_c,
        "coverage_supplement_only": cov_s,
        "extended_road_fclass": bool(ns.extended_road_fclass),
        "rings_in_coverage": bool(ns.rings_in_coverage),
        "n_supplement_segments": int(len(supplement_exp)),
        "n_ring_segments": int(len(rings_exp)),
        "meets_target": bool(cov_c >= float(ns.target_coverage) - 1e-9),
        "supplement_roots": [str(r) for r in roots],
        "site_buffer_m": float(ns.site_buffer_m),
    }

    if ns.emit_walk_assignment_segments_csv:
        n_em = write_walk_assignment_segments_csv(
            official_exp=official_exp,
            supplement_exp=supplement_exp,
            rings_exp=rings_exp,
            out_csv=Path(ns.emit_walk_assignment_segments_csv),
        )
        print(f"Walk assignment segments CSV: {ns.emit_walk_assignment_segments_csv} ({n_em} rows)")

    plot_enriched(
        units=units,
        official_exp=official_exp,
        supplement_exp=supplement_exp,
        rings_exp=rings_exp,
        site_path=site_path,
        out_path=Path(ns.out),
        meta_lines=meta_lines,
        official_sources=[p.name for p in src_paths],
        supplement_notes=supplement_notes,
        clip_poly_ucrs=clip_poly_ucrs,
        plot_margin_frac=float(ns.plot_margin_frac),
    )

    meta_path = Path(ns.out).with_suffix(".json")
    meta_out = {
        "script": "plot_N01_pedestrian_enriched.py",
        "png": str(Path(ns.out).resolve()),
        **meta_lines,
        "inputs": {
            "units": str(ns.units),
            "data_root": str(ns.data_root),
            "site_json": str(site_path) if site_path else None,
            "site_buffer_m": float(ns.site_buffer_m),
        },
        "supplement_notes": supplement_notes,
    }
    meta_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {ns.out}")
    print(json.dumps({k: meta_lines[k] for k in meta_lines if k != "supplement_roots"}, ensure_ascii=False, indent=2))
    if not meta_lines["meets_target"]:
        print(
            "警告：合并覆盖率低于目标。可尝试 --extended-road-fclass 或略增大 --coverage-radius-m。",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
