"""
在底图上叠加「原始矢量交通图层」（GeoJSON），按文件名规则分为四类并分别出图；
SITE 范围内线段稀少或过短的片段用红虚线标为结构性薄弱（几何可读 proxy，非 OD 仿真）。

不使用地块邻接边权 proxy。默认仅在以下目录递归 *.geojson（可用 --scan-all-under-root 扩大至整个 data-root）：
  data/site_3km/04-交通数据/
  data/site_3km/metroflow/

用法：
  python scripts/plot_flow_modality_networks.py ^
    --units data/site_3km/01_units.gpkg ^
    --data-root data/site_3km ^
    --out-dir output/flow/flow_modality_networks
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
from shapely.geometry import LineString, MultiLineString, Point, box

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from site_map_overlay import load_site_gdf, plot_site_boundary, resolve_site_json_path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# 单次读入上限，避免 --scan-all-under-root 时把百万级 POI 整块 concat 导致 OOM
DEFAULT_MAX_FILE_MB = 30.0

# (类别显示名, glob 子串列表；命中任一即归入该类；按顺序匹配，先到先得）
# 不含专用「自行车道」图层（文件名 stem 含「自行车」单独剔除），步行制图与分配对齐 enriched N01。
MODALITY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("轨道与站点（原始矢量）", ("地铁", "轨道", "铁路", "metroflow", "station", "站点")),
    (
        "慢行与人行（原始矢量）",
        ("行人", "步行", "慢行", "绿道", "运动场跑道", "跑道", "轮渡", "渡口", "人渡口"),
    ),
    ("快速路主干（原始矢量）", ("快速路", "高速", "主干道")),
    ("其它机动车道路（原始矢量）", ("公路", "道路", "街坊", "次级", "其它", "一级", "市区")),
]


def is_dedicated_bicycle_geojson_path(path: Path) -> bool:
    """专用自行车道等图层（stem 含「自行车」），不参与慢行步行制图/分配。"""
    return "自行车" in path.stem


def configure_cn_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


configure_cn_font()


def _collect_geojsons(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        out.extend(root.rglob("*.geojson"))
    uniq = []
    seen = set()
    for p in sorted(set(out)):
        key = str(p.resolve())
        if key in seen:
            continue
        if p.stat().st_size > 80 * 1024 * 1024:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _modality_match_string(path: Path) -> str:
    """仅用仓库内相对路径匹配，避免上级目录 ``.../shanghaistation/...`` 误命中英文 ``station``。"""
    try:
        rel = path.resolve().relative_to(REPO.resolve())
    except ValueError:
        parts: list[str] = [path.stem.lower()]
        cur = path.parent
        for _ in range(10):
            if cur.name:
                parts.append(cur.name.lower())
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
        return "/".join(parts)
    segs = [*(s.lower() for s in rel.parent.parts if s not in ("", ".")), rel.stem.lower()]
    return "/".join(segs)


def _assign_modality(path: Path) -> str | None:
    s = _modality_match_string(path)
    for name, keys in MODALITY_RULES:
        if any(k.lower() in s for k in keys):
            return name
    return None


def _explode_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows = []
    for _, r in gdf.iterrows():
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            rows.append(LineString(geom.coords))
        elif isinstance(geom, MultiLineString):
            for g in geom.geoms:
                rows.append(LineString(g.coords))
        elif geom.geom_type == "GeometryCollection":
            for g in geom.geoms:
                if isinstance(g, LineString):
                    rows.append(LineString(g.coords))
                elif isinstance(g, MultiLineString):
                    for gg in g.geoms:
                        rows.append(LineString(gg.coords))
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    return gpd.GeoDataFrame(geometry=rows, crs=gdf.crs)


def _points_xy(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, Point):
            xs.append(geom.x)
            ys.append(geom.y)
    return np.array(xs), np.array(ys)


def _clip_to_bbox(gdf: gpd.GeoDataFrame, bbox_geom) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    try:
        return gdf[gdf.intersects(bbox_geom)].copy()
    except Exception:
        return gdf


def _segment_lengths_deg(gdf: gpd.GeoDataFrame) -> np.ndarray:
    lens = []
    for geom in gdf.geometry:
        if geom is not None and not geom.is_empty:
            lens.append(float(geom.length))
    return np.array(lens, dtype=float) if lens else np.array([], dtype=float)


def plot_modality(
    title: str,
    lines: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
    site_path: Path | None,
    out_path: Path,
    *,
    weak_pct: float,
    sources: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 10.5))
    u = units.copy()
    if u.crs is None:
        u.set_crs(4326, inplace=True)
    bb = u.total_bounds
    pad = max((bb[2] - bb[0]), (bb[3] - bb[1])) * 0.06
    clip_geom = box(bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)

    u.plot(ax=ax, facecolor="#f4f4f4", edgecolor="#bbbbbb", linewidth=0.15, zorder=0)

    site = load_site_gdf(site_path)
    site_buf = None
    if site is not None and len(site):
        sg = site.copy()
        if sg.crs != u.crs:
            sg = sg.to_crs(u.crs)
        geom = sg.geometry
        site_union = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
        try:
            site_buf = site_union.buffer(pad * 0.35)
        except Exception:
            site_buf = site_union

    segs = []
    lens = []
    touch_site = []
    weak_notes = ""

    if not lines.empty:
        L = lines.copy(deep=False)
        if L.crs != u.crs:
            L = L.to_crs(u.crs)
        L = _clip_to_bbox(L, clip_geom)
        gt = L.geometry.geom_type
        pts_g = L[gt == "Point"].copy()
        line_mask = gt.isin(["LineString", "MultiLineString", "GeometryCollection"])
        Llines = L[line_mask].copy()
        Lexp = _explode_lines(Llines) if not Llines.empty else gpd.GeoDataFrame(geometry=[], crs=u.crs)
        px, py = _points_xy(pts_g)
        weak_pts = np.zeros(len(px), dtype=bool)
        if px.size and site_buf is not None:
            try:
                from shapely.geometry import Point as ShpPoint

                for i in range(len(px)):
                    weak_pts[i] = ShpPoint(float(px[i]), float(py[i])).distance(site_buf) < pad * 0.5
            except Exception:
                weak_pts = np.zeros(len(px), dtype=bool)

        if px.size:
            ax.scatter(
                px[~weak_pts],
                py[~weak_pts],
                s=28,
                c="#2563eb",
                edgecolors="#1e3a8a",
                linewidths=0.35,
                zorder=4,
                label="站点/点要素",
            )
            if weak_pts.any():
                ax.scatter(
                    px[weak_pts],
                    py[weak_pts],
                    s=55,
                    c="#d90429",
                    edgecolors="#590d22",
                    linewidths=0.6,
                    zorder=5,
                    marker="o",
                    label="SITE 邻近薄弱点",
                )

        if not Lexp.empty:
            lens_arr = _segment_lengths_deg(Lexp)
            for geom in Lexp.geometry:
                if geom is None or geom.is_empty:
                    continue
                c = np.array(geom.coords)
                if len(c) < 2:
                    continue
                segs.append(c)
                lens.append(float(geom.length))
                inside = False
                if site_buf is not None:
                    try:
                        inside = geom.intersects(site_buf)
                    except Exception:
                        inside = False
                touch_site.append(inside)
            lens = np.array(lens, dtype=float)
            touch_site = np.array(touch_site, dtype=bool)
            if lens.size and touch_site.any():
                cand = lens[touch_site]
                thr = max(float(np.percentile(cand, weak_pct)), 1e-9)
                weak = touch_site & (lens <= thr)
            else:
                weak = np.zeros(len(lens), dtype=bool)
            weak_notes = "线：SITE 邻域内≈P%d%%短片段标红虚线；点：距 SITE 边界缓冲近者标红" % int(weak_pct)

            if segs:
                lc = LineCollection(segs, colors="#2563eb", linewidths=1.15, alpha=0.78, zorder=2)
                ax.add_collection(lc)
            if weak.any():
                wsegs = [segs[i] for i in range(len(segs)) if weak[i]]
                lcw = LineCollection(wsegs, colors="#d90429", linewidths=2.6, alpha=0.95, linestyles="dashed", zorder=3)
                ax.add_collection(lcw)
        elif px.size:
            weak_notes = "仅点要素（站域等）；红圈示 SITE 邻近薄弱点"
        else:
            weak_notes = "裁剪后无线段或点"
    else:
        weak_notes = "未加载到几何"

    _ = plot_site_boundary(ax, u.crs, site_path)
    ax.set_xlim(bb[0] - pad, bb[2] + pad)
    ax.set_ylim(bb[1] - pad, bb[3] + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    src_txt = "\n".join(sources[:12])
    if len(sources) > 12:
        src_txt += f"\n… 共 {len(sources)} 个文件"
    ax.set_title(f"{title}\n{weak_notes}\n图层来源（节选）：\n{src_txt}", fontsize=10, loc="left")

    leg = [
        Line2D([0], [0], color="#2563eb", lw=2, label="原始线位"),
        Line2D([0], [0], color="#d90429", lw=2.4, linestyle=(0, (5, 3)), label="SITE 邻域薄弱片段"),
        Line2D([0], [0], color="#999", lw=1, label="地块单元（浅灰）"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_placeholder(title: str, msg: str, units: gpd.GeoDataFrame, site_path: Path | None, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    u = units.copy()
    if u.crs is None:
        u.set_crs(4326, inplace=True)
    u.plot(ax=ax, facecolor="#eee", edgecolor="#ccc", linewidth=0.2)
    plot_site_boundary(ax, u.crs, site_path)
    ax.set_title(title, fontsize=12)
    ax.text(0.02, 0.98, msg, transform=ax.transAxes, va="top", fontsize=10, color="#444")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="原始 GeoJSON 交通图层四分图（非 proxy）")
    ap.add_argument("--units", type=Path, default=Path("data/site_3km/01_units.gpkg"))
    ap.add_argument("--data-root", type=Path, default=Path("data/site_3km"))
    ap.add_argument(
        "--scan-all-under-root",
        action="store_true",
        help="递归扫描整个 data-root（慢）；默认仅扫描 04-交通数据 与 metroflow",
    )
    ap.add_argument("--extra-root", type=Path, nargs="*", default=[], help="额外搜集目录")
    ap.add_argument("--out-dir", type=Path, default=Path("output/flow/flow_modality_networks"))
    ap.add_argument("--site-json", type=Path, default=None)
    ap.add_argument("--weak-pct", type=float, default=20.0)
    ap.add_argument(
        "--max-file-mb",
        type=float,
        default=DEFAULT_MAX_FILE_MB,
        help=f"跳过大于该尺寸的 geojson（默认 {DEFAULT_MAX_FILE_MB} MB），防全站扫描 OOM",
    )
    ns = ap.parse_args()

    roots: list[Path] = []
    if ns.scan_all_under_root:
        roots.append(ns.data_root.resolve())
    else:
        for sub in ("04-交通数据", "metroflow"):
            p = (ns.data_root / sub).resolve()
            if p.is_dir():
                roots.append(p)
        if not roots:
            roots.append(ns.data_root.resolve())
    roots.extend([Path(p).resolve() for p in ns.extra_root])
    site_path = ns.site_json if ns.site_json is not None and ns.site_json.is_file() else resolve_site_json_path()

    try:
        units = gpd.read_file(ns.units, layer="units")
    except Exception:
        units = gpd.read_file(ns.units)

    all_paths = _collect_geojsons(roots)
    buckets: dict[str, list[Path]] = {name: [] for name, _ in MODALITY_RULES}
    unassigned: list[Path] = []
    for p in all_paths:
        if is_dedicated_bicycle_geojson_path(p):
            continue
        mod = _assign_modality(p)
        if mod is None:
            unassigned.append(p)
            continue
        buckets[mod].append(p)

    out_files = {
        "轨道与站点（原始矢量）": "N04_transit_raw_layers.png",
        "慢行与人行（原始矢量）": "N01_pedestrian_raw_layers.png",
        "快速路主干（原始矢量）": "N02_fast_road_raw_layers.png",
        "其它机动车道路（原始矢量）": "N03_slow_road_raw_layers.png",
    }

    ns.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"roots": [str(r) for r in roots], "modalities": {}, "unassigned_sample": [str(p) for p in unassigned[:40]]}

    for mod_name, _keys in MODALITY_RULES:
        paths = buckets[mod_name]
        manifest["modalities"][mod_name] = []
        for p in paths:
            rp = str(p.resolve())
            try:
                manifest["modalities"][mod_name].append(str(p.relative_to(REPO)))
            except ValueError:
                manifest["modalities"][mod_name].append(rp)
        if not paths:
            plot_placeholder(
                mod_name,
                "未在 data-root 下匹配到文件名关键字。\n请将快速路/主干道/行人路/轨道相关 GeoJSON 放入 data/site_3km/04-交通数据 等目录后重跑。",
                units,
                site_path,
                ns.out_dir / out_files[mod_name],
            )
            print("Placeholder:", ns.out_dir / out_files[mod_name])
            continue
        frames = []
        max_bytes = int(max(1.0, float(ns.max_file_mb)) * 1024 * 1024)
        for p in paths:
            try:
                if p.stat().st_size > max_bytes:
                    continue
                g = gpd.read_file(p)
                if len(g) == 0:
                    continue
                if g.crs is None:
                    g.set_crs(4326, inplace=True)
                frames.append(g)
            except Exception:
                continue
        if not frames:
            plot_placeholder(mod_name, "文件存在但解析失败或为空。", units, site_path, ns.out_dir / out_files[mod_name])
            print("Placeholder (read fail):", ns.out_dir / out_files[mod_name])
            continue
        merged = pd.concat(frames, ignore_index=True)
        merged = gpd.GeoDataFrame(merged, crs=frames[0].crs)
        plot_modality(
            mod_name,
            merged,
            units,
            site_path,
            ns.out_dir / out_files[mod_name],
            weak_pct=ns.weak_pct,
            sources=[p.name for p in paths],
        )
        print("Wrote", ns.out_dir / out_files[mod_name])

    (ns.out_dir / "flow_modality_network_meta.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
