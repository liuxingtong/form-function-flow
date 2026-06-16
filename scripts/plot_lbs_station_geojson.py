"""
Plot FeatureCollection from parse_lbs_vitality_map.py (Point + footprint Polygon).

Writes a PNG with geometries and property annotations. If coordinates are WGS84
and geopandas+contextily+mattaplotlib are available, draws an OSM basemap underlay.

Example:
  python scripts/plot_lbs_station_geojson.py \\
      --geojson outputs/lbs_demo/lbs_demo.geojson \\
      --out outputs/lbs_demo/lbs_demo_map.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


def _configure_matplotlib_fonts() -> None:
    import platform

    if platform.system() == "Windows":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_plain(fc: dict, out_path: Path, title: str | None) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    props0 = {}
    for feat in fc.get("features", []):
        geom = feat.get("geometry") or {}
        pr = feat.get("properties") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        role = pr.get("feature_role", "")
        if not props0 and pr:
            props0 = pr
        if gtype == "Point" and coords:
            ax.scatter(coords[0], coords[1], c="crimson", s=120, zorder=5, edgecolors="white", linewidths=1.5)
        elif gtype == "Polygon" and coords:
            ring = coords[0]
            xy = [(x, y) for x, y in ring[:-1]]
            poly = MplPolygon(xy, closed=True, facecolor=(1, 0.6, 0.2, 0.25), edgecolor="darkorange", linewidth=2)
            ax.add_patch(poly)

    cs = props0.get("coordinate_space", "")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("longitude (°)" if cs == "WGS84" else "image x (px)")
    ax.set_ylabel("latitude (°)" if cs == "WGS84" else "image y (px)")
    ax.grid(True, alpha=0.35)
    ax.set_title(title or "LBS vitality parse — GeoJSON preview")

    lines = [
        props0.get("evaluation_framework", ""),
        props0.get("indicator", ""),
        f"t = {props0.get('timestamp_local', '')}",
        f"bin = {props0.get('vitality_bin', '')}  "
        f"[{props0.get('vitality_min', '')} … {props0.get('vitality_max', '')}]",
        f"pixel ({props0.get('query_pixel_x', '')}, {props0.get('query_pixel_y', '')})",
    ]
    txt = "\n".join(str(x) for x in lines if x)
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
        family="sans-serif",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_basemap(fc: dict, out_path: Path, title: str | None) -> bool:
    try:
        import contextily as ctx
        import geopandas as gpd
    except ImportError:
        return False

    props0: dict = {}
    for feat in fc.get("features", []):
        pr = feat.get("properties") or {}
        if pr:
            props0 = pr
            break

    try:
        gdf = gpd.GeoDataFrame.from_features(fc.get("features", []), crs="EPSG:4326")
    except Exception:
        return False

    if len(gdf) == 0:
        return False

    g3857 = gdf.to_crs(3857)
    fig, ax = plt.subplots(figsize=(10, 9))
    polys = g3857[g3857.geom_type != "Point"]
    pts = g3857[g3857.geom_type == "Point"]
    if len(polys):
        polys.plot(ax=ax, facecolor=(1.0, 0.6, 0.2, 0.35), edgecolor="darkorange", linewidth=2)
    if len(pts):
        pts.plot(ax=ax, color="crimson", markersize=90, edgecolor="white", linewidth=1.2)
    try:
        ctx.add_basemap(ax, crs=g3857.crs, source=ctx.providers.CartoDB.Positron)
    except Exception:
        ctx.add_basemap(ax, crs=g3857.crs)

    ax.set_axis_off()
    cs = props0.get("coordinate_space", "")
    ttl = title or ("LBS vitality parse (WGS84 + basemap)" if cs == "WGS84" else "LBS vitality parse")
    ax.set_title(ttl)
    lines = [
        props0.get("evaluation_framework", ""),
        props0.get("indicator", ""),
        f"t = {props0.get('timestamp_local', '')}",
        f"bin = {props0.get('vitality_bin', '')}",
        f"range ≈ {props0.get('vitality_min', '')} — {props0.get('vitality_max', '')}",
    ]
    txt = "\n".join(str(x) for x in lines if x)
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    _configure_matplotlib_fonts()
    ap = argparse.ArgumentParser(description="Plot GeoJSON from parse_lbs_vitality_map.")
    ap.add_argument("--geojson", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="PNG path (default: same stem + _map.png)")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--no-basemap", action="store_true", help="Skip contextily even if WGS84.")
    args = ap.parse_args()

    path = args.geojson.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Missing {path}")

    with open(path, encoding="utf-8") as f:
        fc = json.load(f)

    if fc.get("type") != "FeatureCollection":
        raise SystemExit("Expected FeatureCollection")

    out = args.out
    if out is None:
        out = path.with_name(path.stem + "_map.png")

    props = (fc.get("features") or [{}])[0].get("properties") or {}
    wgs = props.get("coordinate_space") == "WGS84" and not args.no_basemap

    if wgs and _plot_basemap(fc, out, args.title):
        print(f"Wrote basemap PNG: {out}")
        return

    _plot_plain(fc, out, args.title)
    print(f"Wrote PNG (no basemap): {out}")


if __name__ == "__main__":
    main()
