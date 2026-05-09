"""
QA maps for 01_units.gpkg + 02_edges.csv: ring zones, adjacency overlay, degree histogram,
and edge-attribute choropleth-style line maps (conductance / cost).

Writes to data/site_3km/qa/ (PNG + qa_summary.json). Includes temporal diagnostics from
poi_temporal_synthesis.json and optional metroflow/time_slice_calibration.json.

Run from repo root:
  python scripts/visualize_units_edges_qa.py
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

REPO = Path(__file__).resolve().parents[1]
SITE_3KM = REPO / "data" / "site_3km"
DEFAULT_UNITS = SITE_3KM / "01_units.gpkg"
DEFAULT_EDGES = SITE_3KM / "02_edges.csv"
DEFAULT_OUT_DIR = SITE_3KM / "qa"
DEFAULT_STATION = (121.451257271, 31.249149419)
DEFAULT_SITE_JSON = SITE_3KM / "SITE.json"
T_IDS = ("WD_AM", "WD_PM", "WD_EVE", "WE_PM")

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from site_map_overlay import plot_site_boundary  # noqa: E402


def _edge_segments(sub: pd.DataFrame, cents_wgs84: pd.Series) -> list[list[tuple[float, float]]]:
    segs: list[list[tuple[float, float]]] = []
    for _, r in sub.iterrows():
        a, b = r["source_id"], r["target_id"]
        if a not in cents_wgs84.index or b not in cents_wgs84.index:
            continue
        ca, cb = cents_wgs84.loc[a], cents_wgs84.loc[b]
        segs.append([(float(ca.x), float(ca.y)), (float(cb.x), float(cb.y))])
    return segs


def plot_temporal_site_qa(out_dir: Path, units_path: Path) -> dict[str, str]:
    """POI diurnal curves, slice mass shares, flow proxy; optional MetroFlow blend."""
    site = units_path.parent
    paths: dict[str, str] = {}
    poi_path = site / "poi_temporal_synthesis.json"
    if not poi_path.is_file():
        return paths
    data = json.loads(poi_path.read_text(encoding="utf-8"))
    wd = data["curves"]["weekday"]
    we = data["curves"]["weekend"]
    h = list(range(24))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(h, wd, label="weekday (POI synth)", color="#1f77b4", lw=2)
    ax.plot(h, we, label="weekend (POI synth)", color="#ff7f0e", lw=2)
    ax.set_xlabel("hour (local)")
    ax.set_ylabel("normalized mass / h")
    ax.set_title("POI-based diurnal curves (full 24h)")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 23)
    fig.tight_layout()
    p = out_dir / "temporal_wd_we_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["temporal_wd_we_curves_png"] = str(p).replace("\\", "/")

    slices = data.get("slices") or []
    if slices:
        tids = [s["t_id"] for s in slices]
        mwd = [float(s["mass_share_weekday_in_slice"]) for s in slices]
        mwe = [float(s["mass_share_weekend_in_slice"]) for s in slices]
        x = np.arange(len(tids))
        w = 0.36
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - w / 2, mwd, w, label="weekday mass in slice", color="#1f77b4")
        ax.bar(x + w / 2, mwe, w, label="weekend mass in slice", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(tids)
        ax.set_ylabel("mass share")
        ax.set_title("Four clock slices: WD vs WE curve mass in window")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        p2 = out_dir / "temporal_slice_mass_wd_we.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        paths["temporal_slice_mass_wd_we_png"] = str(p2).replace("\\", "/")

    fp = data.get("flow_proxy_period_weights", {})
    wk = fp.get("weekday") or {}
    if wk:
        inf = [float(wk[t]["period_inflow_weight"]) for t in T_IDS if t in wk]
        if len(inf) == 4:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(T_IDS, inf, color="#2ca02c", edgecolor="white")
            ax.set_ylabel("period_inflow_weight")
            ax.set_title("Flow proxy (weekday): inflow weight by t_id")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            p3 = out_dir / "temporal_flow_inflow_weight_weekday.png"
            fig.savefig(p3, dpi=150)
            plt.close(fig)
            paths["temporal_flow_inflow_weight_weekday_png"] = str(p3).replace("\\", "/")

    cal_path = site / "metroflow" / "time_slice_calibration.json"
    if cal_path.is_file():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        syn_w = cal.get("synthetic_flow_proxy_period_weights_ref", {}).get("weekday", {})
        bl_w = cal.get("flow_proxy_period_weights_blended", {}).get("weekday", {})
        if isinstance(bl_w, dict) and syn_w:
            syn_m = [float(syn_w[t]["curve_mass_share"]) for t in T_IDS if t in syn_w]
            bl_m = [float(bl_w[t]["curve_mass_share"]) for t in T_IDS if t in bl_w]
            if len(syn_m) == 4 and len(bl_m) == 4:
                x = np.arange(4)
                w = 0.36
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.bar(x - w / 2, syn_m, w, label="POI synthetic", color="#7f7f7f")
                ax.bar(x + w / 2, bl_m, w, label="blended (POI+MetroFlow)", color="#9467bd")
                ax.set_xticks(x)
                ax.set_xticklabels(T_IDS)
                ax.set_ylabel("curve_mass_share")
                ax.set_title("Weekday slice mass: synthetic vs MetroFlow blend")
                ax.legend()
                ax.grid(axis="y", alpha=0.25)
                fig.tight_layout()
                p4 = out_dir / "temporal_metroflow_blend_mass_weekday.png"
                fig.savefig(p4, dpi=150)
                plt.close(fig)
                paths["temporal_metroflow_blend_mass_weekday_png"] = str(p4).replace("\\", "/")

    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    ap.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--station-lon", type=float, default=DEFAULT_STATION[0])
    ap.add_argument("--station-lat", type=float, default=DEFAULT_STATION[1])
    ap.add_argument(
        "--site-json",
        type=Path,
        default=DEFAULT_SITE_JSON,
        help="场地红线 GeoJSON（默认 data/SITE.json），叠在地理 QA 图上；不存在则跳过。",
    )
    ap.add_argument(
        "--edge-color-pct",
        type=float,
        default=98.0,
        help="Color scale caps at this percentile (robust to outliers).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        units = gpd.read_file(args.units, layer="units")
    except Exception:
        units = gpd.read_file(args.units)
    edges = pd.read_csv(args.edges)
    units_idx = units.set_index("unit_id")

    # Centroids in projected CRS → WGS84 for correct map geometry (avoid geographic-centroid bias).
    u_m = units_idx.to_crs("EPSG:32651")
    cents_wgs84 = gpd.GeoDataFrame(geometry=u_m.geometry.centroid, crs="EPSG:32651").to_crs(
        "EPSG:4326"
    )["geometry"]
    fig, ax = plt.subplots(figsize=(9, 9))
    units.plot(
        column="ring_zone",
        ax=ax,
        legend=True,
        categorical=True,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.15,
        legend_kwds={"title": "ring_zone", "loc": "lower left"},
    )
    ax.scatter(
        [args.station_lon],
        [args.station_lat],
        c="red",
        s=120,
        marker="*",
        zorder=5,
        label="station ref",
        edgecolors="white",
    )
    site_ok = plot_site_boundary(ax, units.crs, args.site_json)
    ax.set_title("01_units: ring_zone + station")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    h0, l0 = ax.get_legend_handles_labels()
    if site_ok:
        h0.append(Line2D([0], [0], color="#d90429", lw=2.2, linestyle=(0, (5, 3))))
        l0.append("场地红线 (SITE.json)")
    ax.legend(h0, l0, loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()
    p1 = args.out_dir / "units_ring_zones.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # --- Fig 2: edge kinds (LineCollection for speed) ---
    ed = edges[edges["source_id"] < edges["target_id"]].copy()

    fig, ax = plt.subplots(figsize=(9, 9))
    units.plot(ax=ax, color="#e8e8e8", edgecolor="white", linewidth=0.12)

    layers = (
        ("parcel_touch", "#4a4a4a", 0.28, 0.07, 2),
        ("proximity_bridge", "#1f77b4", 0.42, 0.22, 3),
        ("knn_bridge", "#c0392b", 1.0, 0.72, 4),
    )
    for kind, color, lw, alpha, z in layers:
        sub = ed[ed["edge_kind"] == kind]
        segs = _edge_segments(sub, cents_wgs84)
        if not segs:
            continue
        lc = LineCollection(segs, colors=color, linewidths=lw, alpha=alpha, zorder=z)
        ax.add_collection(lc)

    ax.scatter(
        [args.station_lon],
        [args.station_lat],
        c="red",
        s=120,
        marker="*",
        zorder=6,
        edgecolors="white",
    )
    site_ok2 = plot_site_boundary(ax, units.crs, args.site_json)
    ax.set_xlim(units.total_bounds[0], units.total_bounds[2])
    ax.set_ylim(units.total_bounds[1], units.total_bounds[3])
    ax.set_title("02_edges: parcel_touch (gray) + proximity_bridge (blue) + knn_bridge (red)")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal")
    leg = [
        Line2D([0], [0], color="#4a4a4a", lw=2, alpha=0.6, label="parcel_touch"),
        Line2D([0], [0], color="#1f77b4", lw=2, alpha=0.7, label="proximity_bridge"),
        Line2D([0], [0], color="#c0392b", lw=2, alpha=0.9, label="knn_bridge"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=12, linestyle="None", label="station ref"),
    ]
    if site_ok2:
        leg.append(Line2D([0], [0], color="#d90429", lw=2.2, linestyle=(0, (5, 3)), label="场地红线 (SITE.json)"))
    ax.legend(handles=leg, loc="upper right")
    fig.tight_layout()
    p2 = args.out_dir / "units_edges_touch_vs_bridge.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # --- Fig 3: degree (outgoing edges per source) ---
    deg = edges.groupby("source_id").size()
    dmax = int(deg.max())
    fig, ax = plt.subplots(figsize=(7, 4))
    if dmax <= 80:
        bins = range(0, dmax + 2)
    else:
        bins = np.linspace(0, dmax, 81)
    ax.hist(deg.values, bins=bins, color="steelblue", edgecolor="white")
    ax.set_title("Out-degree distribution (directed edges per unit_id)")
    ax.set_xlabel("degree")
    ax.set_ylabel("count")
    fig.tight_layout()
    p3 = args.out_dir / "degree_histogram.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)

    # --- Fig 4: area distribution (log) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    a = units["area"].clip(lower=0.01)
    ax.hist(a, bins=50, color="seagreen", edgecolor="white")
    ax.set_xlabel("area (m²)")
    ax.set_ylabel("count")
    ax.set_title("Unit polygon area (m²)")
    fig.tight_layout()
    p4 = args.out_dir / "unit_area_histogram.png"
    fig.savefig(p4, dpi=140)
    plt.close(fig)

    # --- Fig 5–6: edge conductance / edge_cost (undirected unique rows, color = value) ---
    ed_attr = edges[edges["source_id"] < edges["target_id"]].copy()

    def plot_edge_choropleth(
        column: str,
        cmap: str,
        title: str,
        cbar_label: str,
        out_name: str,
        invert_cmap: bool = False,
    ) -> Path:
        segs: list[list[tuple[float, float]]] = []
        vals: list[float] = []
        for _, r in ed_attr.iterrows():
            a, b = r["source_id"], r["target_id"]
            if a not in cents_wgs84.index or b not in cents_wgs84.index:
                continue
            ca, cb = cents_wgs84.loc[a], cents_wgs84.loc[b]
            segs.append([(float(ca.x), float(ca.y)), (float(cb.x), float(cb.y))])
            vals.append(float(r[column]))
        if not segs:
            raise SystemExit(f"No segments for edge plot ({column})")
        v = np.array(vals, dtype=float)
        lo, hi = float(np.min(v)), float(np.percentile(v, args.edge_color_pct))
        if hi <= lo:
            hi = float(np.max(v))
        norm = plt.Normalize(vmin=lo, vmax=hi, clip=True)
        cmap_obj = plt.get_cmap(cmap)
        if invert_cmap:
            cmap_obj = cmap_obj.reversed()
        fig, ax = plt.subplots(figsize=(10, 10))
        units.plot(ax=ax, color="#f5f5f5", edgecolor="white", linewidth=0.1)
        lc = LineCollection(
            segs,
            cmap=cmap_obj,
            norm=norm,
            array=v,
            linewidths=0.85,
            alpha=0.92,
        )
        ax.add_collection(lc)
        ax.scatter(
            [args.station_lon],
            [args.station_lat],
            c="red",
            s=100,
            marker="*",
            zorder=6,
            edgecolors="white",
        )
        site_ok_e = plot_site_boundary(ax, units.crs, args.site_json)
        ax.set_xlim(units.total_bounds[0], units.total_bounds[2])
        ax.set_ylim(units.total_bounds[1], units.total_bounds[3])
        ax.set_aspect("equal")
        ax.set_title(title + f" (color cap p{args.edge_color_pct:g})")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        cbar = fig.colorbar(lc, ax=ax, shrink=0.55, label=cbar_label)
        cbar.ax.tick_params(labelsize=9)
        lh = [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="red",
                markersize=11,
                linestyle="None",
                label="station ref",
            ),
        ]
        if site_ok_e:
            lh.append(Line2D([0], [0], color="#d90429", lw=2.2, linestyle=(0, (5, 3)), label="场地红线 (SITE.json)"))
        ax.legend(handles=lh, loc="lower right", fontsize=8, framealpha=0.92)
        fig.tight_layout()
        outp = args.out_dir / out_name
        fig.savefig(outp, dpi=170)
        plt.close(fig)
        return outp

    p5 = plot_edge_choropleth(
        "edge_conductance",
        "viridis",
        "Undirected edges: edge_conductance",
        "edge_conductance (exp(-θ·cost)); knn/proximity scaled",
        "edges_choropleth_conductance.png",
        invert_cmap=False,
    )
    p6 = plot_edge_choropleth(
        "edge_cost",
        "magma",
        "Undirected edges: edge_cost",
        "edge_cost (walk + barriers)",
        "edges_choropleth_edge_cost.png",
        invert_cmap=False,
    )

    temporal_paths = plot_temporal_site_qa(args.out_dir, args.units)

    summary = {
        "n_units": int(len(units)),
        "n_directed_edges": int(len(edges)),
        "edge_kind_directed": edges["edge_kind"].value_counts().to_dict(),
        "degree_min": int(deg.min()),
        "degree_max": int(deg.max()),
        "degree_median": float(deg.median()),
        "area_m2_min": float(units["area"].min()),
        "area_m2_median": float(units["area"].median()),
        "outputs": {
            "ring_zones_png": str(p1).replace("\\", "/"),
            "touch_vs_bridge_png": str(p2).replace("\\", "/"),
            "degree_histogram_png": str(p3).replace("\\", "/"),
            "area_histogram_png": str(p4).replace("\\", "/"),
            "edges_choropleth_conductance_png": str(p5).replace("\\", "/"),
            "edges_choropleth_edge_cost_png": str(p6).replace("\\", "/"),
        },
        "temporal_outputs": temporal_paths,
        "edge_conductance_undirected": {
            "min": float(ed_attr["edge_conductance"].min()),
            "p50": float(ed_attr["edge_conductance"].median()),
            "p98": float(np.percentile(ed_attr["edge_conductance"], 98)),
            "max": float(ed_attr["edge_conductance"].max()),
        },
        "edge_cost_undirected": {
            "min": float(ed_attr["edge_cost"].min()),
            "p50": float(ed_attr["edge_cost"].median()),
            "p98": float(np.percentile(ed_attr["edge_cost"], 98)),
            "max": float(ed_attr["edge_cost"].max()),
        },
    }
    (args.out_dir / "qa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Wrote", p1)
    print("Wrote", p2)
    print("Wrote", p3)
    print("Wrote", p4)
    print("Wrote", p5)
    print("Wrote", p6)
    print("Wrote", args.out_dir / "qa_summary.json")
    for _k, _p in temporal_paths.items():
        print("Wrote", _p)


if __name__ == "__main__":
    main()
