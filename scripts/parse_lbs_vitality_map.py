"""
Parse an LBS vitality choropleth slide image: extract legend colors via k-means,
map a query pixel (Shanghai Station area) to the vitality numeric bin, and save
metadata as JSON, CSV, and GeoJSON (FeatureCollection: Point + footprint Polygon).

Dependencies: numpy, opencv-python (or opencv-python-headless)

Examples:
  # Click once on the map at 上海站附近 — saves outputs next to the image
  python scripts/parse_lbs_vitality_map.py --image path/to/slide.png --click

  # Known pixel coordinates (OpenCV x,y from left-top)
  python scripts/parse_lbs_vitality_map.py --image slide.png --qx 820 --qy 540 \\
      --legend-roi 0.02,0.52,0.38,0.45

  # Use baked gradient fallback if legend crop is unreliable
  python scripts/parse_lbs_vitality_map.py --image slide.png --qx 800 --qy 520 --fallback-palette

  # GeoJSON with WGS84 coordinates (linear bbox georef for whole image)
  python scripts/parse_lbs_vitality_map.py --image slide.png --qx 820 --qy 540 \\
      --bbox 121.44,31.23,121.48,31.27
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as e:
    raise SystemExit("Install OpenCV: pip install opencv-python-headless") from e


# ---------------------------------------------------------------------------
# Numeric bins transcribed from slide legend (低 → 高 vitality)
# Edit if your slide uses different breaks.
# ---------------------------------------------------------------------------
VITALITY_BINS: list[tuple[float, float]] = [
    (0.0, 2_327_425.891300),
    (2_327_425.891301, 4_473_550.063190),
    (4_473_550.063191, 6_075_534.413240),
    (6_075_534.413241, 8_165_560.208590),
    (8_165_560.208591, 11_316_459.171300),
    (11_316_459.171301, 15_739_382.875000),
    (15_739_382.875001, 22_638_564.838100),
    (22_638_564.838101, 32_984_451.099200),
    (32_984_451.099201, 49_394_305.289300),
]


@dataclass
class ParseResult:
    evaluation_framework: str
    indicator: str
    timestamp_local: str
    image_path: str
    query_pixel_xy: tuple[int, int]
    patch_radius_px: int
    matched_bin_index: int  # 1..9
    vitality_min: float
    vitality_max: float
    lab_distance_to_center: float
    legend_mode: str
    notes: str


def _parse_roi(s: str, w: int, h: int) -> tuple[int, int, int, int]:
    """ROI string 'x,y,width,height' in pixels OR normalized 0–1 if all values ≤1."""
    parts = [float(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,width,height")
    x, y, rw, rh = parts
    if max(parts) <= 1.0 + 1e-6:
        x, y = int(x * w), int(y * h)
        rw, rh = int(rw * w), int(rh * h)
    else:
        x, y, rw, rh = int(x), int(y), int(rw), int(rh)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(10, min(rw, w - x))
    rh = max(10, min(rh, h - y))
    return x, y, rw, rh


def _mask_colorful_pixels(bgr: np.ndarray, s_min: int = 40, v_min: int = 40) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    return (s > s_min) & (v > v_min)


def _sample_pixels_masked(patch: np.ndarray, mask: np.ndarray, max_n: int = 20_000) -> np.ndarray:
    pts = patch[mask]
    if len(pts) == 0:
        return patch.reshape(-1, 3)
    if len(pts) > max_n:
        idx = np.random.choice(len(pts), max_n, replace=False)
        pts = pts[idx]
    return pts.astype(np.float32)


def _kmeans_centers_bgr(pixels: np.ndarray, k: int) -> np.ndarray:
    """Returns k x 3 BGR centers."""
    if len(pixels) < k * 10:
        raise ValueError("Too few legend pixels for k-means; widen --legend-roi or lower thresholds.")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    _, _, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        attempts=5,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    return centers.reshape(-1, 3)


def _sort_centers_low_to_high_vitality(centers_bgr: np.ndarray) -> np.ndarray:
    """Yellow (低) → red (高): sort by red-dominance ascending."""
    b, g, r = centers_bgr[:, 0], centers_bgr[:, 1], centers_bgr[:, 2]
    score = r.astype(np.float64) - (g.astype(np.float64) + b.astype(np.float64)) / 2.0
    order = np.argsort(score)
    return centers_bgr[order]


def _median_patch_color_lab(img_bgr: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    patch = img_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        raise ValueError("Patch empty; check query coordinates.")
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, axis=0)


def _nearest_center_lab(query_lab: np.ndarray, centers_bgr: np.ndarray) -> tuple[int, float]:
    centers_lab = cv2.cvtColor(
        centers_bgr.reshape(1, -1, 3).astype(np.uint8),
        cv2.COLOR_BGR2LAB,
    ).reshape(-1, 3).astype(np.float64)
    q = query_lab.astype(np.float64)
    d = np.linalg.norm(centers_lab - q[None, :], axis=1)
    j = int(np.argmin(d))
    return j, float(d[j])


def _fallback_palette_bgr(n: int = 9) -> np.ndarray:
    """Synthetic yellow→red ramp if legend extraction fails."""
    colors = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        # BGR
        b = int(60 * (1 - t))
        g = int(220 - 100 * t)
        r = int(120 + 135 * t)
        colors.append([b, g, r])
    return np.array(colors, dtype=np.float32)


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    """west,south,east,north in WGS84 degrees."""
    parts = [float(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be west,south,east,north")
    west, south, east, north = parts
    return west, south, east, north


def _pixel_to_lonlat(
    qx: float, qy: float, img_w: int, img_h: int, west: float, south: float, east: float, north: float
) -> tuple[float, float]:
    """Map image pixel (x right, y down) to WGS84; y=0 → north edge."""
    iw = max(img_w - 1, 1)
    ih = max(img_h - 1, 1)
    lon = west + (east - west) * (qx / iw)
    lat = north - (north - south) * (qy / ih)
    return lon, lat


def _square_polygon_lonlat(
    qx: float,
    qy: float,
    half_px: float,
    img_w: int,
    img_h: int,
    west: float,
    south: float,
    east: float,
    north: float,
) -> list[list[list[float]]]:
    """Axis-aligned square around query in pixel space, corners mapped to lon/lat."""
    corners_px = [
        (qx - half_px, qy - half_px),
        (qx + half_px, qy - half_px),
        (qx + half_px, qy + half_px),
        (qx - half_px, qy + half_px),
    ]
    ring = []
    for x, y in corners_px + [corners_px[0]]:
        lon, lat = _pixel_to_lonlat(x, y, img_w, img_h, west, south, east, north)
        ring.append([lon, lat])
    return [ring]


def _build_geojson(
    result: ParseResult,
    qx: int,
    qy: int,
    W: int,
    H: int,
    bbox: tuple[float, float, float, float] | None,
    footprint_half_px: float,
) -> dict:
    """FeatureCollection: 点 = 查询中心；多边形 = 与 patch 尺度一致的小方形（示意地块像元）。"""
    props = {
        "evaluation_framework": result.evaluation_framework,
        "indicator": result.indicator,
        "timestamp_local": result.timestamp_local,
        "source_image": result.image_path,
        "query_pixel_x": qx,
        "query_pixel_y": qy,
        "image_width_px": W,
        "image_height_px": H,
        "patch_radius_px": result.patch_radius_px,
        "vitality_bin": result.matched_bin_index,
        "vitality_min": result.vitality_min,
        "vitality_max": result.vitality_max,
        "vitality_mid_estimate": (result.vitality_min + result.vitality_max) / 2.0,
        "lab_distance_to_legend_center": result.lab_distance_to_center,
        "legend_mode": result.legend_mode,
        "notes": result.notes,
    }

    if bbox is None:
        props["coordinate_space"] = "image_pixels_xy"
        pt_coords: list[float] = [float(qx), float(qy)]
        poly_coords: list[list[list[float]]] = [
            [
                [qx - footprint_half_px, qy - footprint_half_px],
                [qx + footprint_half_px, qy - footprint_half_px],
                [qx + footprint_half_px, qy + footprint_half_px],
                [qx - footprint_half_px, qy + footprint_half_px],
                [qx - footprint_half_px, qy - footprint_half_px],
            ]
        ]
        crs_note = (
            "geometry coordinates are image pixels (x right, y down), not WGS84. "
            "Pass --bbox west,south,east,north for lon/lat GeoJSON."
        )
    else:
        west, south, east, north = bbox
        props["coordinate_space"] = "WGS84"
        props["georef_bbox_wsen"] = [west, south, east, north]
        lon, lat = _pixel_to_lonlat(float(qx), float(qy), W, H, west, south, east, north)
        pt_coords = [lon, lat]
        poly_coords = _square_polygon_lonlat(float(qx), float(qy), footprint_half_px, W, H, west, south, east, north)
        crs_note = "Coordinates are WGS84 longitude, latitude from linear --bbox georeferencing."

    fc: dict = {
        "type": "FeatureCollection",
        "name": "lbs_vitality_station_sample",
        "description": crs_note,
        "features": [
            {
                "type": "Feature",
                "properties": {**props, "feature_role": "shanghai_station_query_center"},
                "geometry": {"type": "Point", "coordinates": pt_coords},
            },
            {
                "type": "Feature",
                "properties": {**props, "feature_role": "shanghai_station_query_footprint_proxy"},
                "geometry": {"type": "Polygon", "coordinates": poly_coords},
            },
        ],
    }
    return fc


def _interactive_pick_point(img_bgr: np.ndarray) -> tuple[int, int]:
    win = "Click Shanghai Station parcel (press any key after click)"
    coords: list[int] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.clear()
            coords.extend([x, y])
            vis = img_bgr.copy()
            cv2.drawMarker(vis, (x, y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
            cv2.imshow(win, vis)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, img_bgr)
    print("左键点击地图上的上海站对应地块，然后按任意键退出。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(coords) != 2:
        raise SystemExit("No click recorded.")
    return coords[0], coords[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse LBS vitality map slide → station parcel bin + JSON/CSV.")
    ap.add_argument("--image", type=Path, required=True, help="Path to slide screenshot PNG/JPG.")
    ap.add_argument(
        "--legend-roi",
        type=str,
        default="0.02,0.52,0.38,0.45",
        help="Legend crop x,y,w,h — pixels or normalized fractions (if all ≤1).",
    )
    ap.add_argument("--qx", type=int, default=None, help="Query pixel x (overrides --click).")
    ap.add_argument("--qy", type=int, default=None, help="Query pixel y.")
    ap.add_argument("--click", action="store_true", help="Interactive pick query point.")
    ap.add_argument("--patch-radius", type=int, default=8, help="Median color window radius in pixels.")
    ap.add_argument("--fallback-palette", action="store_true", help="Skip legend k-means; use synthetic ramp.")
    ap.add_argument(
        "--timestamp",
        type=str,
        default="2025-06-12T20:00:00",
        help='ISO-like local timestamp for storage (matches slide "25年6月12日20时").',
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: next to image).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Optional WGS84 west,south,east,north — maps image edges to lon/lat for GeoJSON output.",
    )
    args = ap.parse_args()
    np.random.seed(args.seed)

    bbox_tpl: tuple[float, float, float, float] | None = None
    if args.bbox:
        bbox_tpl = _parse_bbox(args.bbox)

    img_path = args.image.expanduser().resolve()
    if not img_path.is_file():
        raise SystemExit(f"Image not found: {img_path}")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise SystemExit(f"OpenCV could not read: {img_path}")

    H, W = img_bgr.shape[:2]
    if args.click or args.qx is None or args.qy is None:
        qx, qy = _interactive_pick_point(img_bgr)
    else:
        qx, qy = args.qx, args.qy

    lx, ly, lw, lh = 0, 0, W, H
    if args.fallback_palette:
        centers_bgr = _fallback_palette_bgr(len(VITALITY_BINS))
        legend_mode = "synthetic_gradient_fallback"
    else:
        lx, ly, lw, lh = _parse_roi(args.legend_roi, W, H)
        legend_crop = img_bgr[ly : ly + lh, lx : lx + lw]
        mask = _mask_colorful_pixels(legend_crop)
        pix = _sample_pixels_masked(legend_crop, mask)
        centers_bgr = _kmeans_centers_bgr(pix, k=len(VITALITY_BINS))
        centers_bgr = _sort_centers_low_to_high_vitality(centers_bgr)
        legend_mode = f"kmeans_legend_roi({lx},{ly},{lw},{lh})"

    query_lab = _median_patch_color_lab(img_bgr, qx, qy, args.patch_radius)
    bin_idx0, dist = _nearest_center_lab(query_lab, centers_bgr)
    bi = bin_idx0 + 1
    vmin, vmax = VITALITY_BINS[bin_idx0]

    out_dir = args.out_dir if args.out_dir else img_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem + "_lbs_station_parse"

    result = ParseResult(
        evaluation_framework="地块更新潜力评价指标",
        indicator="LBS活力",
        timestamp_local=args.timestamp,
        image_path=str(img_path),
        query_pixel_xy=(qx, qy),
        patch_radius_px=args.patch_radius,
        matched_bin_index=bi,
        vitality_min=vmin,
        vitality_max=vmax,
        lab_distance_to_center=dist,
        legend_mode=legend_mode,
        notes=(
            "数值区间来自幻灯片图例九档；颜色匹配为 LAB 空间到图例 k-means 中心的最近邻。"
            "若布局与默认 ROI 不符，请调整 --legend-roi 或使用 --click + --fallback-palette 做对照。"
        ),
    )

    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    # Minimal CSV row for tabular pipelines
    import csv

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        row = [
            result.evaluation_framework,
            result.indicator,
            result.timestamp_local,
            qx,
            qy,
            bi,
            vmin,
            vmax,
            legend_mode,
        ]
        header = [
            "evaluation_framework",
            "indicator",
            "timestamp_local",
            "query_x",
            "query_y",
            "vitality_bin",
            "vitality_min",
            "vitality_max",
            "legend_mode",
        ]
        if bbox_tpl is not None:
            lon_q, lat_q = _pixel_to_lonlat(float(qx), float(qy), W, H, *bbox_tpl)
            header.extend(["longitude_wgs84", "latitude_wgs84"])
            row.extend([lon_q, lat_q])
        w.writerow(header)
        w.writerow(row)

    footprint_half = float(max(args.patch_radius, 1))
    geojson_fc = _build_geojson(result, qx, qy, W, H, bbox_tpl, footprint_half)
    geojson_path = out_dir / f"{stem}.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_fc, f, ensure_ascii=False, indent=2)

    vis = img_bgr.copy()
    cv2.drawMarker(vis, (qx, qy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
    if not args.fallback_palette:
        cv2.rectangle(vis, (lx, ly), (lx + lw, ly + lh), (0, 255, 255), 2)
    preview_path = out_dir / f"{stem}_preview.jpg"
    cv2.imwrite(str(preview_path), vis)

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    print(f"\nWrote: {json_path}\n       {csv_path}\n       {geojson_path}\n       {preview_path}")


if __name__ == "__main__":
    main()
