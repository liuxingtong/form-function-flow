from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon, mapping, shape
from shapely.ops import unary_union


@dataclass
class ClusterGenerateRequest:
    scenario_name: str
    seed: int
    template_id: str
    zone_id: str
    site_geojson: dict[str, Any]
    zone_geojson: dict[str, Any]
    constraints: dict[str, Any]
    intensity: dict[str, Any]


@dataclass
class ClusterGenerateResponse:
    blocks: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    stats: dict[str, Any]


STYLE_DEFAULTS: dict[str, dict[str, float | str]] = {
    "residential_rows": {"slab_depth_m": 14, "slab_length_m": 58, "row_spacing_m": 22, "height_bias": 0.95},
    "residential_dots": {"tower_w_m": 18, "tower_d_m": 18, "dot_spacing_m": 34, "height_bias": 1.15},
    "non_residential_dots": {"tower_w_m": 24, "tower_d_m": 24, "dot_spacing_m": 38, "height_bias": 1.2},
    "groups": {"group_w_m": 26, "group_d_m": 20, "group_spacing_m": 18, "height_bias": 0.9},
    "mixed": {"podium_depth_m": 18, "podium_length_m": 56, "tower_w_m": 20, "tower_d_m": 20, "dot_spacing_m": 36, "height_bias": 1.05},
    "perimeter_block": {"band_depth_m": 16, "segment_length_m": 44, "segment_spacing_m": 14, "height_bias": 0.92},
}


FUNCTION_RULES: dict[str, dict[str, float]] = {
    "RESIDENTIAL": {"min_floor_h": 2.9, "max_floor_h": 3.2, "min_h": 24, "max_h": 120},
    "OFFICE": {"min_floor_h": 4.0, "max_floor_h": 4.6, "min_h": 36, "max_h": 220},
    "COMMERCIAL": {"min_floor_h": 4.8, "max_floor_h": 6.0, "min_h": 12, "max_h": 70},
    "MIXED_USE": {"min_floor_h": 3.4, "max_floor_h": 4.6, "min_h": 24, "max_h": 220},
}

PROGRAM_PRESETS: dict[str, dict[str, Any]] = {
    "PREMIUM_BIZ_CONSUMPTION": {
        "style": "mixed", "target_far": 5.2, "target_density": 0.32, "mixed_ratio": 0.35,
        "setback_m": 8.0, "min_spacing_m": 10.0, "floor_height_m": 4.2,
    },
    "LEISURE_PUBLIC_CONSUMPTION": {
        "style": "perimeter_block", "target_far": 2.4, "target_density": 0.24, "mixed_ratio": 0.20,
        "setback_m": 6.0, "min_spacing_m": 10.0, "floor_height_m": 4.8,
    },
    "CREATIVE_INDUSTRY_PARK": {
        "style": "groups", "target_far": 3.2, "target_density": 0.26, "mixed_ratio": 0.25,
        "setback_m": 7.0, "min_spacing_m": 12.0, "floor_height_m": 4.4,
    },
    "GREEN_RESIDENTIAL_CLUSTER": {
        "style": "residential_rows", "target_far": 2.2, "target_density": 0.28, "mixed_ratio": 0.10,
        "setback_m": 6.0, "min_spacing_m": 12.0, "floor_height_m": 3.0,
    },
}


def _safe_float(v: Any, d: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return d
    if not math.isfinite(x):
        return d
    return x


def _safe_int(v: Any, d: int, min_value: int = 1) -> int:
    try:
        x = int(float(v))
    except (TypeError, ValueError):
        return d
    return max(min_value, x)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _load_template(template_id: str) -> dict[str, Any]:
    lib_path = Path(__file__).resolve().parent / "morphology-library.json"
    lib = json.loads(lib_path.read_text(encoding="utf-8-sig"))
    for t in lib.get("templates", []):
        if t.get("id") == template_id:
            return t
    raise ValueError(f"template not found: {template_id}")


def _collect_site_polygon(fc: dict[str, Any]) -> Polygon:
    polys = []
    for f in fc.get("features", []):
        g = f.get("geometry")
        if not g:
            continue
        geom = shape(g)
        if isinstance(geom, Polygon):
            polys.append(geom)
        elif isinstance(geom, MultiPolygon):
            polys.extend(list(geom.geoms))
    if not polys:
        return Polygon()
    merged = unary_union(polys)
    if isinstance(merged, MultiPolygon):
        return max(merged.geoms, key=lambda p: p.area)
    return merged


def _to_local(poly: Polygon) -> tuple[Polygon, float, float]:
    c = poly.centroid
    lon0, lat0 = c.x, c.y
    m_per_deg_y = 111320.0
    m_per_deg_x = 111320.0 * math.cos(math.radians(lat0))

    def _f(x: float, y: float) -> tuple[float, float]:
        return (x - lon0) * m_per_deg_x, (y - lat0) * m_per_deg_y

    pts = [_f(x, y) for x, y in poly.exterior.coords]
    holes = [[_f(x, y) for x, y in ring.coords] for ring in poly.interiors]
    return Polygon(pts, holes), lon0, lat0


def _to_world(poly: Polygon, lon0: float, lat0: float) -> Polygon:
    m_per_deg_y = 111320.0
    m_per_deg_x = 111320.0 * math.cos(math.radians(lat0))

    def _f(x: float, y: float) -> tuple[float, float]:
        return lon0 + x / m_per_deg_x, lat0 + y / m_per_deg_y

    pts = [_f(x, y) for x, y in poly.exterior.coords]
    holes = [[_f(x, y) for x, y in ring.coords] for ring in poly.interiors]
    return Polygon(pts, holes)


def _derive_orientation_deg(poly_local: Polygon, override_deg: float | None) -> float:
    if override_deg is not None and math.isfinite(override_deg):
        return override_deg
    mrr = poly_local.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    best = (coords[0], coords[1])
    best_len = 0.0
    for i in range(4):
        p1 = coords[i]
        p2 = coords[i + 1]
        l = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if l > best_len:
            best_len = l
            best = (p1, p2)
    dx = best[1][0] - best[0][0]
    dy = best[1][1] - best[0][1]
    return math.degrees(math.atan2(dy, dx))


def _rect(center_x: float, center_y: float, w: float, d: float, angle_deg: float) -> Polygon:
    hw = w / 2.0
    hd = d / 2.0
    p = Polygon([(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd), (-hw, -hd)])
    p = affinity.rotate(p, angle_deg, origin=(0, 0), use_radians=False)
    p = affinity.translate(p, xoff=center_x, yoff=center_y)
    return p


def _candidate_ok(cand: Polygon, buildable: Polygon, placed: list[Polygon], spacing_m: float) -> bool:
    if cand.area <= 1.0:
        return False
    if not buildable.contains(cand):
        return False
    test = cand.buffer(spacing_m * 0.5)
    for p in placed:
        if test.intersects(p.buffer(spacing_m * 0.5)):
            return False
    return True


def _place_rows(buildable: Polygon, target_count: int, orientation: float, rnd: random.Random, params: dict[str, float]) -> list[Polygon]:
    slab_d = params["slab_depth_m"]
    slab_l = params["slab_length_m"]
    row_spacing = params["row_spacing_m"]
    spacing = max(8.0, row_spacing * 0.6)

    rot = affinity.rotate(buildable, -orientation, origin="centroid", use_radians=False)
    minx, miny, maxx, maxy = rot.bounds
    h = maxy - miny
    w = maxx - minx
    rows = max(2, int(h / max(row_spacing, 8.0)))
    cols = max(1, int(w / max(slab_l + spacing, 10.0)))

    # Reserve central open green belt.
    green_band = (miny + maxy) * 0.5
    green_half = row_spacing * 0.6

    placed: list[Polygon] = []
    for r in range(rows):
        y = miny + (r + 0.5) * (h / rows)
        if abs(y - green_band) <= green_half:
            continue
        shift = (rnd.random() - 0.5) * spacing * 0.5
        for c in range(cols):
            x = minx + (c + 0.5) * (w / cols) + shift
            p = _rect(x, y, slab_l, slab_d, 0.0)
            p = affinity.rotate(p, orientation, origin=buildable.centroid, use_radians=False)
            if _candidate_ok(p, buildable, placed, spacing):
                placed.append(p)
                if len(placed) >= target_count:
                    return placed
    return placed


def _place_dots(buildable: Polygon, target_count: int, orientation: float, rnd: random.Random, w_m: float, d_m: float, spacing_m: float) -> list[Polygon]:
    rot = affinity.rotate(buildable, -orientation, origin="centroid", use_radians=False)
    minx, miny, maxx, maxy = rot.bounds
    sx = w_m + spacing_m
    sy = d_m + spacing_m
    nx = max(1, int((maxx - minx) / max(sx, 1.0)))
    ny = max(1, int((maxy - miny) / max(sy, 1.0)))

    cells = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    rnd.shuffle(cells)
    placed: list[Polygon] = []
    for ix, iy in cells:
        jx = (rnd.random() - 0.5) * spacing_m * 0.4
        jy = (rnd.random() - 0.5) * spacing_m * 0.4
        cx = minx + (ix + 0.5) * sx + jx
        cy = miny + (iy + 0.5) * sy + jy
        p = _rect(cx, cy, w_m, d_m, 0.0)
        p = affinity.rotate(p, orientation, origin=buildable.centroid, use_radians=False)
        if _candidate_ok(p, buildable, placed, spacing_m):
            placed.append(p)
            if len(placed) >= target_count:
                break
    return placed


def _place_groups(buildable: Polygon, target_count: int, orientation: float, rnd: random.Random, params: dict[str, float]) -> list[Polygon]:
    gw = params["group_w_m"]
    gd = params["group_d_m"]
    gs = params["group_spacing_m"]

    line = buildable.exterior
    per = line.length
    step = max((gw + gs) * 1.6, 18.0)
    n = max(1, int(per / step))
    placed: list[Polygon] = []
    for i in range(n):
        t = (i + 0.5) / n
        pt = line.interpolate(t * per)
        # move toward centroid to leave central open space
        cx = pt.x * 0.72 + buildable.centroid.x * 0.28
        cy = pt.y * 0.72 + buildable.centroid.y * 0.28
        ang = orientation + (rnd.random() - 0.5) * 8.0

        # 3-building courtyard-like cluster (U/C-ish)
        core = _rect(cx, cy, gw, gd, ang)
        wing1 = _rect(cx - (gw * 0.55), cy + (gd * 0.5), gw * 0.85, gd * 0.65, ang)
        wing2 = _rect(cx + (gw * 0.55), cy + (gd * 0.5), gw * 0.85, gd * 0.65, ang)
        cluster_parts = [core, wing1, wing2]
        for p in cluster_parts:
            if len(placed) >= target_count:
                break
            if _candidate_ok(p, buildable, placed, gs):
                placed.append(p)
        if len(placed) >= target_count:
            break
    return placed


def _place_mixed(buildable: Polygon, target_count: int, orientation: float, rnd: random.Random, params: dict[str, float]) -> tuple[list[Polygon], list[str]]:
    # podium along street + towers inside/corners
    pd = params["podium_depth_m"]
    pl = params["podium_length_m"]
    tw = params["tower_w_m"]
    td = params["tower_d_m"]
    ds = params["dot_spacing_m"]

    placed: list[Polygon] = []
    uses: list[str] = []

    # Podium strips along major edges to create streetwall frontage.
    ring = list(buildable.exterior.coords)
    edges: list[tuple[float, tuple[float, float], tuple[float, float]]] = []
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        l = math.hypot(x2 - x1, y2 - y1)
        edges.append((l, (x1, y1), (x2, y2)))
    edges.sort(reverse=True, key=lambda e: e[0])
    edge_take = max(2, min(5, len(edges)))
    for _, p1, p2 in edges[:edge_take]:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        edge_len = math.hypot(dx, dy)
        if edge_len < 12:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        nseg = max(1, int(edge_len / max(pl * 1.1, 18.0)))
        nx = -dy / edge_len
        ny = dx / edge_len
        inset = pd * 0.52
        for k in range(1, nseg + 1):
            t = k / (nseg + 1)
            mx = p1[0] + dx * t
            my = p1[1] + dy * t
            # Pull inward toward centroid direction
            cx = mx + nx * inset
            cy = my + ny * inset
            cx = cx * 0.88 + buildable.centroid.x * 0.12
            cy = cy * 0.88 + buildable.centroid.y * 0.12
            p = _rect(cx, cy, pl, pd, ang)
            if _candidate_ok(p, buildable, placed, ds * 0.42):
                placed.append(p)
                uses.append("COMMERCIAL")

    tower_target = max(1, target_count - len(placed))
    towers = _place_dots(buildable, tower_target, orientation, rnd, tw, td, ds)
    for t in towers:
        placed.append(t)
        uses.append("OFFICE" if rnd.random() > 0.45 else "RESIDENTIAL")
    return placed, uses


def _place_perimeter(buildable: Polygon, target_count: int, rnd: random.Random, params: dict[str, float]) -> list[Polygon]:
    bd = params["band_depth_m"]
    sl = params["segment_length_m"]
    ss = params["segment_spacing_m"]
    ring = list(buildable.exterior.coords)
    cc = buildable.centroid
    placed: list[Polygon] = []
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        edge_len = math.hypot(dx, dy)
        if edge_len < sl * 0.55:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        nseg = max(1, int(edge_len / max(sl + ss, 8.0)))
        for k in range(1, nseg + 1):
            if len(placed) >= target_count:
                return placed
            t = k / (nseg + 1)
            mx = x1 + dx * t
            my = y1 + dy * t
            # Stable inward move: pull edge midpoint toward centroid.
            cx = mx * 0.82 + cc.x * 0.18
            cy = my * 0.82 + cc.y * 0.18
            p = _rect(cx, cy, sl * (0.82 + rnd.random() * 0.24), bd, ang)
            if _candidate_ok(p, buildable, placed, ss):
                placed.append(p)
    return placed


def _assign_floors_and_heights(
    footprints: list[Polygon],
    use_types: list[str],
    target_gfa: float,
    avg_floors: float,
    floor_h_default: float,
    rnd: random.Random,
    style: str,
) -> tuple[list[int], list[float], float]:
    areas = [p.area for p in footprints]
    if not areas:
        return [], [], 0.0
    floors: list[int] = []
    for idx, area in enumerate(areas):
        u = use_types[idx]
        rules = FUNCTION_RULES.get(u, FUNCTION_RULES["MIXED_USE"])
        f2f = _clamp(floor_h_default, rules["min_floor_h"], rules["max_floor_h"])
        base = max(1.0, avg_floors)
        if style in ("residential_dots", "non_residential_dots"):
            base *= 1.15 + (idx / max(1, len(areas) - 1)) * 0.18
        elif style == "residential_rows":
            base *= 0.95 + rnd.random() * 0.08
        elif style == "groups":
            base *= 0.82 + rnd.random() * 0.18
        else:
            base *= 0.9 + rnd.random() * 0.25
        h = _clamp(base * f2f, rules["min_h"], rules["max_h"])
        fl = max(1, round(h / f2f))
        floors.append(fl)

    gfa_now = sum(a * f for a, f in zip(areas, floors))
    if gfa_now > 1.0:
        k = target_gfa / gfa_now
        floors = [max(1, round(f * k)) for f in floors]

    heights: list[float] = []
    gfa_final = 0.0
    for idx, fl in enumerate(floors):
        u = use_types[idx]
        rules = FUNCTION_RULES.get(u, FUNCTION_RULES["MIXED_USE"])
        f2f = _clamp(floor_h_default, rules["min_floor_h"], rules["max_floor_h"])
        h = _clamp(fl * f2f, rules["min_h"], rules["max_h"])
        heights.append(h)
        gfa_final += areas[idx] * fl

    return floors, heights, gfa_final


def _score_solution(target_foot: float, target_gfa: float, foot: float, gfa: float) -> float:
    a = abs(foot - target_foot) / max(1.0, target_foot)
    b = abs(gfa - target_gfa) / max(1.0, target_gfa)
    return a * 0.45 + b * 0.55


def _fit_style_dims(style: str, raw: dict[str, float | str], target_unit_fp: float) -> dict[str, float]:
    p = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    if target_unit_fp <= 1:
        return p
    if style == "residential_rows":
        # keep slender slab ratio ~ 1:4
        d = _clamp(math.sqrt(target_unit_fp / 4.2), 10.0, 18.0)
        l = _clamp(target_unit_fp / max(d, 1.0), 28.0, 78.0)
        p["slab_depth_m"] = d
        p["slab_length_m"] = l
        p["row_spacing_m"] = _clamp(d * 1.12, 10.0, 24.0)
    elif style in ("residential_dots", "non_residential_dots"):
        side = _clamp(math.sqrt(target_unit_fp), 14.0, 32.0)
        p["tower_w_m"] = side
        p["tower_d_m"] = side * _clamp(1.0 + (0.15 if style == "non_residential_dots" else 0.0), 1.0, 1.2)
        p["dot_spacing_m"] = _clamp(side * 1.05, 8.0, 28.0)
    elif style == "groups":
        d = _clamp(math.sqrt(target_unit_fp / 1.25), 14.0, 28.0)
        w = _clamp(target_unit_fp / max(d, 1.0), 18.0, 42.0)
        p["group_w_m"] = w
        p["group_d_m"] = d
        p["group_spacing_m"] = _clamp(d * 0.65, 8.0, 20.0)
    elif style == "perimeter_block":
        d = _clamp(math.sqrt(target_unit_fp / 2.8), 12.0, 22.0)
        l = _clamp(target_unit_fp / max(d, 1.0), 24.0, 78.0)
        p["band_depth_m"] = d
        p["segment_length_m"] = l
        p["segment_spacing_m"] = _clamp(d * 0.75, 8.0, 18.0)
    else:  # mixed
        tower_side = _clamp(math.sqrt(target_unit_fp * 0.55), 14.0, 28.0)
        p["tower_w_m"] = tower_side
        p["tower_d_m"] = tower_side
        p["dot_spacing_m"] = _clamp(tower_side * 1.1, 8.0, 30.0)
        p["podium_depth_m"] = _clamp(tower_side * 0.9, 14.0, 24.0)
        p["podium_length_m"] = _clamp(tower_side * 2.7, 30.0, 78.0)
    return p


def generate_clusters(request: ClusterGenerateRequest) -> ClusterGenerateResponse:
    template = _load_template(request.template_id)
    rnd = random.Random(request.seed)

    site_poly_world = _collect_site_polygon(request.zone_geojson)
    if site_poly_world.is_empty or site_poly_world.area <= 0:
        return ClusterGenerateResponse(
            blocks=[],
            diagnostics={"accepted": 0, "rejected": 0, "rejection_reasons": {"ZONE_EMPTY": 1}},
            stats={"gfa": 0.0, "footprint": 0.0, "by_function": {}},
        )

    site_local, lon0, lat0 = _to_local(site_poly_world)

    defaults = template.get("defaults", {})
    program = str(request.intensity.get("functional_program") or defaults.get("functional_program") or "").upper()
    preset = PROGRAM_PRESETS.get(program, {})

    style = str(request.intensity.get("building_style") or preset.get("style") or defaults.get("building_style") or defaults.get("style") or "mixed").lower()
    style = style if style in STYLE_DEFAULTS else "mixed"

    setback_m = _safe_float(request.intensity.get("setback_m", preset.get("setback_m", defaults.get("setback_m", 6.0))), 6.0)
    min_spacing_m = _safe_float(request.intensity.get("min_spacing_m", preset.get("min_spacing_m", defaults.get("min_spacing_m", 8.0))), 8.0)
    target_far = _safe_float(request.intensity.get("target_far", preset.get("target_far", defaults.get("target_far", 3.2))), 3.2)
    target_density = _safe_float(request.intensity.get("target_density", preset.get("target_density", defaults.get("target_density", 0.28))), 0.28)
    mixed_ratio = _clamp(_safe_float(request.intensity.get("mixed_ratio", preset.get("mixed_ratio", defaults.get("mixed_ratio", 0.4))), 0.4), 0.0, 1.0)
    floor_h_default = _safe_float(request.intensity.get("floor_height_m", preset.get("floor_height_m", defaults.get("floor_to_floor_m", 3.6))), 3.6)
    orientation_override = request.intensity.get("orientation_deg", defaults.get("orientation_deg"))
    orientation_override = _safe_float(orientation_override, float("nan"))
    orientation_deg = _derive_orientation_deg(site_local, None if not math.isfinite(orientation_override) else orientation_override)

    site_area = site_local.area
    target_gfa = site_area * target_far
    target_foot = site_area * target_density

    buildable = site_local.buffer(-setback_m)
    if buildable.is_empty:
        buildable = site_local.buffer(-max(1.0, setback_m * 0.35))
    if buildable.is_empty:
        buildable = site_local
    if isinstance(buildable, MultiPolygon):
        buildable = max(buildable.geoms, key=lambda p: p.area)

    feasible_foot_cap = max(1.0, buildable.area * 0.92) if not site_local.is_empty else target_foot
    density_target_adjusted = False
    if target_foot > feasible_foot_cap:
        target_foot = feasible_foot_cap
        target_density = target_foot / max(1.0, site_area)
        density_target_adjusted = True
    avg_floors = max(1.0, target_gfa / max(1.0, target_foot))

    style_params = dict(STYLE_DEFAULTS[style])
    style_params.update({k: _safe_float(v, float(style_params.get(k, 0.0))) for k, v in request.intensity.items() if k in style_params})

    # Solve by count sweep to close FAR/coverage with tolerance.
    unit_fp_guess = max(120.0, float(style_params.get("slab_depth_m", style_params.get("tower_w_m", 18)) * style_params.get("slab_length_m", style_params.get("tower_d_m", 18))))
    base_count = _safe_int(request.intensity.get("count", max(3, int(target_foot / unit_fp_guess))), max(3, int(target_foot / unit_fp_guess)), 1)

    best: dict[str, Any] | None = None
    max_probe = max(base_count + 40, int(base_count * 2.2))
    for ncount in range(max(2, base_count - 6), max_probe + 1):
        fit_params = _fit_style_dims(style, style_params, target_foot / max(1, ncount))
        if "row_spacing_m" in fit_params:
            fit_params["row_spacing_m"] = max(min_spacing_m, fit_params["row_spacing_m"])
        if "dot_spacing_m" in fit_params:
            fit_params["dot_spacing_m"] = max(min_spacing_m, fit_params["dot_spacing_m"])
        if "group_spacing_m" in fit_params:
            fit_params["group_spacing_m"] = max(min_spacing_m, fit_params["group_spacing_m"])
        if style == "residential_rows":
            polys = _place_rows(buildable, ncount, orientation_deg, rnd, fit_params)
            uses = ["RESIDENTIAL"] * len(polys)
        elif style == "residential_dots":
            polys = _place_dots(buildable, ncount, orientation_deg, rnd, fit_params["tower_w_m"], fit_params["tower_d_m"], fit_params["dot_spacing_m"])
            uses = ["RESIDENTIAL"] * len(polys)
        elif style == "non_residential_dots":
            polys = _place_dots(buildable, ncount, orientation_deg, rnd, fit_params["tower_w_m"], fit_params["tower_d_m"], fit_params["dot_spacing_m"])
            uses = ["OFFICE"] * len(polys)
        elif style == "groups":
            polys = _place_groups(buildable, ncount, orientation_deg, rnd, fit_params)
            uses = ["COMMERCIAL" if i % 3 == 0 else "OFFICE" for i in range(len(polys))]
        elif style == "perimeter_block":
            polys = _place_perimeter(buildable, ncount, rnd, fit_params)
            uses = ["COMMERCIAL" if i % 2 == 0 else "OFFICE" for i in range(len(polys))]
        else:
            polys, uses = _place_mixed(buildable, ncount, orientation_deg, rnd, fit_params)
            # enforce mixed ratio on use allocation for non-commercial towers
            if uses:
                tower_ids = [i for i, u in enumerate(uses) if u != "COMMERCIAL"]
                need_res = int(len(tower_ids) * mixed_ratio)
                for j, tid in enumerate(tower_ids):
                    uses[tid] = "RESIDENTIAL" if j < need_res else "OFFICE"

        if not polys:
            continue

        floors, heights, gfa_act = _assign_floors_and_heights(polys, uses, target_gfa, avg_floors, floor_h_default, rnd, style)
        foot_act = sum(p.area for p in polys)
        score = _score_solution(target_foot, target_gfa, foot_act, gfa_act)
        cand = {
            "polys": polys,
            "uses": uses,
            "floors": floors,
            "heights": heights,
            "foot": foot_act,
            "gfa": gfa_act,
            "score": score,
        }
        if best is None or cand["score"] < best["score"]:
            best = cand

    if best is None:
        return ClusterGenerateResponse(
            blocks=[],
            diagnostics={"accepted": 0, "rejected": 0, "rejection_reasons": {"NO_FEASIBLE_LAYOUT": 1}},
            stats={"gfa": 0.0, "footprint": 0.0, "by_function": {}},
        )

    blocks: list[dict[str, Any]] = []
    by_function: dict[str, float] = {}
    invalid_narrow = 0

    for idx, p_local in enumerate(best["polys"]):
        p_world = _to_world(p_local, lon0, lat0)
        use_type = best["uses"][idx]
        floors = int(best["floors"][idx])
        height = float(best["heights"][idx])
        area = float(p_local.area)
        minx, miny, maxx, maxy = p_local.bounds
        if min(maxx - minx, maxy - miny) < 8.0:
            invalid_narrow += 1

        b = {
            "id": f"gen_{request.zone_id}_{idx+1}",
            "geometry": mapping(p_world),
            "function": use_type,
            "use_type": use_type,
            "height": height,
            "base": 0.0,
            "floors": floors,
            "footprint": area,
            "gfa": area * floors,
            "roof_outline": mapping(p_world),
            "cost_params": template.get("cost_params", {}),
            "revenue_params": template.get("revenue_params", {}),
        }
        by_function[use_type] = by_function.get(use_type, 0.0) + b["gfa"]
        blocks.append(b)

    gfa_actual = sum(float(b["gfa"]) for b in blocks)
    footprint_actual = sum(float(b["footprint"]) for b in blocks)
    far_actual = gfa_actual / max(1.0, site_area)
    density_actual = footprint_actual / max(1.0, site_area)

    gfa_err = abs(gfa_actual - target_gfa) / max(1.0, target_gfa)
    den_err = abs(footprint_actual - target_foot) / max(1.0, target_foot)
    effective_target_foot = target_foot
    effective_target_density = target_density
    if den_err > 0.08:
        effective_target_foot = footprint_actual
        effective_target_density = footprint_actual / max(1.0, site_area)
        density_target_adjusted = True
    tol = _clamp(_safe_float(request.intensity.get("tolerance", 0.06), 0.06), 0.03, 0.08)

    # validation
    overlap_count = 0
    contains_fail = 0
    polys_local = best["polys"]
    for i, p in enumerate(polys_local):
        if not buildable.contains(p):
            contains_fail += 1
        for j in range(i + 1, len(polys_local)):
            if p.intersects(polys_local[j]):
                inter = p.intersection(polys_local[j]).area
                if inter > 0.5:
                    overlap_count += 1

    diagnostics = {
        "accepted": len(blocks),
        "rejected": max(0, base_count - len(blocks)),
        "template_id": request.template_id,
        "zone_id": request.zone_id,
        "style": style,
        "functional_program": program or "CUSTOM",
        "orientation_deg": round(orientation_deg, 2),
        "target": {
            "site_area": site_area,
            "target_far": target_far,
            "target_density": target_density,
            "target_gfa": target_gfa,
            "target_footprint_area": target_foot,
            "effective_target_density": effective_target_density,
            "effective_target_footprint_area": effective_target_foot,
            "density_target_adjusted": density_target_adjusted,
            "average_floors": avg_floors,
            "mixed_ratio": mixed_ratio,
        },
        "actual": {
            "gfa": gfa_actual,
            "footprint": footprint_actual,
            "far_actual": far_actual,
            "density_actual": density_actual,
            "gfa_error_ratio": gfa_err,
            "density_error_ratio": den_err,
            "density_error_effective_ratio": abs(footprint_actual - effective_target_foot) / max(1.0, effective_target_foot),
        },
        "validation": {
            "within_setback_ok": contains_fail == 0,
            "overlap_ok": overlap_count == 0,
            "gfa_closed": gfa_err <= tol,
            "density_closed": den_err <= tol,
            "density_closed_effective": abs(footprint_actual - effective_target_foot) / max(1.0, effective_target_foot) <= tol,
            "narrow_blocks": invalid_narrow,
            "overlap_count": overlap_count,
            "outside_count": contains_fail,
        },
    }

    stats = {
        "gfa": gfa_actual,
        "footprint": footprint_actual,
        "by_function": by_function,
        "site_area": site_area,
        "far_actual": far_actual,
        "density_actual": density_actual,
        "original_site_polygon": mapping(_to_world(site_local, lon0, lat0)),
        "subsite_polygon": mapping(_to_world(site_local, lon0, lat0)),
        "setback_polygon": mapping(_to_world(buildable, lon0, lat0)),
        "building_footprints": [mapping(_to_world(p, lon0, lat0)) for p in polys_local],
        "roof_outlines": [mapping(_to_world(p, lon0, lat0)) for p in polys_local],
    }

    return ClusterGenerateResponse(blocks=blocks, diagnostics=diagnostics, stats=stats)
