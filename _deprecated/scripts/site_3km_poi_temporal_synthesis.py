"""
Build POI-based synthetic diurnal weights from data/site_3km clipped POI GeoJSONs,
derive **two** sets of four time-slice boundaries — weekday vs weekend — each as equal
cumulative mass quarters on [6,23) of its own curve (wd vs we), and write
03_time_slices.csv + poi_temporal_synthesis.json.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from time_slice_constants import T_IDS_WEEKDAY, T_IDS_WEEKEND

SITE_3KM = Path(__file__).resolve().parents[1] / "data" / "site_3km"
POI_DIR = SITE_3KM / "02-POI&AOI" / "1-POI" / "25.05" / "CSV" / "分类" / "按类别"
OUT_JSON = SITE_3KM / "poi_temporal_synthesis.json"
OUT_CSV = SITE_3KM / "03_time_slices.csv"

QUAL_LABELS = (
    "第一时段（晨—午前）",
    "第二时段（午间）",
    "第三时段（午后—傍晚）",
    "第四时段（晚间）",
)
WD_NAMES = tuple(f"工作日·{QUAL_LABELS[i]}" for i in range(4))
WE_NAMES = tuple(f"周末·{QUAL_LABELS[i]}" for i in range(4))
PERIODS = ("slice_q1", "slice_q2", "slice_q3", "slice_q4")


def _role_hour_weights() -> dict[str, list[float]]:
    """24-hour profiles (sum=1) by functional role; literature-style priors."""
    h = list(range(24))

    def norm(v: list[float]) -> list[float]:
        s = sum(v)
        return [x / s for x in v]

    office = norm(
        [1.2 * math.exp(-0.5 * ((x - 9.5) / 1.4) ** 2) + 0.9 * math.exp(-0.5 * ((x - 15.0) / 2.0) ** 2) + 0.05 for x in h]
    )
    transit = norm(
        [
            1.4 * math.exp(-0.5 * ((x - 8.0) / 1.2) ** 2)
            + 1.3 * math.exp(-0.5 * ((x - 18.0) / 1.4) ** 2)
            + 0.15
            for x in h
        ]
    )
    food = norm(
        [
            0.9 * math.exp(-0.5 * ((x - 12.0) / 1.3) ** 2)
            + 1.1 * math.exp(-0.5 * ((x - 18.5) / 1.8) ** 2)
            + 0.25
            for x in h
        ]
    )
    retail = norm([0.35 + 0.65 * math.exp(-0.5 * ((x - 15.0) / 3.5) ** 2) for x in h])
    residential = norm([0.55 + 0.45 * math.exp(-0.5 * ((x - 2.5) / 3.0) ** 2) for x in h])
    leisure = norm(
        [
            0.2 + 0.5 * math.exp(-0.5 * ((x - 19.5) / 2.2) ** 2) + 0.35 * math.exp(-0.5 * ((x - 14.0) / 3.0) ** 2)
            for x in h
        ]
    )
    services = norm([0.12 + 0.88 * math.exp(-0.5 * ((x - 11.0) / 4.0) ** 2) for x in h])
    hotel = norm([0.15 + 0.85 * math.exp(-0.5 * ((x - 21.0) / 2.5) ** 2) for x in h])
    auto = norm([0.2 + 0.8 * math.exp(-0.5 * ((x - 10.5) / 4.0) ** 2) for x in h])
    other = norm([1.0 for _ in h])
    return {
        "office": office,
        "transit": transit,
        "food": food,
        "retail": retail,
        "residential": residential,
        "leisure": leisure,
        "services": services,
        "hotel": hotel,
        "auto": auto,
        "other": other,
    }


def category_to_role(name: str) -> str:
    if "公司企业" in name or "政府机构" in name:
        return "office"
    if "交通设施" in name or "通行设施" in name:
        return "transit"
    if "餐饮服务" in name:
        return "food"
    if "购物服务" in name:
        return "retail"
    if "商务住宅" in name:
        return "residential"
    if "科教文化" in name:
        return "office"
    if "体育休闲" in name or "风景名胜" in name:
        return "leisure"
    if "住宿服务" in name:
        return "hotel"
    if "医疗保健" in name or "生活服务" in name or "公共设施" in name or "金融保险" in name:
        return "services"
    if "汽车" in name or "摩托车" in name:
        return "auto"
    return "other"


def count_geojson_features(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data.get("features") or [])


def blend_counts(counts: dict[str, int], roles: dict[str, list[float]], weekend_adj: bool) -> list[float]:
    w = [0.0] * 24
    for cat, n in counts.items():
        if n <= 0:
            continue
        role = category_to_role(cat)
        base = roles[role][:]
        if weekend_adj:
            if role in ("retail", "food", "leisure"):
                base = [x * 1.18 for x in base]
            elif role == "office":
                base = [x * 0.52 for x in base]
            elif role == "transit":
                base = [x * 0.92 for x in base]
            elif role == "residential":
                base = [x * 1.08 for x in base]
        for h in range(24):
            w[h] += n * base[h]
    s = sum(w)
    if s <= 0:
        return [1.0 / 24.0] * 24
    return [x / s for x in w]


def normalize_full_day(curve: list[float]) -> list[float]:
    s = sum(curve)
    if s <= 0:
        return [1.0 / 24.0] * 24
    return [x / s for x in curve]


def equal_mass_cuts(curve: list[float], h_start: int, h_end_exclusive: int, n_parts: int) -> list[int]:
    """Return [t0,...,t_n]: t0=h_start, t_n=h_end_exclusive; each [t_i,t_{i+1}) has ~equal mass."""
    acc = [0.0] * 25
    for h in range(24):
        acc[h + 1] = acc[h] + curve[h]
    total = acc[h_end_exclusive] - acc[h_start]
    cuts = [h_start]
    if total <= 0:
        step = max(1, (h_end_exclusive - h_start) // n_parts)
        for i in range(1, n_parts):
            cuts.append(min(h_start + i * step, h_end_exclusive - 1))
        cuts.append(h_end_exclusive)
        return cuts
    thresholds = [total * (k / n_parts) for k in range(1, n_parts)]
    ti = 0
    cum = 0.0
    for h in range(h_start, h_end_exclusive):
        cum += curve[h]
        while ti < len(thresholds) and cum >= thresholds[ti]:
            cuts.append(h + 1)
            ti += 1
    cuts.append(h_end_exclusive)
    return cuts


def format_hr(h: float) -> str:
    hh = int(math.floor(h)) % 24
    mm = int(round((h - math.floor(h)) * 60)) % 60
    return f"{hh:02d}:{mm:02d}"


def mass_seg(curve: list[float], a: int, b: int) -> float:
    if b <= a:
        return 0.0
    return sum(curve[a:b])


def mass_frac_interval(curve: list[float], a: int, b: int) -> float:
    t = sum(curve)
    if t <= 0:
        return 0.0
    return mass_seg(curve, a, b) / t


def rel_to_mean(masses: list[float], i: int) -> float:
    u = sum(masses) / max(len(masses), 1)
    if u <= 0:
        return 1.0
    return round(masses[i] / u, 4)


def main() -> None:
    if not POI_DIR.is_dir():
        raise SystemExit(f"Missing POI dir: {POI_DIR}")

    counts: dict[str, int] = {}
    for p in sorted(POI_DIR.glob("*.geojson")):
        if "虚拟数据" in p.name:
            continue
        m = re.search(r"上海市-(.+)\.geojson$", p.name)
        cat = m.group(1) if m else p.stem
        counts[cat] = count_geojson_features(p)

    roles = _role_hour_weights()
    wd = blend_counts(counts, roles, weekend_adj=False)
    we = blend_counts(counts, roles, weekend_adj=True)
    avg = normalize_full_day([(wd[h] + we[h]) / 2.0 for h in range(24)])

    h0, h1 = 6, 23
    cuts_wd = equal_mass_cuts(wd, h0, h1, 4)
    cuts_we = equal_mass_cuts(we, h0, h1, 4)
    if len(cuts_wd) != 5:
        cuts_wd = [h0, h0 + 4, h0 + 9, h0 + 15, h1]
    if len(cuts_we) != 5:
        cuts_we = [h0, h0 + 4, h0 + 9, h0 + 15, h1]

    slices_wd = [(cuts_wd[i], cuts_wd[i + 1]) for i in range(4)]
    slices_we = [(cuts_we[i], cuts_we[i + 1]) for i in range(4)]

    mw = [mass_frac_interval(wd, a, b) for a, b in slices_wd]
    mwe = [mass_frac_interval(we, a, b) for a, b in slices_we]

    flow_wd: dict[str, dict[str, float]] = {}
    flow_we: dict[str, dict[str, float]] = {}
    rev_w = list(reversed(mw))
    rev_e = list(reversed(mwe))
    for i, tid in enumerate(T_IDS_WEEKDAY):
        a, b = slices_wd[i]
        flow_wd[tid] = {
            "curve_mass_share": round(mw[i], 4),
            "period_inflow_weight": rel_to_mean(mw, i),
            "period_outflow_weight": rel_to_mean(rev_w, i),
        }
    for i, tid in enumerate(T_IDS_WEEKEND):
        a, b = slices_we[i]
        flow_we[tid] = {
            "curve_mass_share": round(mwe[i], 4),
            "period_inflow_weight": rel_to_mean(mwe, i),
            "period_outflow_weight": rel_to_mean(rev_e, i),
        }

    out = {
        "source_dir": str(POI_DIR).replace("\\", "/"),
        "poi_counts_by_category": counts,
        "total_poi": sum(counts.values()),
        "method": {
            "diurnal": "Per-category POI counts × role-specific 24h prior weights (sum-normalized).",
            "weekend_adjustment": "Retail/food/leisure up; office down; transit slightly down (for weekend curve only).",
            "partition": (
                f"Independent four clock windows for weekday (wd) and weekend (we): equal cumulative mass quarters "
                f"on [{h0},{h1}) **separately** on wd(h) and we(h). curve_avg(h) retained for diagnostics only."
            ),
            "t_id_note": (
                "Weekday ids: WD_AM/WD_PM/WD_EVE/WD_NT map to Q1–Q4 on the weekday curve; "
                "weekend ids: WE_AM/WE_MD/WE_EVE/WE_NT map to Q1–Q4 on the weekend curve. "
                "Clock boundaries may differ between day types."
            ),
        },
        "curves": {"weekday": wd, "weekend": we, "partition_avg": avg},
        "partition_cuts_hour_weekday": cuts_wd,
        "partition_cuts_hour_weekend": cuts_we,
        "slices_weekday": [
            {
                "t_id": T_IDS_WEEKDAY[i],
                "hour_range_inclusive_start": slices_wd[i][0],
                "hour_range_exclusive_end": slices_wd[i][1],
                "mass_share_weekday_in_slice": round(mw[i], 4),
            }
            for i in range(4)
        ],
        "slices_weekend": [
            {
                "t_id": T_IDS_WEEKEND[i],
                "hour_range_inclusive_start": slices_we[i][0],
                "hour_range_exclusive_end": slices_we[i][1],
                "mass_share_weekend_in_slice": round(mwe[i], 4),
            }
            for i in range(4)
        ],
        "flow_proxy_period_weights": {
            "note": (
                "curve_mass_share: fraction of full-day curve mass in slice. "
                "period_*_weight: slice mass / mean(slice masses), inflow pattern early>late via index reversal heuristic."
            ),
            "weekday": flow_wd,
            "weekend": flow_we,
        },
    }

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for i in range(4):
        a, b = slices_wd[i]
        rows.append(
            {
                "t_id": T_IDS_WEEKDAY[i],
                "t_name": WD_NAMES[i],
                "day_type": "weekday",
                "period": PERIODS[i],
                "start_local": format_hr(float(a)),
                "end_local": format_hr(float(b)),
                "hour_range_inclusive_start": a,
                "hour_range_exclusive_end": b,
            }
        )
    for i in range(4):
        a, b = slices_we[i]
        rows.append(
            {
                "t_id": T_IDS_WEEKEND[i],
                "t_name": WE_NAMES[i],
                "day_type": "weekend",
                "period": PERIODS[i],
                "start_local": format_hr(float(a)),
                "end_local": format_hr(float(b)),
                "hour_range_inclusive_start": a,
                "hour_range_exclusive_end": b,
            }
        )

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "t_id",
                "t_name",
                "day_type",
                "period",
                "start_local",
                "end_local",
                "hour_range_inclusive_start",
                "hour_range_exclusive_end",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print("Wrote", OUT_JSON)
    print("Wrote", OUT_CSV)
    print("cuts_weekday", cuts_wd, "slices_wd", slices_wd)
    print("cuts_weekend", cuts_we, "slices_we", slices_we)


if __name__ == "__main__":
    main()
