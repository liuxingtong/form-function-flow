"""
Aggregate MetroFlow 10-min in/out within the site buffer to the same four clock
slices as 03_time_slices.csv, split by workday calendar (isWorday), and optionally
blend empirical mass shares with poi_temporal_synthesis.json flow_proxy weights.

Outputs: data/site_3km/metroflow/time_slice_calibration.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SITE_3KM = Path(__file__).resolve().parents[1] / "data" / "site_3km"
METRO_DIR = SITE_3KM / "metroflow"
INOUT_PARQUET = METRO_DIR / "inout_10min_3km.parquet"
WORKDAY_CSV = METRO_DIR / "MetaData_workday_calendar.csv"
POI_SYNTH = SITE_3KM / "poi_temporal_synthesis.json"
OUT_JSON = METRO_DIR / "time_slice_calibration.json"

T_IDS = ("WD_AM", "WD_PM", "WD_EVE", "WE_PM")


def hour_to_tid(h: int) -> str | None:
    if 6 <= h < 11:
        return "WD_AM"
    if 11 <= h < 15:
        return "WD_PM"
    if 15 <= h < 18:
        return "WD_EVE"
    if 18 <= h < 23:
        return "WE_PM"
    return None


def rel_to_mean(masses: list[float], i: int) -> float:
    u = sum(masses) / max(len(masses), 1)
    if u <= 0:
        return 1.0
    return round(masses[i] / u, 4)


def flow_block_from_masses(
    mass_act: list[float], in_m: list[float], out_m: list[float]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, tid in enumerate(T_IDS):
        out[tid] = {
            "curve_mass_share": round(mass_act[i], 4),
            "period_inflow_weight": rel_to_mean(in_m, i),
            "period_outflow_weight": rel_to_mean(out_m, i),
        }
    return out


def blend_flow_proxy(
    syn: dict[str, dict[str, dict[str, float]]],
    emp: dict[str, dict[str, dict[str, float]]],
    alpha: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Convex blend of curve_mass_share; blend in/out weights the same way."""
    note = (
        f"Blended with alpha={alpha}: curve_mass_share and period_*_weights "
        "mix synthetic (POI curves) and MetroFlow empirical buffer totals."
    )
    blended: dict[str, dict[str, dict[str, float]]] = {"note": note, "weekday": {}, "weekend": {}}
    for day in ("weekday", "weekend"):
        for tid in T_IDS:
            s = syn[day][tid]
            e = emp[day][tid]
            blended[day][tid] = {
                "curve_mass_share": round(
                    (1 - alpha) * s["curve_mass_share"] + alpha * e["curve_mass_share"], 4
                ),
                "period_inflow_weight": round(
                    (1 - alpha) * s["period_inflow_weight"] + alpha * e["period_inflow_weight"], 4
                ),
                "period_outflow_weight": round(
                    (1 - alpha) * s["period_outflow_weight"] + alpha * e["period_outflow_weight"], 4
                ),
            }
    return blended  # type: ignore[return-value]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--blend-alpha",
        type=float,
        default=0.5,
        help="Weight on MetroFlow empirical vs POI synthetic for blended flow_proxy (0=synthetic only).",
    )
    args = p.parse_args()
    alpha = max(0.0, min(1.0, args.blend_alpha))

    if not INOUT_PARQUET.is_file():
        raise SystemExit(f"Missing {INOUT_PARQUET}; run build_metroflow_site_subset.py first.")
    if not WORKDAY_CSV.is_file():
        raise SystemExit(f"Missing {WORKDAY_CSV}")
    if not POI_SYNTH.is_file():
        raise SystemExit(f"Missing {POI_SYNTH}")

    cal = pd.read_csv(WORKDAY_CSV)
    # Dataset typo: isWorday
    col = "isWorday" if "isWorday" in cal.columns else "isWorkday"
    cal = cal.rename(columns={col: "is_workday"})
    cal["date"] = cal["date"].astype(int)

    df = pd.read_parquet(INOUT_PARQUET)
    df["hour"] = df["starttime"].astype(str).str.zfill(6).str[:2].astype(int)
    df["t_id"] = df["hour"].map(hour_to_tid)
    df = df[df["t_id"].notna()].copy()
    df = df.merge(cal[["date", "is_workday"]], on="date", how="inner")

    df["activity"] = df["inflow"].astype("float64") + df["outflow"].astype("float64")

    def aggregate(is_wd: int) -> tuple[dict[str, float], dict[str, float], dict[str, float], int]:
        sub = df[df["is_workday"] == is_wd]
        n_dates = int(sub["date"].nunique())
        g = sub.groupby("t_id", as_index=False).agg(
            activity=("activity", "sum"),
            inflow=("inflow", "sum"),
            outflow=("outflow", "sum"),
        )
        row = {r["t_id"]: float(r["activity"]) for _, r in g.iterrows()}
        ri = {r["t_id"]: float(r["inflow"]) for _, r in g.iterrows()}
        ro = {r["t_id"]: float(r["outflow"]) for _, r in g.iterrows()}
        return row, ri, ro, n_dates

    def to_vectors(d_act: dict[str, float], d_in: dict[str, float], d_out: dict[str, float]):
        act = [d_act.get(t, 0.0) for t in T_IDS]
        ins = [d_in.get(t, 0.0) for t in T_IDS]
        outs = [d_out.get(t, 0.0) for t in T_IDS]
        s_act = sum(act)
        if s_act <= 0:
            mass = [0.25, 0.25, 0.25, 0.25]
        else:
            mass = [x / s_act for x in act]
        return mass, ins, outs

    wd_act_d, wd_in_d, wd_out_d, n_wd = aggregate(1)
    we_act_d, we_in_d, we_out_d, n_we = aggregate(0)
    mw, in_wd, out_wd = to_vectors(wd_act_d, wd_in_d, wd_out_d)
    mwe, in_we, out_we = to_vectors(we_act_d, we_in_d, we_out_d)

    emp_wd = flow_block_from_masses(mw, in_wd, out_wd)
    emp_we = flow_block_from_masses(mwe, in_we, out_we)

    poi = json.loads(POI_SYNTH.read_text(encoding="utf-8"))
    syn_fp = poi["flow_proxy_period_weights"]
    syn_wd = syn_fp["weekday"]
    syn_we = syn_fp["weekend"]

    blended = blend_flow_proxy(
        {"weekday": syn_wd, "weekend": syn_we},
        {"weekday": emp_wd, "weekend": emp_we},
        alpha,
    )

    out = {
        "caveat": (
            "MetroFlow 子集为 2017 年样本；与当前 POI/建成环境存在年际差。工作日历字段 isWorday 将法定假日等记为非工作日，"
            "与「仅周六日」的周末定义不完全一致。"
        ),
        "inputs": {
            "inout_parquet": str(INOUT_PARQUET).replace("\\", "/"),
            "workday_calendar": str(WORKDAY_CSV).replace("\\", "/"),
            "poi_synthesis": str(POI_SYNTH).replace("\\", "/"),
        },
        "aggregation": (
            "对缓冲区内各站 10min 行求和 inflow+outflow 为 activity；由 starttime 取整点小时，"
            "映射到与 03_time_slices 相同的四小时窗 WD_AM[6,11)、WD_PM[11,15)、WD_EVE[15,18)、WE_PM[18,23)。"
            "按 is_workday=1/0 分别池化所有样本日后再归一化质量占比。"
        ),
        "n_sample_dates": {"workday_calendar_1": n_wd, "workday_calendar_0": n_we},
        "empirical_flow_proxy_period_weights": {
            "note": (
                "curve_mass_share：各窗 activity 占全日（池化）份额。"
                "period_inflow_weight / period_outflow_weight：该窗总进站或出站 / 四窗均值（真实分向，非 POI 脚本的反转启发式）。"
            ),
            "weekday": emp_wd,
            "weekend": emp_we,
        },
        "blend_alpha_metroflow": alpha,
        "synthetic_flow_proxy_period_weights_ref": {"weekday": syn_wd, "weekend": syn_we},
        "flow_proxy_period_weights_blended": blended,
    }

    METRO_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
