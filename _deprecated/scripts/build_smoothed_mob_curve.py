#!/usr/bin/env python3
"""
Build smoothed 8-bin mobility composition curves for each unit.

The script keeps fixed clock bins (T_ORDER) and estimates per-unit
time-share vector q_1..q_8 by optimizing:

  cross_entropy(raw_share, softmax(z))
  + lambda1 * RW1(z)
  + lambda2 * RW2(z)
  + lambda_c * cyclic(z_1 - z_8)^2

Optionally it applies a soft cap on max share (p_max) by shrinking
toward uniform distribution after optimization.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from time_slice_constants import T_ORDER  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO / "output" / "flow" / "output_mobility_state" / "mob_state.csv"
DEFAULT_OUT = REPO / "output" / "flow" / "output_mobility_state" / "mob_temporal_curve.csv"
DEFAULT_QA = REPO / "output" / "flow" / "output_mobility_state" / "mob_temporal_curve_qa.json"


@dataclass
class HyperParams:
    lambda1: float = 0.08
    lambda2: float = 0.03
    lambda_c: float = 0.05
    lr: float = 0.06
    max_iter: int = 500
    tol: float = 1e-7
    p_max: float = 0.35
    apply_peak_cap: bool = True


def _softmax(z: np.ndarray) -> np.ndarray:
    z0 = z - np.max(z)
    ez = np.exp(z0)
    return ez / (ez.sum() + 1e-12)


def _make_raw_signal(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    # Positive proxy from robust z-scores then clipped to avoid zero rows.
    x = np.zeros(len(df), dtype=float)
    for c in cols:
        if c not in df.columns:
            continue
        v = df[c].to_numpy(dtype=float)
        med = float(np.nanmedian(v))
        mad = float(np.nanmedian(np.abs(v - med))) + 1e-9
        z = (v - med) / (1.4826 * mad)
        x += np.maximum(z, -2.5) + 2.8
    x = np.nan_to_num(x, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(x, 1e-6, None)


def _loss_grad(z: np.ndarray, r: np.ndarray, hp: HyperParams) -> tuple[float, np.ndarray]:
    q = _softmax(z)
    # CE(r, q)
    ce = -float(np.sum(r * np.log(q + 1e-12)))
    g = q - r

    # RW1: sum_{t=2..8} (z_t - z_{t-1})^2
    rw1 = 0.0
    for t in range(1, len(z)):
        d = z[t] - z[t - 1]
        rw1 += d * d
        gd = 2.0 * d * hp.lambda1
        g[t] += gd
        g[t - 1] -= gd

    # RW2: sum_{t=3..8} (z_t - 2z_{t-1} + z_{t-2})^2
    rw2 = 0.0
    for t in range(2, len(z)):
        d2 = z[t] - 2.0 * z[t - 1] + z[t - 2]
        rw2 += d2 * d2
        gd2 = 2.0 * d2 * hp.lambda2
        g[t] += gd2
        g[t - 1] += -2.0 * gd2
        g[t - 2] += gd2

    # Cyclic continuity: (z1 - z8)^2
    dc = z[0] - z[-1]
    cyc = dc * dc
    g[0] += 2.0 * hp.lambda_c * dc
    g[-1] -= 2.0 * hp.lambda_c * dc

    loss = ce + hp.lambda1 * rw1 + hp.lambda2 * rw2 + hp.lambda_c * cyc
    return float(loss), g


def _fit_unit_curve(raw_share: np.ndarray, hp: HyperParams) -> np.ndarray:
    r = np.clip(raw_share.astype(float), 1e-12, None)
    r = r / (r.sum() + 1e-12)
    z = np.log(r + 1e-12)
    prev = 1e18
    lr = hp.lr
    for _ in range(hp.max_iter):
        loss, grad = _loss_grad(z, r, hp)
        if abs(prev - loss) < hp.tol:
            break
        prev = loss
        # tiny backoff for stability
        z_new = z - lr * grad
        new_loss, _ = _loss_grad(z_new, r, hp)
        if new_loss > loss:
            lr = max(lr * 0.5, 1e-4)
            z = z - lr * grad
        else:
            z = z_new
            lr = min(lr * 1.03, hp.lr)
    q = _softmax(z)

    if hp.apply_peak_cap:
        mx = float(q.max())
        if mx > hp.p_max:
            # shrink toward uniform until max share <= p_max
            u = np.full_like(q, 1.0 / len(q))
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                qq = (1.0 - mid) * q + mid * u
                if float(qq.max()) > hp.p_max:
                    lo = mid
                else:
                    hi = mid
            q = (1.0 - hi) * q + hi * u
            q = q / (q.sum() + 1e-12)
    return q


def build_curves(
    df: pd.DataFrame,
    hp: HyperParams,
    raw_proxy_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    t_index = {t: i for i, t in enumerate(T_ORDER)}
    work = df.copy()
    work = work[work["t_id"].isin(T_ORDER)].copy()
    work["_t_ord"] = work["t_id"].map(t_index)
    work = work.sort_values(["unit_id", "_t_ord"]).reset_index(drop=True)
    if work["unit_id"].nunique() == 0:
        raise ValueError("No rows after filtering to T_ORDER.")

    raw_signal = _make_raw_signal(work, raw_proxy_cols)
    work["_raw_signal"] = raw_signal

    rows: list[dict] = []
    peak_raw: list[float] = []
    peak_smooth: list[float] = []
    cyc_gap_raw: list[float] = []
    cyc_gap_smooth: list[float] = []

    for uid, g in work.groupby("unit_id", sort=False):
        gg = g.sort_values("_t_ord")
        if len(gg) != len(T_ORDER):
            # require full 8 bins for stable fit
            continue
        y = gg["_raw_signal"].to_numpy(dtype=float)
        raw = y / (y.sum() + 1e-12)
        q = _fit_unit_curve(raw, hp)
        amp = float(np.log1p(np.sum(y)))
        peak_raw.append(float(raw.max()))
        peak_smooth.append(float(q.max()))
        cyc_gap_raw.append(float(abs(raw[0] - raw[-1])))
        cyc_gap_smooth.append(float(abs(q[0] - q[-1])))
        for i, tid in enumerate(T_ORDER):
            rows.append(
                {
                    "unit_id": uid,
                    "t_id": tid,
                    "q_share": float(q[i]),
                    "raw_share": float(raw[i]),
                    "amp": amp,
                    **{f"q_{j+1}": float(q[j]) for j in range(len(T_ORDER))},
                }
            )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No complete 8-bin units found; cannot build curves.")

    # normalization safety
    sm = out.groupby("unit_id")["q_share"].sum().rename("sum_q")
    out = out.merge(sm, left_on="unit_id", right_index=True, how="left")
    out["q_share"] = out["q_share"] / (out["sum_q"] + 1e-12)
    out = out.drop(columns=["sum_q"])

    qa = {
        "units": int(out["unit_id"].nunique()),
        "rows": int(len(out)),
        "peak_raw_p95": float(np.quantile(np.array(peak_raw), 0.95)),
        "peak_smooth_p95": float(np.quantile(np.array(peak_smooth), 0.95)),
        "cyc_gap_raw_mean": float(np.mean(cyc_gap_raw)),
        "cyc_gap_smooth_mean": float(np.mean(cyc_gap_smooth)),
        "hyper_params": {
            "lambda1": hp.lambda1,
            "lambda2": hp.lambda2,
            "lambda_c": hp.lambda_c,
            "lr": hp.lr,
            "max_iter": hp.max_iter,
            "tol": hp.tol,
            "p_max": hp.p_max,
            "apply_peak_cap": hp.apply_peak_cap,
        },
        "raw_proxy_cols": raw_proxy_cols,
        "t_order": list(T_ORDER),
    }
    return out, qa


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--qa-json", type=Path, default=DEFAULT_QA)
    ap.add_argument(
        "--raw-proxy-cols",
        nargs="+",
        default=["traffic_intensity", "population_density", "stay_proxy"],
        help="Columns used to build raw per-bin signal before smoothing.",
    )
    ap.add_argument("--lambda1", type=float, default=0.08)
    ap.add_argument("--lambda2", type=float, default=0.03)
    ap.add_argument("--lambda-c", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--p-max", type=float, default=0.35)
    ap.add_argument(
        "--disable-peak-cap",
        action="store_true",
        help="Disable max-share cap post-processing.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.in_csv.is_file():
        raise FileNotFoundError(f"Missing input: {args.in_csv}")

    df = pd.read_csv(args.in_csv)
    need = {"unit_id", "t_id"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Input missing required columns: {miss}")

    hp = HyperParams(
        lambda1=float(args.lambda1),
        lambda2=float(args.lambda2),
        lambda_c=float(args.lambda_c),
        lr=float(args.lr),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        p_max=float(args.p_max),
        apply_peak_cap=not bool(args.disable_peak_cap),
    )
    out, qa = build_curves(df=df, hp=hp, raw_proxy_cols=list(args.raw_proxy_cols))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.qa_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    args.qa_json.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.qa_json}")
    print(
        "QA:",
        f"units={qa['units']}",
        f"peak_raw_p95={qa['peak_raw_p95']:.4f}",
        f"peak_smooth_p95={qa['peak_smooth_p95']:.4f}",
    )


if __name__ == "__main__":
    main()
