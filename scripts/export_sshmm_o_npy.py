"""
Export observation_normalized.csv → NumPy array O with shape (R, T, L) for SSHMM training.

Official reference implementation loads a single .npy file:
  O = np.load('.../o_1hours_31day.npy')   # shape [R, total_hours, L]
See https://github.com/XTxiatong/SSHMM/blob/master/code/2training_spark_SSHMM.py

This script aligns channel order with sshmm_manifest.json when provided:
  L = 3 flow dims (norm_o_arrive, norm_o_leave, norm_o_stay)
    + len(poi_columns) semantics (norm_<poi_col>).

Run from repo root:
  python scripts/export_sshmm_o_npy.py \\
    --normalized-csv data/sshmm/out_demo/observation_normalized.csv \\
    --manifest data/sshmm/out_demo/sshmm_manifest.json \\
    --out-npy data/sshmm/out_demo/o_hours.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _norm_feature_cols(manifest: dict | None, df: pd.DataFrame) -> list[str]:
    if manifest is not None:
        poi = list(manifest.get("poi_columns") or [])
        flows = ["norm_o_arrive", "norm_o_leave", "norm_o_stay"]
        cols = flows + [f"norm_{c}" for c in poi]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"CSV missing columns expected from manifest: {missing}")
        return cols

    flows = ["norm_o_arrive", "norm_o_leave", "norm_o_stay"]
    have_flow = [c for c in flows if c in df.columns]
    rest = sorted(c for c in df.columns if c.startswith("norm_") and c not in flows)
    if not have_flow and not rest:
        raise SystemExit("No norm_* columns found in CSV.")
    return have_flow + rest


def main() -> int:
    p = argparse.ArgumentParser(description="Export SSHMM O tensor as .npy (R,T,L).")
    p.add_argument("--normalized-csv", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None, help="sshmm_manifest.json (recommended)")
    p.add_argument("--out-npy", type=Path, required=True)
    p.add_argument(
        "--out-meta",
        type=Path,
        default=None,
        help="Optional JSON: unit_ids, timestamps (ISO), L, feature_cols",
    )
    p.add_argument(
        "--fill-missing",
        choices=("zero", "nan"),
        default="zero",
        help="Value for (unit, time) pairs absent from long table (default: zero).",
    )
    args = p.parse_args()

    csv_path = args.normalized_csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"Missing CSV: {csv_path}")

    manifest: dict | None = None
    if args.manifest is not None:
        mp = args.manifest.resolve()
        if not mp.is_file():
            raise SystemExit(f"Missing manifest: {mp}")
        manifest = json.loads(mp.read_text(encoding="utf-8"))

    df = pd.read_csv(csv_path)
    if "unit_id" not in df.columns or "timestamp" not in df.columns:
        raise SystemExit("CSV must contain unit_id and timestamp.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    feat_cols = _norm_feature_cols(manifest, df)
    L = len(feat_cols)

    units = sorted(df["unit_id"].astype(str).unique())
    times = sorted(df["timestamp"].unique())

    # Dense index for fast pivot
    ui = {u: i for i, u in enumerate(units)}
    ti = {t: i for i, t in enumerate(times)}
    R, T = len(units), len(times)

    O = np.zeros((R, T, L), dtype=np.float64)
    if args.fill_missing == "nan":
        O.fill(np.nan)

    # Fill observed rows
    uid_s = df["unit_id"].astype(str).values
    ts_s = df["timestamp"].values
    ir = np.array([ui[x] for x in uid_s], dtype=np.intp)
    it = np.array([ti[pd.Timestamp(x)] for x in ts_s], dtype=np.intp)
    X = df[feat_cols].astype(np.float64).values
    O[ir, it, :] = X

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, O)

    meta = {
        "shape_RT_L": [int(R), int(T), int(L)],
        "feature_cols": feat_cols,
        "unit_ids": units,
        "timestamps_iso": [pd.Timestamp(t).isoformat() for t in times],
    }
    if manifest is not None:
        meta["manifest_L"] = manifest.get("L")

    out_meta = args.out_meta
    if out_meta is None:
        out_meta = args.out_npy.with_suffix(".meta.json")
    out_meta = out_meta.resolve()
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {args.out_npy} shape=({R}, {T}, {L}) dtype={O.dtype}")
    print(f"Wrote {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
