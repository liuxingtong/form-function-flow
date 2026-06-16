#!/usr/bin/env python3
"""
仅重跑「分时段 × 多方式」路网交通分配（Frank–Wolfe 链式），不重新生成 WorldPop / OD。

前提：``out_dir`` 下已有 ``synthetic_od_modal_by_period_long.csv``（及可选已有 ``flow_road_assignment_edges.csv``）。

用法（仓库根目录）：
  python scripts/rerun_period_edge_assignment.py ^
    --out-dir output/synthetic_flow_worldpop ^
    --rebuild-assignment-edges

更新 ``method_manifest.json`` 中的 assignment 段落（若存在则合并写入）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from generate_worldpop_four_step_flow import (  # noqa: E402
    assign_period_edges,
    load_time_slice_catalog,
)


def _json_default(o: object) -> str:
    try:
        import numpy as np

        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except Exception:
        pass
    return str(o)


def main() -> int:
    ap = argparse.ArgumentParser(description="仅重跑分时段路网分配（基于已有 modal OD 长表）")
    ap.add_argument("--out-dir", type=Path, default=REPO / "output" / "synthetic_flow_worldpop")
    ap.add_argument(
        "--modal-od-csv",
        type=Path,
        default=None,
        help="默认: <out-dir>/synthetic_od_modal_by_period_long.csv",
    )
    ap.add_argument(
        "--assignment-edges-csv",
        type=Path,
        default=None,
        help="默认: <out-dir>/flow_road_assignment_edges.csv",
    )
    ap.add_argument("--units", type=Path, default=REPO / "output" / "function" / "数据包" / "01_units.gpkg")
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "site_3km")
    ap.add_argument("--site-json", type=Path, default=REPO / "data" / "site_3km" / "SITE.json")
    ap.add_argument("--site-buffer-m", type=float, default=3000.0)
    ap.add_argument("--rebuild-assignment-edges", action="store_true")
    ap.add_argument("--time-slices-csv", type=Path, default=None)
    ap.add_argument("--aon-max-origins", type=int, default=0)
    ap.add_argument("--assignment-iters", type=int, default=5)
    ap.add_argument("--assignment-delay-alpha", type=float, default=0.22)
    ap.add_argument("--assignment-delay-power", type=float, default=2.0)
    ap.add_argument(
        "--assignment-scheme",
        choices=("frank_wolfe", "aon_replace"),
        default="frank_wolfe",
    )
    ns = ap.parse_args()

    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modal_path = Path(ns.modal_od_csv) if ns.modal_od_csv else out_dir / "synthetic_od_modal_by_period_long.csv"
    if not modal_path.is_file():
        raise SystemExit(f"缺少 modal OD 表: {modal_path}")

    edges_csv = Path(ns.assignment_edges_csv) if ns.assignment_edges_csv else out_dir / "flow_road_assignment_edges.csv"
    if ns.rebuild_assignment_edges or not edges_csv.is_file():
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "build_flow_road_assignment_edges.py"),
            "--units",
            str(ns.units),
            "--data-root",
            str(ns.data_root),
            "--site-buffer-m",
            str(ns.site_buffer_m),
            "--out-csv",
            str(edges_csv),
        ]
        if ns.site_json.is_file():
            cmd.extend(["--site-json", str(ns.site_json)])
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
        build_meta: dict[str, Any] = {"assignment_edges_csv": str(edges_csv), "rebuilt": True, "command": cmd}
    else:
        build_meta = {"assignment_edges_csv": str(edges_csv), "rebuilt": False}

    _t_ids, _dt_map, _slices_df, chain_order = load_time_slice_catalog(ns.time_slices_csv)
    modal_od = pd.read_csv(modal_path, encoding="utf-8-sig")
    _edge_df, assign_stats = assign_period_edges(
        modal_od,
        edges_csv,
        out_dir,
        max_origins=int(ns.aon_max_origins),
        n_iters=int(ns.assignment_iters),
        delay_alpha=float(ns.assignment_delay_alpha),
        delay_power=float(ns.assignment_delay_power),
        scheme=str(ns.assignment_scheme),
        period_chain_order=chain_order,
    )
    assignment_meta: dict[str, Any] = {**build_meta, **assign_stats, "skipped": False}

    manifest_path = out_dir / "method_manifest.json"
    if manifest_path.is_file():
        try:
            meta = json.loads(manifest_path.read_text(encoding="utf-8"))
            meta["assignment"] = assignment_meta
            meta.setdefault("parameters", {})
            if isinstance(meta["parameters"], dict):
                meta["parameters"]["skip_assignment"] = False
                meta["parameters"]["force_rebuild_assignment_edges"] = bool(ns.rebuild_assignment_edges)
            manifest_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"WARN: 未更新 method_manifest.json: {exc}", file=sys.stderr)

    print(json.dumps({"out_dir": str(out_dir), "assignment": assignment_meta}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
