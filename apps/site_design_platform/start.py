from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from apps.site_design_platform.backend.serve import run_server
from apps.site_design_platform.backend.agent_trace_board import run_trace_board


def load_env_file(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def main() -> None:
    ap = argparse.ArgumentParser(description="One-command startup for site design platform.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--trace-port", type=int, default=8099)
    ap.add_argument("--no-open", action="store_true", help="Do not auto-open browser.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    load_env_file(repo_root)

    summary_file = repo_root / "data" / "site_design_platform" / "datasets" / "summary.json"
    if summary_file.exists():
        import json as _json
        summary = _json.loads(summary_file.read_text(encoding="utf-8"))
        print(f"Using pre-built datasets: buildings={summary.get('buildings_features')}, landuse={summary.get('landuse_features')}")
    else:
        from apps.site_design_platform.tools.prepare_datasets import prepare
        summary = prepare(repo_root)
        print(
            f"Prepared datasets: buildings={summary['buildings_features']}, "
            f"landuse={summary['landuse_features']}, "
            f"height=[{summary.get('height_min', 0):.2f}, {summary.get('height_max', 0):.2f}] m"
        )

    url = f"http://{args.host}:{args.port}/apps/site_design_platform/frontend/index.html"
    trace_url = f"http://{args.host}:{args.trace_port}/"
    t = threading.Thread(target=run_trace_board, args=(repo_root, args.host, args.trace_port), daemon=True)
    t.start()
    print(f"Agent trace board: {trace_url}")
    if not args.no_open:
        webbrowser.open(url)
        webbrowser.open(trace_url)
    run_server(repo_root, args.host, args.port)


if __name__ == "__main__":
    main()
