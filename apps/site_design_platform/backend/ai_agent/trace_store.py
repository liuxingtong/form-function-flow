from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _trace_file(root: Path) -> Path:
    p = root / "data" / "site_design_platform" / "logs" / "agent_trace.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append_trace(root: Path, event: dict[str, Any]) -> None:
    row = dict(event)
    row.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    f = _trace_file(root)
    line = json.dumps(row, ensure_ascii=False)
    with _LOCK:
      with f.open("a", encoding="utf-8") as fp:
        fp.write(line + "\n")


def tail_traces(root: Path, limit: int = 120) -> list[dict[str, Any]]:
    f = _trace_file(root)
    if not f.exists():
      return []
    with _LOCK:
      lines = f.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for ln in lines[-max(1, limit):]:
      try:
        out.append(json.loads(ln))
      except Exception:
        continue
    return out
