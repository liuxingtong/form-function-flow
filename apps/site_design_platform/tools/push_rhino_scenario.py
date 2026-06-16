from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with request.urlopen(req, timeout=20) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {"ok": True}


def _build_ground_fc(payload: dict) -> dict:
    existing = payload.get("rhino_ground")
    if isinstance(existing, dict) and isinstance(existing.get("features"), list):
        return existing

    parcels = payload.get("rhino_parcels", {})
    if not isinstance(parcels, dict):
        return {"type": "FeatureCollection", "features": []}
    feats = parcels.get("features", [])
    if not isinstance(feats, list):
        return {"type": "FeatureCollection", "features": []}

    ground_layers = {"GROUND", "TOD_GROUND"}
    out = []
    for f in feats:
        layer = str((f or {}).get("properties", {}).get("layer", "")).upper()
        if layer in ground_layers:
            out.append(f)
    return {"type": "FeatureCollection", "features": out}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Push Rhino-exported scenario JSON to Site Design Platform /api/site-design/rhino/sync."
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to Rhino-exported scenario JSON (must contain blocks[]).",
    )
    ap.add_argument(
        "--url",
        default="http://127.0.0.1:8088/api/site-design/rhino/sync",
        help="Sync endpoint URL.",
    )
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"input not found: {args.input}")

    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("blocks"), list):
        raise ValueError("invalid scenario JSON: expected object with blocks[]")
    payload["rhino_ground"] = _build_ground_fc(payload)

    out = post_json(args.url, payload)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
