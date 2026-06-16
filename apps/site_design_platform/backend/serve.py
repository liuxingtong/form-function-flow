from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from apps.site_design_platform.backend.cluster_generator.generator import (
    ClusterGenerateRequest,
    generate_clusters,
)
from apps.site_design_platform.backend.ai_agent.service import infer_audience_profile, infer_floor_stack
from apps.site_design_platform.backend.ai_agent.trace_store import append_trace
from apps.site_design_platform.backend.site_alignment import align_payload_to_site


def run_server(root: Path, host: str, port: int) -> None:
    rhino_dir = root / "data" / "site_design_platform" / "scenarios"
    rhino_dir.mkdir(parents=True, exist_ok=True)
    rhino_file = rhino_dir / "rhino_live.json"
    econ_snapshot_file = rhino_dir / "economics_snapshots.jsonl"
    econ_latest_file = rhino_dir / "economics_snapshot_latest.json"
    site_file = root / "data" / "site_dxf" / "crs84" / "SITE.geojson"

    class RootHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def _send_json(self, status: int, payload: dict, extra_headers: dict[str, str] | None = None) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _extract_layer_fc(self, payload: dict, source_key: str, layer_names: set[str]) -> dict:
            source = payload.get(source_key, {"type": "FeatureCollection", "features": []})
            if not isinstance(source, dict):
                return {"type": "FeatureCollection", "features": []}
            feats = source.get("features", [])
            if not isinstance(feats, list):
                return {"type": "FeatureCollection", "features": []}
            filtered = []
            for f in feats:
                layer = str((f or {}).get("properties", {}).get("layer", "")).upper()
                if layer in layer_names:
                    filtered.append(f)
            return {"type": "FeatureCollection", "features": filtered}

        def _extract_ground_from_blocks(self, payload: dict) -> dict:
            blocks = payload.get("blocks", [])
            if not isinstance(blocks, list):
                return {"type": "FeatureCollection", "features": []}
            features = []
            for i, b in enumerate(blocks):
                if not isinstance(b, dict):
                    continue
                fn = str(b.get("function", "")).upper()
                if fn not in {"GROUND", "TOD_GROUND"}:
                    continue
                g = b.get("geometry")
                if not isinstance(g, dict):
                    continue
                bid = str(b.get("id", f"ground_block_{i+1}"))
                features.append(
                    {
                        "type": "Feature",
                        "properties": {"id": bid, "layer": fn},
                        "geometry": g,
                    }
                )
            return {"type": "FeatureCollection", "features": features}

        def do_POST(self) -> None:
            if self.path == "/api/site-design/economics/snapshot":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        self._send_json(400, {"error": "invalid payload: expected object"})
                        return
                    row = {
                        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "saved_at_epoch_ms": int(time.time() * 1000),
                        "source": "ui_recalc",
                        **payload,
                    }
                    with econ_snapshot_file.open("a", encoding="utf-8") as fp:
                        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    econ_latest_file.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "saved": str(econ_snapshot_file),
                            "latest": str(econ_latest_file),
                            "scenario_name": row.get("scenario_name", "scenario_mvp"),
                        },
                    )
                except Exception as exc:
                    traceback.print_exc()
                    self._send_json(400, {"error": str(exc)})
                return

            if self.path == "/api/site-design/rhino/sync":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict) or not isinstance(payload.get("blocks"), list):
                        self._send_json(400, {"error": "invalid payload: expected scenario JSON with blocks[]"})
                        return
                    # Map Rhino local SITE outline → WGS84 SITE.geojson; affine-transform all geometries.
                    aligned = False
                    if payload.get("rhino_site_outline") and site_file.exists():
                        try:
                            site_fc = json.loads(site_file.read_text(encoding="utf-8-sig"))
                            aligned = align_payload_to_site(payload, site_fc)
                        except Exception:
                            traceback.print_exc()
                    rhino_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    self._send_json(200, {"ok": True, "saved": str(rhino_file), "blocks": len(payload.get("blocks", [])), "aligned_to_site": aligned})
                except Exception as exc:
                    traceback.print_exc()
                    self._send_json(400, {"error": str(exc)})
                return

            if self.path == "/api/site-design/ai/floor-stack":
                t0 = time.time()
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    payload = json.loads(raw.decode("utf-8"))
                    prompt = str(payload.get("prompt", "")).strip()
                    audience_profile = str(payload.get("audience_profile", "")).strip()
                    blocks_fc = payload.get("blocks", {"type": "FeatureCollection", "features": []})
                    parcels_fc = payload.get("parcels", {"type": "FeatureCollection", "features": []})
                    if not prompt:
                        self._send_json(400, {"error": "prompt is required"})
                        return
                    out = infer_floor_stack(prompt, blocks_fc, parcels_fc, audience_profile)
                    append_trace(
                        root,
                        {
                            "api": "floor-stack",
                            "ok": True,
                            "engine": "agent",
                            "ms": int((time.time() - t0) * 1000),
                            "model": os.getenv("SITE_AI_MODEL", ""),
                            "prompt": prompt[:400],
                            "blocks": len((blocks_fc or {}).get("features", []) if isinstance(blocks_fc, dict) else []),
                        },
                    )
                    self._send_json(200, out)
                except Exception as exc:
                    traceback.print_exc()
                    append_trace(
                        root,
                        {
                            "api": "floor-stack",
                            "ok": False,
                            "engine": "agent",
                            "ms": int((time.time() - t0) * 1000),
                            "model": os.getenv("SITE_AI_MODEL", ""),
                            "prompt": str(payload.get("prompt", ""))[:400] if "payload" in locals() else "",
                            "error": str(exc),
                        },
                    )
                    self._send_json(400, {"error": str(exc)})
                return

            if self.path == "/api/site-design/ai/complete-audience":
                t0 = time.time()
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    payload = json.loads(raw.decode("utf-8"))
                    prompt = str(payload.get("prompt", "")).strip()
                    if not prompt:
                        self._send_json(400, {"error": "prompt is required"})
                        return
                    out = infer_audience_profile(prompt)
                    append_trace(
                        root,
                        {
                            "api": "complete-audience",
                            "ok": True,
                            "engine": str(out.get("source", "agent")),
                            "ms": int((time.time() - t0) * 1000),
                            "model": os.getenv("SITE_AI_MODEL", ""),
                            "prompt": prompt[:400],
                            "note": str(out.get("note", "")),
                        },
                    )
                    self._send_json(200, out)
                except Exception as exc:
                    traceback.print_exc()
                    append_trace(
                        root,
                        {
                            "api": "complete-audience",
                            "ok": False,
                            "engine": "agent",
                            "ms": int((time.time() - t0) * 1000),
                            "model": os.getenv("SITE_AI_MODEL", ""),
                            "prompt": str(payload.get("prompt", ""))[:400] if "payload" in locals() else "",
                            "error": str(exc),
                        },
                    )
                    self._send_json(400, {"error": str(exc)})
                return

            if self.path != "/api/site-design/cluster/generate":
                self._send_json(404, {"error": "not found"})
                return

            try:
                # Deprecated: cluster generator endpoint is planned for removal in a future release.
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                req = ClusterGenerateRequest(
                    scenario_name=payload.get("scenario_name", "scenario_mvp"),
                    seed=int(payload.get("seed", 42)),
                    template_id=payload.get("template_id", "OFFICE_CAMPUS_BOX"),
                    zone_id=payload.get("zone_id", "CBD"),
                    site_geojson=payload.get("site_geojson", {"type": "FeatureCollection", "features": []}),
                    zone_geojson=payload.get("zone_geojson", {"type": "FeatureCollection", "features": []}),
                    constraints=payload.get("constraints", {}),
                    intensity=payload.get("intensity", {}),
                )
                out = generate_clusters(req)
                self._send_json(
                    200,
                    {
                        "blocks": out.blocks,
                        "diagnostics": out.diagnostics,
                        "stats": out.stats,
                        "deprecated": True,
                        "deprecation_note": "cluster/generate will be removed in a future release",
                    },
                )
            except Exception as exc:
                traceback.print_exc()
                self._send_json(400, {"error": str(exc)})

        def do_GET(self) -> None:
            if self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if self.path == "/api/site-design/rhino/latest":
                if not rhino_file.exists():
                    self._send_json(404, {"error": "rhino scenario not found"})
                    return
                try:
                    payload = json.loads(rhino_file.read_text(encoding="utf-8-sig"))
                    if not isinstance(payload, dict):
                        self._send_json(400, {"error": "invalid scenario file"})
                        return
                    stat = rhino_file.stat()
                    blocks = payload.get("blocks", [])
                    block_count = len(blocks) if isinstance(blocks, list) else 0
                    self._send_json(
                        200,
                        payload,
                        {
                            "Cache-Control": "no-store",
                            "X-Rhino-Updated-At": str(stat.st_mtime),
                            "X-Rhino-Blocks-Count": str(block_count),
                        },
                    )
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return
            if self.path == "/api/site-design/rhino/parcels":
                if not rhino_file.exists():
                    self._send_json(404, {"error": "rhino scenario not found"})
                    return
                try:
                    payload = json.loads(rhino_file.read_text(encoding="utf-8-sig"))
                    parcels = payload.get("rhino_parcels", {"type": "FeatureCollection", "features": []})
                    self._send_json(200, parcels if isinstance(parcels, dict) else {"type": "FeatureCollection", "features": []})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return
            if self.path == "/api/site-design/rhino/original-buildings":
                if not rhino_file.exists():
                    self._send_json(404, {"error": "rhino scenario not found"})
                    return
                try:
                    payload = json.loads(rhino_file.read_text(encoding="utf-8-sig"))
                    fc = payload.get("rhino_original_buildings", {"type": "FeatureCollection", "features": []})
                    self._send_json(200, fc if isinstance(fc, dict) else {"type": "FeatureCollection", "features": []})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return
            if self.path == "/api/site-design/rhino/walking":
                if not rhino_file.exists():
                    self._send_json(404, {"error": "rhino scenario not found"})
                    return
                try:
                    payload = json.loads(rhino_file.read_text(encoding="utf-8-sig"))
                    fc = payload.get("rhino_walking", {"type": "FeatureCollection", "features": []})
                    self._send_json(200, fc if isinstance(fc, dict) else {"type": "FeatureCollection", "features": []})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return
            if self.path == "/api/site-design/rhino/ground":
                if not rhino_file.exists():
                    self._send_json(404, {"error": "rhino scenario not found"})
                    return
                try:
                    payload = json.loads(rhino_file.read_text(encoding="utf-8-sig"))
                    fc = payload.get("rhino_ground", {"type": "FeatureCollection", "features": []})
                    if not isinstance(fc, dict) or not isinstance(fc.get("features"), list) or not fc.get("features"):
                        ground_like_layers = {"GROUND", "TOD_GROUND"}
                        from_parcels = self._extract_layer_fc(payload, "rhino_parcels", ground_like_layers)
                        from_original = self._extract_layer_fc(payload, "rhino_original_buildings", ground_like_layers)
                        from_blocks = self._extract_ground_from_blocks(payload)
                        fc = {
                            "type": "FeatureCollection",
                            "features": [
                                *(from_parcels.get("features", []) or []),
                                *(from_original.get("features", []) or []),
                                *(from_blocks.get("features", []) or []),
                            ],
                        }
                    self._send_json(200, fc if isinstance(fc, dict) else {"type": "FeatureCollection", "features": []})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return
            super().do_GET()

    server = ThreadingHTTPServer((host, port), RootHandler)
    print(f"Serving {root} at http://{host}:{port}/apps/site_design_platform/frontend/index.html")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve site design platform frontend.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = ap.parse_args()
    run_server(args.root.resolve(), args.host, args.port)


if __name__ == "__main__":
    main()
