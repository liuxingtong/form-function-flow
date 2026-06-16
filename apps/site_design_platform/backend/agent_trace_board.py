from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from apps.site_design_platform.backend.ai_agent.trace_store import tail_traces


def _html() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'/><title>Agent Trace Board</title>
<style>body{font-family:Segoe UI,Arial;margin:0;background:#0f172a;color:#e2e8f0}header{padding:12px 16px;background:#111827;position:sticky;top:0}table{width:100%;border-collapse:collapse;font-size:12px}th,td{border-bottom:1px solid #243041;padding:6px 8px;vertical-align:top}th{position:sticky;top:48px;background:#0b1220}.ok{color:#34d399}.err{color:#f87171}.mono{font-family:Consolas,monospace;white-space:pre-wrap}</style></head>
<body><header><b>Agent Trace Board</b> <span id='s'></span></header><table><thead><tr><th>time</th><th>api</th><th>engine</th><th>ok</th><th>ms</th><th>model</th><th>prompt</th><th>detail</th></tr></thead><tbody id='tb'></tbody></table>
<script>
async function load(){const r=await fetch('/api/traces?limit=200');const d=await r.json();const tb=document.getElementById('tb');tb.innerHTML='';(d.items||[]).reverse().forEach(x=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${x.ts||''}</td><td>${x.api||''}</td><td>${x.engine||''}</td><td class='${x.ok?'ok':'err'}'>${x.ok?'OK':'ERR'}</td><td>${x.ms??''}</td><td>${x.model||''}</td><td class='mono'>${(x.prompt||'').slice(0,120)}</td><td class='mono'>${x.error||x.note||''}</td>`;tb.appendChild(tr)});document.getElementById('s').textContent=`rows:${(d.items||[]).length} updated:${new Date().toLocaleTimeString()}`;}
load();setInterval(load,2000);
</script></body></html>"""


def run_trace_board(root: Path, host: str = "127.0.0.1", port: int = 8099) -> None:
    class H(BaseHTTPRequestHandler):
        def _json(self, obj: dict, code: int = 200):
            b = json.dumps(obj, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == '/api/traces':
                q = parse_qs(u.query)
                limit = int((q.get('limit') or ['120'])[0])
                self._json({'items': tail_traces(root, limit)})
                return
            if u.path in ['/', '/index.html']:
                b = _html().encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            self._json({'error': 'not found'}, 404)

    srv = ThreadingHTTPServer((host, port), H)
    print(f'Agent trace board at http://{host}:{port}/')
    srv.serve_forever()
