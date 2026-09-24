"""TypeSafe-compatible /v1/systemone HTTP Server.

Serves the Laya Browser Decision model over HTTP as a direct drop-in replacement
for TypeSafe Jev API.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from katai.core.engine import KataiEngine

DEFAULT_PORT = int(os.environ.get("KATAI_SERVER_PORT", "8791"))
DEFAULT_HOST = os.environ.get("KATAI_SERVER_HOST", "127.0.0.1")


class SystemOneHandler(BaseHTTPRequestHandler):
    engine: Optional[KataiEngine] = None
    log_history: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {"requests": 0, "errors": 0}

    def log_message(self, format: str, *args: Any):
        # Suppress default stdlib access log; we print a custom summary
        pass

    def _send_json(self, code: int, body: Dict[str, Any]):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_GET(self):
        if self.path in ("/", "/health", "/v1/health"):
            self._send_json(
                200,
                {
                    "status": "online",
                    "model": f"laya-{self.engine.checkpoint_name if self.engine else 'unknown'}",
                    "device": str(self.engine.device if self.engine else 'unknown'),
                    "requests": self.stats["requests"],
                    "errors": self.stats["errors"],
                    "recent_latencies_ms": [x.get("ms") for x in self.log_history[-10:]],
                },
            )
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self._send_json(404, {"error": "Endpoint not found. Use POST /v1/systemone"})
            return

        try:
            content_len = int(self.headers.get("Content-Length", 0))
            req_raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
            payload = json.loads(req_raw.decode("utf-8"))

            state = payload.get("state", {})
            questions = payload.get("questions", {})

            if not questions:
                self._send_json(400, {"error": "Missing 'questions' in request body"})
                return

            t0 = time.perf_counter()
            result = self.engine.predict(state, questions)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            self.stats["requests"] += 1
            log_entry = {
                "ms": latency_ms,
                "questions": len(questions),
                "options": sum(len(q.get("criteria", {})) for q in questions.values()),
                "passes": result.get("passes", 1),
                "tokens": result.get("usage", {}).get("input_tokens", 0),
            }
            self.log_history.append(log_entry)
            if len(self.log_history) > 100:
                self.log_history.pop(0)

            op_choice = result.get("answers", {}).get("operation", {}).get("choice", "?")
            op_conf = result.get("answers", {}).get("operation", {}).get("confidence", 0.0)
            print(
                f"[SystemOne Server] {log_entry['questions']} Qs | {log_entry['options']} opts | "
                f"{log_entry['tokens']} tok | {latency_ms:.1f}ms -> op={op_choice} ({op_conf:.2f})",
                file=sys.stderr,
                flush=True,
            )

            self._send_json(200, result)

        except Exception as e:
            self.stats["errors"] += 1
            traceback.print_exc()
            self._send_json(500, {"error": f"{type(e).__name__}: {str(e)}"})


def start_server(
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    checkpoint: str = "v10s",
    device: Optional[str] = None,
    engine: Optional[KataiEngine] = None,
):
    """Start the SystemOne HTTP server."""
    if not engine:
        engine = KataiEngine(checkpoint=checkpoint, device=device)

    SystemOneHandler.engine = engine
    server = ThreadingHTTPServer((host, port), SystemOneHandler)
    print(f"Katai SystemOne Server listening at http://{host}:{port}/v1/systemone", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SystemOne server...", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    ck = sys.argv[2] if len(sys.argv) > 2 else "v10s"
    start_server(port=port, checkpoint=ck)
