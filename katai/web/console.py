"""Interactive Web Console for Katai Browser Agent.

Provides a modern GUI with live browser screenshots, DOM element inspection,
step-by-step execution, calibrated decision probabilities, and action space exploration.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
import traceback
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from katai.core.agent import KataiAgent
from katai.core.engine import (
    DEFAULT_CHECKPOINT,
    KataiEngine,
    list_available_models,
)

DEFAULT_WEB_PORT = int(os.environ.get("KATAI_WEB_PORT", "8766"))
DEFAULT_WEB_HOST = os.environ.get("KATAI_WEB_HOST", "127.0.0.1")
STATIC_DIR = Path(__file__).resolve().parent / "static"


class ModelManager:
    """Thread-safe manager for model checkpoints, hot-swapping, and progress status."""

    _lock = threading.Lock()
    current_model: str = DEFAULT_CHECKPOINT
    current_device: str = "cpu"
    engine: Optional[KataiEngine] = None
    loading_status: str = "idle"  # idle | loading | success | error
    loading_stage: str = ""
    loading_progress: float = 0.0
    loading_error: Optional[str] = None
    last_elapsed_ms: int = 0
    model_being_loaded: str = ""

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        with cls._lock:
            return {
                "status": cls.loading_status,
                "stage": cls.loading_stage,
                "progress": cls.loading_progress,
                "model": cls.model_being_loaded if cls.loading_status == "loading" else cls.current_model,
                "device": cls.current_device,
                "error": cls.loading_error,
                "elapsed_ms": cls.last_elapsed_ms,
            }

    @classmethod
    def update_progress(cls, stage: str, progress: float):
        with cls._lock:
            cls.loading_stage = stage
            cls.loading_progress = progress

    @classmethod
    def load_model(
        cls,
        model_name: str,
        device: Optional[str] = None,
        agent: Optional[KataiAgent] = None,
    ) -> Dict[str, Any]:
        model_name = model_name.strip()
        with cls._lock:
            cls.loading_status = "loading"
            cls.model_being_loaded = model_name
            cls.loading_stage = f"Initializing load for {model_name}..."
            cls.loading_progress = 0.05
            cls.loading_error = None

        t0 = time.perf_counter()
        try:
            def cb(stage: str, progress: float):
                cls.update_progress(stage, progress)

            if agent:
                res = agent.switch_model(model_name, device=device, progress_callback=cb)
                engine = agent.engine
            elif cls.engine:
                res = cls.engine.load_model(model_name, device=device, progress_callback=cb)
                engine = cls.engine
            else:
                engine = KataiEngine(checkpoint=model_name, device=device)
                cls.engine = engine
                res = {
                    "checkpoint": model_name,
                    "device": str(engine.device),
                    "model_name": getattr(engine.agent, "cfg", {}).get("model_name", "laya"),
                }

            cls.engine = engine
            elapsed_ms = round((time.perf_counter() - t0) * 1000)

            with cls._lock:
                cls.current_model = model_name
                cls.current_device = str(getattr(engine, "device", "cpu"))
                cls.loading_status = "success"
                cls.loading_stage = f"Loaded {model_name} successfully"
                cls.loading_progress = 1.0
                cls.last_elapsed_ms = elapsed_ms
                cls.model_being_loaded = ""

            res["elapsed_ms"] = elapsed_ms
            return res

        except Exception as e:
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            with cls._lock:
                cls.loading_status = "error"
                cls.loading_stage = f"Failed to load {model_name}"
                cls.loading_error = str(e)
                cls.last_elapsed_ms = elapsed_ms
                cls.model_being_loaded = ""
            raise


MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
}


class WebConsoleHandler(BaseHTTPRequestHandler):
    agent: Optional[KataiAgent] = None

    def log_message(self, format, *args):
        # Mute normal request logs to keep terminal output clean
        pass

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send(self, code: int, body: Any, content_type: str = "application/json"):
        if content_type == "application/json":
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        else:
            data = body.encode("utf-8") if isinstance(body, str) else body

        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def _serve_file(self, file_path: Path):
        try:
            # Ensure path does not escape STATIC_DIR
            resolved = file_path.resolve()
            if not str(resolved).startswith(str(STATIC_DIR.resolve())):
                self._send(403, {"error": "Forbidden"})
                return

            if not resolved.is_file():
                self._send(404, {"error": "Not Found"})
                return

            ext = resolved.suffix.lower()
            content_type = MIME_TYPES.get(ext, "application/octet-stream")
            data = resolved.read_bytes()

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if "/assets/" in self.path:
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            else:
                self.send_header("Cache-Control", "no-cache")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self._send(500, {"error": str(e)})

    def do_GET(self):
        path = self.path.split("?")[0]

        # API Endpoints
        if path == "/api/models":
            models = list_available_models()
            self._send(
                200,
                {
                    "current_model": ModelManager.current_model,
                    "current_device": ModelManager.current_device,
                    "models": models,
                    "loading_state": ModelManager.get_status(),
                },
            )
            return

        elif path == "/api/models/status":
            self._send(200, ModelManager.get_status())
            return

        elif path == "/api/agent/state":
            if self.agent:
                self._send(200, self.agent.snapshot())
            else:
                self._send(
                    200,
                    {
                        "status": "idle",
                        "step_count": 0,
                        "elapsed_ms": 0,
                        "history": [],
                        "elements": [],
                        "model": ModelManager.current_model,
                        "device": ModelManager.current_device,
                    },
                )
            return

        elif path == "/api/system/info":
            device = ModelManager.current_device
            model = ModelManager.current_model
            if self.agent and hasattr(self.agent, "engine") and self.agent.engine:
                device = getattr(self.agent.engine, "device", device)
                model = getattr(self.agent.engine, "checkpoint_name", getattr(self.agent.engine, "checkpoint", model))
            self._send(
                200,
                {
                    "model": model,
                    "device": str(device),
                    "port": DEFAULT_WEB_PORT,
                    "host": DEFAULT_WEB_HOST,
                    "python": sys.version,
                    "version": "1.1",
                },
            )
            return

        # Branding icon endpoint
        if path in ("/katai.png", "/branding/katai.png", "/favicon.ico"):
            if (STATIC_DIR / "katai.png").is_file():
                self._serve_file(STATIC_DIR / "katai.png")
                return
            for bc in [
                Path(__file__).resolve().parents[2] / "branding" / "katai.png",
                Path(__file__).resolve().parents[1] / "branding" / "katai.png",
            ]:
                if bc.is_file():
                    self._serve_file(bc)
                    return

        # Serve static assets
        if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
            if path in ("/", "/index.html", ""):
                self._serve_file(STATIC_DIR / "index.html")
            else:
                rel_path = path.lstrip("/")
                target_file = STATIC_DIR / rel_path
                if target_file.is_file():
                    self._serve_file(target_file)
                else:
                    # SPA Fallback
                    self._serve_file(STATIC_DIR / "index.html")
            return

        # Fallback if static assets not built
        self._send(
            200,
            "<html><body><h2>Katai Console UI is being prepared. Run 'npm run build' inside frontend/.</h2></body></html>",
            "text/html; charset=utf-8",
        )

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
        except Exception:
            body = {}

        path = self.path.split("?")[0]

        try:
            if path == "/api/models/load":
                model_name = body.get("model") or body.get("checkpoint")
                if not model_name:
                    self._send(400, {"error": "Missing 'model' or 'checkpoint' parameter in request body."})
                    return
                device = body.get("device")
                try:
                    res = ModelManager.load_model(model_name=model_name, device=device, agent=self.agent)
                    self._send(200, res)
                except Exception as e:
                    self._send(500, {"error": str(e), "model": model_name})
                return

            elif path == "/api/models/refresh":
                models = list_available_models()
                self._send(
                    200,
                    {
                        "refreshed": True,
                        "models": models,
                        "current_model": ModelManager.current_model,
                        "current_device": ModelManager.current_device,
                    },
                )
                return

            elif path == "/api/agent/start":
                url = body.get("url", "https://en.wikipedia.org/wiki/Main_Page")
                goal = body.get("goal", "Search Wikipedia for 'Python programming language'")
                checkpoint = body.get("checkpoint") or ModelManager.current_model
                headless = body.get("headless", False)

                if not self.agent:
                    if ModelManager.engine and (ModelManager.current_model == checkpoint or not checkpoint):
                        self.agent = KataiAgent(engine=ModelManager.engine, headless=headless)
                    else:
                        self.agent = KataiAgent(checkpoint=checkpoint, headless=headless)
                        ModelManager.engine = self.agent.engine
                        ModelManager.current_model = getattr(self.agent.engine, "checkpoint_name", checkpoint)
                        ModelManager.current_device = str(getattr(self.agent.engine, "device", "cpu"))
                else:
                    current_engine_model = getattr(self.agent.engine, "checkpoint_name", "")
                    if checkpoint and current_engine_model != checkpoint:
                        ModelManager.load_model(checkpoint, agent=self.agent)

                self.agent.start(url, goal)
                self._send(200, self.agent.snapshot())

            elif path == "/api/agent/step":
                if not self.agent:
                    self._send(400, {"error": "Agent not initialized. Please call /api/agent/start first."})
                    return
                state = self.agent.step()
                self._send(200, state)

            elif path == "/api/agent/predict":
                if not self.agent:
                    self._send(400, {"error": "Agent not initialized."})
                    return
                decision = self.agent.predict()
                snapshot = self.agent.snapshot()
                snapshot["decision"] = decision
                self._send(200, snapshot)

            elif path == "/api/agent/act":
                if not self.agent:
                    self._send(400, {"error": "Agent not initialized."})
                    return
                state = self.agent.act()
                self._send(200, state)

            elif path == "/api/agent/stop":
                if self.agent:
                    self.agent.state["status"] = "ready"
                    self._send(200, self.agent.snapshot())
                else:
                    self._send(200, {"status": "idle"})

            elif path == "/api/agent/reset":
                if self.agent:
                    try:
                        self.agent.close()
                    except Exception:
                        pass
                    self.agent = None
                self._send(
                    200,
                    {
                        "status": "idle",
                        "step_count": 0,
                        "elapsed_ms": 0,
                        "history": [],
                        "elements": [],
                        "model": ModelManager.current_model,
                        "device": ModelManager.current_device,
                    },
                )

            else:
                self._send(404, {"error": f"Endpoint not found: {path}"})

        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": str(e), "traceback": traceback.format_exc()})


# looks
def get_ascii_banner() -> str:
    """Load ASCII banner art from ascii.txt."""
    return "\n"


def start_console(
    port: int = DEFAULT_WEB_PORT,
    host: str = DEFAULT_WEB_HOST,
    agent: Optional[KataiAgent] = None,
):
    """Start the interactive web console."""
    banner = get_ascii_banner()
    if banner:
        print(banner, file=sys.stderr)
    WebConsoleHandler.agent = agent
    if agent and hasattr(agent, "engine") and agent.engine:
        ModelManager.engine = agent.engine
        ModelManager.current_model = getattr(agent.engine, "checkpoint_name", "v10s")
        ModelManager.current_device = str(getattr(agent.engine, "device", "cpu"))
    server = ThreadingHTTPServer((host, port), WebConsoleHandler)
    print("=" * 65, file=sys.stderr)
    print("  Katai Web Console", file=sys.stderr)
    print(f"  URL:     http://{host}:{port}", file=sys.stderr)
    print("=" * 65, file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Web Console...", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    start_console()
