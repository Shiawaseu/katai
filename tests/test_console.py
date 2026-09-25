import json
import threading
import time
import urllib.request
import urllib.error
import pytest

from katai.web.console import start_console, WebConsoleHandler


@pytest.fixture(scope="module")
def console_server():
    port = 8789
    host = "127.0.0.1"
    server_thread = threading.Thread(
        target=start_console,
        kwargs={"port": port, "host": host, "agent": None},
        daemon=True,
    )
    server_thread.start()
    time.sleep(1.0)
    yield f"http://{host}:{port}"


def test_console_root_serves_html(console_server):
    with urllib.request.urlopen(f"{console_server}/", timeout=5.0) as resp:
        assert resp.status == 200
        content_type = resp.headers.get("Content-Type", "")
        assert "text/html" in content_type
        body = resp.read().decode("utf-8")
        assert "Katai" in body
        assert '<div id="root">' in body
        assert "/assets/" in body

        # Verify JS asset serves properly with correct content type
        import re
        js_matches = re.findall(r'src="(/assets/[^"]+\.js)"', body)
        assert len(js_matches) > 0
        js_url = f"{console_server}{js_matches[0]}"
        with urllib.request.urlopen(js_url, timeout=5.0) as js_resp:
            assert js_resp.status == 200
            assert "javascript" in js_resp.headers.get("Content-Type", "")
            assert len(js_resp.read()) > 1000

        # Verify CSS asset serves properly with correct content type
        css_matches = re.findall(r'href="(/assets/[^"]+\.css)"', body)
        assert len(css_matches) > 0
        css_url = f"{console_server}{css_matches[0]}"
        with urllib.request.urlopen(css_url, timeout=5.0) as css_resp:
            assert css_resp.status == 200
            assert "text/css" in css_resp.headers.get("Content-Type", "")
            assert len(css_resp.read()) > 1000


def test_console_branding_icon(console_server):
    for endpoint in ("/katai.png", "/branding/katai.png"):
        with urllib.request.urlopen(f"{console_server}{endpoint}", timeout=5.0) as resp:
            assert resp.status == 200
            assert "image/png" in resp.headers.get("Content-Type", "")
            data = resp.read()
            assert len(data) > 10000
            # Verify PNG header magic bytes
            assert data.startswith(b"\x89PNG\r\n\x1a\n")


def test_console_api_state(console_server):
    with urllib.request.urlopen(f"{console_server}/api/agent/state", timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "status" in data
        assert "step_count" in data
        assert "elements" in data
        assert "history" in data


def test_console_api_system_info(console_server):
    with urllib.request.urlopen(f"{console_server}/api/system/info", timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "model" in data
        assert "device" in data
        assert "port" in data


def test_console_api_stop_and_reset(console_server):
    req_stop = urllib.request.Request(
        f"{console_server}/api/agent/stop",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req_stop, timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "status" in data

    req_reset = urllib.request.Request(
        f"{console_server}/api/agent/reset",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req_reset, timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "idle"
        assert data["step_count"] == 0


def test_console_api_models(console_server):
    with urllib.request.urlopen(f"{console_server}/api/models", timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "current_model" in data
        assert "models" in data
        assert isinstance(data["models"], list)
        model_ids = [m["id"] for m in data["models"]]
        assert "ichenney/laya-browser-v32b" in model_ids
        assert "abedinia/laya-web-agent" in model_ids
        assert "v10s" in model_ids


def test_console_api_models_status(console_server):
    with urllib.request.urlopen(f"{console_server}/api/models/status", timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "status" in data
        assert "progress" in data
        assert "model" in data


def test_console_api_models_refresh(console_server):
    req = urllib.request.Request(
        f"{console_server}/api/models/refresh",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data.get("refreshed") is True
        assert len(data.get("models", [])) >= 7


def test_console_api_models_load_and_hotswap(console_server):
    # Hot-swap to ichenney/laya-browser-v32b (already cached)
    req = urllib.request.Request(
        f"{console_server}/api/models/load",
        data=json.dumps({"model": "https://huggingface.co/ichenney/laya-browser-v32b"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data.get("checkpoint") == "https://huggingface.co/ichenney/laya-browser-v32b"
        assert "device" in data

    # Verify status reflects new model
    with urllib.request.urlopen(f"{console_server}/api/models/status", timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data.get("status") == "success"
        assert data.get("model") == "https://huggingface.co/ichenney/laya-browser-v32b"

