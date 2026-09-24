import json
import threading
import time
import urllib.request
import pytest

from katai.server import start_server, SystemOneHandler
from katai.core.engine import KataiEngine


@pytest.fixture(scope="module")
def systemone_server():
    port = 8798
    engine = KataiEngine(checkpoint="v10s")
    t = threading.Thread(
        target=start_server,
        kwargs={"port": port, "host": "127.0.0.1", "engine": engine},
        daemon=True,
    )
    t.start()
    time.sleep(1.0)
    yield f"http://127.0.0.1:{port}"


def test_server_health(systemone_server):
    with urllib.request.urlopen(f"{systemone_server}/health", timeout=5.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert data["status"] == "online"
        assert "laya-v10s" in data["model"]


def test_server_systemone_post(systemone_server):
    import os
    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_request.json")
    req_body = json.load(open(sample_path))

    req = urllib.request.Request(
        f"{systemone_server}/v1/systemone",
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "answers" in data
        assert data["answers"]["operation"]["choice"] == "TYPE_TEXT"
