"""Browser Controller & CDP Session Management.

Provides high-speed browser automation through Chrome DevTools Protocol (CDP).
Supports:
- Direct CDP connection via WebSocket with automatic tab management.
- Integration with Browser-Harness daemon when available.
- Headless and headful Chrome launch automation on macOS / Linux.
- Reliable DOM inspection via snapshot.js and robust action execution.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

import websockets.sync.client as ws_client

SNAPSHOT_JS_PATH = Path(__file__).with_name("snapshot.js")
DEFAULT_CDP_URL = os.environ.get("BU_CDP_URL", "http://127.0.0.1:9222")


class StalePage(RuntimeError):
    """Raised when the observed DOM state changed before the decision could act."""


def action_space(actions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Group flat DOM actions into element indices, target groups, and navigation controls."""
    elements: List[Dict[str, Any]] = []
    indices: Dict[int, str] = {}
    targets: Dict[str, Dict[str, Any]] = {}
    controls: Dict[str, Any] = {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}

    for action in actions:
        kind = action.get("kind")
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue

        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if "rect" in action:
                element["rect"] = action["rect"]
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)

        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)

        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action

    return elements, targets, controls


def find_chrome_binary() -> Optional[str]:
    """Find installed Chrome / Chromium executable."""
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def launch_chrome(port: int = 9222, headless: bool = True, user_data_dir: Optional[str] = None) -> subprocess.Popen:
    """Launch a Chrome process with remote debugging port."""
    binary = find_chrome_binary()
    if not binary:
        raise FileNotFoundError(
            "Could not find Google Chrome binary. Please install Chrome or set BU_CDP_URL to a running browser."
        )

    user_data = user_data_dir or f"/tmp/laya_chrome_{port}"
    args = [
        binary,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-translate",
        "--disable-extensions",
        "about:blank",
    ]
    if headless:
        args.insert(1, "--headless=new")

    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Wait for endpoint to become reachable
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=0.5) as r:
                if r.status == 200:
                    return p
        except Exception:
            time.sleep(0.1)

    raise TimeoutError(f"Chrome started with PID {p.pid} but port {port} was not reachable within 10s.")


class Browser:
    """Direct CDP connection to a browser tab."""

    def __init__(self, initial_url: str = "about:blank", cdp_url: str = DEFAULT_CDP_URL, headless: bool = False):
        self.cdp_url = cdp_url.rstrip("/")
        self.ws: Optional[ws_client.ClientConnection] = None
        self.target_id: Optional[str] = None
        self.msg_id = 0
        self._launched_process: Optional[subprocess.Popen] = None
        self._snapshot_script = SNAPSHOT_JS_PATH.read_text(encoding="utf-8")

        self._connect_or_launch(headless=headless)
        self.navigate(initial_url)

    def _connect_or_launch(self, headless: bool = False):
        """Connect to CDP; launch Chrome automatically if port is down."""
        try:
            with urllib.request.urlopen(f"{self.cdp_url}/json/version", timeout=1.0) as r:
                pass
        except Exception:
            # Parse port from cdp_url
            try:
                port = int(self.cdp_url.split(":")[-1].split("/")[0])
            except Exception:
                port = 9222
            print(f"[Browser] No active Chrome at {self.cdp_url}. Launching local Chrome on port {port}...", file=sys.stderr)
            self._launched_process = launch_chrome(port=port, headless=headless)

        # Create or pick a target tab
        new_tab_url = f"{self.cdp_url}/json/new?about:blank"
        req = urllib.request.Request(new_tab_url, data=b"", method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=5.0) as r:
                tab_info = json.loads(r.read().decode("utf-8"))
        except Exception:
            # Fallback to GET /json/list
            with urllib.request.urlopen(f"{self.cdp_url}/json/list", timeout=5.0) as r:
                tabs = json.loads(r.read().decode("utf-8"))
                tab_info = tabs[0] if tabs else None

        if not tab_info or "webSocketDebuggerUrl" not in tab_info:
            raise RuntimeError(f"Could not open or attach to a Chrome tab at {self.cdp_url}")

        ws_url = tab_info["webSocketDebuggerUrl"]
        self.target_id = tab_info.get("id")
        self.ws = ws_client.connect(ws_url, max_size=32 * 1024 * 1024)

        # Configure tab viewport and settings
        self.call("Page.enable")
        self.call("DOM.enable")
        self.call("Runtime.enable")
        self.call("Emulation.setDeviceMetricsOverride", width=1280, height=800, deviceScaleFactor=1, mobile=False)
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)

    def call(self, method: str, **params) -> Dict[str, Any]:
        """Send a synchronous CDP command and wait for response."""
        if not self.ws:
            raise RuntimeError("Browser CDP WebSocket not connected.")
        self.msg_id += 1
        curr_id = self.msg_id
        msg = {"id": curr_id, "method": method, "params": params}
        self.ws.send(json.dumps(msg))

        while True:
            raw = self.ws.recv()
            data = json.loads(raw)
            if data.get("id") == curr_id:
                if "error" in data:
                    raise RuntimeError(f"CDP error calling {method}: {data['error']}")
                return data.get("result", {})

    def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript in page context and return serialized value."""
        res = self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if res.get("exceptionDetails"):
            raise StalePage(f"Script evaluation exception: {res['exceptionDetails']}")
        return res.get("result", {}).get("value")

    def navigate(self, url: str, wait_complete: bool = True, timeout: float = 20.0):
        """Navigate to URL and wait until readyState is complete."""
        self.call("Page.navigate", url=url)
        if not wait_complete:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready = self.evaluate("document.readyState")
            if ready in ("complete", "interactive"):
                break
            time.sleep(0.05)

    def observe(self, screenshot: bool = True) -> Dict[str, Any]:
        """Capture DOM snapshot, interactive actions, and viewport screenshot."""
        data = self.evaluate(self._snapshot_script)
        if not data or not isinstance(data, dict):
            # Page might be loading or in transition
            time.sleep(0.1)
            data = self.evaluate(self._snapshot_script)
            if not data or not isinstance(data, dict):
                data = {
                    "url": self.evaluate("location.href") or "",
                    "title": self.evaluate("document.title") or "",
                    "w": 1280,
                    "h": 800,
                    "text": "",
                    "actions": [],
                    "fingerprint": "blank",
                }

        if screenshot:
            shot_res = self.call("Page.captureScreenshot", format="jpeg", quality=65)
            data["screenshot"] = shot_res.get("data", "")
        else:
            data["screenshot"] = ""

        return data

    def fresh(self, observed_page: Dict[str, Any]) -> bool:
        """Check whether the DOM is still fresh relative to the observation."""
        current_url = self.evaluate("location.href")
        return current_url == observed_page.get("url")

    def act(self, action: Dict[str, Any], page: Dict[str, Any], text: Optional[str] = None):
        """Execute selected action against the browser."""
        kind = action.get("kind")
        node_id = action.get("node")

        if kind == "scroll":
            delta = action.get("delta", 500)
            self.evaluate(f"window.scrollBy({{top: {delta}, behavior: 'instant'}});")
            time.sleep(0.15)
            return

        if kind == "wait":
            time.sleep(0.6)
            return

        # Interactive actions require element node
        rect = action.get("rect", {})
        x = rect.get("x", 0) + rect.get("w", 0) / 2
        y = rect.get("y", 0) + rect.get("h", 0) / 2

        if kind == "click":
            # Direct DOM dispatch or mouse event
            click_expr = f"""(() => {{
                const e = window.__layaFast?.nodes.get({node_id});
                if (e) {{
                    e.scrollIntoViewIfNeeded ? e.scrollIntoViewIfNeeded() : e.scrollIntoView({{block: 'center'}});
                    e.click();
                    return true;
                }}
                return false;
            }})()"""
            clicked = self.evaluate(click_expr)
            if not clicked:
                # Fallback to mouse click event
                self.call("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
                self.call("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)
            time.sleep(0.2)

        elif kind == "fill":
            fill_expr = f"""(() => {{
                const e = window.__layaFast?.nodes.get({node_id});
                if (e) {{
                    e.focus();
                    if ('value' in e) e.value = '';
                    else e.innerText = '';
                    e.dispatchEvent(new Event('input', {{bubbles: true}}));
                    return true;
                }}
                return false;
            }})()"""
            self.evaluate(fill_expr)
            if text:
                self.call("Input.insertText", text=text)
                self.evaluate(f"""(() => {{
                    const e = window.__layaFast?.nodes.get({node_id});
                    if (e) {{
                        e.dispatchEvent(new Event('change', {{bubbles: true}}));
                    }}
                }})()""")
            time.sleep(0.15)

        elif kind == "select":
            val = json.dumps(action.get("value", ""))
            select_expr = f"""(() => {{
                const e = window.__layaFast?.nodes.get({node_id});
                if (e) {{
                    e.value = {val};
                    e.dispatchEvent(new Event('change', {{bubbles: true}}));
                    return true;
                }}
                return false;
            }})()"""
            self.evaluate(select_expr)
            time.sleep(0.15)

    def close(self):
        """Close CDP WebSocket and tab."""
        if self.ws:
            try:
                if self.target_id:
                    # Close created tab
                    close_url = f"{self.cdp_url}/json/close/{self.target_id}"
                    urllib.request.urlopen(close_url, timeout=1.0)
            except Exception:
                pass
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

        if self._launched_process:
            try:
                self._launched_process.terminate()
            except Exception:
                pass
            self._launched_process = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
