"""KataiAgent: High-Level Autonomous Web Agent.

Coordinates:
- Browser observation (snapshot.js + CDP)
- Laya Decision Engine (in-process or remote SystemOne)
- Text Helper (heuristic extraction or LLM completion)
- Execution guardrails (stale-page detection, loop prevention, max-step bounds)
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .browser import Browser, StalePage, action_space
from .engine import LayaEngine
from .text_helper import get_field_text

DEFAULT_MAX_STEPS = int(os.environ.get("MAX_STEPS", "60"))


class KataiAgent:
    """End-to-end browser agent driven by Katai / Laya non-autoregressive decision models."""

    def __init__(
        self,
        checkpoint: str = "v10s",
        device: Optional[str] = None,
        cdp_url: Optional[str] = None,
        headless: bool = False,
        engine: Optional[LayaEngine] = None,
        screenshots: bool = True,
        record_dir: Optional[str] = None,
    ):
        self.engine = engine or LayaEngine(checkpoint=checkpoint, device=device)
        self.cdp_url = cdp_url or os.environ.get("BU_CDP_URL", "http://127.0.0.1:9222")
        self.headless = headless
        self.screenshots = screenshots
        self.record_dir = Path(record_dir) if record_dir else None
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)

        self.browser: Optional[Browser] = None
        self.state: Dict[str, Any] = {
            "status": "idle",
            "url": "",
            "goal": "",
            "page": {},
            "decision": None,
            "history": [],
            "decisions": [],
            "step_count": 0,
            "elapsed_ms": 0,
            "started_at": None,
        }

    def start(self, url: str, goal: str):
        """Initialize browser and observe initial page."""
        self.state["url"] = url
        self.state["goal"] = goal.strip()
        self.state["history"] = []
        self.state["decisions"] = []
        self.state["status"] = "ready"
        self.state["started_at"] = time.perf_counter()

        if not self.browser:
            self.browser = Browser(initial_url=url, cdp_url=self.cdp_url, headless=self.headless)
        else:
            self.browser.navigate(url)

        self.state["page"] = self.browser.observe(screenshot=self.screenshots)
        if self.record_dir and self.state["page"].get("screenshot"):
            (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(self.state["page"]["screenshot"]))

    def snapshot(self) -> Dict[str, Any]:
        """Export current agent state without raw browser handle."""
        actions = self.state["page"].get("actions", [])
        elements, targets, controls = action_space(actions)
        return {
            "status": self.state["status"],
            "url": self.state["url"],
            "goal": self.state["goal"],
            "page_title": self.state["page"].get("title", ""),
            "page_url": self.state["page"].get("url", ""),
            "decision": self.state["decision"],
            "history": self.state["history"],
            "elements": elements,
            "step_count": len(self.state["history"]),
            "elapsed_ms": self.state["elapsed_ms"],
            "screenshot": self.state["page"].get("screenshot", ""),
            "model": getattr(self.engine, "checkpoint", "v10s"),
            "device": getattr(self.engine, "device", "mps"),
        }

    def predict(self) -> Dict[str, Any]:
        """Predict the next action based on current DOM observation."""
        if not self.browser:
            raise RuntimeError("Agent not started. Call start(url, goal) first.")
        if self.state["status"] in ("done", "blocked"):
            return self.snapshot()

        page = self.state["page"]
        elements, targets, controls = action_space(page.get("actions", []))

        decision = self.engine.choose_action(
            page=page,
            goal=self.state["goal"],
            history=self.state["history"],
            targets=targets,
            controls=controls,
        )

        self.state["decision"] = decision
        self.state["status"] = "predicted"
        return decision

    def act(self) -> Dict[str, Any]:
        """Execute the predicted action in the browser."""
        decision = self.state.get("decision")
        page = self.state.get("page")
        if not decision or not page:
            raise RuntimeError("No pending decision. Call predict() before act().")

        self.state["decision"] = None
        selected = decision["choice"]

        if selected in ("DONE", "BLOCKED"):
            self.state["status"] = "done" if selected == "DONE" else "blocked"
            self.state["elapsed_ms"] = round((time.perf_counter() - self.state["started_at"]) * 1000)
            return self.snapshot()

        # Find matching action in current page
        matching_action = next((a for a in page.get("actions", []) if a["id"] == selected), None)
        if not matching_action:
            # Action not found (DOM might have updated)
            self.state["page"] = self.browser.observe(screenshot=self.screenshots)
            self.state["status"] = "ready"
            return self.snapshot()

        text_to_type = None
        text_meta = None
        if matching_action["kind"] == "fill":
            field_context = {
                "goal": self.state["goal"],
                "field": {k: matching_action.get(k) for k in ("label", "role", "value")},
                "page": {"title": page.get("title", ""), "text": page.get("text", "")[:4000]},
                "recent_actions": [
                    {k: h.get(k) for k in ("action", "text")} for h in self.state["history"][-6:]
                ],
            }
            text_to_type, text_meta = get_field_text(field_context)

        # Execute action in browser
        self.browser.act(matching_action, page, text=text_to_type)
        self.state["elapsed_ms"] = round((time.perf_counter() - self.state["started_at"]) * 1000)

        # Log action to history
        step_idx = len(self.state["history"]) + 1
        history_item = {
            "step": step_idx,
            "action": matching_action.get("label", ""),
            "kind": matching_action.get("kind", ""),
            "choice": selected,
            "operation": decision.get("operation"),
            "target": decision.get("target"),
            "confidence": decision.get("confidence", 0.0),
            "latency_ms": decision.get("latency_ms", 0),
            "text": text_to_type,
            "text_meta": text_meta,
            "url": page.get("url", ""),
            "elapsed_ms": self.state["elapsed_ms"],
        }
        self.state["history"].append(history_item)

        # Observe new page state
        new_page = self.browser.observe(screenshot=self.screenshots)
        page_changed = (
            new_page.get("url") != page.get("url")
            or new_page.get("fingerprint") != page.get("fingerprint")
        )
        history_item["page_changed"] = page_changed
        self.state["page"] = new_page

        if self.record_dir and new_page.get("screenshot"):
            (self.record_dir / f"{step_idx:06d}.jpg").write_bytes(base64.b64decode(new_page["screenshot"]))

        # Detect infinite stagnation loops (3 consecutive unmutating actions)
        repeated = self.state["history"][-3:]
        if (
            len(repeated) == 3
            and all(h.get("page_changed") is False and h.get("kind") != "wait" for h in repeated)
        ):
            self.state["status"] = "blocked"
        else:
            self.state["status"] = "ready"

        return self.snapshot()

    def step(self) -> Dict[str, Any]:
        """Perform a single complete predict-and-act cycle."""
        try:
            self.predict()
            return self.act()
        except StalePage:
            # Recover from stale page
            self.state["decision"] = None
            self.state["status"] = "ready"
            self.state["page"] = self.browser.observe(screenshot=self.screenshots)
            return self.snapshot()

    def run(
        self,
        url: Optional[str] = None,
        goal: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """Run agent autonomously, yielding state after each step."""
        if url and goal:
            self.start(url, goal)

        while self.state["status"] not in ("done", "blocked"):
            if len(self.state["history"]) >= max_steps:
                self.state["status"] = "blocked"
                break
            yield self.step()

        return self.snapshot()

    def close(self):
        """Clean up browser connection."""
        if self.browser:
            self.browser.close()
            self.browser = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
