import pytest
from katai.core.browser import action_space, find_chrome_binary


def test_action_space_grouping():
    raw_actions = [
        {"id": "e1", "node": 101, "kind": "click", "label": "Search"},
        {"id": "e2", "node": 102, "kind": "fill", "label": "Query Input", "value": ""},
        {"id": "e3", "node": 103, "kind": "select", "label": "Country", "value": "US"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down"},
        {"id": "wait", "kind": "wait", "label": "Wait"},
    ]

    elements, targets, controls = action_space(raw_actions)

    # Check elements
    assert len(elements) == 3
    assert elements[0]["label"] == "Search"
    assert elements[1]["label"] == "Query Input"

    # Check targets
    assert "CLICK" in targets
    assert "TYPE_TEXT" in targets
    assert "SELECT" in targets
    assert "1" in targets["CLICK"]
    assert "2" in targets["TYPE_TEXT"]

    # Check controls
    assert "SCROLL_DOWN" in controls
    assert "WAIT" in controls


def test_chrome_binary_discovery():
    binary = find_chrome_binary()
    assert binary is not None
    assert "Chrome" in binary or "chromium" in binary.lower()
