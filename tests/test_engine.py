import json
import os
import pytest
from katai.core.engine import KataiEngine, compact_element


def test_compact_element_v3():
    cand = {
        "element": "[1] Submit Query Button With A Very Long Name That Exceeds Head Budget",
        "role": "button",
        "current_value": "Search Now",
        "checked": "false",
    }
    compacted = compact_element(cand, fmt="v3")
    assert "[1] Submit Query" in compacted
    assert "(button)" in compacted
    assert "checked=false" in compacted


def test_engine_inference():
    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_request.json")
    if not os.path.exists(sample_path):
        pytest.skip("sample_request.json not found")

    req = json.load(open(sample_path))
    engine = KataiEngine(checkpoint="v10s")

    result = engine.predict(req["state"], req["questions"])
    assert "answers" in result
    assert "operation" in result["answers"]

    op = result["answers"]["operation"]["choice"]
    conf = result["answers"]["operation"]["confidence"]

    assert op == "TYPE_TEXT"
    assert conf > 0.8
