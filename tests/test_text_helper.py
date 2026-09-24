import pytest
from katai.core.text_helper import extract_heuristic_text, get_field_text


def test_quoted_text_extraction():
    ctx = {
        "goal": "Search Wikipedia for 'Python programming language' and open the article.",
        "field": {"label": "Search Wikipedia", "role": "searchbox"},
    }
    extracted = extract_heuristic_text(ctx)
    assert extracted == "Python programming language"


def test_double_quoted_text():
    ctx = {
        "goal": 'Find "Albert Einstein" biography',
        "field": {"label": "Search", "role": "searchbox"},
    }
    extracted = extract_heuristic_text(ctx)
    assert extracted == "Albert Einstein"


def test_unquoted_search_phrase():
    ctx = {
        "goal": "Search for deep learning papers and click search",
        "field": {"label": "Query", "role": "textbox"},
    }
    extracted = extract_heuristic_text(ctx)
    assert "deep learning papers" in extracted


def test_email_extraction():
    ctx = {
        "goal": "Enter user@example.com into the subscriber field",
        "field": {"label": "Email Address", "role": "textbox"},
    }
    extracted = extract_heuristic_text(ctx)
    assert extracted == "user@example.com"


def test_get_field_text_fallback():
    ctx = {
        "goal": "Type 'hello world'",
        "field": {"label": "Greeting", "role": "textbox"},
    }
    val, meta = get_field_text(ctx)
    assert val == "hello world"
    assert meta["source"] == "heuristic"
