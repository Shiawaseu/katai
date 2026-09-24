"""Direct Katai Decision Engine usage on arbitrary DOM states."""

import json
from katai import KataiEngine

def main():
    engine = KataiEngine(checkpoint="v10s")

    state = {
        "page": {
            "title": "Hacker News",
            "url": "https://news.ycombinator.com/",
            "text": "1. Show HN: Katai Non-autoregressive Browser Agent\n2. Ask HN: Favorite developer tools?"
        },
        "recent_actions": []
    }

    questions = {
        "operation": {
            "type": "choice",
            "criteria": {
                "CLICK": "Click a link or button",
                "SCROLL_DOWN": "Scroll down for more items",
                "DONE": "Goal satisfied"
            },
            "instructions": {"goal": "Open the newest submissions page"}
        },
        "click_target": {
            "type": "choice",
            "criteria": {
                "1": "[1] Hacker News (link)",
                "2": "[2] new (link)",
                "3": "[3] past (link)",
                "4": "[4] comments (link)",
                "5": "[5] ask (link)",
                "6": "[6] show (link)"
            },
            "instructions": {"goal": "Open the newest submissions page"}
        }
    }

    result = engine.predict(state, questions)
    answers = result["answers"]

    print("Operation decision:", answers["operation"]["choice"], f"(confidence: {answers['operation']['confidence']:.2f})")
    print("Click target:      ", answers["click_target"]["choice"], f"(confidence: {answers['click_target']['confidence']:.2f})")
    print("Latency:           ", result["latency_ms"], "ms")

if __name__ == "__main__":
    main()
