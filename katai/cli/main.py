"""Command Line Interface for Katai Browser-Use Agent.

Usage:
  katai run <url> <goal>    Run an autonomous browser session
  katai server              Start TypeSafe-compatible SystemOne API
  katai console             Start the interactive web dashboard
  katai verify              Run model diagnostic verification
  katai chrome              Launch local Chrome with remote debugging
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from katai.core.agent import KataiAgent
from katai.core.browser import launch_chrome
from katai.core.engine import KataiEngine
from katai.server.systemone import start_server
from katai.web.console import start_console


def get_ascii_banner() -> str:
    """Load ASCII banner art from ascii.txt."""
    candidates = [
        Path(__file__).resolve().parent / "ascii.txt",
        Path(__file__).resolve().parents[2] / "branding" / "ascii.txt",
        Path(__file__).resolve().parents[1] / "branding" / "ascii.txt",
    ]
    for p in candidates:
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8").rstrip()
            except Exception:
                pass
    return ""


def render_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Render a clean unicode progress bar with percentage."""
    if total <= 0:
        return "░" * width + "   0%"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    bar = "█" * filled + "░" * (width - filled)
    pct = int(round(ratio * 100))
    return f"[{bar}] {pct:3d}%"


def render_confidence_bar(conf: float, width: int = 10) -> str:
    """Render a calibrated confidence progress bar with percentage."""
    ratio = min(max(conf, 0.0), 1.0)
    filled = int(round(ratio * width))
    bar = "█" * filled + "░" * (width - filled)
    pct = int(round(ratio * 100))
    return f"[{bar}] {pct:2d}%"


def print_cli_banner():
    """Print the ASCII banner on initialization."""
    banner = False #get_ascii_banner()
    if banner:
        print(banner)
        print()


def cmd_run(args):
    """Run autonomous browser agent from terminal with modern progress tracking."""
    print("=" * 65)
    print("  KATAI AUTONOMOUS BROWSER AGENT")
    print("=" * 65)
    print(f"  Target URL:   {args.url}")
    print(f"  Goal:         {args.goal}")
    print(f"  Model:        {args.checkpoint}")
    print(f"  Headless:     {args.headless}")
    print(f"  Max Steps:    {args.max_steps}")
    print("=" * 65)

    print(f"\n[1/3] Initializing Decision Engine ({args.checkpoint})...")
    engine = KataiEngine(checkpoint=args.checkpoint, device=args.device)

    print("[2/3] Connecting to Chrome via DevTools Protocol (CDP)...")
    agent = KataiAgent(
        checkpoint=args.checkpoint,
        device=args.device,
        headless=args.headless,
        engine=engine,
    )
    print(f"      CDP Connected ({agent.engine.device.upper()}) [OK]")

    print("[3/3] Observing initial page viewport & DOM action space...", end="", flush=True)
    agent.start(args.url, args.goal)
    actions_count = len(agent.state["page"].get("actions", []))
    page_title = agent.state["page"].get("title", "")
    print(f" [{actions_count} controls detected]")
    if page_title:
        print(f"      Initial Title: {page_title}")
    print("-" * 65 + "\n")

    step = 0
    try:
        while agent.state["status"] not in ("done", "blocked") and step < args.max_steps:
            step += 1
            progress_str = render_progress_bar(step, args.max_steps, width=20)
            print(f"Step {step:02d}/{args.max_steps} {progress_str}")

            # Predict phase
            t0 = time.perf_counter()
            agent.predict()
            d = agent.state.get("decision", {})
            op = d.get("operation", "UNKNOWN")
            conf = d.get("confidence", 0.0)
            lat = d.get("latency_ms", 0)
            target = d.get("target") or d.get("choice") or "None"
            conf_bar = render_confidence_bar(conf, width=10)

            # Act phase
            agent.act()
            h = agent.state["history"][-1] if agent.state["history"] else {}
            action_label = h.get("action", "")
            text_typed = h.get("text")

            status = agent.state["status"].upper()

            # Render tree layout
            print(f"  ├─ Operation:  {op:10s} (conf: {conf_bar}) · {lat:3d}ms")
            print(f"  ├─ Target:     [{target}] {action_label[:50]}")
            if text_typed:
                print(f"  ├─ Typed text: \"{text_typed}\"")
            page_changed = h.get("page_changed", False)
            change_status = "DOM Mutated" if page_changed else "Unchanged"
            print(f"  └─ Status:     [{status}] · {change_status}")
            print()

        final_status = agent.state["status"].upper()
        elapsed_sec = agent.state["elapsed_ms"] / 1000.0
        print("=" * 65)
        print(f"  SESSION RESULT: {final_status}")
        print(f"  Completed in {step} steps ({agent.state['elapsed_ms']} ms / {elapsed_sec:.2f}s total)")
        print(f"  Final URL:   {agent.state['page'].get('url')}")
        print(f"  Final Title: {agent.state['page'].get('title')}")
        print("=" * 65 + "\n")

    finally:
        agent.close()


def cmd_server(args):
    """Start SystemOne server."""
    print("=" * 65)
    print("  KATAI SYSTEMONE API SERVER")
    print(f"  Listening on: http://{args.host}:{args.port}/v1/systemone")
    print(f"  Checkpoint:   {args.checkpoint}")
    print("=" * 65 + "\n")
    start_server(port=args.port, host=args.host, checkpoint=args.checkpoint, device=args.device)


def cmd_console(args):
    """Start interactive Web Console."""
    agent = KataiAgent(checkpoint=args.checkpoint, device=args.device, headless=args.headless)
    start_console(port=args.port, host=args.host, agent=agent)


def cmd_verify(args):
    """Run model diagnostic verification with visual progress tracking."""
    print("=" * 65)
    print("  KATAI DECISION ENGINE DIAGNOSTICS")
    print("=" * 65)

    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "sample_request.json",
    )
    if not os.path.exists(sample_path):
        print(f"Error: {sample_path} not found.")
        sys.exit(1)

    req = json.load(open(sample_path))
    engine = KataiEngine(checkpoint=args.checkpoint, device=args.device)

    # Warmup with progress indicator
    print("  [1/2] Warming up decision model:    ", end="", flush=True)
    t_warm = time.perf_counter()
    engine.predict(req["state"], req["questions"])
    warm_ms = (time.perf_counter() - t_warm) * 1000
    print(f"{render_progress_bar(1, 1, width=16)} ({warm_ms:.1f} ms)")

    # Inference evaluation with progress indicator
    print("  [2/2] Evaluating Mind2Web sample:   ", end="", flush=True)
    t0 = time.perf_counter()
    res = engine.predict(req["state"], req["questions"])
    ms = (time.perf_counter() - t0) * 1000
    print(f"{render_progress_bar(1, 1, width=16)} ({ms:.1f} ms)")

    answers = res["answers"]
    op_ans = answers.get("operation", {})
    op = op_ans.get("choice")
    conf = op_ans.get("confidence", 0.0)
    conf_bar = render_confidence_bar(conf, width=10)

    print("-" * 65)
    print(f"  Model:       {res['model']}")
    print(f"  Device:      {engine.device.upper()}")
    print(f"  Latency:     {ms:.1f} ms (single forward pass)")
    print(f"  Tokens:      {res['usage']['input_tokens']:,}")
    print(f"  Operation:   {op} (confidence: {conf_bar})")

    tq_name = f"{op.lower()}_target"
    if tq_name in answers:
        target_choice = answers[tq_name].get("choice")
        target_crit = req["questions"][tq_name]["criteria"].get(target_choice, "")
        print(f"  Target:      [{target_choice}] {target_crit[:55]}")

    expected = req.get("expected", {}).get("operation")
    if expected:
        match = "MATCH [OK]" if op == expected else "MISMATCH [WARN]"
        print(f"  Expected:    {expected} -> {match}")
    print("=" * 65 + "\n")


def cmd_chrome(args):
    """Launch local Chrome with remote debugging port."""
    print("=" * 65)
    print("  CHROME DEVTOOLS PROTOCOL HELPER")
    print(f"  Port:     {args.port}")
    print(f"  Headless: {args.headless}")
    print("=" * 65)
    print(f"Launching Chrome with remote debugging on port {args.port}...")
    p = launch_chrome(port=args.port, headless=args.headless)
    print(f"Chrome successfully launched with PID {p.pid} at http://127.0.0.1:{args.port}\n")
    try:
        p.wait()
    except KeyboardInterrupt:
        p.terminate()


def main():
    # Always print ASCII art banner on CLI initialization
    print_cli_banner()

    parser = argparse.ArgumentParser(
        description="Katai: Ultra-fast non-autoregressive browser automation agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    p_run = subparsers.add_parser("run", help="Execute an autonomous web browsing goal")
    p_run.add_argument("url", help="Target start URL")
    p_run.add_argument("goal", help="Task goal instructions")
    p_run.add_argument("--checkpoint", default="v10s", help="Model checkpoint (default: v10s)")
    p_run.add_argument("--device", default=None, help="Device (mps, cuda, cpu)")
    p_run.add_argument("--headless", action="store_true", help="Run browser headless")
    p_run.add_argument("--max-steps", type=int, default=60, help="Maximum step budget")
    p_run.set_defaults(func=cmd_run)

    # server command
    p_srv = subparsers.add_parser("server", help="Start SystemOne API server")
    p_srv.add_argument("--port", type=int, default=8791, help="Port (default: 8791)")
    p_srv.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    p_srv.add_argument("--checkpoint", default="v10s", help="Model checkpoint")
    p_srv.add_argument("--device", default=None, help="Device (mps, cuda, cpu)")
    p_srv.set_defaults(func=cmd_server)

    # console command
    p_con = subparsers.add_parser("console", help="Start interactive web dashboard")
    p_con.add_argument("--port", type=int, default=8766, help="Port (default: 8766)")
    p_con.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    p_con.add_argument("--checkpoint", default="v10s", help="Model checkpoint")
    p_con.add_argument("--device", default=None, help="Device (mps, cuda, cpu)")
    p_con.add_argument("--headless", action="store_true", help="Run browser in background")
    p_con.set_defaults(func=cmd_console)

    # verify command
    p_ver = subparsers.add_parser("verify", help="Run model verification test")
    p_ver.add_argument("--checkpoint", default="v10s", help="Model checkpoint")
    p_ver.add_argument("--device", default=None, help="Device (mps, cuda, cpu)")
    p_ver.set_defaults(func=cmd_verify)

    # chrome command
    p_chr = subparsers.add_parser("chrome", help="Launch Chrome with CDP enabled")
    p_chr.add_argument("--port", type=int, default=9222, help="CDP Port (default: 9222)")
    p_chr.add_argument("--headless", action="store_true", help="Launch in headless mode")
    p_chr.set_defaults(func=cmd_chrome)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
