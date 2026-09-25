"""Katai: Ultra-Fast Non-Autoregressive Browser Agent.

Driven by fine-tuned decision heads for sub-second, calibrated web automation.
"""

import os
import sys
from pathlib import Path

# Auto-re-exec into local .venv if executed with an external Python interpreter missing packages
venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
if venv_python.exists() and sys.executable != str(venv_python):
    try:
        import websockets
        import torch
    except (ImportError, ModuleNotFoundError):
        if hasattr(sys, "orig_argv") and len(sys.orig_argv) > 1:
            re_args = [str(venv_python)] + sys.orig_argv[1:]
        else:
            re_args = [str(venv_python), "-m", "katai"] + sys.argv[1:]
        os.execv(str(venv_python), re_args)

from katai.core.agent import KataiAgent
from katai.core.browser import Browser, StalePage, action_space, launch_chrome
from katai.core.engine import KataiEngine
from katai.core.text_helper import extract_heuristic_text, extract_llm_text, get_field_text
from katai.server.systemone import start_server
from katai.web.console import start_console

__version__ = "1.1"
__all__ = [
    "KataiAgent",
    "KataiEngine",
    "Browser",
    "StalePage",
    "action_space",
    "launch_chrome",
    "extract_heuristic_text",
    "extract_llm_text",
    "get_field_text",
    "start_server",
    "start_console",
]
