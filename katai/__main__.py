"""Entry point for python -m katai or python3 katai."""

import os
import sys
from pathlib import Path

# Auto-re-exec into .venv if dependencies are missing in the current interpreter
venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
if venv_python.exists() and sys.executable != str(venv_python):
    try:
        import katai
        import laya
    except (ImportError, ModuleNotFoundError):
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from katai.cli import main
except ImportError:
    from .cli import main

if __name__ == "__main__":
    main()
