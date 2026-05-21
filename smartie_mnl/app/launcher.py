"""
SMARTIE entry point.

Installed as the `SMARTIE` command when you run `pip install smartie`.
Launches the Streamlit web interface in your default browser.

Usage:
    SMARTIE                    # starts on port 8501, opens browser
    SMARTIE --port 8502        # custom port
    SMARTIE --no-browser       # don't open browser automatically

If SMARTIE is not recognised by your terminal (common on Windows), use:
    python -m smartie_mnl.app.launcher
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


APP_FILE = Path(__file__).parent / "app.py"


def _fix_path() -> None:
    """
    Ensure the Scripts / bin directory that contains `streamlit` is on PATH.

    pip  → Windows: <python>/Scripts
    pip  → Mac/Linux: <python>/../bin  (user install: ~/.local/bin)
    pipx → Mac/Linux: ~/.local/bin  (pipx ensurepath not always run)
    """
    candidates: list[Path] = []

    if sys.platform == "win32":
        # pip user install on Windows
        candidates.append(Path(sys.executable).parent / "Scripts")
        # also the user-level Scripts folder
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            candidates.append(Path(appdata) / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts")
    else:
        # pip system / venv install
        candidates.append(Path(sys.executable).parent)
        # pip --user install on Mac/Linux
        candidates.append(Path.home() / ".local" / "bin")
        # pipx default bin dir
        candidates.append(Path.home() / ".local" / "bin")
        # Homebrew pipx on Mac
        candidates.append(Path("/opt/homebrew/bin"))
        candidates.append(Path("/usr/local/bin"))

    current_path = os.environ.get("PATH", "")
    for d in candidates:
        if d.exists() and str(d) not in current_path:
            os.environ["PATH"] = str(d) + os.pathsep + current_path
            current_path = os.environ["PATH"]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="SMARTIE",
        description="Launch the SMARTIE web interface in your browser.",
    )
    parser.add_argument(
        "--port", type=int, default=8501,
        help="Port to run the server on (default: 8501).",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Start the server without opening a browser tab automatically.",
    )
    args = parser.parse_args()

    _fix_path()

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(APP_FILE),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.no_browser else "false",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light",
        "--theme.primaryColor", "#4CAF50",
    ]

    print(f"\n🧬 SMARTIE — web interface")
    print(f"   URL: http://localhost:{args.port}")
    print(f"   Press Ctrl+C to stop.\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
    except FileNotFoundError:
        print(
            "\n❌ Streamlit is not installed.\n"
            "   Run:  pip install streamlit\n"
            "   Then try again.\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
