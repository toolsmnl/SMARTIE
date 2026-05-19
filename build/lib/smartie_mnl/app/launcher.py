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


def _fix_windows_path() -> None:
    """
    On Windows, pip installs scripts to a folder that is often not on PATH.
    This function adds the correct Scripts directory to the current process
    PATH so that `streamlit` can be found even when the user has not set
    up their environment variables.
    """
    if sys.platform != "win32":
        return
    scripts_dir = Path(sys.executable).parent / "Scripts"
    if scripts_dir.exists() and str(scripts_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")


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

    _fix_windows_path()

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
