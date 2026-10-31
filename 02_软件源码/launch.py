"""Launch the packaged application using the active Python environment."""
import argparse
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    port = args.port
    if not port:
        for candidate in range(7860, 7881):
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", candidate))
                    port = candidate
                    break
                except OSError:
                    continue
    if not port:
        parser.error("No available local port in 7860..7880; choose --port.")
    import app
    app.build_ui().queue().launch(server_name="127.0.0.1", server_port=port, inbrowser=not args.no_browser)

if __name__ == "__main__":
    main()
