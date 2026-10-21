"""
Human Activity Recognition (HAR) Web Dashboard Launcher
Launches a local HTTP server and automatically opens the dashboard in your default browser.
"""
import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

PORT = 8000
DASHBOARD_DIR = Path(__file__).parent / "dashboard"

class CustomHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

def main():
    if not (DASHBOARD_DIR / "index.html").exists():
        print(f"Error: Dashboard files not found in {DASHBOARD_DIR}")
        sys.exit(1)

    url = f"http://localhost:{PORT}/index.html"
    print("=" * 65)
    print("  HUMAN ACTIVITY RECOGNITION (HAR) — WEB DASHBOARD")
    print("=" * 65)
    print(f"  Serving dashboard at: {url}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 65)

    webbrowser.open(url)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), CustomHTTPHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()
