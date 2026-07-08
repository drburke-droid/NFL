"""
Local helper for the draft tool's "↻ Buzz" button.

Serves the draft tool at http://127.0.0.1:8765 and exposes:
  POST /api/refresh-buzz   -> runs, in order:
                                fetch_wiki_buzz.py refresh   (daily Jun-Sep pageviews)
                                late_breakout_final.py       (rescore darts w/ buzz)
                                build_draft_tool.py          (rebuild data.js)
  GET  /api/refresh-status -> {"state": "idle|running|done|error", "step": ..., "error": ...}

CORS is open (and private-network preflights answered) so the button also works
from the GitHub Pages copy or a file:// open — the pipeline always runs locally
and updates outputs/draft_tool/ + docs/; push to publish the refreshed board.

Run:  python scripts/buzz_server.py        (Ctrl-C to stop)
"""
import json, os, subprocess, sys, threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "outputs", "draft_tool")
PORT = 8765

STEPS = [
    ("pulling Wikipedia pageviews", ["fetch_wiki_buzz.py", "refresh"]),
    ("rescoring late-breakout darts", ["late_breakout_final.py"]),
    ("rebuilding draft board", ["build_draft_tool.py"]),
]
STATUS = {"state": "idle", "step": "", "error": ""}


def run_pipeline():
    for label, cmd in STEPS:
        STATUS.update(state="running", step=label)
        print(f"== {label}: {' '.join(cmd)}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", cmd[0])] + cmd[1:],
                           cwd=ROOT, capture_output=True, text=True)
        print(r.stdout[-2000:] if r.stdout else "", flush=True)
        if r.returncode != 0:
            print(r.stderr[-2000:], flush=True)
            STATUS.update(state="error", error=f"{cmd[0]} failed (exit {r.returncode})")
            return
    STATUS.update(state="done", step="")
    print("== refresh complete — reload the page", flush=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=TOOL, **kw)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/refresh-status"):
            return self._json(STATUS)
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/refresh-buzz"):
            if STATUS["state"] == "running":
                return self._json({"ok": False, "reason": "already running"}, 409)
            STATUS.update(state="running", step="starting", error="")
            threading.Thread(target=run_pipeline, daemon=True).start()
            return self._json({"ok": True})
        self.send_error(404)

    def log_message(self, fmt, *args):
        if "/api/refresh-status" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print(f"draft tool + buzz refresh helper: http://127.0.0.1:{PORT}  (Ctrl-C to stop)")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
