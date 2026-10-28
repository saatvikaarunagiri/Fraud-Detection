"""Plain metrics page for the fraud pipeline. Reads the artifacts the demo
writes. No framework — a stdlib HTTP server.  `make dashboard` -> :9000
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(path: str) -> dict:
    try:
        return json.loads((ROOT / path).read_text())
    except Exception:
        return {}


def _count(path: str) -> int:
    try:
        return sum(1 for _ in (ROOT / path).open())
    except Exception:
        return 0


def render() -> str:
    fraud = _load("p2_fraud/fraud/artifacts/metrics.json")

    def row(label, value):
        return f"<tr><td>{label}</td><td><b>{value}</b></td></tr>"

    def na(d, k, fmt="{}"):
        return fmt.format(d[k]) if k in d else "<i>run make p2-demo</i>"

    cases = _count("p2_fraud/investigator/cases.jsonl")
    rows = "".join([
        row("Fraud model AUC", na(fraud, "auc")),
        row("Fraud model PR-AUC", na(fraud, "pr_auc")),
        row("Test transactions", na(fraud, "n_test")),
        row("Top features", ", ".join(f[0] for f in fraud.get("top_features", [])[:4]) or "<i>—</i>"),
        row("Investigations logged", cases or "<i>—</i>"),
    ])

    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>Fraud Detection — Dashboard</title>
<style>
body{{font:14px/1.5 -apple-system,system-ui,sans-serif;color:#111;background:#fff;max-width:720px;margin:30px auto;padding:0 16px}}
h1{{font-size:20px}} table{{border-collapse:collapse;width:100%;margin-top:6px}}
td{{padding:6px 8px;border-bottom:1px solid #f0f0f0}} td:first-child{{color:#555;width:55%}}
.muted{{color:#888}} code{{background:#f6f6f6;padding:1px 5px;border-radius:4px}}
</style></head><body>
<h1>Fraud Detection + Personalization — dashboard</h1>
<p class=muted>Live from the pipeline's artifacts. Populate with <code>make p2-demo</code>.</p>
<table>{rows}</table>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = render().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main(port: int = 9000) -> None:
    print(f"dashboard -> http://localhost:{port}  (Ctrl-C to stop)")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
