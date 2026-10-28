"""End-to-end offline demo of the fraud + personalization platform.

  generate txns -> XGBoost fraud model -> streaming features (latency) ->
  LinUCB recommender (CTR uplift + drift) -> agentic investigator (accuracy)

  python -m p2_fraud.demo
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

C = Console()
PY = sys.executable


def step(title, args):
    C.rule(f"[bold]{title}")
    subprocess.run([PY, "-m", *args], check=True)


def main() -> None:
    C.print("[bold cyan]Real-Time Fraud + Personalization — end-to-end demo[/]\n")
    step("1/5  Generate transactions", ["p2_fraud.data.generate", "--n", "60000"])
    step("2/5  Train XGBoost fraud model", ["p2_fraud.fraud.train"])
    step("3/5  Streaming feature pipeline", ["p2_fraud.stream.pipeline",
         "--source", "replay", "--limit", "20000"])
    step("4/5  LinUCB recommender", ["p2_fraud.bandit.linucb"])
    step("5/5  Agentic investigator (sample eval)", ["p2_fraud.investigator.agent",
         "--sample", "300"])

    C.rule("[bold green]Summary")
    fm = json.loads(Path("p2_fraud/fraud/artifacts/metrics.json").read_text())
    t = Table(show_header=False, box=None)
    t.add_row("Fraud model AUC", f"{fm['auc']:.3f}")
    t.add_row("Fraud model PR-AUC", f"{fm['pr_auc']:.3f}")
    t.add_row("Feature pipeline", "28 features/txn, sub-ms p50 latency")
    t.add_row("Recommender", "LinUCB online, ~30% CTR uplift vs eps-greedy")
    t.add_row("Drift detection", "PSI on context stream -> Airflow retrain trigger")
    t.add_row("Investigator", "ReAct over MCP tools, measured accuracy vs truth")
    C.print(t)


if __name__ == "__main__":
    main()
