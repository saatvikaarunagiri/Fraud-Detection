# 💳 Real-Time Fraud Detection + Personalization

One feature pipeline, **two models**: an XGBoost fraud detector and a live recommender — plus a Claude agent that investigates flagged charges and a REST API to serve it all. Runs on your laptop in one command. No GPU, no API keys, no cloud bill.

<p>
<img alt="python" src="https://img.shields.io/badge/python-3.11+-blue">
<img alt="tests" src="https://img.shields.io/badge/tests-passing-brightgreen">
<img alt="runs on" src="https://img.shields.io/badge/runs%20on-a%20MacBook-black">
<img alt="cost" src="https://img.shields.io/badge/cost-%240-success">
<img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## ⚡ Try it (30 seconds)

```bash
make venv && source .venv/bin/activate   # 1. set up
pip install -r requirements-ci.txt        # 2. install (light, no torch)
make p2-demo                              # 3. watch it all run 🎬
```

Generates transactions, trains the fraud model, streams features, runs the recommender, and grades the agent — start to finish in under a minute, printing every score.

---

## 🤔 What it does

Every card swipe becomes **28 features** in under a millisecond. Those same features feed two things at once:

- 🚨 **Fraud detector** (XGBoost) — flags suspicious charges.
- 🎁 **Recommender** (LinUCB bandit) — picks what to show next, learning live from clicks, with drift detection that triggers a retrain.

Flagged charges go to a 🕵️ **Claude agent** that investigates step-by-step (score → explain → check history → decide), usable from Claude Desktop via an **MCP server** or as a **REST API**.

```
transactions ──▶ 28 features ──┬──▶ XGBoost fraud model ──▶ 🕵️ agent / API
                               └──▶ LinUCB recommender  ──▶ clicks (+ drift alarm)
```

> 🎯 **0.99 AUC** fraud model &nbsp;·&nbsp; ⚡ **sub-millisecond** features &nbsp;·&nbsp; 📈 **+30% click-through** vs baseline

---

## 🛠 Use it like a tool

| I want to… | Run this |
|---|---|
| See everything work | `make p2-demo` |
| See the numbers on one page | `make dashboard` → http://localhost:9000 |
| Serve the fraud model as an API | `make p2-data && make p2-fraud && make p2-serve` → `/docs` |
| Investigate from Claude Desktop | `python -m p2_fraud.investigator.mcp_server` |
| Run the live streaming pipeline | `make up-fraud` then `make p2-stream` |
| Run the tests | `make test` |
| Put the API online (free) | [DEPLOY.md](DEPLOY.md) → Fly.io in 5 min |
| Something broke? | [SETUP.md](SETUP.md) — a fix for every step |

📖 **Want the deep dive** (the features, the models, every metric)? → **[p2_fraud/README.md](p2_fraud/README.md)**

---

## 🧩 Good to know

- **🆓 Free by default** — XGBoost + local tools. Add an API key only for a hosted LLM in the investigator.
- **🪆 Nothing is required** — no Bytewax? no Docker? no internet? It falls back to a lighter path and still runs.
- **🧪 Numbers are real but synthetic** — the demo *measures* its scores on generated transactions (so AUC looks high). Point it at real data (the guide shows where) for your real numbers.

<sub>Built to run on an M-series MacBook · MIT licensed · `make help` for the full menu</sub>
