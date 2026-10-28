# SETUP — fraud-detection, step by step (with a fix for every step)

Gets the pipeline running on a Mac (works on Linux too), free. Each step has an
**if it fails** note.

## 1. Prerequisites

```bash
# Homebrew (if you don't have it)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12 libomp    # need Python >= 3.11; libomp is for xgboost
brew install --cask docker         # only needed for the live streaming path
# optional, free local LLM for the investigator:
brew install ollama && ollama serve & ollama pull llama3.2:3b
```
**If it fails:** `brew: command not found` → `eval "$(/opt/homebrew/bin/brew shellenv)"`. `xgboost` import error about libomp → `brew install libomp` then `pip install --force-reinstall xgboost`.

## 2. Run the offline demo (no Docker, no keys)

```bash
make venv && source .venv/bin/activate
make env                      # creates .env (optional for the demo)
pip install -r requirements-ci.txt
make p2-demo
```
**If it fails:**
- `make: command not found` → `xcode-select --install`.
- `ModuleNotFoundError: shared` → run from the repo root with the venv active (the `make` targets do this).
- 500k rows slow/heavy → `python -m p2_fraud.data.generate --n 100000` (it's linear).

## 3. Serve the fraud model as an API

```bash
make p2-data && make p2-fraud      # produce data + model
make p2-serve                      # uvicorn on :8000 — open http://localhost:8000/docs
curl -X POST localhost:8000/score -H 'Content-Type: application/json' \
  -d '{"amount":1450,"merchant_category":"gift_card","country":"BR"}'
```
**If it fails:** `/healthz` says model not trained → run `make p2-fraud` first.

## 4. The live streaming pipeline (optional, needs Docker)

```bash
make up-fraud          # redpanda + redis
make p2-data
make p2-fraud
make p2-stream         # feature latency report
make p2-bandit         # CTR uplift + PSI drift
```
**If it fails:**
- `bytewax` won't install → you don't need it; the stream auto-falls back to a single-process engine.
- `redis connection refused` → `make up-fraud`, or use `--sink memory` (default).

## 5. Use the agent from Claude Desktop (MCP)

```bash
python -m p2_fraud.investigator.mcp_server
```
Add the block printed at the top of `p2_fraud/investigator/mcp_server.py` to
Claude Desktop's config (use **absolute** paths), restart it, and ask Claude to
"investigate transaction <id>".

## 6. Keeping it free

Default is **$0** (XGBoost + local tools). The investigator uses a local ReAct
fallback unless you set a key in `.env`:
```bash
ANTHROPIC_API_KEY=...    # and set USE_LOCAL_LLM=false
```

## 7. Publish / deploy

```bash
# push to GitHub
gh repo create fraud-detection --public --source=. --remote=origin --push
# or: git remote add origin <url> && git push -u origin main
```
Deploy the scoring API to Fly.io (free, scale-to-zero) — see **[DEPLOY.md](DEPLOY.md)**.
The GitHub Action auto-deploys after CI passes.

## Troubleshooting cheatsheet
| Symptom | Fix |
|---|---|
| `make: command not found` | `xcode-select --install` |
| `xgboost` libomp error | `brew install libomp` |
| `ModuleNotFoundError: shared` | run from repo root with the venv active |
| `redis` refused | `make up-fraud`, or use `--sink memory` |
| tests fail on fresh clone | `pip install -r requirements-ci.txt -r requirements-dev.txt && bash scripts/ci_smoke.sh` |
