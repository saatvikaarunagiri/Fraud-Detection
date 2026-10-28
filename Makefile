# =============================================================================
# Fraud Detection + Personalization — one-command helpers. Run `make help`.
# All python targets run from the repo root so `import shared...` resolves.
# =============================================================================
.DEFAULT_GOAL := help
SHELL := /bin/bash
PY := .venv/bin/python

.PHONY: help venv env doctor up-fraud up-dash down clean \
        p2-data p2-fraud p2-stream p2-bandit p2-investigator p2-serve p2-demo \
        dashboard test

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

# ---- setup ----
venv:  ## Create the Python virtualenv
	python3 -m venv .venv && .venv/bin/pip install -U pip
env:  ## Create .env from the example if missing
	@test -f .env || (cp .env.example .env && echo "created .env")
doctor:  ## Check your machine has what it needs
	$(PY) scripts/doctor.py

# ---- infra (docker profiles) ----
up-fraud:  ## Start infra (redpanda, redis)
	docker compose --profile fraud up -d
up-dash:  ## Start Grafana
	docker compose --profile dash up -d
down:  ## Stop all containers (keeps data)
	docker compose down
clean:  ## Stop all containers AND delete data volumes
	docker compose down -v

# ---- the pipeline ----
p2-data:         ## Generate synthetic transactions
	$(PY) -m p2_fraud.data.generate --n 500000
p2-fraud:        ## Train the XGBoost fraud model
	$(PY) -m p2_fraud.fraud.train
p2-stream:       ## Run the streaming feature pipeline (Bytewax)
	$(PY) -m p2_fraud.stream.pipeline
p2-bandit:       ## Run the LinUCB recommender online demo
	$(PY) -m p2_fraud.bandit.linucb
p2-investigator: ## Start the agentic fraud investigator (+ MCP server)
	$(PY) -m p2_fraud.investigator.agent
p2-serve:        ## Serve the fraud-scoring REST API (FastAPI on :8000)
	.venv/bin/uvicorn p2_fraud.serve.app:app --reload --port 8000
p2-demo:         ## End-to-end offline demo
	$(PY) -m p2_fraud.demo

# ---- extras ----
dashboard:  ## Serve the metrics page at :9000
	$(PY) -m dashboard.server
test:  ## Run the fast unit tests
	$(PY) -m pytest -q
