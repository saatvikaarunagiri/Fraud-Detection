"""Fraud-scoring REST API (Project 2).

Serves the trained XGBoost fraud model over HTTP — the deployable face of the
platform. It reuses the exact scoring/explanation logic from the investigator
tools, so the API and the agent agree by construction.

Endpoints
---------
  GET  /healthz                 liveness + whether the model is loaded
  GET  /score/{txn_id}          fraud probability + FLAG/ALLOW for a known txn
  GET  /explain/{txn_id}        the feature factors that drove the score
  GET  /investigate/{txn_id}    full agentic investigation (ReAct, local fallback)
  POST /score                   score an arbitrary transaction body (stateless)
  GET  /docs                    interactive OpenAPI UI (FastAPI built-in)

Run locally:
  make p2-data && make p2-fraud          # produce data + model first
  make p2-serve                          # uvicorn on :8000
  curl localhost:8000/healthz
"""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..features import FeatureStore, FEATURE_NAMES
from ..investigator import tools

app = FastAPI(
    title="Fraud Scoring API",
    description="XGBoost fraud model served over HTTP, with agentic investigation.",
    version="1.0.0",
)


class TxnIn(BaseModel):
    user_id: str = "u_demo"
    amount: float
    merchant: str = "m_demo"
    merchant_category: str = "electronics"
    country: str = "US"
    device_id: str = "d_demo"


@app.get("/healthz")
def healthz():
    model_loaded = tools._model() is not None
    return {"ok": True, "model_loaded": model_loaded,
            "hint": None if model_loaded else "train it: make p2-data && make p2-fraud"}


@app.get("/score/{txn_id}")
def score(txn_id: str):
    out = tools.score_transaction(txn_id)
    if "error" in out:
        raise HTTPException(404 if "no transaction" in out["error"] else 503, out["error"])
    return out


@app.get("/explain/{txn_id}")
def explain(txn_id: str):
    out = tools.explain_score(txn_id)
    if "error" in out:
        raise HTTPException(404 if "no transaction" in out["error"] else 503, out["error"])
    return out


@app.get("/investigate/{txn_id}")
def investigate(txn_id: str):
    from ..investigator.agent import investigate as run
    if txn_id not in tools._index()[0]:
        raise HTTPException(404, f"no transaction {txn_id}")
    return run(txn_id)


@app.post("/score")
def score_body(txn: TxnIn):
    """Stateless scoring of an arbitrary transaction. Note: velocity/history
    features are zero here (no per-user history), so this is a lower-signal score
    than /score/{txn_id} over the indexed stream — good for quick what-ifs."""
    model = tools._model()
    if model is None:
        raise HTTPException(503, "fraud model not trained; run `make p2-fraud`")
    feats = FeatureStore().compute_and_update(txn.model_dump() | {"ts": __import__("time").time()})
    p = float(model.predict_proba(np.asarray([feats], dtype="float32"))[0, 1])
    return {"fraud_probability": round(p, 4),
            "decision": "FLAG" if p > 0.5 else "ALLOW",
            "features_used": dict(zip(FEATURE_NAMES, [round(float(x), 3) for x in feats]))}
