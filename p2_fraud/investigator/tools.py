"""Investigation tools over the fraud model + transaction history.

Defined once here, then exposed two ways:
  * as an MCP server (mcp_server.py) for Claude Desktop / any MCP client
  * imported directly by the ReAct agent (agent.py)

So the agent and the MCP server share identical tool behavior.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..features import FeatureStore, FEATURE_NAMES

DATA = Path("p2_fraud/data/txns.jsonl")
MODEL = Path("p2_fraud/fraud/artifacts/fraud_xgb.json")


@lru_cache(maxsize=1)
def _index():
    """Build: txn_id -> txn, user_id -> [txns], and features per txn (causal)."""
    txns = [json.loads(l) for l in DATA.read_text().splitlines()]
    txns.sort(key=lambda t: t["ts"])
    store = FeatureStore()
    by_id, by_user, feats = {}, {}, {}
    for t in txns:
        feats[t["txn_id"]] = store.compute_and_update(t)
        by_id[t["txn_id"]] = t
        by_user.setdefault(t["user_id"], []).append(t)
    return by_id, by_user, feats


@lru_cache(maxsize=1)
def _model():
    if not MODEL.exists():
        return None
    import xgboost as xgb
    m = xgb.XGBClassifier(); m.load_model(str(MODEL)); return m


# --- tools ------------------------------------------------------------------
def get_transaction(txn_id: str) -> dict:
    """Return the raw transaction by id."""
    by_id, _, _ = _index()
    return by_id.get(txn_id, {"error": f"no transaction {txn_id}"})


def get_user_history(user_id: str, limit: int = 10) -> dict:
    """Return a user's recent transactions (most recent first)."""
    _, by_user, _ = _index()
    hist = sorted(by_user.get(user_id, []), key=lambda t: -t["ts"])[:limit]
    return {"user_id": user_id, "n": len(hist),
            "transactions": [{k: t[k] for k in
                              ("txn_id", "amount", "merchant_category", "country",
                               "device_id", "ts")} for t in hist]}


def score_transaction(txn_id: str) -> dict:
    """Run the fraud model on a transaction. Returns probability + decision."""
    by_id, _, feats = _index()
    if txn_id not in by_id:
        return {"error": f"no transaction {txn_id}"}
    m = _model()
    if m is None:
        return {"error": "fraud model not trained; run `make p2-fraud`"}
    p = float(m.predict_proba(np.asarray([feats[txn_id]], dtype="float32"))[0, 1])
    return {"txn_id": txn_id, "fraud_probability": round(p, 4),
            "decision": "FLAG" if p > 0.5 else "ALLOW"}


def explain_score(txn_id: str, top_k: int = 6) -> dict:
    """Explain a score: the feature values that most pushed it toward fraud,
    using the model's feature importances weighted by this txn's values."""
    by_id, _, feats = _index()
    if txn_id not in by_id:
        return {"error": f"no transaction {txn_id}"}
    m = _model()
    if m is None:
        return {"error": "model not trained"}
    x = np.asarray(feats[txn_id])
    contrib = m.feature_importances_ * (np.abs(x) / (np.abs(x).max() + 1e-9))
    order = np.argsort(-contrib)[:top_k]
    return {"txn_id": txn_id,
            "top_factors": [{"feature": FEATURE_NAMES[i],
                             "value": round(float(x[i]), 3),
                             "weight": round(float(contrib[i]), 4)} for i in order]}


def flag_transaction(txn_id: str, reason: str) -> dict:
    """Record an investigation decision (writes to a local case file)."""
    out = Path("p2_fraud/investigator/cases.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps({"txn_id": txn_id, "reason": reason}) + "\n")
    return {"ok": True, "txn_id": txn_id, "logged_reason": reason}


TOOLS = {t.__name__: t for t in
         (get_transaction, get_user_history, score_transaction, explain_score, flag_transaction)}
