"""Train the XGBoost fraud model over engineered features.

Streams the (time-sorted) transactions through the SAME FeatureStore the live
pipeline uses, so offline training and online serving share feature code.
Reports AUC + PR-AUC and saves the model for the investigator/MCP server.

  python -m p2_fraud.fraud.train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..features import FeatureStore, FEATURE_NAMES

ART = Path("p2_fraud/fraud/artifacts")
ART.mkdir(parents=True, exist_ok=True)


def build_matrix(path: str):
    store = FeatureStore()
    X, y = [], []
    for line in Path(path).read_text().splitlines():
        txn = json.loads(line)
        X.append(store.compute_and_update(txn))
        y.append(int(txn["is_fraud"]))
    return np.asarray(X, dtype="float32"), np.asarray(y), store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="p2_fraud/data/txns.jsonl")
    args = ap.parse_args()

    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

    X, y, _ = build_matrix(args.data)
    print(f"matrix {X.shape}  fraud={y.mean():.2%}")
    # time-aware split: first 80% train, last 20% test (no leakage)
    cut = int(len(X) * 0.8)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    scale = (ytr == 0).sum() / max(1, (ytr == 1).sum())   # class imbalance
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
        scale_pos_weight=scale, n_jobs=-1, tree_method="hist")
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

    proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    prauc = average_precision_score(yte, proba)
    pred = (proba > 0.5).astype(int)
    print(f"\nAUC={auc:.4f}  PR-AUC={prauc:.4f}")
    print(classification_report(yte, pred, digits=3, zero_division=0))

    # feature importance (top 8)
    imp = sorted(zip(FEATURE_NAMES, model.feature_importances_),
                 key=lambda x: -x[1])[:8]
    print("top features:", ", ".join(f"{n}({v:.2f})" for n, v in imp))

    model.save_model(str(ART / "fraud_xgb.json"))
    (ART / "metrics.json").write_text(json.dumps(
        {"auc": round(float(auc), 4), "pr_auc": round(float(prauc), 4),
         "n_test": int(len(yte)), "fraud_rate": round(float(y.mean()), 4),
         "top_features": [[n, round(float(v), 4)] for n, v in imp]}, indent=2))
    print(f"saved -> {ART/'fraud_xgb.json'}")


if __name__ == "__main__":
    main()
