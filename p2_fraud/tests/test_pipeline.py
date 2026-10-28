"""Fast tests for Project 2. Run: .venv/bin/pytest p2_fraud"""
import numpy as np

from p2_fraud.features import FeatureStore, FEATURE_NAMES
from p2_fraud.data.generate import gen
from p2_fraud.bandit.linucb import LinUCB, psi, simulate


def test_features_count_and_determinism():
    store = FeatureStore()
    txns = [t.__dict__ for t in gen(200, seed=1, fraud_rate=0.1)]
    for t in txns:
        f = store.compute_and_update(t)
        assert len(f) == 28 == len(FEATURE_NAMES)
        assert all(isinstance(x, float) for x in f)


def test_generator_has_both_classes():
    txns = [t.__dict__ for t in gen(2000, seed=2, fraud_rate=0.1)]
    labels = {t["is_fraud"] for t in txns}
    assert labels == {0, 1}


def test_linucb_beats_baseline():
    res = simulate(n=4000)
    assert res["linucb_ctr"] > res["baseline_ctr"]       # contextual should win
    assert res["drift_alert"] is True                    # injected drift fires


def test_psi_zero_for_same_dist():
    x = np.random.RandomState(0).normal(size=5000)
    assert psi(x, x.copy()) < 0.01


def test_investigator_local_runs():
    # needs a trained model + data; skip gracefully if absent
    from pathlib import Path
    if not Path("p2_fraud/fraud/artifacts/fraud_xgb.json").exists():
        import pytest; pytest.skip("model not trained yet")
    from p2_fraud.investigator import tools, agent
    tid = next(iter(tools._index()[0]))
    res = agent.investigate_local(tid)
    assert res["verdict"] in ("FLAG", "ALLOW") and res["steps"] >= 4
