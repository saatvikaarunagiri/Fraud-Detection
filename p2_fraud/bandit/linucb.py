"""LinUCB contextual-bandit recommender.

Serves recommendations from the SAME streaming features as the fraud model
(single workflow, two consumers). Updates online from each click. We include:

  * LinUCB (disjoint) — the recommender.
  * A simulation harness with a hidden user-preference model so we can MEASURE
    CTR uplift vs an epsilon-greedy baseline (the "22% CTR uplift" claim).
  * PSI (Population Stability Index) drift detection on the context stream, the
    trigger for Airflow retraining in production.

  python -m p2_fraud.bandit.linucb
"""
from __future__ import annotations

import argparse
import math
import random

import numpy as np


class LinUCB:
    def __init__(self, n_arms: int, d: int, alpha: float = 0.6):
        self.n_arms, self.d, self.alpha = n_arms, d, alpha
        self.A = [np.identity(d) for _ in range(n_arms)]      # d x d per arm
        self.b = [np.zeros(d) for _ in range(n_arms)]
        self.updates = 0

    def recommend(self, x: np.ndarray) -> int:
        best, best_p = 0, -1e9
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p = float(theta @ x + self.alpha * math.sqrt(x @ A_inv @ x))
            if p > best_p:
                best_p, best = p, a
        return best

    def update(self, a: int, x: np.ndarray, reward: float) -> None:
        self.A[a] += np.outer(x, x)
        self.b[a] += reward * x
        self.updates += 1


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D distributions.
    >0.25 conventionally signals significant drift -> retrain."""
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0] / len(expected) + 1e-6
    a = np.histogram(actual, qs)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def simulate(n: int = 20000, n_arms: int = 6, d: int = 8, seed: int = 0):
    """Hidden preference model: clicks are Bernoulli with a context-dependent
    rate per arm. LinUCB (uses context) is compared against a FAIR epsilon-greedy
    baseline that actually learns each arm's average reward but ignores context —
    which is exactly the lift a contextual model should buy you."""
    rng = np.random.default_rng(seed)
    true_W = rng.normal(size=(n_arms, d))

    def click_prob(a, x):
        return 1 / (1 + math.exp(-(true_W[a] @ x)))

    lin = LinUCB(n_arms, d, alpha=0.6)
    eps = 0.1
    Q = np.zeros(n_arms); N = np.zeros(n_arms)        # eps-greedy value estimates
    lin_clicks = base_clicks = 0
    pre, post = [], []                                 # a context feature for PSI

    for t in range(n):
        x = rng.normal(size=d)
        # real covariate shift in the SECOND half: mean of feature 0 moves +1.5
        if t >= n // 2:
            x[0] += 1.5
        xn = x / (np.linalg.norm(x) + 1e-9)
        (pre if t < n // 2 else post).append(x[0])

        # LinUCB (contextual)
        a = lin.recommend(xn)
        r = 1.0 if rng.random() < click_prob(a, xn) else 0.0
        lin.update(a, xn, r); lin_clicks += r

        # epsilon-greedy (context-free, but genuinely learning)
        ba = int(rng.integers(n_arms)) if rng.random() < eps else int(np.argmax(Q))
        br = 1.0 if rng.random() < click_prob(ba, xn) else 0.0
        N[ba] += 1; Q[ba] += (br - Q[ba]) / N[ba]
        base_clicks += br

    lin_ctr, base_ctr = lin_clicks / n, base_clicks / n
    drift = psi(np.array(pre), np.array(post))
    return {
        "linucb_ctr": round(lin_ctr, 4),
        "baseline_ctr": round(base_ctr, 4),
        "ctr_uplift_pct": round(100 * (lin_ctr - base_ctr) / max(base_ctr, 1e-6), 1),
        "updates": lin.updates,
        "context_psi": round(drift, 3),
        "drift_alert": drift > 0.25,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    res = simulate(args.n)
    print("LinUCB recommender simulation")
    for k, v in res.items():
        print(f"  {k:16} {v}")
    if res["drift_alert"]:
        print("  -> PSI > 0.25: context drift detected, trigger Airflow retrain")


if __name__ == "__main__":
    main()
