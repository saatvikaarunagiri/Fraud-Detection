"""Synthetic transaction generator with realistic, learnable fraud patterns.

Fraud is injected via latent mechanisms (account takeover bursts, card testing,
high-value foreign gift-card buys, odd-hour spikes) so the engineered features
carry genuine signal and XGBoost can reach a high AUC — the same way a real
model separates fraud, just on data we're allowed to share.

  python -m p2_fraud.data.generate --n 500000
"""
from __future__ import annotations

import argparse
import calendar
import json
import random
import time
from pathlib import Path

from shared.schemas import Transaction


def ts_at_hour(now: float, days_ago: int, hour: int, rng) -> float:
    """A timestamp `days_ago` days back, pinned to a given GMT hour (so the
    is_night / hour features are actually correct)."""
    base = now - days_ago * 86400
    g = time.gmtime(base)
    midnight = calendar.timegm((g.tm_year, g.tm_mon, g.tm_mday, 0, 0, 0, 0, 0, 0))
    return midnight + hour * 3600 + rng.uniform(0, 3600)

MERCHANTS = [f"m_{i}" for i in range(400)]
CATEGORIES = ["electronics", "gift_card", "travel", "groceries", "fuel",
              "subscription", "jewelry"]
COUNTRIES = ["US", "GB", "DE", "IN", "BR", "JP"]


def gen(n: int, seed: int, fraud_rate: float):
    rng = random.Random(seed)
    users = [f"u_{i}" for i in range(max(50, n // 40))]
    # each user has a "home" profile
    home = {u: {"country": rng.choice(COUNTRIES),
                "device": f"d_{rng.getrandbits(32):08x}",
                "avg": rng.choice([20, 40, 80, 150])} for u in users}
    now = time.time()

    emitted = 0
    while emitted < n:
        u = rng.choice(users)
        h = home[u]

        if rng.random() < fraud_rate:
            mode = rng.choice(["takeover", "card_testing", "giftcard", "night_spike"])
            subtle = rng.random() < 0.28             # ~28% of fraud is hard to catch
            if mode == "takeover":                   # new device, foreign, big
                country = h["country"] if subtle else rng.choice(
                    [c for c in COUNTRIES if c != h["country"]])
                device = h["device"] if subtle else f"d_{rng.getrandbits(32):08x}"
                amount = h["avg"] * (rng.uniform(1.8, 3.0) if subtle else rng.uniform(6, 20))
                category = rng.choice(["electronics", "jewelry", "travel"])
                t = now - rng.uniform(0, 30 * 86400)
                yield Transaction(user_id=u, amount=round(amount, 2),
                                  merchant=rng.choice(MERCHANTS), merchant_category=category,
                                  country=country, device_id=device, is_fraud=1, ts=t)
                emitted += 1

            elif mode == "card_testing":             # a BURST of tiny txns, same user/min
                device = f"d_{rng.getrandbits(32):08x}"
                t0 = now - rng.uniform(0, 30 * 86400)
                for _ in range(rng.randint(6, 14)):
                    yield Transaction(
                        user_id=u, amount=round(rng.uniform(0.5, 4.0), 2),
                        merchant=rng.choice(MERCHANTS),
                        merchant_category=rng.choice(["subscription", "gift_card"]),
                        country=h["country"], device_id=device, is_fraud=1,
                        ts=t0 + rng.uniform(0, 300))   # all within ~5 min
                    emitted += 1

            elif mode == "giftcard":                 # high-value gift cards abroad
                country = h["country"] if subtle else rng.choice(["BR", "IN"])
                device = h["device"] if rng.random() < 0.3 else f"d_{rng.getrandbits(32):08x}"
                amount = rng.uniform(150, 400) if subtle else rng.uniform(400, 1500)
                t = now - rng.uniform(0, 30 * 86400)
                yield Transaction(user_id=u, amount=round(amount, 2),
                                  merchant=rng.choice(MERCHANTS), merchant_category="gift_card",
                                  country=country, device_id=device, is_fraud=1, ts=t)
                emitted += 1

            else:                                    # night_spike: 0-5am GMT, high amount
                t = ts_at_hour(now, rng.randint(0, 29), rng.randint(0, 5), rng)
                yield Transaction(
                    user_id=u, amount=round(h["avg"] * rng.uniform(4, 10), 2),
                    merchant=rng.choice(MERCHANTS), merchant_category=rng.choice(CATEGORIES),
                    country=h["country"], device_id=h["device"], is_fraud=1, ts=t)
                emitted += 1
        else:                                        # legit (daytime-biased)
            anomalous = rng.random() < 0.05          # legit-but-suspicious -> overlap
            country = h["country"] if rng.random() < 0.97 else rng.choice(COUNTRIES)
            device = (f"d_{rng.getrandbits(32):08x}" if (anomalous and rng.random() < 0.5)
                      else (h["device"] if rng.random() < 0.92 else f"d_{rng.getrandbits(32):08x}"))
            amount = max(1.0, rng.gauss(h["avg"] * (3.0 if anomalous else 1.0),
                                        h["avg"] * 0.35))
            hr = rng.randint(0, 5) if (anomalous and rng.random() < 0.4) else rng.randint(8, 22)
            t = ts_at_hour(now, rng.randint(0, 29), hr, rng)
            yield Transaction(
                user_id=u, amount=round(amount, 2), merchant=rng.choice(MERCHANTS),
                merchant_category=rng.choice(CATEGORIES), country=country,
                device_id=device, is_fraud=0, ts=t)
            emitted += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--fraud-rate", type=float, default=0.03)
    ap.add_argument("--out", default="p2_fraud/data/txns.jsonl")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n_fraud = 0
    # sort by ts so streaming/feature history is causal
    rows = sorted(gen(args.n, args.seed, args.fraud_rate), key=lambda r: r.ts)
    with out.open("w") as f:
        for r in rows:
            n_fraud += r.is_fraud
            f.write(json.dumps(r.__dict__) + "\n")
    print(f"wrote {len(rows)} txns ({n_fraud} fraud, {n_fraud/len(rows):.2%}) -> {out}")


if __name__ == "__main__":
    main()
