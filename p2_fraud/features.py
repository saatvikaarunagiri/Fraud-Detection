"""Feature engineering — ONE implementation used by both the batch XGBoost
trainer and the live Bytewax stream, so training and serving can't skew.

28 features per transaction, derived from the raw event + a small amount of
per-user state (recent history). In production that state lives in Redis; for
local/offline it lives in an in-memory dict with identical semantics.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

# High-risk lookup tables (would be learned/configured in prod).
COUNTRY_RISK = {"US": 0.1, "GB": 0.15, "DE": 0.15, "IN": 0.4, "BR": 0.5, "JP": 0.2}
CATEGORY_RISK = {"electronics": 0.5, "gift_card": 0.8, "travel": 0.4,
                 "groceries": 0.05, "fuel": 0.1, "subscription": 0.2, "jewelry": 0.6}

FEATURE_NAMES = [
    "amount", "log_amount", "hour", "is_night", "is_weekend",
    "country_risk", "category_risk",
    "amt_zscore_user", "amt_vs_user_mean", "amt_max_ratio_user",
    "txn_count_1h", "txn_count_24h", "amt_sum_1h", "amt_sum_24h",
    "time_since_last_s", "velocity_1h", "distinct_merchants_24h",
    "distinct_countries_24h", "new_device", "device_count_user",
    "merchant_seen_before", "is_round_amount", "amount_gt_100",
    "amount_gt_500", "night_high_amount", "foreign_high_risk_cat",
    "burst_flag", "user_txn_rank",
]
assert len(FEATURE_NAMES) == 28


@dataclass
class UserState:
    amounts: deque = field(default_factory=lambda: deque(maxlen=200))
    times: deque = field(default_factory=lambda: deque(maxlen=200))
    merchants: deque = field(default_factory=lambda: deque(maxlen=200))
    countries: deque = field(default_factory=lambda: deque(maxlen=200))
    devices: set = field(default_factory=set)
    n: int = 0


class FeatureStore:
    """In-memory per-user state. Swap `.state` for Redis hashes in prod — the
    update/compute logic is identical."""
    def __init__(self):
        self.state: dict[str, UserState] = defaultdict(UserState)

    def compute(self, txn: dict) -> list[float]:
        s = self.state[txn["user_id"]]
        now = txn["ts"]
        amt = float(txn["amount"])

        # time-windowed aggregates from history (before adding current txn)
        amts_1h = [a for a, t in zip(s.amounts, s.times) if now - t <= 3600]
        amts_24h = [a for a, t in zip(s.amounts, s.times) if now - t <= 86400]
        merch_24h = {m for m, t in zip(s.merchants, s.times) if now - t <= 86400}
        ctry_24h = {c for c, t in zip(s.countries, s.times) if now - t <= 86400}

        mean = (sum(s.amounts) / len(s.amounts)) if s.amounts else amt
        std = (statistics_pstdev(s.amounts)) if len(s.amounts) > 1 else 1.0
        mx = max(s.amounts) if s.amounts else amt
        last_t = s.times[-1] if s.times else now
        hour = time.gmtime(now).tm_hour
        wday = time.gmtime(now).tm_wday

        feats = {
            "amount": amt,
            "log_amount": math.log1p(amt),
            "hour": float(hour),
            "is_night": 1.0 if (hour < 6 or hour >= 23) else 0.0,
            "is_weekend": 1.0 if wday >= 5 else 0.0,
            "country_risk": COUNTRY_RISK.get(txn["country"], 0.5),
            "category_risk": CATEGORY_RISK.get(txn["merchant_category"], 0.3),
            "amt_zscore_user": (amt - mean) / (std + 1e-6),
            "amt_vs_user_mean": amt / (mean + 1e-6),
            "amt_max_ratio_user": amt / (mx + 1e-6),
            "txn_count_1h": float(len(amts_1h)),
            "txn_count_24h": float(len(amts_24h)),
            "amt_sum_1h": float(sum(amts_1h)),
            "amt_sum_24h": float(sum(amts_24h)),
            "time_since_last_s": float(now - last_t),
            "velocity_1h": len(amts_1h) / 1.0,
            "distinct_merchants_24h": float(len(merch_24h)),
            "distinct_countries_24h": float(len(ctry_24h)),
            "new_device": 0.0 if txn["device_id"] in s.devices else 1.0,
            "device_count_user": float(len(s.devices)),
            "merchant_seen_before": 1.0 if txn["merchant"] in s.merchants else 0.0,
            "is_round_amount": 1.0 if abs(amt - round(amt, -1)) < 1e-6 else 0.0,
            "amount_gt_100": 1.0 if amt > 100 else 0.0,
            "amount_gt_500": 1.0 if amt > 500 else 0.0,
            "night_high_amount": 1.0 if (hour < 6 and amt > 200) else 0.0,
            "foreign_high_risk_cat": (COUNTRY_RISK.get(txn["country"], 0.5)
                                      * CATEGORY_RISK.get(txn["merchant_category"], 0.3)),
            "burst_flag": 1.0 if len(amts_1h) >= 5 else 0.0,
            "user_txn_rank": float(s.n),
        }
        return [feats[k] for k in FEATURE_NAMES]

    def update(self, txn: dict) -> None:
        s = self.state[txn["user_id"]]
        s.amounts.append(float(txn["amount"]))
        s.times.append(txn["ts"])
        s.merchants.append(txn["merchant"])
        s.countries.append(txn["country"])
        s.devices.add(txn["device_id"])
        s.n += 1

    def compute_and_update(self, txn: dict) -> list[float]:
        f = self.compute(txn)
        self.update(txn)
        return f


def statistics_pstdev(xs) -> float:
    xs = list(xs)
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
