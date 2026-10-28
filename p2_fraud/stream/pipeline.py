"""Real-time feature pipeline: raw txns -> 28 features -> fraud score + feature
store, measuring per-event latency (the "sub-second feature latency" claim).

Two execution engines behind one interface:

  * bytewax  — the production dataflow (Flink-equivalent, pure-Python, scales
               across workers). Used automatically if bytewax is installed.
  * simple   — a single-process streaming loop. Identical FeatureStore + output,
               used for local runs and anywhere bytewax isn't available.

Source: kafka | replay(jsonl). Sink: redis feature store | memory.

  python -m p2_fraud.stream.pipeline --source replay --limit 20000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from shared import config
from ..features import FeatureStore, FEATURE_NAMES


class RedisFeatureSink:
    def __init__(self):
        import redis
        self.r = redis.from_url(config.REDIS_URL)

    def put(self, txn_id: str, feats: list[float]) -> None:
        self.r.hset(f"feat:{txn_id}", mapping={n: v for n, v in zip(FEATURE_NAMES, feats)})
        self.r.expire(f"feat:{txn_id}", 3600)


class MemoryFeatureSink:
    def __init__(self): self.store = {}
    def put(self, txn_id, feats): self.store[txn_id] = feats


def load_fraud_model():
    p = Path("p2_fraud/fraud/artifacts/fraud_xgb.json")
    if not p.exists():
        return None
    import xgboost as xgb
    m = xgb.XGBClassifier()
    m.load_model(str(p))
    return m


def iter_txns(source: str, path: str, limit: int):
    if source == "replay":
        lines = Path(path).read_text().splitlines()
        if limit:
            lines = lines[:limit]
        for line in lines:
            yield json.loads(line)
    else:  # kafka
        from confluent_kafka import Consumer
        c = Consumer({"bootstrap.servers": config.KAFKA_BROKERS,
                      "group.id": "feature-pipeline", "auto.offset.reset": "earliest"})
        c.subscribe([config.TOPIC_TXNS])
        try:
            while True:
                m = c.poll(1.0)
                if m and not m.error():
                    yield json.loads(m.value())
        finally:
            c.close()


def run_simple(args):
    store = FeatureStore()
    sink = RedisFeatureSink() if args.sink == "redis" else MemoryFeatureSink()
    model = load_fraud_model()
    lat = []
    n = flagged = 0
    for txn in iter_txns(args.source, args.path, args.limit):
        t0 = time.perf_counter()
        feats = store.compute_and_update(txn)
        if model is not None:
            score = float(model.predict_proba(np.asarray([feats], dtype="float32"))[0, 1])
            if score > 0.5:
                flagged += 1
        sink.put(txn.get("txn_id", str(n)), feats)
        lat.append((time.perf_counter() - t0) * 1000)
        n += 1
    lat = np.array(lat)
    print(f"engine=simple  processed={n}  flagged_fraud={flagged}")
    print(f"feature latency ms: p50={np.percentile(lat,50):.3f} "
          f"p99={np.percentile(lat,99):.3f} max={lat.max():.3f}")
    print(f"throughput: {n/ (lat.sum()/1000):.0f} txns/s single-process")
    return n


def run_bytewax(args):
    """Production dataflow. Requires `pip install bytewax`."""
    from bytewax.dataflow import Dataflow
    import bytewax.operators as op
    from bytewax.connectors.stdio import StdOutSink
    from bytewax.testing import TestingSource

    store = FeatureStore()
    model = load_fraud_model()
    txns = list(iter_txns(args.source, args.path, args.limit))

    def featurize(txn):
        feats = store.compute_and_update(txn)
        score = (float(model.predict_proba(np.asarray([feats], dtype="float32"))[0, 1])
                 if model is not None else None)
        return f"{txn.get('txn_id','?')} fraud_score={score}"

    flow = Dataflow("fraud-features")
    stream = op.input("in", flow, TestingSource(txns))
    scored = op.map("featurize", stream, featurize)
    op.output("out", scored, StdOutSink())

    from bytewax.run import cli_main
    cli_main(flow)                        # blocks; run via `python -m bytewax.run`


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="auto", choices=["auto", "simple", "bytewax"])
    ap.add_argument("--source", default="replay", choices=["replay", "kafka"])
    ap.add_argument("--sink", default="memory", choices=["memory", "redis"])
    ap.add_argument("--path", default="p2_fraud/data/txns.jsonl")
    ap.add_argument("--limit", type=int, default=20000)
    args = ap.parse_args()

    if args.engine == "bytewax" or (args.engine == "auto" and _has_bytewax()):
        try:
            return run_bytewax(args)
        except Exception as e:  # noqa: BLE001
            print(f"[stream] bytewax path failed ({e}); using simple engine")
    run_simple(args)


def _has_bytewax() -> bool:
    try:
        import bytewax  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
