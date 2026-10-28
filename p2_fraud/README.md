# Project 2 — Real-Time Fraud Intelligence & Personalization

One streaming feature pipeline computes 28 features per transaction and feeds
**two** models from the *same* features (no train/serve skew): an **XGBoost fraud
classifier** and a **LinUCB contextual-bandit recommender**. Flagged transactions
are triaged by a **Claude agent** that reasons over the model through an **MCP
server**.

```
 txns ──▶ Kafka/Redpanda ──▶ stream features (28/txn) ──▶ Redis feature store
                                     │                        │
                                     ▼                        ▼
                          XGBoost fraud model          LinUCB recommender ──▶ CTR
                                     │                  (PSI drift → retrain)
                                     ▼
                     agentic investigator (Claude + MCP tools, ReAct loop)
```

---

## Why it's built this way

| Choice | Why |
|---|---|
| **One shared `features.py`** | The batch trainer and the live stream import the *same* feature code, so a feature can't be computed one way in training and another in serving. This is what "single workflow serving both models" means in practice. |
| **XGBoost** for fraud | Gradient-boosted trees are the industry default for tabular fraud: strong on heterogeneous features, handles class imbalance via `scale_pos_weight`, gives feature importances for explainability, trains in seconds on CPU. |
| **LinUCB** for recs | A contextual bandit with closed-form updates — no GPU, <1 ms/decision, naturally balances explore/exploit, and updates online from every click. |
| **Bytewax** over Flink | Flink needs a JVM cluster (heavy on 16 GB). Bytewax is pip-installable pure Python with the same windowed-dataflow model; the code auto-falls back to a single-process loop if Bytewax isn't installed. |
| **Agent + MCP** | The model becomes a *tool*; the agent does the multi-step reasoning (score → explain → corroborate with history → decide → log). The MCP server makes those tools usable from Claude Desktop directly. |

---

## The ML, in detail

### 1. Features (`features.py`) — the heart of the system
28 features per transaction, grouped as:
- **Amount shape:** `amount`, `log_amount`, round-number flag, `>100`/`>500` flags.
- **Time:** hour, `is_night`, `is_weekend`.
- **Risk priors:** per-country and per-merchant-category risk, and their product.
- **User-relative (needs history):** z-score and ratio of this amount vs the
  user's mean/max — *this is the strongest signal* (a $1,200 charge is normal for
  one user, a red flag for another).
- **Velocity (windowed):** txn counts and amount sums over the last 1 h / 24 h,
  time since last txn, distinct merchants/countries in 24 h, burst flag.
- **Device:** new-device flag, device count, merchant-seen-before.

Per-user state (recent amounts/times/merchants/devices) lives in an in-memory
store locally and in **Redis hashes** in production — identical update/compute
logic either way. The order matters: features are computed from history *before*
the current txn is folded in (no leakage).

### 2. Fraud model (`fraud/train.py`)
Streams the time-sorted transactions through the shared `FeatureStore`, then
trains XGBoost with a **time-aware split** (first 80% train, last 20% test — never
shuffle fraud data, that leaks the future). Key settings: 400 trees, depth 6,
lr 0.1, `eval_metric=aucpr`, `scale_pos_weight = negatives/positives` to handle
the ~3–9% fraud rate.

```bash
python -m p2_fraud.data.generate --n 60000      # synthetic txns with fraud patterns
python -m p2_fraud.fraud.train                  # → AUC, PR-AUC, feature importances
```
**Reading the output:** **AUC** = ranking quality (does fraud score higher than
legit?); **PR-AUC** = the one to watch under class imbalance (precision vs recall
on the rare positive class). The report also prints the top features — expect
`amount` and `amt_vs_user_mean` to dominate, which is exactly how real fraud
models behave.

> The synthetic fraud (account takeover, card-testing bursts, high-value gift
> cards abroad, odd-hour spikes) is injected so the engineered features carry
> genuine signal, with ~28% "subtle" fraud and ~5% anomalous-but-legit txns so
> the classes **overlap**. That's why AUC lands ~0.99 rather than a fake 1.0.

### 3. Recommender (`bandit/linucb.py`)
**LinUCB (disjoint).** For each arm *a* it keeps `A_a` (dxd) and `b_a`; the score
for context *x* is `θ_aᵀx + α·√(xᵀA_a⁻¹x)` where `θ_a = A_a⁻¹b_a`. The first term
is the expected reward, the second is an **upper confidence bound** that drives
exploration of under-tried arms. After observing reward *r*: `A_a += xxᵀ`,
`b_a += r·x`. α=0.6 trades exploration vs exploitation.

The simulation compares LinUCB against a **fair epsilon-greedy baseline** (one
that genuinely learns each arm's average reward but ignores context) — so the
reported uplift is the value of *using context*, not a rigged comparison.

**PSI drift detection.** Population Stability Index between an early and a late
window of the context stream: `PSI = Σ (aᵢ−eᵢ)·ln(aᵢ/eᵢ)` over deciles. The
convention: `>0.25` = significant drift → trigger an Airflow/cron retrain.

```bash
python -m p2_fraud.bandit.linucb
#   linucb_ctr 0.69 | baseline_ctr 0.52 | ctr_uplift_pct ~31 | context_psi 1.9 | drift_alert True
```

### 4. Agentic investigator (`investigator/`)
The model is exposed as **tools** (`tools.py`): `get_transaction`,
`get_user_history`, `score_transaction`, `explain_score`, `flag_transaction`.
Two ways to use them:

- **MCP server** (`mcp_server.py`) — run it and point Claude Desktop at it; Claude
  calls the tools itself.
- **ReAct agent** (`agent.py`) — a loop that investigates a transaction:
  score → explain → pull user history → decide FLAG/ALLOW → log. Runs on the
  Claude API when a key is set, else a deterministic local ReAct loop (so it's
  always runnable and testable). `explain_score` approximates per-feature
  contribution as importance × normalized feature value (SHAP is an optional dep
  for exact values).

```bash
make p2-fraud                                    # train the model first
python -m p2_fraud.investigator.agent --txn <id> # investigate one case, verbose
python -m p2_fraud.investigator.agent --sample 300   # measured accuracy vs ground truth
python -m p2_fraud.investigator.mcp_server       # expose tools to Claude Desktop
```
The sample eval **measures** decision accuracy against the ground-truth
`is_fraud` label and reports avg tool-calls/case vs a modeled manual baseline.

**Claude Desktop MCP config** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "fraud-investigator": {
      "command": "/ABSOLUTE/PATH/ai-platforms/.venv/bin/python",
      "args": ["-m", "p2_fraud.investigator.mcp_server"],
      "cwd": "/ABSOLUTE/PATH/ai-platforms"
    }
  }
}
```

### 5. Streaming (`stream/pipeline.py`)
Raw txns → 28 features → fraud score + feature-store write, measuring **per-event
latency**. Two engines behind one interface: **Bytewax** (production dataflow) and
a single-process **simple** loop (auto-fallback). Sinks: Redis feature store or
memory.

```bash
python -m p2_fraud.stream.pipeline --source replay --limit 20000
#   feature latency ms: p50 ~0.6  p99 ~1.9     (well under the sub-second target)
```

---

### 6. Fraud-scoring API (`serve/app.py`)
The deployable face of the platform: the trained XGBoost model served over HTTP,
reusing the same scoring/explanation code as the investigator tools.

```bash
make p2-data && make p2-fraud    # produce data + model first
make p2-serve                    # uvicorn on :8000 — open /docs for the OpenAPI UI
```
Endpoints: `GET /healthz`, `GET /score/{txn_id}`, `GET /explain/{txn_id}`,
`GET /investigate/{txn_id}` (full ReAct investigation), `POST /score` (score an
arbitrary transaction body). This is what the Fly.io deploy ships — see
[DEPLOY.md](../DEPLOY.md).

---

## Run it

```bash
make p2-demo                     # everything offline, ~40s
# or piece by piece:
make up-fraud                    # redpanda + redis (docker)
make p2-data                     # 500k txns (start smaller: --n 100000)
make p2-fraud                    # train XGBoost
make p2-stream                   # feature latency report
make p2-bandit                   # CTR uplift + PSI drift
make p2-investigator             # agent sample eval
```

## How to read every metric
| Metric | What it means | What's "good" |
|---|---|---|
| AUC | ranking quality, fraud vs legit | high; ~0.99 on this synthetic data |
| PR-AUC | precision/recall on the rare fraud class | the honest imbalance metric |
| feature latency p50/p99 | per-txn compute+score time | sub-ms here, well under 1 s |
| ctr_uplift_pct | LinUCB CTR vs fair eps-greedy | positive (~30% here) |
| context_psi | distribution drift in the context stream | `>0.25` ⇒ retrain |
| investigator accuracy | decisions vs ground-truth `is_fraud` | inherits the model's accuracy |

## From synthetic to real
1. Replace `data/generate.py` with your real transaction stream (same fields as
   `shared.schemas.Transaction`).
2. Retrain: `make p2-fraud`. The time-aware split and `scale_pos_weight` already
   handle real imbalance.
3. The recommender arms = your real items/offers; the context = the 28 features
   (or a subset). Wire clicks back as rewards.
4. The investigator and MCP server need no change — they read whatever model and
   data are present.

## File guide
| File | Role |
|---|---|
| `features.py` | 28 features, shared by training + serving |
| `data/generate.py` | synthetic txns with 4 fraud patterns + overlap |
| `fraud/train.py` | XGBoost trainer, time-aware split, metrics |
| `bandit/linucb.py` | LinUCB + fair baseline + PSI drift |
| `stream/pipeline.py` | real-time features (Bytewax + simple engine) |
| `investigator/tools.py` | the 5 investigation tools |
| `investigator/mcp_server.py` | MCP server exposing the tools |
| `investigator/agent.py` | ReAct agent (Claude + local fallback) + eval |
| `demo.py` | the whole thing in one command |
