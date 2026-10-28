# Deploying the Fraud Scoring API to Fly.io (free, scale-to-zero)

This ships **Project 2's fraud model** as an HTTP API on one self-contained Fly
machine: light image (numpy + xgboost + fastapi, no torch). At boot it generates
a small dataset, trains the XGBoost model (~10s), and serves it. It sleeps when
idle (`min_machines_running = 0`) so it costs ~$0, and wakes on the next request.
Config lives in [`fly.toml`](fly.toml); the auto-deploy lives in
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml).

**Endpoints:** `GET /healthz`, `GET /score/{txn_id}`, `GET /explain/{txn_id}`,
`GET /investigate/{txn_id}`, `POST /score`, and interactive docs at `/docs`.

## One-time setup (~5 minutes)

### 1. Install flyctl and sign in
```bash
brew install flyctl          # macOS; or: curl -L https://fly.io/install.sh | sh
fly auth signup              # or: fly auth login   (free, no card for the hobby tier)
```

### 2. Create the app (the name must be globally unique)
```bash
cd ai-platforms
fly apps create your-unique-fraud-api
```
Then edit `fly.toml` and set `app = "your-unique-fraud-api"`. Pick a nearby
`primary_region` from `fly platform regions` (e.g. `iad`, `lhr`, `sjc`).

### 3. First deploy (from your laptop)
```bash
fly deploy
fly open            # opens https://your-unique-fraud-api.fly.dev/docs
```
Test it:
```bash
curl https://your-unique-fraud-api.fly.dev/healthz
# {"ok":true,"model_loaded":true,"hint":null}

# score an arbitrary transaction:
curl -X POST https://your-unique-fraud-api.fly.dev/score \
  -H 'Content-Type: application/json' \
  -d '{"amount": 1450, "merchant_category": "gift_card", "country": "BR"}'
# {"fraud_probability": ..., "decision": "FLAG", "features_used": {...}}
```

## Auto-deploy on every push (GitHub Action)

The workflow deploys **only after CI passes on `main`** (via `workflow_run`), so
a red build never ships.

1. Create a deploy token:
   ```bash
   fly tokens create deploy -x 999999h
   ```
2. In GitHub: **repo → Settings → Secrets and variables → Actions → New
   repository secret**
   - Name: `FLY_API_TOKEN`
   - Value: the token from step 1
3. Push to `main`. Watch **Actions → CI** go green, then **Deploy Fraud API
   (Fly.io)** run automatically. You can also trigger it manually from the
   Actions tab (**Run workflow**).

## How it runs on Fly

At boot the machine runs `generate → train → uvicorn :8000` (Fly fronts it with
HTTPS on 443). The filesystem is ephemeral, so it regenerates+retrains on each
restart (~10s — the health check's `grace_period` covers it). For a persistent,
pre-trained model, bake `make p2-data && make p2-fraud` artifacts into the image
in the `Dockerfile` instead of training at boot.

## Other hosts

The same root `Dockerfile` deploys anywhere that runs a container — **Render**
(free web service), **Railway**, or an **Oracle Cloud Free** VM. See
[SETUP.md](SETUP.md) § 6.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `Error: app name is not unique` | pick another `app` name; Fly app names are global |
| deploy OK but 503 for ~10s | the model is training at boot; the health-check grace period covers it — wait |
| `/score/{id}` returns 404 | that txn id isn't in the freshly generated dataset; use `POST /score` with a body, or copy an id from `GET /docs` examples |
| Action fails: `No access token available` | add the `FLY_API_TOKEN` repo secret (setup step 2) |
| want it always-on (no cold start) | set `min_machines_running = 1` in `fly.toml` (leaves the free tier) |
