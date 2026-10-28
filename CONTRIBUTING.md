# Contributing

Thanks for taking a look! See the [README](README.md) for what this is.

## Dev setup
```bash
make venv && source .venv/bin/activate
pip install -r requirements-ci.txt -r requirements-dev.txt
bash scripts/ci_smoke.sh      # builds tiny data + model so all tests run
make test
```

## Conventions
- **Python 3.11+**, linted with [ruff](https://docs.astral.sh/ruff/) (`ruff check .`).
- **Training and serving share `p2_fraud/features.py`** — never compute a feature
  one way in training and another at serving. Keep them in that one module.
- Keep the **light fallbacks working** — the pipeline must still run without
  Bytewax, Docker, or API keys (see `p2_fraud/stream/pipeline.py`,
  `p2_fraud/investigator/agent.py`).
- New behavior needs a fast test under `p2_fraud/tests/` (no network, no Docker).

## PRs
Branch off `main`, keep it focused, make sure `make test` passes (CI runs it on
3.11 and 3.12).
