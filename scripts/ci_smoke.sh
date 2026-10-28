#!/usr/bin/env bash
# Build the minimal artifacts so the full test suite runs (nothing skips).
# Used by CI and reproducible locally:  bash scripts/ci_smoke.sh
set -euo pipefail

echo "::group::tiny data + fraud model"
python -m p2_fraud.data.generate --n 8000
python -m p2_fraud.fraud.train
echo "::endgroup::"

echo "smoke artifacts ready"
