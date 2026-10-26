#!/usr/bin/env bash
# One resumable public-data job, separate from the running Flask service.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TORCH_FORCE_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

output="${1:?Usage: bash server/scripts/train-common-plants.sh NEW_MODEL_DIRECTORY}"
if [[ -e "$output" ]]; then
  echo "Model directory exists; choose a new version." >&2
  exit 1
fi
server/.venv/bin/python -m yizhu_server.training_data collect
# A class lacking licensed train/validation/test data is omitted and reported.
# This never pretends that a smaller model is a 1,000-class classifier.
server/.venv/bin/python -m yizhu_server.training_classifier train \
  --allow-partial --output "$output"
