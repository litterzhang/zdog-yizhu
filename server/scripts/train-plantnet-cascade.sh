#!/usr/bin/env bash
# Offline training and evaluation only. No downloads, API restarts or model activation.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TORCH_FORCE_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
data="${1:?Usage: bash server/scripts/train-plantnet-cascade.sh FROZEN_DATA NEW_MODEL [--allow-partial]}"
model="${2:?Use a new model output directory}"
shift 2
server/.venv/bin/python -u -m yizhu_server.plantnet_transfer train \
  --data "$data" --model "$model" "$@"
server/.venv/bin/python -u -m yizhu_server.plantnet_transfer evaluate \
  --data "$data" --model "$model" --output "$model/test.json"
