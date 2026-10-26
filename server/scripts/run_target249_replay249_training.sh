#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhang/workspace/zdog-yizhu"
cd "$ROOT"
HOST_ROOT="/home/zhang/workspace/zdog-yizhu-collection-20260930"
BCP_ROOT="/home/zhang/workspace/zdog-yizhu-training-20260924"
LOG="server/.artifacts/target249-replay249-training-20261001.log"
PLAN="server/.artifacts/features/target249-replay249-bioclip2-plan-20261001"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"
}

stage() {
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage"])' "$1" \
    2>/dev/null
}

remote_stage() {
  ssh -o BatchMode=yes "$1" \
    "python3 -c 'import json; print(json.load(open(\"$2\"))[\"stage\"])'" 2>/dev/null
}

until [[ "$(stage server/.artifacts/features/target249-replay249-part0-bioclip2/progress.json || true)" == complete ]]; do
  log "waiting: replay partition 0"
  sleep 30
done
until [[ "$(remote_stage zhang@host.942295.xyz "$HOST_ROOT/server/.artifacts/features/target249-replay249-part1-bioclip2/progress.json" || true)" == complete ]]; do
  log "waiting: replay partition 1"
  sleep 30
done
until [[ "$(remote_stage zhang@bcp.942295.xyz "$BCP_ROOT/server/.artifacts/features/target249-replay249-part2-bioclip2/progress.json" || true)" == complete ]]; do
  log "waiting: replay partition 2"
  sleep 30
done

rsync -a zhang@host.942295.xyz:"$HOST_ROOT/server/.artifacts/features/target249-replay249-part1-bioclip2/" \
  server/.artifacts/features/target249-replay249-part1-bioclip2/
rsync -a zhang@bcp.942295.xyz:"$BCP_ROOT/server/.artifacts/features/target249-replay249-part2-bioclip2/" \
  server/.artifacts/features/target249-replay249-part2-bioclip2/

MERGED="server/.artifacts/features/target249-replay249-bioclip2"
if [[ ! -e "$MERGED" ]]; then
  server/.venv/bin/python -m yizhu_server.target_feature_expansion merge \
    --plan "$PLAN/plan.json" \
    --reused server/.artifacts/features/target249-complete-bioclip2-replay324 \
    --partition server/.artifacts/features/target249-replay249-part0-bioclip2 \
    --partition server/.artifacts/features/target249-replay249-part1-bioclip2 \
    --partition server/.artifacts/features/target249-replay249-part2-bioclip2 \
    --output "$MERGED"
fi
log "249 failed-independent images appended as balanced replay training rows"

run_head() {
  local weighting="$1"
  local output="server/.artifacts/models/target249-replay249-center-l2-${weighting}-v1"
  if [[ ! -e "$output" ]]; then
    ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    server/.venv/bin/python -m yizhu_server.target_only_feature_training \
      --features "$MERGED" \
      --classes server/.cache/training/target249-union-v2-complete-20261001/train-validation.json \
      --output "$output" --epochs 250 --patience 30 --seed 20261001 --threads 1 \
      --feature-transform center-l2 --class-weighting "$weighting" \
      --learning-rate .001 --label-smoothing .1 --dropout .1 --batch-size 256
  fi
  if [[ ! -f "$output/model.metadata.json" ]]; then
    server/.venv/bin/python -m yizhu_server.target_model_metadata \
      --model "$output" --features "$MERGED" \
      --snapshot server/.cache/training/target249-union-v2-complete-20261001 \
      --solution-id "target249-replay249-center-l2-${weighting}-v1"
  fi
}

run_head inverse
run_head none

server/.venv/bin/python - <<'PY'
import json
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json

models = [
    Path("server/.artifacts/models/target249-replay249-center-l2-inverse-v1"),
    Path("server/.artifacts/models/target249-replay249-center-l2-none-v1"),
]
rows = []
for model in models:
    result = json.loads((model / "results.json").read_text())
    validation = result["metrics"]["validation"]
    rows.append({
        "model": str(model),
        "metadataSha256": sha256(model / "model.metadata.json"),
        "validationMacroTop1": validation["macroTop1"],
        "validationMacroTop5": validation["macroTop5"],
        "validationTop1": validation["top1"], "validationTop5": validation["top5"],
    })
rows.sort(key=lambda row: (
    row["validationMacroTop1"], row["validationMacroTop5"],
    row["validationTop1"], row["validationTop5"],
), reverse=True)
write_json(Path("server/.artifacts/target249-replay249-selection-20261001.json"), {
    "createdAt": now(), "status": "frozen-before-second-independent-evaluation",
    "selectionRule": "original development validation macro Top-1; tie Top-5",
    "candidates": rows, "selectedModel": rows[0]["model"],
    "failedIndependentResultsUsedOnlyAsReplayData": True,
    "secondIndependentOutputsReadForSelection": False,
    "productionChanged": False,
})
PY
log "replay249 candidates trained and frozen; production unchanged"
