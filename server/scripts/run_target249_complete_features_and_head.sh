#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhang/workspace/zdog-yizhu"
cd "$ROOT"
BCP_ROOT="/home/zhang/workspace/zdog-yizhu-training-20260924"
LOG="server/.artifacts/target249-complete-training-20261001.log"
PLAN="server/.artifacts/features/target249-complete-bioclip2-plan-20261001"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"
}

remote_stage() {
  local path="$1"
  ssh -o BatchMode=yes zhang@bcp.942295.xyz \
    "python3 -c 'import json; print(json.load(open(\"$path\"))[\"stage\"])'" \
    2>/dev/null
}

until [[ -f server/.artifacts/models/target249-base242-bioclip2-center-l2-inverse-v1/results.json ]]; do
  log "waiting: base 242-class feature reuse and development head"
  sleep 60
done

SUPPLEMENT="$BCP_ROOT/server/.artifacts/features/target249-complete-supplement-bioclip2"
if [[ "$(remote_stage "$SUPPLEMENT/progress.json" || true)" != complete ]]; then
  ssh -o BatchMode=yes zhang@bcp.942295.xyz \
    "systemd-run --user --unit=yizhu-target249-complete-supplement-bioclip \
      --property=MemoryMax=8G --property=CPUQuota=300% \
      --setenv=PYTHONPATH=$BCP_ROOT/server/src \
      $BCP_ROOT/server/.venv/bin/python -m yizhu_server.bioclip_partition_features \
      --partition $BCP_ROOT/$PLAN/supplement-partition-bcp.json \
      --output $SUPPLEMENT --threads 3 --batch-size 6"
fi
until [[ "$(remote_stage "$SUPPLEMENT/progress.json" || true)" == complete ]]; do
  log "waiting: 67 admitted-candidate BioCLIP features"
  sleep 60
done
rsync -a zhang@bcp.942295.xyz:"$SUPPLEMENT/" \
  server/.artifacts/features/target249-complete-supplement-bioclip2/
log "admitted-candidate feature supplement synchronized"

MERGED="server/.artifacts/features/target249-complete-bioclip2-replay324"
if [[ ! -e "$MERGED" ]]; then
  PART0="server/.artifacts/features/target249-base242-part0-bioclip2"
  if [[ -f server/.artifacts/features/target249-base242-part0-bioclip2-fast/index.json ]]; then
    PART0="server/.artifacts/features/target249-base242-part0-bioclip2-fast"
  fi
  server/.venv/bin/python -m yizhu_server.target_feature_expansion merge \
    --plan "$PLAN/plan.json" \
    --reused server/.artifacts/features/target162-union-v3-bioclip2-replay324 \
    --partition "$PART0" \
    --partition server/.artifacts/features/target249-base242-part1-bioclip2 \
    --partition server/.artifacts/features/target249-base242-part2-bioclip2 \
    --partition server/.artifacts/features/target249-complete-supplement-bioclip2 \
    --output "$MERGED"
fi
log "complete 249-class features merged with 324 replay rows"

run_head() {
  local transform="$1" weighting="$2" initialization="${3:-}" suffix="${4:-}"
  local output="server/.artifacts/models/target249-complete-bioclip2-${transform}-${weighting}${suffix}-v1"
  local initialization_args=()
  local learning_rate=.001 label_smoothing=.1 dropout=.1
  if [[ -n "$initialization" ]]; then
    initialization_args=(--initialization "$initialization")
    # Preserve useful zero-shot geometry for one/few-shot classes instead of
    # immediately overwriting text prototypes with the standard random-head rate.
    learning_rate=.0003
    label_smoothing=.05
    dropout=.05
  fi
  if [[ ! -e "$output" ]]; then
    ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    server/.venv/bin/python -m yizhu_server.target_only_feature_training \
      --features "$MERGED" \
      --classes server/.cache/training/target249-union-v2-complete-20261001/train-validation.json \
      --output "$output" --epochs 250 --patience 30 --seed 20261001 --threads 1 \
      --feature-transform "$transform" --class-weighting "$weighting" \
      --learning-rate "$learning_rate" --label-smoothing "$label_smoothing" \
      --dropout "$dropout" --batch-size 256 \
      "${initialization_args[@]}"
  fi
  if [[ ! -f "$output/model.metadata.json" ]]; then
    server/.venv/bin/python -m yizhu_server.target_model_metadata \
      --model "$output" --features "$MERGED" \
      --snapshot server/.cache/training/target249-union-v2-complete-20261001 \
      --solution-id "target249-bioclip2-${transform}-${weighting}${suffix}-v1"
  fi
  log "trained: $output"
}

run_head center-l2 inverse
run_head center-l2 none
TEXT_INIT="server/.artifacts/features/target249-bioclip2-text-init-20261001"
until [[ -f "$TEXT_INIT/index.json" ]]; do
  log "waiting: BioCLIP 2 scientific-name text initialization"
  sleep 60
done
run_head l2 inverse "$TEXT_INIT" "-text"
run_head l2 none "$TEXT_INIT" "-text"
log "complete 249-class development heads trained; independent acceptance still required"
