#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhang/workspace/zdog-yizhu"
cd "$ROOT"
HOST_ROOT="/home/zhang/workspace/zdog-yizhu-collection-20260930"
BCP_ROOT="/home/zhang/workspace/zdog-yizhu-training-20260924"
LOG="server/.artifacts/target249-base242-training-20261001.log"
PLAN="server/.artifacts/features/target249-base242-bioclip2-plan-20261001"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"
}

stage() {
  local path="$1"
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage"])' "$path" \
    2>/dev/null
}

remote_stage() {
  local host="$1" path="$2"
  ssh -o BatchMode=yes "$host" \
    "python3 -c 'import json; print(json.load(open(\"$path\"))[\"stage\"])'" \
    2>/dev/null
}

LOCAL_PART0="server/.artifacts/features/target249-base242-part0-bioclip2"
FAST_PART0="$HOST_ROOT/server/.artifacts/features/target249-base242-part0-bioclip2-fast"
while true; do
  local_part0_stage="$(stage "$LOCAL_PART0/progress.json" || true)"
  fast_part0_stage="$(
    remote_stage zhang@host.942295.xyz "$FAST_PART0/progress.json" || true
  )"
  if [[ "$local_part0_stage" == complete || "$fast_part0_stage" == complete ]]; then
    break
  fi
  log "waiting: local or host-fast BioCLIP partition 0"
  sleep 60
done
PART0="$LOCAL_PART0"
if [[ "$(stage "$LOCAL_PART0/progress.json" || true)" != complete ]]; then
  PART0="server/.artifacts/features/target249-base242-part0-bioclip2-fast"
  rsync -a zhang@host.942295.xyz:"$FAST_PART0/" "$PART0/"
  log "using host-fast BioCLIP partition 0"
fi
until [[ "$(remote_stage zhang@host.942295.xyz \
  "$HOST_ROOT/server/.artifacts/features/target249-base242-part1-bioclip2/progress.json" || true)" == complete ]]; do
  log "waiting: host BioCLIP partition 1"
  sleep 60
done
until [[ "$(remote_stage zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.artifacts/features/target249-base242-part2-bioclip2/progress.json" || true)" == complete ]]; do
  log "waiting: bcp BioCLIP partition 2"
  sleep 60
done

rsync -a zhang@host.942295.xyz:"$HOST_ROOT/server/.artifacts/features/target249-base242-part1-bioclip2/" \
  server/.artifacts/features/target249-base242-part1-bioclip2/
rsync -a zhang@bcp.942295.xyz:"$BCP_ROOT/server/.artifacts/features/target249-base242-part2-bioclip2/" \
  server/.artifacts/features/target249-base242-part2-bioclip2/
log "all three BioCLIP partitions synchronized"

MERGED="server/.artifacts/features/target249-base242-bioclip2-replay324"
if [[ ! -e "$MERGED" ]]; then
  server/.venv/bin/python -m yizhu_server.target_feature_expansion merge \
    --plan "$PLAN/plan.json" \
    --reused server/.artifacts/features/target162-union-v3-bioclip2-replay324 \
    --partition "$PART0" \
    --partition server/.artifacts/features/target249-base242-part1-bioclip2 \
    --partition server/.artifacts/features/target249-base242-part2-bioclip2 \
    --output "$MERGED"
fi
log "base 242-class features merged with 324 replay rows"

MODEL="server/.artifacts/models/target249-base242-bioclip2-center-l2-inverse-v1"
if [[ ! -e "$MODEL" ]]; then
  ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  server/.venv/bin/python -m yizhu_server.target_only_feature_training \
    --features "$MERGED" \
    --classes server/.cache/training/target249-union-v1-20261001/train-validation.json \
    --output "$MODEL" --epochs 250 --patience 30 --seed 20261001 --threads 1 \
    --feature-transform center-l2 --class-weighting inverse \
    --label-smoothing .1 --dropout .1 --batch-size 256
fi
log "base 242/249 development head trained; no production change"
