#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhang/workspace/zdog-yizhu"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a \
    server/.artifacts/target162-feature-experiments-20260930.log
}

remote_complete() {
  local host="$1" path="$2"
  ssh -o BatchMode=yes "$host" \
    "python3 -c 'import json; print(json.load(open(\"$path\"))[\"stage\"])'" \
    2>/dev/null | grep -qx complete
}

wait_remote() {
  local host="$1" path="$2" label="$3"
  until remote_complete "$host" "$path"; do
    log "waiting: $label"
    sleep 120
  done
  log "complete: $label"
}

wait_local() {
  local path="$1" label="$2"
  until server/.venv/bin/python -c \
    "import json; print(json.load(open('$path'))['stage'])" 2>/dev/null |
    grep -qx complete; do
    log "waiting: $label"
    sleep 120
  done
  log "complete: $label"
}

HOST_ROOT="/home/zhang/workspace/zdog-yizhu-collection-20260930"
BCP_ROOT="/home/zhang/workspace/zdog-yizhu-training-20260924"
wait_remote zhang@host.942295.xyz \
  "$HOST_ROOT/server/.artifacts/features/target162-plantclef-local-v1/progress.json" \
  "PlantCLEF host local partition"
wait_remote zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.artifacts/features/target162-plantclef-bcp-v1/progress.json" \
  "PlantCLEF bcp partition"

mkdir -p server/.artifacts/features
rsync -a \
  zhang@host.942295.xyz:"$HOST_ROOT/server/.artifacts/features/target162-plantclef-host-v1/" \
  server/.artifacts/features/target162-plantclef-host-v1/
rsync -a \
  zhang@host.942295.xyz:"$HOST_ROOT/server/.artifacts/features/target162-plantclef-local-v1/" \
  server/.artifacts/features/target162-plantclef-local-v1/
rsync -a \
  zhang@bcp.942295.xyz:"$BCP_ROOT/server/.artifacts/features/target162-plantclef-bcp-v1/" \
  server/.artifacts/features/target162-plantclef-bcp-v1/
log "PlantCLEF partitions synchronized"

MERGED="server/.artifacts/features/target162-union-v1-plantclef-dinov2"
if [[ ! -f "$MERGED/index.json" ]]; then
  server/.venv/bin/python -m yizhu_server.merge_partition_features \
    --training server/.cache/training/target162-union-v1/train-validation.json \
    --test server/.cache/training/target162-union-v1/test-heldout.json \
    --partition server/.artifacts/features/target162-plantclef-host-v1 \
    --partition server/.artifacts/features/target162-plantclef-local-v1 \
    --partition server/.artifacts/features/target162-plantclef-bcp-v1 \
    --output "$MERGED"
fi
log "PlantCLEF partitions merged"

run_head() {
  local features="$1" architecture="$2" transform="$3" weighting="$4"
  local output="server/.artifacts/models/target162-${architecture}-${transform}-${weighting}-145-v1"
  if [[ -f "$output/results.json" ]]; then
    log "skip existing: $output"
    return
  fi
  server/.venv/bin/python -m yizhu_server.target_only_feature_training \
    --features "$features" \
    --classes server/.cache/training/target162-union-v1/train-validation.json \
    --output "$output" --epochs 200 --patience 25 --seed 20260930 --threads 2 \
    --feature-transform "$transform" --class-weighting "$weighting" \
    --label-smoothing .1 --dropout .1 --batch-size 256
  log "trained: $output"
}

run_head "$MERGED" plantclef none inverse
run_head "$MERGED" plantclef zscore inverse
run_head "$MERGED" plantclef zscore none
run_head "$MERGED" plantclef l2 inverse
run_head "$MERGED" plantclef center-l2 inverse

wait_local \
  server/.artifacts/features/target162-union-v1-bioclip2/progress.json \
  "BioCLIP 2 full feature extraction"
BIOCLIP="server/.artifacts/features/target162-union-v1-bioclip2"
run_head "$BIOCLIP" bioclip2 none inverse
run_head "$BIOCLIP" bioclip2 none none
run_head "$BIOCLIP" bioclip2 center-l2 inverse

server/.venv/bin/python - <<'PY'
import json
from pathlib import Path

rows = []
for path in Path("server/.artifacts/models").glob("target162-*-145-v1/results.json"):
  result = json.loads(path.read_text())
  metrics = result["metrics"]
  rows.append({
    "model": path.parent.name,
    "classCount": result["classCount"],
    "validationMacroTop1": metrics["validation"]["macroTop1"],
    "validationMacroTop5": metrics["validation"]["macroTop5"],
    "testMacroTop1": metrics["test"]["macroTop1"],
    "testMacroTop5": metrics["test"]["macroTop5"],
    "coverageGoalMet": result["coverageGoalMet"],
    "finalIndependentGoalMet": result["finalIndependentGoalMet"],
  })
rows.sort(key=lambda row: (
  row["validationMacroTop1"], row["validationMacroTop5"],
), reverse=True)
Path("server/.artifacts/target162-feature-experiment-summary.json").write_text(
  json.dumps({"status": "development-comparison", "models": rows},
             ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(rows, ensure_ascii=False))
PY
log "all frozen-feature development experiments complete"
