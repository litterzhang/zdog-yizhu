#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhang/workspace/zdog-yizhu"
cd "$ROOT"
LOG="server/.artifacts/target162-full-coverage-20260930.log"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"
}

remote_stage() {
  local host="$1" path="$2"
  ssh -o BatchMode=yes "$host" \
    "python3 -c 'import json; print(json.load(open(\"$path\"))[\"stage\"])'" \
    2>/dev/null
}

wait_remote_collection() {
  local host="$1" path="$2" label="$3"
  until [[ "$(remote_stage "$host" "$path" || true)" == "collection-pass-finished" ]]; do
    log "waiting: $label"
    sleep 120
  done
  log "complete: $label"
}

wait_remote_features() {
  local host="$1" path="$2" label="$3"
  until [[ "$(remote_stage "$host" "$path" || true)" == "complete" ]]; do
    log "waiting: $label"
    sleep 120
  done
  log "complete: $label"
}

HOST_ROOT="/home/zhang/workspace/zdog-yizhu-collection-20260930"
BCP_ROOT="/home/zhang/workspace/zdog-yizhu-training-20260924"
GBIF="server/.cache/training/target162-gbif-final-20260930"
SNAPSHOT="server/.cache/training/target162-union-v2-gbif"
if [[ ! -f "$SNAPSHOT/snapshot.json" ]]; then
  wait_remote_collection zhang@host.942295.xyz \
    "$HOST_ROOT/server/.cache/training/target162-gbif-part1b-20260930/progress.json" \
    "GBIF part1b"
  wait_remote_collection zhang@host.942295.xyz \
    "$HOST_ROOT/server/.cache/training/target162-gbif-buxus-20260930/progress.json" \
    "GBIF Buxus"
  wait_remote_collection zhang@bcp.942295.xyz \
    "$BCP_ROOT/server/.cache/training/target162-gbif-part2b-20260930/progress.json" \
    "GBIF part2b"
  wait_remote_collection zhang@bcp.942295.xyz \
    "$BCP_ROOT/server/.cache/training/target162-gbif-synonyms-20260930/progress.json" \
    "GBIF accepted synonyms"
  mkdir -p "$GBIF/part1b" "$GBIF/buxus" "$GBIF/part2b" "$GBIF/synonyms"
  rsync -a zhang@host.942295.xyz:"$HOST_ROOT/server/.cache/training/target162-gbif-part1b-20260930/" "$GBIF/part1b/"
  rsync -a zhang@host.942295.xyz:"$HOST_ROOT/server/.cache/training/target162-gbif-buxus-20260930/" "$GBIF/buxus/"
  rsync -a zhang@bcp.942295.xyz:"$BCP_ROOT/server/.cache/training/target162-gbif-part2b-20260930/" "$GBIF/part2b/"
  rsync -a zhang@bcp.942295.xyz:"$BCP_ROOT/server/.cache/training/target162-gbif-synonyms-20260930/" "$GBIF/synonyms/"
  log "GBIF candidate collections synchronized"
  server/.venv/bin/python -m yizhu_server.target162_gbif_snapshot \
    --base server/.cache/training/target162-union-v1 \
    --collection "$GBIF/part1b" --collection "$GBIF/buxus" \
    --collection "$GBIF/part2b" --collection "$GBIF/synonyms" \
    --output "$SNAPSHOT"
fi
log "full 162-class snapshot built"

BASE_FEATURES="server/.artifacts/features/target162-union-v1-plantclef-dinov2"
until [[ -f "$BASE_FEATURES/index.json" ]]; do
  log "waiting: merged base PlantCLEF features"
  sleep 120
done

SUPPLEMENT_PLAN="server/.artifacts/features/target162-v2-plantclef-supplement-plan"
mkdir -p "$SUPPLEMENT_PLAN"
server/.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("/home/zhang/workspace/zdog-yizhu")
v1 = json.loads((root / "server/.artifacts/features/target162-union-v1-plantclef-dinov2/index.json").read_text())
v2root = root / "server/.cache/training/target162-union-v2-gbif"
training = json.loads((v2root / "train-validation.json").read_text())
test = json.loads((v2root / "test-heldout.json").read_text())
known = {row["sampleId"] for row in v1["samples"]}
rows = [
  row for row in sorted(
    training["samples"] + test["samples"],
    key=lambda value: (value["split"], value["speciesId"], value["sampleId"]),
  )
  if row["sampleId"] not in known
]
remote_root = Path(
  "/home/zhang/workspace/zdog-yizhu-collection-20260930/"
  "server/.cache/training/target162-v2-supplement"
)
samples, files = [], []
for row in rows:
  relative = Path(row["path"])
  files.append(str(relative))
  samples.append({
    **{key: row[key] for key in (
      "sampleId", "speciesId", "imageSha256", "observerGroup", "photoId",
      "observationId", "split",
    )},
    "sourcePath": str(remote_root / relative),
  })
plan = root / "server/.artifacts/features/target162-v2-plantclef-supplement-plan"
(plan / "files.txt").write_text("\n".join(files) + "\n")
(plan / "partition.json").write_text(json.dumps({
  "createdAt": "2026-09-30",
  "purpose": "target162-plantclef-feature-partition",
  "node": "host-v2-supplement",
  "samples": samples,
}, ensure_ascii=False, indent=2) + "\n")
print(len(samples))
PY

ssh -o BatchMode=yes zhang@host.942295.xyz \
  "mkdir -p '$HOST_ROOT/server/.cache/training/target162-v2-supplement'"
rsync -a --files-from="$SUPPLEMENT_PLAN/files.txt" "$SNAPSHOT/" \
  zhang@host.942295.xyz:"$HOST_ROOT/server/.cache/training/target162-v2-supplement/"
scp -q "$SUPPLEMENT_PLAN/partition.json" \
  zhang@host.942295.xyz:"$HOST_ROOT/server/.artifacts/target162-v2-supplement-partition.json"

REMOTE_FEATURES="$HOST_ROOT/server/.artifacts/features/target162-v2-plantclef-supplement"
if ! remote_stage zhang@host.942295.xyz "$REMOTE_FEATURES/progress.json" 2>/dev/null |
  grep -qx complete; then
  ssh -o BatchMode=yes zhang@host.942295.xyz \
    "systemd-run --user --unit=yizhu-target162-v2-plantclef-supplement \
      --property=MemoryMax=6G --property=CPUQuota=400% \
      --setenv=PYTHONPATH=$HOST_ROOT/server/src \
      $HOST_ROOT/.venv/bin/python -m yizhu_server.plantclef_partition_features \
      --partition $HOST_ROOT/server/.artifacts/target162-v2-supplement-partition.json \
      --model $HOST_ROOT/server/.cache/models/sparrow-plantclef-dinov2/unpacked/plantclef-dinov2/1/model.onnx \
      --output $REMOTE_FEATURES --threads 4 --batch-size 1"
fi
wait_remote_features zhang@host.942295.xyz "$REMOTE_FEATURES/progress.json" \
  "PlantCLEF full-coverage supplement"

LOCAL_SUPPLEMENT="server/.artifacts/features/target162-v2-plantclef-supplement"
rsync -a zhang@host.942295.xyz:"$REMOTE_FEATURES/" "$LOCAL_SUPPLEMENT/"
MERGED="server/.artifacts/features/target162-union-v2-plantclef-dinov2"
if [[ ! -f "$MERGED/index.json" ]]; then
  server/.venv/bin/python -m yizhu_server.merge_partition_features \
    --training "$SNAPSHOT/train-validation.json" \
    --test "$SNAPSHOT/test-heldout.json" \
    --partition "$BASE_FEATURES" --partition "$LOCAL_SUPPLEMENT" \
    --output "$MERGED"
fi
log "full 162-class PlantCLEF features merged"

run_head() {
  local transform="$1" weighting="$2"
  local output="server/.artifacts/models/target162-plantclef-v2-${transform}-${weighting}-162-v1"
  if [[ ! -f "$output/results.json" ]]; then
    ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    server/.venv/bin/python -m yizhu_server.target_only_feature_training \
      --features "$MERGED" --classes "$SNAPSHOT/train-validation.json" \
      --output "$output" --epochs 250 --patience 30 --seed 20260930 --threads 1 \
      --feature-transform "$transform" --class-weighting "$weighting" \
      --label-smoothing .1 --dropout .1 --batch-size 256
  fi
  log "trained: $output"
}

run_head none inverse
run_head zscore inverse
run_head zscore none
run_head l2 inverse
run_head center-l2 inverse

server/.venv/bin/python - <<'PY'
import json
from pathlib import Path

rows = []
for path in Path("server/.artifacts/models").glob(
  "target162-plantclef-v2-*-162-v1/results.json"
):
  result = json.loads(path.read_text())
  rows.append({
    "model": path.parent.name,
    "classCount": result["classCount"],
    "validation": {
      key: result["metrics"]["validation"][key]
      for key in ("top1", "top5", "macroTop1", "macroTop5")
    },
    "developmentTest": {
      key: result["metrics"]["test"][key]
      for key in ("top1", "top5", "macroTop1", "macroTop5")
    },
    "coverageGoalMet": result["coverageGoalMet"],
    "validationGoalMet": result["validationGoalMet"],
    "finalIndependentGoalMet": result["finalIndependentGoalMet"],
  })
rows.sort(
  key=lambda row: (
    row["validation"]["macroTop1"], row["validation"]["macroTop5"],
  ),
  reverse=True,
)
Path("server/.artifacts/target162-full-coverage-summary.json").write_text(
  json.dumps({"status": "development-comparison", "models": rows},
             ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(rows, ensure_ascii=False))
PY
log "full 162-class frozen-feature experiments complete"
