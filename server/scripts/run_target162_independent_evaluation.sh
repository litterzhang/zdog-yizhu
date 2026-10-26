#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhang/workspace/zdog-yizhu"
cd "$ROOT"
LOG="server/.artifacts/target162-independent-evaluation-20260930.log"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"
}

remote_stage() {
  local host="$1" path="$2"
  ssh -o BatchMode=yes "$host" \
    "python3 -c 'import json; print(json.load(open(\"$path\"))[\"stage\"])'" \
    2>/dev/null
}

wait_collection() {
  local host="$1" path="$2" label="$3"
  until [[ "$(remote_stage "$host" "$path" || true)" == "collection-pass-finished" ]]; do
    log "waiting: $label"
    sleep 180
  done
  log "complete: $label"
}

wait_local_collection() {
  local path="$1" label="$2"
  until [[ "$(python3 -c "import json; print(json.load(open('$path'))['stage'])" \
    2>/dev/null || true)" == "collection-pass-finished" ]]; do
    log "waiting: $label"
    sleep 180
  done
  log "complete: $label"
}

HOST_ROOT="/home/zhang/workspace/zdog-yizhu-collection-20260930"
BCP_ROOT="/home/zhang/workspace/zdog-yizhu-training-20260924"
HOST_FINAL1="target162-independent-final-part1b-20260930"
HOST_FINAL2="target162-independent-final-part2b-20260930"
BCP_FINAL3="target162-independent-final-part3b-20260930"
BCP_FINAL4="target162-independent-final-part4b-20260930"
HOST_BACKUP="target162-independent-backup-host-20260930"
BCP_BACKUP="target162-independent-backup-bcp-20260930"
HOST_MISSING="target162-independent-missing35-part1-20260930"
BCP_MISSING="target162-independent-missing35-part2-20260930"

wait_collection zhang@host.942295.xyz \
  "$HOST_ROOT/server/.cache/training/$HOST_FINAL1/progress.json" "host final part1"
wait_collection zhang@host.942295.xyz \
  "$HOST_ROOT/server/.cache/training/$HOST_FINAL2/progress.json" "host final part2"
wait_collection zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/$BCP_FINAL3/progress.json" "bcp final part3"
wait_collection zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/$BCP_FINAL4/progress.json" "bcp final part4"
wait_collection zhang@host.942295.xyz \
  "$HOST_ROOT/server/.cache/training/$HOST_BACKUP/progress.json" "host shortfall backup"
wait_collection zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/$BCP_BACKUP/progress.json" "bcp shortfall backup"
wait_local_collection \
  "server/.cache/training/target162-independent-commons-shortfalls-v1/progress.json" \
  "Commons shortfall backup"
wait_collection zhang@host.942295.xyz \
  "$HOST_ROOT/server/.cache/training/$HOST_MISSING/progress.json" \
  "host photo-id-excluded shortfalls"
wait_collection zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/$BCP_MISSING/progress.json" \
  "bcp photo-id-excluded shortfalls"

META="server/.cache/training/target162-independent-meta-v1"
mkdir -p "$META"/{host-final1,host-final2,bcp-final3,bcp-final4,host-backup,bcp-backup,host-missing,bcp-missing}
FILTER=(--include='/plan.json' --include='/progress.json' --include='/candidates.json' --exclude='*')
rsync -a "${FILTER[@]}" \
  zhang@host.942295.xyz:"$HOST_ROOT/server/.cache/training/$HOST_FINAL1/" \
  "$META/host-final1/"
rsync -a "${FILTER[@]}" \
  zhang@host.942295.xyz:"$HOST_ROOT/server/.cache/training/$HOST_FINAL2/" \
  "$META/host-final2/"
rsync -a "${FILTER[@]}" \
  zhang@bcp.942295.xyz:"$BCP_ROOT/server/.cache/training/$BCP_FINAL3/" \
  "$META/bcp-final3/"
rsync -a "${FILTER[@]}" \
  zhang@bcp.942295.xyz:"$BCP_ROOT/server/.cache/training/$BCP_FINAL4/" \
  "$META/bcp-final4/"
rsync -a "${FILTER[@]}" \
  zhang@host.942295.xyz:"$HOST_ROOT/server/.cache/training/$HOST_BACKUP/" \
  "$META/host-backup/"
rsync -a "${FILTER[@]}" \
  zhang@bcp.942295.xyz:"$BCP_ROOT/server/.cache/training/$BCP_BACKUP/" \
  "$META/bcp-backup/"
rsync -a "${FILTER[@]}" \
  zhang@host.942295.xyz:"$HOST_ROOT/server/.cache/training/$HOST_MISSING/" \
  "$META/host-missing/"
rsync -a "${FILTER[@]}" \
  zhang@bcp.942295.xyz:"$BCP_ROOT/server/.cache/training/$BCP_MISSING/" \
  "$META/bcp-missing/"
log "independent candidate metadata synchronized"

MODEL="server/.artifacts/models/target162-bioclip2-v2-none-inverse-162-v1"
SNAPSHOT="server/.cache/training/target162-union-v2-gbif"
COLLECTION_ARGS=(
  --collection "$META/host-final1"
  --collection "$META/host-final2"
  --collection "$META/bcp-final3"
  --collection "$META/bcp-final4"
  --collection "$META/host-backup"
  --collection "$META/bcp-backup"
  --collection "$META/host-missing"
  --collection "$META/bcp-missing"
  --collection "server/.cache/training/target162-supplemental-review-20260930/commons-part1"
  --collection "server/.cache/training/target162-supplemental-review-20260930/commons-part2"
  --collection "server/.cache/training/target162-independent-commons-shortfalls-v1"
  --collection "server/.cache/training/target162-independent-juniperus-v1"
  --collection "server/.cache/training/target162-independent-commons-final3-v2"
  --collection "server/.cache/training/target162-independent-commons-magnolia-chapensis-v1"
)
for target in 1 5; do
  out="server/.artifacts/target162-independent-preaudit-${target}-v3"
  if [[ ! -f "$out/audit.json" ]]; then
    server/.venv/bin/python -m yizhu_server.target162_independent_test \
      --snapshot "$SNAPSHOT" --model-metadata "$MODEL/model.metadata.json" \
      "${COLLECTION_ARGS[@]}" --output "$out" --per-species "$target" \
      --metadata-only --allow-known-source-groups
  fi
done
STRICT="server/.artifacts/target162-independent-preaudit-strict-1-v3"
if [[ ! -f "$STRICT/audit.json" ]]; then
  server/.venv/bin/python -m yizhu_server.target162_independent_test \
    --snapshot "$SNAPSHOT" --model-metadata "$MODEL/model.metadata.json" \
    "${COLLECTION_ARGS[@]}" --output "$STRICT" --per-species 1 --metadata-only
fi

TARGET="$(
  server/.venv/bin/python - <<'PY'
import json
five = json.load(open("server/.artifacts/target162-independent-preaudit-5-v3/audit.json"))
one = json.load(open("server/.artifacts/target162-independent-preaudit-1-v3/audit.json"))
print(5 if five["readyForOneShotEvaluation"] else
      1 if one["readyForOneShotEvaluation"] else 0)
PY
)"
if [[ "$TARGET" == 0 ]]; then
  log "independent audit lacks absolute 162-species coverage; no model output was read"
  exit 0
fi
export TARGET
log "independent one-shot evaluation target selected without model output: ${TARGET} per species"

server/.venv/bin/python - <<'PY'
import json
from pathlib import Path
import os

manifest = json.load(open(
  f"server/.artifacts/target162-independent-preaudit-{os.environ['TARGET']}-v3/manifest.json"
))
root = Path("server/.cache/training/target162-independent-meta-v1").resolve()
groups = {
  "host-final1": [], "host-final2": [],
  "bcp-final3": [], "bcp-final4": [],
  "host-backup": [], "bcp-backup": [],
  "host-missing": [], "bcp-missing": [],
}
for row in manifest["samples"]:
  path = Path(row["path"])
  if not path.is_relative_to(root):
    # Local Commons candidates already have verified images in place.
    continue
  relative = path.relative_to(root)
  group = relative.parts[0]
  groups[group].append("/".join(relative.parts[1:]))
plan = Path("server/.artifacts/target162-independent-selected-files")
plan.mkdir(parents=True, exist_ok=True)
for group, files in groups.items():
  (plan / f"{group}.txt").write_text("\n".join(sorted(set(files))) + "\n")
PY

rsync -a --files-from=server/.artifacts/target162-independent-selected-files/host-final1.txt \
  zhang@host.942295.xyz:"$HOST_ROOT/server/.cache/training/$HOST_FINAL1/" \
  "$META/host-final1/" &
rsync -a --files-from=server/.artifacts/target162-independent-selected-files/host-final2.txt \
  zhang@host.942295.xyz:"$HOST_ROOT/server/.cache/training/$HOST_FINAL2/" \
  "$META/host-final2/" &
rsync -a --files-from=server/.artifacts/target162-independent-selected-files/bcp-final3.txt \
  zhang@bcp.942295.xyz:"$BCP_ROOT/server/.cache/training/$BCP_FINAL3/" \
  "$META/bcp-final3/" &
rsync -a --files-from=server/.artifacts/target162-independent-selected-files/bcp-final4.txt \
  zhang@bcp.942295.xyz:"$BCP_ROOT/server/.cache/training/$BCP_FINAL4/" \
  "$META/bcp-final4/" &
rsync -a --files-from=server/.artifacts/target162-independent-selected-files/host-backup.txt \
  zhang@host.942295.xyz:"$HOST_ROOT/server/.cache/training/$HOST_BACKUP/" \
  "$META/host-backup/" &
rsync -a --files-from=server/.artifacts/target162-independent-selected-files/bcp-backup.txt \
  zhang@bcp.942295.xyz:"$BCP_ROOT/server/.cache/training/$BCP_BACKUP/" \
  "$META/bcp-backup/" &
rsync -a --files-from=server/.artifacts/target162-independent-selected-files/host-missing.txt \
  zhang@host.942295.xyz:"$HOST_ROOT/server/.cache/training/$HOST_MISSING/" \
  "$META/host-missing/" &
rsync -a --files-from=server/.artifacts/target162-independent-selected-files/bcp-missing.txt \
  zhang@bcp.942295.xyz:"$BCP_ROOT/server/.cache/training/$BCP_MISSING/" \
  "$META/bcp-missing/" &
wait
log "selected independent images synchronized"

FINAL="server/.artifacts/target162-independent-final-${TARGET}-v3"
if [[ ! -f "$FINAL/manifest.json" ]]; then
  server/.venv/bin/python -m yizhu_server.target162_independent_test \
    --snapshot "$SNAPSHOT" --model-metadata "$MODEL/model.metadata.json" \
    "${COLLECTION_ARGS[@]}" --output "$FINAL" --per-species "$TARGET" \
    --allow-known-source-groups
fi
EVALUATION="server/.artifacts/models/target162-bioclip2-independent-evaluation-v3"
if [[ ! -f "$EVALUATION/results.json" ]]; then
  server/.venv/bin/python -m yizhu_server.evaluate_independent_bioclip \
    --manifest "$FINAL/manifest.json" --model "$MODEL" --output "$EVALUATION" \
    --threads 4 --batch-size 8
fi
log "one-shot independent evaluation complete"
