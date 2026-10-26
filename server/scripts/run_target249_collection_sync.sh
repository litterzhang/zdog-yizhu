#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhang/workspace/zdog-yizhu"
cd "$ROOT"
HOST_ROOT="/home/zhang/workspace/zdog-yizhu-collection-20260930"
BCP_ROOT="/home/zhang/workspace/zdog-yizhu-training-20260924"
LOCAL="$ROOT/server/.cache/training/target249-supplement-candidates-20261001"
LOG="$ROOT/server/.artifacts/target249-collection-sync-20261001.log"
mkdir -p "$LOCAL"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"
}

remote_stage() {
  local host="$1" path="$2"
  ssh -o BatchMode=yes "$host" \
    "python3 -c 'import json; print(json.load(open(\"$path\"))[\"stage\"])'" \
    2>/dev/null
}

wait_complete() {
  local host="$1" path="$2" label="$3"
  until [[ "$(remote_stage "$host" "$path" || true)" == "collection-pass-finished" ]]; do
    log "waiting: $label"
    sleep 60
  done
  log "complete: $label"
}

HOST_CACHE="$HOST_ROOT/server/.cache/training"
BCP_CACHE="$BCP_ROOT/server/.cache/training"
wait_complete zhang@host.942295.xyz \
  "$HOST_CACHE/target249-gbif-clematis-v2-20261001/progress.json" "GBIF Clematis"
wait_complete zhang@host.942295.xyz \
  "$HOST_CACHE/target249-gbif-spathiphyllum-v2-20261001/progress.json" "GBIF Spathiphyllum"
wait_complete zhang@bcp.942295.xyz \
  "$BCP_CACHE/target249-commons-aliases-20261001/progress.json" "Commons source aliases"

sync_one() {
  local host="$1" remote="$2" local_name="$3"
  rsync -a "$host:$remote/" "$LOCAL/$local_name/"
}

sync_one zhang@host.942295.xyz \
  "$HOST_CACHE/target249-gbif-clematis-v2-20261001" gbif-clematis
sync_one zhang@host.942295.xyz \
  "$HOST_CACHE/target249-gbif-spathiphyllum-v2-20261001" gbif-spathiphyllum
sync_one zhang@host.942295.xyz \
  "$HOST_CACHE/target249-gbif-celosia-20261001" gbif-celosia
sync_one zhang@host.942295.xyz \
  "$HOST_CACHE/target249-gbif-goeppertia-20261001" gbif-goeppertia
sync_one zhang@bcp.942295.xyz \
  "$BCP_CACHE/target249-gbif-begonia-hiemalis-20261001" gbif-begonia-hiemalis
sync_one zhang@bcp.942295.xyz \
  "$BCP_CACHE/target249-gbif-begonia-tuberhybrida-20261001" gbif-begonia-tuberhybrida
sync_one zhang@bcp.942295.xyz \
  "$BCP_CACHE/target249-gbif-curio-20261001" gbif-curio
sync_one zhang@host.942295.xyz \
  "$HOST_CACHE/target249-commons-missing7-20261001" commons-missing7
sync_one zhang@bcp.942295.xyz \
  "$BCP_CACHE/target249-commons-aliases-20261001" commons-aliases

PYTHONPATH="$ROOT/server/src" "$ROOT/server/.venv/bin/python" - <<'PY'
import json
from collections import Counter
from pathlib import Path

from yizhu_server.training_data import now, write_json

root = Path("server/.cache/training/target249-supplement-candidates-20261001")
counts = Counter()
sources = []
for progress_path in sorted(root.glob("*/progress.json")):
    progress = json.loads(progress_path.read_text())
    source_counts = progress.get("counts", {})
    counts.update(source_counts)
    sources.append({
        "source": progress_path.parent.name,
        "stage": progress.get("stage"),
        "candidateImages": (
            progress.get("candidateImages")
            or progress.get("downloadedCandidateImages")
            or 0
        ),
        "counts": source_counts,
    })
write_json(root / "summary.json", {
    "createdAt": now(), "status": "collection-synchronized-review-required",
    "candidateCountsBeforeCrossSourceDedup": dict(counts),
    "sources": sources, "trainingSupportChanged": False,
    "productionChanged": False,
})
PY
log "all target249 candidate collections synchronized; review required"
