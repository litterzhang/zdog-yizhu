#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhang/workspace/zdog-yizhu"
cd "$ROOT"
HOST_ROOT="/home/zhang/workspace/zdog-yizhu-collection-20260930"
BCP_ROOT="/home/zhang/workspace/zdog-yizhu-training-20260924"
LOG="server/.artifacts/target249-independent-evaluation-20261001.log"
CANDIDATES="server/.cache/training/target249-independent-candidates-20261001"
mkdir -p "$CANDIDATES"

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

MODEL_INVERSE="server/.artifacts/models/target249-complete-bioclip2-center-l2-inverse-v1"
MODEL_NONE="server/.artifacts/models/target249-complete-bioclip2-center-l2-none-v1"
MODEL_TEXT_INVERSE="server/.artifacts/models/target249-complete-bioclip2-l2-inverse-text-v1"
MODEL_TEXT_NONE="server/.artifacts/models/target249-complete-bioclip2-l2-none-text-v1"
until [[
  -f "$MODEL_INVERSE/model.metadata.json"
  && -f "$MODEL_NONE/model.metadata.json"
  && -f "$MODEL_TEXT_INVERSE/model.metadata.json"
  && -f "$MODEL_TEXT_NONE/model.metadata.json"
]]; do
  log "waiting: four frozen 249-class development candidates"
  sleep 60
done

SELECTION="server/.artifacts/target249-development-selection-20261001.json"
if [[ ! -f "$SELECTION" ]]; then
  server/.venv/bin/python - <<'PY'
import json
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json

models = [
    Path("server/.artifacts/models/target249-complete-bioclip2-center-l2-inverse-v1"),
    Path("server/.artifacts/models/target249-complete-bioclip2-center-l2-none-v1"),
    Path("server/.artifacts/models/target249-complete-bioclip2-l2-inverse-text-v1"),
    Path("server/.artifacts/models/target249-complete-bioclip2-l2-none-text-v1"),
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
        "validationTop1": validation["top1"],
        "validationTop5": validation["top5"],
        "developmentTestNotUsedForSelection": True,
    })
rows.sort(
    key=lambda row: (
        row["validationMacroTop1"], row["validationMacroTop5"],
        row["validationTop1"], row["validationTop5"],
    ),
    reverse=True,
)
write_json(Path("server/.artifacts/target249-development-selection-20261001.json"), {
    "createdAt": now(), "status": "frozen-before-independent-evaluation",
    "selectionRule": "validation macro Top-1; tie macro Top-5; tie micro Top-1/Top-5",
    "candidates": rows, "selectedModel": rows[0]["model"],
    "independentOutputsReadForSelection": False, "productionChanged": False,
})
PY
fi
MODEL="$(python -c 'import json; print(json.load(open("'"$SELECTION"'"))["selectedModel"])')"
log "frozen independent candidate: $MODEL"

PART0_PROGRESS="$BCP_ROOT/server/.cache/training/target249-independent-gbif-part0-bcp-20261001/progress.json"
HOST_PROGRESS="$HOST_ROOT/server/.cache/training/target249-independent-gbif-part1-20261001/progress.json"
BCP_PROGRESS="$BCP_ROOT/server/.cache/training/target249-independent-gbif-part2-20261001/progress.json"
REMAINDER0_PROGRESS="server/.cache/training/target249-independent-gbif-remainder0-20261001/progress.json"
REMAINDER1_PROGRESS="$HOST_ROOT/server/.cache/training/target249-independent-gbif-remainder1-20261001/progress.json"
REMAINDER2_PROGRESS="$BCP_ROOT/server/.cache/training/target249-independent-gbif-remainder2-20261001/progress.json"
COMMONS_PROGRESS="$BCP_ROOT/server/.cache/training/target249-independent-commons-by-sa-missing7-20261001/progress.json"
COMMONS_SPLIT0_PROGRESS="server/.cache/training/target249-independent-commons-split0-20261001/progress.json"
COMMONS_SPLIT1_PROGRESS="$HOST_ROOT/server/.cache/training/target249-independent-commons-split1-20261001/progress.json"
COMMONS_SPLIT2_PROGRESS="$BCP_ROOT/server/.cache/training/target249-independent-commons-split2-20261001/progress.json"
until [[ "$(remote_stage zhang@bcp.942295.xyz "$PART0_PROGRESS" || true)" == collection-pass-finished ]]; do
  log "waiting: independent GBIF part 0"
  sleep 60
done
until [[ "$(remote_stage zhang@host.942295.xyz "$HOST_PROGRESS" || true)" == collection-partial-finalized ]]; do
  log "waiting: independent GBIF part 1 prefix"
  sleep 60
done
until [[ "$(remote_stage zhang@bcp.942295.xyz "$BCP_PROGRESS" || true)" == collection-partial-finalized ]]; do
  log "waiting: independent GBIF part 2 prefix"
  sleep 60
done
until [[ "$(stage "$REMAINDER0_PROGRESS" || true)" == collection-pass-finished ]]; do
  log "waiting: independent GBIF remainder 0"
  sleep 60
done
until [[ "$(remote_stage zhang@host.942295.xyz "$REMAINDER1_PROGRESS" || true)" == collection-pass-finished ]]; do
  log "waiting: independent GBIF remainder 1"
  sleep 60
done
until [[ "$(remote_stage zhang@bcp.942295.xyz "$REMAINDER2_PROGRESS" || true)" == collection-pass-finished ]]; do
  log "waiting: independent GBIF remainder 2"
  sleep 60
done
until [[ "$(remote_stage zhang@bcp.942295.xyz "$COMMONS_PROGRESS" || true)" == collection-pass-finished ]]; do
  log "waiting: independent Commons BY-SA supplement"
  sleep 60
done
until [[ "$(stage "$COMMONS_SPLIT0_PROGRESS" || true)" == collection-pass-finished ]]; do
  log "waiting: independent Commons split 0"
  sleep 60
done
until [[ "$(remote_stage zhang@host.942295.xyz "$COMMONS_SPLIT1_PROGRESS" || true)" == collection-pass-finished ]]; do
  log "waiting: independent Commons split 1"
  sleep 60
done
until [[ "$(remote_stage zhang@bcp.942295.xyz "$COMMONS_SPLIT2_PROGRESS" || true)" == collection-pass-finished ]]; do
  log "waiting: independent Commons split 2"
  sleep 60
done

sync_metadata() {
  local host="$1" remote="$2" local_root="$3"
  mkdir -p "$local_root"
  rsync -a \
    --include='*/' --include='progress.json' --include='candidates.json' \
    --include='partial.json' --include='plan.json' --exclude='*' \
    "$host:$remote/" "$local_root/"
}

sync_metadata zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/target249-independent-gbif-part0-bcp-20261001" \
  "$CANDIDATES/gbif-part0"
sync_metadata zhang@host.942295.xyz \
  "$HOST_ROOT/server/.cache/training/target249-independent-gbif-part1-20261001" \
  "$CANDIDATES/gbif-part1"
sync_metadata zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/target249-independent-gbif-part2-20261001" \
  "$CANDIDATES/gbif-part2"
sync_metadata zhang@host.942295.xyz \
  "$HOST_ROOT/server/.cache/training/target249-independent-gbif-remainder1-20261001" \
  "$CANDIDATES/gbif-remainder1"
sync_metadata zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/target249-independent-gbif-remainder2-20261001" \
  "$CANDIDATES/gbif-remainder2"
sync_metadata zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/target249-independent-commons-by-sa-missing7-20261001" \
  "$CANDIDATES/commons-by-sa"
sync_metadata zhang@host.942295.xyz \
  "$HOST_ROOT/server/.cache/training/target249-independent-commons-split1-20261001" \
  "$CANDIDATES/commons-split1"
sync_metadata zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/target249-independent-commons-split2-20261001" \
  "$CANDIDATES/commons-split2"
log "independent candidate metadata synchronized"

LOCAL_REMAINDER0="server/.cache/training/target249-independent-gbif-remainder0-20261001"
LOCAL_COMMONS0="server/.cache/training/target249-independent-commons-split0-20261001"
PREAUDIT="server/.artifacts/target249-independent-preaudit-1-v1"
if [[ ! -e "$PREAUDIT" ]]; then
  server/.venv/bin/python -m yizhu_server.target162_independent_test \
    --snapshot server/.cache/training/target249-union-v2-complete-20261001 \
    --model-metadata "$MODEL/model.metadata.json" \
    --collection "$CANDIDATES/gbif-part0" \
    --collection "$CANDIDATES/gbif-part1" \
    --collection "$CANDIDATES/gbif-part2" \
    --collection "$LOCAL_REMAINDER0" \
    --collection "$CANDIDATES/gbif-remainder1" \
    --collection "$CANDIDATES/gbif-remainder2" \
    --collection "$CANDIDATES/commons-by-sa" \
    --collection "$LOCAL_COMMONS0" \
    --collection "$CANDIDATES/commons-split1" \
    --collection "$CANDIDATES/commons-split2" \
    --exclude-manifest server/.artifacts/target162-independent-final-1-v3/manifest.json \
    --exclude-manifest server/.artifacts/target162-ensemble-independent-final-1-v1/manifest.json \
    --exclude-manifest server/.artifacts/target162-third-independent-final-1-v1/manifest.json \
    --output "$PREAUDIT" --per-species 1 --metadata-only
fi

READY="$(python -c 'import json; print(str(json.load(open("'"$PREAUDIT"'/audit.json"))["readyForOneShotEvaluation"]).lower())')"
if [[ "$READY" != true ]]; then
  log "independent preaudit incomplete; inspect $PREAUDIT/audit.json"
  exit 0
fi

sync_selected() {
  local host="$1" remote="$2" local_root="$3" label="$4"
  local file_list="server/.artifacts/target249-selected-${label}-files.txt"
  server/.venv/bin/python - "$PREAUDIT/manifest.json" "$local_root" "$file_list" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2]).resolve()
paths = []
for row in manifest["samples"]:
    path = Path(row["path"])
    if path.is_relative_to(root):
        paths.append(str(path.relative_to(root)))
Path(sys.argv[3]).write_text("".join(path + "\n" for path in sorted(paths)))
PY
  if [[ -s "$file_list" ]]; then
    rsync -a --files-from="$file_list" "$host:$remote/" "$local_root/"
  fi
}

sync_selected zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/target249-independent-gbif-part0-bcp-20261001" \
  "$CANDIDATES/gbif-part0" gbif-part0
sync_selected zhang@host.942295.xyz \
  "$HOST_ROOT/server/.cache/training/target249-independent-gbif-part1-20261001" \
  "$CANDIDATES/gbif-part1" gbif-part1
sync_selected zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/target249-independent-gbif-part2-20261001" \
  "$CANDIDATES/gbif-part2" gbif-part2
sync_selected zhang@host.942295.xyz \
  "$HOST_ROOT/server/.cache/training/target249-independent-gbif-remainder1-20261001" \
  "$CANDIDATES/gbif-remainder1" gbif-remainder1
sync_selected zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/target249-independent-gbif-remainder2-20261001" \
  "$CANDIDATES/gbif-remainder2" gbif-remainder2
sync_selected zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/target249-independent-commons-by-sa-missing7-20261001" \
  "$CANDIDATES/commons-by-sa" commons-by-sa
sync_selected zhang@host.942295.xyz \
  "$HOST_ROOT/server/.cache/training/target249-independent-commons-split1-20261001" \
  "$CANDIDATES/commons-split1" commons-split1
sync_selected zhang@bcp.942295.xyz \
  "$BCP_ROOT/server/.cache/training/target249-independent-commons-split2-20261001" \
  "$CANDIDATES/commons-split2" commons-split2
log "only selected independent images synchronized"

AUDIT="server/.artifacts/target249-independent-final-1-v1"
if [[ ! -e "$AUDIT" ]]; then
  server/.venv/bin/python -m yizhu_server.target162_independent_test \
    --snapshot server/.cache/training/target249-union-v2-complete-20261001 \
    --model-metadata "$MODEL/model.metadata.json" \
    --collection "$CANDIDATES/gbif-part0" \
    --collection "$CANDIDATES/gbif-part1" \
    --collection "$CANDIDATES/gbif-part2" \
    --collection "$LOCAL_REMAINDER0" \
    --collection "$CANDIDATES/gbif-remainder1" \
    --collection "$CANDIDATES/gbif-remainder2" \
    --collection "$CANDIDATES/commons-by-sa" \
    --collection "$LOCAL_COMMONS0" \
    --collection "$CANDIDATES/commons-split1" \
    --collection "$CANDIDATES/commons-split2" \
    --exclude-manifest server/.artifacts/target162-independent-final-1-v3/manifest.json \
    --exclude-manifest server/.artifacts/target162-ensemble-independent-final-1-v1/manifest.json \
    --exclude-manifest server/.artifacts/target162-third-independent-final-1-v1/manifest.json \
    --output "$AUDIT" --per-species 1
fi

EVALUATION="server/.artifacts/models/target249-independent-evaluation-v1"
if [[ ! -e "$EVALUATION" ]]; then
  server/.venv/bin/python -m yizhu_server.evaluate_independent_bioclip \
    --manifest "$AUDIT/manifest.json" --model "$MODEL" --output "$EVALUATION" \
    --threads 4 --batch-size 8
fi
GOAL_MET="$(python -c 'import json; print(str(json.load(open("'"$EVALUATION"'/results.json"))["goalMet"]).lower())')"
if [[ "$GOAL_MET" == true && ! -f "$MODEL/production.metadata.json" ]]; then
  server/.venv/bin/python -m yizhu_server.target_model_promotion \
    --model "$MODEL" --evaluation "$EVALUATION"
fi
log "one-shot 249-class independent evaluation complete; runtime selection unchanged"
