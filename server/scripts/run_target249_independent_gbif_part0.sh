#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhang/workspace/zdog-yizhu
cd "$ROOT"
PROGRESS=server/.artifacts/features/target249-base242-part0-bioclip2/progress.json
FAST=/home/zhang/workspace/zdog-yizhu-collection-20260930/server/.artifacts/features/target249-base242-part0-bioclip2-fast/progress.json
while true; do
  local_stage="$(
    python -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage"])' \
      "$PROGRESS" 2>/dev/null || true
  )"
  fast_stage="$(
    ssh -o BatchMode=yes zhang@host.942295.xyz \
      "python3 -c 'import json; print(json.load(open(\"$FAST\"))[\"stage\"])'" \
      2>/dev/null || true
  )"
  if [[ "$local_stage" == complete || "$fast_stage" == complete ]]; then break; fi
  sleep 60
done
PYTHONPATH="$ROOT/server/src" exec "$ROOT/server/.venv/bin/python" -m yizhu_server.gbif_target_collection \
  --targets server/.artifacts/landscape-union-249-20260925/targets.json \
  --output server/.cache/training/target249-independent-gbif-part0-20261001 \
  --names-file server/.artifacts/target249-independent-collection-20261001/names-0.json \
  --exclusions server/.artifacts/target249-independent-collection-20261001/exclusions.json \
  --per-species 5 --author-cap 2 --max-pages 20 --skip-unresolved
