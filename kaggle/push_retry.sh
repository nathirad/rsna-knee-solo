#!/usr/bin/env bash
# Push a Kaggle kernel folder and make sure the session really starts.
# A session sometimes dies within a minute of the push with an empty log and no quota used
# (seen 2026-10-05/06, same notebook starts on a later try); this re-pushes until one survives.
# Usage: bash push_retry.sh <kernel-folder> [accelerator] [max attempts, default 6]
set -uo pipefail
DIR=${1:?usage: push_retry.sh <kernel-folder> [accelerator] [attempts]}
ACC=${2:-}
MAX=${3:-6}
KAGGLE=${KAGGLE:-$(command -v kaggle || echo "$HOME/.local/bin/kaggle")}
ID=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['id'])" "$DIR/kernel-metadata.json")
for attempt in $(seq 1 "$MAX"); do
  if [ -n "$ACC" ]; then OUT=$("$KAGGLE" kernels push -p "$DIR" --accelerator "$ACC" 2>&1 | tail -1); else OUT=$("$KAGGLE" kernels push -p "$DIR" 2>&1 | tail -1); fi
  echo "$(date -u +%H:%M:%S) attempt $attempt: ${OUT:0:70}"
  case "$OUT" in *"successfully pushed"*) ;; *) echo "push refused: $OUT"; exit 2 ;; esac
  alive=1
  for i in $(seq 1 10); do
    sleep 20
    S=$("$KAGGLE" kernels status "$ID" 2>&1 | tail -1)
    case "$S" in *KernelWorkerStatus.ERROR*|*KernelWorkerStatus.CANCEL*) alive=0; break ;; *KernelWorkerStatus.COMPLETE*) break ;; esac
  done
  if [ "$alive" = 1 ]; then echo "$(date -u +%H:%M:%S) started (attempt $attempt)"; exit 0; fi
  echo "$(date -u +%H:%M:%S) died at start, retrying in 60 s"
  sleep 60
done
echo "gave up after $MAX attempts"; exit 1
