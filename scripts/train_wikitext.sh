#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GPU_ID="0"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  GPU_ID="$1"
  shift
fi

MODEL_OVERRIDE=""
PASS_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --model)
      MODEL_OVERRIDE="$2"
      shift 2
      ;;
    *)
      PASS_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -n "$MODEL_OVERRIDE" ]; then
  exec "$SCRIPT_DIR/train_baselines.sh" "$GPU_ID" --dataset wikitext --model "$MODEL_OVERRIDE" --algo-set regular "${PASS_ARGS[@]}"
fi

"$SCRIPT_DIR/train_baselines.sh" "$GPU_ID" --dataset wikitext --model base --algo-set regular "${PASS_ARGS[@]}"
"$SCRIPT_DIR/train_baselines.sh" "$GPU_ID" --dataset wikitext --model llama --algo-set regular "${PASS_ARGS[@]}"
