#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GPU_ID="0"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  GPU_ID="$1"
  shift
fi

exec "$SCRIPT_DIR/train_baselines.sh" "$GPU_ID" --dataset fineweb --model base "$@"
