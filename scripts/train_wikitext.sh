#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Wikitext is typically stored in the user-local datasets folder.
# Can be overridden by exporting DATASETS_DIR before running this script.
WT_DATASETS_DIR="${DATASETS_DIR:-$HOME/datasets}"

GPU_ID="0"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  GPU_ID="$1"
  shift
fi

if [[ " $* " == *" --model "* ]]; then
  exec env DATASETS_DIR="$WT_DATASETS_DIR" "$SCRIPT_DIR/train_baselines.sh" "$GPU_ID" --dataset wikitext --algo-set regular "$@"
fi

env DATASETS_DIR="$WT_DATASETS_DIR" "$SCRIPT_DIR/train_baselines.sh" "$GPU_ID" --dataset wikitext --model base --algo-set regular "$@"
env DATASETS_DIR="$WT_DATASETS_DIR" "$SCRIPT_DIR/train_baselines.sh" "$GPU_ID" --dataset wikitext --model llama --algo-set regular "$@"
