#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

HAS_ALGO_SET=0
for arg in "$@"; do
	if [ "$arg" = "--algo-set" ]; then
		HAS_ALGO_SET=1
		break
	fi
done

if [ "$HAS_ALGO_SET" -eq 1 ]; then
	exec "$SCRIPT_DIR/train_tuning.sh" "$@" --dataset fineweb --model base
else
	exec "$SCRIPT_DIR/train_tuning.sh" "$@" --dataset fineweb --model base --algo-set all
fi
