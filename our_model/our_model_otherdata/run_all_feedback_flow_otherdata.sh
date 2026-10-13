#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
DATASETS=(copa storycloze mmlu wino boolq)

for dataset in "${DATASETS[@]}"; do
  "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/train_feedback_flow_otherdata.py" \
    --dataset "${dataset}" \
    "$@"
done
