#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

export MOLMO_DATA_DIR="$ROOT_DIR/molmo_data"

echo "Project root : $ROOT_DIR"
echo "Model        : $ROOT_DIR/models/molmo2-4b"
echo "Data         : $ROOT_DIR/molmo_data/custom/shrimp"
echo "Output       : $ROOT_DIR/eval_outputs/base_4b_native_2"

python "$SCRIPT_DIR/eval_molmo2_base_metrics_2.py" \
  --base-model-path "$ROOT_DIR/models/molmo2-4b" \
  --data-root "$ROOT_DIR/molmo_data/custom/shrimp" \
  --output-dir "$ROOT_DIR/eval_outputs/base_4b_native_2" \
  --mode both \
  --max-new-tokens 512 \
  --no-resume