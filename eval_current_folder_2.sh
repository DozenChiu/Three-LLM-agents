#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

export MOLMO_DATA_DIR="$ROOT_DIR/molmo_data"

echo "Project root : $ROOT_DIR"
echo "Model        : $ROOT_DIR/models/molmo2-4b"
echo "LoRA         : $ROOT_DIR/Molmo2/molmo_runs/shrimp_lora_full"
echo "Data         : $ROOT_DIR/molmo_data/custom/shrimp"
echo "Output       : $ROOT_DIR/Molmo2/eval_outputs/shrimp_lora_full_0708_2"

python "$SCRIPT_DIR/eval_shrimp_metrics.py" \
  --base-model-path "$ROOT_DIR/models/molmo2-4b" \
  --lora-path "$ROOT_DIR/Molmo2/molmo_runs/shrimp_lora_full" \
  --data-root "$ROOT_DIR/molmo_data/custom/shrimp" \
  --output-dir "$ROOT_DIR/Molmo2/eval_outputs/shrimp_lora_full_0708_2" \
  --mode both
