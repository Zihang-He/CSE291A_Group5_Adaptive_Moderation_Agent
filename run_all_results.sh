#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv Python at $PYTHON_BIN"
  echo "Create the environment first."
  exit 1
fi

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  echo "Missing $ROOT_DIR/.env"
  echo "Create the API config first."
  exit 1
fi

if [[ ! -f "$ROOT_DIR/train.csv" ]]; then
  echo "Missing $ROOT_DIR/train.csv"
  echo "Extract or place the Jigsaw dataset in the project root first."
  exit 1
fi

cd "$ROOT_DIR"

echo "[1/4] Running perception on Jigsaw dataset..."
"$PYTHON_BIN" - <<'PY'
import asyncio
from run_perception_on_jigsaw import main

asyncio.run(main(split="train", limit=100, output_path="jigsaw_perception_output.jsonl"))
PY

echo "[2/4] Running ReAct action chooser..."
"$PYTHON_BIN" run_action_chooser_on_output.py

echo "[3/4] Evaluating perception outputs..."
"$PYTHON_BIN" evaluate_perception.py

echo "[4/4] Running simulator rollout demo..."
"$PYTHON_BIN" roll_out_demo.py

echo
echo "Finished."
echo "Perception output: $ROOT_DIR/jigsaw_perception_output.jsonl"
echo "Action output:     $ROOT_DIR/jigsaw_action_output.jsonl"
echo "Plots:             $ROOT_DIR/evaluation_plots/"
