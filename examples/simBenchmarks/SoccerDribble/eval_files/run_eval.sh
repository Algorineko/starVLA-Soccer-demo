#!/usr/bin/env bash
# Evaluate a trained SoccerDribble model. Usage:
#   ./run_eval.sh runs/ppo_v1/best_model/best_model.zip [episodes]
set -euo pipefail
cd "$(dirname "$0")"

MODEL_PATH="${1:?usage: run_eval.sh <model.zip> [episodes]}"
EPISODES="${2:-20}"

source ../train_files/.venv/bin/activate
python3 eval_soccer.py --model "../train_files/${MODEL_PATH}" --episodes "$EPISODES"
