#!/usr/bin/env bash
# Creates an isolated venv for the SoccerDribble demo (mujoco + gymnasium +
# stable-baselines3), separate from starVLA's main conda environment.
#
# Requires Python 3.11 (mujoco/torch prebuilt wheels are not yet available
# for 3.12+ on all platforms as of writing). Usage:
#   ./setup_env.sh [python-binary]
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${1:-python3.11}"

if ! command -v "$PYTHON_BIN" &>/dev/null; then
  echo "error: $PYTHON_BIN not found. Install Python 3.11 or pass its path as \$1." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CPU torch first (motion.pt only needs CPU inference); numpy<2 avoids a
# torch/numpy ABI mismatch (prebuilt torch wheels are compiled against numpy 1.x).
pip install "numpy<2" torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "Done. Activate with: source $(pwd)/.venv/bin/activate"
