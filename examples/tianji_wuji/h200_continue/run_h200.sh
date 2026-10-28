#!/usr/bin/env bash
set -euo pipefail
: "${TIANJI_WORK_ROOT:?Set the shared work root}"
: "${TIANJI_JOB_ID:?Set a unique job ID}"
[[ "$TIANJI_JOB_ID" =~ ^[a-zA-Z0-9._-]+$ ]] || exit 2
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PATH="$TIANJI_WORK_ROOT/tools/bootstrap/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$TIANJI_WORK_ROOT/tools/python"
export UV_CACHE_DIR="$TIANJI_WORK_ROOT/uv-cache"
export HF_HOME="$TIANJI_WORK_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 UV_OFFLINE=true
export WANDB_MODE=offline PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
unset TRANSFORMERS_CACHE
bash examples/tianji_wuji/h200_smoke/bootstrap_system_offline.sh
uv run --no-sync python examples/tianji_wuji/h200_continue/continue_training.py \
  --work-root "$TIANJI_WORK_ROOT" --job-root "$TIANJI_WORK_ROOT/jobs/$TIANJI_JOB_ID"
