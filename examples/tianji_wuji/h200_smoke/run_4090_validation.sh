#!/usr/bin/env bash
set -euo pipefail
: "${TIANJI_WORK_ROOT:?Set the prepared shared-disk directory}"
export TIANJI_JOB_ID="${TIANJI_JOB_ID:-tianji-4090-environment-$(date +%Y%m%d-%H%M%S)}"
[[ "$TIANJI_JOB_ID" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Invalid job ID" >&2; exit 2; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
export PATH="$TIANJI_WORK_ROOT/tools/bootstrap/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$TIANJI_WORK_ROOT/tools/python"
export HF_HOME="$TIANJI_WORK_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline UV_OFFLINE=true PYTHONUNBUFFERED=1
export UV_CACHE_DIR="$TIANJI_WORK_ROOT/uv-cache"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
job_root="$TIANJI_WORK_ROOT/jobs/$TIANJI_JOB_ID"
[[ ! -e "$job_root" ]] || { echo "Job directory exists; choose a new job ID" >&2; exit 2; }
mkdir -p "$job_root/logs"
exec > >(tee -a "$job_root/logs/console.log") 2>&1
bash examples/tianji_wuji/h200_smoke/bootstrap_system_offline.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$job_root/gpu.csv"
uv run --no-sync python examples/tianji_wuji/h200_smoke/preflight.py \
  --work-root "$TIANJI_WORK_ROOT" --job-root "$job_root" --profile 4090-environment
uv run --no-sync lerobot-train --config_path="$job_root/train_config.json"
uv run --no-sync python examples/tianji_wuji/h200_smoke/verify_4090.py --job-root "$job_root"
echo "4090 environment validation passed: $job_root/verification.json"
