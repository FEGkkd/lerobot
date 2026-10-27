#!/usr/bin/env bash
set -euo pipefail
: "${TIANJI_WORK_ROOT:?Set the prepared shared-disk directory}"
export TIANJI_JOB_ID="${TIANJI_JOB_ID:-tianji-h200-smoke100-$(date +%Y%m%d-%H%M%S)}"
[[ "$TIANJI_JOB_ID" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Invalid job ID" >&2; exit 2; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
export HF_HOME="$TIANJI_WORK_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export UV_OFFLINE=true
export UV_CACHE_DIR="$TIANJI_WORK_ROOT/uv-cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
job_root="$TIANJI_WORK_ROOT/jobs/$TIANJI_JOB_ID"
[[ ! -e "$job_root" ]] || { echo "Job directory exists; choose a new job ID" >&2; exit 2; }
mkdir -p "$job_root/logs"
exec > >(tee -a "$job_root/logs/console.log") 2>&1
echo "Job root: $job_root"
echo "Task: one H200, partial finetuning, 100 optimizer updates; restart at 50"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$job_root/gpu.csv"
uv run --no-sync python examples/tianji_wuji/h200_smoke/preflight.py \
  --work-root "$TIANJI_WORK_ROOT" --job-root "$job_root"
uv run --no-sync lerobot-train --config_path="$job_root/train_config.json" \
  2>&1 | tee "$job_root/logs/phase1.log"
uv run --no-sync lerobot-train \
  --config_path="$job_root/train/checkpoints/000050/pretrained_model/train_config.json" \
  --resume=true --stop_after_steps=100 \
  2>&1 | tee "$job_root/logs/phase2-resume.log"
uv run --no-sync python examples/tianji_wuji/h200_smoke/verify_run.py --job-root "$job_root" \
  2>&1 | tee "$job_root/logs/verification.log"
echo "Verification passed; inspect $job_root/verification.json"
