#!/usr/bin/env bash
set -euo pipefail
: "${TIANJI_WORK_ROOT:?Set the prepared shared-disk directory}"
export TIANJI_JOB_ID="${TIANJI_JOB_ID:-tianji-h200-formal-v1-$(date +%Y%m%d-%H%M%S)}"
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
echo "Formal baseline: batch4, accumulation1,30000 real updates; check/restart at1000"
uv run --no-sync python examples/tianji_wuji/h200_formal/check_storage.py --job-root "$job_root"
bash examples/tianji_wuji/h200_smoke/bootstrap_system_offline.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$job_root/gpu.csv"
uv run --no-sync python examples/tianji_wuji/h200_smoke/preflight.py \
  --work-root "$TIANJI_WORK_ROOT" --job-root "$job_root" --profile h200-formal \
  --config examples/tianji_wuji/h200_formal/train_config.json
uv run --no-sync lerobot-train --config_path="$job_root/train_config.json" \
  2>&1 | tee "$job_root/logs/phase1.log"
uv run --no-sync python examples/tianji_wuji/h200_formal/verify_formal.py \
  --job-root "$job_root" --expected-stop 1000 | tee "$job_root/logs/phase1-verification.log"
uv run --no-sync lerobot-train \
  --config_path="$job_root/train/checkpoints/001000/pretrained_model/train_config.json" \
  --resume=true --stop_after_steps=30000 \
  2>&1 | tee "$job_root/logs/phase2-resume.log"
uv run --no-sync python examples/tianji_wuji/h200_formal/verify_formal.py --job-root "$job_root" \
  2>&1 | tee "$job_root/logs/verification.log"
echo "Completed: $job_root/verification.json and best_checkpoint.json"
