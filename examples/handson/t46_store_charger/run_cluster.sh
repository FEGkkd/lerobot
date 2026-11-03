#!/usr/bin/env bash
set -euo pipefail
: "${HANDSON_WORK_ROOT:?Set the separate HandsOn shared-disk root}"
: "${HANDSON_RUNTIME_ROOT:?Set the existing prepared runtime/model root}"
: "${HANDSON_PYTHON:?Set the prepared Python executable}"
: "${HANDSON_JOB_ID:?Use a unique job ID}"
profile="${HANDSON_PROFILE:-h200}"
[[ "$profile" == 4090 || "$profile" == h200 ]] || exit 2
[[ "$HANDSON_JOB_ID" =~ ^[a-zA-Z0-9._-]+$ ]] || exit 2
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HANDSON_RUNTIME_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HANDSON_WORK_ROOT/hf-datasets"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
job_root="$HANDSON_WORK_ROOT/jobs/$HANDSON_JOB_ID"
[[ ! -e "$job_root" ]] || { echo "Job directory exists; refusing duplicate start" >&2; exit 2; }
mkdir -p "$job_root/logs"
exec > >(tee -a "$job_root/logs/console.log") 2>&1
if [[ "$profile" == h200 ]]; then
  TIANJI_WORK_ROOT="$HANDSON_RUNTIME_ROOT" bash examples/tianji_wuji/h200_smoke/bootstrap_system_offline.sh
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$job_root/gpu.csv"
"$HANDSON_PYTHON" -m examples.handson.t46_store_charger.cluster_check preflight \
  --job-root "$job_root" --data-root "$HANDSON_WORK_ROOT/datasets/v3" \
  --assets "$HANDSON_RUNTIME_ROOT/assets.json" --profile "$profile"
if [[ "$profile" == 4090 ]]; then
  "$HANDSON_PYTHON" -m lerobot.scripts.lerobot_train --config_path="$job_root/train_config.json" \
    2>&1 | tee "$job_root/logs/phase1.log"
  "$HANDSON_PYTHON" -m lerobot.scripts.lerobot_train \
    --config_path="$job_root/train/checkpoints/0000003/pretrained_model/train_config.json" \
    --resume=true --stop_after_steps=4 \
    2>&1 | tee "$job_root/logs/phase2-resume.log"
  "$HANDSON_PYTHON" -m examples.handson.t46_store_charger.cluster_check inference \
    --job-root "$job_root" --profile "$profile" 2>&1 | tee "$job_root/logs/inference.log"
else
  "$HANDSON_PYTHON" -m examples.handson.t46_store_charger.run_long --job-root "$job_root"
fi
"$HANDSON_PYTHON" -m examples.handson.t46_store_charger.cluster_check verify \
  --job-root "$job_root" --profile "$profile" 2>&1 | tee "$job_root/logs/verification.log"
