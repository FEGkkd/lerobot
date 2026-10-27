#!/usr/bin/env bash
set -euo pipefail
: "${TIANJI_WORK_ROOT:?Set a writable directory on the /inspire shared disk}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
export HF_HOME="$TIANJI_WORK_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR="$TIANJI_WORK_ROOT/uv-cache"
export HF_HUB_DISABLE_TELEMETRY=1
uv sync --locked --extra groot --extra training --extra test
uv pip install --python .venv/bin/python 'tensorboard==2.20.0' 'setuptools==80.9.0'
uv run --no-sync python examples/tianji_wuji/h200_smoke/prepare_assets.py --work-root "$TIANJI_WORK_ROOT"
uv run --no-sync python examples/tianji_wuji/h200_smoke/prepare_splits.py \
  --root "$TIANJI_WORK_ROOT/datasets/bag_v3" --verify-only
uv pip freeze --python .venv/bin/python > "$TIANJI_WORK_ROOT/prepared-environment.txt"
echo "Assets are ready. Save this preparation machine as a project image, or reuse the same persistent .venv and system libraries."
