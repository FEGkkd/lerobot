#!/usr/bin/env bash
set -euo pipefail
: "${TIANJI_WORK_ROOT:?Set a writable directory on the /inspire shared disk}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
export PATH="$TIANJI_WORK_ROOT/tools/bootstrap/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$TIANJI_WORK_ROOT/tools/python"
export HF_HOME="$TIANJI_WORK_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR="$TIANJI_WORK_ROOT/uv-cache"
export HF_HUB_DISABLE_TELEMETRY=1
internal_mirror=false
environment_only=false
for argument in "$@"; do
  case "$argument" in
    --internal-mirror) internal_mirror=true ;;
    --environment-only) environment_only=true ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done
index_args=()
if "$internal_mirror"; then
  index_args=(--default-index http://nexus.sii.shaipower.online/repository/pypi/simple --allow-insecure-host nexus.sii.shaipower.online)
  [[ -x .venv/bin/python ]] || uv venv --python 3.12.14 .venv
  mkdir -p "$TIANJI_WORK_ROOT/preparation"
  uv export --frozen --format requirements.txt --extra groot --extra training --extra test \
    --no-emit-project --output-file "$TIANJI_WORK_ROOT/preparation/requirements.lock.txt" > /dev/null
  uv run --no-sync python examples/tianji_wuji/h200_smoke/prepare_environment_requirements.py \
    --lock uv.lock --requirements "$TIANJI_WORK_ROOT/preparation/requirements.lock.txt" \
    --output-dir "$TIANJI_WORK_ROOT/preparation"
  uv pip sync --python .venv/bin/python --require-hashes \
    "$TIANJI_WORK_ROOT/preparation/requirements.pypi.lock.txt" \
    "$TIANJI_WORK_ROOT/preparation/requirements.pytorch.lock.txt" "${index_args[@]}"
  uv pip install --python .venv/bin/python --no-deps -e . "${index_args[@]}"
else
  uv sync --locked --python 3.12.14 --extra groot --extra training --extra test
fi
uv pip install --python .venv/bin/python 'tensorboard==2.20.0' 'setuptools==80.9.0' "${index_args[@]}"
uv run --no-sync python examples/tianji_wuji/h200_smoke/validate_cpu_environment.py --work-root "$TIANJI_WORK_ROOT"
uv pip freeze --python .venv/bin/python > "$TIANJI_WORK_ROOT/prepared-environment.txt"
if "$environment_only"; then
  echo "Python environment prepared. Model weights and data require separate verification."
  exit 0
fi
uv run --no-sync python examples/tianji_wuji/h200_smoke/prepare_assets.py --work-root "$TIANJI_WORK_ROOT"
uv run --no-sync python examples/tianji_wuji/h200_smoke/prepare_splits.py \
  --root "$TIANJI_WORK_ROOT/datasets/bag_v3" --verify-only
echo "Assets are ready. Save this preparation machine as a project image, or reuse the same persistent .venv and system libraries."
