#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
exec "$repo_root/.venv/bin/lerobot-train" \
  --config_path="$repo_root/examples/tianji_wuji/train_config.json" \
  "$@"
