#!/usr/bin/env bash
# Install the verified Ubuntu runtime bundle without using any package repository.
set -euo pipefail
: "${TIANJI_WORK_ROOT:?Set the prepared shared-disk directory}"
source /etc/os-release
[[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 22.04 ]] || {
  echo "The offline runtime bundle requires the verified Ubuntu 22.04 base image." >&2
  exit 2
}
[[ "$(id -u)" == 0 ]] || { echo "The runtime bundle requires the container's root user." >&2; exit 2; }
bundle="$TIANJI_WORK_ROOT/apt-cache"
[[ -f "$bundle/SHA256SUMS" ]] || { echo "Offline system package bundle is not ready." >&2; exit 2; }
(
  cd "$bundle"
  sha256sum --check SHA256SUMS
  shopt -s nullglob
  packages=(./*.deb)
  ((${#packages[@]} > 0)) || { echo "The runtime bundle is empty." >&2; exit 2; }
  export DEBIAN_FRONTEND=noninteractive
  dpkg --unpack "${packages[@]}"
  dpkg --configure -a
)
command -v git
command -v ffmpeg
echo "Offline system runtime installed. No network package installation was used."
