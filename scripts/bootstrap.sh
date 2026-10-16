#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export npm_config_cache="$PWD/.cache/npm"
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/playwright"
if [[ -x .tools/uv/bin/uv ]]; then
  export PATH="$PWD/.tools/uv/bin:$PATH"
fi
if [[ -x .tools/node-v24.21.0-darwin-arm64/bin/node ]]; then
  export PATH="$PWD/.tools/node-v24.21.0-darwin-arm64/bin:$PATH"
fi
command -v uv >/dev/null || { echo 'Install uv 0.12.23 from the official Astral instructions.'; exit 1; }
read -r task_uv_name task_uv_release task_uv_build <<< "$(uv --version)"
[[ $task_uv_name == uv && $task_uv_release == 0.12.23 ]] || { echo 'Use pinned uv 0.12.23.'; exit 1; }
[[ $(node --version) == v24.21.0 ]] || { echo 'Install the pinned official Node 24.21.0 runtime.'; exit 1; }
uv python install 3.14.8 --no-bin
uv sync --locked
npm ci --prefix apps/web
uv run --locked disguise-lab host
