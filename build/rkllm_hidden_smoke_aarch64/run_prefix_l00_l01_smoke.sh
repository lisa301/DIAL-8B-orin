#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-${DIR}/qwen3vl_text_prefix_l00_l01_w8a8_calib2.rkllm}"

export LD_LIBRARY_PATH="${DIR}/lib:${LD_LIBRARY_PATH:-}"
export RKLLM_LOG_LEVEL="${RKLLM_LOG_LEVEL:-1}"

if [[ -x "${DIR}/fix_freq_rk3588.sh" || -f "${DIR}/fix_freq_rk3588.sh" ]]; then
  sh "${DIR}/fix_freq_rk3588.sh" || true
fi

exec "${DIR}/rkllm_hidden_smoke" "${MODEL}" 1 4096 4096
