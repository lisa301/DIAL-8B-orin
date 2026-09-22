#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-/home/seaway/sdb/ljl/model/qwen3vl_text_block_l00_l01_dummyio48_causal_no_norm_fp.rkllm}"
DATA_DIR="${2:-${DIR}/testdata_l00_l01}"

export LD_LIBRARY_PATH="${DIR}/lib:${LD_LIBRARY_PATH:-}"
export RKLLM_LOG_LEVEL="${RKLLM_LOG_LEVEL:-1}"

if [[ -x "${DIR}/fix_freq_rk3588.sh" || -f "${DIR}/fix_freq_rk3588.sh" ]]; then
  sh "${DIR}/fix_freq_rk3588.sh" || true
fi

exec "${DIR}/rkllm_hidden_smoke" \
  "${MODEL}" \
  6 \
  4096 \
  4096 \
  "${DATA_DIR}/embed_f32.bin" \
  "${DATA_DIR}/ref_layer1_raw_hidden_f32.bin"
