#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RKLLM_ROOT="${RKLLM_ROOT:-/home/seaway/sdb/ljl/rknn-llm-src}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/rkllm_hidden_smoke_aarch64}"
CXX="${CXX:-aarch64-linux-gnu-g++}"

INCLUDE_DIR="${RKLLM_ROOT}/rkllm-runtime/Linux/librkllm_api/include"
LIB_DIR="${RKLLM_ROOT}/rkllm-runtime/Linux/librkllm_api/aarch64"
RUNTIME_LIB="${LIB_DIR}/librkllmrt.so"

mkdir -p "${OUT_DIR}/lib"

"${CXX}" \
  -std=c++17 \
  -O2 \
  -Wall \
  -Wextra \
  -I"${INCLUDE_DIR}" \
  "${ROOT_DIR}/tools/rkllm_hidden_smoke.cpp" \
  "${RUNTIME_LIB}" \
  -Wl,-rpath,'$ORIGIN/lib' \
  -o "${OUT_DIR}/rkllm_hidden_smoke"

cp "${RUNTIME_LIB}" "${OUT_DIR}/lib/"
cp "${RKLLM_ROOT}/scripts/fix_freq_rk3588.sh" "${OUT_DIR}/" 2>/dev/null || true

echo "built: ${OUT_DIR}/rkllm_hidden_smoke"
echo "runtime: ${OUT_DIR}/lib/librkllmrt.so"
