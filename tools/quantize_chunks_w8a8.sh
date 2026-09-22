#!/usr/bin/env bash
set -euo pipefail

# Quantize RKNN text chunk ONNX files to w8a8 and optionally delete old fp16 rknn chunks.
#
# Usage:
#   tools/quantize_chunks_w8a8.sh \
#     --onnx-dir /path/to/onnx \
#     --dataset /path/to/dataset.txt \
#     --out-dir /path/to/chunks_rknn_w8a8 \
#     [--target rk3588] \
#     [--delete-old-rknn-dir /path/to/old_chunks_rknn]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONVERT_PY="$ROOT_DIR/tools/convert_onnx_to_rknn.py"

ONNX_DIR=""
DATASET=""
OUT_DIR=""
TARGET="rk3588"
DELETE_OLD_RKNN_DIR=""
DYNAMIC_SEQ_LENS="128,256,512,1024"
PATTERN="qwen3_vl_8b_text_*_l*_l*.onnx"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx-dir)
      ONNX_DIR="$2"; shift 2;;
    --dataset)
      DATASET="$2"; shift 2;;
    --out-dir)
      OUT_DIR="$2"; shift 2;;
    --target)
      TARGET="$2"; shift 2;;
    --dynamic-seq-lens)
      DYNAMIC_SEQ_LENS="$2"; shift 2;;
    --pattern)
      PATTERN="$2"; shift 2;;
    --delete-old-rknn-dir)
      DELETE_OLD_RKNN_DIR="$2"; shift 2;;
    *)
      echo "unknown arg: $1" >&2
      exit 1;;
  esac
done

if [[ -z "$ONNX_DIR" || -z "$DATASET" || -z "$OUT_DIR" ]]; then
  echo "missing required args. need --onnx-dir --dataset --out-dir" >&2
  exit 1
fi

if [[ ! -d "$ONNX_DIR" ]]; then
  echo "onnx dir not found: $ONNX_DIR" >&2
  exit 1
fi

if [[ ! -f "$DATASET" ]]; then
  echo "dataset file not found: $DATASET" >&2
  exit 1
fi

if [[ ! -f "$CONVERT_PY" ]]; then
  echo "convert script not found: $CONVERT_PY" >&2
  exit 1
fi

if ! python3 - <<'PY' >/dev/null 2>&1
import importlib.util as u
assert u.find_spec("rknn"), "rknn module not found"
PY
then
  echo "python module 'rknn' not found in current environment." >&2
  echo "activate RKNN env first, then rerun." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

echo "[INFO] ONNX_DIR=$ONNX_DIR"
echo "[INFO] DATASET=$DATASET"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] TARGET=$TARGET"
echo "[INFO] PATTERN=$PATTERN"
echo "[INFO] DYNAMIC_SEQ_LENS=$DYNAMIC_SEQ_LENS"

python3 "$CONVERT_PY" \
  --onnx-dir "$ONNX_DIR" \
  --pattern "$PATTERN" \
  --out-dir "$OUT_DIR" \
  --target-platform "$TARGET" \
  --quantize \
  --quantized-dtype w8a8 \
  --dataset "$DATASET" \
  --dynamic-input-seq-lens "$DYNAMIC_SEQ_LENS"

echo "[INFO] quantization done."
du -sh "$OUT_DIR" || true

if [[ -n "$DELETE_OLD_RKNN_DIR" ]]; then
  if [[ ! -d "$DELETE_OLD_RKNN_DIR" ]]; then
    echo "[WARN] old rknn dir not found, skip delete: $DELETE_OLD_RKNN_DIR"
    exit 0
  fi
  echo "[WARN] deleting old rknn dir: $DELETE_OLD_RKNN_DIR"
  rm -rf "$DELETE_OLD_RKNN_DIR"
  echo "[INFO] old rknn dir deleted."
fi

