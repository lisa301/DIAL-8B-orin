#!/bin/bash
# 1. 
export RUST_LOG=info
export QWEN3VL_VISION_CACHE=0
# 2. 
./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --cpu \
  --text-decode-mode cpu-only \
#  --text-mlp-rknn-dir /home/firefly/Documents/Dial_llama/transmodel/mlp_down_rknn_l00_l03 \
  --vision-rknn /home/firefly/Documents/Dial_llama/qwen3_vl_8b_vision_448_deepstack_fp16.rknn \
  --vision-fixed-side 448
