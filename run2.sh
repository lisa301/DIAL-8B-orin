export DIAL_TEXT_DECODE_PAST_BUCKETS=128,256,512,1024

/home/firefly/Documents/Dial_llama/target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /home/firefly/Documents/Qwen3-VL-8B-Instruct \
  --text-rknn-dir /home/firefly/Documents/Dial_llama/transmodel/transmodel2 \
  --text-rknn-prefill \
  --text-decode-mode npu-gpu \
  --vision-rknn /home/firefly/Documents/Dial_llama/qwen3_vl_8b_vision_448_deepstack_fp16.rknn \
  --vision-fixed-side 448
