# Qwen3-VL Text RKLLM Prefix 0-1

Target split:

```text
RKLLM text layers 0..1 -> GPU/Candle text layers 2..35 -> final norm -> lm_head
```

This is a prefix split. RKLLM must be used for both prefill and decode so its
internal KV cache for layers 0 and 1 stays consistent.

## Generated Files

```text
/home/seaway/sdb/ljl/model/qwen3vl_text_prefix_l00_l01
/home/seaway/sdb/ljl/model/qwen3vl_text_prefix_l00_l01_w8a8_calib2.rkllm
```

The `.rkllm` file was exported directly as W8A8 with Rockchip
`rkllm-toolkit==1.2.3`.

## Commands Used

```bash
python /home/seaway/sdb/ljl/Dial_llama/tools/make_qwen3_vl_text_prefix_model.py \
  --src-model-dir /home/seaway/sdb/ljl/model/Qwen3-VL-8B-Instruct \
  --out-model-dir /home/seaway/sdb/ljl/model/qwen3vl_text_prefix_l00_l01 \
  --end-layer 1 \
  --force
```

```bash
HF_HOME=/home/seaway/sdb/ljl/.cache/huggingface \
HF_DATASETS_CACHE=/home/seaway/sdb/ljl/.cache/huggingface/datasets \
TMPDIR=/home/seaway/sdb/ljl/.cache/tmp \
/home/lijilin/miniconda3/envs/qwen3/bin/python \
  /home/seaway/sdb/ljl/Dial_llama/tools/export_qwen3_vl_text_suffix_rkllm.py \
  --model-dir /home/seaway/sdb/ljl/model/qwen3vl_text_prefix_l00_l01 \
  --out /home/seaway/sdb/ljl/model/qwen3vl_text_prefix_l00_l01_w8a8_calib2.rkllm \
  --dataset /home/seaway/sdb/ljl/Dial_llama/tools/qwen3_vl_text_suffix_calib_samples.jsonl \
  --target-platform rk3588 \
  --num-npu-core 3 \
  --quantized-dtype w8a8
```

## Correctness Blocker

This candidate uses the official built-in Qwen3 conversion path. The generated
text prefix HF model is a standard `Qwen3Model`, whose `last_hidden_state` is
after the model final RMSNorm.

For this split, GPU layer 2 needs the raw hidden state after layer 1 and before
the final RMSNorm. A local HF boundary check showed the standard model output is
not equivalent to that raw hidden:

```text
pre_norm rms:  0.3777859509
post_norm rms: 2.2804446220
max_abs_diff:  85.3179016113
rms_diff:      2.0678000450
```

So the exported candidate must not be wired into GPU layer 2 unless RKLLM
runtime is verified to return the pre-output-norm hidden for
`RKLLM_INFER_GET_LAST_HIDDEN_LAYER`.

Rockchip's public custom model config exposes `OUTPUT_NORM`, but it does not
expose Qwen3's `self_attn.q_norm` and `self_attn.k_norm` mappings. Rebuilding
Qwen3 through that custom path would risk changing attention math and is not a
final-precision solution.

## Runtime Requirements

- New dialog: call `rkllm_clear_kv_cache`.
- Prefill: send raw token embeddings to RKLLM, request last hidden, then run GPU
  text layers `2..35`, final norm, and `lm_head`.
- Decode: send one new token embedding to the same RKLLM handle, then run GPU
  text layers `2..35`.
- Do not run GPU layers 0 and 1 in this mode.
- Do not sync RKLLM KV into Candle; RKLLM owns KV for layers 0 and 1.
- Multimodal prefill is unsafe with the current runtime's deepstack injection if
  injection remains mapped to layers 0 and 1, because RKLLM has no API for
  injecting layer-intermediate visual hidden states.
