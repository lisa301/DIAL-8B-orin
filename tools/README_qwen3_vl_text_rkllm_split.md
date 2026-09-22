# Qwen3-VL Text RKLLM Split

This directory contains the first-stage tooling for split text deployment:

```text
GPU/Candle layers 0..L-1 -> RKLLM suffix layers L..last -> existing lm_head
```

The generated RKLLM segment must be used for both prefill and decode so that
its internal self-attention KV cache stays consistent.

## 1. Build a suffix HF model

Choose `L` after the final language-layer deepstack injection for multimodal
use. For pure text, any suffix start layer can be tested.

```bash
python /home/seaway/sdb/ljl/Dial_llama/tools/make_qwen3_vl_text_rkllm_suffix_model.py \
  --src-model-dir /home/seaway/sdb/ljl/model/Qwen3-VL-8B-Instruct \
  --out-model-dir /home/seaway/sdb/ljl/model/qwen3vl_text_suffix_l24_l35 \
  --start-layer 24 \
  --include-lm-head auto \
  --force
```

Use `--dry-run` first to inspect the tensor map without writing large shards.

## 2. Install Official RKLLM Toolkit

Use Rockchip's official `rkllm-toolkit` package from `airockchip/rknn-llm`.
Version `1.2.3` supports Python 3.9-3.12. On this machine, the `qwen3` conda
environment is Python 3.10:

```bash
/home/lijilin/miniconda3/envs/qwen3/bin/python -m pip install --upgrade \
  https://raw.githubusercontent.com/airockchip/rknn-llm/main/rkllm-toolkit/packages/rkllm_toolkit-1.2.3-cp310-cp310-linux_x86_64.whl
```

The toolkit pins `transformers==4.55.2` and `torch==2.6.0`.

## 3. Export to RKLLM

First verify the model can be compiled without quantization:

```bash
/home/lijilin/miniconda3/envs/qwen3/bin/python \
  /home/seaway/sdb/ljl/Dial_llama/tools/export_qwen3_vl_text_suffix_rkllm.py \
  --model-dir /home/seaway/sdb/ljl/model/qwen3vl_text_suffix_l24_l35 \
  --out /home/seaway/sdb/ljl/model/qwen3vl_text_suffix_l24_l35_fp.rkllm \
  --target-platform rk3588 \
  --num-npu-core 3 \
  --no-quantization
```

Then quantize with a dataset whose inputs are boundary hidden states at layer
`L`, not raw prompts. Since the official toolkit's pinned Transformers can load
the generated text-only `qwen3` models but cannot load the original `qwen3_vl`
checkpoint directly, generate a text-only prefix model for layers `0..L-1` and
use it to create calibration embeddings.

```bash
python /home/seaway/sdb/ljl/Dial_llama/tools/make_qwen3_vl_text_prefix_model.py \
  --src-model-dir /home/seaway/sdb/ljl/model/Qwen3-VL-8B-Instruct \
  --out-model-dir /home/seaway/sdb/ljl/model/qwen3vl_text_prefix_l00_l23 \
  --end-layer 23 \
  --force
```

```bash
/home/lijilin/miniconda3/envs/qwen3/bin/python \
  /home/seaway/sdb/ljl/Dial_llama/tools/make_qwen3_vl_text_suffix_quant_data.py \
  --prefix-model-dir /home/seaway/sdb/ljl/model/qwen3vl_text_prefix_l00_l23 \
  --samples /path/to/text_calib.jsonl \
  --out /home/seaway/sdb/ljl/model/qwen3vl_text_suffix_l24_l35_calib.json \
  --start-layer 24 \
  --apply-chat-template \
  --device cuda \
  --dtype bfloat16 \
  --max-length 2048
```

```bash
/home/lijilin/miniconda3/envs/qwen3/bin/python \
  /home/seaway/sdb/ljl/Dial_llama/tools/export_qwen3_vl_text_suffix_rkllm.py \
  --model-dir /home/seaway/sdb/ljl/model/qwen3vl_text_suffix_l24_l35 \
  --out /home/seaway/sdb/ljl/model/qwen3vl_text_suffix_l24_l35_w8a8.rkllm \
  --dataset /home/seaway/sdb/ljl/model/qwen3vl_text_suffix_l24_l35_calib.json \
  --target-platform rk3588 \
  --num-npu-core 3 \
  --quantized-dtype w8a8
```

The checked-in `tools/qwen3_vl_text_suffix_calib_samples.jsonl` has only two
short records and is for pipeline validation only. Use representative business
prompts for final quantization.

## Verified Local Outputs

With `L=24`, these outputs were generated successfully:

```text
/home/seaway/sdb/ljl/model/qwen3vl_text_suffix_l24_l35
/home/seaway/sdb/ljl/model/qwen3vl_text_suffix_l24_l35_fp.rkllm
/home/seaway/sdb/ljl/model/qwen3vl_text_prefix_l00_l23
/home/seaway/sdb/ljl/model/qwen3vl_text_suffix_l24_l35_calib_2.json
/home/seaway/sdb/ljl/model/qwen3vl_text_suffix_l24_l35_w8a8_calib2.rkllm
```

The `w8a8_calib2` file proves the official quantization path accepts boundary
`input_embed` data. Do not treat it as a final-quality quantized model.

## Runtime contract

- New dialog: clear the RKLLM KV cache.
- Prefill: run local layers `0..L-1`, pass the resulting hidden states via
  `RKLLM_INPUT_EMBED`, request `RKLLM_INFER_GET_LAST_HIDDEN_LAYER`, then run the
  existing final `lm_head`.
- Decode: repeat the same path with one token. Keep the same RKLLM handle so the
  suffix model owns its KV cache.
- If the RKLLM suffix fails mid-dialog, replay from the prompt with the fallback
  path. The suffix KV cache is internal to RKLLM and cannot be synced into the
  existing Candle cache.

This first-stage suffix model keeps the real final language RMSNorm. An
arbitrary middle segment requires a custom model with identity output norm and a
second GPU/Candle boundary after the RKLLM segment.
