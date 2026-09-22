# Orin Worker GGUF quantization

This backend uses the same GGUF mixed quantization family as Ollama/llama.cpp. For
Qwen3-VL 8B `Q4_K_M`, most transformer matrices are Q4_K and selected matrices
are Q6_K. DIAL loads only the transformer layers assigned to the current Worker.

Master embeddings, vision, final norm and lm_head are still loaded from the
matching Hugging Face checkpoint. Activations transferred between Master and
Worker, residual values, attention state and KV cache remain F16.

The GGUF must be the same model variant as the Hugging Face model. Do not combine
an Instruct Hugging Face checkpoint with a Thinking GGUF.

With Ollama installed, obtain the matching file with:

```bash
ollama pull qwen3-vl:8b-instruct-q4_K_M
ollama show qwen3-vl:8b-instruct-q4_K_M --modelfile
```

Use the absolute blob path printed by the `FROM` line as
`--worker-quantized-gguf`.

## Build

```bash
cargo build --release --features cuda
```

## Worker on Orin

```bash
RUST_LOG=info ./target/release/dial-cli \
  --mode worker \
  --name worker0 \
  --address 0.0.0.0:10138 \
  --topology ./topology_qwen3vl_worker_gguf.yml \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --worker-quantized-gguf /path/to/qwen3-vl-8b-instruct-q4_K_M.gguf \
  --worker-gguf-fp16-prefill true \
  --worker-gguf-output-head true \
  --worker-gguf-sample-token true \
  --worker-w8a16 false \
  --local-linear-q8 false \
  --dtype f16
```

## Master on RK3588

```bash
RUST_LOG=info ./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --topology ./topology_qwen3vl_worker_gguf.yml \
  --model /path/to/Qwen3-VL-8B-Instruct
```

The layer range can be changed freely in the topology. A Worker with layers
`12-23`, for example, reads only the seven quantized projection matrices for
those twelve layers. If both `--worker-w8a16 true` and
`--worker-quantized-gguf` are supplied, GGUF takes precedence.

`--worker-gguf-fp16-prefill true` enables the hybrid execution path. At Worker
startup, DIAL dequantizes the assigned GGUF projection matrices to F16 and
fuses Q/K/V and gate/up. Multi-token inputs use these four fused F16 projections
per layer; single-token decode keeps using the seven original GGUF projections.
Both copies remain resident, so there is no model reload between prefill and
decode and both paths use values derived from the same GGUF file.

For all 36 Qwen3-VL 8B layers, the extra F16 projections occupy approximately
13 GiB in addition to the roughly 4 GiB GGUF projections, KV cache, CUDA
workspace, norms and the remaining model components. This mode is intended for
a 32 GB Orin. Leave the flag `false` on a 16 GB device. The startup summary must
show non-zero `fp16_prefill_linears` and `fp16_prefill_memory`; for 36 layers the
expected FP16 projection count is 144.

On the first request, `RUST_LOG=info` prints one route line for prefill and one
for decode. They should contain `projection_path=fused-FP16-prefill` for the
multi-token request and `projection_path=GGUF-decode` for each subsequent
single-token route (the message itself is emitted only once to avoid log noise).

`--worker-gguf-output-head true` moves the final RMSNorm and quantized output
projection to the Worker that owns the final transformer layer. The Worker
selects the final hidden state, returns F16 logits with shape `[1, vocab_size]`,
and the Master keeps the existing repeat-penalty and sampling implementation.
This increases each response from an 8 KiB hidden state to about 297 KiB for
Qwen3-VL 8B, but avoids the much slower RK3588 CPU lm_head. Both the Worker and
Master must be rebuilt from the updated source because the Master must recognize
remote logits and skip its local final norm and lm_head.

`--worker-gguf-sample-token true` also moves repeat penalty and sampling to the
final-layer Worker. The Master sends the sampling parameters and recent token
context with each request; the Worker returns one `[1] U32` token id instead of
the full F16 vocabulary logits. This removes the roughly 297 KiB response per
generated token. It requires `--worker-gguf-output-head true`. Set it to `false`
while leaving the output head enabled for the full-logits A/B baseline.

Expected Worker startup and first-request messages include:

```text
worker GGUF output head loaded: norm=output_norm.weight output=output.weight ...
worker GGUF output-head route selected: ... logits_shape=[1, 151936] logits_dtype=F16
```

The Master should print:

```text
remote GGUF output head active: received logits shape=[1, 151936] dtype=F16; skipping Master final_norm and lm_head
```

With remote sampling enabled, the first request should additionally print:

```text
worker GGUF remote sampling requested
worker GGUF sampled-token route selected: ... response_shape=[1] response_dtype=U32
remote GGUF sampling active: received token shape=[1] dtype=U32
```

At startup, DIAL warms both decode and prefill kernels for each GGUF weight type
so the first user request does not pay CUDA module initialization latency. Set
`DIAL_WORKER_GGUF_WARMUP=0` only when measuring cold-start behavior.
