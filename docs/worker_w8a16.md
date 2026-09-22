# Worker-only W8A16

This path is for the CUDA Worker only. The Master keeps its existing F16/CPU-Q8 behavior.

Build the CUDA binary on the Orin host:

```bash
export PATH=/usr/local/cuda/bin:$PATH
cargo build --release --features cuda
```

Start a Worker with W8A16 enabled:

```bash
RUST_LOG=info ./target/release/dial-cli \
  --mode worker \
  --name worker0 \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --topology /path/to/topology_qwen3vl.yml \
  --dtype f16 \
  --worker-w8a16 true
```

For the A/B baseline, use the same command with `--worker-w8a16 false`.

The implementation quantizes each linear weight as it is loaded by the Worker. It follows the
topology, so it works with one layer, a prefix, a suffix, or all 36 layers without changing the
model files. QKV, O, gate/up, and down projections use row-wise INT8 weights; activations,
outputs, KV cache, and network tensors remain F16. The Worker prints the number of converted
linears and their total weight memory after loading.

`--worker-w8a16` is deliberately separate from `--local-linear-q8`: the latter is the existing
CPU-only RK3588 path. Passing `--worker-w8a16 true` to a Master, a CPU Worker, or a non-F16 Worker
does not change its weights and logs a fallback message.

This is a weight-only CUDA kernel, not Ollama's Q4_K_M implementation. It reduces Worker weight
storage and weight-read bandwidth, but the decode speed must be measured on the target Orin. Keep
the prompt, image size, topology, context limit, and generated token count identical for both A/B
runs; compare `decode_tps`, `ttft_s`, and `remote_compute_s` separately.
