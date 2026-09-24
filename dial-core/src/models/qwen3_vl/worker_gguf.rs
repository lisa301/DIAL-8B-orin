use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex, OnceLock, RwLock,
    },
};

use candle_core::{
    quantized::{gguf_file, GgmlDType, QMatMul, QTensor},
    DType, Device, IndexOp, Module, Tensor,
};

use crate::models::llama3::Config;

use super::TextLinear;

struct WorkerGgufSource {
    path: PathBuf,
    content: gguf_file::Content,
    file: Mutex<File>,
    device: Device,
}

#[derive(Debug)]
struct WorkerGgufOutputHead {
    norm: candle_nn::RmsNorm,
    output: QMatMul,
    final_block_idx: usize,
}

/// Result of the optional Worker-side output-head path.
///
/// SampledToken keeps the fast path scalar all the way into the transport layer,
/// avoiding Tensor -> RawTensor wrapping for a four-byte token id.
pub enum WorkerGgufOutput {
    Tensor(Tensor),
    SampledToken(u32),
}

static WORKER_GGUF: OnceLock<RwLock<Option<Arc<WorkerGgufSource>>>> = OnceLock::new();
static WORKER_GGUF_LINEAR_COUNT: AtomicUsize = AtomicUsize::new(0);
static WORKER_GGUF_WEIGHT_BYTES: AtomicUsize = AtomicUsize::new(0);
static WORKER_GGUF_WARMED_DTYPES: AtomicUsize = AtomicUsize::new(0);
static WORKER_GGUF_DIRECT_F16_LOGGED: AtomicBool = AtomicBool::new(false);
static WORKER_GGUF_FP16_PREFILL: AtomicBool = AtomicBool::new(false);
static WORKER_GGUF_FP16_PREFILL_LINEAR_COUNT: AtomicUsize = AtomicUsize::new(0);
static WORKER_GGUF_FP16_PREFILL_WEIGHT_BYTES: AtomicUsize = AtomicUsize::new(0);
static WORKER_GGUF_HYBRID_ROUTES_LOGGED: AtomicUsize = AtomicUsize::new(0);
static WORKER_GGUF_OUTPUT_HEAD_REQUESTED: AtomicBool = AtomicBool::new(false);
static WORKER_GGUF_SAMPLE_TOKEN_REQUESTED: AtomicBool = AtomicBool::new(false);
static WORKER_GGUF_OUTPUT_HEAD_ROUTE_LOGGED: AtomicBool = AtomicBool::new(false);
static WORKER_GGUF_OUTPUT_HEAD: OnceLock<RwLock<Option<Arc<WorkerGgufOutputHead>>>> =
    OnceLock::new();

fn source_slot() -> &'static RwLock<Option<Arc<WorkerGgufSource>>> {
    WORKER_GGUF.get_or_init(|| RwLock::new(None))
}

fn output_head_slot() -> &'static RwLock<Option<Arc<WorkerGgufOutputHead>>> {
    WORKER_GGUF_OUTPUT_HEAD.get_or_init(|| RwLock::new(None))
}

fn gguf_tensor_to_f16(tensor: &QTensor, device: &Device) -> candle_core::Result<Tensor> {
    match tensor.dtype() {
        // Candle's CUDA-specific dequantize_f16 kernel only supports quantized dtypes.
        // GGUF norm weights are commonly stored as plain F32/F16 tensors.
        GgmlDType::F32 | GgmlDType::F16 => tensor.dequantize(device)?.to_dtype(DType::F16),
        _ => tensor.dequantize_f16(device),
    }
}

pub fn configure_worker_gguf_from_args(
    path: Option<&str>,
    fp16_prefill: bool,
    output_head: bool,
    sample_token: bool,
    worker_mode: bool,
    device: &Device,
    dtype: DType,
    cfg: &Config,
) -> anyhow::Result<()> {
    WORKER_GGUF_FP16_PREFILL.store(false, Ordering::Relaxed);
    WORKER_GGUF_OUTPUT_HEAD_REQUESTED.store(false, Ordering::Relaxed);
    WORKER_GGUF_SAMPLE_TOKEN_REQUESTED.store(false, Ordering::Relaxed);
    WORKER_GGUF_OUTPUT_HEAD_ROUTE_LOGGED.store(false, Ordering::Relaxed);
    *source_slot()
        .write()
        .map_err(|_| anyhow!("worker GGUF state poisoned"))? = None;
    *output_head_slot()
        .write()
        .map_err(|_| anyhow!("worker GGUF output-head state poisoned"))? = None;

    if !worker_mode {
        if path.is_some() || fp16_prefill || output_head || sample_token {
            log::info!(
                "worker GGUF options ignored in master mode; master weights remain unchanged"
            );
        }
        return Ok(());
    }
    let Some(path) = path else {
        if fp16_prefill || output_head || sample_token {
            bail!("--worker-gguf-fp16-prefill/--worker-gguf-output-head requires --worker-quantized-gguf");
        }
        return Ok(());
    };
    if !device.is_cuda() {
        bail!("--worker-quantized-gguf currently requires a CUDA worker");
    }
    if dtype != DType::F16 {
        bail!("--worker-quantized-gguf requires --dtype f16 so DIAL activation and KV-cache contracts remain unchanged");
    }
    if sample_token && !output_head {
        bail!("--worker-gguf-sample-token true requires --worker-gguf-output-head true");
    }

    let path = Path::new(path);
    let mut file =
        File::open(path).map_err(|e| anyhow!("cannot open worker GGUF {}: {e}", path.display()))?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow!("cannot parse worker GGUF {}: {e}", path.display()))?;

    match content.metadata.get("general.architecture") {
        Some(gguf_file::Value::String(arch)) if arch == "qwen3vl" => {}
        Some(value) => bail!(
            "worker GGUF {} has unsupported architecture metadata {:?}; expected qwen3vl",
            path.display(),
            value
        ),
        None => bail!(
            "worker GGUF {} is missing general.architecture",
            path.display()
        ),
    }
    if let Some(value) = content.metadata.get("qwen3vl.block_count") {
        let block_count = value
            .to_u64()
            .map_err(|e| anyhow!("invalid qwen3vl.block_count: {e}"))?
            as usize;
        if block_count != cfg.num_hidden_layers {
            bail!(
                "worker GGUF block count {} does not match Hugging Face config {}",
                block_count,
                cfg.num_hidden_layers
            );
        }
    }

    WORKER_GGUF_LINEAR_COUNT.store(0, Ordering::Relaxed);
    WORKER_GGUF_WEIGHT_BYTES.store(0, Ordering::Relaxed);
    WORKER_GGUF_WARMED_DTYPES.store(0, Ordering::Relaxed);
    WORKER_GGUF_FP16_PREFILL_LINEAR_COUNT.store(0, Ordering::Relaxed);
    WORKER_GGUF_FP16_PREFILL_WEIGHT_BYTES.store(0, Ordering::Relaxed);
    WORKER_GGUF_HYBRID_ROUTES_LOGGED.store(0, Ordering::Relaxed);
    let source = WorkerGgufSource {
        path: path.to_path_buf(),
        content,
        file: Mutex::new(file),
        device: device.clone(),
    };
    *source_slot()
        .write()
        .map_err(|_| anyhow!("worker GGUF state poisoned"))? = Some(Arc::new(source));
    WORKER_GGUF_FP16_PREFILL.store(fp16_prefill, Ordering::Relaxed);
    WORKER_GGUF_OUTPUT_HEAD_REQUESTED.store(output_head, Ordering::Relaxed);
    WORKER_GGUF_SAMPLE_TOKEN_REQUESTED.store(sample_token, Ordering::Relaxed);
    log::info!(
        "worker GGUF enabled: file={} format=Ollama/llama.cpp activations=F16 kv_cache=F16",
        path.display()
    );
    if fp16_prefill {
        log::info!(
            "worker GGUF hybrid enabled: prefill=fused-FP16 decode=GGUF (both resident; routing by sequence length)"
        );
    }
    if output_head {
        log::info!(
            "worker GGUF output head requested: final RMSNorm and quantized output projection will run on the Worker that owns the final layer"
        );
    }
    if sample_token {
        log::info!(
            "worker GGUF remote sampling requested: repetition penalty and sampling run on the final-layer Worker; response is one U32 token id"
        );
    }
    Ok(())
}

pub fn worker_gguf_enabled() -> bool {
    source_slot()
        .read()
        .map(|source| source.is_some())
        .unwrap_or(false)
}

pub fn worker_gguf_fp16_prefill_enabled() -> bool {
    WORKER_GGUF_FP16_PREFILL.load(Ordering::Relaxed)
}

pub fn worker_gguf_output_head_requested() -> bool {
    WORKER_GGUF_OUTPUT_HEAD_REQUESTED.load(Ordering::Relaxed)
}

pub fn load_worker_gguf_output_head(
    cfg: &Config,
    owns_final_layer: bool,
) -> candle_core::Result<()> {
    *output_head_slot()
        .write()
        .map_err(|_| candle_core::Error::Msg("worker GGUF output-head state poisoned".into()))? =
        None;
    if !worker_gguf_output_head_requested() {
        return Ok(());
    }
    if !owns_final_layer {
        log::info!("worker GGUF output head not loaded: this Worker does not own the final layer");
        return Ok(());
    }
    let source = source_slot()
        .read()
        .map_err(|_| candle_core::Error::Msg("worker GGUF state poisoned".into()))?
        .clone()
        .ok_or_else(|| candle_core::Error::Msg("worker GGUF source is not configured".into()))?;

    let norm_name = "output_norm.weight";
    let norm_info = source.content.tensor_infos.get(norm_name).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "worker GGUF {} has no tensor {norm_name}",
            source.path.display()
        ))
    })?;
    if norm_info.shape.dims() != [cfg.hidden_size] {
        candle_core::bail!(
            "worker GGUF tensor {norm_name} shape {:?}, expected [{}]",
            norm_info.shape.dims(),
            cfg.hidden_size
        );
    }

    let output_name = if source.content.tensor_infos.contains_key("output.weight") {
        "output.weight"
    } else if source
        .content
        .tensor_infos
        .contains_key("token_embd.weight")
    {
        log::info!("worker GGUF output.weight missing; using tied token_embd.weight");
        "token_embd.weight"
    } else {
        candle_core::bail!(
            "worker GGUF {} has neither output.weight nor token_embd.weight",
            source.path.display()
        );
    };
    let output_info = source
        .content
        .tensor_infos
        .get(output_name)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!("worker GGUF tensor {output_name} disappeared"))
        })?;
    if output_info.shape.dims() != [cfg.vocab_size, cfg.hidden_size] {
        candle_core::bail!(
            "worker GGUF tensor {output_name} shape {:?}, expected [{}, {}]",
            output_info.shape.dims(),
            cfg.vocab_size,
            cfg.hidden_size
        );
    }

    let mut file = source
        .file
        .lock()
        .map_err(|_| candle_core::Error::Msg("worker GGUF file lock poisoned".into()))?;
    let norm_q = source
        .content
        .tensor(&mut *file, norm_name, &source.device)?;
    let norm_dtype = norm_q.dtype();
    let norm_weight = gguf_tensor_to_f16(&norm_q, &source.device)?;
    let norm = candle_nn::RmsNorm::new(norm_weight, cfg.rms_norm_eps);
    let output_q = source
        .content
        .tensor(&mut *file, output_name, &source.device)?;
    let output_dtype = output_q.dtype();
    let output_bytes = output_q.storage_size_in_bytes();
    let output = QMatMul::from_arc(Arc::new(output_q))?;
    warmup_worker_gguf(&output, output_dtype, cfg.hidden_size, &source.device)?;
    drop(file);

    let head = WorkerGgufOutputHead {
        norm,
        output,
        final_block_idx: cfg.num_hidden_layers.saturating_sub(1),
    };
    if !matches!(
        std::env::var("DIAL_WORKER_GGUF_WARMUP").ok().as_deref(),
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO")
    ) {
        let started = std::time::Instant::now();
        let hidden = Tensor::zeros((1, 1, cfg.hidden_size), DType::F16, &source.device)?;
        let _logits = forward_output_head(&head, &hidden)?;
        source.device.synchronize()?;
        log::info!(
            "worker GGUF output head warmup complete: elapsed_ms={:.1}",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
    *output_head_slot()
        .write()
        .map_err(|_| candle_core::Error::Msg("worker GGUF output-head state poisoned".into()))? =
        Some(Arc::new(head));
    log::info!(
        "worker GGUF output head loaded: norm={} norm_dtype={:?} output={} dtype={:?} shape=[{}, {}] quantized_memory={:.1} MiB return_dtype=F16",
        norm_name,
        norm_dtype,
        output_name,
        output_dtype,
        cfg.vocab_size,
        cfg.hidden_size,
        output_bytes as f64 / 1048576.0
    );
    Ok(())
}

pub fn worker_gguf_output_head_enabled() -> bool {
    output_head_slot()
        .read()
        .map(|head| head.is_some())
        .unwrap_or(false)
}

pub fn worker_gguf_sample_token_enabled() -> bool {
    worker_gguf_output_head_enabled() && WORKER_GGUF_SAMPLE_TOKEN_REQUESTED.load(Ordering::Relaxed)
}

fn forward_output_head(head: &WorkerGgufOutputHead, x: &Tensor) -> candle_core::Result<Tensor> {
    let last_hidden = match x.dims() {
        [1, seq_len, _] if *seq_len > 0 => x.i((.., seq_len - 1, ..))?.contiguous()?,
        [1, _] => x.contiguous()?,
        dims => candle_core::bail!(
            "worker GGUF output head expects [1, seq, hidden] or [1, hidden], got {:?}",
            dims
        ),
    };
    let normalized = head.norm.forward(&last_hidden)?;
    forward_worker_gguf(&head.output, &normalized)
}

pub fn maybe_forward_worker_gguf_output_head(
    x: &Tensor,
    final_request_block_idx: Option<usize>,
    sampling: Option<&crate::spm::SamplingRequest>,
) -> candle_core::Result<Option<WorkerGgufOutput>> {
    let head = output_head_slot()
        .read()
        .map_err(|_| candle_core::Error::Msg("worker GGUF output-head state poisoned".into()))?
        .clone();
    let Some(head) = head else {
        return Ok(None);
    };
    if final_request_block_idx != Some(head.final_block_idx) {
        return Ok(None);
    }
    let logits = forward_output_head(&head, x)?;
    if !WORKER_GGUF_OUTPUT_HEAD_ROUTE_LOGGED.swap(true, Ordering::Relaxed) {
        log::info!(
            "worker GGUF output-head route selected: final_block={} hidden_shape={:?} logits_shape={:?} logits_dtype={:?}",
            head.final_block_idx,
            x.dims(),
            logits.dims(),
            logits.dtype()
        );
    }
    if worker_gguf_sample_token_enabled() {
        let sampling = sampling.ok_or_else(|| {
            candle_core::Error::Msg(
                "GGUF remote sampling is enabled but the Master did not send sampling parameters; rebuild and restart both endpoints"
                    .into(),
            )
        })?;
        let token = super::sample_remote_logits(&logits, sampling)
            .map_err(|e| candle_core::Error::Msg(format!("remote sampling failed: {e}")))?;
        static SAMPLE_LOGGED: AtomicBool = AtomicBool::new(false);
        if !SAMPLE_LOGGED.swap(true, Ordering::Relaxed) {
            log::info!(
                "worker GGUF sampled-token route selected: logits_shape={:?} response_shape=[1] response_dtype=U32",
                logits.dims()
            );
        }
        return Ok(Some(WorkerGgufOutput::SampledToken(token)));
    }
    Ok(Some(WorkerGgufOutput::Tensor(logits)))
}

pub fn worker_gguf_summary() -> Option<(usize, usize, usize, usize)> {
    worker_gguf_enabled().then(|| {
        (
            WORKER_GGUF_LINEAR_COUNT.load(Ordering::Relaxed),
            WORKER_GGUF_WEIGHT_BYTES.load(Ordering::Relaxed),
            WORKER_GGUF_FP16_PREFILL_LINEAR_COUNT.load(Ordering::Relaxed),
            WORKER_GGUF_FP16_PREFILL_WEIGHT_BYTES.load(Ordering::Relaxed),
        )
    })
}

pub(crate) fn worker_gguf_prefill_for_dims(dims: &[usize]) -> bool {
    let seq_len = dims
        .len()
        .checked_sub(2)
        .and_then(|index| dims.get(index))
        .copied()
        .unwrap_or(1);
    let fp16_prefill = seq_len > 1;
    if worker_gguf_fp16_prefill_enabled() {
        let route_bit = if fp16_prefill { 1 } else { 2 };
        let previous = WORKER_GGUF_HYBRID_ROUTES_LOGGED.fetch_or(route_bit, Ordering::Relaxed);
        if previous & route_bit == 0 {
            log::info!(
                "worker GGUF hybrid route selected: seq_len={} projection_path={}",
                seq_len,
                if fp16_prefill {
                    "fused-FP16-prefill"
                } else {
                    "GGUF-decode"
                }
            );
        }
    }
    fp16_prefill
}

pub(crate) fn build_worker_gguf_fp16_prefill_linear(
    name: &str,
    source_linears: &[&TextLinear],
) -> candle_core::Result<candle_nn::Linear> {
    if source_linears.is_empty() {
        candle_core::bail!("cannot build empty FP16 GGUF prefill projection")
    }
    let weights = source_linears
        .iter()
        .map(|linear| linear.gguf_dequantize_f16_weight())
        .collect::<candle_core::Result<Vec<_>>>()?;
    let weight = if weights.len() == 1 {
        weights[0].clone()
    } else {
        let weight_refs = weights.iter().collect::<Vec<_>>();
        Tensor::cat(&weight_refs, 0)?
    };
    let weight_bytes = weight.elem_count() * 2;
    WORKER_GGUF_FP16_PREFILL_LINEAR_COUNT.fetch_add(1, Ordering::Relaxed);
    WORKER_GGUF_FP16_PREFILL_WEIGHT_BYTES.fetch_add(weight_bytes, Ordering::Relaxed);
    log::debug!(
        "built worker GGUF FP16 prefill projection {} shape={:?} memory={:.1} MiB",
        name,
        weight.dims(),
        weight_bytes as f64 / 1048576.0
    );
    Ok(candle_nn::Linear::new(weight, None))
}

pub(crate) fn attach_worker_gguf_fp16_prefill(
    name: &str,
    linear: TextLinear,
) -> candle_core::Result<TextLinear> {
    let prefill = build_worker_gguf_fp16_prefill_linear(name, &[&linear])?;
    Ok(linear.with_gguf_fp16_prefill(prefill))
}

pub(crate) fn load_worker_gguf_linear(
    layer_idx: Option<usize>,
    tensor_suffix: &str,
    in_dim: usize,
    out_dim: usize,
) -> candle_core::Result<Option<TextLinear>> {
    let source = source_slot()
        .read()
        .map_err(|_| candle_core::Error::Msg("worker GGUF state poisoned".into()))?
        .clone();
    let Some(source) = source else {
        return Ok(None);
    };
    let layer_idx = layer_idx.ok_or_else(|| {
        candle_core::Error::Msg("worker GGUF requires a numeric transformer layer name".into())
    })?;
    let tensor_name = format!("blk.{layer_idx}.{tensor_suffix}.weight");
    let info = source
        .content
        .tensor_infos
        .get(&tensor_name)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "worker GGUF {} has no tensor {tensor_name}",
                source.path.display()
            ))
        })?;
    if info.shape.dims() != [out_dim, in_dim] {
        return Err(candle_core::Error::Msg(format!(
            "worker GGUF tensor {tensor_name} shape {:?}, expected [{out_dim}, {in_dim}]",
            info.shape.dims()
        )));
    }

    let mut file = source
        .file
        .lock()
        .map_err(|_| candle_core::Error::Msg("worker GGUF file lock poisoned".into()))?;
    let tensor = source
        .content
        .tensor(&mut *file, &tensor_name, &source.device)?;
    let weight_bytes = tensor.storage_size_in_bytes();
    let ggml_dtype = tensor.dtype();
    let matmul = QMatMul::from_arc(Arc::new(tensor))?;
    warmup_worker_gguf(&matmul, ggml_dtype, in_dim, &source.device)?;
    WORKER_GGUF_LINEAR_COUNT.fetch_add(1, Ordering::Relaxed);
    WORKER_GGUF_WEIGHT_BYTES.fetch_add(weight_bytes, Ordering::Relaxed);
    log::debug!(
        "loaded worker GGUF tensor {} dtype={:?} shape=[{}, {}] memory={:.1} MiB",
        tensor_name,
        ggml_dtype,
        out_dim,
        in_dim,
        weight_bytes as f64 / 1048576.0
    );
    Ok(Some(TextLinear::Gguf {
        name: tensor_name,
        matmul,
        fp16_prefill: None,
    }))
}

fn warmup_worker_gguf(
    matmul: &QMatMul,
    ggml_dtype: GgmlDType,
    in_dim: usize,
    device: &Device,
) -> candle_core::Result<()> {
    if matches!(
        std::env::var("DIAL_WORKER_GGUF_WARMUP").ok().as_deref(),
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO")
    ) {
        return Ok(());
    }
    let dtype_bit = match ggml_dtype {
        GgmlDType::Q4K => 1,
        GgmlDType::Q6K => 2,
        _ => 4,
    };
    let previous = WORKER_GGUF_WARMED_DTYPES.fetch_or(dtype_bit, Ordering::Relaxed);
    if previous & dtype_bit != 0 {
        return Ok(());
    }

    let started = std::time::Instant::now();
    let result = (|| {
        let decode = Tensor::zeros((1, 1, in_dim), DType::F16, device)?;
        let decode_output = forward_worker_gguf(matmul, &decode)?;
        if decode_output.dtype() != DType::F16 {
            candle_core::bail!(
                "GGUF F16 decode warmup returned {:?}, expected F16",
                decode_output.dtype()
            );
        }

        // Keep the legacy F32 fallback warm for deployments that disable hybrid FP16 prefill.
        let prefill = Tensor::zeros((1, 9, in_dim), DType::F32, device)?;
        let _prefill_output = matmul.forward(&prefill)?;
        device.synchronize()
    })();
    if let Err(err) = result {
        WORKER_GGUF_WARMED_DTYPES.fetch_and(!dtype_bit, Ordering::Relaxed);
        return Err(err);
    }
    log::info!(
        "worker GGUF CUDA warmup complete: dtype={:?} decode=F16/direct seq=1 fallback=F32 seq=9 elapsed_ms={:.1}",
        ggml_dtype,
        started.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

pub(crate) fn forward_worker_gguf(
    matmul: &QMatMul,
    x: &candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    let x = x.contiguous()?;
    let rows = match x.dims() {
        [batch, seq_len, _] => batch.saturating_mul(*seq_len),
        [batch, _] => *batch,
        _ => usize::MAX,
    };
    let direct_cuda = matches!(
        x.device().location(),
        candle_core::DeviceLocation::Cuda { .. }
    ) && rows <= 8
        && matches!(x.dtype(), DType::F16 | DType::BF16 | DType::F32);
    if direct_cuda {
        // Candle 0.11 fast-MMVQ accepts F16/BF16/F32 activations directly and returns the
        // same dtype. Keep decode activations in F16 instead of launching a cast before and
        // after every quantized projection.
        let output = matmul.forward(&x)?;
        if output.dtype() != x.dtype() {
            candle_core::bail!(
                "direct CUDA GGUF matmul changed activation dtype {:?} -> {:?}",
                x.dtype(),
                output.dtype()
            );
        }
        if x.dtype() == DType::F16 && !WORKER_GGUF_DIRECT_F16_LOGGED.swap(true, Ordering::Relaxed) {
            log::info!("worker GGUF direct F16 CUDA decode enabled: no per-linear F16/F32 casts");
        }
        return Ok(output);
    }

    // The CPU/Metal fallback QMatMul still consumes F32 activations.
    let output_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    matmul.forward(&x)?.to_dtype(output_dtype)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use candle_core::{
        quantized::{GgmlDType, QMatMul, QTensor},
        DType, Device, IndexOp, Module, Tensor,
    };

    use super::{
        forward_output_head, forward_worker_gguf, gguf_tensor_to_f16, worker_gguf_prefill_for_dims,
        WorkerGgufOutputHead,
    };
    use crate::models::qwen3_vl::TextLinear;

    #[test]
    fn hybrid_routing_uses_fp16_only_for_multi_token_inputs() {
        assert!(!worker_gguf_prefill_for_dims(&[1, 1, 4096]));
        assert!(worker_gguf_prefill_for_dims(&[1, 2, 4096]));
        assert!(worker_gguf_prefill_for_dims(&[1, 224, 4096]));
        assert!(!worker_gguf_prefill_for_dims(&[1, 4096]));
    }

    fn q4k_linear(offset: usize) -> candle_core::Result<TextLinear> {
        let values = (0..512)
            .map(|index| (((index + offset) % 37) as f32 - 18.0) / 19.0)
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(values, (2, 256), &Device::Cpu)?;
        let tensor = QTensor::quantize(&weight, GgmlDType::Q4K)?;
        Ok(TextLinear::Gguf {
            name: format!("test.{offset}"),
            matmul: QMatMul::from_arc(Arc::new(tensor))?,
            fp16_prefill: None,
        })
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_q4k_decode_keeps_f16_activations() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let values = (0..512)
            .map(|index| ((index % 37) as f32 - 18.0) / 19.0)
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(values, (2, 256), &Device::Cpu)?;
        let tensor = QTensor::quantize_onto(&weight, GgmlDType::Q4K, &device)?;
        let matmul = QMatMul::from_arc(Arc::new(tensor))?;
        let input = Tensor::from_vec(
            (0..256)
                .map(|index| ((index % 29) as f32 - 14.0) / 15.0)
                .collect::<Vec<_>>(),
            (1, 1, 256),
            &device,
        )?
        .to_dtype(DType::F16)?;

        let output = forward_worker_gguf(&matmul, &input)?;
        device.synchronize()?;
        assert_eq!(output.dtype(), DType::F16);
        assert_eq!(output.dims(), [1, 1, 2]);
        assert!(output
            .flatten_all()?
            .to_vec1::<half::f16>()?
            .iter()
            .all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn plain_f32_gguf_tensor_converts_to_f16() -> candle_core::Result<()> {
        let values = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, &Device::Cpu)?;
        let tensor = QTensor::quantize(&values, GgmlDType::F32)?;
        let converted = gguf_tensor_to_f16(&tensor, &Device::Cpu)?;
        assert_eq!(converted.dtype(), DType::F16);
        assert_eq!(
            converted.to_vec1::<half::f16>()?,
            vec![
                half::f16::from_f32(1.0),
                half::f16::from_f32(2.0),
                half::f16::from_f32(3.0),
                half::f16::from_f32(4.0),
            ]
        );
        Ok(())
    }

    #[test]
    fn fused_fp16_prefill_matches_separate_dequantized_q4k_projections() -> candle_core::Result<()>
    {
        let first = q4k_linear(0)?;
        let second = q4k_linear(11)?;
        let first_dense = candle_nn::Linear::new(first.gguf_dequantize_f16_weight()?, None);
        let second_dense = candle_nn::Linear::new(second.gguf_dequantize_f16_weight()?, None);
        let fused = super::build_worker_gguf_fp16_prefill_linear("test.fused", &[&first, &second])?;
        let input = Tensor::from_vec(
            (0..768)
                .map(|index| ((index % 23) as f32 - 11.0) / 12.0)
                .collect::<Vec<_>>(),
            (1, 3, 256),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?;

        let expected = Tensor::cat(
            &[first_dense.forward(&input)?, second_dense.forward(&input)?],
            candle_core::D::Minus1,
        )?;
        let actual = fused.forward(&input)?;
        assert_eq!(
            actual.flatten_all()?.to_vec1::<half::f16>()?,
            expected.flatten_all()?.to_vec1::<half::f16>()?
        );
        Ok(())
    }

    #[test]
    fn hybrid_linear_keeps_q4k_for_single_token_decode() -> candle_core::Result<()> {
        let linear = q4k_linear(7)?;
        let matmul = match &linear {
            TextLinear::Gguf { matmul, .. } => matmul.clone(),
            _ => unreachable!(),
        };
        let fp16 = candle_nn::Linear::new(linear.gguf_dequantize_f16_weight()?, None);
        let hybrid = linear.with_gguf_fp16_prefill(fp16);
        let input = Tensor::from_vec(
            (0..256)
                .map(|index| ((index % 29) as f32 - 14.0) / 15.0)
                .collect::<Vec<_>>(),
            (1, 1, 256),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?;

        let expected = forward_worker_gguf(&matmul, &input)?;
        let actual = hybrid.forward(&input)?;
        assert_eq!(
            actual.flatten_all()?.to_vec1::<half::f16>()?,
            expected.flatten_all()?.to_vec1::<half::f16>()?
        );
        Ok(())
    }

    #[test]
    fn output_head_uses_last_hidden_and_quantized_projection() -> candle_core::Result<()> {
        let output = match q4k_linear(17)? {
            TextLinear::Gguf { matmul, .. } => matmul,
            _ => unreachable!(),
        };
        let norm_weight = Tensor::ones(256, DType::F16, &Device::Cpu)?;
        let head = WorkerGgufOutputHead {
            norm: candle_nn::RmsNorm::new(norm_weight, 1e-6),
            output: output.clone(),
            final_block_idx: 35,
        };
        let input = Tensor::from_vec(
            (0..768)
                .map(|index| ((index % 31) as f32 - 15.0) / 16.0)
                .collect::<Vec<_>>(),
            (1, 3, 256),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?;

        let last_hidden = input.i((.., 2, ..))?.contiguous()?;
        let normalized = head.norm.forward(&last_hidden)?;
        let expected = forward_worker_gguf(&output, &normalized)?;
        let actual = forward_output_head(&head, &input)?;
        assert_eq!(actual.dims(), [1, 2]);
        assert_eq!(
            actual.flatten_all()?.to_vec1::<half::f16>()?,
            expected.flatten_all()?.to_vec1::<half::f16>()?
        );
        Ok(())
    }
}
