use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{linear_no_bias as linear, Embedding, Linear, Module, RmsNorm};
use half::f16;
use rand::{distr::Distribution, SeedableRng};
use rayon::prelude::*;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Instant;
use tokenizers::Tokenizer;

use crate::{
    models::llama3::{Cache, Config},
    models::{chat::Message, Generator, Token},
    spm::{Context, Forwarder},
};

use super::{
    configure_local_q8_from_args, configure_mlp_rknn_from_args, configure_qkv_rknn_from_args,
    image_to_tensor, ImageSpan, PromptEncoder, Qwen3VlConfig, TextRknnRunner, Transformer,
    VideoPatchSpec, VisionConfig, VisionEncoder, VisionOutputs, VisionRknn, VisualTokenSpec,
};

/// 定义一个字符串常量结束标志符.
const DEFAULT_EOS_TOKEN: &str = "<|im_end|>";

#[allow(dead_code)]
struct RknnDeltaLayoutDebugCompat {
    k_head_seq_dim: Vec<f32>,
    k_seq_dim_head: Vec<f32>,
    v_head_seq_dim: Vec<f32>,
    v_seq_dim_head: Vec<f32>,
}

#[allow(dead_code)]
trait TextRknnRunnerCompat {
    fn sync_to_cache(
        &self,
        cache: &mut Cache,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<()>;
    fn layer_kv_cache_f32(&self, layer_idx: usize) -> Option<(&[f32], &[f32], usize)>;
    fn layer_delta_layout_debug(&self, layer_idx: usize) -> Option<&RknnDeltaLayoutDebugCompat>;
    fn can_shrink_decode_coverage(&self) -> bool;
    fn shrink_last_chunk(&mut self) -> Result<(usize, usize)>;
    fn max_supported_bucket(&self) -> usize;
}

impl TextRknnRunnerCompat for TextRknnRunner {
    fn sync_to_cache(
        &self,
        _cache: &mut Cache,
        _device: &candle_core::Device,
        _dtype: DType,
    ) -> Result<()> {
        Ok(())
    }

    fn layer_kv_cache_f32(&self, _layer_idx: usize) -> Option<(&[f32], &[f32], usize)> {
        None
    }

    fn layer_delta_layout_debug(&self, _layer_idx: usize) -> Option<&RknnDeltaLayoutDebugCompat> {
        None
    }

    fn can_shrink_decode_coverage(&self) -> bool {
        false
    }

    fn shrink_last_chunk(&mut self) -> Result<(usize, usize)> {
        bail!("this text_rknn runner build does not support shrinking decode coverage")
    }

    fn max_supported_bucket(&self) -> usize {
        0
    }
}

/// 从模型目录里把 tokenizer 文件加载出来
fn load_tokenizer(ctx: &Context) -> Result<Tokenizer> {
    let tokenizer_filename = ctx.data_path.join("tokenizer.json");
    log::info!("loading tokenizer from {}", tokenizer_filename.display());
    Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)
}
fn create_fast_logits_processor(ctx: &Context) -> FastLogitsProcessor {
    let temperature = ctx.args.temperature;
    let sampling = if temperature <= 0. {
        FastSampling::ArgMax
    } else {
        match (ctx.args.top_k, ctx.args.top_p) {
            (None, None) => FastSampling::All { temperature },
            (Some(k), None) => FastSampling::TopK { k, temperature },
            (None, Some(p)) => FastSampling::TopP { p, temperature },
            (Some(k), Some(p)) => FastSampling::TopKThenTopP { k, p, temperature },
        }
    };
    FastLogitsProcessor::new(ctx.args.seed, sampling)
}

#[derive(Clone, Debug)]
enum FastSampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

struct FastLogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: FastSampling,
}

impl FastLogitsProcessor {
    fn new(seed: u64, sampling: FastSampling) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            sampling,
        }
    }

    fn sample(
        &mut self,
        logits: &Tensor,
        repeat_penalty: f32,
        repeat_context: &[u32],
    ) -> Result<u32> {
        // Keep greedy sampling on the inference device. On CUDA this transfers
        // only the selected token id to the host instead of the full vocabulary.
        if matches!(&self.sampling, FastSampling::ArgMax) && repeat_penalty == 1.0 {
            return logits
                .argmax(D::Minus1)
                .and_then(|token| token.to_scalar::<u32>())
                .map_err(|e| anyhow!("device argmax failed: {e}"));
        }

        let mut logits = logits_to_f32_vec(logits)?;
        if repeat_penalty != 1.0 {
            apply_repeat_penalty_in_place(&mut logits, repeat_penalty, repeat_context);
        }

        match self.sampling.clone() {
            FastSampling::ArgMax => logits
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow!("empty logits")),
            FastSampling::All { temperature } => {
                softmax_in_place(&mut logits, temperature as f32);
                self.sample_multinomial(&logits)
            }
            FastSampling::TopK { k, temperature } => {
                softmax_in_place(&mut logits, temperature as f32);
                self.sample_topk(&mut logits, k)
            }
            FastSampling::TopP { p, temperature } => {
                softmax_in_place(&mut logits, temperature as f32);
                self.sample_topp(&mut logits, p as f32)
            }
            FastSampling::TopKThenTopP { k, p, temperature } => {
                softmax_in_place(&mut logits, temperature as f32);
                self.sample_topk_topp(&mut logits, k, p as f32)
            }
        }
    }

    fn sample_multinomial(&mut self, prs: &[f32]) -> Result<u32> {
        let distr = rand::distr::weighted::WeightedIndex::new(prs)
            .map_err(|e| anyhow!("weighted sampling failed: {e}"))?;
        Ok(distr.sample(&mut self.rng) as u32)
    }

    fn sample_topp(&mut self, prs: &mut [f32], top_p: f32) -> Result<u32> {
        if top_p <= 0.0 || top_p >= 1.0 {
            return self.sample_multinomial(prs);
        }
        let mut indices = (0..prs.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|&i, &j| prs[j].total_cmp(&prs[i]));

        let mut cumsum = 0.0f32;
        for index in indices {
            if cumsum >= top_p {
                prs[index] = 0.0;
            } else {
                cumsum += prs[index];
            }
        }
        self.sample_multinomial(prs)
    }

    fn sample_topk(&mut self, prs: &mut [f32], top_k: usize) -> Result<u32> {
        if top_k >= prs.len() {
            return self.sample_multinomial(prs);
        }
        let mut indices = (0..prs.len()).collect::<Vec<_>>();
        let (top_indices, _, _) =
            indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
        let top_prs = top_indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
        let selected = self.sample_multinomial(&top_prs)? as usize;
        Ok(top_indices[selected] as u32)
    }

    fn sample_topk_topp(&mut self, prs: &mut [f32], top_k: usize, top_p: f32) -> Result<u32> {
        if top_k >= prs.len() {
            return self.sample_topp(prs, top_p);
        }
        let mut indices = (0..prs.len()).collect::<Vec<_>>();
        let (top_indices, _, _) =
            indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
        let mut top_prs = top_indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
        let selected = self.sample_topp(&mut top_prs, top_p)? as usize;
        Ok(top_indices[selected] as u32)
    }
}

fn apply_repeat_penalty_in_place(logits: &mut [f32], penalty: f32, context: &[u32]) {
    let mut seen = std::collections::HashSet::with_capacity(context.len());
    for token_id in context {
        if !seen.insert(*token_id) {
            continue;
        }
        if let Some(logit) = logits.get_mut(*token_id as usize) {
            if *logit >= 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}

fn logits_to_f32_vec(logits: &Tensor) -> Result<Vec<f32>> {
    match logits.dtype() {
        DType::F32 => logits.to_vec1::<f32>().map_err(anyhow::Error::msg),
        DType::F16 => Ok(logits
            .to_vec1::<f16>()
            .map_err(anyhow::Error::msg)?
            .into_iter()
            .map(f16::to_f32)
            .collect()),
        other => Ok(logits
            .to_dtype(DType::F32)
            .map_err(anyhow::Error::msg)?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("cannot convert {other:?} logits to f32 vec: {e}"))?),
    }
}

fn is_remote_output_logits(dims: &[usize], vocab_size: usize) -> bool {
    dims == [1, vocab_size]
}

fn is_remote_sampled_token(tensor: &Tensor) -> bool {
    tensor.dtype() == DType::U32 && tensor.dims() == [1]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RemoteOutputKind {
    SampledToken,
    Logits,
}

fn classify_remote_output(tensor: &Tensor, vocab_size: usize) -> Option<RemoteOutputKind> {
    if is_remote_sampled_token(tensor) {
        Some(RemoteOutputKind::SampledToken)
    } else if is_remote_output_logits(tensor.dims(), vocab_size) {
        Some(RemoteOutputKind::Logits)
    } else {
        None
    }
}

pub(crate) fn sample_remote_logits(
    logits: &Tensor,
    request: &crate::spm::SamplingRequest,
) -> Result<u32> {
    let logits = match logits.dims() {
        [_] => logits.clone(),
        [1, _] => logits.squeeze(0)?,
        dims => bail!(
            "remote sampling expects [vocab] or [1, vocab] logits, got {:?}",
            dims
        ),
    };
    let sampling = if request.temperature <= 0.0 {
        FastSampling::ArgMax
    } else {
        match (request.top_k, request.top_p) {
            (None, None) => FastSampling::All {
                temperature: request.temperature,
            },
            (Some(k), None) => FastSampling::TopK {
                k,
                temperature: request.temperature,
            },
            (None, Some(p)) => FastSampling::TopP {
                p,
                temperature: request.temperature,
            },
            (Some(k), Some(p)) => FastSampling::TopKThenTopP {
                k,
                p,
                temperature: request.temperature,
            },
        }
    };
    // Each request is self-contained, so concurrent dialogs cannot share RNG state.
    let seed = request.seed ^ (request.sample_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    FastLogitsProcessor::new(seed, sampling).sample(
        &logits,
        request.repeat_penalty,
        &request.repeat_context,
    )
}

fn softmax_in_place(values: &mut [f32], temperature: f32) {
    let temperature = temperature.max(1e-7);
    let mut max = f32::NEG_INFINITY;
    for value in values.iter_mut() {
        *value /= temperature;
        max = max.max(*value);
    }

    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 && sum.is_finite() {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
}

struct QuantizedLmHead {
    qweight: Vec<i8>,
    scales: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
}

impl QuantizedLmHead {
    fn from_linear(linear: &Linear) -> Result<Self> {
        if linear.bias().is_some() {
            bail!("quantized lm_head only supports bias-free linear");
        }
        let weight = if linear.weight().is_contiguous() {
            linear.weight().clone()
        } else {
            linear.weight().contiguous()?
        };
        // QuantizedLmHead stores host Vecs and executes on the CPU. Copy CUDA
        // weights to the host once instead of synchronizing once per vocab row.
        let weight = weight
            .to_device(&Device::Cpu)
            .map_err(|e| anyhow!("cannot stage lm_head weights on CPU for q8: {e}"))?;
        let (out_dim, in_dim) = weight.dims2()?;

        let mut qweight = Vec::with_capacity(out_dim * in_dim);
        let mut scales = Vec::with_capacity(out_dim);
        for row_idx in 0..out_dim {
            let row_tensor = weight.narrow(0, row_idx, 1)?.flatten_all()?;
            let row = logits_to_f32_vec(&row_tensor)?;
            let max_abs = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales.push(scale);
            let inv_scale = 1.0 / scale;
            qweight.extend(row.iter().map(|v| {
                let q = (v * inv_scale).round().clamp(-127.0, 127.0);
                q as i8
            }));
            if row_idx > 0 && row_idx % 16384 == 0 {
                log::info!("lm_head q8 quantizing rows {}/{}", row_idx, out_dim);
            }
        }

        Ok(Self {
            qweight,
            scales,
            out_dim,
            in_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dims = x.dims();
        if x_dims.len() != 2 || x_dims[0] != 1 || x_dims[1] != self.in_dim {
            bail!(
                "quantized lm_head expects [1, {}], got {:?}",
                self.in_dim,
                x_dims
            );
        }
        let device = x.device().clone();
        let x_values = logits_to_f32_vec(&x.flatten_all()?)?;
        let x_max_abs = x_values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let x_scale = if x_max_abs > 0.0 {
            x_max_abs / 127.0
        } else {
            1.0
        };
        let inv_x_scale = 1.0 / x_scale;
        let x_q = x_values
            .iter()
            .map(|v| (v * inv_x_scale).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();
        let dotprod = aarch64_dotprod_available();

        let mut logits = vec![0.0f32; self.out_dim];
        // One task per vocab row is too fine-grained for RK3588. Chunk rows so
        // Rayon scheduling overhead does not dominate the int8 dot products.
        const ROWS_PER_TASK: usize = 64;
        logits
            .par_chunks_mut(ROWS_PER_TASK)
            .enumerate()
            .for_each(|(chunk_idx, outs)| {
                let row_base = chunk_idx * ROWS_PER_TASK;
                for (row_offset, out) in outs.iter_mut().enumerate() {
                    let row_idx = row_base + row_offset;
                    let row_start = row_idx * self.in_dim;
                    let row = &self.qweight[row_start..row_start + self.in_dim];
                    let acc = dot_i8(row, &x_q, dotprod);
                    *out = (acc as f32) * self.scales[row_idx] * x_scale;
                }
            });
        Tensor::from_vec(logits, (1, self.out_dim), &device)
            .map_err(|e| anyhow!("quantized lm_head logits tensor: {e}"))
    }
}

fn dot_i8(row: &[i8], x: &[i8], _dotprod: bool) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        if _dotprod {
            // SAFETY: guarded by runtime dotprod detection and equal slice length.
            return unsafe { dot_i8_sdot_asm(row.as_ptr(), x.as_ptr(), row.len()) };
        }
        // Stable fallback: multiply i8 lanes into i16 and accumulate i32.
        return unsafe { dot_i8_neon(row.as_ptr(), x.as_ptr(), row.len()) };
    }

    row.iter()
        .zip(x.iter())
        .fold(0i32, |acc, (&w, &xv)| acc + (w as i32) * (xv as i32))
}

fn aarch64_dotprod_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_neon(row: *const i8, x: *const i8, len: usize) -> i32 {
    use std::arch::aarch64::*;

    let mut acc = vdupq_n_s32(0);
    let mut i = 0usize;
    while i + 16 <= len {
        let wv = vld1q_s8(row.add(i));
        let xv = vld1q_s8(x.add(i));

        let prod_lo = vmull_s8(vget_low_s8(wv), vget_low_s8(xv));
        let prod_hi = vmull_s8(vget_high_s8(wv), vget_high_s8(xv));
        acc = vaddq_s32(acc, vpaddlq_s16(prod_lo));
        acc = vaddq_s32(acc, vpaddlq_s16(prod_hi));
        i += 16;
    }

    let mut sum = vaddvq_s32(acc);
    while i < len {
        sum += (*row.add(i) as i32) * (*x.add(i) as i32);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_i8_sdot_asm(mut row: *const i8, mut x: *const i8, mut len: usize) -> i32 {
    let mut sum: i32;
    std::arch::asm!(
        "movi v0.4s, #0",
        "1:",
        "cmp {len}, #16",
        "blt 2f",
        "ldr q1, [{row}], #16",
        "ldr q2, [{x}], #16",
        "sdot v0.4s, v1.16b, v2.16b",
        "sub {len}, {len}, #16",
        "b 1b",
        "2:",
        "addv s0, v0.4s",
        "umov {sum:w}, v0.s[0]",
        row = inout(reg) row,
        x = inout(reg) x,
        len = inout(reg) len,
        sum = lateout(reg) sum,
        out("v0") _,
        out("v1") _,
        out("v2") _,
        options(nostack)
    );

    while len > 0 {
        sum += (*row as i32) * (*x as i32);
        row = row.add(1);
        x = x.add(1);
        len -= 1;
    }
    sum
}

fn lm_head_q8_enabled(arg_enabled: bool) -> bool {
    match std::env::var("QWEN3VL_LM_HEAD_Q8").ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES") => true,
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO") => false,
        Some(_) => false,
        None => arg_enabled,
    }
}

/// 成功时返回图片的二进制字节数组，失败时返回错误
fn decode_base64_image(data: &str) -> Result<Vec<u8>> {
    // Accept raw base64 or data URLs: data:image/png;base64,....
    let payload = if let Some(idx) = data.find("base64,") {
        &data[idx + "base64,".len()..]
    } else {
        data
    };
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|e| anyhow!("invalid base64 image: {e}"))
}
/// 把已经解码出来的图片二进制数据加载成 RGB 图像，并返回原始像素数组、高度和宽度
fn load_image_rgb(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::load_from_memory(bytes).map_err(|e| anyhow!("image decode failed: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), h as usize, w as usize))
}

#[derive(Debug, Clone, Copy)]
struct VideoProbe {
    width: usize,
    height: usize,
    fps: f64,
    estimated_frames: Option<usize>,
}

struct TemporaryVideoFile {
    path: PathBuf,
}

impl TemporaryVideoFile {
    fn create(bytes: &[u8]) -> Result<Self> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let temp_dir = std::env::var_os("DIAL_VIDEO_TEMP_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&temp_dir).map_err(|e| {
            anyhow!(
                "cannot create video temp directory {}: {e}",
                temp_dir.display()
            )
        })?;

        for _ in 0..32 {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = temp_dir.join(format!("dial-video-{}-{id}.input", std::process::id()));
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            match options.open(&path) {
                Ok(mut file) => {
                    use std::io::Write;
                    if let Err(error) = file.write_all(bytes).and_then(|_| file.flush()) {
                        let _ = std::fs::remove_file(&path);
                        return Err(anyhow!(
                            "cannot write temporary video {}: {error}",
                            path.display()
                        ));
                    }
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(anyhow!(
                        "cannot create temporary video in {}: {error}",
                        temp_dir.display()
                    ))
                }
            }
        }
        bail!(
            "cannot allocate a unique temporary video in {}",
            temp_dir.display()
        )
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryVideoFile {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.path) {
            log::warn!(
                "failed to remove temporary video {}: {}",
                self.path.display(),
                error
            );
        }
    }
}

fn media_payload(data: &str) -> &str {
    data.find("base64,")
        .map(|idx| &data[idx + "base64,".len()..])
        .unwrap_or(data)
        .trim()
}

fn decode_base64_video(data: &str, max_bytes: usize) -> Result<Vec<u8>> {
    let payload = media_payload(data);
    let estimated_bytes = payload.len().saturating_mul(3) / 4;
    if estimated_bytes > max_bytes {
        bail!(
            "video payload is about {} bytes, exceeding --video-max-bytes {}",
            estimated_bytes,
            max_bytes
        );
    }
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|e| anyhow!("invalid base64 video: {e}"))?;
    if bytes.len() > max_bytes {
        bail!(
            "video payload is {} bytes, exceeding --video-max-bytes {}",
            bytes.len(),
            max_bytes
        );
    }
    Ok(bytes)
}

fn command_from_env(env_name: &str, default_name: &str, bundled_path: &str) -> String {
    std::env::var(env_name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            if Path::new(bundled_path).is_file() {
                bundled_path.to_string()
            } else {
                default_name.to_string()
            }
        })
}

fn parse_frame_rate(raw: &str) -> Result<f64> {
    let fps = if let Some((numerator, denominator)) = raw.split_once('/') {
        let numerator: f64 = numerator.parse()?;
        let denominator: f64 = denominator.parse()?;
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    } else {
        raw.parse()?
    };
    if !fps.is_finite() || fps <= 0.0 {
        bail!("invalid video frame rate: {raw}");
    }
    Ok(fps)
}

fn uniform_sample_indices(
    total_frames: usize,
    source_fps: f64,
    target_fps: f64,
    min_frames: usize,
    max_frames: usize,
) -> Result<Vec<usize>> {
    if total_frames == 0 {
        bail!("video contains no frames");
    }
    if !source_fps.is_finite()
        || source_fps <= 0.0
        || min_frames == 0
        || max_frames == 0
        || min_frames > max_frames
    {
        bail!(
            "invalid video sampling settings: source_fps={source_fps} min={min_frames} max={max_frames}"
        );
    }
    let requested = if target_fps > 0.0 {
        ((total_frames as f64 / source_fps) * target_fps).floor() as usize
    } else {
        max_frames
    };
    let count = requested.max(min_frames).min(max_frames).min(total_frames);
    if count == 1 {
        return Ok(vec![0]);
    }
    let last = (total_frames - 1) as f64;
    Ok((0..count)
        .map(|index| {
            let position = index as f64 * last / (count - 1) as f64;
            position.round().clamp(0.0, last) as usize
        })
        .collect())
}

fn ffmpeg_select_filter(indices: &[usize], scale: &str) -> String {
    let expression = indices
        .iter()
        .map(|index| format!("eq(n,{index})"))
        .collect::<Vec<_>>()
        .join("+");
    format!("select='{expression}',{scale}")
}

fn probe_video(video_path: &Path) -> Result<VideoProbe> {
    let ffprobe = command_from_env(
        "DIAL_FFPROBE_BIN",
        "ffprobe",
        "/opt/sophon/sophon-ffmpeg-latest/bin/ffprobe",
    );
    let output = Command::new(&ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
            "-of",
            "json",
            video_path
                .to_str()
                .ok_or_else(|| anyhow!("video temp path is not valid UTF-8"))?,
        ])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| anyhow!("cannot run ffprobe ({ffprobe}): {e}"))?;
    if !output.status.success() {
        bail!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| anyhow!("invalid ffprobe output: {e}"))?;
    let stream = value["streams"]
        .as_array()
        .and_then(|streams| streams.first())
        .ok_or_else(|| anyhow!("video has no decodable video stream"))?;
    let width = stream["width"]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .filter(|v| *v > 0)
        .ok_or_else(|| anyhow!("ffprobe did not return a valid video width"))?;
    let height = stream["height"]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .filter(|v| *v > 0)
        .ok_or_else(|| anyhow!("ffprobe did not return a valid video height"))?;
    let fps = parse_frame_rate(
        stream["avg_frame_rate"]
            .as_str()
            .ok_or_else(|| anyhow!("ffprobe did not return the video frame rate"))?,
    )?;
    let estimated_frames = stream["nb_frames"]
        .as_str()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|count| *count > 0)
        .or_else(|| {
            stream["duration"]
                .as_str()
                .or_else(|| value["format"]["duration"].as_str())
                .and_then(|raw| raw.parse::<f64>().ok())
                .map(|duration| (duration * fps).round() as usize)
                .filter(|count| *count > 0)
        });
    Ok(VideoProbe {
        width,
        height,
        fps,
        estimated_frames,
    })
}

fn decode_all_video_frames<F>(
    video_path: &Path,
    output_h: usize,
    output_w: usize,
    filter: String,
    mut on_frame: F,
) -> Result<usize>
where
    F: FnMut(usize, Vec<u8>) -> Result<()>,
{
    let ffmpeg = command_from_env(
        "DIAL_FFMPEG_BIN",
        "ffmpeg",
        "/opt/sophon/sophon-ffmpeg-latest/bin/ffmpeg",
    );
    let mut child = Command::new(&ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path
                .to_str()
                .ok_or_else(|| anyhow!("video temp path is not valid UTF-8"))?,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &filter,
            "-vsync",
            "0",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow!("cannot start ffmpeg ({ffmpeg}): {e}"))?;

    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("ffmpeg stderr unavailable"))?;
    let error_thread = thread::spawn(move || {
        let mut message = String::new();
        let _ = stderr.read_to_string(&mut message);
        message
    });
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("ffmpeg stdout unavailable"))?;

    let frame_bytes = output_h
        .checked_mul(output_w)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| anyhow!("decoded video frame size overflow"))?;
    let mut frame_count = 0usize;
    let decode_result = loop {
        let mut frame = vec![0u8; frame_bytes];
        match stdout.read(&mut frame[..1]) {
            Ok(0) => break Ok(()),
            Ok(_) => {}
            Err(error) => break Err(anyhow!("reading decoded video failed: {error}")),
        }
        if let Err(error) = stdout.read_exact(&mut frame[1..]) {
            break Err(anyhow!("ffmpeg returned a partial video frame: {error}"));
        }
        if let Err(error) = on_frame(frame_count, frame) {
            break Err(error);
        }
        frame_count += 1;
    };

    if decode_result.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    let stderr = error_thread.join().unwrap_or_default();
    decode_result?;
    if !status.success() {
        bail!("ffmpeg video decode failed: {}", stderr.trim());
    }
    if frame_count == 0 {
        bail!("video contains no decodable frames");
    }
    Ok(frame_count)
}

/// 把任意尺寸的 RGB 图像，缩放到符合模型输入要求的标准尺寸
fn smart_resize_like_pipeline(
    rgb: Vec<u8>,
    h: usize,
    w: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    num_position_embeddings: usize,
    max_side_override: Option<u32>,
    allow_upscale: bool,
) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow!("invalid rgb buffer"))?;

    let (th, tw) = smart_resize_dimensions(
        h,
        w,
        patch_size,
        spatial_merge_size,
        num_position_embeddings,
        max_side_override,
        allow_upscale,
    )?;

    let resized = image::imageops::resize(
        &img,
        tw as u32,
        th as u32,
        // 比 Triangle(双线性) 保细节一些，更适合图文场景（类似 bicubic）。
        image::imageops::FilterType::CatmullRom,
    );
    let (rw, rh) = resized.dimensions();
    Ok((resized.into_raw(), rh as usize, rw as usize))
}

fn smart_resize_dimensions(
    h: usize,
    w: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    num_position_embeddings: usize,
    max_side_override: Option<u32>,
    allow_upscale: bool,
) -> Result<(usize, usize)> {
    if h == 0 || w == 0 {
        bail!("video/image dimensions must be positive, got {w}x{h}");
    }

    // 这里用 factor=patch_size*spatial_merge_size（默认 32）保证：
    // - 像素尺寸能整除 patch_size（生成 patch grid）
    // - patch grid 能整除 spatial_merge_size（避免 encode() 再次裁剪）
    let factor = (patch_size * spatial_merge_size).max(1) as u32;

    // pipeline.py 里 image 的 min_pixels 传的是 4*32*32（即 4096），这里按同样思路取默认最小像素数。
    let min_pixels: u64 = 4u64 * factor as u64 * factor as u64;

    // 最大像素数由位置嵌入容量决定：最多 num_position_embeddings 个 patch token，
    // 每个 patch 对应 patch_size^2 像素。
    let max_pixels: u64 = num_position_embeddings as u64 * patch_size as u64 * patch_size as u64;

    let oh = h as u64;
    let ow = w as u64;
    let op = oh * ow;

    // 计算缩放比例，使得总像素落在 [min_pixels, max_pixels] 之间（保持宽高比）。
    let mut scale = 1.0f64;
    if op > max_pixels {
        scale = (max_pixels as f64 / op as f64).sqrt();
    } else if op < min_pixels && allow_upscale {
        scale = (min_pixels as f64 / op as f64).sqrt();
    }

    let mut th = ((h as f64) * scale).round().max(factor as f64) as u32;
    let mut tw = ((w as f64) * scale).round().max(factor as f64) as u32;

    // 额外约束：把 patch grid 的长宽都限制在 base_grid(48) 以内。
    // 原因：当前 Rust 版 VisionEncoder 的位置编码是固定表（48x48），不做插值。
    // 如果出现 hp>48 或 wp>48，会导致位置编码与网格不匹配，模型容易“看不懂图”。
    let base_side = (num_position_embeddings as f64).sqrt() as u32;
    if base_side * base_side != num_position_embeddings as u32 {
        bail!(
            "num_position_embeddings {} is not a square",
            num_position_embeddings
        );
    }
    let default_max_side_pixels = base_side * patch_size as u32;
    let max_side_pixels = max_side_override
        .filter(|v| *v > 0)
        .map(|v| v.min(default_max_side_pixels))
        .unwrap_or(default_max_side_pixels);
    let max_dim = th.max(tw);
    if max_dim > max_side_pixels {
        let s = max_side_pixels as f64 / max_dim as f64;
        th = ((th as f64) * s).floor().max(factor as f64) as u32;
        tw = ((tw as f64) * s).floor().max(factor as f64) as u32;
    }

    // 对齐到 factor 的整数倍（不做 crop，而是直接 resize 到对齐后的尺寸）。
    th = (th / factor).max(1) * factor;
    tw = (tw / factor).max(1) * factor;

    // 再次确认不会超过 max_pixels：如果超过则向下对齐一档。
    let capped_max_pixels = (max_side_pixels as u64) * (max_side_pixels as u64);
    let max_pixels = max_pixels.min(capped_max_pixels);
    while (th as u64) * (tw as u64) > max_pixels && th > factor && tw > factor {
        th -= factor;
        tw -= factor;
    }

    Ok((th.max(1) as usize, tw.max(1) as usize))
}

/// Resize to a fixed square side length (static shape), keeping aspect ratio with letterbox padding.
///
/// This is intended for hardware backends requiring static input shapes (e.g. BM1684 bmodel).
/// Padding uses mid-gray (128) so that after `(x/255 - 0.5)/0.5` it is ~0.
/// 把任意尺寸的长方形图像 → 缩放到固定大小的正方形，保持原图比例不变形，多余区域用灰色居中填充
fn resize_to_fixed_square_letterbox(
    rgb: Vec<u8>,
    h: usize,
    w: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    num_position_embeddings: usize,
    fixed_side: u32,
    allow_upscale: bool,
) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow!("invalid rgb buffer"))?;

    // Align side to the grid factor so patch/merge grids are exact and encode() won't crop.
    let factor = (patch_size * spatial_merge_size).max(1) as u32;

    // Cap by the learned position embedding table capacity (no interpolation in this implementation).
    let base_side = (num_position_embeddings as f64).sqrt() as u32;
    if base_side * base_side != num_position_embeddings as u32 {
        bail!(
            "num_position_embeddings {} is not a square",
            num_position_embeddings
        );
    }
    let max_side_pixels = base_side * patch_size as u32;

    let mut side = fixed_side.min(max_side_pixels).max(factor);
    side = (side / factor).max(1) * factor;

    // Fit into the square canvas.
    let max_dim = (h.max(w) as f64).max(1.0);
    let mut scale = side as f64 / max_dim;
    if !allow_upscale {
        scale = scale.min(1.0);
    }
    let new_h = ((h as f64) * scale).round().max(1.0) as u32;
    let new_w = ((w as f64) * scale).round().max(1.0) as u32;

    let resized = image::imageops::resize(
        &img,
        new_w.min(side).max(1),
        new_h.min(side).max(1),
        image::imageops::FilterType::CatmullRom,
    );
    let (rw, rh) = resized.dimensions();

    // Letterbox pad to square.
    let pad = image::Rgb([128u8, 128u8, 128u8]);
    let mut canvas = image::RgbImage::from_pixel(side, side, pad);
    let x0 = ((side - rw) / 2) as i64;
    let y0 = ((side - rh) / 2) as i64;
    image::imageops::overlay(&mut canvas, &resized, x0, y0);

    Ok((canvas.into_raw(), side as usize, side as usize))
}

/// Qwen3-VL 多模态大模型（文本 + 图像） 的核心数据结构定义.
pub struct Qwen3Vl {
    ctx: Context,

    // Language model pieces.
    tokenizer: Tokenizer,
    embedding: Embedding,
    ln_f: RmsNorm,
    lm_head: Option<Linear>,
    lm_head_q8: Option<QuantizedLmHead>,
    blocks: Vec<Box<dyn Forwarder>>,
    prefill_shadow_blocks: Vec<Option<Box<dyn Forwarder>>>,
    force_shadow_blocks_for_dialog: bool,
    text_rknn: Option<TextRknnRunner>,
    text_rknn_dir: Option<PathBuf>,
    text_rknn_lib: Option<PathBuf>,
    text_cfg: Config,
    text_decode_mode: TextDecodeMode,
    text_rknn_disabled_for_dialog: bool,
    text_rknn_disabled_for_runtime: bool,
    text_rknn_prefix_validated_for_dialog: bool,
    text_rknn_validated_last_layer: Option<usize>,

    // Vision model.
    vision_cfg: VisionConfig,
    vision: Option<VisionEncoder>,
    vision_rknn: Option<VisionRknn>,
    vision_rknn_path: Option<PathBuf>,
    vision_rknn_lib: Option<PathBuf>,
    vision_rknn_side: Option<u32>,
    vision_cache: HashMap<u64, VisionOutputs>,
    prompt_encoder: PromptEncoder,

    // Special tokens.
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    eos_token_id: Option<u32>,

    // Chat state.
    history: Vec<Message>,
    tokens: Vec<u32>,
    image_spans: Vec<(ImageSpan, Tensor)>,
    deepstack_spans: Vec<Vec<(ImageSpan, Tensor)>>,

    index_pos: usize,
    generated: usize,
    fast_logits_processor: FastLogitsProcessor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextDecodeMode {
    Auto,
    FullNpu,
    NpuCpu,
    CpuOnly,
}

impl TextDecodeMode {
    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "full-npu" => Self::FullNpu,
            "npu-cpu" | "npu-gpu" => Self::NpuCpu,
            "cpu-only" | "cpu-gpu" => Self::CpuOnly,
            _ => Self::Auto,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::FullNpu => "full-npu",
            Self::NpuCpu => "npu-cpu",
            Self::CpuOnly => "cpu-only",
        }
    }

    fn allows_text_rknn_decode(self) -> bool {
        !matches!(self, Self::CpuOnly)
    }

    fn needs_text_rknn_runtime(self, prefill_enabled: bool) -> bool {
        prefill_enabled || self.allows_text_rknn_decode()
    }

    fn allows_partial_coverage(self) -> bool {
        matches!(self, Self::Auto | Self::NpuCpu)
    }
}

#[derive(Default)]
struct TextStepProfile {
    prompt_ms: f64,
    embedding_ms: f64,
    image_inject_ms: f64,
    deepstack_inject_ms: f64,
    local_ms: f64,
    local_layers: usize,
    remote_ms: f64,
    remote_batches: usize,
    remote_layers: usize,
    text_rknn_ms: f64,
    final_norm_ms: f64,
    slice_ms: f64,
    lm_head_ms: f64,
    logits_to_f32_ms: f64,
    repeat_penalty_ms: f64,
    sample_ms: f64,
    token_decode_ms: f64,
    forward_total_ms: f64,
}

impl TextStepProfile {
    fn add_elapsed_ms(slot: &mut f64, started: Instant) {
        *slot += started.elapsed().as_secs_f64() * 1000.0;
    }

    fn bottleneck(&self) -> (&'static str, f64) {
        let items = [
            ("local", self.local_ms),
            ("remote", self.remote_ms),
            ("rknn", self.text_rknn_ms),
            ("lm_head", self.lm_head_ms),
            ("sample", self.sample_ms),
        ];
        items
            .into_iter()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or(("unknown", 0.0))
    }

    fn other_ms(&self, total_ms: f64) -> f64 {
        (total_ms
            - self.prompt_ms
            - self.embedding_ms
            - self.image_inject_ms
            - self.deepstack_inject_ms
            - self.local_ms
            - self.remote_ms
            - self.text_rknn_ms
            - self.final_norm_ms
            - self.slice_ms
            - self.lm_head_ms
            - self.logits_to_f32_ms
            - self.repeat_penalty_ms
            - self.sample_ms
            - self.token_decode_ms)
            .max(0.0)
    }

    fn pct(value_ms: f64, total_ms: f64) -> f64 {
        if total_ms > 0.0 {
            value_ms * 100.0 / total_ms
        } else {
            0.0
        }
    }
}

impl Qwen3Vl {
    const TEXT_RKNN_MIN_HEADROOM: usize = 64;
    const TEXT_RKNN_PREFIX_MAX_ABS_THRESHOLD: f32 = 1e-2;
    const TEXT_RKNN_PREFIX_RMS_THRESHOLD: f32 = 1e-3;
    const TEXT_RKNN_PREFIX_RELAXED_HIDDEN_MAX_ABS_THRESHOLD: f32 = 4.0;
    const TEXT_RKNN_PREFIX_RELAXED_HIDDEN_RMS_THRESHOLD: f32 = 1.5e-1;
    const TEXT_RKNN_PREFIX_DELTA_K_MAX_ABS_THRESHOLD: f32 = 1.0;
    const TEXT_RKNN_PREFIX_DELTA_K_RMS_THRESHOLD: f32 = 5.0e-2;
    const TEXT_RKNN_PREFIX_DELTA_V_MAX_ABS_THRESHOLD: f32 = 1.0e-2;
    const TEXT_RKNN_PREFIX_DELTA_V_RMS_THRESHOLD: f32 = 1.0e-3;

    fn text_profile_enabled() -> bool {
        matches!(
            std::env::var("QWEN3VL_PROFILE_TEXT").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        )
    }

    fn text_profile_summary_enabled() -> bool {
        matches!(
            std::env::var("QWEN3VL_PROFILE_SUMMARY").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        )
    }

    fn should_log_text_profile_step(index: usize) -> bool {
        index == 0 || index == 1 || index == 8 || index == 32 || index % 64 == 0
    }

    fn keep_vision_rknn_loaded() -> bool {
        !matches!(
            std::env::var("QWEN3VL_KEEP_VISION_RKNN").ok().as_deref(),
            Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO")
        )
    }

    fn prompt_profile_enabled() -> bool {
        matches!(
            std::env::var("QWEN3VL_PROFILE_PROMPT").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        )
    }

    fn vision_cache_enabled() -> bool {
        !matches!(
            std::env::var("QWEN3VL_VISION_CACHE").ok().as_deref(),
            Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO")
        )
    }

    fn vision_cache_key(rgb: &[u8], h: usize, w: usize, fixed_side: Option<u32>) -> u64 {
        let mut hasher = DefaultHasher::new();
        h.hash(&mut hasher);
        w.hash(&mut hasher);
        fixed_side.hash(&mut hasher);
        rgb.hash(&mut hasher);
        hasher.finish()
    }

    fn shadow_prefill_enabled() -> bool {
        matches!(
            std::env::var("QWEN3VL_SHADOW_PREFILL").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        )
    }

    fn shadow_prefill_promote_decode() -> bool {
        !matches!(
            std::env::var("QWEN3VL_SHADOW_PREFILL_PROMOTE_DECODE")
                .ok()
                .as_deref(),
            Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO")
        )
    }

    fn shadow_prefill_host(ctx: &Context) -> Option<String> {
        if let Ok(host) = std::env::var("QWEN3VL_SHADOW_PREFILL_HOST") {
            let host = host.trim();
            if !host.is_empty() {
                return Some(host.to_string());
            }
        }
        ctx.topology.values().next().map(|node| node.host.clone())
    }

    fn lm_head_forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(q8) = self.lm_head_q8.as_ref() {
            q8.forward(x)
        } else {
            self.lm_head
                .as_ref()
                .ok_or_else(|| anyhow!("lm_head is not available"))?
                .forward(x)
                .map_err(|e| anyhow!("lm_head.forward: {e}"))
        }
    }

    fn ensure_vision_rknn_loaded(&mut self) -> Result<()> {
        if self.vision_rknn.is_some() {
            return Ok(());
        }
        let Some(model_path) = self.vision_rknn_path.as_deref() else {
            return Ok(());
        };
        let lib_path = self.vision_rknn_lib.as_deref().map(Path::new);
        let rknn = VisionRknn::load(model_path, lib_path)?;
        if let Some(side) = self.vision_rknn_side {
            let actual = rknn.expected_side().ok_or_else(|| {
                anyhow!(
                    "rknn input must be square NCHW/NHWC with 3 channels, got {}",
                    rknn.input_summary()
                )
            })?;
            if actual != side {
                bail!(
                    "vision rknn side changed after reload: expected {}, got {}",
                    side,
                    actual
                );
            }
        }
        self.vision_rknn = Some(rknn);
        log::info!("vision rknn reloaded on demand");
        Ok(())
    }

    fn release_vision_rknn(&mut self, reason: &str) {
        if Self::keep_vision_rknn_loaded() {
            log::info!(
                "vision rknn context kept after {}; set QWEN3VL_KEEP_VISION_RKNN=0 to release it",
                reason
            );
            return;
        }
        if self.vision_rknn.take().is_some() {
            log::info!("vision rknn context released after {}", reason);
        }
    }

    fn release_text_rknn(&mut self, reason: &str) {
        if self.text_rknn.take().is_some() {
            log::info!("text rknn context released after {}", reason);
        }
    }

    fn release_text_mlp_rknn(&self, reason: &str) {
        let released: usize = self
            .blocks
            .iter()
            .map(|block| block.release_mlp_rknn(reason))
            .sum();
        if released > 0 {
            log::info!(
                "released {} text mlp rknn context(s) before {}",
                released,
                reason
            );
        }
    }

    fn sync_text_rknn_to_cache(&mut self, reason: &str) {
        if let Some(runner) = self.text_rknn.as_ref() {
            if let Err(err) =
                runner.sync_to_cache(&mut self.ctx.cache, &self.ctx.device, self.ctx.dtype)
            {
                log::warn!("text rknn cache sync before {} failed: {}", reason, err);
            }
        }
    }

    fn tensor_diff_stats(a: &[f32], b: &[f32]) -> Result<(f32, f32)> {
        if a.len() != b.len() {
            bail!("tensor diff length mismatch: {} != {}", a.len(), b.len());
        }
        if a.is_empty() {
            return Ok((0.0, 0.0));
        }
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > max_abs {
                max_abs = d;
            }
            let df = (x - y) as f64;
            sum_sq += df * df;
        }
        let rms = (sum_sq / a.len() as f64).sqrt() as f32;
        Ok((max_abs, rms))
    }

    fn last_kv_token_cache_layout(
        data: &[f32],
        heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let expected = heads
            .checked_mul(seq)
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(|| {
                anyhow!("kv shape overflow: heads={heads} seq={seq} head_dim={head_dim}")
            })?;
        if data.len() != expected {
            bail!(
                "kv length mismatch for last-token slice: data={} expected={} heads={} seq={} head_dim={}",
                data.len(),
                expected,
                heads,
                seq,
                head_dim
            );
        }
        if seq == 0 {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(heads * head_dim);
        for h in 0..heads {
            let offset = h * seq * head_dim + (seq - 1) * head_dim;
            out.extend_from_slice(&data[offset..offset + head_dim]);
        }
        Ok(out)
    }

    async fn validate_text_rknn_prefix_once(&mut self, x: &Tensor, idx: usize) -> Result<()> {
        if self.text_rknn_prefix_validated_for_dialog || self.text_rknn_disabled_for_dialog {
            return Ok(());
        }
        let Some(_) = self.text_rknn.as_mut() else {
            return Ok(());
        };
        loop {
            let (decode_last_layer, rknn_out) = {
                let runner = self
                    .text_rknn
                    .as_mut()
                    .ok_or_else(|| anyhow!("text rknn runner missing during prefix validation"))?;
                let decode_last_layer = runner.decode_last_layer();
                let rknn_out = runner.forward_decode(
                    x,
                    idx,
                    &self.ctx.cache,
                    &self.ctx.device,
                    self.ctx.dtype,
                )?;
                (decode_last_layer, rknn_out)
            };

            let mut ref_cache = self.ctx.cache.clone();
            let mut ref_x = x.clone();
            for block_idx in 0..=decode_last_layer {
                ref_x = self.blocks[block_idx]
                    .forward_mut(&ref_x, idx, block_idx, &mut ref_cache)
                    .await
                    .map_err(|e| {
                        anyhow!("text rknn prefix reference block {block_idx} failed: {e}")
                    })?;
            }

            let rknn_vec = rknn_out
                .to_device(&candle_core::Device::Cpu)?
                .to_dtype(candle_core::DType::F32)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let ref_vec = ref_x
                .to_device(&candle_core::Device::Cpu)?
                .to_dtype(candle_core::DType::F32)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let (max_abs, rms) = Self::tensor_diff_stats(&rknn_vec, &ref_vec)?;
            log::info!(
                "text rknn prefix validation: layers=0..{} past_len={} hidden max_abs={:.6e} rms={:.6e}",
                decode_last_layer,
                idx,
                max_abs,
                rms
            );
            let mut saw_delta_debug = false;
            let mut worst_delta_k_max_abs = 0.0f32;
            let mut worst_delta_k_rms = 0.0f32;
            let mut worst_delta_v_max_abs = 0.0f32;
            let mut worst_delta_v_rms = 0.0f32;
            let mut worst_delta_k_layer = None;
            let mut worst_delta_v_layer = None;
            if let Some(runner) = self.text_rknn.as_ref() {
                let kv_heads = self.text_cfg.num_key_value_heads;
                let head_dim = self.text_cfg.hidden_size / self.text_cfg.num_attention_heads;
                for layer_idx in 0..=decode_last_layer {
                    let Some((rk_k, rk_v, rk_len)) = runner.layer_kv_cache_f32(layer_idx) else {
                        continue;
                    };
                    let Some((ref_k, ref_v)) = ref_cache.kv_clone(layer_idx) else {
                        continue;
                    };
                    let ref_k_vec = ref_k
                        .to_device(&candle_core::Device::Cpu)?
                        .to_dtype(candle_core::DType::F32)?
                        .contiguous()?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let ref_v_vec = ref_v
                        .to_device(&candle_core::Device::Cpu)?
                        .to_dtype(candle_core::DType::F32)?
                        .contiguous()?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    if rk_k.len() == ref_k_vec.len() && rk_v.len() == ref_v_vec.len() {
                        let (k_max_abs, k_rms) = Self::tensor_diff_stats(rk_k, &ref_k_vec)?;
                        let (v_max_abs, v_rms) = Self::tensor_diff_stats(rk_v, &ref_v_vec)?;
                        log::info!(
                            "text rknn prefix validation kv-cache: layer={} len={} k max_abs={:.6e} rms={:.6e} v max_abs={:.6e} rms={:.6e}",
                            layer_idx,
                            rk_len,
                            k_max_abs,
                            k_rms,
                            v_max_abs,
                            v_rms
                        );
                    } else {
                        log::warn!(
                            "text rknn prefix validation kv length mismatch: layer={} rknn_k={} ref_k={} rknn_v={} ref_v={}",
                            layer_idx,
                            rk_k.len(),
                            ref_k_vec.len(),
                            rk_v.len(),
                            ref_v_vec.len()
                        );
                    }
                    if let Some(debug) = runner.layer_delta_layout_debug(layer_idx) {
                        let ref_seq = ref_k_vec.len() / (kv_heads * head_dim);
                        if ref_seq > 0 && ref_v_vec.len() == ref_k_vec.len() {
                            let ref_k_delta = Self::last_kv_token_cache_layout(
                                &ref_k_vec, kv_heads, ref_seq, head_dim,
                            )?;
                            let ref_v_delta = Self::last_kv_token_cache_layout(
                                &ref_v_vec, kv_heads, ref_seq, head_dim,
                            )?;
                            let (k_hsd_max, k_hsd_rms) =
                                Self::tensor_diff_stats(&debug.k_head_seq_dim, &ref_k_delta)?;
                            let (k_sdh_max, k_sdh_rms) =
                                Self::tensor_diff_stats(&debug.k_seq_dim_head, &ref_k_delta)?;
                            let (v_hsd_max, v_hsd_rms) =
                                Self::tensor_diff_stats(&debug.v_head_seq_dim, &ref_v_delta)?;
                            let (v_sdh_max, v_sdh_rms) =
                                Self::tensor_diff_stats(&debug.v_seq_dim_head, &ref_v_delta)?;
                            saw_delta_debug = true;
                            if k_hsd_max > worst_delta_k_max_abs || k_hsd_rms > worst_delta_k_rms {
                                worst_delta_k_layer = Some(layer_idx);
                            }
                            if v_hsd_max > worst_delta_v_max_abs || v_hsd_rms > worst_delta_v_rms {
                                worst_delta_v_layer = Some(layer_idx);
                            }
                            worst_delta_k_max_abs = worst_delta_k_max_abs.max(k_hsd_max);
                            worst_delta_k_rms = worst_delta_k_rms.max(k_hsd_rms);
                            worst_delta_v_max_abs = worst_delta_v_max_abs.max(v_hsd_max);
                            worst_delta_v_rms = worst_delta_v_rms.max(v_hsd_rms);
                            log::info!(
                                "text rknn prefix validation kv-layout delta: layer={} len={} k HeadSeqDim max_abs={:.6e} rms={:.6e} k SeqDimHead max_abs={:.6e} rms={:.6e} v HeadSeqDim max_abs={:.6e} rms={:.6e} v SeqDimHead max_abs={:.6e} rms={:.6e}",
                                layer_idx,
                                ref_seq,
                                k_hsd_max,
                                k_hsd_rms,
                                k_sdh_max,
                                k_sdh_rms,
                                v_hsd_max,
                                v_hsd_rms,
                                v_sdh_max,
                                v_sdh_rms
                            );
                        }
                    }
                }
            }
            let strict_ok = max_abs <= Self::TEXT_RKNN_PREFIX_MAX_ABS_THRESHOLD
                && rms <= Self::TEXT_RKNN_PREFIX_RMS_THRESHOLD;
            let relaxed_ok = saw_delta_debug
                && max_abs <= Self::TEXT_RKNN_PREFIX_RELAXED_HIDDEN_MAX_ABS_THRESHOLD
                && rms <= Self::TEXT_RKNN_PREFIX_RELAXED_HIDDEN_RMS_THRESHOLD
                && worst_delta_k_max_abs <= Self::TEXT_RKNN_PREFIX_DELTA_K_MAX_ABS_THRESHOLD
                && worst_delta_k_rms <= Self::TEXT_RKNN_PREFIX_DELTA_K_RMS_THRESHOLD
                && worst_delta_v_max_abs <= Self::TEXT_RKNN_PREFIX_DELTA_V_MAX_ABS_THRESHOLD
                && worst_delta_v_rms <= Self::TEXT_RKNN_PREFIX_DELTA_V_RMS_THRESHOLD;
            if strict_ok || relaxed_ok {
                if relaxed_ok && !strict_ok {
                    log::info!(
                        "text rknn prefix validation accepted with relaxed thresholds: hidden max_abs={:.6e} rms={:.6e}; delta k max_abs={:.6e} rms={:.6e}; delta v max_abs={:.6e} rms={:.6e}",
                        max_abs,
                        rms,
                        worst_delta_k_max_abs,
                        worst_delta_k_rms,
                        worst_delta_v_max_abs,
                        worst_delta_v_rms
                    );
                }
                self.text_rknn_prefix_validated_for_dialog = true;
                self.text_rknn_validated_last_layer = Some(decode_last_layer);
                self.release_text_rknn("prefix validation");
                self.ensure_text_rknn_loaded()?;
                return Ok(());
            }

            let can_shrink = self
                .text_rknn
                .as_ref()
                .map(|runner| runner.can_shrink_decode_coverage())
                .unwrap_or(false);
            if can_shrink {
                let (old_last, new_last) = self
                    .text_rknn
                    .as_mut()
                    .ok_or_else(|| anyhow!("text rknn runner missing during shrink"))?
                    .shrink_last_chunk()?;
                log::warn!(
                    "text rknn prefix validation mismatch on layers 0..{} (max_abs={:.6e}, rms={:.6e}); shrink local rknn coverage to 0..{} and retry",
                    old_last,
                    max_abs,
                    rms,
                    new_last
                );
                continue;
            }

            self.text_rknn_disabled_for_dialog = true;
            self.text_rknn_disabled_for_runtime = true;
            self.release_text_rknn("prefix validation mismatch");
            bail!(
                "text rknn prefix validation mismatch: hidden max_abs={:.6e} rms={:.6e} exceeds strict thresholds ({:.6e}, {:.6e}) and relaxed gate failed (delta_seen={} delta_k layer={:?} max_abs={:.6e} rms={:.6e} delta_v layer={:?} max_abs={:.6e} rms={:.6e})",
                max_abs,
                rms,
                Self::TEXT_RKNN_PREFIX_MAX_ABS_THRESHOLD,
                Self::TEXT_RKNN_PREFIX_RMS_THRESHOLD,
                saw_delta_debug,
                worst_delta_k_layer,
                worst_delta_k_max_abs,
                worst_delta_k_rms,
                worst_delta_v_layer,
                worst_delta_v_max_abs,
                worst_delta_v_rms
            );
        }
    }

    fn encode_vision_image(&mut self, img: &Tensor) -> Result<VisionOutputs> {
        if self.vision_rknn_path.is_some() {
            if self.vision_rknn.is_none() {
                self.release_text_mlp_rknn("vision rknn reload");
            }
            self.ensure_vision_rknn_loaded()?;
        }
        if let Some(rknn) = &self.vision_rknn {
            rknn.encode(img, &self.ctx.device, self.ctx.dtype)
        } else {
            self.vision
                .as_ref()
                .ok_or_else(|| anyhow!("vision encoder is not loaded"))?
                .encode(img)
        }
    }

    fn encode_vision_image_cached(
        &mut self,
        cache_key: u64,
        img: &Tensor,
        prompt_profile: bool,
        vision_encode_ms: &mut f64,
    ) -> Result<VisionOutputs> {
        if Self::vision_cache_enabled() {
            if let Some(cached) = self.vision_cache.get(&cache_key) {
                log::info!("vision embedding cache hit key={cache_key:016x}");
                return Ok(cached.clone());
            }
        }

        let started = prompt_profile.then(Instant::now);
        let outputs = self.encode_vision_image(img)?;
        if let Some(started) = started {
            *vision_encode_ms += started.elapsed().as_secs_f64() * 1000.0;
        }
        if Self::vision_cache_enabled() {
            log::info!("vision embedding cache store key={cache_key:016x}");
            self.vision_cache.insert(cache_key, outputs.clone());
        }
        Ok(outputs)
    }

    fn encode_video_bytes(
        &self,
        bytes: Vec<u8>,
        prompt_profile: bool,
        vision_encode_ms: &mut f64,
    ) -> Result<(VisualTokenSpec, Vec<Tensor>, Vec<Vec<Tensor>>, usize)> {
        if self.vision_rknn_path.is_some() {
            bail!(
                "native video analysis requires the software vision encoder; start the Master without --vision-rknn"
            );
        }
        let encoder = self
            .vision
            .as_ref()
            .ok_or_else(|| anyhow!("vision encoder is not loaded"))?
            .clone();
        let factor = (self.vision_cfg.patch_size * self.vision_cfg.spatial_merge_size).max(1);
        if self.ctx.args.video_max_side < factor as u32 {
            bail!(
                "--video-max-side must be at least {} for this model",
                factor
            );
        }
        if self.ctx.args.video_max_frames == 0 {
            bail!("--video-max-frames must be greater than zero");
        }
        // MP4/MOV commonly stores seek-dependent metadata. Give ffmpeg a
        // seekable temporary source instead of piping the container through stdin.
        let temporary_video = TemporaryVideoFile::create(&bytes)?;
        drop(bytes);
        let probe = probe_video(temporary_video.path())?;
        let total_frames = probe.estimated_frames.ok_or_else(|| {
            anyhow!("could not determine video frame count for official sampling")
        })?;
        let sample_indices = if self.ctx.args.video_no_sample {
            if total_frames > self.ctx.args.video_max_frames {
                bail!(
                    "video contains {} frames, exceeding --video-max-frames {} in --video-no-sample mode",
                    total_frames,
                    self.ctx.args.video_max_frames
                );
            }
            (0..total_frames).collect::<Vec<_>>()
        } else {
            uniform_sample_indices(
                total_frames,
                probe.fps,
                self.ctx.args.video_fps,
                self.ctx.args.video_min_frames,
                self.ctx.args.video_max_frames,
            )?
        };
        if sample_indices.is_empty() {
            bail!("video sampling produced no frames");
        }

        let (output_h, output_w) = smart_resize_dimensions(
            probe.height,
            probe.width,
            self.vision_cfg.patch_size,
            self.vision_cfg.spatial_merge_size,
            self.vision_cfg.num_position_embeddings,
            Some(self.ctx.args.video_max_side),
            !self.ctx.args.vision_no_upscale,
        )?;
        let scale = format!("scale={output_w}:{output_h}:flags=bicubic");
        let filter = if self.ctx.args.video_no_sample {
            scale
        } else {
            ffmpeg_select_filter(&sample_indices, &scale)
        };

        let temporal = self.vision_cfg.temporal_patch_size.max(1);
        let device = self.ctx.device.clone();
        let dtype = self.ctx.dtype;
        let expected_frames = sample_indices.len();
        let batch_size = self.ctx.args.video_batch_size;
        if batch_size == 0 {
            bail!("--video-batch-size must be greater than zero");
        }
        let mut sampled_frames: Vec<(usize, Vec<u8>)> = Vec::with_capacity(expected_frames);
        let mut patch_specs = Vec::new();
        let mut embeds = Vec::new();
        let mut deepstack_embeds = Vec::new();

        let frame_count = decode_all_video_frames(
            temporary_video.path(),
            output_h,
            output_w,
            filter,
            |frame_index, rgb| {
                let source_index = *sample_indices
                    .get(frame_index)
                    .ok_or_else(|| anyhow!("ffmpeg returned more frames than the sampling plan"))?;
                sampled_frames.push((source_index, rgb));
                Ok(())
            },
        )?;
        if frame_count != expected_frames {
            bail!(
                "ffmpeg returned {} processed frames, expected {}; video was not silently truncated",
                frame_count,
                expected_frames
            );
        }

        // Qwen3-VL pads a short final temporal patch with its last frame.
        let mut patch_groups: Vec<Vec<(usize, Vec<u8>)>> = Vec::new();
        for chunk in sampled_frames.chunks(temporal) {
            let mut group = chunk.to_vec();
            while group.len() < temporal {
                group.push(
                    group
                        .last()
                        .cloned()
                        .ok_or_else(|| anyhow!("cannot pad an empty temporal video patch"))?,
                );
            }
            patch_groups.push(group);
        }

        for batch in patch_groups.chunks(batch_size) {
            let started = prompt_profile.then(Instant::now);
            let mut batch_tensors = Vec::with_capacity(batch.len());
            for group in batch {
                let mut tensors = Vec::with_capacity(temporal);
                for (_, rgb) in group {
                    tensors
                        .push(image_to_tensor(rgb, output_h, output_w, &device)?.to_dtype(dtype)?);
                }
                let refs: Vec<&Tensor> = tensors.iter().collect();
                batch_tensors.push(Tensor::cat(&refs, 0)?);
            }
            let refs: Vec<&Tensor> = batch_tensors.iter().collect();
            let frames = Tensor::stack(&refs, 0)?;
            let VisionOutputs {
                embeds: batch_embeds,
                deepstack: batch_deepstack,
            } = encoder.encode_video_patches(&frames)?;
            if let Some(started) = started {
                *vision_encode_ms += started.elapsed().as_secs_f64() * 1000.0;
            }
            let (encoded_batch, _seq, _hidden) = batch_embeds.dims3()?;
            if encoded_batch != batch.len() {
                bail!(
                    "vision encoder returned {} video patches, expected {}",
                    encoded_batch,
                    batch.len()
                );
            }
            for (batch_index, group) in batch.iter().enumerate() {
                let patch_embeds = batch_embeds.narrow(0, batch_index, 1)?.squeeze(0)?;
                let token_count = patch_embeds.dims2()?.0;
                let first_index = group.first().unwrap().0;
                let last_index = group.last().unwrap().0;
                let timestamp_s = (first_index as f64 + last_index as f64) / (2.0 * probe.fps);
                patch_specs.push(VideoPatchSpec {
                    timestamp_s,
                    token_count,
                });
                embeds.push(patch_embeds);
                let mut patch_deepstack = Vec::with_capacity(batch_deepstack.len());
                for output in &batch_deepstack {
                    patch_deepstack.push(output.narrow(0, batch_index, 1)?.squeeze(0)?);
                }
                deepstack_embeds.push(patch_deepstack);
            }
        }
        log::info!(
            "video encoded with {} sampling: source={}x{} fps={:.3} source_frames={} processed_frames={} temporal_patches={} output={}x{}",
            if self.ctx.args.video_no_sample { "no" } else { "official" },
            probe.width,
            probe.height,
            probe.fps,
            total_frames,
            frame_count,
            patch_specs.len(),
            output_w,
            output_h
        );
        Ok((
            VisualTokenSpec::Video {
                temporal_patches: patch_specs,
            },
            embeds,
            deepstack_embeds,
            frame_count,
        ))
    }

    fn ensure_text_rknn_loaded(&mut self) -> Result<()> {
        if !self
            .text_decode_mode
            .needs_text_rknn_runtime(self.ctx.args.text_rknn_prefill)
        {
            return Ok(());
        }
        if self.text_rknn_disabled_for_runtime {
            return Ok(());
        }
        if self.text_rknn_disabled_for_dialog {
            return Ok(());
        }
        if self.text_rknn.is_some() {
            return Ok(());
        }
        let Some(model_dir) = self.text_rknn_dir.clone() else {
            return Ok(());
        };
        let lib_path = self.text_rknn_lib.as_deref().map(Path::new);
        let mut runner = TextRknnRunner::load(model_dir.as_path(), lib_path, &self.text_cfg)?;
        if let Some(validated_last_layer) = self.text_rknn_validated_last_layer {
            while runner.decode_last_layer() > validated_last_layer {
                let (old_last, new_last) = runner.shrink_last_chunk()?;
                log::info!(
                    "text rknn decode coverage limited by prefix validation: shrink 0..{} to 0..{}",
                    old_last,
                    new_last
                );
            }
        }
        if !runner.decode_covers_all_layers() {
            if !self.text_decode_mode.allows_text_rknn_decode() && self.ctx.args.text_rknn_prefill {
                log::info!(
                    "text rknn prefill-only runtime enabled from {} (decode cover=0..{}, decode path stays on CPU/software)",
                    model_dir.display(),
                    runner.decode_last_layer()
                );
            } else if self.text_decode_mode.allows_partial_coverage() {
                log::info!(
                    "text rknn partial decode enabled from {} (cover=0..{}, full=false); remaining layers will fallback to software/remote path",
                    model_dir.display(),
                    runner.decode_last_layer()
                );
            } else {
                log::warn!(
                    "text rknn decode disabled for this dialog: partial coverage 0..{} is not allowed in {:?} mode",
                    runner.decode_last_layer(),
                    self.text_decode_mode
                );
                self.text_rknn_disabled_for_dialog = true;
                self.text_rknn_disabled_for_runtime = true;
                return Ok(());
            }
        }
        log::info!(
            "text rknn runtime enabled from {} (decode cover=0..{}, decode_full={})",
            model_dir.display(),
            runner.decode_last_layer(),
            runner.decode_covers_all_layers()
        );
        self.text_rknn = Some(runner);
        Ok(())
    }

    async fn forward_blocks_from(
        &mut self,
        mut x: Tensor,
        idx: usize,
        inject_images: bool,
        start_layer: usize,
        mut step_profile: Option<&mut TextStepProfile>,
    ) -> Result<Tensor> {
        let num_blocks = self.blocks.len();
        let mut block_idx = start_layer;
        let profile = Self::text_profile_enabled();

        while block_idx < num_blocks {
            let curr_block_id = self.blocks[block_idx].ident().to_owned();
            if inject_images {
                let started = step_profile.as_ref().map(|_| Instant::now());
                x = self.inject_deepstack_embeddings(&x, block_idx)?;
                if let (Some(profile), Some(started)) = (step_profile.as_mut(), started) {
                    TextStepProfile::add_elapsed_ms(&mut profile.deepstack_inject_ms, started);
                }
            }
            if curr_block_id == "local" {
                let use_shadow = self.force_shadow_blocks_for_dialog
                    || (idx == 0
                        && x.dims().get(1).copied().unwrap_or_default() > 1
                        && self
                            .prefill_shadow_blocks
                            .get(block_idx)
                            .and_then(Option::as_ref)
                            .is_some());
                if use_shadow {
                    if idx == 0
                        && x.dims().get(1).copied().unwrap_or_default() > 1
                        && Self::shadow_prefill_promote_decode()
                    {
                        self.force_shadow_blocks_for_dialog = true;
                    }
                    let first = block_idx;
                    let shadow_ident = self
                        .prefill_shadow_blocks
                        .get(block_idx)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| anyhow!("shadow block missing for local layer {block_idx}"))?
                        .ident()
                        .to_string();
                    let mut batch = vec![];
                    while block_idx < num_blocks
                        && self.blocks[block_idx].ident() == "local"
                        && self
                            .prefill_shadow_blocks
                            .get(block_idx)
                            .and_then(Option::as_ref)
                            .is_some_and(|block| block.ident() == shadow_ident)
                    {
                        batch.push((
                            self.blocks[block_idx].layer_name().to_string(),
                            idx,
                            block_idx,
                        ));
                        block_idx += 1;
                    }
                    let started = profile.then(Instant::now);
                    let summary_started = step_profile.as_ref().map(|_| Instant::now());
                    let last = block_idx.saturating_sub(1);
                    let num_shadow_layers = last.saturating_sub(first) + 1;
                    let shadow = self
                        .prefill_shadow_blocks
                        .get_mut(first)
                        .and_then(Option::as_mut)
                        .ok_or_else(|| anyhow!("shadow block missing for local layer {first}"))?;
                    x = shadow
                        .forward_batch(&x, batch, &mut self.ctx.cache)
                        .await
                        .map_err(|e| {
                            anyhow!(
                                "error in shadow forward batch operation for local blocks {}..{}: {e}",
                                first,
                                last
                            )
                        })?;
                    if let (Some(profile), Some(started)) = (step_profile.as_mut(), summary_started)
                    {
                        TextStepProfile::add_elapsed_ms(&mut profile.remote_ms, started);
                        profile.remote_batches += 1;
                        profile.remote_layers += num_shadow_layers;
                    }
                    if let Some(started) = started {
                        log::info!(
                            "profile text shadow layers={}..{} idx={} elapsed_ms={:.3}",
                            first,
                            last,
                            idx,
                            started.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    continue;
                }

                let started = profile.then(Instant::now);
                let summary_started = step_profile.as_ref().map(|_| Instant::now());
                let layer_idx = block_idx;
                x = self.blocks[block_idx]
                    .forward_mut(&x, idx, block_idx, &mut self.ctx.cache)
                    .await
                    .map_err(|e| {
                        anyhow!("error in forward operation of local block {block_idx}: {e}")
                    })?;
                if let (Some(profile), Some(started)) = (step_profile.as_mut(), summary_started) {
                    TextStepProfile::add_elapsed_ms(&mut profile.local_ms, started);
                    profile.local_layers += 1;
                }
                if let Some(started) = started {
                    log::info!(
                        "profile text local layer={} idx={} elapsed_ms={:.3}",
                        layer_idx,
                        idx,
                        started.elapsed().as_secs_f64() * 1000.0
                    );
                }
                block_idx += 1;
            } else {
                let mut batch = vec![];
                let first = block_idx;
                while block_idx < num_blocks && self.blocks[block_idx].ident() == curr_block_id {
                    batch.push((
                        self.blocks[block_idx].layer_name().to_string(),
                        idx,
                        block_idx,
                    ));
                    block_idx += 1;
                }
                let started = profile.then(Instant::now);
                let summary_started = step_profile.as_ref().map(|_| Instant::now());
                let last = block_idx.saturating_sub(1);
                let num_remote_layers = last.saturating_sub(first) + 1;
                x = self.blocks[first]
                    .forward_batch(&x, batch, &mut self.ctx.cache)
                    .await
                    .map_err(|e| {
                        anyhow!("error in forward batch operation for block {block_idx}: {e}")
                    })?;
                if let (Some(profile), Some(started)) = (step_profile.as_mut(), summary_started) {
                    TextStepProfile::add_elapsed_ms(&mut profile.remote_ms, started);
                    profile.remote_batches += 1;
                    profile.remote_layers += num_remote_layers;
                }
                if let Some(started) = started {
                    log::info!(
                        "profile text remote layers={}..{} idx={} elapsed_ms={:.3}",
                        first,
                        last,
                        idx,
                        started.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
        }

        Ok(x)
    }

    async fn forward_embeds(
        &mut self,
        mut x: Tensor,
        idx: usize,
        inject_images: bool,
        mut step_profile: Option<&mut TextStepProfile>,
    ) -> Result<Tensor> {
        let (_batch_size, seq_len, hidden_size) = x.dims3()?;

        if inject_images && !self.image_spans.is_empty() {
            let started = step_profile.as_ref().map(|_| Instant::now());
            x = self.inject_image_embeddings(&x)?;
            if let (Some(profile), Some(started)) = (step_profile.as_mut(), started) {
                TextStepProfile::add_elapsed_ms(&mut profile.image_inject_ms, started);
            }
        }

        if self.text_decode_mode.allows_text_rknn_decode()
            && seq_len == 1
            && idx > 0
            && !inject_images
            && self.ctx.cache.with_kv_cache()
        {
            self.ensure_text_rknn_loaded()?;
            if let Some(runner) = self.text_rknn.as_ref() {
                let max_bucket = runner.max_supported_bucket();
                if self.text_decode_mode.allows_partial_coverage() {
                    let required_seq = idx.max(1);
                    if max_bucket > 0 && required_seq > max_bucket {
                        log::info!(
                            "text rknn decode fallback for this dialog: past_len={} requires past bucket {} but max bucket is {}; sync prefix kv back to cache and continue on software/remote path",
                            idx,
                            required_seq,
                            max_bucket
                        );
                        self.text_rknn_disabled_for_dialog = true;
                        self.sync_text_rknn_to_cache("decode max-bucket fallback");
                        self.release_text_rknn("decode max-bucket fallback");
                    }
                } else {
                    let headroom = max_bucket.saturating_sub(idx);
                    if max_bucket > 0 && headroom < Self::TEXT_RKNN_MIN_HEADROOM {
                        log::info!(
                            "text rknn decode skipped for this dialog: past_len={} too close to max bucket {} (headroom={} < {})",
                            idx,
                            max_bucket,
                            headroom,
                            Self::TEXT_RKNN_MIN_HEADROOM
                        );
                        self.text_rknn_disabled_for_dialog = true;
                        self.sync_text_rknn_to_cache("decode headroom guard");
                        self.release_text_rknn("decode headroom guard");
                    }
                }
            }
        }

        let use_text_rknn_decode = self.text_decode_mode.allows_text_rknn_decode()
            && self.text_rknn.is_some()
            && seq_len == 1
            && idx > 0
            && !inject_images
            && self.ctx.cache.with_kv_cache();
        if use_text_rknn_decode {
            if let Err(err) = self.validate_text_rknn_prefix_once(&x, idx).await {
                log::warn!(
                    "text rknn prefix validation failed: {}; disable text rknn and fallback to software/remote decode path",
                    err
                );
                x = self
                    .forward_blocks_from(x, idx, false, 0, step_profile.as_deref_mut())
                    .await?;
                let ln_started = step_profile.as_ref().map(|_| Instant::now());
                let x = self
                    .ln_f
                    .forward(&x)
                    .map_err(|e| anyhow!("ln_f.forward: {e}"))?;
                if let (Some(profile), Some(started)) = (step_profile.as_mut(), ln_started) {
                    TextStepProfile::add_elapsed_ms(&mut profile.final_norm_ms, started);
                }
                let slice_started = step_profile.as_ref().map(|_| Instant::now());
                let x = x
                    .i((.., seq_len - 1, ..))
                    .map_err(|e| anyhow!("x.i: {e}"))?
                    .contiguous()
                    .map_err(|e| anyhow!("x.i.contiguous: {e}"))?;
                if let (Some(profile), Some(started)) = (step_profile.as_mut(), slice_started) {
                    TextStepProfile::add_elapsed_ms(&mut profile.slice_ms, started);
                }
                return Ok(x);
            }
            let decode_result = {
                let runner = self
                    .text_rknn
                    .as_mut()
                    .ok_or_else(|| anyhow!("text rknn runner not initialized"))?;
                let last_layer = runner.decode_last_layer();
                let rknn_started = step_profile.as_ref().map(|_| Instant::now());
                match runner.forward_decode(
                    &x,
                    idx,
                    &self.ctx.cache,
                    &self.ctx.device,
                    self.ctx.dtype,
                ) {
                    Ok(out) => {
                        if let (Some(profile), Some(started)) =
                            (step_profile.as_mut(), rknn_started)
                        {
                            TextStepProfile::add_elapsed_ms(&mut profile.text_rknn_ms, started);
                        }
                        Ok((out, last_layer))
                    }
                    Err(err) => {
                        if let (Some(profile), Some(started)) =
                            (step_profile.as_mut(), rknn_started)
                        {
                            TextStepProfile::add_elapsed_ms(&mut profile.text_rknn_ms, started);
                        }
                        Err(err)
                    }
                }
            };
            match decode_result {
                Ok((out, decode_last_layer)) => {
                    x = out;
                    let start_layer = decode_last_layer + 1;
                    if start_layer < self.blocks.len() {
                        x = self
                            .forward_blocks_from(
                                x,
                                idx,
                                false,
                                start_layer,
                                step_profile.as_deref_mut(),
                            )
                            .await?;
                    }
                }
                Err(err) => {
                    log::warn!(
                        "text rknn decode failed: {}; fallback to software/remote decode path for remaining tokens",
                        err
                    );
                    self.sync_text_rknn_to_cache("decode fallback");
                    self.text_rknn_disabled_for_dialog = true;
                    self.text_rknn_disabled_for_runtime = true;
                    self.release_text_rknn("decode fallback");
                    x = self
                        .forward_blocks_from(x, idx, false, 0, step_profile.as_deref_mut())
                        .await?;
                }
            }
        } else {
            if idx == 0 && self.ctx.args.text_rknn_prefill {
                self.ensure_text_rknn_loaded()?;
            }
            let use_text_rknn_prefill = self.ctx.args.text_rknn_prefill
                && self
                    .text_rknn
                    .as_ref()
                    .map(|runner| runner.has_prefill())
                    .unwrap_or(false)
                && idx == 0
                && self.ctx.cache.with_kv_cache()
                && (!inject_images
                    || self
                        .text_rknn
                        .as_ref()
                        .map(|runner| runner.prefill_supports_deepstack())
                        .unwrap_or(false));
            if use_text_rknn_prefill {
                let layer_adds = if inject_images {
                    Some(self.build_prefill_layer_adds(seq_len, hidden_size)?)
                } else {
                    None
                };
                let prefill_result = {
                    let runner = self
                        .text_rknn
                        .as_mut()
                        .ok_or_else(|| anyhow!("text rknn runner not initialized"))?;
                    let rknn_started = step_profile.as_ref().map(|_| Instant::now());
                    let out = runner.forward_prefill(
                        &x,
                        idx,
                        layer_adds.as_deref(),
                        &self.ctx.device,
                        self.ctx.dtype,
                    );
                    if let (Some(profile), Some(started)) = (step_profile.as_mut(), rknn_started) {
                        TextStepProfile::add_elapsed_ms(&mut profile.text_rknn_ms, started);
                    }
                    out
                };
                match prefill_result {
                    Ok(out) => {
                        x = out;
                        let need_cpu_tail_cache = self
                            .text_rknn
                            .as_ref()
                            .map(|runner| {
                                !self.text_decode_mode.allows_text_rknn_decode()
                                    || !runner.decode_covers_all_layers()
                            })
                            .unwrap_or(false);
                        if need_cpu_tail_cache {
                            let sync_started = step_profile.as_ref().map(|_| Instant::now());
                            if let Some(runner) = self.text_rknn.as_ref() {
                                runner.sync_to_cache(
                                    &mut self.ctx.cache,
                                    &self.ctx.device,
                                    self.ctx.dtype,
                                )?;
                            }
                            if let (Some(profile), Some(started)) =
                                (step_profile.as_mut(), sync_started)
                            {
                                TextStepProfile::add_elapsed_ms(&mut profile.text_rknn_ms, started);
                            }
                        }
                    }
                    Err(err) => {
                        log::warn!(
                            "text rknn prefill failed: {}; fallback to software/remote prefill path",
                            err
                        );
                        self.text_rknn_disabled_for_dialog = true;
                        self.text_rknn_disabled_for_runtime = true;
                        self.release_text_rknn("prefill fallback");
                        x = self
                            .forward_blocks_from(
                                x,
                                idx,
                                inject_images,
                                0,
                                step_profile.as_deref_mut(),
                            )
                            .await?;
                    }
                }
            } else {
                x = self
                    .forward_blocks_from(x, idx, inject_images, 0, step_profile.as_deref_mut())
                    .await?;
            }
        }

        match classify_remote_output(&x, self.text_cfg.vocab_size) {
            Some(RemoteOutputKind::SampledToken) => {
                let released_local_head =
                    self.lm_head_q8.take().is_some() | self.lm_head.take().is_some();
                static LOGGED: std::sync::Once = std::sync::Once::new();
                LOGGED.call_once(|| {
                    log::info!(
                        "remote GGUF sampled token received before final norm: shape={:?} dtype={:?}; skipping Master final_norm and lm_head local_head_released={}",
                        x.dims(),
                        x.dtype(),
                        released_local_head
                    );
                });
                return Ok(x);
            }
            Some(RemoteOutputKind::Logits) => {
                let released_local_head =
                    self.lm_head_q8.take().is_some() | self.lm_head.take().is_some();
                static LOGGED: std::sync::Once = std::sync::Once::new();
                LOGGED.call_once(|| {
                    log::info!(
                        "remote GGUF output head active: received logits shape={:?} dtype={:?}; skipping Master final_norm and lm_head local_head_released={}",
                        x.dims(),
                        x.dtype(),
                        released_local_head
                    );
                });
                return Ok(x);
            }
            None => {}
        }

        let profile = Self::text_profile_enabled();
        let ln_started = profile.then(Instant::now);
        let summary_ln_started = step_profile.as_ref().map(|_| Instant::now());
        let x = self
            .ln_f
            .forward(&x)
            .map_err(|e| anyhow!("ln_f.forward: {e}"))?;
        if let (Some(profile), Some(started)) = (step_profile.as_mut(), summary_ln_started) {
            TextStepProfile::add_elapsed_ms(&mut profile.final_norm_ms, started);
        }
        if let Some(started) = ln_started {
            log::info!(
                "profile text final_norm idx={} elapsed_ms={:.3}",
                idx,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        let summary_slice_started = step_profile.as_ref().map(|_| Instant::now());
        let x = x
            .i((.., seq_len - 1, ..))
            .map_err(|e| anyhow!("x.i: {e}"))?
            .contiguous()
            .map_err(|e| anyhow!("x.i.contiguous: {e}"))?;
        if let (Some(profile), Some(started)) = (step_profile.as_mut(), summary_slice_started) {
            TextStepProfile::add_elapsed_ms(&mut profile.slice_ms, started);
        }
        let lm_started = profile.then(Instant::now);
        let summary_lm_started = step_profile.as_ref().map(|_| Instant::now());
        let logits = self.lm_head_forward(&x)?;
        if let (Some(profile), Some(started)) = (step_profile.as_mut(), summary_lm_started) {
            TextStepProfile::add_elapsed_ms(&mut profile.lm_head_ms, started);
        }
        if let Some(started) = lm_started {
            log::info!(
                "profile text lm_head idx={} elapsed_ms={:.3}",
                idx,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(logits)
    }

    fn inject_image_embeddings(&self, x: &Tensor) -> Result<Tensor> {
        // x: (1, seq, hidden)
        let (_b, seq, _h) = x.dims3()?;
        let mut out = x.clone();
        // Apply spans in order (they are non-overlapping).
        // We rebuild the tensor with concatenation to "replace" the span.
        for (span, embeds) in &self.image_spans {
            let n = span.end - span.start;
            let embeds = embeds.unsqueeze(0)?; // (1, n, hidden)
            if embeds.dims3()?.1 != n {
                bail!(
                    "image embeds length mismatch, span {}..{} expects {}",
                    span.start,
                    span.end,
                    n
                );
            }
            let before = out.narrow(1, 0, span.start)?;
            let after = out.narrow(1, span.end, seq - span.end)?;
            out = Tensor::cat(&[&before, &embeds, &after], 1)?;
        }
        Ok(out)
    }

    fn inject_deepstack_embeddings(&self, x: &Tensor, layer_idx: usize) -> Result<Tensor> {
        let Some(spans) = self.deepstack_spans.get(layer_idx) else {
            return Ok(x.clone());
        };
        if spans.is_empty() {
            return Ok(x.clone());
        }

        let (_b, seq, _h) = x.dims3()?;
        let mut out = x.clone();
        for (span, embeds) in spans {
            let n = span.end - span.start;
            let embeds = embeds.unsqueeze(0)?; // (1, n, hidden)
            if embeds.dims3()?.1 != n {
                bail!(
                    "deepstack embeds length mismatch for layer {} span {}..{} expects {}",
                    layer_idx,
                    span.start,
                    span.end,
                    n
                );
            }
            let before = out.narrow(1, 0, span.start)?;
            let local = out.narrow(1, span.start, n)?;
            let local = (local + &embeds)?;
            let after = out.narrow(1, span.end, seq - span.end)?;
            out = Tensor::cat(&[&before, &local, &after], 1)?;
        }
        Ok(out)
    }

    fn build_prefill_layer_adds(
        &self,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let mut layer_adds: Vec<Vec<f32>> = vec![vec![]; self.blocks.len()];
        for (layer_idx, spans) in self.deepstack_spans.iter().enumerate() {
            if layer_idx >= layer_adds.len() {
                bail!(
                    "deepstack layer index {} out of range for text layers {}",
                    layer_idx,
                    layer_adds.len()
                );
            }
            if spans.is_empty() {
                continue;
            }

            let mut merged = vec![0f32; seq_len * hidden_size];
            for (span, embeds) in spans {
                if span.end < span.start {
                    bail!(
                        "invalid deepstack span for layer {}: start {} > end {}",
                        layer_idx,
                        span.start,
                        span.end
                    );
                }
                let n = span.end - span.start;
                if span.end > seq_len {
                    bail!(
                        "deepstack span out of range for layer {}: span {}..{}, seq_len={}",
                        layer_idx,
                        span.start,
                        span.end,
                        seq_len
                    );
                }
                let embeds = embeds
                    .to_device(&Device::Cpu)
                    .map_err(|e| {
                        anyhow!("deepstack layer {} embeds.to_cpu failed: {e}", layer_idx)
                    })?
                    .to_dtype(DType::F32)
                    .map_err(|e| {
                        anyhow!("deepstack layer {} embeds.to_f32 failed: {e}", layer_idx)
                    })?
                    .contiguous()
                    .map_err(|e| {
                        anyhow!(
                            "deepstack layer {} embeds.contiguous failed: {e}",
                            layer_idx
                        )
                    })?;
                let (embed_seq, embed_hidden) = embeds.dims2().map_err(|e| {
                    anyhow!("deepstack layer {} embeds.dims2 failed: {e}", layer_idx)
                })?;
                if embed_seq != n || embed_hidden != hidden_size {
                    bail!(
                        "deepstack embeds shape mismatch for layer {} span {}..{}: got ({},{}), expected ({},{})",
                        layer_idx,
                        span.start,
                        span.end,
                        embed_seq,
                        embed_hidden,
                        n,
                        hidden_size
                    );
                }
                let data = embeds
                    .flatten_all()
                    .map_err(|e| {
                        anyhow!("deepstack layer {} embeds.flatten failed: {e}", layer_idx)
                    })?
                    .to_vec1::<f32>()
                    .map_err(|e| {
                        anyhow!("deepstack layer {} embeds.to_vec failed: {e}", layer_idx)
                    })?;
                for row in 0..n {
                    let dst_off = (span.start + row) * hidden_size;
                    let src_off = row * hidden_size;
                    for (dst, src) in merged[dst_off..dst_off + hidden_size]
                        .iter_mut()
                        .zip(data[src_off..src_off + hidden_size].iter())
                    {
                        *dst += *src;
                    }
                }
            }
            layer_adds[layer_idx] = merged;
        }
        Ok(layer_adds)
    }

    fn start_dialog_prompt(&mut self) -> Result<()> {
        let prompt_profile = Self::prompt_profile_enabled();
        let prompt_total = prompt_profile.then(Instant::now);
        let mut image_decode_ms = 0.0;
        let mut image_resize_ms = 0.0;
        let mut image_tensor_ms = 0.0;
        let mut vision_encode_ms = 0.0;
        let mut image_count = 0usize;
        let mut video_count = 0usize;
        let mut video_frame_count = 0usize;

        self.tokens.clear();
        self.image_spans.clear();
        self.deepstack_spans.clear();
        self.ctx.cache.clear();
        self.text_rknn_disabled_for_dialog = false;
        self.text_rknn_prefix_validated_for_dialog = false;
        self.text_rknn_validated_last_layer = None;
        self.force_shadow_blocks_for_dialog = false;
        self.index_pos = 0;

        let fixed_side = self.vision_rknn_side.or(self.ctx.args.vision_fixed_side);
        let history = self.history.clone();

        // 1) Encode visual inputs in message order. A video contributes one span per temporal patch.
        let mut visual_embeds: Vec<Tensor> = vec![];
        let mut visual_deepstack_embeds: Vec<Vec<Tensor>> = vec![];
        let mut visual_specs: Vec<VisualTokenSpec> = vec![];

        for msg in &history {
            match &msg.content {
                crate::models::chat::MessageContent::Text(_) => {}
                crate::models::chat::MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            crate::models::chat::ContentPart::Text { .. } => {}
                            crate::models::chat::ContentPart::ImageBase64 { data, .. } => {
                                image_count += 1;
                                let started = prompt_profile.then(Instant::now);
                                let bytes = decode_base64_image(data)?;
                                let (rgb, h, w) = load_image_rgb(&bytes)?;
                                if let Some(started) = started {
                                    image_decode_ms += started.elapsed().as_secs_f64() * 1000.0;
                                }
                                let started = prompt_profile.then(Instant::now);
                                let (rgb, h, w) = if let Some(side) = fixed_side {
                                    resize_to_fixed_square_letterbox(
                                        rgb,
                                        h,
                                        w,
                                        self.vision_cfg.patch_size,
                                        self.vision_cfg.spatial_merge_size,
                                        self.vision_cfg.num_position_embeddings,
                                        side,
                                        !self.ctx.args.vision_no_upscale,
                                    )?
                                } else {
                                    // （重要）按 pipeline.py 的思路做“网格对齐 + 像素上下限”缩放，尽量避免 crop 造成的信息丢失。
                                    smart_resize_like_pipeline(
                                        rgb,
                                        h,
                                        w,
                                        self.vision_cfg.patch_size,
                                        self.vision_cfg.spatial_merge_size,
                                        self.vision_cfg.num_position_embeddings,
                                        self.ctx.args.vision_max_side,
                                        !self.ctx.args.vision_no_upscale,
                                    )?
                                };
                                if let Some(started) = started {
                                    image_resize_ms += started.elapsed().as_secs_f64() * 1000.0;
                                }
                                let cache_key = Self::vision_cache_key(&rgb, h, w, fixed_side);
                                let started = prompt_profile.then(Instant::now);
                                let img = image_to_tensor(&rgb, h, w, &self.ctx.device)?
                                    .to_dtype(self.ctx.dtype)?;
                                if let Some(started) = started {
                                    image_tensor_ms += started.elapsed().as_secs_f64() * 1000.0;
                                }
                                let VisionOutputs { embeds, deepstack } = self
                                    .encode_vision_image_cached(
                                        cache_key,
                                        &img,
                                        prompt_profile,
                                        &mut vision_encode_ms,
                                    )?;
                                let n = embeds.dims3()?.1;
                                visual_specs.push(VisualTokenSpec::Image { token_count: n });
                                visual_embeds.push(embeds.squeeze(0)?); // (n, hidden)
                                let mut deepstack_embeds = Vec::with_capacity(deepstack.len());
                                for d in deepstack {
                                    deepstack_embeds.push(d.squeeze(0)?);
                                }
                                visual_deepstack_embeds.push(deepstack_embeds);
                            }
                            crate::models::chat::ContentPart::ImageUrl { image_url } => {
                                // Support data URLs only (no network fetch).
                                if !image_url.url.starts_with("data:") {
                                    bail!("image_url is only supported as data URLs in this build");
                                }
                                image_count += 1;
                                let started = prompt_profile.then(Instant::now);
                                let bytes = decode_base64_image(&image_url.url)?;
                                let (rgb, h, w) = load_image_rgb(&bytes)?;
                                if let Some(started) = started {
                                    image_decode_ms += started.elapsed().as_secs_f64() * 1000.0;
                                }
                                let started = prompt_profile.then(Instant::now);
                                let (rgb, h, w) = if let Some(side) = fixed_side {
                                    resize_to_fixed_square_letterbox(
                                        rgb,
                                        h,
                                        w,
                                        self.vision_cfg.patch_size,
                                        self.vision_cfg.spatial_merge_size,
                                        self.vision_cfg.num_position_embeddings,
                                        side,
                                        !self.ctx.args.vision_no_upscale,
                                    )?
                                } else {
                                    smart_resize_like_pipeline(
                                        rgb,
                                        h,
                                        w,
                                        self.vision_cfg.patch_size,
                                        self.vision_cfg.spatial_merge_size,
                                        self.vision_cfg.num_position_embeddings,
                                        self.ctx.args.vision_max_side,
                                        !self.ctx.args.vision_no_upscale,
                                    )?
                                };
                                if let Some(started) = started {
                                    image_resize_ms += started.elapsed().as_secs_f64() * 1000.0;
                                }
                                let cache_key = Self::vision_cache_key(&rgb, h, w, fixed_side);
                                let started = prompt_profile.then(Instant::now);
                                let img = image_to_tensor(&rgb, h, w, &self.ctx.device)?
                                    .to_dtype(self.ctx.dtype)?;
                                if let Some(started) = started {
                                    image_tensor_ms += started.elapsed().as_secs_f64() * 1000.0;
                                }
                                let VisionOutputs { embeds, deepstack } = self
                                    .encode_vision_image_cached(
                                        cache_key,
                                        &img,
                                        prompt_profile,
                                        &mut vision_encode_ms,
                                    )?;
                                let n = embeds.dims3()?.1;
                                visual_specs.push(VisualTokenSpec::Image { token_count: n });
                                visual_embeds.push(embeds.squeeze(0)?);
                                let mut deepstack_embeds = Vec::with_capacity(deepstack.len());
                                for d in deepstack {
                                    deepstack_embeds.push(d.squeeze(0)?);
                                }
                                visual_deepstack_embeds.push(deepstack_embeds);
                            }
                            crate::models::chat::ContentPart::VideoBase64 { data, .. } => {
                                video_count += 1;
                                let started = prompt_profile.then(Instant::now);
                                let bytes =
                                    decode_base64_video(data, self.ctx.args.video_max_bytes)?;
                                if let Some(started) = started {
                                    image_decode_ms += started.elapsed().as_secs_f64() * 1000.0;
                                }
                                let (spec, embeds, deepstack, frames) = self.encode_video_bytes(
                                    bytes,
                                    prompt_profile,
                                    &mut vision_encode_ms,
                                )?;
                                video_frame_count += frames;
                                visual_specs.push(spec);
                                visual_embeds.extend(embeds);
                                visual_deepstack_embeds.extend(deepstack);
                            }
                            crate::models::chat::ContentPart::VideoUrl { video_url } => {
                                if !video_url.url.starts_with("data:") {
                                    bail!("video_url is only supported as data URLs in this build");
                                }
                                video_count += 1;
                                let started = prompt_profile.then(Instant::now);
                                let bytes = decode_base64_video(
                                    &video_url.url,
                                    self.ctx.args.video_max_bytes,
                                )?;
                                if let Some(started) = started {
                                    image_decode_ms += started.elapsed().as_secs_f64() * 1000.0;
                                }
                                let (spec, embeds, deepstack, frames) = self.encode_video_bytes(
                                    bytes,
                                    prompt_profile,
                                    &mut vision_encode_ms,
                                )?;
                                video_frame_count += frames;
                                visual_specs.push(spec);
                                visual_embeds.extend(embeds);
                                visual_deepstack_embeds.extend(deepstack);
                            }
                        }
                    }
                }
            }
        }
        self.release_vision_rknn("visual pre-encode");

        // 2) Encode text + vision placeholders into token ids.
        let prompt_encode_started = prompt_profile.then(Instant::now);
        let (ids, spans) = self
            .prompt_encoder
            .encode(&self.tokenizer, &history, &visual_specs)?;
        let prompt_encode_ms = prompt_encode_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        self.tokens = ids;
        let max_context = if self.ctx.args.kv_cache_max_len > 0 {
            self.ctx
                .args
                .kv_cache_max_len
                .min(self.text_cfg.max_seq_len)
        } else {
            self.text_cfg.max_seq_len
        };
        if self.tokens.len() > max_context {
            bail!(
                "multimodal prompt requires {} tokens, exceeding context limit {}; reduce video duration/resolution or raise --kv-cache-max-len (frames were not sampled)",
                self.tokens.len(),
                max_context
            );
        }

        // 3) Map spans to embeds.
        if spans.len() != visual_embeds.len() {
            bail!(
                "internal error: spans {} != visual_embeds {}",
                spans.len(),
                visual_embeds.len()
            );
        }
        self.image_spans = spans.clone().into_iter().zip(visual_embeds).collect();

        let max_deepstack = visual_deepstack_embeds
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        self.deepstack_spans = vec![vec![]; max_deepstack];
        for (span, deepstack_embeds) in spans.into_iter().zip(visual_deepstack_embeds) {
            for (layer_idx, embeds) in deepstack_embeds.into_iter().enumerate() {
                self.deepstack_spans[layer_idx].push((span.clone(), embeds));
            }
        }

        if let Some(started) = prompt_total {
            log::info!(
                "profile prompt total_ms={:.3} images={} videos={} video_frames={} decode={:.3} resize={:.3} tensor={:.3} vision={:.3} tokenize={:.3} tokens={}",
                started.elapsed().as_secs_f64() * 1000.0,
                image_count,
                video_count,
                video_frame_count,
                image_decode_ms,
                image_resize_ms,
                image_tensor_ms,
                vision_encode_ms,
                prompt_encode_ms,
                self.tokens.len()
            );
        }

        Ok(())
    }
}

#[async_trait]
impl Generator for Qwen3Vl {
    type Shardable = Transformer;

    const MODEL_NAME: &'static str = "qwen3_vl";

    async fn load(ctx: Context) -> Result<Box<Self>> {
        let cfg_path = ctx.data_path.join("config.json");
        let full_cfg = Qwen3VlConfig::from_path(&cfg_path)?;
        let mut text_cfg: Config = full_cfg.text();
        text_cfg.attn_f32 = ctx.args.attn_f32;
        let tie_word_embeddings =
            full_cfg.tie_word_embeddings || full_cfg.text_config.tie_word_embeddings;

        let tokenizer = load_tokenizer(&ctx)?;
        let eos_token_id = text_cfg
            .eos_token_id
            .or_else(|| tokenizer.token_to_id(DEFAULT_EOS_TOKEN));

        let vision_cfg = full_cfg.vision_config.clone();

        let mut vision_rknn = None;
        let vision_rknn_path = ctx.args.vision_rknn.as_deref().map(PathBuf::from);
        let vision_rknn_lib = ctx.args.vision_rknn_lib.as_deref().map(PathBuf::from);
        let mut vision_rknn_side = None;
        if let Some(model_path) = ctx.args.vision_rknn.as_deref() {
            let lib_path = ctx.args.vision_rknn_lib.as_deref().map(Path::new);
            let rknn = VisionRknn::load(Path::new(model_path), lib_path)?;
            let side = rknn.expected_side().ok_or_else(|| {
                anyhow!(
                    "rknn input must be square NCHW/NHWC with 3 channels, got {}",
                    rknn.input_summary()
                )
            })?;
            if let Some(user_side) = ctx.args.vision_fixed_side {
                if user_side != side {
                    bail!(
                        "vision_fixed_side {} does not match rknn input side {}",
                        user_side,
                        side
                    );
                }
            }
            vision_rknn_side = Some(side);
            if Self::keep_vision_rknn_loaded() {
                vision_rknn = Some(rknn);
                log::info!("vision rknn enabled (side={})", side);
            } else {
                drop(rknn);
                log::info!(
                    "vision rknn probed and released before text loading (side={}); it will reload on image pre-encode"
                    ,
                    side
                );
            }
        }

        let vision = if vision_rknn_path.is_some() {
            None
        } else {
            log::info!("loading vision encoder ...");
            Some(VisionEncoder::load(
                vision_cfg.clone(),
                ctx.var_builder.clone(),
                ctx.args.attn_f32,
            )?)
        };

        let text_rknn_dir = ctx.args.text_rknn_dir.as_deref().map(PathBuf::from);
        let text_rknn_lib = ctx.args.text_rknn_lib.as_deref().map(PathBuf::from);
        let text_decode_mode = TextDecodeMode::parse(&ctx.args.text_decode_mode);
        let mut text_rknn_disabled_for_runtime = false;
        let text_rknn = if !text_decode_mode.needs_text_rknn_runtime(ctx.args.text_rknn_prefill) {
            log::info!("text decode mode: {}", text_decode_mode.as_str());
            None
        } else if let Some(model_dir) = text_rknn_dir.as_deref() {
            let lib_path = text_rknn_lib.as_deref().map(Path::new);
            let runner = TextRknnRunner::load(model_dir, lib_path, &text_cfg)?;
            log::info!(
                "text rknn capabilities from {}: decode_cover=0..{} decode_full={} prefill_rknn={} prefill_deepstack={}",
                model_dir.display(),
                runner.decode_last_layer(),
                runner.decode_covers_all_layers(),
                runner.has_prefill(),
                runner.prefill_supports_deepstack()
            );
            if !runner.decode_covers_all_layers() {
                if !text_decode_mode.allows_text_rknn_decode() && ctx.args.text_rknn_prefill {
                    log::info!(
                        "text rknn prefill-only startup enabled from {} (decode cover=0..{}, decode path stays on CPU/software)",
                        model_dir.display(),
                        runner.decode_last_layer()
                    );
                    Some(runner)
                } else if text_decode_mode.allows_partial_coverage() {
                    log::info!(
                        "text rknn partial decode enabled at startup from {} (cover=0..{}, full=false); remaining layers will fallback to software/remote path",
                        model_dir.display(),
                        runner.decode_last_layer()
                    );
                    Some(runner)
                } else {
                    log::warn!(
                        "text rknn decode disabled for this runtime at startup: partial coverage 0..{} is not allowed in {:?} mode",
                        runner.decode_last_layer(),
                        text_decode_mode
                    );
                    text_rknn_disabled_for_runtime = true;
                    None
                }
            } else {
                log::info!(
                    "text rknn runtime enabled from {} (decode cover=0..{}, decode_full={})",
                    model_dir.display(),
                    runner.decode_last_layer(),
                    runner.decode_covers_all_layers()
                );
                Some(runner)
            }
        } else {
            None
        };
        if matches!(
            text_decode_mode,
            TextDecodeMode::NpuCpu | TextDecodeMode::CpuOnly
        ) && !matches!(ctx.device, Device::Cpu)
        {
            log::warn!(
                "text decode mode '{}' assumes a CPU/software tail, but the local software device is not CPU; pass --cpu to force a true NPU+CPU path",
                text_decode_mode.as_str()
            );
        }
        if ctx.args.text_rknn_prefill
            && text_rknn
                .as_ref()
                .map(|runner| !runner.has_prefill())
                .unwrap_or(true)
        {
            log::warn!(
                "--text-rknn-prefill was requested, but no usable prefill RKNN chunks were loaded; prompt prefill will fallback to the local software path"
            );
        }
        configure_qkv_rknn_from_args(&ctx.args)?;
        configure_mlp_rknn_from_args(&ctx.args)?;
        configure_local_q8_from_args(ctx.args.local_linear_q8);

        let video_token_id = full_cfg
            .video_token_id
            .or_else(|| tokenizer.token_to_id("<|video_pad|>"))
            .unwrap_or_else(|| {
                log::warn!(
                    "model has no video_token_id or <|video_pad|>; video input will be unavailable"
                );
                full_cfg.image_token_id
            });
        let prompt_encoder = PromptEncoder::from_tokenizer(
            &tokenizer,
            full_cfg.image_token_id,
            video_token_id,
            full_cfg.vision_start_token_id,
            full_cfg.vision_end_token_id,
        )?;

        log::info!("loading language embeddings ...");
        let embedding: Embedding = candle_nn::embedding(
            text_cfg.vocab_size,
            text_cfg.hidden_size,
            ctx.var_builder.pp("model.language_model.embed_tokens"),
        )?;

        log::info!("loading language norm ...");
        let ln_f = candle_nn::rms_norm(
            text_cfg.hidden_size,
            text_cfg.rms_norm_eps,
            ctx.var_builder.pp("model.language_model.norm"),
        )?;

        log::info!("loading lm_head ...");
        let lm_head = match linear(
            text_cfg.hidden_size,
            text_cfg.vocab_size,
            ctx.var_builder.pp("lm_head"),
        ) {
            Ok(v) => v,
            Err(e1) => match linear(
                text_cfg.hidden_size,
                text_cfg.vocab_size,
                ctx.var_builder.pp("model.language_model.lm_head"),
            ) {
                Ok(v) => v,
                Err(e2) => {
                    if tie_word_embeddings {
                        log::warn!(
                            "lm_head.weight not found ({}; {}), using tied word embeddings from model.language_model.embed_tokens.weight",
                            e1,
                            e2
                        );
                        Linear::new(embedding.embeddings().clone(), None)
                    } else {
                        return Err(anyhow!(
                            "cannot find tensor lm_head.weight (tried `lm_head.weight` and `model.language_model.lm_head.weight`): {e1}; {e2}"
                        ));
                    }
                }
            },
        };
        let lm_head_q8 = if lm_head_q8_enabled(ctx.args.lm_head_q8) {
            let started = Instant::now();
            match QuantizedLmHead::from_linear(&lm_head) {
                Ok(q8) => {
                    log::info!(
                        "lm_head q8 enabled: out_dim={} in_dim={} aarch64_dotprod={} elapsed_ms={:.3}",
                        q8.out_dim,
                        q8.in_dim,
                        aarch64_dotprod_available(),
                        started.elapsed().as_secs_f64() * 1000.0
                    );
                    Some(q8)
                }
                Err(err) => {
                    log::warn!("lm_head q8 disabled: {err}");
                    None
                }
            }
        } else {
            log::info!("lm_head q8 disabled by --lm-head-q8/QWEN3VL_LM_HEAD_Q8");
            None
        };
        let lm_head = if lm_head_q8.is_some() {
            log::info!("lm_head f16 weights released after q8 quantization");
            None
        } else {
            Some(lm_head)
        };

        log::info!("loading {} text blocks ...", text_cfg.num_hidden_layers);
        let mut blocks: Vec<Box<dyn Forwarder>> = vec![];
        let shadow_host = if Self::shadow_prefill_enabled() {
            Self::shadow_prefill_host(&ctx)
        } else {
            None
        };
        if Self::shadow_prefill_enabled() && shadow_host.is_none() {
            log::warn!(
                "QWEN3VL_SHADOW_PREFILL=1 but no shadow host is available; set QWEN3VL_SHADOW_PREFILL_HOST=host:port"
            );
        }
        let mut worker_connections = crate::spm::ClientPool::default();
        let mut prefill_shadow_blocks: Vec<Option<Box<dyn Forwarder>>> = Vec::new();
        for i in 0..text_cfg.num_hidden_layers {
            let block_layer_name = format!("model.language_model.layers.{i}");
            let node_for_layer = ctx.topology.get_node_for_layer(&block_layer_name);
            if let Some((node_name, node)) = node_for_layer {
                log::debug!("node {node_name} will serve {}", &block_layer_name);
                let client = worker_connections
                    .client_for_layer(ctx.device.clone(), &node.host, &block_layer_name)
                    .await?;
                blocks.push(Box::new(client));
            } else {
                blocks.push(Transformer::load(
                    block_layer_name.clone(),
                    ctx.var_builder.pp(&block_layer_name),
                    &text_cfg,
                )?);
            }
            let shadow = if let Some(host) = shadow_host.as_deref() {
                if node_for_layer.is_none() {
                    log::info!(
                        "shadow prefill client for local layer {} -> {}",
                        block_layer_name,
                        host
                    );
                    let client = worker_connections
                        .client_for_layer(ctx.device.clone(), host, &block_layer_name)
                        .await?;
                    Some(Box::new(client) as Box<dyn Forwarder>)
                } else {
                    None
                }
            } else {
                None
            };
            prefill_shadow_blocks.push(shadow);
        }
        for block in &blocks {
            log::info!("  {}", block)
        }

        let fast_logits_processor = create_fast_logits_processor(&ctx);

        Ok(Box::new(Self {
            tokenizer,
            ctx,
            embedding,
            ln_f,
            lm_head,
            lm_head_q8,
            blocks,
            prefill_shadow_blocks,
            force_shadow_blocks_for_dialog: false,
            text_rknn,
            text_rknn_dir,
            text_rknn_lib,
            text_cfg: text_cfg.clone(),
            text_decode_mode,
            text_rknn_disabled_for_dialog: false,
            text_rknn_disabled_for_runtime,
            text_rknn_prefix_validated_for_dialog: false,
            text_rknn_validated_last_layer: None,
            vision_cfg,
            vision,
            vision_rknn,
            vision_rknn_path,
            vision_rknn_lib,
            vision_rknn_side,
            vision_cache: HashMap::new(),
            prompt_encoder,
            image_token_id: full_cfg.image_token_id,
            vision_start_token_id: full_cfg.vision_start_token_id,
            vision_end_token_id: full_cfg.vision_end_token_id,
            eos_token_id,
            fast_logits_processor,
            history: vec![],
            tokens: vec![],
            image_spans: vec![],
            deepstack_spans: vec![],
            index_pos: 0,
            generated: 0,
        }))
    }

    fn add_message(&mut self, message: Message) -> Result<()> {
        self.history.push(message);
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.tokens.clear();
        self.history.clear();
        self.image_spans.clear();
        self.deepstack_spans.clear();
        self.ctx.cache.clear();
        self.text_rknn_disabled_for_dialog = false;
        self.text_rknn_prefix_validated_for_dialog = false;
        self.text_rknn_validated_last_layer = None;
        self.force_shadow_blocks_for_dialog = false;
        if let Some(runner) = self.text_rknn.as_mut() {
            runner.clear();
        }
        self.index_pos = 0;
        self.generated = 0;
        Ok(())
    }

    async fn next_token(&mut self, index: usize) -> Result<Token> {
        let profile_enabled = Self::text_profile_summary_enabled();
        let mut step_profile = profile_enabled.then(TextStepProfile::default);
        let token_started = profile_enabled.then(Instant::now);

        if self.generated == 0 {
            let started = profile_enabled.then(Instant::now);
            self.start_dialog_prompt()?;
            if let (Some(profile), Some(started)) = (step_profile.as_mut(), started) {
                TextStepProfile::add_elapsed_ms(&mut profile.prompt_ms, started);
            }
        }

        let num_tokens = self.tokens.len();
        let (context_size, context_index) = if self.ctx.cache.with_kv_cache() && index > 0 {
            (1, self.index_pos)
        } else {
            (num_tokens, 0)
        };

        let context_offset = num_tokens.saturating_sub(context_size);
        let context_tokens = &self.tokens[context_offset..];
        let num_context_tokens = context_tokens.len();

        let input_ids = Tensor::new(context_tokens, &self.ctx.device)?.unsqueeze(0)?;
        let started = profile_enabled.then(Instant::now);
        let x = self.embedding.forward(&input_ids)?;
        if let (Some(profile), Some(started)) = (step_profile.as_mut(), started) {
            TextStepProfile::add_elapsed_ms(&mut profile.embedding_ms, started);
        }

        // Only inject images for the full "prefill" pass.
        let inject_images = self.index_pos == 0 && context_index == 0 && context_size == num_tokens;

        let forward_started = profile_enabled.then(Instant::now);
        let repeat_start = num_tokens.saturating_sub(self.ctx.args.repeat_last_n);
        let sampling_request = crate::spm::SamplingRequest {
            seed: self.ctx.args.seed,
            sample_index: self.generated,
            temperature: self.ctx.args.temperature,
            top_k: self.ctx.args.top_k,
            top_p: self.ctx.args.top_p,
            repeat_penalty: self.ctx.args.repeat_penalty,
            repeat_context: self.tokens[repeat_start..].to_vec(),
        };
        let logits = crate::spm::with_remote_sampling_request(
            sampling_request,
            self.forward_embeds(x, context_index, inject_images, step_profile.as_mut()),
        )
        .await
        .map_err(|e| anyhow!("forward failed: {e}"))?;
        if let (Some(profile), Some(started)) = (step_profile.as_mut(), forward_started) {
            TextStepProfile::add_elapsed_ms(&mut profile.forward_total_ms, started);
        }

        self.index_pos += num_context_tokens;
        let started = profile_enabled.then(Instant::now);
        let next_token = if classify_remote_output(&logits, self.text_cfg.vocab_size)
            == Some(RemoteOutputKind::SampledToken)
        {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                log::info!(
                    "remote GGUF sampling active: received token shape={:?} dtype={:?}; skipping Master logits transfer and sampling",
                    logits.dims(),
                    logits.dtype()
                );
            });
            logits.to_vec1::<u32>()?[0]
        } else {
            let start_at = num_tokens.saturating_sub(self.ctx.args.repeat_last_n);
            let logits = logits.squeeze(0)?;
            self.fast_logits_processor.sample(
                &logits,
                self.ctx.args.repeat_penalty,
                &self.tokens[start_at..],
            )?
        };
        if let (Some(profile), Some(started)) = (step_profile.as_mut(), started) {
            TextStepProfile::add_elapsed_ms(&mut profile.sample_ms, started);
        }

        self.generated += 1;
        self.tokens.push(next_token);

        let started = profile_enabled.then(Instant::now);
        let text = match self.tokenizer.decode(&[next_token], false) {
            Ok(s) => Some(s),
            Err(e) => {
                log::error!("could not decode token {next_token}: {e}");
                None
            }
        };
        if let (Some(profile), Some(started)) = (step_profile.as_mut(), started) {
            TextStepProfile::add_elapsed_ms(&mut profile.token_decode_ms, started);
        }

        if let (Some(profile), Some(started)) = (step_profile.as_ref(), token_started) {
            if Self::should_log_text_profile_step(index) {
                let total_ms = started.elapsed().as_secs_f64() * 1000.0;
                log::info!(
                    "profile text summary idx={} context={} total_ms={:.3} prompt={:.3} embed={:.3} img_inject={:.3} deepstack={:.3} local={:.3}/{} remote={:.3}/{}b/{}l rknn={:.3} norm={:.3} slice={:.3} lm_head={:.3} cast={:.3} repeat={:.3} sample={:.3} tok_decode={:.3} forward_total={:.3}",
                    index,
                    num_context_tokens,
                    total_ms,
                    profile.prompt_ms,
                    profile.embedding_ms,
                    profile.image_inject_ms,
                    profile.deepstack_inject_ms,
                    profile.local_ms,
                    profile.local_layers,
                    profile.remote_ms,
                    profile.remote_batches,
                    profile.remote_layers,
                    profile.text_rknn_ms,
                    profile.final_norm_ms,
                    profile.slice_ms,
                    profile.lm_head_ms,
                    profile.logits_to_f32_ms,
                    profile.repeat_penalty_ms,
                    profile.sample_ms,
                    profile.token_decode_ms,
                    profile.forward_total_ms
                );
                let (bottleneck, bottleneck_ms) = profile.bottleneck();
                let other_ms = profile.other_ms(total_ms);
                log::info!(
                    "profile text bottleneck idx={} context={} top={} {:.3}ms/{:.1}% local={:.3}ms/{:.1}% remote={:.3}ms/{:.1}% lm_head={:.3}ms/{:.1}% sample={:.3}ms/{:.1}% other={:.3}ms/{:.1}%",
                    index,
                    num_context_tokens,
                    bottleneck,
                    bottleneck_ms,
                    TextStepProfile::pct(bottleneck_ms, total_ms),
                    profile.local_ms,
                    TextStepProfile::pct(profile.local_ms, total_ms),
                    profile.remote_ms,
                    TextStepProfile::pct(profile.remote_ms, total_ms),
                    profile.lm_head_ms,
                    TextStepProfile::pct(profile.lm_head_ms, total_ms),
                    profile.sample_ms,
                    TextStepProfile::pct(profile.sample_ms, total_ms),
                    other_ms,
                    TextStepProfile::pct(other_ms, total_ms)
                );
            }
        }

        Ok(Token {
            id: next_token,
            text,
            is_end_of_stream: Some(next_token) == self.eos_token_id,
        })
    }

    fn generated_tokens(&self) -> usize {
        self.generated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_uses_device_argmax() {
        let logits = Tensor::new(&[-1.0f32, 0.5, 3.0, 2.0], &Device::Cpu).unwrap();
        let mut processor = FastLogitsProcessor::new(0, FastSampling::ArgMax);

        let token = processor.sample(&logits, 1.0, &[]).unwrap();

        assert_eq!(token, 2);
    }

    #[test]
    fn parses_fractional_and_integer_video_frame_rates() {
        assert!((parse_frame_rate("30000/1001").unwrap() - 29.970).abs() < 0.001);
        assert_eq!(parse_frame_rate("25").unwrap(), 25.0);
        assert!(parse_frame_rate("0/0").is_err());
    }

    #[test]
    fn official_video_sampling_covers_the_full_source_uniformly() {
        let indices = uniform_sample_indices(1253, 25.0, 2.0, 4, 768).unwrap();
        assert_eq!(indices.len(), 100);
        assert_eq!(indices.first(), Some(&0));
        assert_eq!(indices.last(), Some(&1252));
        assert!(indices.windows(2).all(|pair| pair[1] > pair[0]));
    }

    #[test]
    fn official_video_sampling_honors_minimum_and_maximum() {
        assert_eq!(
            uniform_sample_indices(2, 25.0, 2.0, 4, 768).unwrap(),
            vec![0, 1]
        );
        assert_eq!(
            uniform_sample_indices(10_000, 25.0, 2.0, 4, 8)
                .unwrap()
                .len(),
            8
        );
        assert_eq!(
            uniform_sample_indices(100, 25.0, 0.0, 4, 8).unwrap().len(),
            8
        );
    }

    #[test]
    fn video_resize_grid_is_patch_and_merge_aligned() {
        let (height, width) =
            smart_resize_dimensions(1080, 1920, 16, 2, 2304, Some(384), false).unwrap();
        assert_eq!(height % 32, 0);
        assert_eq!(width % 32, 0);
        assert!(height <= 384);
        assert!(width <= 384);
    }

    #[test]
    fn full_video_decoder_preserves_every_input_frame() {
        let ffmpeg = command_from_env(
            "DIAL_FFMPEG_BIN",
            "ffmpeg",
            "/opt/sophon/sophon-ffmpeg-latest/bin/ffmpeg",
        );
        let output = match Command::new(&ffmpeg)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x32:rate=7",
                "-frames:v",
                "7",
                "-c:v",
                "ffv1",
                "-f",
                "matroska",
                "pipe:1",
            ])
            .output()
        {
            Ok(output) if output.status.success() => output,
            _ => return, // Optional integration check when ffmpeg/lavfi is unavailable.
        };

        let temporary_video = TemporaryVideoFile::create(&output.stdout).unwrap();
        let probe = probe_video(temporary_video.path()).unwrap();
        assert_eq!((probe.width, probe.height), (64, 32));
        assert_eq!(probe.fps, 7.0);

        let mut indexes = Vec::new();
        let count = decode_all_video_frames(
            temporary_video.path(),
            32,
            64,
            "scale=64:32".to_string(),
            |index, rgb| {
                assert_eq!(rgb.len(), 64 * 32 * 3);
                indexes.push(index);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(count, 7);
        assert_eq!(indexes, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn remote_output_logits_are_distinguished_from_hidden_states() {
        assert!(is_remote_output_logits(&[1, 151936], 151936));
        assert!(!is_remote_output_logits(&[1, 1, 151936], 151936));
        assert!(!is_remote_output_logits(&[1, 4096], 151936));
        assert!(!is_remote_output_logits(&[1, 224, 4096], 151936));
    }

    #[test]
    fn remote_sampled_token_has_an_unambiguous_wire_shape() {
        let token = Tensor::new(&[42u32], &Device::Cpu).unwrap();
        let logits = Tensor::zeros((1, 151936), DType::F16, &Device::Cpu).unwrap();
        let hidden = Tensor::zeros((1, 224, 4096), DType::F16, &Device::Cpu).unwrap();

        assert!(is_remote_sampled_token(&token));
        assert!(!is_remote_sampled_token(&logits));
        assert_eq!(
            classify_remote_output(&token, 151936),
            Some(RemoteOutputKind::SampledToken)
        );
        assert_eq!(
            classify_remote_output(&logits, 151936),
            Some(RemoteOutputKind::Logits)
        );
        assert_eq!(classify_remote_output(&hidden, 151936), None);
    }

    #[test]
    fn remote_sampling_applies_repeat_penalty_before_greedy_selection() {
        // The GGUF output head returns a batch dimension even though generation
        // currently samples one sequence at a time.
        let logits = Tensor::new(&[[5.0f32, 4.0]], &Device::Cpu).unwrap();
        let request = crate::spm::SamplingRequest {
            seed: 7,
            sample_index: 0,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repeat_penalty: 2.0,
            repeat_context: vec![0],
        };

        assert_eq!(sample_remote_logits(&logits, &request).unwrap(), 1);
    }
}
