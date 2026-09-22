//! Qwen3 text (language model) building blocks.

use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{linear_no_bias as linear, Module, RmsNorm, VarBuilder};
use half::f16;
use rayon::prelude::*;
use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

use crate::spm::Forwarder;

use crate::models::llama3::{Cache, Config};

use super::{
    attach_worker_gguf_fp16_prefill, build_worker_gguf_fp16_prefill_linear,
    load_worker_gguf_linear, worker_gguf_enabled, worker_gguf_fp16_prefill_enabled,
    worker_gguf_prefill_for_dims, MlpRknnRunner, MlpRknnSpec, QkvRknnRunner, TextLinear,
};

static LOCAL_Q8_RESERVED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LOCAL_Q8_ARG_ENABLED: AtomicBool = AtomicBool::new(true);

pub fn configure_local_q8_from_args(enabled: bool) {
    LOCAL_Q8_ARG_ENABLED.store(enabled, Ordering::Relaxed);
}

fn tensor_to_f32_vec(tensor: &Tensor) -> candle_core::Result<Vec<f32>> {
    match tensor.dtype() {
        DType::F32 => tensor.to_vec1::<f32>(),
        DType::F16 => Ok(tensor
            .to_vec1::<f16>()?
            .into_iter()
            .map(f16::to_f32)
            .collect()),
        _ => tensor.to_dtype(DType::F32)?.to_vec1::<f32>(),
    }
}

fn tensor_diff_stats(lhs: &[f32], rhs: &[f32]) -> Result<(f32, f32)> {
    if lhs.len() != rhs.len() {
        bail!(
            "tensor diff length mismatch: lhs={} rhs={}",
            lhs.len(),
            rhs.len()
        );
    }
    let mut max_abs = 0.0f32;
    let mut sq_sum = 0.0f64;
    for (l, r) in lhs.iter().zip(rhs.iter()) {
        let diff = (l - r).abs();
        max_abs = max_abs.max(diff);
        sq_sum += f64::from(diff) * f64::from(diff);
    }
    let rms = if lhs.is_empty() {
        0.0
    } else {
        (sq_sum / lhs.len() as f64).sqrt() as f32
    };
    Ok((max_abs, rms))
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

fn local_q8_enabled() -> bool {
    match std::env::var("QWEN3VL_LOCAL_Q8").ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES") => true,
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO") => false,
        Some(_) => false,
        None => LOCAL_Q8_ARG_ENABLED.load(Ordering::Relaxed),
    }
}

fn local_q8_budget_bytes() -> usize {
    const DEFAULT_BUDGET_MB: usize = 256;
    match std::env::var("QWEN3VL_LOCAL_Q8_BUDGET_MB").ok().as_deref() {
        Some("all") | Some("ALL") | Some("unlimited") | Some("UNLIMITED") => usize::MAX,
        Some(v) => v
            .parse::<usize>()
            .unwrap_or(DEFAULT_BUDGET_MB)
            .saturating_mul(1024 * 1024),
        None => DEFAULT_BUDGET_MB * 1024 * 1024,
    }
}

fn local_q8_max_layers() -> usize {
    match std::env::var("QWEN3VL_LOCAL_Q8_MAX_LAYERS").ok().as_deref() {
        Some("all") | Some("ALL") | Some("unlimited") | Some("UNLIMITED") => usize::MAX,
        Some(v) => v.parse::<usize>().unwrap_or(usize::MAX),
        None => usize::MAX,
    }
}

fn local_q8_scope_allows(scope: &str) -> bool {
    let value = std::env::var("QWEN3VL_LOCAL_Q8_SCOPE").unwrap_or_else(|_| "all".to_string());
    value
        .split(',')
        .map(|part| part.trim().to_ascii_lowercase())
        .any(|part| part == "all" || part == scope || (scope == "attn" && part == "attention"))
}

fn mlp_rknn_gate_up_allowed(layer_idx: Option<usize>) -> bool {
    let Some(raw) = std::env::var("QWEN3VL_MLP_RKNN_GATE_UP_LAYERS").ok() else {
        return true;
    };
    let raw = raw.trim();
    if matches!(raw, "0" | "false" | "FALSE" | "no" | "NO") {
        return false;
    }
    if matches!(raw, "all" | "ALL" | "unlimited" | "UNLIMITED") {
        return true;
    }
    match raw.parse::<usize>() {
        Ok(limit) => layer_idx.map(|idx| idx < limit).unwrap_or(true),
        Err(_) => true,
    }
}

fn mlp_rknn_lazy_load_enabled() -> bool {
    match std::env::var("QWEN3VL_MLP_RKNN_LAZY").ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES") => true,
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO") => false,
        Some(_) | None => false,
    }
}

fn parse_layer_idx(name: &str) -> Option<usize> {
    name.rsplit('.').next()?.parse().ok()
}

fn local_q8_enabled_for_layer(layer_idx: Option<usize>) -> bool {
    if !local_q8_enabled() {
        return false;
    }
    let Some(layer_idx) = layer_idx else {
        return true;
    };
    layer_idx < local_q8_max_layers()
}

fn local_q8_device_allows(device: &Device) -> bool {
    matches!(device, Device::Cpu)
}

fn local_profile_enabled() -> bool {
    matches!(
        std::env::var("QWEN3VL_PROFILE_LOCAL").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn local_prefill_profile_enabled() -> bool {
    matches!(
        std::env::var("QWEN3VL_PROFILE_LOCAL_PREFILL")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

struct LocalQ8Reservation {
    bytes: usize,
    committed: bool,
}

impl LocalQ8Reservation {
    fn reserve(name: &str, bytes: usize) -> candle_core::Result<Self> {
        let budget = local_q8_budget_bytes();
        let mut current = LOCAL_Q8_RESERVED_BYTES.load(Ordering::Relaxed);
        loop {
            let Some(next) = current.checked_add(bytes) else {
                candle_core::bail!("local q8 byte counter overflow for {name}");
            };
            if next > budget {
                candle_core::bail!(
                    "local q8 budget exceeded for {name}: need {:.1} MiB, reserved {:.1} MiB, budget {:.1} MiB",
                    bytes as f64 / 1048576.0,
                    current as f64 / 1048576.0,
                    budget as f64 / 1048576.0
                );
            }
            match LOCAL_Q8_RESERVED_BYTES.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Ok(Self {
                        bytes,
                        committed: false,
                    })
                }
                Err(actual) => current = actual,
            }
        }
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for LocalQ8Reservation {
    fn drop(&mut self) {
        if !self.committed {
            LOCAL_Q8_RESERVED_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
        }
    }
}

fn maybe_quantized_linear(name: &str, linear: &candle_nn::Linear) -> Option<QuantizedLinear> {
    match QuantizedLinear::from_linear(name, linear) {
        Ok(q8) => Some(q8),
        Err(e) => {
            log::info!("local q8 linear skipped: {name}: {e}");
            None
        }
    }
}

#[derive(Debug, Clone)]
struct QuantizedLinear {
    qweight: Vec<i8>,
    scales: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
    dotprod: bool,
}

impl QuantizedLinear {
    fn from_linear(name: &str, linear: &candle_nn::Linear) -> candle_core::Result<Self> {
        if linear.bias().is_some() {
            candle_core::bail!("local q8 linear only supports bias-free linear: {name}");
        }
        let weight = if linear.weight().is_contiguous() {
            linear.weight().clone()
        } else {
            linear.weight().contiguous()?
        };
        let (out_dim, in_dim) = weight.dims2()?;
        let Some(weight_bytes) = out_dim
            .checked_mul(in_dim)
            .and_then(|v| v.checked_add(out_dim * std::mem::size_of::<f32>()))
        else {
            candle_core::bail!("local q8 size overflow: {name}");
        };
        let reservation = LocalQ8Reservation::reserve(name, weight_bytes)?;

        let mut qweight = Vec::with_capacity(out_dim * in_dim);
        let mut scales = Vec::with_capacity(out_dim);
        for row_idx in 0..out_dim {
            let row_tensor = weight.narrow(0, row_idx, 1)?.flatten_all()?;
            let row = tensor_to_f32_vec(&row_tensor)?;
            let max_abs = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;
            scales.push(scale);
            qweight.extend(row.iter().map(|v| {
                let q = (v * inv_scale).round().clamp(-127.0, 127.0);
                q as i8
            }));
        }

        log::info!("local q8 linear enabled: {name} out_dim={out_dim} in_dim={in_dim}");
        reservation.commit();
        Ok(Self {
            qweight,
            scales,
            out_dim,
            in_dim,
            dotprod: aarch64_dotprod_available(),
        })
    }

    fn forward_decode(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        if dims != [1, 1, self.in_dim] {
            candle_core::bail!(
                "local q8 linear expects decode [1, 1, {}], got {:?}",
                self.in_dim,
                dims
            );
        }
        let out_dtype = x.dtype();
        let device = x.device().clone();
        let x_values = tensor_to_f32_vec(&x.flatten_all()?)?;
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
        let mut out = vec![0.0f32; self.out_dim];
        const ROWS_PER_TASK: usize = 64;
        out.par_chunks_mut(ROWS_PER_TASK)
            .enumerate()
            .for_each(|(chunk_idx, outs)| {
                let row_base = chunk_idx * ROWS_PER_TASK;
                for (row_offset, dst) in outs.iter_mut().enumerate() {
                    let row_idx = row_base + row_offset;
                    let row_start = row_idx * self.in_dim;
                    let row = &self.qweight[row_start..row_start + self.in_dim];
                    let acc = dot_i8(row, &x_q, self.dotprod);
                    *dst = (acc as f32) * self.scales[row_idx] * x_scale;
                }
            });

        let out = Tensor::from_vec(out, (1, 1, self.out_dim), &device)?;
        if out_dtype == DType::F32 {
            Ok(out)
        } else {
            out.to_dtype(out_dtype)
        }
    }
}

fn dot_i8(row: &[i8], x: &[i8], _dotprod: bool) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        if _dotprod {
            // SAFETY: guarded by runtime dotprod detection and equal slice length.
            return unsafe { dot_i8_sdot_asm(row.as_ptr(), x.as_ptr(), row.len()) };
        }
        return unsafe { dot_i8_neon(row.as_ptr(), x.as_ptr(), row.len()) };
    }

    row.iter()
        .zip(x.iter())
        .fold(0i32, |acc, (&w, &xv)| acc + (w as i32) * (xv as i32))
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

#[inline]
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle_core::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone)]
pub struct Mlp {
    gate_up_proj: MlpGateUpProjection,
    down_proj: TextLinear,
    gate_up_q8: Option<QuantizedLinear>,
    down_q8: Option<QuantizedLinear>,
    mlp_rknn: Option<Arc<Mutex<MlpRknnState>>>,
    mlp_gate_up_rknn: Option<Arc<Mutex<MlpRknnState>>>,
    mlp_down_rknn: Option<Arc<Mutex<MlpRknnState>>>,
    split_rknn_validated_decode: Arc<Mutex<bool>>,
    intermediate_size: usize,
}

#[derive(Debug, Clone)]
enum MlpGateUpProjection {
    Fused(TextLinear),
    Split {
        gate: TextLinear,
        up: TextLinear,
    },
    Hybrid {
        fp16_prefill: candle_nn::Linear,
        gate: TextLinear,
        up: TextLinear,
    },
}

impl MlpGateUpProjection {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Fused(proj) => proj.forward(x),
            Self::Split { gate, up } => Tensor::cat(&[gate.forward(x)?, up.forward(x)?], D::Minus1),
            Self::Hybrid {
                fp16_prefill,
                gate,
                up,
            } => {
                if worker_gguf_prefill_for_dims(x.dims()) {
                    fp16_prefill.forward(x)
                } else {
                    Tensor::cat(&[gate.forward(x)?, up.forward(x)?], D::Minus1)
                }
            }
        }
    }
}

#[derive(Debug)]
struct MlpRknnState {
    spec: MlpRknnSpec,
    runner: Option<MlpRknnRunner>,
    validated_decode: bool,
    disabled: bool,
}

impl MlpRknnState {
    fn new(runner: MlpRknnRunner) -> Self {
        Self {
            spec: runner.spec(),
            runner: Some(runner),
            validated_decode: false,
            disabled: false,
        }
    }

    fn new_lazy(spec: MlpRknnSpec) -> Self {
        Self {
            spec,
            runner: None,
            validated_decode: false,
            disabled: false,
        }
    }

    fn runner_or_reload(&mut self) -> Option<&MlpRknnRunner> {
        if self.disabled {
            return None;
        }
        if self.runner.is_none() {
            match MlpRknnRunner::load_spec(&self.spec) {
                Ok(runner) => {
                    log::info!(
                        "text mlp {} rknn layer {} reloaded on demand",
                        self.spec.label(),
                        self.spec.layer_idx()
                    );
                    self.runner = Some(runner);
                }
                Err(err) => {
                    self.disabled = true;
                    log::warn!(
                        "text mlp {} rknn layer {} reload failed: {}; fallback to cpu/q8",
                        self.spec.label(),
                        self.spec.layer_idx(),
                        err
                    );
                    return None;
                }
            }
        }
        self.runner.as_ref()
    }

    fn release(&mut self, reason: &str) -> bool {
        if self.runner.take().is_some() {
            log::info!(
                "text mlp {} rknn layer {} released before {}",
                self.spec.label(),
                self.spec.layer_idx(),
                reason
            );
            true
        } else {
            false
        }
    }
}

fn make_mlp_rknn_state(spec: MlpRknnSpec) -> Option<Arc<Mutex<MlpRknnState>>> {
    if mlp_rknn_lazy_load_enabled() {
        log::info!(
            "text mlp {} rknn layer {} lazy load enabled; defer rknn_init until first decode",
            spec.label(),
            spec.layer_idx()
        );
        return Some(Arc::new(Mutex::new(MlpRknnState::new_lazy(spec))));
    }
    match MlpRknnRunner::load_spec(&spec) {
        Ok(runner) => Some(Arc::new(Mutex::new(MlpRknnState::new(runner)))),
        Err(err) => {
            log::warn!(
                "text mlp {} rknn layer {} load failed: {}; fallback to cpu/q8",
                spec.label(),
                spec.layer_idx(),
                err
            );
            None
        }
    }
}

impl Mlp {
    const MLP_RKNN_MAX_ABS_THRESHOLD: f32 = 5e-2;
    const MLP_RKNN_RMS_THRESHOLD: f32 = 5e-3;
    const MLP_GATE_UP_RKNN_MAX_ABS_THRESHOLD: f32 = 2e-1;
    const MLP_GATE_UP_RKNN_RMS_THRESHOLD: f32 = 1e-2;

    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        layer_idx: Option<usize>,
    ) -> candle_core::Result<Self> {
        let gguf_enabled = worker_gguf_enabled();
        let q8_allowed = !gguf_enabled
            && local_q8_enabled_for_layer(layer_idx)
            && local_q8_device_allows(vb.device());
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let mlp_rknn = if gguf_enabled {
            None
        } else {
            MlpRknnRunner::maybe_full_spec(layer_idx, h).and_then(make_mlp_rknn_state)
        };
        let mlp_gate_up_rknn =
            if !gguf_enabled && mlp_rknn.is_none() && mlp_rknn_gate_up_allowed(layer_idx) {
                MlpRknnRunner::maybe_gate_up_spec(layer_idx, h, i * 2).and_then(make_mlp_rknn_state)
            } else {
                None
            };
        let mlp_down_rknn = if !gguf_enabled && mlp_rknn.is_none() {
            MlpRknnRunner::maybe_down_spec(layer_idx, i, h).and_then(make_mlp_rknn_state)
        } else {
            None
        };
        let (gate_up_proj, down_proj, gate_up_q8, down_q8) = if gguf_enabled {
            let gate = load_worker_gguf_linear(layer_idx, "ffn_gate", h, i)?.ok_or_else(|| {
                candle_core::Error::Msg("worker GGUF gate projection missing".into())
            })?;
            let up = load_worker_gguf_linear(layer_idx, "ffn_up", h, i)?.ok_or_else(|| {
                candle_core::Error::Msg("worker GGUF up projection missing".into())
            })?;
            let down = load_worker_gguf_linear(layer_idx, "ffn_down", i, h)?.ok_or_else(|| {
                candle_core::Error::Msg("worker GGUF down projection missing".into())
            })?;
            if worker_gguf_fp16_prefill_enabled() {
                let layer_label = layer_idx
                    .map(|idx| format!("blk.{idx}"))
                    .unwrap_or_else(|| "blk.unknown".to_string());
                let fp16_prefill = build_worker_gguf_fp16_prefill_linear(
                    &format!("{layer_label}.ffn_gate_up.weight"),
                    &[&gate, &up],
                )?;
                let down = attach_worker_gguf_fp16_prefill(
                    &format!("{layer_label}.ffn_down.weight"),
                    down,
                )?;
                (
                    MlpGateUpProjection::Hybrid {
                        fp16_prefill,
                        gate,
                        up,
                    },
                    down,
                    None,
                    None,
                )
            } else {
                (MlpGateUpProjection::Split { gate, up }, down, None, None)
            }
        } else {
            let gate_proj = linear(h, i, vb.pp("gate_proj"))?;
            let up_proj = linear(h, i, vb.pp("up_proj"))?;
            let gate_up_weight = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?;
            let gate_up_dense = candle_nn::Linear::new(gate_up_weight, None);
            let down_dense = linear(i, h, vb.pp("down_proj"))?;
            let gate_up_q8 = if q8_allowed
                && local_q8_scope_allows("mlp")
                && mlp_rknn.is_none()
                && mlp_gate_up_rknn.is_none()
            {
                maybe_quantized_linear("local.mlp.gate_up_proj", &gate_up_dense)
            } else {
                None
            };
            let down_q8 = if q8_allowed
                && local_q8_scope_allows("mlp")
                && mlp_rknn.is_none()
                && mlp_down_rknn.is_none()
            {
                maybe_quantized_linear("local.mlp.down_proj", &down_dense)
            } else {
                None
            };
            let layer_label = layer_idx
                .map(|idx| format!("layer.{idx}"))
                .unwrap_or_else(|| "layer.unknown".to_string());
            (
                MlpGateUpProjection::Fused(TextLinear::from_dense(
                    &format!("{layer_label}.mlp.gate_up_proj"),
                    gate_up_dense,
                )?),
                TextLinear::from_dense(&format!("{layer_label}.mlp.down_proj"), down_dense)?,
                gate_up_q8,
                down_q8,
            )
        };
        Ok(Self {
            gate_up_proj,
            down_proj,
            gate_up_q8,
            down_q8,
            mlp_rknn,
            mlp_gate_up_rknn,
            mlp_down_rknn,
            split_rknn_validated_decode: Arc::new(Mutex::new(false)),
            intermediate_size: i,
        })
    }

    fn gate_up_rknn_active(&self) -> candle_core::Result<bool> {
        let Some(state) = &self.mlp_gate_up_rknn else {
            return Ok(false);
        };
        let state = state
            .lock()
            .map_err(|_| candle_core::Error::Msg("mlp gate_up rknn state poisoned".into()))?;
        Ok(!state.disabled)
    }

    fn split_rknn_validation_needed(&self) -> candle_core::Result<bool> {
        if !self.gate_up_rknn_active()? {
            return Ok(false);
        }
        let validated = self
            .split_rknn_validated_decode
            .lock()
            .map_err(|_| candle_core::Error::Msg("mlp split rknn state poisoned".into()))?;
        Ok(!*validated)
    }

    fn mark_split_rknn_validated(&self) -> candle_core::Result<()> {
        let mut validated = self
            .split_rknn_validated_decode
            .lock()
            .map_err(|_| candle_core::Error::Msg("mlp split rknn state poisoned".into()))?;
        *validated = true;
        Ok(())
    }

    fn disable_gate_up_rknn(&self, reason: &str) -> candle_core::Result<()> {
        if let Some(state) = &self.mlp_gate_up_rknn {
            let mut state = state
                .lock()
                .map_err(|_| candle_core::Error::Msg("mlp gate_up rknn state poisoned".into()))?;
            state.disabled = true;
            state.runner = None;
            log::warn!(
                "mlp gate_up rknn layer {} disabled: {}",
                state.spec.layer_idx(),
                reason
            );
        }
        Ok(())
    }

    fn forward_gate_up_rknn_or_cpu(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if x.dims().len() == 3 && x.dims()[0] == 1 && x.dims()[1] == 1 {
            if let Some(state) = &self.mlp_gate_up_rknn {
                let mut state = state.lock().map_err(|_| {
                    candle_core::Error::Msg("mlp gate_up rknn state poisoned".into())
                })?;
                if !state.disabled {
                    let validate_decode = !state.validated_decode;
                    let ref_out = if validate_decode {
                        Some(self.gate_up_proj.forward(x)?)
                    } else {
                        None
                    };
                    let out_dtype = ref_out
                        .as_ref()
                        .map(|t| t.dtype())
                        .unwrap_or_else(|| x.dtype());
                    let Some(runner) = state.runner_or_reload() else {
                        return match ref_out {
                            Some(ref_out) => Ok(ref_out),
                            None => self.gate_up_decode_or_cpu(x),
                        };
                    };
                    let layer_idx = runner.layer_idx();
                    let label = runner.label();
                    let rknn_result = runner.forward(x, x.device(), out_dtype);
                    match rknn_result {
                        Ok(rknn_out) => {
                            if let Some(ref_out) = ref_out {
                                let rknn_vec = tensor_to_f32_vec(&rknn_out.flatten_all()?)?;
                                let ref_vec = tensor_to_f32_vec(&ref_out.flatten_all()?)?;
                                let (max_abs, rms) = tensor_diff_stats(&rknn_vec, &ref_vec)
                                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                                if max_abs <= Self::MLP_GATE_UP_RKNN_MAX_ABS_THRESHOLD
                                    && rms <= Self::MLP_GATE_UP_RKNN_RMS_THRESHOLD
                                {
                                    state.validated_decode = true;
                                    log::info!(
                                        "mlp {} rknn layer {} validation accepted: max_abs={:.6e} rms={:.6e}",
                                        label,
                                        layer_idx,
                                        max_abs,
                                        rms
                                    );
                                } else {
                                    state.disabled = true;
                                    log::warn!(
                                            "mlp {} rknn layer {} validation failed: max_abs={:.6e} rms={:.6e}; fallback to cpu/q8 gate_up_proj",
                                            label,
                                            layer_idx,
                                        max_abs,
                                        rms
                                    );
                                    return Ok(ref_out);
                                }
                            }
                            return Ok(rknn_out);
                        }
                        Err(err) => {
                            state.disabled = true;
                            log::warn!(
                                "mlp {} rknn layer {} execution failed: {}; fallback to cpu/q8 gate_up_proj",
                                label,
                                layer_idx,
                                err
                            );
                            return match ref_out {
                                Some(ref_out) => Ok(ref_out),
                                None => self.gate_up_decode_or_cpu(x),
                            };
                        }
                    }
                }
            }
        }
        self.gate_up_decode_or_cpu(x)
    }

    fn gate_up_decode_or_cpu(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if let Some(q8) = &self.gate_up_q8 {
            if x.dims() == [1, 1, q8.in_dim] {
                return q8.forward_decode(x);
            }
        }
        self.gate_up_proj.forward(x)
    }

    fn mlp_mid_from_gate_up(&self, gate_up: &Tensor) -> candle_core::Result<Tensor> {
        let gate = gate_up.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = gate_up.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        candle_nn::ops::silu(&gate)? * up
    }

    fn down_q8_or_cpu(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if let Some(q8) = &self.down_q8 {
            if x.dims() == [1, 1, q8.in_dim] {
                return q8.forward_decode(x);
            }
        }
        self.down_proj.forward(x)
    }

    fn forward_down_rknn_or_cpu(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if x.dims().len() == 3 && x.dims()[0] == 1 && x.dims()[1] == 1 {
            if let Some(state) = &self.mlp_down_rknn {
                let mut state = state
                    .lock()
                    .map_err(|_| candle_core::Error::Msg("mlp down rknn state poisoned".into()))?;
                if !state.disabled {
                    let validate_decode = !state.validated_decode;
                    let ref_out = if validate_decode {
                        Some(self.down_proj.forward(x)?)
                    } else {
                        None
                    };
                    let out_dtype = ref_out
                        .as_ref()
                        .map(|t| t.dtype())
                        .unwrap_or_else(|| x.dtype());
                    let Some(runner) = state.runner_or_reload() else {
                        return match ref_out {
                            Some(ref_out) => Ok(ref_out),
                            None => self.down_q8_or_cpu(x),
                        };
                    };
                    let layer_idx = runner.layer_idx();
                    let label = runner.label();
                    let rknn_result = runner.forward(x, x.device(), out_dtype);
                    match rknn_result {
                        Ok(rknn_out) => {
                            if let Some(ref_out) = ref_out {
                                let rknn_vec = tensor_to_f32_vec(&rknn_out.flatten_all()?)?;
                                let ref_vec = tensor_to_f32_vec(&ref_out.flatten_all()?)?;
                                let (max_abs, rms) = tensor_diff_stats(&rknn_vec, &ref_vec)
                                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                                if max_abs <= Self::MLP_RKNN_MAX_ABS_THRESHOLD
                                    && rms <= Self::MLP_RKNN_RMS_THRESHOLD
                                {
                                    state.validated_decode = true;
                                    log::info!(
                                        "mlp {} rknn layer {} validation accepted: max_abs={:.6e} rms={:.6e}",
                                        label,
                                        layer_idx,
                                        max_abs,
                                        rms
                                    );
                                } else {
                                    state.disabled = true;
                                    log::warn!(
                                        "mlp {} rknn layer {} validation failed: max_abs={:.6e} rms={:.6e}; fallback to cpu/q8 down_proj",
                                        label,
                                        layer_idx,
                                        max_abs,
                                        rms
                                    );
                                    return Ok(ref_out);
                                }
                            }
                            return Ok(rknn_out);
                        }
                        Err(err) => {
                            state.disabled = true;
                            log::warn!(
                                "mlp {} rknn layer {} execution failed: {}; fallback to cpu/q8 down_proj",
                                label,
                                layer_idx,
                                err
                            );
                            return match ref_out {
                                Some(ref_out) => Ok(ref_out),
                                None => self.down_q8_or_cpu(x),
                            };
                        }
                    }
                }
            }
        }
        self.down_q8_or_cpu(x)
    }

    fn forward_cpu(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let x = self.mlp_mid_from_gate_up(&gate_up)?;
        self.down_proj.forward(&x)
    }

    fn forward_q8_or_cpu(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let validate_split = x.dims().len() == 3
            && x.dims()[0] == 1
            && x.dims()[1] == 1
            && self.split_rknn_validation_needed()?;
        let ref_out = if validate_split {
            Some(self.forward_cpu(x)?)
        } else {
            None
        };
        let gate_up = self.forward_gate_up_rknn_or_cpu(x)?;
        let x = self.mlp_mid_from_gate_up(&gate_up)?;
        let out = self.forward_down_rknn_or_cpu(&x)?;
        if let Some(ref_out) = ref_out {
            let out_vec = tensor_to_f32_vec(&out.flatten_all()?)?;
            let ref_vec = tensor_to_f32_vec(&ref_out.flatten_all()?)?;
            let (max_abs, rms) = tensor_diff_stats(&out_vec, &ref_vec)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            if max_abs <= Self::MLP_RKNN_MAX_ABS_THRESHOLD && rms <= Self::MLP_RKNN_RMS_THRESHOLD {
                self.mark_split_rknn_validated()?;
                log::info!(
                    "mlp split rknn final validation accepted: max_abs={:.6e} rms={:.6e}",
                    max_abs,
                    rms
                );
                Ok(out)
            } else {
                self.disable_gate_up_rknn("split final validation failed")?;
                log::warn!(
                    "mlp split rknn final validation failed: max_abs={:.6e} rms={:.6e}; fallback to cpu/q8 gate_up + down",
                    max_abs,
                    rms
                );
                Ok(ref_out)
            }
        } else {
            Ok(out)
        }
    }

    pub fn release_rknn(&self, reason: &str) -> usize {
        let mut released = 0usize;
        for state in [
            self.mlp_rknn.as_ref(),
            self.mlp_gate_up_rknn.as_ref(),
            self.mlp_down_rknn.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            match state.lock() {
                Ok(mut state) => {
                    if state.release(reason) {
                        released += 1;
                    }
                }
                Err(_) => {
                    log::warn!("text mlp rknn state poisoned while releasing before {reason}");
                }
            }
        }
        released
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if x.dims().len() == 3 && x.dims()[0] == 1 && x.dims()[1] == 1 {
            if let Some(state) = &self.mlp_rknn {
                let mut state = state
                    .lock()
                    .map_err(|_| candle_core::Error::Msg("mlp rknn state poisoned".into()))?;
                if !state.disabled {
                    let validate_decode = !state.validated_decode;
                    let ref_out = if validate_decode {
                        Some(self.forward_cpu(x)?)
                    } else {
                        None
                    };
                    let out_dtype = ref_out
                        .as_ref()
                        .map(|t| t.dtype())
                        .unwrap_or_else(|| x.dtype());
                    let Some(runner) = state.runner_or_reload() else {
                        return match ref_out {
                            Some(ref_out) => Ok(ref_out),
                            None => self.forward_q8_or_cpu(x),
                        };
                    };
                    let layer_idx = runner.layer_idx();
                    let label = runner.label();
                    let rknn_result = runner.forward(x, x.device(), out_dtype);
                    match rknn_result {
                        Ok(rknn_out) => {
                            if let Some(ref_out) = ref_out {
                                let rknn_vec = tensor_to_f32_vec(&rknn_out.flatten_all()?)?;
                                let ref_vec = tensor_to_f32_vec(&ref_out.flatten_all()?)?;
                                let (max_abs, rms) = tensor_diff_stats(&rknn_vec, &ref_vec)
                                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                                if max_abs <= Self::MLP_RKNN_MAX_ABS_THRESHOLD
                                    && rms <= Self::MLP_RKNN_RMS_THRESHOLD
                                {
                                    state.validated_decode = true;
                                    log::info!(
                                        "mlp {} rknn layer {} validation accepted: max_abs={:.6e} rms={:.6e}",
                                        label,
                                        layer_idx,
                                        max_abs,
                                        rms
                                    );
                                } else {
                                    state.disabled = true;
                                    log::warn!(
                                        "mlp {} rknn layer {} validation failed: max_abs={:.6e} rms={:.6e}; fallback to cpu/q8 mlp",
                                        label,
                                        layer_idx,
                                        max_abs,
                                        rms
                                    );
                                    return Ok(ref_out);
                                }
                            }
                            return Ok(rknn_out);
                        }
                        Err(err) => {
                            state.disabled = true;
                            log::warn!(
                                "mlp {} rknn layer {} execution failed: {}; fallback to cpu/q8 mlp",
                                label,
                                layer_idx,
                                err
                            );
                            return match ref_out {
                                Some(ref_out) => Ok(ref_out),
                                None => self.forward_q8_or_cpu(x),
                            };
                        }
                    }
                }
            }
        }
        self.forward_q8_or_cpu(x)
    }
}

#[derive(Debug, Clone)]
pub struct QwenAttention {
    qkv_proj: QkvProjection,
    o_proj: TextLinear,
    qkv_q8: Option<QuantizedLinear>,
    o_q8: Option<QuantizedLinear>,
    qkv_rknn: Option<Arc<Mutex<QkvRknnState>>>,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
    attn_f32: bool,
}

#[derive(Debug, Clone)]
enum QkvProjection {
    Fused(TextLinear),
    Split {
        q: TextLinear,
        k: TextLinear,
        v: TextLinear,
    },
    Hybrid {
        fp16_prefill: candle_nn::Linear,
        q: TextLinear,
        k: TextLinear,
        v: TextLinear,
    },
}

impl QkvProjection {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Fused(proj) => proj.forward(x),
            Self::Split { q, k, v } => {
                Tensor::cat(&[q.forward(x)?, k.forward(x)?, v.forward(x)?], D::Minus1)
            }
            Self::Hybrid {
                fp16_prefill,
                q,
                k,
                v,
            } => {
                if worker_gguf_prefill_for_dims(x.dims()) {
                    fp16_prefill.forward(x)
                } else {
                    Tensor::cat(&[q.forward(x)?, k.forward(x)?, v.forward(x)?], D::Minus1)
                }
            }
        }
    }
}

#[derive(Debug)]
struct QkvRknnState {
    runner: QkvRknnRunner,
    validated_decode: bool,
    disabled: bool,
}

impl QwenAttention {
    const QKV_RKNN_MAX_ABS_THRESHOLD: f32 = 1e-2;
    const QKV_RKNN_RMS_THRESHOLD: f32 = 1e-3;

    fn strided_decode_kv_enabled() -> bool {
        !matches!(
            std::env::var("QWEN3VL_STRIDED_DECODE_KV").ok().as_deref(),
            Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO")
        )
    }

    fn grouped_prefill_attention_enabled() -> bool {
        !matches!(
            std::env::var("QWEN3VL_GROUPED_PREFILL_ATTENTION")
                .ok()
                .as_deref(),
            Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO")
        )
    }

    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &Cache,
    ) -> candle_core::Result<Tensor> {
        let (_batch_size, _, seq_len, _head_dim) = x.dims4()?;
        let cos = cache.cosine(index_pos, seq_len)?;
        let sin = cache.sine(index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn tensor_diff_stats(lhs: &[f32], rhs: &[f32]) -> Result<(f32, f32)> {
        if lhs.len() != rhs.len() {
            bail!(
                "qkv diff length mismatch: lhs={} rhs={}",
                lhs.len(),
                rhs.len()
            );
        }
        let mut max_abs = 0.0f32;
        let mut sq_sum = 0.0f64;
        for (l, r) in lhs.iter().zip(rhs.iter()) {
            let diff = (l - r).abs();
            max_abs = max_abs.max(diff);
            sq_sum += f64::from(diff) * f64::from(diff);
        }
        let rms = if lhs.is_empty() {
            0.0
        } else {
            (sq_sum / lhs.len() as f64).sqrt() as f32
        };
        Ok((max_abs, rms))
    }

    fn project_qkv_cpu(&self, x: &Tensor, q8_allowed: bool) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3().map_err(|e| anyhow!("x.dims3 -> {e}"))?;
        if q8_allowed {
            if let Some(q8) = &self.qkv_q8 {
                if b_sz == 1 && seq_len == 1 && x.dims() == [1, 1, q8.in_dim] {
                    return q8.forward_decode(x).map_err(anyhow::Error::from);
                }
            }
        }
        self.qkv_proj.forward(x).map_err(anyhow::Error::from)
    }

    fn project_qkv(&self, x: &Tensor, decode_step: bool) -> Result<Tensor> {
        if decode_step {
            if let Some(state) = &self.qkv_rknn {
                let mut state = state
                    .lock()
                    .map_err(|_| anyhow!("qkv rknn state poisoned"))?;
                if !state.disabled {
                    let validate_decode = !state.validated_decode;
                    let ref_qkv = if validate_decode {
                        Some(self.project_qkv_cpu(x, false)?)
                    } else {
                        None
                    };
                    let out_dtype = ref_qkv
                        .as_ref()
                        .map(|t| t.dtype())
                        .unwrap_or_else(|| x.dtype());
                    match state.runner.forward_concat(x, x.device(), out_dtype) {
                        Ok(rknn_qkv) => {
                            if let Some(ref_qkv) = ref_qkv {
                                let rknn_vec = tensor_to_f32_vec(&rknn_qkv.flatten_all()?)
                                    .map_err(anyhow::Error::from)?;
                                let ref_vec = tensor_to_f32_vec(&ref_qkv.flatten_all()?)
                                    .map_err(anyhow::Error::from)?;
                                let (max_abs, rms) = Self::tensor_diff_stats(&rknn_vec, &ref_vec)?;
                                if max_abs <= Self::QKV_RKNN_MAX_ABS_THRESHOLD
                                    && rms <= Self::QKV_RKNN_RMS_THRESHOLD
                                {
                                    state.validated_decode = true;
                                    log::info!(
                                        "qkv rknn validation accepted: max_abs={:.6e} rms={:.6e}",
                                        max_abs,
                                        rms
                                    );
                                } else {
                                    state.disabled = true;
                                    log::warn!(
                                        "qkv rknn validation failed: max_abs={:.6e} rms={:.6e}; fallback to cpu qkv",
                                        max_abs,
                                        rms
                                    );
                                    return Ok(ref_qkv);
                                }
                            }
                            return Ok(rknn_qkv);
                        }
                        Err(err) => {
                            state.disabled = true;
                            log::warn!("qkv rknn execution failed: {}; fallback to cpu qkv", err);
                            return match ref_qkv {
                                Some(ref_qkv) => Ok(ref_qkv),
                                None => self.project_qkv_cpu(x, true),
                            };
                        }
                    }
                }
            }
        }
        self.project_qkv_cpu(x, true)
    }

    fn repeat_kv(&self, x: Tensor) -> candle_core::Result<Tensor> {
        candle_transformers::utils::repeat_kv(
            x,
            self.num_attention_heads / self.num_key_value_heads,
        )
    }

    fn grouped_decode_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (b_sz, q_heads, seq_len, head_dim) = q.dims4()?;
        let (_, kv_heads, kv_seq_len, _) = k.dims4()?;
        let n_rep = q_heads / kv_heads;
        let scale = (head_dim as f64).sqrt();

        let q = q.contiguous()?.reshape((kv_heads, n_rep, head_dim))?;
        let (k, v) = if Self::strided_decode_kv_enabled() {
            (k.squeeze(0)?, v.squeeze(0)?)
        } else {
            (
                k.contiguous()?.reshape((kv_heads, kv_seq_len, head_dim))?,
                v.contiguous()?.reshape((kv_heads, kv_seq_len, head_dim))?,
            )
        };
        let att = (q.matmul(&k.t()?)? / scale)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        att.matmul(&v)?.reshape((b_sz, q_heads, seq_len, head_dim))
    }

    fn grouped_prefill_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: &mut Cache,
        in_dtype: DType,
    ) -> Result<Tensor> {
        let (b_sz, q_heads, seq_len, head_dim) = q.dims4()?;
        let (_, kv_heads, kv_seq_len, _) = k.dims4()?;
        if seq_len != kv_seq_len
            || q_heads <= kv_heads
            || q_heads % kv_heads != 0
            || self.num_attention_heads != q_heads
            || self.num_key_value_heads != kv_heads
        {
            bail!(
                "grouped prefill attention shape mismatch q={:?} k={:?}",
                q.dims(),
                k.dims()
            );
        }

        let n_rep = q_heads / kv_heads;
        let scale = (head_dim as f64).sqrt();
        let q = q
            .contiguous()?
            .reshape((b_sz, kv_heads, n_rep * seq_len, head_dim))?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let att = (q.matmul(&k.t()?)? / scale)?;
        let att = att.reshape((b_sz, kv_heads, n_rep, seq_len, kv_seq_len))?;
        let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
        let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let att = att
            .contiguous()?
            .reshape((b_sz, kv_heads, n_rep * seq_len, kv_seq_len))?;
        att.matmul(&v)?
            .reshape((b_sz, kv_heads, n_rep, seq_len, head_dim))?
            .reshape((b_sz, q_heads, seq_len, head_dim))?
            .to_dtype(in_dtype)
            .map_err(anyhow::Error::from)
    }

    fn repeated_attention(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cache: &mut Cache,
        in_dtype: DType,
        seq_len: usize,
    ) -> Result<Tensor> {
        let k = self
            .repeat_kv(k)
            .map_err(|e| anyhow!("repeat_kv(k) -> {e}"))?;
        let v = self
            .repeat_kv(v)
            .map_err(|e| anyhow!("repeat_kv(v) -> {e}"))?;
        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = if seq_len == 1 {
            att
        } else {
            let mask = cache
                .mask(seq_len)
                .map_err(|e| anyhow!("cache.mask({seq_len}) -> {e}"))?
                .broadcast_as(att.shape())
                .map_err(|e| anyhow!("mask.broadcast_as({:?}) -> {e}", att.shape()))?;
            masked_fill(&att, &mask, f32::NEG_INFINITY)
                .map_err(|e| anyhow!("masked_fill -> {e}"))?
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        Ok(att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?)
    }

    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        layer_idx: Option<usize>,
    ) -> candle_core::Result<Self> {
        let gguf_enabled = worker_gguf_enabled();
        let q8_allowed = !gguf_enabled
            && local_q8_enabled_for_layer(layer_idx)
            && local_q8_device_allows(vb.device());
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let qkv_rknn = if gguf_enabled {
            None
        } else {
            QkvRknnRunner::maybe_load(layer_idx, cfg.hidden_size, size_q + 2 * size_kv).map(
                |runner| {
                    Arc::new(Mutex::new(QkvRknnState {
                        runner,
                        validated_decode: false,
                        disabled: false,
                    }))
                },
            )
        };
        let (qkv_proj, o_proj, qkv_q8, o_q8) = if gguf_enabled {
            let q = load_worker_gguf_linear(layer_idx, "attn_q", size_in, size_q)?.ok_or_else(
                || candle_core::Error::Msg("worker GGUF q projection missing".into()),
            )?;
            let k = load_worker_gguf_linear(layer_idx, "attn_k", size_in, size_kv)?.ok_or_else(
                || candle_core::Error::Msg("worker GGUF k projection missing".into()),
            )?;
            let v = load_worker_gguf_linear(layer_idx, "attn_v", size_in, size_kv)?.ok_or_else(
                || candle_core::Error::Msg("worker GGUF v projection missing".into()),
            )?;
            let o = load_worker_gguf_linear(layer_idx, "attn_output", size_q, size_in)?
                .ok_or_else(|| {
                    candle_core::Error::Msg("worker GGUF output projection missing".into())
                })?;
            if worker_gguf_fp16_prefill_enabled() {
                let layer_label = layer_idx
                    .map(|idx| format!("blk.{idx}"))
                    .unwrap_or_else(|| "blk.unknown".to_string());
                let fp16_prefill = build_worker_gguf_fp16_prefill_linear(
                    &format!("{layer_label}.attn_qkv.weight"),
                    &[&q, &k, &v],
                )?;
                let o = attach_worker_gguf_fp16_prefill(
                    &format!("{layer_label}.attn_output.weight"),
                    o,
                )?;
                (
                    QkvProjection::Hybrid {
                        fp16_prefill,
                        q,
                        k,
                        v,
                    },
                    o,
                    None,
                    None,
                )
            } else {
                (QkvProjection::Split { q, k, v }, o, None, None)
            }
        } else {
            let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
            let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
            let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
            let qkv_weight = Tensor::cat(&[q_proj.weight(), k_proj.weight(), v_proj.weight()], 0)?;
            let qkv_dense = candle_nn::Linear::new(qkv_weight, None);
            let o_dense = linear(size_q, size_in, vb.pp("o_proj"))?;
            let attn_q8_allowed = q8_allowed && local_q8_scope_allows("attn");
            let qkv_q8 = if attn_q8_allowed && qkv_rknn.is_none() {
                maybe_quantized_linear("local.self_attn.qkv_proj", &qkv_dense)
            } else {
                None
            };
            let o_q8 = if attn_q8_allowed {
                maybe_quantized_linear("local.self_attn.o_proj", &o_dense)
            } else {
                None
            };
            let layer_label = layer_idx
                .map(|idx| format!("layer.{idx}"))
                .unwrap_or_else(|| "layer.unknown".to_string());
            (
                QkvProjection::Fused(TextLinear::from_dense(
                    &format!("{layer_label}.self_attn.qkv_proj"),
                    qkv_dense,
                )?),
                TextLinear::from_dense(&format!("{layer_label}.self_attn.o_proj"), o_dense)?,
                qkv_q8,
                o_q8,
            )
        };

        Ok(Self {
            qkv_proj,
            o_proj,
            qkv_q8,
            o_q8,
            qkv_rknn,

            q_norm: candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,

            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
            q_size: size_q,
            kv_size: size_kv,
            attn_f32: cfg.attn_f32,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3().map_err(|e| anyhow!("x.dims3 -> {e}"))?;
        let qkv = self
            .project_qkv(x, b_sz == 1 && seq_len == 1)
            .map_err(|e| anyhow!("qkv.forward -> {e}"))?;
        let q = qkv
            .narrow(D::Minus1, 0, self.q_size)
            .map_err(|e| anyhow!("qkv.q narrow -> {e}"))?;
        let k = qkv
            .narrow(D::Minus1, self.q_size, self.kv_size)
            .map_err(|e| anyhow!("qkv.k narrow -> {e}"))?;
        let v = qkv
            .narrow(D::Minus1, self.q_size + self.kv_size, self.kv_size)
            .map_err(|e| anyhow!("qkv.v narrow -> {e}"))?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply q/k RMSNorm per head-dim (broadcast across batch/head/seq).
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = self
            .apply_rotary_emb(&q, index_pos, cache)
            .map_err(|e| anyhow!("q.rope -> {e}"))?;
        let k = self
            .apply_rotary_emb(&k, index_pos, cache)
            .map_err(|e| anyhow!("k.rope -> {e}"))?;

        // （新增）Qwen3-VL 的文本注意力同样复用 llama3::Cache（rope + KV-cache + mask）。
        // 这里传入 index_pos，用于在新请求 prefill 时覆盖旧 kv，避免第二次请求崩溃。
        let decode_fast_path = b_sz == 1 && seq_len == 1 && index_pos > 0;
        let (k, v) = if decode_fast_path {
            cache
                .process_kv_decode_in_place(block_idx, index_pos, k, v)
                .map_err(|e| {
                    anyhow!("cache.process_kv_decode_in_place(block={block_idx}) -> {e}")
                })?
        } else {
            cache
                .process_kv(block_idx, index_pos, k, v)
                .map_err(|e| anyhow!("cache.process_kv(block={block_idx}) -> {e}"))?
        };

        let y = {
            let in_dtype = q.dtype();
            let compute_dtype = if self.attn_f32 { DType::F32 } else { in_dtype };
            let q = if compute_dtype == in_dtype {
                q
            } else {
                q.to_dtype(compute_dtype)?
            };
            let k = if compute_dtype == k.dtype() {
                k
            } else {
                k.to_dtype(compute_dtype)?
            };
            let v = if compute_dtype == v.dtype() {
                v
            } else {
                v.to_dtype(compute_dtype)?
            };

            if b_sz == 1
                && seq_len == 1
                && self.num_attention_heads > self.num_key_value_heads
                && self.num_attention_heads % self.num_key_value_heads == 0
            {
                self.grouped_decode_attention(&q, &k, &v)?
                    .to_dtype(in_dtype)?
            } else if b_sz == 1
                && seq_len > 1
                && self.num_attention_heads > self.num_key_value_heads
                && self.num_attention_heads % self.num_key_value_heads == 0
                && Self::grouped_prefill_attention_enabled()
            {
                match self.grouped_prefill_attention(&q, &k, &v, cache, in_dtype) {
                    Ok(y) => y,
                    Err(err) => {
                        log::warn!(
                            "grouped prefill attention failed, fallback to repeated attention: {}",
                            err
                        );
                        self.repeated_attention(q, k, v, cache, in_dtype, seq_len)?
                    }
                }
            } else {
                self.repeated_attention(q, k, v, cache, in_dtype, seq_len)?
            }
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = if let Some(q8) = &self.o_q8 {
            if b_sz == 1 && seq_len == 1 && y.dims() == [1, 1, q8.in_dim] {
                q8.forward_decode(&y)
            } else {
                self.o_proj.forward(&y)
            }
        } else {
            self.o_proj.forward(&y)
        }
        .map_err(|e| anyhow!("o_proj.forward -> {e}"))?;
        Ok(y)
    }
}

#[derive(Debug, Clone)]
pub struct Transformer {
    name: String,
    rms_1: RmsNorm,
    attn: QwenAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl std::fmt::Display for Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

#[async_trait]
impl Forwarder for Transformer {
    fn load(name: String, vb: VarBuilder, cfg: &Config) -> Result<Box<Self>> {
        let layer_idx = parse_layer_idx(&name);
        let attn = QwenAttention::load(vb.pp("self_attn"), cfg, layer_idx)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg, layer_idx)?;
        let rms_1 =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Box::new(Self {
            name,
            rms_1,
            attn,
            rms_2,
            mlp,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let seq_len = x.dims().get(1).copied().unwrap_or_default();
        let profile = local_profile_enabled() || (seq_len > 1 && local_prefill_profile_enabled());
        let total_start = profile.then(Instant::now);

        let residual = x;
        let step_start = profile.then(Instant::now);
        let x = self.rms_1.forward(x).map_err(|e| anyhow!("rms_1: {e}"))?;
        let rms1_ms = step_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);

        let step_start = profile.then(Instant::now);
        let x = (self
            .attn
            .forward(&x, index_pos, block_idx, cache)
            .map_err(|e| anyhow!("attention: {e}"))?
            + residual)
            .map_err(|e| anyhow!("residual: {e}"))?;
        let attn_ms = step_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);

        let residual = &x;
        let step_start = profile.then(Instant::now);
        let x = self.rms_2.forward(&x).map_err(|e| anyhow!("rms_2: {e}"))?;
        let rms2_ms = step_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);

        let step_start = profile.then(Instant::now);
        let x = (self.mlp.forward(&x).map_err(|e| anyhow!("mlp: {e}"))? + residual)
            .map_err(|e| anyhow!("mlp residual: {e}"))?;
        let mlp_ms = step_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);

        if let (Some(total_start), Some(rms1_ms), Some(attn_ms), Some(rms2_ms), Some(mlp_ms)) =
            (total_start, rms1_ms, attn_ms, rms2_ms, mlp_ms)
        {
            log::info!(
                "profile local layer name={} block={} index={} seq={} total_ms={:.3} rms1={:.3} attn={:.3} rms2={:.3} mlp={:.3}",
                self.name,
                block_idx,
                index_pos,
                seq_len,
                total_start.elapsed().as_secs_f64() * 1000.0,
                rms1_ms,
                attn_ms,
                rms2_ms,
                mlp_ms
            );
        }
        Ok(x)
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        self.forward(x, index_pos, block_idx, cache).await
    }

    fn layer_name(&self) -> &str {
        &self.name
    }

    fn release_mlp_rknn(&self, reason: &str) -> usize {
        self.mlp.release_rknn(reason)
    }
}
