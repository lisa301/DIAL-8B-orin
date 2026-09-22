use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor};
use regex::Regex;
use rknpu2::{
    api::{runtime::RuntimeAPI, RknnInitFlags},
    bf16, f16,
    io::{
        buffer::{BufMutView, BufView},
        input::Input,
        output::{Output, OutputKind},
    },
    query::{
        CurrentOutputAttr, InputAttr, InputDynamicRange, InputOutputNum, OutputAttr, TensorAttrView,
    },
    rknn::NpuCores,
    tensor::DataTypeKind,
    tensor::TensorFormatKind,
    utils::find_rknn_library,
    RKNN,
};

use crate::models::llama3::{Cache, Config};

#[derive(Debug, Clone)]
struct LayerKv {
    past_len: usize,
    k: Vec<f32>,
    v: Vec<f32>,
    native: Option<NativeLayerKv>,
}

#[derive(Debug, Clone)]
pub struct LayerKvDeltaLayoutDebug {
    pub k_head_seq_dim: Vec<f32>,
    pub k_seq_dim_head: Vec<f32>,
    pub v_head_seq_dim: Vec<f32>,
    pub v_seq_dim_head: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KvStorageDType {
    F32,
    F16,
    BF16,
}

impl KvStorageDType {
    fn from_rknn_dtype(dtype: &DataTypeKind) -> Result<Self> {
        match dtype {
            DataTypeKind::Float32(_) => Ok(Self::F32),
            DataTypeKind::Float16(_) => Ok(Self::F16),
            DataTypeKind::BFloat16(_) => Ok(Self::BF16),
            other => bail!("unsupported rknn kv dtype: {other:?}"),
        }
    }
}

#[derive(Debug, Clone)]
enum KvData {
    F32(Vec<f32>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
}

impl KvData {
    fn zeros(dtype: KvStorageDType, n: usize) -> Self {
        match dtype {
            KvStorageDType::F32 => Self::F32(vec![0.0; n]),
            KvStorageDType::F16 => Self::F16(vec![f16::from_f32(0.0); n]),
            KvStorageDType::BF16 => Self::BF16(vec![bf16::from_f32(0.0); n]),
        }
    }

    fn from_f32_slice(data: &[f32], dtype: KvStorageDType) -> Self {
        match dtype {
            KvStorageDType::F32 => Self::F32(data.to_vec()),
            KvStorageDType::F16 => Self::F16(data.iter().map(|v| f16::from_f32(*v)).collect()),
            KvStorageDType::BF16 => Self::BF16(data.iter().map(|v| bf16::from_f32(*v)).collect()),
        }
    }

    fn replace_from_f32_prefix(&mut self, data: &[f32], n: usize) -> Result<()> {
        if data.len() < n {
            bail!(
                "replace_from_f32_prefix: data too short {} < {}",
                data.len(),
                n
            );
        }
        match self {
            Self::F32(buf) => {
                buf.clear();
                buf.extend_from_slice(&data[..n]);
            }
            Self::F16(buf) => {
                buf.clear();
                buf.reserve(n);
                for v in &data[..n] {
                    buf.push(f16::from_f32(*v));
                }
            }
            Self::BF16(buf) => {
                buf.clear();
                buf.reserve(n);
                for v in &data[..n] {
                    buf.push(bf16::from_f32(*v));
                }
            }
        }
        Ok(())
    }

    fn append_from_f32_prefix(&mut self, data: &[f32], n: usize) -> Result<()> {
        if data.len() < n {
            bail!(
                "append_from_f32_prefix: data too short {} < {}",
                data.len(),
                n
            );
        }
        match self {
            Self::F32(buf) => buf.extend_from_slice(&data[..n]),
            Self::F16(buf) => {
                buf.reserve(n);
                for v in &data[..n] {
                    buf.push(f16::from_f32(*v));
                }
            }
            Self::BF16(buf) => {
                buf.reserve(n);
                for v in &data[..n] {
                    buf.push(bf16::from_f32(*v));
                }
            }
        }
        Ok(())
    }

    fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Self::F32(buf) => buf.clone(),
            Self::F16(buf) => buf.iter().map(|v| v.to_f32()).collect(),
            Self::BF16(buf) => buf.iter().map(|v| v.to_f32()).collect(),
        }
    }

    fn write_cache_layout_seq(
        &mut self,
        data: &[f32],
        heads: usize,
        seq_idx: usize,
        head_dim: usize,
        format: TensorFormatKind,
        bucket_len: usize,
    ) -> Result<()> {
        let expected = heads.checked_mul(head_dim).ok_or_else(|| {
            anyhow!(
                "kv write size overflow: heads={} head_dim={}",
                heads,
                head_dim
            )
        })?;
        if data.len() < expected {
            bail!(
                "kv write_cache_layout_seq: data too short {} < {}",
                data.len(),
                expected
            );
        }
        match self {
            Self::F32(buf) => {
                write_cache_layout_seq_impl(buf, data, heads, seq_idx, head_dim, format, bucket_len)
            }
            Self::F16(buf) => {
                let converted: Vec<f16> =
                    data[..expected].iter().map(|v| f16::from_f32(*v)).collect();
                write_cache_layout_seq_impl(
                    buf, &converted, heads, seq_idx, head_dim, format, bucket_len,
                )
            }
            Self::BF16(buf) => {
                let converted: Vec<bf16> = data[..expected]
                    .iter()
                    .map(|v| bf16::from_f32(*v))
                    .collect();
                write_cache_layout_seq_impl(
                    buf, &converted, heads, seq_idx, head_dim, format, bucket_len,
                )
            }
        }
    }
}

#[derive(Debug, Clone)]
struct NativeLayerKv {
    bucket_len: usize,
    k_format: TensorFormatKind,
    v_format: TensorFormatKind,
    k_dtype: KvStorageDType,
    v_dtype: KvStorageDType,
    k: KvData,
    v: KvData,
}

impl LayerKv {
    fn ensure_native_buffers(
        &mut self,
        bucket_len: usize,
        k_format: TensorFormatKind,
        v_format: TensorFormatKind,
        k_dtype: KvStorageDType,
        v_dtype: KvStorageDType,
        heads: usize,
        head_dim: usize,
    ) {
        let rebuild = match &self.native {
            Some(native) => {
                native.bucket_len != bucket_len
                    || native.k_dtype != k_dtype
                    || native.v_dtype != v_dtype
                    || !same_tensor_format(&native.k_format, &k_format)
                    || !same_tensor_format(&native.v_format, &v_format)
            }
            None => true,
        };
        if !rebuild {
            return;
        }

        let k_native = KvData::from_f32_slice(
            &kv_cache_to_rknn_layout(
                &pad_kv_to_bucket(&self.k, heads, self.past_len, head_dim, bucket_len),
                heads,
                bucket_len,
                head_dim,
                k_format,
            ),
            k_dtype,
        );
        let v_native = KvData::from_f32_slice(
            &kv_cache_to_rknn_layout(
                &pad_kv_to_bucket(&self.v, heads, self.past_len, head_dim, bucket_len),
                heads,
                bucket_len,
                head_dim,
                v_format,
            ),
            v_dtype,
        );
        self.native = Some(NativeLayerKv {
            bucket_len,
            k_format,
            v_format,
            k_dtype,
            v_dtype,
            k: k_native,
            v: v_native,
        });
    }

    fn append_cache_layout_delta(
        &mut self,
        present_k: &[f32],
        present_v: &[f32],
        heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let expected = heads.checked_mul(head_dim).ok_or_else(|| {
            anyhow!(
                "kv append size overflow: heads={} head_dim={}",
                heads,
                head_dim
            )
        })?;
        if present_k.len() < expected || present_v.len() < expected {
            bail!(
                "append_cache_layout_delta: present delta too short k={} v={} expected={}",
                present_k.len(),
                present_v.len(),
                expected
            );
        }
        append_cache_layout_delta_vec(&mut self.k, present_k, self.past_len, heads, head_dim)?;
        append_cache_layout_delta_vec(&mut self.v, present_v, self.past_len, heads, head_dim)?;
        if let Some(native) = self.native.as_mut() {
            if self.past_len < native.bucket_len {
                native.k.write_cache_layout_seq(
                    present_k,
                    heads,
                    self.past_len,
                    head_dim,
                    native.k_format,
                    native.bucket_len,
                )?;
                native.v.write_cache_layout_seq(
                    present_v,
                    heads,
                    self.past_len,
                    head_dim,
                    native.v_format,
                    native.bucket_len,
                )?;
            } else {
                self.native = None;
            }
        }
        self.past_len += 1;
        Ok(())
    }

    fn replace_cache_layout_full(
        &mut self,
        present_k: &[f32],
        present_v: &[f32],
        present_len: usize,
        heads: usize,
        head_dim: usize,
    ) {
        let expected = heads * present_len * head_dim;
        self.k.clear();
        self.k.extend_from_slice(&present_k[..expected]);
        self.v.clear();
        self.v.extend_from_slice(&present_v[..expected]);
        self.past_len = present_len;
        if let Some(native) = self.native.as_mut() {
            native.k = KvData::from_f32_slice(
                &kv_cache_to_rknn_layout(
                    &pad_kv_to_bucket(&self.k, heads, self.past_len, head_dim, native.bucket_len),
                    heads,
                    native.bucket_len,
                    head_dim,
                    native.k_format,
                ),
                native.k_dtype,
            );
            native.v = KvData::from_f32_slice(
                &kv_cache_to_rknn_layout(
                    &pad_kv_to_bucket(&self.v, heads, self.past_len, head_dim, native.bucket_len),
                    heads,
                    native.bucket_len,
                    head_dim,
                    native.v_format,
                ),
                native.v_dtype,
            );
        }
    }
}

fn append_cache_layout_delta_vec(
    dst: &mut Vec<f32>,
    delta: &[f32],
    old_seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<()> {
    let old_per_head = old_seq.checked_mul(head_dim).ok_or_else(|| {
        anyhow!(
            "kv old per-head size overflow: old_seq={} head_dim={}",
            old_seq,
            head_dim
        )
    })?;
    let old_expected = heads.checked_mul(old_per_head).ok_or_else(|| {
        anyhow!(
            "kv old size overflow: heads={} old_seq={} head_dim={}",
            heads,
            old_seq,
            head_dim
        )
    })?;
    let delta_expected = heads.checked_mul(head_dim).ok_or_else(|| {
        anyhow!(
            "kv delta size overflow: heads={} head_dim={}",
            heads,
            head_dim
        )
    })?;
    if dst.len() != old_expected {
        bail!(
            "kv cache length mismatch before append: got={} expected={} heads={} old_seq={} head_dim={}",
            dst.len(),
            old_expected,
            heads,
            old_seq,
            head_dim
        );
    }
    if delta.len() < delta_expected {
        bail!(
            "kv delta too short before append: got={} expected={} heads={} head_dim={}",
            delta.len(),
            delta_expected,
            heads,
            head_dim
        );
    }

    let mut out = Vec::with_capacity(heads * (old_seq + 1) * head_dim);
    for h in 0..heads {
        let old_off = h * old_per_head;
        out.extend_from_slice(&dst[old_off..old_off + old_per_head]);

        let delta_off = h * head_dim;
        out.extend_from_slice(&delta[delta_off..delta_off + head_dim]);
    }
    *dst = out;
    Ok(())
}

fn same_tensor_format(a: &TensorFormatKind, b: &TensorFormatKind) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

fn write_cache_layout_seq_impl<T: Copy>(
    dst: &mut [T],
    src: &[T],
    heads: usize,
    seq_idx: usize,
    head_dim: usize,
    format: TensorFormatKind,
    bucket_len: usize,
) -> Result<()> {
    let expected_total = heads
        .checked_mul(bucket_len)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| anyhow!("kv native buffer size overflow"))?;
    if dst.len() < expected_total {
        bail!(
            "kv native buffer too short {} < {}",
            dst.len(),
            expected_total
        );
    }
    match format {
        TensorFormatKind::NHWC(_) => {
            for d in 0..head_dim {
                for h in 0..heads {
                    let src_idx = h * head_dim + d;
                    let dst_idx = seq_idx * head_dim * heads + d * heads + h;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
        _ => {
            for h in 0..heads {
                for d in 0..head_dim {
                    let src_idx = h * head_dim + d;
                    let dst_idx = h * bucket_len * head_dim + seq_idx * head_dim + d;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
    Ok(())
}

struct ChunkModel {
    stage: ChunkStage,
    layer_start: usize,
    layer_end: usize,
    rknn: RKNN<RuntimeAPI>,
    input_attrs: Vec<InputAttr>,
    output_attrs: Vec<OutputAttr>,
    has_past_mask: bool,
    input_dynamic_shapes: Vec<Vec<Vec<u32>>>,
    is_dynamic_model: bool,
    supported_buckets: Vec<usize>,
    prefill_expects_layer_add: bool,
}

impl ChunkModel {
    fn num_layers(&self) -> usize {
        self.layer_end - self.layer_start + 1
    }

    fn has_dynamic_inputs(&self) -> bool {
        self.input_dynamic_shapes
            .iter()
            .any(|shapes| !shapes.is_empty())
    }

    fn prefill_expects_layer_add(&self) -> bool {
        matches!(self.stage, ChunkStage::Prefill) && self.prefill_expects_layer_add
    }

    fn load(
        path: &Path,
        layer_start: usize,
        layer_end: usize,
        lib_path: &Path,
        stage: ChunkStage,
    ) -> Result<Self> {
        let mut model_data = std::fs::read(path)
            .map_err(|e| anyhow!("failed to read rknn model {}: {e}", path.display()))?;

        let rknn = RKNN::new_with_library(
            lib_path.to_path_buf(),
            &mut model_data,
            RknnInitFlags::builder(),
        )
        .map_err(|e| anyhow!("rknn init failed for {}: {e}", path.display()))?;

        if let Err(e) = rknn.set_core_mask(NpuCores::cores_0_1_2()) {
            log::warn!(
                "rknn set_core_mask(0_1_2) failed for {}: {}; fallback to auto",
                path.display(),
                e
            );
            if let Err(e2) = rknn.set_core_mask(NpuCores::auto()) {
                log::warn!(
                    "rknn set_core_mask(auto) failed for {}: {}; continue with runtime default",
                    path.display(),
                    e2
                );
            }
        }

        let io_num = rknn
            .query::<InputOutputNum>()
            .map_err(|e| anyhow!("rknn query io_num failed for {}: {e}", path.display()))?;

        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let is_dynamic_model = file_name.contains("_dynamic_");
        let mut input_attrs = Vec::with_capacity(io_num.input_num() as usize);
        let mut input_dynamic_shapes = Vec::with_capacity(io_num.input_num() as usize);
        for i in 0..io_num.input_num() {
            input_attrs.push(rknn.query_with_input::<InputAttr>(i).map_err(|e| {
                anyhow!(
                    "rknn query input attr[{i}] failed for {}: {e}",
                    path.display()
                )
            })?);
            let dyn_shapes = if is_dynamic_model {
                rknn.query_with_input::<InputDynamicRange>(i)
                    .ok()
                    .map(|r| r.shapes())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            input_dynamic_shapes.push(dyn_shapes);
        }

        let mut output_attrs = Vec::with_capacity(io_num.output_num() as usize);
        for i in 0..io_num.output_num() {
            output_attrs.push(rknn.query_with_input::<OutputAttr>(i).map_err(|e| {
                anyhow!(
                    "rknn query output attr[{i}] failed for {}: {e}",
                    path.display()
                )
            })?);
        }

        let n_layers = layer_end - layer_start + 1;
        let expected_outputs = 1 + 2 * n_layers;
        let (has_past_mask, supported_buckets) = match stage {
            ChunkStage::Decode => {
                let expected_inputs = 3 + 2 * n_layers;
                let has_past_mask = input_attrs.len() == expected_inputs + 1;
                if input_attrs.len() != expected_inputs && !has_past_mask {
                    bail!(
                        "{} {} input_num mismatch: expected {} or {}, got {}",
                        path.display(),
                        stage.as_str(),
                        expected_inputs,
                        expected_inputs + 1,
                        input_attrs.len()
                    );
                }
                (
                    has_past_mask,
                    infer_supported_buckets(
                        n_layers,
                        &input_attrs,
                        &input_dynamic_shapes,
                        has_past_mask,
                    ),
                )
            }
            ChunkStage::Prefill => {
                let without_add = 3;
                let with_add = 3 + n_layers;
                if input_attrs.len() != without_add && input_attrs.len() != with_add {
                    bail!(
                        "{} {} input_num mismatch: expected {} or {}, got {}",
                        path.display(),
                        stage.as_str(),
                        without_add,
                        with_add,
                        input_attrs.len()
                    );
                }
                (false, Vec::new())
            }
        };
        if output_attrs.len() != expected_outputs {
            bail!(
                "{} {} output_num mismatch: expected {}, got {}",
                path.display(),
                stage.as_str(),
                expected_outputs,
                output_attrs.len()
            );
        }
        let prefill_expects_layer_add =
            matches!(stage, ChunkStage::Prefill) && input_attrs.len() == 3 + n_layers;

        Ok(Self {
            stage,
            layer_start,
            layer_end,
            rknn,
            input_attrs,
            output_attrs,
            has_past_mask,
            input_dynamic_shapes,
            is_dynamic_model,
            supported_buckets,
            prefill_expects_layer_add,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChunkStage {
    Decode,
    Prefill,
}

impl ChunkStage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Prefill => "prefill",
        }
    }
}

enum OwnedInputBuffer<'a> {
    BorrowF32(&'a [f32]),
    BorrowF16(&'a [f16]),
    BorrowBF16(&'a [bf16]),
    OwnF32(Vec<f32>),
    OwnF16(Vec<f16>),
    OwnBF16(Vec<bf16>),
}

impl<'a> OwnedInputBuffer<'a> {
    fn from_f32(data: &'a [f32], dtype: DataTypeKind) -> Result<Self> {
        let out = match dtype {
            DataTypeKind::Float32(_) => Self::BorrowF32(data),
            DataTypeKind::Float16(_) => {
                let converted: Vec<f16> = data.iter().map(|v| f16::from_f32(*v)).collect();
                Self::OwnF16(converted)
            }
            DataTypeKind::BFloat16(_) => {
                let converted: Vec<bf16> = data.iter().map(|v| bf16::from_f32(*v)).collect();
                Self::OwnBF16(converted)
            }
            other => bail!("unsupported rknn input dtype: {other:?}"),
        };
        Ok(out)
    }

    fn from_kv_data(data: &'a KvData) -> Self {
        match data {
            KvData::F32(v) => Self::BorrowF32(v),
            KvData::F16(v) => Self::BorrowF16(v),
            KvData::BF16(v) => Self::BorrowBF16(v),
        }
    }

    fn as_buf_view(&'a self) -> BufView<'a> {
        match self {
            Self::BorrowF32(v) => BufView::F32(v),
            Self::BorrowF16(v) => BufView::F16(v),
            Self::BorrowBF16(v) => BufView::BF16(v),
            Self::OwnF32(v) => BufView::F32(v),
            Self::OwnF16(v) => BufView::F16(v),
            Self::OwnBF16(v) => BufView::BF16(v),
        }
    }
}

fn build_rotary(position: usize, head_dim: usize, rope_theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut freqs = vec![0f32; half];
    for (i, f) in freqs.iter_mut().enumerate() {
        let theta = 2 * i;
        let inv = 1f64 / rope_theta.powf(theta as f64 / head_dim as f64);
        *f = position as f32 * inv as f32;
    }

    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for i in 0..half {
        cos[i] = freqs[i].cos();
        sin[i] = freqs[i].sin();
    }
    (cos, sin)
}

fn parse_chunk_path(path: &Path, re: &Regex) -> Option<(usize, usize)> {
    let name = path.file_name()?.to_str()?;
    let caps = re.captures(name)?;
    let s = caps.get(1)?.as_str().parse::<usize>().ok()?;
    let e = caps.get(2)?.as_str().parse::<usize>().ok()?;
    Some((s, e))
}

fn validate_chunk_coverage(
    mut parsed: Vec<(usize, usize, PathBuf)>,
    expected_layers: usize,
    stage_name: &str,
) -> Result<Vec<(usize, usize, PathBuf)>> {
    if parsed.is_empty() {
        bail!("no {stage_name} chunk filename matched pattern lXX_lYY");
    }
    parsed.sort_by_key(|(s, _, _)| *s);

    let mut expected = 0usize;
    for (s, e, path) in &parsed {
        if *s != expected {
            bail!(
                "{stage_name} chunk coverage gap: expected layer {}, got {} ({})",
                expected,
                s,
                path.display()
            );
        }
        if e < s {
            bail!(
                "{stage_name} invalid chunk range {}..{} in {}",
                s,
                e,
                path.display()
            );
        }
        expected = e + 1;
    }
    if expected != expected_layers {
        bail!(
            "{stage_name} chunk coverage mismatch: got 0..{}, expected 0..{}",
            expected.saturating_sub(1),
            expected_layers.saturating_sub(1)
        );
    }
    Ok(parsed)
}

fn static_bucket_from_path(path: &Path) -> Option<usize> {
    let name = path.file_name()?.to_str()?;
    let marker = "_static_p";
    let start = name.find(marker)? + marker.len();
    let digits = name[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        None
    } else {
        digits.parse::<usize>().ok()
    }
}

fn chunk_candidate_sort_key(path: &Path) -> (u8, std::cmp::Reverse<usize>, String) {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    if let Some(bucket) = static_bucket_from_path(path) {
        return (0, std::cmp::Reverse(bucket), name);
    }
    if name.contains("_dynamic_") {
        return (1, std::cmp::Reverse(usize::MAX), name);
    }
    (2, std::cmp::Reverse(0), name)
}

fn tensor_to_f32_vec3(x: &Tensor, expected_hidden: usize) -> Result<(usize, Vec<f32>)> {
    let x = x
        .to_device(&Device::Cpu)
        .map_err(|e| anyhow!("x.to_cpu failed: {e}"))?
        .to_dtype(DType::F32)
        .map_err(|e| anyhow!("x.to_f32 failed: {e}"))?
        .contiguous()
        .map_err(|e| anyhow!("x.contiguous failed: {e}"))?;
    let (b, s, h) = x.dims3().map_err(|e| anyhow!("x.dims3 failed: {e}"))?;
    if b != 1 || h != expected_hidden {
        bail!("text rknn expects x shape (1,seq,{expected_hidden}), got ({b},{s},{h})");
    }
    let data = x
        .flatten_all()
        .map_err(|e| anyhow!("x.flatten failed: {e}"))?
        .to_vec1::<f32>()
        .map_err(|e| anyhow!("x.to_vec1 failed: {e}"))?;
    Ok((s, data))
}

fn tensor_to_f32_vec3_seq1(x: &Tensor, expected_hidden: usize) -> Result<Vec<f32>> {
    let (seq_len, data) = tensor_to_f32_vec3(x, expected_hidden)?;
    if seq_len != 1 {
        bail!("text rknn decode expects seq_len=1, got {seq_len}");
    }
    Ok(data)
}

fn build_rotary_window(
    position: usize,
    seq_len: usize,
    head_dim: usize,
    rope_theta: f64,
) -> (Vec<f32>, Vec<f32>) {
    let mut cos = Vec::with_capacity(seq_len * head_dim);
    let mut sin = Vec::with_capacity(seq_len * head_dim);
    for p in position..(position + seq_len) {
        let (row_cos, row_sin) = build_rotary(p, head_dim, rope_theta);
        cos.extend_from_slice(&row_cos);
        sin.extend_from_slice(&row_sin);
    }
    (cos, sin)
}

fn varying_dim_index(shapes: &[Vec<u32>]) -> Option<usize> {
    if shapes.is_empty() {
        return None;
    }
    let rank = shapes[0].len();
    let mut varying = None;
    for dim_idx in 0..rank {
        let base = shapes[0][dim_idx];
        if shapes.iter().any(|s| s.len() != rank || s[dim_idx] != base) {
            if varying.is_some() {
                return None;
            }
            varying = Some(dim_idx);
        }
    }
    varying
}

fn select_shape_for_bucket(shapes: &[Vec<u32>], bucket: usize) -> Option<Vec<u32>> {
    if shapes.is_empty() {
        return None;
    }
    if shapes.len() == 1 {
        return Some(shapes[0].clone());
    }
    let varying = varying_dim_index(shapes)?;
    let mut sorted = shapes.to_vec();
    sorted.sort_by_key(|s| s[varying]);
    for shape in &sorted {
        if shape[varying] as usize >= bucket {
            return Some(shape.clone());
        }
    }
    sorted.last().cloned()
}

const DEFAULT_DECODE_BUCKETS: &[usize] = &[16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChunkInputKind {
    Hidden,
    Cos,
    Sin,
    PastK,
    PastV,
    PastMask,
}

fn chunk_input_kind_from_counts(
    num_layers: usize,
    has_past_mask: bool,
    idx: usize,
) -> ChunkInputKind {
    let kv_input_end = 3 + 2 * num_layers;
    match idx {
        0 => ChunkInputKind::Hidden,
        1 => ChunkInputKind::Cos,
        2 => ChunkInputKind::Sin,
        i if has_past_mask && i == kv_input_end => ChunkInputKind::PastMask,
        i if i >= 3 && i < kv_input_end && (i - 3) % 2 == 0 => ChunkInputKind::PastK,
        i if i >= 3 && i < kv_input_end => ChunkInputKind::PastV,
        _ => ChunkInputKind::Hidden,
    }
}

fn chunk_input_kind(chunk: &ChunkModel, idx: usize) -> ChunkInputKind {
    chunk_input_kind_from_counts(chunk.num_layers(), chunk.has_past_mask, idx)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KvTensorLayout {
    HeadSeqDim,
    SeqDimHead,
}

fn infer_output_kv_layout<T: TensorAttrView>(
    attr: &T,
    heads: usize,
    head_dim: usize,
) -> KvTensorLayout {
    let dims = attr.dims();
    if dims.len() == 4 {
        let d1 = dims[1] as usize;
        let d2 = dims[2] as usize;
        let d3 = dims[3] as usize;
        if d2 == head_dim && d3 == heads {
            return KvTensorLayout::SeqDimHead;
        }
        if d1 == heads && d3 == head_dim {
            return KvTensorLayout::HeadSeqDim;
        }
    }
    match attr.format() {
        TensorFormatKind::NHWC(_) => KvTensorLayout::SeqDimHead,
        _ => KvTensorLayout::HeadSeqDim,
    }
}

fn bucket_candidates_from_env_or_default() -> Vec<usize> {
    let mut vals = if let Ok(spec) = std::env::var("DIAL_TEXT_DECODE_PAST_BUCKETS") {
        spec.split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    if vals.is_empty() {
        vals.extend_from_slice(DEFAULT_DECODE_BUCKETS);
    }
    vals.retain(|v| *v > 0);
    vals.sort_unstable();
    vals.dedup();
    vals
}

fn select_decode_bucket(chunk: &ChunkModel, run_past_len: usize) -> usize {
    let required_seq = run_past_len.max(1);
    let mut vals = if chunk.supported_buckets.is_empty() {
        bucket_candidates_from_env_or_default()
    } else {
        chunk.supported_buckets.clone()
    };
    if vals.is_empty() {
        for idx in 0..chunk.input_attrs.len() {
            let shapes = chunk
                .input_dynamic_shapes
                .get(idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if let Some(varying) = varying_dim_index(shapes) {
                vals.extend(shapes.iter().map(|shape| shape[varying] as usize));
            }
        }
        vals.retain(|v| *v > 0);
        vals.sort_unstable();
        vals.dedup();
    }
    if !vals.is_empty() {
        for v in &vals {
            if *v >= required_seq {
                return *v;
            }
        }
        return *vals.last().unwrap();
    }
    required_seq
}

fn seq_dim_index_for_kind(kind: ChunkInputKind, attr: &InputAttr, shape: &[u32]) -> Option<usize> {
    match kind {
        ChunkInputKind::PastK | ChunkInputKind::PastV if shape.len() == 4 => match attr.format() {
            TensorFormatKind::NHWC(_) => Some(1),
            _ => Some(2),
        },
        ChunkInputKind::PastMask if shape.len() == 4 => match attr.format() {
            TensorFormatKind::NHWC(_) => Some(2),
            _ => Some(3),
        },
        _ => None,
    }
}

fn preferred_input_shape(
    chunk: &ChunkModel,
    idx: usize,
    requested_bucket: usize,
    hidden_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Vec<u32> {
    let attr = &chunk.input_attrs[idx];
    let kind = chunk_input_kind(chunk, idx);
    let shapes = chunk
        .input_dynamic_shapes
        .get(idx)
        .map(Vec::as_slice)
        .unwrap_or(&[]);

    match kind {
        ChunkInputKind::Hidden => return vec![1, 1, hidden_size as u32],
        ChunkInputKind::Cos | ChunkInputKind::Sin => return attr.dims().to_vec(),
        ChunkInputKind::PastK | ChunkInputKind::PastV if attr.dims().len() == 4 => {
            if shapes.is_empty() {
                return attr.dims().to_vec();
            }
            return match attr.format() {
                TensorFormatKind::NHWC(_) => vec![
                    1,
                    requested_bucket as u32,
                    head_dim as u32,
                    num_kv_heads as u32,
                ],
                _ => vec![
                    1,
                    num_kv_heads as u32,
                    requested_bucket as u32,
                    head_dim as u32,
                ],
            };
        }
        ChunkInputKind::PastMask if attr.dims().len() == 4 => {
            if shapes.is_empty() {
                return attr.dims().to_vec();
            }
            return match attr.format() {
                TensorFormatKind::NHWC(_) => vec![1, 1, requested_bucket as u32, 1],
                _ => vec![1, 1, 1, requested_bucket as u32],
            };
        }
        _ => {}
    }

    if let Some(shape) = select_shape_for_bucket(shapes, requested_bucket) {
        return shape;
    }
    attr.dims().to_vec()
}

fn chunk_supports_bucket(chunk: &ChunkModel, bucket: usize) -> bool {
    for idx in 0..chunk.input_attrs.len() {
        let attr = &chunk.input_attrs[idx];
        let kind = chunk_input_kind(chunk, idx);
        match kind {
            ChunkInputKind::PastK | ChunkInputKind::PastV | ChunkInputKind::PastMask => {
                let shape = preferred_input_shape(chunk, idx, bucket, 4096, 8, 128);
                if let Some(seq_idx) = seq_dim_index_for_kind(kind, attr, &shape) {
                    if (shape.get(seq_idx).copied().unwrap_or(0) as usize) < bucket {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
}

fn input_bucket_from_attr(attr: &InputAttr, kind: ChunkInputKind) -> Option<usize> {
    let dims = attr.dims();
    if dims.len() != 4 {
        return None;
    }
    let seq_idx = seq_dim_index_for_kind(kind, attr, dims)?;
    let bucket = dims.get(seq_idx).copied().unwrap_or(0) as usize;
    if bucket > 0 {
        Some(bucket)
    } else {
        None
    }
}

fn infer_supported_buckets(
    num_layers: usize,
    input_attrs: &[InputAttr],
    input_dynamic_shapes: &[Vec<Vec<u32>>],
    has_past_mask: bool,
) -> Vec<usize> {
    let mut vals = Vec::new();
    for (idx, attr) in input_attrs.iter().enumerate() {
        let kind = chunk_input_kind_from_counts(num_layers, has_past_mask, idx);
        match kind {
            ChunkInputKind::PastK | ChunkInputKind::PastV | ChunkInputKind::PastMask => {
                if let Some(bucket) = input_bucket_from_attr(attr, kind) {
                    vals.push(bucket);
                }
                let shapes = input_dynamic_shapes
                    .get(idx)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                if let Some(varying) = varying_dim_index(shapes) {
                    vals.extend(shapes.iter().map(|shape| shape[varying] as usize));
                }
            }
            _ => {}
        }
    }
    vals.retain(|v| *v > 0);
    vals.sort_unstable();
    vals.dedup();
    vals
}

fn pad_kv_to_bucket(
    data: &[f32],
    heads: usize,
    run_past_len: usize,
    head_dim: usize,
    bucket: usize,
) -> Vec<f32> {
    if bucket <= run_past_len {
        return data.to_vec();
    }
    let mut out = vec![0f32; heads * bucket * head_dim];
    let src_stride = run_past_len * head_dim;
    let dst_stride = bucket * head_dim;
    for h in 0..heads {
        let src_off = h * src_stride;
        let dst_off = h * dst_stride;
        out[dst_off..dst_off + src_stride].copy_from_slice(&data[src_off..src_off + src_stride]);
    }
    out
}

fn kv_cache_to_rknn_layout(
    data: &[f32],
    heads: usize,
    seq: usize,
    head_dim: usize,
    format: TensorFormatKind,
) -> Vec<f32> {
    match format {
        TensorFormatKind::NHWC(_) => {
            let mut out = vec![0f32; data.len()];
            for h in 0..heads {
                for s in 0..seq {
                    for d in 0..head_dim {
                        let src = h * seq * head_dim + s * head_dim + d;
                        let dst = s * head_dim * heads + d * heads + h;
                        out[dst] = data[src];
                    }
                }
            }
            out
        }
        _ => data.to_vec(),
    }
}

fn kv_rknn_to_cache_layout(
    data: &[f32],
    heads: usize,
    seq: usize,
    head_dim: usize,
    layout: KvTensorLayout,
) -> Vec<f32> {
    match layout {
        KvTensorLayout::SeqDimHead => {
            let mut out = vec![0f32; data.len()];
            for s in 0..seq {
                for d in 0..head_dim {
                    for h in 0..heads {
                        let src = s * head_dim * heads + d * heads + h;
                        let dst = h * seq * head_dim + s * head_dim + d;
                        out[dst] = data[src];
                    }
                }
            }
            out
        }
        KvTensorLayout::HeadSeqDim => data.to_vec(),
    }
}

fn truncate_cache_layout_seq(
    data: &[f32],
    heads: usize,
    seq: usize,
    present_len: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if present_len > seq {
        bail!(
            "truncate_cache_layout_seq: present_len {} > seq {}",
            present_len,
            seq
        );
    }
    let per_head = seq
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow!("truncate_cache_layout_seq: seq/head overflow"))?;
    let keep_per_head = present_len
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow!("truncate_cache_layout_seq: present/head overflow"))?;
    let expected_total = heads
        .checked_mul(per_head)
        .ok_or_else(|| anyhow!("truncate_cache_layout_seq: total overflow"))?;
    if data.len() < expected_total {
        bail!(
            "truncate_cache_layout_seq: data too short {} < {}",
            data.len(),
            expected_total
        );
    }
    let mut out = Vec::with_capacity(heads * keep_per_head);
    for head in 0..heads {
        let start = head * per_head;
        out.extend_from_slice(&data[start..start + keep_per_head]);
    }
    Ok(out)
}

fn effective_bucket_from_selected_shapes(
    chunk: &ChunkModel,
    selected_shapes: &[Vec<u32>],
) -> Option<usize> {
    let mut vals = Vec::new();

    for idx in 0..selected_shapes.len().min(chunk.input_attrs.len()) {
        let kind = chunk_input_kind(chunk, idx);
        let shape = &selected_shapes[idx];
        if let Some(seq_idx) = seq_dim_index_for_kind(kind, &chunk.input_attrs[idx], shape) {
            if let Some(v) = shape.get(seq_idx) {
                vals.push(*v as usize);
            }
        }
    }
    vals.into_iter().max()
}

fn dtype_num_bytes(dtype: &DataTypeKind) -> Option<usize> {
    match dtype {
        DataTypeKind::Float32(_) => Some(std::mem::size_of::<f32>()),
        DataTypeKind::Float16(_) => Some(std::mem::size_of::<f16>()),
        DataTypeKind::BFloat16(_) => Some(std::mem::size_of::<bf16>()),
        DataTypeKind::Int64(_) => Some(std::mem::size_of::<i64>()),
        DataTypeKind::Int32(_) => Some(std::mem::size_of::<i32>()),
        DataTypeKind::UInt32(_) => Some(std::mem::size_of::<u32>()),
        DataTypeKind::UInt16(_) => Some(std::mem::size_of::<u16>()),
        DataTypeKind::UInt8(_) => Some(std::mem::size_of::<u8>()),
        DataTypeKind::Int8(_) => Some(std::mem::size_of::<i8>()),
        _ => None,
    }
}

fn expected_bytes_for_shape(shape: &[u32], dtype: &DataTypeKind) -> Option<usize> {
    let elem_bytes = dtype_num_bytes(dtype)?;
    let num_elems = shape
        .iter()
        .try_fold(1usize, |acc, d| acc.checked_mul(*d as usize))?;
    num_elems.checked_mul(elem_bytes)
}

fn tensor_attr_num_elements<T: TensorAttrView>(attr: &T) -> Result<usize> {
    let n = attr.num_elements() as usize;
    if n > 0 {
        return Ok(n);
    }
    let dims = attr.dims();
    if dims.is_empty() {
        bail!("tensor {} has no dims and n_elems=0", attr.name());
    }
    dims.iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim as usize))
        .ok_or_else(|| {
            anyhow!(
                "tensor {} num_elements overflow for dims {:?}",
                attr.name(),
                dims
            )
        })
}

enum ResolvedOutputAttr<'a> {
    Static(&'a OutputAttr),
    Dynamic(CurrentOutputAttr),
}

impl TensorAttrView for ResolvedOutputAttr<'_> {
    fn io(&self) -> rknpu2::query::Io {
        match self {
            Self::Static(attr) => attr.io(),
            Self::Dynamic(attr) => attr.io(),
        }
    }

    fn index(&self) -> u32 {
        match self {
            Self::Static(attr) => attr.index(),
            Self::Dynamic(attr) => attr.index(),
        }
    }

    fn name(&self) -> String {
        match self {
            Self::Static(attr) => attr.name(),
            Self::Dynamic(attr) => attr.name(),
        }
    }

    fn dtype(&self) -> DataTypeKind {
        match self {
            Self::Static(attr) => attr.dtype(),
            Self::Dynamic(attr) => attr.dtype(),
        }
    }

    fn num_dims(&self) -> u32 {
        match self {
            Self::Static(attr) => attr.num_dims(),
            Self::Dynamic(attr) => attr.num_dims(),
        }
    }

    fn dims(&self) -> &[u32] {
        match self {
            Self::Static(attr) => attr.dims(),
            Self::Dynamic(attr) => attr.dims(),
        }
    }

    fn format(&self) -> TensorFormatKind {
        match self {
            Self::Static(attr) => attr.format(),
            Self::Dynamic(attr) => attr.format(),
        }
    }

    fn qnt_type(&self) -> rknpu2::tensor::QuantTypeKind {
        match self {
            Self::Static(attr) => attr.qnt_type(),
            Self::Dynamic(attr) => attr.qnt_type(),
        }
    }

    fn num_elements(&self) -> u32 {
        match self {
            Self::Static(attr) => attr.num_elements(),
            Self::Dynamic(attr) => attr.num_elements(),
        }
    }

    fn scale(&self) -> f32 {
        match self {
            Self::Static(attr) => attr.scale(),
            Self::Dynamic(attr) => attr.scale(),
        }
    }

    fn zero_point(&self) -> i32 {
        match self {
            Self::Static(attr) => attr.zero_point(),
            Self::Dynamic(attr) => attr.zero_point(),
        }
    }

    fn fl(&self) -> i8 {
        match self {
            Self::Static(attr) => attr.fl(),
            Self::Dynamic(attr) => attr.fl(),
        }
    }

    fn w_stride(&self) -> u32 {
        match self {
            Self::Static(attr) => attr.w_stride(),
            Self::Dynamic(attr) => attr.w_stride(),
        }
    }

    fn h_stride(&self) -> u32 {
        match self {
            Self::Static(attr) => attr.h_stride(),
            Self::Dynamic(attr) => attr.h_stride(),
        }
    }

    fn size(&self) -> u32 {
        match self {
            Self::Static(attr) => attr.size(),
            Self::Dynamic(attr) => attr.size(),
        }
    }

    fn size_with_stride(&self) -> u32 {
        match self {
            Self::Static(attr) => attr.size_with_stride(),
            Self::Dynamic(attr) => attr.size_with_stride(),
        }
    }
}

fn query_current_output_attrs(chunk: &ChunkModel) -> Result<Vec<ResolvedOutputAttr<'_>>> {
    if !chunk.is_dynamic_model {
        return Ok(chunk
            .output_attrs
            .iter()
            .map(ResolvedOutputAttr::Static)
            .collect());
    }
    let mut attrs = Vec::with_capacity(chunk.output_attrs.len());
    for i in 0..chunk.output_attrs.len() {
        attrs.push(
            chunk
                .rknn
                .query_with_input::<CurrentOutputAttr>(i as u32)
                .map_err(|e| {
                    anyhow!(
                        "text chunk {}-{} query current output attr[{}] failed: {e}",
                        chunk.layer_start,
                        chunk.layer_end,
                        i
                    )
                })
                .map(ResolvedOutputAttr::Dynamic)?,
        );
    }
    Ok(attrs)
}

fn kv_seq_len_from_output_attr<T: TensorAttrView>(
    attr: &T,
    data_len: usize,
    heads: usize,
    head_dim: usize,
) -> Result<usize> {
    let per_seq = heads.checked_mul(head_dim).ok_or_else(|| {
        anyhow!(
            "kv seq size overflow: heads={} head_dim={}",
            heads,
            head_dim
        )
    })?;
    if per_seq == 0 || data_len % per_seq != 0 {
        bail!(
            "tensor {} invalid kv output len {} for heads={} head_dim={}",
            attr.name(),
            data_len,
            heads,
            head_dim
        );
    }
    if attr.dims().len() == 4 {
        let seq_idx = match infer_output_kv_layout(attr, heads, head_dim) {
            KvTensorLayout::SeqDimHead => 1,
            KvTensorLayout::HeadSeqDim => 2,
        };
        if let Some(seq) = attr.dims().get(seq_idx).copied() {
            if seq > 0 {
                return Ok(seq as usize);
            }
        }
    }
    Ok(data_len / per_seq)
}

/// Run Qwen3-VL text decode via RKNN chunk models (e.g. 2 layers/chunk).
pub struct TextRknnRunner {
    chunks: Vec<ChunkModel>,
    chunk_last_selected_shapes: Vec<Option<Vec<Vec<u32>>>>,
    prefill_chunks: Option<Vec<ChunkModel>>,
    prefill_supports_deepstack: bool,
    hidden_size: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    num_hidden_layers: usize,
    decode_last_layer: usize,
    rope_theta: f64,
    layer_k_storage_dtypes: Vec<KvStorageDType>,
    layer_v_storage_dtypes: Vec<KvStorageDType>,
    kv: Vec<Option<LayerKv>>,
    last_delta_layout_debug: Vec<Option<LayerKvDeltaLayoutDebug>>,
    synced_from_cache: bool,
    max_supported_bucket: usize,
}

impl TextRknnRunner {
    fn load_prefill_chunks(
        parsed: Vec<(usize, usize, PathBuf)>,
        expected_layers: usize,
        lib_path: &Path,
    ) -> Result<Vec<ChunkModel>> {
        let mut grouped: BTreeMap<(usize, usize), Vec<PathBuf>> = BTreeMap::new();
        for (s, e, path) in parsed {
            grouped.entry((s, e)).or_default().push(path);
        }
        for paths in grouped.values_mut() {
            paths.sort_by_key(|path| chunk_candidate_sort_key(path));
        }

        let coverage = validate_chunk_coverage(
            grouped
                .iter()
                .map(|((s, e), paths)| {
                    (
                        *s,
                        *e,
                        paths.first().cloned().unwrap_or_else(|| PathBuf::from("")),
                    )
                })
                .collect(),
            expected_layers,
            "prefill",
        )?;

        let mut loaded = Vec::with_capacity(coverage.len());
        for (s, e, _) in coverage {
            let paths = grouped
                .remove(&(s, e))
                .ok_or_else(|| anyhow!("prefill chunk candidate list missing for {}-{}", s, e))?;
            let mut last_err = None;
            let mut picked = None;
            for path in paths {
                match ChunkModel::load(&path, s, e, lib_path, ChunkStage::Prefill) {
                    Ok(chunk) => {
                        log::info!(
                            "text rknn prefill chunk loaded: {} (layers {}-{})",
                            path.display(),
                            s,
                            e
                        );
                        picked = Some(chunk);
                        break;
                    }
                    Err(err) => {
                        log::warn!(
                            "text rknn prefill chunk candidate load failed for {} (layers {}-{}): {}",
                            path.display(),
                            s,
                            e,
                            err
                        );
                        last_err = Some(err);
                    }
                }
            }
            let Some(chunk) = picked else {
                if let Some(err) = last_err {
                    return Err(err);
                }
                bail!(
                    "no loadable prefill chunk candidates for layers {}-{}",
                    s,
                    e
                );
            };
            loaded.push(chunk);
        }
        Ok(loaded)
    }

    pub fn load(model_dir: &Path, lib_path: Option<&Path>, cfg: &Config) -> Result<Self> {
        if cfg.hidden_size == 0 || cfg.num_hidden_layers == 0 || cfg.num_key_value_heads == 0 {
            bail!("invalid text config for text rknn runner");
        }
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let lib_path = match lib_path {
            Some(path) => path.to_path_buf(),
            None => find_rknn_library()
                .next()
                .ok_or_else(|| anyhow!("cannot find librknnrt.so (set --text-rknn-lib)"))?,
        };

        let mut files: Vec<PathBuf> = fs::read_dir(model_dir)
            .map_err(|e| anyhow!("read_dir {} failed: {e}", model_dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("rknn"))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no .rknn files found in {}", model_dir.display());
        }

        let re = Regex::new(r"l(\d+)_l(\d+)").map_err(|e| anyhow!("regex build failed: {e}"))?;
        let mut decode_tagged = Vec::new();
        let mut prefill_tagged = Vec::new();
        let mut untagged = Vec::new();
        for path in files {
            if let Some((s, e)) = parse_chunk_path(&path, &re) {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.to_ascii_lowercase())
                    .unwrap_or_default();
                if name.contains("prefill") {
                    prefill_tagged.push((s, e, path));
                } else if name.contains("decode") {
                    decode_tagged.push((s, e, path));
                } else {
                    untagged.push((s, e, path));
                }
            }
        }
        let decode_candidates = if !decode_tagged.is_empty() {
            decode_tagged
        } else {
            untagged
        };
        if decode_candidates.is_empty() {
            bail!(
                "no decode chunk filename matched pattern lXX_lYY under {}",
                model_dir.display()
            );
        }
        let mut grouped: BTreeMap<(usize, usize), Vec<PathBuf>> = BTreeMap::new();
        for (s, e, path) in decode_candidates {
            grouped.entry((s, e)).or_default().push(path);
        }
        for paths in grouped.values_mut() {
            paths.sort_by_key(|path| chunk_candidate_sort_key(path));
        }

        let mut chunks = Vec::with_capacity(grouped.len());
        let mut loaded_last_layer = None;
        let mut expected = 0usize;
        while expected < cfg.num_hidden_layers {
            let Some((range, paths)) =
                grouped
                    .iter()
                    .filter(|((s, e), _)| *s == expected && e >= s)
                    .min_by_key(|((s, e), paths)| {
                        let first = paths.first();
                        let file_key = first
                            .map(|path| chunk_candidate_sort_key(path))
                            .unwrap_or((3, std::cmp::Reverse(0), String::new()));
                        (file_key.0, file_key.1, e.saturating_sub(*s), file_key.2)
                    })
                    .map(|(range, paths)| (*range, paths.clone()))
            else {
                if chunks.is_empty() {
                    bail!(
                        "chunk coverage gap: expected layer {} but no text rknn chunk starts there under {}",
                        expected,
                        model_dir.display()
                    );
                }
                log::info!(
                    "text rknn chunk coverage stops at layer {}; no chunk starts at layer {}; remaining layers fallback to software/remote path",
                    loaded_last_layer.unwrap_or(0),
                    expected
                );
                break;
            };
            grouped.remove(&range);
            let (s, e) = range;
            let mut last_err = None;
            let mut loaded = None;
            for path in paths {
                match ChunkModel::load(&path, s, e, &lib_path, ChunkStage::Decode) {
                    Ok(chunk) => {
                        let max_bucket = chunk.supported_buckets.iter().copied().max().unwrap_or(0);
                        log::info!(
                            "text rknn decode chunk loaded: {} (layers {}-{}, max_bucket={})",
                            path.display(),
                            s,
                            e,
                            max_bucket
                        );
                        for local_idx in 0..chunk.num_layers() {
                            let layer = s + local_idx;
                            let k_out_idx = 1 + 2 * local_idx;
                            let v_out_idx = 1 + 2 * local_idx + 1;
                            log::info!(
                                "text rknn chunk {}-{} layer {} output formats: hidden={:?} present_k={:?} present_v={:?}",
                                s,
                                e,
                                layer,
                                chunk.output_attrs[0].format(),
                                chunk.output_attrs[k_out_idx].format(),
                                chunk.output_attrs[v_out_idx].format(),
                            );
                        }
                        loaded = Some(chunk);
                        break;
                    }
                    Err(err) => {
                        log::warn!(
                            "text rknn chunk candidate load failed for {} (layers {}-{}): {}",
                            path.display(),
                            s,
                            e,
                            err
                        );
                        last_err = Some(err);
                    }
                }
            }
            let Some(chunk) = loaded else {
                if chunks.is_empty() {
                    if let Some(err) = last_err {
                        return Err(err);
                    }
                    bail!(
                        "no loadable text rknn chunk candidates for layers {}-{}",
                        s,
                        e
                    );
                }
                log::warn!(
                    "text rknn chunk load failed for all candidates (layers {}-{}); shrink decode coverage to 0..{} and fallback remaining layers to software/remote path",
                    s,
                    e,
                    loaded_last_layer.unwrap_or(0)
                );
                break;
            };
            loaded_last_layer = Some(e);
            expected = e + 1;
            chunks.push(chunk);
        }
        if chunks.is_empty() {
            bail!(
                "no text rknn chunks could be loaded from {}",
                model_dir.display()
            );
        }
        let decode_last_layer = loaded_last_layer.unwrap_or(0);
        let expected = decode_last_layer + 1;
        if expected < cfg.num_hidden_layers {
            log::info!(
                "text rknn prefix coverage only spans layers 0..{}; remaining layers {}..{} will fallback to software/remote path",
                decode_last_layer,
                expected,
                cfg.num_hidden_layers.saturating_sub(1)
            );
        }

        let mut layer_k_storage_dtypes = vec![None; expected];
        let mut layer_v_storage_dtypes = vec![None; expected];
        let mut max_supported_bucket = 0usize;
        for chunk in &chunks {
            let chunk_max_bucket = if !chunk.supported_buckets.is_empty() {
                chunk.supported_buckets.iter().copied().max().unwrap_or(0)
            } else {
                bucket_candidates_from_env_or_default()
                    .into_iter()
                    .filter(|bucket| chunk_supports_bucket(chunk, *bucket))
                    .max()
                    .unwrap_or(0)
            };
            max_supported_bucket = max_supported_bucket.max(chunk_max_bucket);
            for local_idx in 0..chunk.num_layers() {
                let layer = chunk.layer_start + local_idx;
                let k_attr_idx = 3 + 2 * local_idx;
                let v_attr_idx = 3 + 2 * local_idx + 1;
                let k_dtype =
                    KvStorageDType::from_rknn_dtype(&chunk.input_attrs[k_attr_idx].dtype())?;
                let v_dtype =
                    KvStorageDType::from_rknn_dtype(&chunk.input_attrs[v_attr_idx].dtype())?;
                layer_k_storage_dtypes[layer] = Some(k_dtype);
                layer_v_storage_dtypes[layer] = Some(v_dtype);
            }
        }
        let layer_k_storage_dtypes: Vec<KvStorageDType> = layer_k_storage_dtypes
            .into_iter()
            .enumerate()
            .map(|(layer, v)| v.ok_or_else(|| anyhow!("missing k dtype mapping for layer {layer}")))
            .collect::<Result<Vec<_>>>()?;
        let layer_v_storage_dtypes: Vec<KvStorageDType> = layer_v_storage_dtypes
            .into_iter()
            .enumerate()
            .map(|(layer, v)| v.ok_or_else(|| anyhow!("missing v dtype mapping for layer {layer}")))
            .collect::<Result<Vec<_>>>()?;
        let chunk_count = chunks.len();
        let prefill_chunks = if prefill_tagged.is_empty() {
            log::info!("text rknn prefill chunks not found; prefill will fallback to local path");
            None
        } else {
            match Self::load_prefill_chunks(prefill_tagged, cfg.num_hidden_layers, &lib_path) {
                Ok(loaded) => Some(loaded),
                Err(err) => {
                    log::warn!(
                        "text rknn prefill chunks unavailable; fallback to local prefill path: {}",
                        err
                    );
                    None
                }
            }
        };
        let prefill_supports_deepstack = prefill_chunks
            .as_ref()
            .map(|chunks| chunks.iter().all(|chunk| chunk.prefill_expects_layer_add()))
            .unwrap_or(false);

        Ok(Self {
            chunks,
            chunk_last_selected_shapes: vec![None; chunk_count],
            prefill_chunks,
            prefill_supports_deepstack,
            hidden_size: cfg.hidden_size,
            head_dim,
            num_key_value_heads: cfg.num_key_value_heads,
            num_hidden_layers: cfg.num_hidden_layers,
            decode_last_layer,
            rope_theta: cfg.rope_theta as f64,
            layer_k_storage_dtypes,
            layer_v_storage_dtypes,
            kv: vec![None; cfg.num_hidden_layers],
            last_delta_layout_debug: vec![None; cfg.num_hidden_layers],
            synced_from_cache: false,
            max_supported_bucket,
        })
    }

    pub fn clear(&mut self) {
        self.kv = vec![None; self.num_hidden_layers];
        self.last_delta_layout_debug = vec![None; self.num_hidden_layers];
        self.synced_from_cache = false;
        for item in &mut self.chunk_last_selected_shapes {
            *item = None;
        }
    }

    pub fn has_prefill(&self) -> bool {
        self.prefill_chunks.is_some()
    }

    pub fn decode_last_layer(&self) -> usize {
        self.decode_last_layer
    }

    pub fn decode_covers_all_layers(&self) -> bool {
        self.decode_last_layer + 1 == self.num_hidden_layers
    }

    pub fn max_supported_bucket(&self) -> usize {
        self.max_supported_bucket
    }

    pub fn prefill_supports_deepstack(&self) -> bool {
        self.prefill_supports_deepstack
    }

    pub fn layer_kv_cache_f32(&self, layer_idx: usize) -> Option<(&[f32], &[f32], usize)> {
        self.kv.get(layer_idx).and_then(|kv| {
            kv.as_ref()
                .map(|kv| (kv.k.as_slice(), kv.v.as_slice(), kv.past_len))
        })
    }

    pub fn layer_delta_layout_debug(&self, layer_idx: usize) -> Option<&LayerKvDeltaLayoutDebug> {
        self.last_delta_layout_debug
            .get(layer_idx)
            .and_then(|debug| debug.as_ref())
    }

    pub fn can_shrink_decode_coverage(&self) -> bool {
        self.chunks.len() > 1
    }

    pub fn shrink_last_chunk(&mut self) -> Result<(usize, usize)> {
        let removed = self
            .chunks
            .pop()
            .ok_or_else(|| anyhow!("cannot shrink text rknn coverage: no chunks loaded"))?;
        let prev_last = removed.layer_end;
        let new_last = self
            .chunks
            .last()
            .map(|chunk| chunk.layer_end)
            .ok_or_else(|| anyhow!("cannot shrink text rknn coverage below layer 0"))?;

        self.decode_last_layer = new_last;
        self.chunk_last_selected_shapes.pop();
        let keep = new_last + 1;
        self.layer_k_storage_dtypes.truncate(keep);
        self.layer_v_storage_dtypes.truncate(keep);
        for layer_kv in self.kv.iter_mut().skip(keep) {
            *layer_kv = None;
        }
        for debug in self.last_delta_layout_debug.iter_mut().skip(keep) {
            *debug = None;
        }
        self.synced_from_cache = false;
        self.max_supported_bucket = self
            .chunks
            .iter()
            .map(|chunk| {
                if !chunk.supported_buckets.is_empty() {
                    chunk.supported_buckets.iter().copied().max().unwrap_or(0)
                } else {
                    bucket_candidates_from_env_or_default()
                        .into_iter()
                        .filter(|bucket| chunk_supports_bucket(chunk, *bucket))
                        .max()
                        .unwrap_or(0)
                }
            })
            .max()
            .unwrap_or(0);

        Ok((prev_last, new_last))
    }

    fn sync_from_cache(&mut self, cache: &Cache) -> Result<()> {
        for layer_idx in 0..=self.decode_last_layer {
            let (k, v) = cache.kv_clone(layer_idx).ok_or_else(|| {
                anyhow!("cache missing kv for layer {layer_idx}; run prefill first")
            })?;

            let k = k
                .to_device(&Device::Cpu)
                .map_err(|e| anyhow!("layer {layer_idx} k.to_cpu failed: {e}"))?
                .to_dtype(DType::F32)
                .map_err(|e| anyhow!("layer {layer_idx} k.to_f32 failed: {e}"))?
                .contiguous()
                .map_err(|e| anyhow!("layer {layer_idx} k.contiguous failed: {e}"))?;
            let v = v
                .to_device(&Device::Cpu)
                .map_err(|e| anyhow!("layer {layer_idx} v.to_cpu failed: {e}"))?
                .to_dtype(DType::F32)
                .map_err(|e| anyhow!("layer {layer_idx} v.to_f32 failed: {e}"))?
                .contiguous()
                .map_err(|e| anyhow!("layer {layer_idx} v.contiguous failed: {e}"))?;

            let (b, h, seq, d) = k
                .dims4()
                .map_err(|e| anyhow!("layer {layer_idx} k.dims4 failed: {e}"))?;
            if b != 1 || h != self.num_key_value_heads || d != self.head_dim {
                bail!(
                    "layer {layer_idx} k shape mismatch, expected (1,{},{},{}), got ({},{},{},{})",
                    self.num_key_value_heads,
                    "past_len",
                    self.head_dim,
                    b,
                    h,
                    seq,
                    d
                );
            }
            let (vb, vh, vseq, vd) = v
                .dims4()
                .map_err(|e| anyhow!("layer {layer_idx} v.dims4 failed: {e}"))?;
            if (vb, vh, vseq, vd) != (b, h, seq, d) {
                bail!(
                    "layer {layer_idx} kv shape mismatch, k=({},{},{},{}) v=({},{},{},{})",
                    b,
                    h,
                    seq,
                    d,
                    vb,
                    vh,
                    vseq,
                    vd
                );
            }

            let k_vec = k
                .flatten_all()
                .map_err(|e| anyhow!("layer {layer_idx} k.flatten failed: {e}"))?
                .to_vec1::<f32>()
                .map_err(|e| anyhow!("layer {layer_idx} k.to_vec failed: {e}"))?;
            let v_vec = v
                .flatten_all()
                .map_err(|e| anyhow!("layer {layer_idx} v.flatten failed: {e}"))?
                .to_vec1::<f32>()
                .map_err(|e| anyhow!("layer {layer_idx} v.to_vec failed: {e}"))?;

            self.kv[layer_idx] = Some(LayerKv {
                past_len: seq,
                k: k_vec,
                v: v_vec,
                native: None,
            });
        }
        self.synced_from_cache = true;
        Ok(())
    }

    pub fn sync_to_cache(&self, cache: &mut Cache, device: &Device, dtype: DType) -> Result<()> {
        for (layer_idx, layer_kv) in self.kv.iter().enumerate() {
            let Some(layer_kv) = layer_kv.as_ref() else {
                continue;
            };

            let k = Tensor::from_vec(
                layer_kv.k.clone(),
                Shape::from_dims(&[
                    1,
                    self.num_key_value_heads,
                    layer_kv.past_len,
                    self.head_dim,
                ]),
                &Device::Cpu,
            )
            .map_err(|e| anyhow!("layer {layer_idx} cache sync k build failed: {e}"))?
            .to_dtype(dtype)
            .map_err(|e| anyhow!("layer {layer_idx} cache sync k to_dtype failed: {e}"))?
            .to_device(device)
            .map_err(|e| anyhow!("layer {layer_idx} cache sync k to_device failed: {e}"))?;

            let v = Tensor::from_vec(
                layer_kv.v.clone(),
                Shape::from_dims(&[
                    1,
                    self.num_key_value_heads,
                    layer_kv.past_len,
                    self.head_dim,
                ]),
                &Device::Cpu,
            )
            .map_err(|e| anyhow!("layer {layer_idx} cache sync v build failed: {e}"))?
            .to_dtype(dtype)
            .map_err(|e| anyhow!("layer {layer_idx} cache sync v to_dtype failed: {e}"))?
            .to_device(device)
            .map_err(|e| anyhow!("layer {layer_idx} cache sync v to_device failed: {e}"))?;

            cache
                .set_kv(layer_idx, k, v)
                .map_err(|e| anyhow!("layer {layer_idx} cache sync set_kv failed: {e}"))?;
        }
        Ok(())
    }

    fn run_chunk(
        chunk: &ChunkModel,
        last_selected_shapes: &mut Option<Vec<Vec<u32>>>,
        hidden: &[f32],
        cos: &[f32],
        sin: &[f32],
        kv: &mut [Option<LayerKv>],
        delta_layout_debug: &mut [Option<LayerKvDeltaLayoutDebug>],
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let run_past_len = kv[chunk.layer_start]
            .as_ref()
            .ok_or_else(|| {
                anyhow!(
                    "missing kv for layer {} before chunk run",
                    chunk.layer_start
                )
            })?
            .past_len;
        let requested_bucket_past_len = select_decode_bucket(chunk, run_past_len.max(1));
        let mut selected_shapes = Vec::with_capacity(chunk.input_attrs.len());
        for idx in 0..chunk.input_attrs.len() {
            selected_shapes.push(preferred_input_shape(
                chunk,
                idx,
                requested_bucket_past_len,
                hidden.len(),
                num_kv_heads,
                head_dim,
            ));
        }
        let bucket_past_len = effective_bucket_from_selected_shapes(chunk, &selected_shapes)
            .unwrap_or(requested_bucket_past_len);
        if bucket_past_len < run_past_len {
            bail!(
                "text chunk {}-{} selected bucket {} is smaller than past_len {}",
                chunk.layer_start,
                chunk.layer_end,
                bucket_past_len,
                run_past_len
            );
        }
        if !selected_shapes.is_empty() {
            log::debug!(
                "text rknn chunk {}-{} decode start: past_len={} bucket={} requested_bucket={}",
                chunk.layer_start,
                chunk.layer_end,
                run_past_len,
                bucket_past_len,
                requested_bucket_past_len
            );
            if chunk.has_dynamic_inputs() {
                let need_set_shapes = last_selected_shapes
                    .as_ref()
                    .map(|prev| prev != &selected_shapes)
                    .unwrap_or(true);
                if need_set_shapes {
                    chunk
                        .rknn
                        .set_input_shapes(&chunk.input_attrs, &selected_shapes)
                        .map_err(|e| {
                            anyhow!(
                                "text chunk {}-{} set_input_shapes failed: {e}",
                                chunk.layer_start,
                                chunk.layer_end
                            )
                        })?;
                    *last_selected_shapes = Some(selected_shapes.clone());
                }
            }
        }
        let mut input_buffers = Vec::with_capacity(chunk.input_attrs.len());
        input_buffers.push(OwnedInputBuffer::from_f32(
            hidden,
            chunk.input_attrs[0].dtype(),
        )?);
        input_buffers.push(OwnedInputBuffer::from_f32(
            cos,
            chunk.input_attrs[1].dtype(),
        )?);
        input_buffers.push(OwnedInputBuffer::from_f32(
            sin,
            chunk.input_attrs[2].dtype(),
        )?);

        for layer in chunk.layer_start..=chunk.layer_end {
            let layer_input_offset = 3 + 2 * (layer - chunk.layer_start);
            let k_input_idx = layer_input_offset;
            let v_input_idx = layer_input_offset + 1;
            let k_format = chunk.input_attrs[k_input_idx].format();
            let v_format = chunk.input_attrs[v_input_idx].format();
            let k_dtype = KvStorageDType::from_rknn_dtype(&chunk.input_attrs[k_input_idx].dtype())?;
            let v_dtype = KvStorageDType::from_rknn_dtype(&chunk.input_attrs[v_input_idx].dtype())?;
            let layer_kv = kv[layer]
                .as_mut()
                .ok_or_else(|| anyhow!("missing kv for layer {} before chunk run", layer))?;
            layer_kv.ensure_native_buffers(
                bucket_past_len,
                k_format,
                v_format,
                k_dtype,
                v_dtype,
                num_kv_heads,
                head_dim,
            );
        }

        for layer in chunk.layer_start..=chunk.layer_end {
            let layer_kv = kv[layer]
                .as_ref()
                .ok_or_else(|| anyhow!("missing kv for layer {} before chunk run", layer))?;
            let native = layer_kv
                .native
                .as_ref()
                .ok_or_else(|| anyhow!("layer {} native kv buffers missing after ensure", layer))?;
            input_buffers.push(OwnedInputBuffer::from_kv_data(&native.k));
            input_buffers.push(OwnedInputBuffer::from_kv_data(&native.v));
        }
        if chunk.has_past_mask && bucket_past_len > 0 {
            let mut past_mask = vec![0.0f32; bucket_past_len];
            for v in past_mask.iter_mut().take(run_past_len) {
                *v = 1.0;
            }
            input_buffers.push(
                match chunk
                    .input_attrs
                    .last()
                    .ok_or_else(|| {
                        anyhow!(
                            "text chunk {}-{} past_mask attr missing",
                            chunk.layer_start,
                            chunk.layer_end
                        )
                    })?
                    .dtype()
                {
                    DataTypeKind::Float32(_) => OwnedInputBuffer::OwnF32(past_mask),
                    DataTypeKind::Float16(_) => {
                        OwnedInputBuffer::OwnF16(past_mask.into_iter().map(f16::from_f32).collect())
                    }
                    DataTypeKind::BFloat16(_) => OwnedInputBuffer::OwnBF16(
                        past_mask.into_iter().map(bf16::from_f32).collect(),
                    ),
                    other => bail!("unsupported past_mask dtype: {other:?}"),
                },
            );
        }

        let mut inputs = Vec::with_capacity(input_buffers.len());
        for (idx, buf) in input_buffers.iter().enumerate() {
            if let Some(expected_bytes) =
                expected_bytes_for_shape(&selected_shapes[idx], &chunk.input_attrs[idx].dtype())
            {
                let actual_bytes = buf.as_buf_view().num_bytes();
                if actual_bytes < expected_bytes {
                    bail!(
                        "text chunk {}-{} input[{}] {} undersized: actual_bytes={} expected_bytes={} shape={:?} dtype={:?}",
                        chunk.layer_start,
                        chunk.layer_end,
                        idx,
                        chunk.input_attrs[idx].name(),
                        actual_bytes,
                        expected_bytes,
                        selected_shapes[idx],
                        chunk.input_attrs[idx].dtype()
                    );
                }
            }
            inputs.push(Input::new(
                idx as u32,
                buf.as_buf_view(),
                true,
                chunk.input_attrs[idx].format(),
            ));
        }

        chunk.rknn.set_inputs(inputs).map_err(|e| {
            anyhow!(
                "text chunk {}-{} set_inputs failed: {e}",
                chunk.layer_start,
                chunk.layer_end
            )
        })?;
        chunk.rknn.run().map_err(|e| {
            anyhow!(
                "text chunk {}-{} run failed: {e}",
                chunk.layer_start,
                chunk.layer_end
            )
        })?;
        // Release immutable borrows of `kv` before mutating kv cache with present outputs.
        drop(input_buffers);

        let current_output_attrs = query_current_output_attrs(chunk)?;

        let mut output_bufs: Vec<Vec<f32>> = current_output_attrs
            .iter()
            .enumerate()
            .map(|(i, attr)| {
                let n = tensor_attr_num_elements(attr)?;
                if n == 0 {
                    bail!(
                        "text chunk {}-{} output[{}] has n_elems=0 (dynamic outputs must expose n_elems via RKNN metadata)",
                        chunk.layer_start,
                        chunk.layer_end,
                        i
                    );
                }
                Ok(vec![0f32; n])
            })
            .collect::<Result<Vec<_>>>()?;

        let mut outputs = Vec::with_capacity(output_bufs.len());
        for (i, out) in output_bufs.iter_mut().enumerate() {
            outputs.push(Output {
                index: i as u32,
                kind: OutputKind::Preallocated {
                    buf: BufMutView::F32(out),
                    want_float: true,
                },
            });
        }

        chunk.rknn.get_outputs(&mut outputs).map_err(|e| {
            anyhow!(
                "text chunk {}-{} get_outputs failed: {e}",
                chunk.layer_start,
                chunk.layer_end
            )
        })?;

        let hidden_out = output_bufs.first().ok_or_else(|| {
            anyhow!(
                "text chunk {}-{} returned no outputs",
                chunk.layer_start,
                chunk.layer_end
            )
        })?;
        if let Some(hidden_attr) = current_output_attrs.first() {
            log::debug!(
                "text rknn chunk {}-{} hidden output: dims={:?} format={:?} len={}",
                chunk.layer_start,
                chunk.layer_end,
                hidden_attr.dims(),
                hidden_attr.format(),
                hidden_out.len()
            );
        }
        if hidden_out.len() < hidden.len() {
            bail!(
                "text chunk {}-{} hidden_out too short: got {}, expected at least {}",
                chunk.layer_start,
                chunk.layer_end,
                hidden_out.len(),
                hidden.len()
            );
        }

        for local_idx in 0..chunk.num_layers() {
            let layer = chunk.layer_start + local_idx;
            let prev = kv[layer]
                .as_mut()
                .ok_or_else(|| anyhow!("missing kv for layer {} when updating outputs", layer))?;
            let present_len = prev.past_len + 1;
            let expected_full_elems = num_kv_heads * present_len * head_dim;
            let expected_delta_elems = num_kv_heads * head_dim;

            let k_idx = 1 + 2 * local_idx;
            let v_idx = 1 + 2 * local_idx + 1;
            let present_k = output_bufs.get(k_idx).ok_or_else(|| {
                anyhow!(
                    "missing present_k output for layer {} from chunk {}-{}",
                    layer,
                    chunk.layer_start,
                    chunk.layer_end
                )
            })?;
            let present_v = output_bufs.get(v_idx).ok_or_else(|| {
                anyhow!(
                    "missing present_v output for layer {} from chunk {}-{}",
                    layer,
                    chunk.layer_start,
                    chunk.layer_end
                )
            })?;
            let k_attr = current_output_attrs.get(k_idx).ok_or_else(|| {
                anyhow!(
                    "missing current present_k attr for layer {} from chunk {}-{}",
                    layer,
                    chunk.layer_start,
                    chunk.layer_end
                )
            })?;
            let v_attr = current_output_attrs.get(v_idx).ok_or_else(|| {
                anyhow!(
                    "missing current present_v attr for layer {} from chunk {}-{}",
                    layer,
                    chunk.layer_start,
                    chunk.layer_end
                )
            })?;
            let k_seq =
                kv_seq_len_from_output_attr(k_attr, present_k.len(), num_kv_heads, head_dim)?;
            let v_seq =
                kv_seq_len_from_output_attr(v_attr, present_v.len(), num_kv_heads, head_dim)?;
            log::debug!(
                "text rknn chunk {}-{} layer {} kv outputs: k_dims={:?} k_format={:?} k_len={} k_seq={} v_dims={:?} v_format={:?} v_len={} v_seq={}",
                chunk.layer_start,
                chunk.layer_end,
                layer,
                k_attr.dims(),
                k_attr.format(),
                present_k.len(),
                k_seq,
                v_attr.dims(),
                v_attr.format(),
                present_v.len(),
                v_seq
            );
            if k_seq != v_seq {
                bail!(
                    "layer {} present kv seq mismatch: k_seq={} v_seq={} chunk {}-{}",
                    layer,
                    k_seq,
                    v_seq,
                    chunk.layer_start,
                    chunk.layer_end
                );
            }

            // Preferred path: model outputs delta kv (seq=1) and we append locally.
            if k_seq == 1
                && present_k.len() >= expected_delta_elems
                && present_v.len() >= expected_delta_elems
            {
                let k_layout = infer_output_kv_layout(k_attr, num_kv_heads, head_dim);
                let v_layout = infer_output_kv_layout(v_attr, num_kv_heads, head_dim);
                let present_k_head_seq_dim = kv_rknn_to_cache_layout(
                    &present_k[..expected_delta_elems],
                    num_kv_heads,
                    1,
                    head_dim,
                    KvTensorLayout::HeadSeqDim,
                );
                let present_k_seq_dim_head = kv_rknn_to_cache_layout(
                    &present_k[..expected_delta_elems],
                    num_kv_heads,
                    1,
                    head_dim,
                    KvTensorLayout::SeqDimHead,
                );
                let present_v_head_seq_dim = kv_rknn_to_cache_layout(
                    &present_v[..expected_delta_elems],
                    num_kv_heads,
                    1,
                    head_dim,
                    KvTensorLayout::HeadSeqDim,
                );
                let present_v_seq_dim_head = kv_rknn_to_cache_layout(
                    &present_v[..expected_delta_elems],
                    num_kv_heads,
                    1,
                    head_dim,
                    KvTensorLayout::SeqDimHead,
                );
                if let Some(slot) = delta_layout_debug.get_mut(layer) {
                    *slot = Some(LayerKvDeltaLayoutDebug {
                        k_head_seq_dim: present_k_head_seq_dim.clone(),
                        k_seq_dim_head: present_k_seq_dim_head.clone(),
                        v_head_seq_dim: present_v_head_seq_dim.clone(),
                        v_seq_dim_head: present_v_seq_dim_head.clone(),
                    });
                }
                let present_k = match k_layout {
                    KvTensorLayout::HeadSeqDim => &present_k_head_seq_dim,
                    KvTensorLayout::SeqDimHead => &present_k_seq_dim_head,
                };
                let present_v = match v_layout {
                    KvTensorLayout::HeadSeqDim => &present_v_head_seq_dim,
                    KvTensorLayout::SeqDimHead => &present_v_seq_dim_head,
                };
                prev.append_cache_layout_delta(present_k, present_v, num_kv_heads, head_dim)
                    .map_err(|e| anyhow!("layer {} append delta kv failed: {e}", layer))?;
                continue;
            }

            // Legacy path: model outputs full present kv (past+1) every token.
            if k_seq >= present_len && v_seq >= present_len {
                let k_layout = infer_output_kv_layout(k_attr, num_kv_heads, head_dim);
                let v_layout = infer_output_kv_layout(v_attr, num_kv_heads, head_dim);
                let present_k =
                    kv_rknn_to_cache_layout(present_k, num_kv_heads, k_seq, head_dim, k_layout);
                let present_v =
                    kv_rknn_to_cache_layout(present_v, num_kv_heads, v_seq, head_dim, v_layout);
                prev.replace_cache_layout_full(
                    &present_k,
                    &present_v,
                    present_len,
                    num_kv_heads,
                    head_dim,
                );
                continue;
            }

            {
                bail!(
                    "layer {} present kv shape mismatch: k_len={} v_len={} k_seq={} v_seq={} expected_full={} expected_delta={}",
                    layer,
                    present_k.len(),
                    present_v.len(),
                    k_seq,
                    v_seq,
                    expected_full_elems,
                    expected_delta_elems
                );
            }
        }

        Ok(hidden_out[..hidden.len()].to_vec())
    }

    fn run_prefill_chunk(
        chunk: &ChunkModel,
        hidden: &[f32],
        seq_len: usize,
        cos: &[f32],
        sin: &[f32],
        layer_adds: Option<&[Vec<f32>]>,
        kv: &mut [Option<LayerKv>],
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        if chunk.stage != ChunkStage::Prefill {
            bail!(
                "internal error: run_prefill_chunk received {} chunk {}-{}",
                chunk.stage.as_str(),
                chunk.layer_start,
                chunk.layer_end
            );
        }
        if seq_len == 0 {
            bail!("prefill seq_len must be > 0");
        }
        if hidden.len() % seq_len != 0 || cos.len() % seq_len != 0 || sin.len() % seq_len != 0 {
            bail!(
                "prefill input size mismatch: hidden={} cos={} sin={} seq_len={}",
                hidden.len(),
                cos.len(),
                sin.len(),
                seq_len
            );
        }
        let hidden_size = hidden.len() / seq_len;
        let rope_dim = cos.len() / seq_len;
        if rope_dim != sin.len() / seq_len {
            bail!(
                "prefill rope size mismatch: cos_per_token={} sin_per_token={}",
                rope_dim,
                sin.len() / seq_len
            );
        }

        let mut expected_input_dims = vec![
            vec![1, seq_len as u32, hidden_size as u32],
            vec![seq_len as u32, rope_dim as u32],
            vec![seq_len as u32, rope_dim as u32],
        ];
        if chunk.prefill_expects_layer_add() {
            for _ in 0..chunk.num_layers() {
                expected_input_dims.push(vec![1, seq_len as u32, hidden_size as u32]);
            }
        }
        let needs_shape_update = chunk
            .input_attrs
            .iter()
            .zip(expected_input_dims.iter())
            .any(|(attr, dims)| {
                let curr = attr.dims();
                curr.len() != dims.len() || curr.iter().zip(dims.iter()).any(|(a, b)| *a != *b)
            });
        if needs_shape_update {
            chunk
                .rknn
                .set_input_shapes(&chunk.input_attrs, &expected_input_dims)
                .map_err(|e| {
                    anyhow!(
                        "text prefill chunk {}-{} set_input_shapes failed for seq_len {}: {}",
                        chunk.layer_start,
                        chunk.layer_end,
                        seq_len,
                        e
                    )
                })?;
        }

        let mut input_buffers = Vec::with_capacity(chunk.input_attrs.len());
        input_buffers.push(OwnedInputBuffer::from_f32(
            hidden,
            chunk.input_attrs[0].dtype(),
        )?);
        input_buffers.push(OwnedInputBuffer::from_f32(
            cos,
            chunk.input_attrs[1].dtype(),
        )?);
        input_buffers.push(OwnedInputBuffer::from_f32(
            sin,
            chunk.input_attrs[2].dtype(),
        )?);

        let zero_layer_add = if chunk.prefill_expects_layer_add() {
            Some(vec![0f32; hidden.len()])
        } else {
            None
        };
        if chunk.prefill_expects_layer_add() {
            for local_idx in 0..chunk.num_layers() {
                let layer = chunk.layer_start + local_idx;
                let add_slice = layer_adds
                    .and_then(|adds| adds.get(layer))
                    .map(|buf| buf.as_slice())
                    .unwrap_or_else(|| {
                        zero_layer_add
                            .as_ref()
                            .expect("zero layer_add buffer must exist")
                            .as_slice()
                    });
                if add_slice.len() != hidden.len() {
                    bail!(
                        "prefill layer_add size mismatch for layer {}: got {}, expected {}",
                        layer,
                        add_slice.len(),
                        hidden.len()
                    );
                }
                let add_attr_idx = 3 + local_idx;
                input_buffers.push(OwnedInputBuffer::from_f32(
                    add_slice,
                    chunk.input_attrs[add_attr_idx].dtype(),
                )?);
            }
        }

        let mut inputs = Vec::with_capacity(input_buffers.len());
        for (idx, buf) in input_buffers.iter().enumerate() {
            inputs.push(Input::new(
                idx as u32,
                buf.as_buf_view(),
                true,
                chunk.input_attrs[idx].format(),
            ));
        }

        chunk.rknn.set_inputs(inputs).map_err(|e| {
            anyhow!(
                "text prefill chunk {}-{} set_inputs failed: {e}",
                chunk.layer_start,
                chunk.layer_end
            )
        })?;
        chunk.rknn.run().map_err(|e| {
            anyhow!(
                "text prefill chunk {}-{} run failed: {e}",
                chunk.layer_start,
                chunk.layer_end
            )
        })?;

        let current_output_attrs = query_current_output_attrs(chunk)?;
        let mut output_bufs: Vec<Vec<f32>> = current_output_attrs
            .iter()
            .enumerate()
            .map(|(i, attr)| {
                let n = tensor_attr_num_elements(attr)?;
                if n == 0 {
                    bail!(
                        "text prefill chunk {}-{} output[{}] has n_elems=0",
                        chunk.layer_start,
                        chunk.layer_end,
                        i
                    );
                }
                Ok(vec![0f32; n])
            })
            .collect::<Result<Vec<_>>>()?;
        let mut outputs = Vec::with_capacity(output_bufs.len());
        for (i, out) in output_bufs.iter_mut().enumerate() {
            outputs.push(Output {
                index: i as u32,
                kind: OutputKind::Preallocated {
                    buf: BufMutView::F32(out),
                    want_float: true,
                },
            });
        }
        chunk.rknn.get_outputs(&mut outputs).map_err(|e| {
            anyhow!(
                "text prefill chunk {}-{} get_outputs failed: {e}",
                chunk.layer_start,
                chunk.layer_end
            )
        })?;

        let hidden_out = output_bufs.first().ok_or_else(|| {
            anyhow!(
                "text prefill chunk {}-{} returned no outputs",
                chunk.layer_start,
                chunk.layer_end
            )
        })?;
        if hidden_out.len() < hidden.len() {
            bail!(
                "text prefill chunk {}-{} hidden_out too short: got {}, expected at least {}",
                chunk.layer_start,
                chunk.layer_end,
                hidden_out.len(),
                hidden.len()
            );
        }

        for local_idx in 0..chunk.num_layers() {
            let layer = chunk.layer_start + local_idx;
            let k_idx = 1 + 2 * local_idx;
            let v_idx = 1 + 2 * local_idx + 1;
            let present_k = output_bufs.get(k_idx).ok_or_else(|| {
                anyhow!(
                    "missing prefill present_k output for layer {} from chunk {}-{}",
                    layer,
                    chunk.layer_start,
                    chunk.layer_end
                )
            })?;
            let present_v = output_bufs.get(v_idx).ok_or_else(|| {
                anyhow!(
                    "missing prefill present_v output for layer {} from chunk {}-{}",
                    layer,
                    chunk.layer_start,
                    chunk.layer_end
                )
            })?;
            let k_attr = current_output_attrs.get(k_idx).ok_or_else(|| {
                anyhow!(
                    "missing prefill current present_k attr for layer {} from chunk {}-{}",
                    layer,
                    chunk.layer_start,
                    chunk.layer_end
                )
            })?;
            let v_attr = current_output_attrs.get(v_idx).ok_or_else(|| {
                anyhow!(
                    "missing prefill current present_v attr for layer {} from chunk {}-{}",
                    layer,
                    chunk.layer_start,
                    chunk.layer_end
                )
            })?;
            let k_seq =
                kv_seq_len_from_output_attr(k_attr, present_k.len(), num_kv_heads, head_dim)?;
            let v_seq =
                kv_seq_len_from_output_attr(v_attr, present_v.len(), num_kv_heads, head_dim)?;
            if k_seq != v_seq {
                bail!(
                    "prefill layer {} present kv seq mismatch: k_seq={} v_seq={}",
                    layer,
                    k_seq,
                    v_seq
                );
            }
            if k_seq < seq_len {
                bail!(
                    "prefill layer {} present kv seq too short: got {}, expected at least {}",
                    layer,
                    k_seq,
                    seq_len
                );
            }

            let k_layout = infer_output_kv_layout(k_attr, num_kv_heads, head_dim);
            let v_layout = infer_output_kv_layout(v_attr, num_kv_heads, head_dim);
            let present_k =
                kv_rknn_to_cache_layout(present_k, num_kv_heads, k_seq, head_dim, k_layout);
            let present_v =
                kv_rknn_to_cache_layout(present_v, num_kv_heads, v_seq, head_dim, v_layout);
            let present_k =
                truncate_cache_layout_seq(&present_k, num_kv_heads, k_seq, seq_len, head_dim)?;
            let present_v =
                truncate_cache_layout_seq(&present_v, num_kv_heads, v_seq, seq_len, head_dim)?;

            kv[layer] = Some(LayerKv {
                past_len: seq_len,
                k: present_k,
                v: present_v,
                native: None,
            });
        }

        Ok(hidden_out[..hidden.len()].to_vec())
    }

    pub fn forward_prefill(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        layer_adds: Option<&[Vec<f32>]>,
        out_device: &Device,
        out_dtype: DType,
    ) -> Result<Tensor> {
        let prefill_chunks = self
            .prefill_chunks
            .as_ref()
            .ok_or_else(|| anyhow!("text rknn prefill chunks are not loaded"))?;
        if index_pos != 0 {
            bail!("text rknn prefill only supports index_pos=0, got {index_pos}");
        }

        let (seq_len, mut hidden) = tensor_to_f32_vec3(x, self.hidden_size)?;
        let expected_layer_add_len = seq_len
            .checked_mul(self.hidden_size)
            .ok_or_else(|| anyhow!("prefill layer_add size overflow"))?;
        if let Some(adds) = layer_adds {
            if adds.len() != self.num_hidden_layers {
                bail!(
                    "prefill layer_add count mismatch: got {}, expected {}",
                    adds.len(),
                    self.num_hidden_layers
                );
            }
            if !self.prefill_supports_deepstack && adds.iter().any(|add| !add.is_empty()) {
                bail!("prefill rknn chunks do not support deepstack layer_add inputs");
            }
            for (layer_idx, add) in adds.iter().enumerate() {
                if !add.is_empty() && add.len() != expected_layer_add_len {
                    bail!(
                        "prefill layer_add[{}] size mismatch: got {}, expected {}",
                        layer_idx,
                        add.len(),
                        expected_layer_add_len
                    );
                }
            }
        }

        let (cos, sin) = build_rotary_window(index_pos, seq_len, self.head_dim, self.rope_theta);
        let mut kv = vec![None; self.num_hidden_layers];
        for chunk in prefill_chunks {
            hidden = Self::run_prefill_chunk(
                chunk,
                &hidden,
                seq_len,
                &cos,
                &sin,
                layer_adds,
                &mut kv,
                self.num_key_value_heads,
                self.head_dim,
            )?;
        }

        self.kv = kv;
        self.last_delta_layout_debug = vec![None; self.num_hidden_layers];
        self.synced_from_cache = true;
        for item in &mut self.chunk_last_selected_shapes {
            *item = None;
        }

        Tensor::from_vec(
            hidden,
            Shape::from_dims(&[1, seq_len, self.hidden_size]),
            &Device::Cpu,
        )
        .map_err(|e| anyhow!("prefill hidden tensor build failed: {e}"))?
        .to_dtype(out_dtype)
        .map_err(|e| anyhow!("prefill hidden to_dtype failed: {e}"))?
        .to_device(out_device)
        .map_err(|e| anyhow!("prefill hidden to_device failed: {e}"))
    }

    pub fn forward_decode(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        cache: &Cache,
        out_device: &Device,
        out_dtype: DType,
    ) -> Result<Tensor> {
        if !self.synced_from_cache {
            self.sync_from_cache(cache)?;
        }
        let current_past_len = self
            .kv
            .first()
            .and_then(|kv| kv.as_ref().map(|kv| kv.past_len))
            .unwrap_or(index_pos);
        let next_bucket = select_decode_bucket(
            self.chunks
                .first()
                .ok_or_else(|| anyhow!("text rknn has no decode chunks"))?,
            current_past_len.max(1),
        );
        if self.max_supported_bucket > 0 && next_bucket > self.max_supported_bucket {
            bail!(
                "text rknn decode bucket {} exceeds max supported bucket {}",
                next_bucket,
                self.max_supported_bucket
            );
        }
        let mut hidden = tensor_to_f32_vec3_seq1(x, self.hidden_size)?;
        let (cos, sin) = build_rotary(index_pos, self.head_dim, self.rope_theta);
        for item in &mut self.last_delta_layout_debug {
            *item = None;
        }

        for (chunk_idx, chunk) in self.chunks.iter().enumerate() {
            hidden = Self::run_chunk(
                chunk,
                &mut self.chunk_last_selected_shapes[chunk_idx],
                &hidden,
                &cos,
                &sin,
                &mut self.kv,
                &mut self.last_delta_layout_debug,
                self.num_key_value_heads,
                self.head_dim,
            )?;
        }

        Tensor::from_vec(
            hidden,
            Shape::from_dims(&[1, 1, self.hidden_size]),
            &Device::Cpu,
        )
        .map_err(|e| anyhow!("hidden tensor build failed: {e}"))?
        .to_dtype(out_dtype)
        .map_err(|e| anyhow!("hidden to_dtype failed: {e}"))?
        .to_device(out_device)
        .map_err(|e| anyhow!("hidden to_device failed: {e}"))
    }
}
