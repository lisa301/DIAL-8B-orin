use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
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

use crate::Args;

#[derive(Clone, Debug, PartialEq, Eq)]
struct QkvRknnConfig {
    model_dir: PathBuf,
    lib_path: Option<PathBuf>,
    layer_models: BTreeMap<usize, PathBuf>,
}

static QKV_RKNN_CONFIG: OnceLock<Option<QkvRknnConfig>> = OnceLock::new();

pub fn configure_qkv_rknn_from_args(args: &Args) -> Result<()> {
    let _ = QKV_RKNN_CONFIG.get_or_init(|| {
        let model_dir = args
            .text_qkv_rknn_dir
            .as_deref()
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var("QWEN3VL_QKV_RKNN_DIR")
                    .ok()
                    .map(PathBuf::from)
            });
        let lib_path = args
            .text_qkv_rknn_lib
            .as_deref()
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var("QWEN3VL_QKV_RKNN_LIB")
                    .ok()
                    .map(PathBuf::from)
            });

        let Some(model_dir) = model_dir else {
            return None;
        };
        match build_layer_model_index(&model_dir) {
            Ok(layer_models) => {
                log::info!(
                    "text qkv rknn configured from {} ({} layer models)",
                    model_dir.display(),
                    layer_models.len()
                );
                Some(QkvRknnConfig {
                    model_dir,
                    lib_path,
                    layer_models,
                })
            }
            Err(err) => {
                log::warn!(
                    "text qkv rknn config ignored for {}: {}",
                    model_dir.display(),
                    err
                );
                None
            }
        }
    });
    Ok(())
}

fn config() -> Option<&'static QkvRknnConfig> {
    QKV_RKNN_CONFIG.get().and_then(|cfg| cfg.as_ref())
}

fn build_layer_model_index(model_dir: &Path) -> Result<BTreeMap<usize, PathBuf>> {
    let re_single = Regex::new(r"l(\d{1,2})").map_err(|e| anyhow!("regex build failed: {e}"))?;
    let re_range =
        Regex::new(r"l(\d{1,2})_l(\d{1,2})").map_err(|e| anyhow!("regex build failed: {e}"))?;
    let mut out = BTreeMap::new();
    for entry in fs::read_dir(model_dir)
        .map_err(|e| anyhow!("read_dir {} failed: {e}", model_dir.display()))?
    {
        let path = entry
            .map_err(|e| anyhow!("read_dir entry failed: {e}"))?
            .path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rknn") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !name.contains("qkv") {
            continue;
        }
        if let Some(caps) = re_range.captures(&name) {
            let start = caps
                .get(1)
                .and_then(|m| m.as_str().parse::<usize>().ok())
                .unwrap_or(usize::MAX);
            let stop = caps
                .get(2)
                .and_then(|m| m.as_str().parse::<usize>().ok())
                .unwrap_or(usize::MAX);
            if start == stop {
                out.insert(start, path.clone());
                continue;
            }
        }
        if let Some(caps) = re_single.captures(&name) {
            if let Some(layer_idx) = caps.get(1).and_then(|m| m.as_str().parse::<usize>().ok()) {
                out.insert(layer_idx, path.clone());
            }
        }
    }
    Ok(out)
}

enum OwnedInputBuffer {
    F32(Vec<f32>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
}

impl OwnedInputBuffer {
    fn from_tensor3(
        x: &Tensor,
        expected_hidden: usize,
        dtype: DataTypeKind,
    ) -> Result<(usize, Self)> {
        let x = x
            .to_device(&Device::Cpu)
            .map_err(|e| anyhow!("qkv rknn x.to_cpu failed: {e}"))?
            .contiguous()
            .map_err(|e| anyhow!("qkv rknn x.contiguous failed: {e}"))?;
        let (b, s, h) = x
            .dims3()
            .map_err(|e| anyhow!("qkv rknn x.dims3 failed: {e}"))?;
        if b != 1 || h != expected_hidden {
            bail!("qkv rknn expects x shape (1,seq,{expected_hidden}), got ({b},{s},{h})");
        }
        let flat = x
            .flatten_all()
            .map_err(|e| anyhow!("qkv rknn x.flatten failed: {e}"))?;
        let out = match dtype {
            DataTypeKind::Float32(_) => Self::F32(
                flat.to_dtype(DType::F32)
                    .map_err(|e| anyhow!("qkv rknn x.to_f32 failed: {e}"))?
                    .to_vec1::<f32>()
                    .map_err(|e| anyhow!("qkv rknn x.to_vec1<f32> failed: {e}"))?,
            ),
            DataTypeKind::Float16(_) => Self::F16(
                flat.to_dtype(DType::F16)
                    .map_err(|e| anyhow!("qkv rknn x.to_f16 failed: {e}"))?
                    .to_vec1::<f16>()
                    .map_err(|e| anyhow!("qkv rknn x.to_vec1<f16> failed: {e}"))?,
            ),
            DataTypeKind::BFloat16(_) => Self::BF16(
                flat.to_dtype(DType::BF16)
                    .map_err(|e| anyhow!("qkv rknn x.to_bf16 failed: {e}"))?
                    .to_vec1::<bf16>()
                    .map_err(|e| anyhow!("qkv rknn x.to_vec1<bf16> failed: {e}"))?,
            ),
            other => bail!("unsupported qkv rknn input dtype: {other:?}"),
        };
        Ok((s, out))
    }

    fn as_buf_view(&self) -> BufView<'_> {
        match self {
            Self::F32(v) => BufView::F32(v),
            Self::F16(v) => BufView::F16(v),
            Self::BF16(v) => BufView::BF16(v),
        }
    }
}

enum OwnedOutputBuffer {
    F32(Vec<f32>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
}

impl OwnedOutputBuffer {
    fn for_output(dtype: DataTypeKind, n: usize) -> (Self, bool) {
        match dtype {
            DataTypeKind::Float16(_) => (Self::F16(vec![f16::ZERO; n]), false),
            DataTypeKind::BFloat16(_) => (Self::BF16(vec![bf16::ZERO; n]), false),
            DataTypeKind::Float32(_) => (Self::F32(vec![0f32; n]), false),
            _ => (Self::F32(vec![0f32; n]), true),
        }
    }

    fn as_mut_buf_view(&mut self) -> BufMutView<'_> {
        match self {
            Self::F32(v) => BufMutView::F32(v),
            Self::F16(v) => BufMutView::F16(v),
            Self::BF16(v) => BufMutView::BF16(v),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::BF16(v) => v.len(),
        }
    }

    fn into_tensor(
        self,
        expected: usize,
        seq_len: usize,
        qkv_size: usize,
        out_dtype: DType,
        out_device: &Device,
        layer_idx: usize,
    ) -> Result<Tensor> {
        if self.len() < expected {
            bail!(
                "qkv rknn layer {} output too short: got {}, expected at least {}",
                layer_idx,
                self.len(),
                expected
            );
        }
        let shape = Shape::from_dims(&[1, seq_len, qkv_size]);
        let tensor = match self {
            Self::F32(v) => Tensor::from_vec(v[..expected].to_vec(), shape, &Device::Cpu),
            Self::F16(v) => Tensor::from_vec(v[..expected].to_vec(), shape, &Device::Cpu),
            Self::BF16(v) => Tensor::from_vec(v[..expected].to_vec(), shape, &Device::Cpu),
        }
        .map_err(|e| {
            anyhow!(
                "qkv rknn layer {} output tensor build failed: {e}",
                layer_idx
            )
        })?;
        tensor
            .to_dtype(out_dtype)
            .map_err(|e| anyhow!("qkv rknn layer {} output to_dtype failed: {e}", layer_idx))?
            .to_device(out_device)
            .map_err(|e| anyhow!("qkv rknn layer {} output to_device failed: {e}", layer_idx))
    }
}

fn expected_output_elements<T: TensorAttrView>(attr: &T) -> Result<usize> {
    let elems = attr
        .dims()
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim as usize))
        .ok_or_else(|| anyhow!("qkv rknn output shape overflow for {}", attr.name()))?;
    if elems == 0 {
        bail!("qkv rknn output {} has zero elements", attr.name());
    }
    Ok(elems)
}

fn query_output_attr<'a>(
    rknn: &'a RKNN<RuntimeAPI>,
    output_attrs: &'a [OutputAttr],
    dynamic: bool,
) -> Result<ResolvedOutputAttr<'a>> {
    if !dynamic {
        return Ok(ResolvedOutputAttr::Static(
            output_attrs
                .first()
                .ok_or_else(|| anyhow!("qkv rknn missing static output attr"))?,
        ));
    }
    rknn.query_with_input::<CurrentOutputAttr>(0)
        .map(ResolvedOutputAttr::Dynamic)
        .map_err(|e| anyhow!("qkv rknn query current output attr failed: {e}"))
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

pub struct QkvRknnRunner {
    layer_idx: usize,
    rknn: RKNN<RuntimeAPI>,
    input_attrs: Vec<InputAttr>,
    output_attrs: Vec<OutputAttr>,
    input_dynamic_shapes: Vec<Vec<Vec<u32>>>,
    is_dynamic_model: bool,
    hidden_size: usize,
    qkv_size: usize,
}

impl std::fmt::Debug for QkvRknnRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QkvRknnRunner")
            .field("layer_idx", &self.layer_idx)
            .field("is_dynamic_model", &self.is_dynamic_model)
            .field("hidden_size", &self.hidden_size)
            .field("qkv_size", &self.qkv_size)
            .finish()
    }
}

impl QkvRknnRunner {
    pub fn maybe_load(
        layer_idx: Option<usize>,
        hidden_size: usize,
        qkv_size: usize,
    ) -> Option<Self> {
        let layer_idx = layer_idx?;
        let cfg = config()?;
        let model_path = cfg.layer_models.get(&layer_idx)?;
        match Self::load(
            layer_idx,
            model_path,
            cfg.lib_path.as_deref(),
            hidden_size,
            qkv_size,
        ) {
            Ok(runner) => Some(runner),
            Err(err) => {
                log::warn!(
                    "text qkv rknn layer {} load failed from {}: {}",
                    layer_idx,
                    model_path.display(),
                    err
                );
                None
            }
        }
    }

    fn load(
        layer_idx: usize,
        model_path: &Path,
        lib_path: Option<&Path>,
        hidden_size: usize,
        qkv_size: usize,
    ) -> Result<Self> {
        let lib_path = match lib_path {
            Some(path) => path.to_path_buf(),
            None => find_rknn_library()
                .next()
                .ok_or_else(|| anyhow!("cannot find librknnrt.so (set --text-qkv-rknn-lib)"))?,
        };
        let mut model_data = fs::read(model_path).map_err(|e| {
            anyhow!(
                "failed to read qkv rknn model {}: {e}",
                model_path.display()
            )
        })?;
        let rknn = RKNN::new_with_library(lib_path, &mut model_data, RknnInitFlags::builder())
            .map_err(|e| anyhow!("qkv rknn init failed for {}: {e}", model_path.display()))?;
        if let Err(e) = rknn.set_core_mask(NpuCores::cores_0_1_2()) {
            log::warn!(
                "qkv rknn set_core_mask(0_1_2) failed for {}: {}; fallback to auto",
                model_path.display(),
                e
            );
            let _ = rknn.set_core_mask(NpuCores::auto());
        }

        let io_num = rknn.query::<InputOutputNum>().map_err(|e| {
            anyhow!(
                "qkv rknn query io_num failed for {}: {e}",
                model_path.display()
            )
        })?;
        if io_num.input_num() != 1 || io_num.output_num() != 1 {
            bail!(
                "qkv rknn {} must have 1 input and 1 output, got {}/{}",
                model_path.display(),
                io_num.input_num(),
                io_num.output_num()
            );
        }

        let file_name = model_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let is_dynamic_model = file_name.contains("_dynamic_");
        let input_attr = rknn.query_with_input::<InputAttr>(0).map_err(|e| {
            anyhow!(
                "qkv rknn query input attr failed for {}: {e}",
                model_path.display()
            )
        })?;
        let input_dynamic_shapes = if is_dynamic_model {
            rknn.query_with_input::<InputDynamicRange>(0)
                .ok()
                .map(|r| r.shapes())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let output_attr = rknn.query_with_input::<OutputAttr>(0).map_err(|e| {
            anyhow!(
                "qkv rknn query output attr failed for {}: {e}",
                model_path.display()
            )
        })?;

        let input_dims = input_attr.dims();
        if input_dims.len() != 3 || input_dims[0] != 1 || input_dims[2] != hidden_size as u32 {
            bail!(
                "qkv rknn {} input shape must be [1,seq,{}], got {:?}",
                model_path.display(),
                hidden_size,
                input_dims
            );
        }
        let output_dims = output_attr.dims();
        if output_dims.len() != 3 || output_dims[0] != 1 || output_dims[2] != qkv_size as u32 {
            bail!(
                "qkv rknn {} output shape must be [1,seq,{}], got {:?}",
                model_path.display(),
                qkv_size,
                output_dims
            );
        }

        Ok(Self {
            layer_idx,
            rknn,
            input_attrs: vec![input_attr],
            output_attrs: vec![output_attr],
            input_dynamic_shapes: vec![input_dynamic_shapes],
            is_dynamic_model,
            hidden_size,
            qkv_size,
        })
    }

    pub fn forward_concat(
        &self,
        x: &Tensor,
        out_device: &Device,
        out_dtype: DType,
    ) -> Result<Tensor> {
        let (seq_len, input_buf) =
            OwnedInputBuffer::from_tensor3(x, self.hidden_size, self.input_attrs[0].dtype())?;
        if seq_len == 0 {
            bail!("qkv rknn layer {} got empty sequence", self.layer_idx);
        }

        if self.is_dynamic_model {
            let wanted = vec![vec![1, seq_len as u32, self.hidden_size as u32]];
            let current = self.input_attrs[0].dims();
            if current != wanted[0].as_slice() {
                self.rknn
                    .set_input_shapes(&self.input_attrs, &wanted)
                    .map_err(|e| {
                        anyhow!(
                            "qkv rknn layer {} set_input_shapes failed: {e}",
                            self.layer_idx
                        )
                    })?;
            }
        } else if self.input_attrs[0].dims()[1] as usize != seq_len {
            bail!(
                "qkv rknn layer {} only supports seq_len={}, got {}",
                self.layer_idx,
                self.input_attrs[0].dims()[1],
                seq_len
            );
        }

        let input_tensor = Input::new(
            0,
            input_buf.as_buf_view(),
            true,
            self.input_attrs[0].format(),
        );
        self.rknn
            .set_inputs(vec![input_tensor])
            .map_err(|e| anyhow!("qkv rknn layer {} set_inputs failed: {e}", self.layer_idx))?;
        self.rknn
            .run()
            .map_err(|e| anyhow!("qkv rknn layer {} run failed: {e}", self.layer_idx))?;

        let output_attr = query_output_attr(&self.rknn, &self.output_attrs, self.is_dynamic_model)?;
        let n = expected_output_elements(&output_attr)?;
        let (mut out, want_float) = OwnedOutputBuffer::for_output(output_attr.dtype(), n);
        {
            let mut outputs = vec![Output {
                index: 0,
                kind: OutputKind::Preallocated {
                    buf: out.as_mut_buf_view(),
                    want_float,
                },
            }];
            self.rknn.get_outputs(&mut outputs).map_err(|e| {
                anyhow!("qkv rknn layer {} get_outputs failed: {e}", self.layer_idx)
            })?;
        }

        let expected = seq_len
            .checked_mul(self.qkv_size)
            .ok_or_else(|| anyhow!("qkv rknn layer {} output size overflow", self.layer_idx))?;
        out.into_tensor(
            expected,
            seq_len,
            self.qkv_size,
            out_dtype,
            out_device,
            self.layer_idx,
        )
    }
}
