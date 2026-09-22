use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{anyhow, bail, Result};
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
    query::{InputAttr, InputOutputNum, OutputAttr, TensorAttrView},
    rknn::NpuCores,
    tensor::DataTypeKind,
    utils::find_rknn_library,
    RKNN,
};

use crate::Args;

#[derive(Clone, Debug, PartialEq, Eq)]
struct MlpRknnConfig {
    model_dir: PathBuf,
    lib_path: Option<PathBuf>,
    full_layer_models: BTreeMap<usize, PathBuf>,
    gate_up_layer_models: BTreeMap<usize, PathBuf>,
    down_layer_models: BTreeMap<usize, PathBuf>,
}

static MLP_RKNN_CONFIG: OnceLock<Option<MlpRknnConfig>> = OnceLock::new();

pub fn configure_mlp_rknn_from_args(args: &Args) -> Result<()> {
    let _ = MLP_RKNN_CONFIG.get_or_init(|| {
        let model_dir = args
            .text_mlp_rknn_dir
            .as_deref()
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var("QWEN3VL_MLP_RKNN_DIR")
                    .ok()
                    .map(PathBuf::from)
            });
        let lib_path = args
            .text_mlp_rknn_lib
            .as_deref()
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var("QWEN3VL_MLP_RKNN_LIB")
                    .ok()
                    .map(PathBuf::from)
            });

        let Some(model_dir) = model_dir else {
            return None;
        };
        match build_layer_model_index(&model_dir) {
            Ok((full_layer_models, gate_up_layer_models, down_layer_models)) => {
                log::info!(
                    "text mlp rknn configured from {} ({} full layer models, {} gate_up layer models, {} down layer models)",
                    model_dir.display(),
                    full_layer_models.len(),
                    gate_up_layer_models.len(),
                    down_layer_models.len()
                );
                Some(MlpRknnConfig {
                    model_dir,
                    lib_path,
                    full_layer_models,
                    gate_up_layer_models,
                    down_layer_models,
                })
            }
            Err(err) => {
                log::warn!(
                    "text mlp rknn config ignored for {}: {}",
                    model_dir.display(),
                    err
                );
                None
            }
        }
    });
    Ok(())
}

fn config() -> Option<&'static MlpRknnConfig> {
    MLP_RKNN_CONFIG.get().and_then(|cfg| cfg.as_ref())
}

fn build_layer_model_index(
    model_dir: &Path,
) -> Result<(
    BTreeMap<usize, PathBuf>,
    BTreeMap<usize, PathBuf>,
    BTreeMap<usize, PathBuf>,
)> {
    let re_single = Regex::new(r"l(\d{1,2})").map_err(|e| anyhow!("regex build failed: {e}"))?;
    let re_range =
        Regex::new(r"l(\d{1,2})_l(\d{1,2})").map_err(|e| anyhow!("regex build failed: {e}"))?;
    let mut full = BTreeMap::new();
    let mut gate_up = BTreeMap::new();
    let mut down = BTreeMap::new();
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
        if !name.contains("mlp") {
            continue;
        }
        let is_gate_up = name.contains("gate_up") || name.contains("gateup");
        let is_down = name.contains("down");
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
                if is_gate_up {
                    gate_up.insert(start, path.clone());
                } else if is_down {
                    down.insert(start, path.clone());
                } else {
                    full.insert(start, path.clone());
                }
                continue;
            }
        }
        if let Some(caps) = re_single.captures(&name) {
            if let Some(layer_idx) = caps.get(1).and_then(|m| m.as_str().parse::<usize>().ok()) {
                if is_gate_up {
                    gate_up.insert(layer_idx, path.clone());
                } else if is_down {
                    down.insert(layer_idx, path.clone());
                } else {
                    full.insert(layer_idx, path.clone());
                }
            }
        }
    }
    Ok((full, gate_up, down))
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
    ) -> Result<(usize, Self, bool)> {
        let x = x
            .to_device(&Device::Cpu)
            .map_err(|e| anyhow!("mlp rknn x.to_cpu failed: {e}"))?
            .contiguous()
            .map_err(|e| anyhow!("mlp rknn x.contiguous failed: {e}"))?;
        let (b, s, h) = x
            .dims3()
            .map_err(|e| anyhow!("mlp rknn x.dims3 failed: {e}"))?;
        if b != 1 || h != expected_hidden {
            bail!("mlp rknn expects x shape (1,seq,{expected_hidden}), got ({b},{s},{h})");
        }
        let flat = x
            .flatten_all()
            .map_err(|e| anyhow!("mlp rknn x.flatten failed: {e}"))?;
        let out = match dtype {
            DataTypeKind::Float32(_) => Self::F32(
                flat.to_dtype(DType::F32)
                    .map_err(|e| anyhow!("mlp rknn x.to_f32 failed: {e}"))?
                    .to_vec1::<f32>()
                    .map_err(|e| anyhow!("mlp rknn x.to_vec1<f32> failed: {e}"))?,
            ),
            DataTypeKind::Float16(_) => Self::F16(
                flat.to_dtype(DType::F16)
                    .map_err(|e| anyhow!("mlp rknn x.to_f16 failed: {e}"))?
                    .to_vec1::<f16>()
                    .map_err(|e| anyhow!("mlp rknn x.to_vec1<f16> failed: {e}"))?,
            ),
            DataTypeKind::BFloat16(_) => Self::BF16(
                flat.to_dtype(DType::BF16)
                    .map_err(|e| anyhow!("mlp rknn x.to_bf16 failed: {e}"))?
                    .to_vec1::<bf16>()
                    .map_err(|e| anyhow!("mlp rknn x.to_vec1<bf16> failed: {e}"))?,
            ),
            DataTypeKind::Int8(_) | DataTypeKind::UInt8(_) => Self::F32(
                flat.to_dtype(DType::F32)
                    .map_err(|e| anyhow!("mlp rknn x.to_f32 failed: {e}"))?
                    .to_vec1::<f32>()
                    .map_err(|e| anyhow!("mlp rknn x.to_vec1<f32> failed: {e}"))?,
            ),
            other => bail!("unsupported mlp rknn input dtype: {other:?}"),
        };
        let pass_through = !matches!(dtype, DataTypeKind::Int8(_) | DataTypeKind::UInt8(_));
        Ok((s, out, pass_through))
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
        hidden_size: usize,
        out_dtype: DType,
        out_device: &Device,
        layer_idx: usize,
    ) -> Result<Tensor> {
        if self.len() < expected {
            bail!(
                "mlp rknn layer {} output too short: got {}, expected at least {}",
                layer_idx,
                self.len(),
                expected
            );
        }
        let shape = Shape::from_dims(&[1, seq_len, hidden_size]);
        let tensor = match self {
            Self::F32(v) => Tensor::from_vec(v[..expected].to_vec(), shape, &Device::Cpu),
            Self::F16(v) => Tensor::from_vec(v[..expected].to_vec(), shape, &Device::Cpu),
            Self::BF16(v) => Tensor::from_vec(v[..expected].to_vec(), shape, &Device::Cpu),
        }
        .map_err(|e| {
            anyhow!(
                "mlp rknn layer {} output tensor build failed: {e}",
                layer_idx
            )
        })?;
        tensor
            .to_dtype(out_dtype)
            .map_err(|e| anyhow!("mlp rknn layer {} output to_dtype failed: {e}", layer_idx))?
            .to_device(out_device)
            .map_err(|e| anyhow!("mlp rknn layer {} output to_device failed: {e}", layer_idx))
    }
}

fn expected_output_elements<T: TensorAttrView>(attr: &T) -> Result<usize> {
    let elems = attr
        .dims()
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim as usize))
        .ok_or_else(|| anyhow!("mlp rknn output shape overflow for {}", attr.name()))?;
    if elems == 0 {
        bail!("mlp rknn output {} has zero elements", attr.name());
    }
    Ok(elems)
}

#[derive(Clone, Debug)]
pub struct MlpRknnSpec {
    layer_idx: usize,
    input_size: usize,
    output_size: usize,
    label: &'static str,
    model_path: PathBuf,
    lib_path: PathBuf,
}

impl MlpRknnSpec {
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    pub fn label(&self) -> &'static str {
        self.label
    }
}

pub struct MlpRknnRunner {
    spec: MlpRknnSpec,
    rknn: RKNN<RuntimeAPI>,
    input_attr: InputAttr,
    output_attr: OutputAttr,
}

impl std::fmt::Debug for MlpRknnRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlpRknnRunner")
            .field("layer_idx", &self.spec.layer_idx)
            .field("input_size", &self.spec.input_size)
            .field("output_size", &self.spec.output_size)
            .field("label", &self.spec.label)
            .finish()
    }
}

impl MlpRknnRunner {
    pub fn layer_idx(&self) -> usize {
        self.spec.layer_idx
    }

    pub fn label(&self) -> &'static str {
        self.spec.label
    }

    pub fn spec(&self) -> MlpRknnSpec {
        self.spec.clone()
    }

    pub fn load_spec(spec: &MlpRknnSpec) -> Result<Self> {
        Self::load_resolved(spec.clone())
    }

    pub fn maybe_full_spec(layer_idx: Option<usize>, hidden_size: usize) -> Option<MlpRknnSpec> {
        let layer_idx = layer_idx?;
        let cfg = config()?;
        let model_path = cfg.full_layer_models.get(&layer_idx)?;
        match Self::build_spec(
            layer_idx,
            "full",
            model_path,
            cfg.lib_path.as_deref(),
            hidden_size,
            hidden_size,
        ) {
            Ok(spec) => Some(spec),
            Err(err) => {
                log::warn!(
                    "text mlp full rknn layer {} spec ignored from {}: {}",
                    layer_idx,
                    model_path.display(),
                    err
                );
                None
            }
        }
    }

    pub fn maybe_load_full(layer_idx: Option<usize>, hidden_size: usize) -> Option<Self> {
        let spec = Self::maybe_full_spec(layer_idx, hidden_size)?;
        match Self::load_spec(&spec) {
            Ok(runner) => Some(runner),
            Err(err) => {
                log::warn!(
                    "text mlp full rknn layer {} load failed from {}: {}",
                    spec.layer_idx,
                    spec.model_path.display(),
                    err
                );
                None
            }
        }
    }

    pub fn maybe_down_spec(
        layer_idx: Option<usize>,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> Option<MlpRknnSpec> {
        let layer_idx = layer_idx?;
        let cfg = config()?;
        let model_path = cfg.down_layer_models.get(&layer_idx)?;
        match Self::build_spec(
            layer_idx,
            "down",
            model_path,
            cfg.lib_path.as_deref(),
            intermediate_size,
            hidden_size,
        ) {
            Ok(spec) => Some(spec),
            Err(err) => {
                log::warn!(
                    "text mlp down rknn layer {} spec ignored from {}: {}",
                    layer_idx,
                    model_path.display(),
                    err
                );
                None
            }
        }
    }

    pub fn maybe_load_down(
        layer_idx: Option<usize>,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> Option<Self> {
        let spec = Self::maybe_down_spec(layer_idx, intermediate_size, hidden_size)?;
        match Self::load_spec(&spec) {
            Ok(runner) => Some(runner),
            Err(err) => {
                log::warn!(
                    "text mlp down rknn layer {} load failed from {}: {}",
                    spec.layer_idx,
                    spec.model_path.display(),
                    err
                );
                None
            }
        }
    }

    pub fn maybe_gate_up_spec(
        layer_idx: Option<usize>,
        hidden_size: usize,
        gate_up_size: usize,
    ) -> Option<MlpRknnSpec> {
        let layer_idx = layer_idx?;
        let cfg = config()?;
        let model_path = cfg.gate_up_layer_models.get(&layer_idx)?;
        match Self::build_spec(
            layer_idx,
            "gate_up",
            model_path,
            cfg.lib_path.as_deref(),
            hidden_size,
            gate_up_size,
        ) {
            Ok(spec) => Some(spec),
            Err(err) => {
                log::warn!(
                    "text mlp gate_up rknn layer {} spec ignored from {}: {}",
                    layer_idx,
                    model_path.display(),
                    err
                );
                None
            }
        }
    }

    pub fn maybe_load_gate_up(
        layer_idx: Option<usize>,
        hidden_size: usize,
        gate_up_size: usize,
    ) -> Option<Self> {
        let spec = Self::maybe_gate_up_spec(layer_idx, hidden_size, gate_up_size)?;
        match Self::load_spec(&spec) {
            Ok(runner) => Some(runner),
            Err(err) => {
                log::warn!(
                    "text mlp gate_up rknn layer {} load failed from {}: {}",
                    spec.layer_idx,
                    spec.model_path.display(),
                    err
                );
                None
            }
        }
    }

    fn build_spec(
        layer_idx: usize,
        label: &'static str,
        model_path: &Path,
        lib_path: Option<&Path>,
        input_size: usize,
        output_size: usize,
    ) -> Result<MlpRknnSpec> {
        let lib_path = match lib_path {
            Some(path) => path.to_path_buf(),
            None => find_rknn_library()
                .next()
                .ok_or_else(|| anyhow!("cannot find librknnrt.so (set --text-mlp-rknn-lib)"))?,
        };
        Ok(MlpRknnSpec {
            layer_idx,
            input_size,
            output_size,
            label,
            model_path: model_path.to_path_buf(),
            lib_path,
        })
    }

    fn load_resolved(spec: MlpRknnSpec) -> Result<Self> {
        let mut model_data = fs::read(&spec.model_path).map_err(|e| {
            anyhow!(
                "failed to read mlp rknn model {}: {e}",
                spec.model_path.display()
            )
        })?;
        let rknn = RKNN::new_with_library(
            spec.lib_path.clone(),
            &mut model_data,
            RknnInitFlags::builder(),
        )
        .map_err(|e| {
            anyhow!(
                "mlp rknn init failed for {}: {e}",
                spec.model_path.display()
            )
        })?;
        if let Err(e) = rknn.set_core_mask(NpuCores::cores_0_1_2()) {
            log::warn!(
                "mlp rknn set_core_mask(0_1_2) failed for {}: {}; fallback to auto",
                spec.model_path.display(),
                e
            );
            let _ = rknn.set_core_mask(NpuCores::auto());
        }

        let io_num = rknn.query::<InputOutputNum>().map_err(|e| {
            anyhow!(
                "mlp rknn query io_num failed for {}: {e}",
                spec.model_path.display()
            )
        })?;
        if io_num.input_num() != 1 || io_num.output_num() != 1 {
            bail!(
                "mlp rknn {} must have 1 input and 1 output, got {}/{}",
                spec.model_path.display(),
                io_num.input_num(),
                io_num.output_num()
            );
        }

        let input_attr = rknn.query_with_input::<InputAttr>(0).map_err(|e| {
            anyhow!(
                "mlp rknn query input attr failed for {}: {e}",
                spec.model_path.display()
            )
        })?;
        let output_attr = rknn.query_with_input::<OutputAttr>(0).map_err(|e| {
            anyhow!(
                "mlp rknn query output attr failed for {}: {e}",
                spec.model_path.display()
            )
        })?;

        let input_dims = input_attr.dims();
        if input_dims.len() != 3 || input_dims[0] != 1 || input_dims[2] != spec.input_size as u32 {
            bail!(
                "mlp rknn {} input shape must be [1,seq,{}], got {:?}",
                spec.model_path.display(),
                spec.input_size,
                input_dims
            );
        }
        let output_dims = output_attr.dims();
        if output_dims.len() != 3
            || output_dims[0] != 1
            || output_dims[2] != spec.output_size as u32
        {
            bail!(
                "mlp rknn {} output shape must be [1,seq,{}], got {:?}",
                spec.model_path.display(),
                spec.output_size,
                output_dims
            );
        }

        log::info!(
            "text mlp {} rknn layer {} loaded from {} input={:?} {:?} output={:?} {:?}",
            spec.label,
            spec.layer_idx,
            spec.model_path.display(),
            input_dims,
            input_attr.dtype(),
            output_dims,
            output_attr.dtype()
        );

        Ok(Self {
            spec,
            rknn,
            input_attr,
            output_attr,
        })
    }

    pub fn forward(&self, x: &Tensor, out_device: &Device, out_dtype: DType) -> Result<Tensor> {
        let (seq_len, input_buf, input_pass_through) =
            OwnedInputBuffer::from_tensor3(x, self.spec.input_size, self.input_attr.dtype())?;
        if seq_len == 0 {
            bail!("mlp rknn layer {} got empty sequence", self.spec.layer_idx);
        }
        if self.input_attr.dims()[1] as usize != seq_len {
            bail!(
                "mlp rknn layer {} only supports seq_len={}, got {}",
                self.spec.layer_idx,
                self.input_attr.dims()[1],
                seq_len
            );
        }

        let input_tensor = Input::new(
            0,
            input_buf.as_buf_view(),
            input_pass_through,
            self.input_attr.format(),
        );
        self.rknn.set_inputs(vec![input_tensor]).map_err(|e| {
            anyhow!(
                "mlp rknn layer {} set_inputs failed: {e}",
                self.spec.layer_idx
            )
        })?;
        self.rknn
            .run()
            .map_err(|e| anyhow!("mlp rknn layer {} run failed: {e}", self.spec.layer_idx))?;

        let n = expected_output_elements(&self.output_attr)?;
        let (mut out, want_float) = OwnedOutputBuffer::for_output(self.output_attr.dtype(), n);
        {
            let mut outputs = vec![Output {
                index: 0,
                kind: OutputKind::Preallocated {
                    buf: out.as_mut_buf_view(),
                    want_float,
                },
            }];
            self.rknn.get_outputs(&mut outputs).map_err(|e| {
                anyhow!(
                    "mlp rknn layer {} get_outputs failed: {e}",
                    self.spec.layer_idx
                )
            })?;
        }

        let expected = seq_len.checked_mul(self.spec.output_size).ok_or_else(|| {
            anyhow!(
                "mlp rknn layer {} output size overflow",
                self.spec.layer_idx
            )
        })?;
        out.into_tensor(
            expected,
            seq_len,
            self.spec.output_size,
            out_dtype,
            out_device,
            self.spec.layer_idx,
        )
    }
}
