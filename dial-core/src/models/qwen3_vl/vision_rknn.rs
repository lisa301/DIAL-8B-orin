use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor};
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
    tensor::{DataTypeKind, TensorFormatKind},
    utils::find_rknn_library,
    RKNN,
};

use super::VisionOutputs;

pub struct VisionRknn {
    rknn: RKNN<RuntimeAPI>,
    input_attr: InputAttr,
    output_attrs: Vec<OutputAttr>,
}

impl VisionRknn {
    pub fn load(model_path: &Path, lib_path: Option<&Path>) -> Result<Self> {
        let mut model_data = std::fs::read(model_path)
            .map_err(|e| anyhow!("failed to read rknn model {}: {e}", model_path.display()))?;

        let lib_path = match lib_path {
            Some(path) => path.to_path_buf(),
            None => find_rknn_library()
                .next()
                .ok_or_else(|| anyhow!("cannot find librknnrt.so (set --vision-rknn-lib)"))?,
        };

        let rknn = RKNN::new_with_library(lib_path, &mut model_data, RknnInitFlags::builder())
            .map_err(|e| anyhow!("rknn init failed: {e}"))?;

        if let Err(e) = rknn.set_core_mask(NpuCores::cores_0_1_2()) {
            log::warn!("rknn set_core_mask(0_1_2) failed: {}; fallback to auto", e);
            if let Err(e2) = rknn.set_core_mask(NpuCores::auto()) {
                log::warn!(
                    "rknn set_core_mask(auto) failed: {}; continue with runtime default",
                    e2
                );
            }
        }

        let io_num = rknn
            .query::<InputOutputNum>()
            .map_err(|e| anyhow!("rknn query in/out num failed: {e}"))?;
        if io_num.input_num() != 1 || io_num.output_num() == 0 {
            bail!(
                "rknn model must have 1 input and at least 1 output, got {}/{}",
                io_num.input_num(),
                io_num.output_num()
            );
        }

        let input_attr = rknn
            .query_with_input::<InputAttr>(0)
            .map_err(|e| anyhow!("rknn query input attr failed: {e}"))?;
        let mut output_attrs = Vec::with_capacity(io_num.output_num() as usize);
        for i in 0..io_num.output_num() {
            output_attrs.push(
                rknn.query_with_input::<OutputAttr>(i)
                    .map_err(|e| anyhow!("rknn query output attr {i} failed: {e}"))?,
            );
        }
        log::info!(
            "vision rknn input: dims={:?} fmt={:?} dtype={:?}",
            input_attr.dims(),
            input_attr.format(),
            input_attr.dtype(),
        );
        for output_attr in &output_attrs {
            log::info!(
                "vision rknn output[{}]: name={} dims={:?} fmt={:?} dtype={:?}",
                output_attr.index(),
                output_attr.name(),
                output_attr.dims(),
                output_attr.format(),
                output_attr.dtype()
            );
        }

        Ok(Self {
            rknn,
            input_attr,
            output_attrs,
        })
    }

    pub fn expected_side(&self) -> Option<u32> {
        let dims = self.input_attr.dims();
        if dims.len() != 4 {
            return None;
        }
        match self.input_attr.format() {
            TensorFormatKind::NCHW(_) => {
                let (_, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
                if c != 3 || h != w {
                    return None;
                }
                Some(h)
            }
            TensorFormatKind::NHWC(_) => {
                let (_, h, w, c) = (dims[0], dims[1], dims[2], dims[3]);
                if c != 3 || h != w {
                    return None;
                }
                Some(h)
            }
            _ => None,
        }
    }

    pub fn input_summary(&self) -> String {
        format!(
            "dims={:?} format={:?} dtype={:?}",
            self.input_attr.dims(),
            self.input_attr.format(),
            self.input_attr.dtype()
        )
    }

    pub fn encode(&self, image: &Tensor, device: &Device, dtype: DType) -> Result<VisionOutputs> {
        let image = image
            .to_device(&Device::Cpu)
            .map_err(|e| anyhow!("vision input to cpu failed: {e}"))?;
        let image = image
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("vision input to f32 failed: {e}"))?
            .contiguous()
            .map_err(|e| anyhow!("vision input contiguous failed: {e}"))?;

        let (b, c, h, w) = image
            .dims4()
            .map_err(|e| anyhow!("vision input dims4: {e}"))?;
        self.check_input_shape(b, c, h, w)?;

        let input_f32 = image
            .flatten_all()
            .map_err(|e| anyhow!("vision input flatten failed: {e}"))?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("vision input to_vec failed: {e}"))?;

        let input_fmt = self.input_attr.format();
        let input_f32 = match input_fmt {
            TensorFormatKind::NCHW(_) => input_f32,
            TensorFormatKind::NHWC(_) => Self::nchw_to_nhwc(&input_f32, b, c, h, w),
            _ => bail!("unsupported rknn input format: {input_fmt:?}"),
        };
        enum InputBuffer {
            F32(Vec<f32>),
            F16(Vec<f16>),
            BF16(Vec<bf16>),
        }

        let input_dtype = self.input_attr.dtype();
        let input_buf = match input_dtype {
            DataTypeKind::Float32(_) => InputBuffer::F32(input_f32),
            DataTypeKind::Float16(_) => {
                InputBuffer::F16(input_f32.iter().map(|v| f16::from_f32(*v)).collect())
            }
            DataTypeKind::BFloat16(_) => {
                InputBuffer::BF16(input_f32.iter().map(|v| bf16::from_f32(*v)).collect())
            }
            DataTypeKind::Int8(_) | DataTypeKind::UInt8(_) => InputBuffer::F32(input_f32),
            other => bail!("unsupported rknn input dtype: {other:?}"),
        };
        let input_pass_through =
            !matches!(input_dtype, DataTypeKind::Int8(_) | DataTypeKind::UInt8(_));

        let input = match &input_buf {
            InputBuffer::F32(buf) => {
                Input::new(0, BufView::F32(buf), input_pass_through, input_fmt)
            }
            InputBuffer::F16(buf) => {
                Input::new(0, BufView::F16(buf), input_pass_through, input_fmt)
            }
            InputBuffer::BF16(buf) => {
                Input::new(0, BufView::BF16(buf), input_pass_through, input_fmt)
            }
        };

        self.rknn
            .set_inputs(input)
            .map_err(|e| anyhow!("rknn set_inputs failed: {e}"))?;
        self.rknn
            .run()
            .map_err(|e| anyhow!("rknn run failed: {e}"))?;

        let mut output_bufs: Vec<Vec<f32>> = self
            .output_attrs
            .iter()
            .map(|attr| vec![0f32; attr.num_elements() as usize])
            .collect();
        let mut outputs = Vec::with_capacity(self.output_attrs.len());
        for (i, out) in output_bufs.iter_mut().enumerate() {
            outputs.push(Output {
                index: i as u32,
                kind: OutputKind::Preallocated {
                    buf: BufMutView::F32(out),
                    want_float: true,
                },
            });
        }

        self.rknn
            .get_outputs(&mut outputs)
            .map_err(|e| anyhow!("rknn get_outputs failed: {e}"))?;

        let mut tensors = Vec::with_capacity(self.output_attrs.len());
        for (attr, out) in self.output_attrs.iter().zip(output_bufs) {
            let out_dims: Vec<usize> = attr.dims().iter().map(|d| *d as usize).collect();
            let out = Tensor::from_vec(out, Shape::from_dims(&out_dims), &Device::Cpu)
                .map_err(|e| {
                    anyhow!(
                        "vision output tensor failed for output {}: {e}",
                        attr.index()
                    )
                })?
                .to_dtype(dtype)
                .map_err(|e| {
                    anyhow!(
                        "vision output to dtype failed for output {}: {e}",
                        attr.index()
                    )
                })?
                .to_device(device)
                .map_err(|e| {
                    anyhow!(
                        "vision output to device failed for output {}: {e}",
                        attr.index()
                    )
                })?;
            tensors.push(out);
        }

        let mut tensors = tensors.into_iter();
        let embeds = tensors
            .next()
            .ok_or_else(|| anyhow!("rknn returned no outputs"))?;
        let deepstack = tensors.collect();
        Ok(VisionOutputs { embeds, deepstack })
    }

    fn check_input_shape(&self, b: usize, c: usize, h: usize, w: usize) -> Result<()> {
        let dims = self.input_attr.dims();
        if dims.len() != 4 {
            bail!("rknn input dims not 4, got {:?}", dims);
        }
        match self.input_attr.format() {
            TensorFormatKind::NCHW(_) => {
                let expected = (
                    dims[0] as usize,
                    dims[1] as usize,
                    dims[2] as usize,
                    dims[3] as usize,
                );
                if expected != (b, c, h, w) {
                    bail!(
                        "rknn input shape mismatch, expected {:?} got ({},{},{},{})",
                        expected,
                        b,
                        c,
                        h,
                        w
                    );
                }
            }
            TensorFormatKind::NHWC(_) => {
                let expected = (
                    dims[0] as usize,
                    dims[1] as usize,
                    dims[2] as usize,
                    dims[3] as usize,
                );
                if expected != (b, h, w, c) {
                    bail!(
                        "rknn input shape mismatch, expected {:?} got ({},{},{},{})",
                        expected,
                        b,
                        h,
                        w,
                        c
                    );
                }
            }
            other => bail!("unsupported rknn input format: {other:?}"),
        }
        Ok(())
    }

    fn nchw_to_nhwc(input: &[f32], b: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
        let mut out = vec![0f32; input.len()];
        for bi in 0..b {
            for ci in 0..c {
                for yi in 0..h {
                    for xi in 0..w {
                        let in_idx = ((bi * c + ci) * h + yi) * w + xi;
                        let out_idx = ((bi * h + yi) * w + xi) * c + ci;
                        out[out_idx] = input[in_idx];
                    }
                }
            }
        }
        out
    }
}
