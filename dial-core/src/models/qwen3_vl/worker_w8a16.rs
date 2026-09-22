use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "cuda")]
use std::sync::atomic::AtomicUsize;

use candle_core::{DType, Device, Tensor};
use candle_nn::Module;

static WORKER_W8A16_ENABLED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "cuda")]
static WORKER_W8A16_LINEAR_COUNT: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static WORKER_W8A16_WEIGHT_BYTES: AtomicUsize = AtomicUsize::new(0);

pub fn configure_worker_w8a16_from_args(
    requested: bool,
    worker_mode: bool,
    device: &Device,
    dtype: DType,
) {
    let gguf_enabled = super::worker_gguf_enabled();
    let enabled =
        requested && !gguf_enabled && worker_mode && device.is_cuda() && dtype == DType::F16;
    WORKER_W8A16_ENABLED.store(enabled, Ordering::Relaxed);

    if requested && gguf_enabled {
        log::warn!(
            "both worker W8A16 and GGUF were requested; GGUF takes precedence and W8A16 is disabled"
        );
    } else if requested && !worker_mode {
        log::info!("worker W8A16 ignored in master mode; master weights remain unchanged");
    } else if requested && !device.is_cuda() {
        log::warn!("worker W8A16 requested on a non-CUDA device; falling back to dense weights");
    } else if requested && dtype != DType::F16 {
        log::warn!(
            "worker W8A16 requires --dtype f16 (got {:?}); falling back to dense weights",
            dtype
        );
    } else if enabled {
        log::info!(
            "worker W8A16 enabled: CUDA linear weights use row-wise int8, activations/outputs remain F16"
        );
    }
}

pub fn worker_w8a16_enabled() -> bool {
    WORKER_W8A16_ENABLED.load(Ordering::Relaxed)
}

pub fn worker_w8a16_summary() -> Option<(usize, usize)> {
    if !worker_w8a16_enabled() {
        return None;
    }
    #[cfg(feature = "cuda")]
    {
        return Some((
            WORKER_W8A16_LINEAR_COUNT.load(Ordering::Relaxed),
            WORKER_W8A16_WEIGHT_BYTES.load(Ordering::Relaxed),
        ));
    }
    #[cfg(not(feature = "cuda"))]
    None
}

#[derive(Debug, Clone)]
pub enum TextLinear {
    Dense(candle_nn::Linear),
    Gguf {
        name: String,
        matmul: candle_core::quantized::QMatMul,
        fp16_prefill: Option<candle_nn::Linear>,
    },
    #[cfg(feature = "cuda")]
    W8A16(std::sync::Arc<CudaW8A16Linear>),
}

impl TextLinear {
    pub fn from_dense(name: &str, dense: candle_nn::Linear) -> candle_core::Result<Self> {
        if !worker_w8a16_enabled() {
            return Ok(Self::Dense(dense));
        }

        #[cfg(feature = "cuda")]
        {
            match CudaW8A16Linear::from_linear(name, &dense) {
                Ok(linear) => return Ok(Self::W8A16(std::sync::Arc::new(linear))),
                Err(err) => {
                    log::warn!(
                        "worker W8A16 conversion failed for {name}: {err}; falling back to dense F16"
                    );
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        log::warn!(
            "worker W8A16 requested for {name}, but this binary has no cuda feature; falling back to dense F16"
        );

        Ok(Self::Dense(dense))
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(x),
            Self::Gguf {
                matmul,
                fp16_prefill,
                ..
            } => {
                if let Some(prefill) = fp16_prefill {
                    if super::worker_gguf_prefill_for_dims(x.dims()) {
                        return prefill.forward(x);
                    }
                }
                super::forward_worker_gguf(matmul, x)
            }
            #[cfg(feature = "cuda")]
            Self::W8A16(linear) => linear.forward(x),
        }
    }

    pub(crate) fn gguf_dequantize_f16_weight(&self) -> candle_core::Result<Tensor> {
        match self {
            Self::Gguf { matmul, .. } => matmul.dequantize_f16(),
            _ => candle_core::bail!("FP16 GGUF prefill requires a GGUF linear"),
        }
    }

    pub(crate) fn with_gguf_fp16_prefill(self, prefill: candle_nn::Linear) -> Self {
        match self {
            Self::Gguf { name, matmul, .. } => Self::Gguf {
                name,
                matmul,
                fp16_prefill: Some(prefill),
            },
            other => other,
        }
    }
}

#[cfg(feature = "cuda")]
const W8A16_CUDA_SOURCE: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void dial_w8a16_decode_f16(
    const half* x,
    const signed char* weight,
    const float* scale,
    half* out,
    int rows,
    int in_dim,
    int out_dim) {
    const int out_idx = (int)blockIdx.x;
    const int row = (int)blockIdx.y;
    if (out_idx >= out_dim || row >= rows) return;

    const signed char* w = weight + ((long long)out_idx * in_dim);
    const half* input = x + ((long long)row * in_dim);
    float sum = 0.0f;
    for (int k = (int)threadIdx.x; k < in_dim; k += (int)blockDim.x) {
        sum += (float)w[k] * __half2float(input[k]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float warp_sums[8];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = lane < ((int)blockDim.x >> 5) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            out[(long long)row * out_dim + out_idx] = __float2half_rn(sum * scale[out_idx]);
        }
    }
}

extern "C" __global__ void dial_w8a16_prefill_f16(
    const half* x,
    const signed char* weight,
    const float* scale,
    half* out,
    int rows,
    int in_dim,
    int out_dim) {
    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;
    const int out_idx = (int)blockIdx.x * 16 + tx;
    const int row = (int)blockIdx.y * 16 + ty;
    __shared__ half x_tile[16][16];
    __shared__ signed char w_tile[16][16];
    float sum = 0.0f;

    for (int k0 = 0; k0 < in_dim; k0 += 16) {
        const int k = k0 + tx;
        x_tile[ty][tx] = (row < rows && k < in_dim)
            ? x[(long long)row * in_dim + k]
            : __float2half(0.0f);

        const int load_out = (int)blockIdx.x * 16 + ty;
        w_tile[ty][tx] = (load_out < out_dim && k < in_dim)
            ? weight[(long long)load_out * in_dim + k]
            : 0;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            sum += __half2float(x_tile[ty][kk]) * (float)w_tile[tx][kk];
        }
        __syncthreads();
    }

    if (row < rows && out_idx < out_dim) {
        out[(long long)row * out_dim + out_idx] = __float2half_rn(sum * scale[out_idx]);
    }
}
"#;

#[cfg(feature = "cuda")]
const W8A16_MODULE: &str = "dial_w8a16";
#[cfg(feature = "cuda")]
const W8A16_DECODE_KERNEL: &str = "dial_w8a16_decode_f16";
#[cfg(feature = "cuda")]
const W8A16_PREFILL_KERNEL: &str = "dial_w8a16_prefill_f16";
#[cfg(feature = "cuda")]
static W8A16_PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();

#[cfg(feature = "cuda")]
fn ensure_w8a16_kernels(device: &candle_core::cuda::CudaDevice) -> candle_core::Result<()> {
    use candle_core::cuda::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

    if let Some(ptx) = W8A16_PTX.get() {
        device.get_or_load_custom_func(W8A16_DECODE_KERNEL, W8A16_MODULE, ptx)?;
        device.get_or_load_custom_func(W8A16_PREFILL_KERNEL, W8A16_MODULE, ptx)?;
        return Ok(());
    }

    static COMPILE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = COMPILE_LOCK
        .lock()
        .map_err(|_| candle_core::Error::Msg("worker W8A16 compile lock poisoned".into()))?;
    if let Some(ptx) = W8A16_PTX.get() {
        device.get_or_load_custom_func(W8A16_DECODE_KERNEL, W8A16_MODULE, ptx)?;
        device.get_or_load_custom_func(W8A16_PREFILL_KERNEL, W8A16_MODULE, ptx)?;
        return Ok(());
    }

    let mut include_paths = Vec::new();
    for root in [
        std::env::var("CUDA_HOME").ok(),
        std::env::var("CUDA_PATH").ok(),
        Some("/usr/local/cuda".to_string()),
        Some("/opt/cuda".to_string()),
    ]
    .into_iter()
    .flatten()
    {
        let include = std::path::Path::new(&root).join("include");
        if include.join("cuda_fp16.h").is_file() {
            let include = include.to_string_lossy().into_owned();
            if !include_paths.contains(&include) {
                include_paths.push(include);
            }
        }
    }
    let options = CompileOptions {
        use_fast_math: Some(true),
        include_paths,
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(W8A16_CUDA_SOURCE, options)
        .map_err(|err| candle_core::Error::Msg(format!("NVRTC W8A16 compile failed: {err}")))?
        .to_src();
    let _ = W8A16_PTX.set(ptx);
    let ptx = W8A16_PTX
        .get()
        .ok_or_else(|| candle_core::Error::Msg("W8A16 PTX cache is empty".into()))?;
    device.get_or_load_custom_func(W8A16_DECODE_KERNEL, W8A16_MODULE, ptx)?;
    device.get_or_load_custom_func(W8A16_PREFILL_KERNEL, W8A16_MODULE, ptx)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaW8A16Linear {
    qweight: candle_core::cuda::cudarc::driver::CudaSlice<i8>,
    scales: candle_core::cuda::cudarc::driver::CudaSlice<f32>,
    in_dim: usize,
    out_dim: usize,
}

#[cfg(feature = "cuda")]
impl CudaW8A16Linear {
    fn from_linear(name: &str, linear: &candle_nn::Linear) -> candle_core::Result<Self> {
        use rayon::prelude::*;

        if linear.bias().is_some() {
            candle_core::bail!("W8A16 only supports bias-free linear layers");
        }
        let weight = if linear.weight().is_contiguous() {
            linear.weight().clone()
        } else {
            linear.weight().contiguous()?
        };
        if weight.dtype() != DType::F16 {
            candle_core::bail!("expected F16 weight, got {:?}", weight.dtype());
        }
        let Device::Cuda(cuda) = weight.device() else {
            candle_core::bail!("W8A16 CUDA linear requires a CUDA weight tensor");
        };
        let cuda = cuda.clone();
        ensure_w8a16_kernels(&cuda)?;

        let (out_dim, in_dim) = weight.dims2()?;
        let host_weight = weight
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut qweight = vec![0i8; host_weight.len()];
        let mut scales = vec![1.0f32; out_dim];
        qweight
            .par_chunks_mut(in_dim)
            .zip(scales.par_iter_mut())
            .zip(host_weight.par_chunks(in_dim))
            .for_each(|((qrow, scale), row)| {
                let max_abs = row.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
                *scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
                let inv_scale = 1.0 / *scale;
                for (dst, value) in qrow.iter_mut().zip(row.iter()) {
                    *dst = (value * inv_scale).round().clamp(-127.0, 127.0) as i8;
                }
            });
        drop(host_weight);

        let stream = cuda.cuda_stream();
        let qweight = stream.clone_htod(&qweight).map_err(|err| {
            candle_core::Error::Msg(format!("uploading W8A16 weight failed: {err}"))
        })?;
        let scales = stream.clone_htod(&scales).map_err(|err| {
            candle_core::Error::Msg(format!("uploading W8A16 scales failed: {err}"))
        })?;
        let bytes = out_dim * in_dim + out_dim * std::mem::size_of::<f32>();
        let count = WORKER_W8A16_LINEAR_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        let total = WORKER_W8A16_WEIGHT_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
        log::info!(
            "worker W8A16 linear enabled: {name} shape=[{out_dim},{in_dim}] weight={:.1} MiB total_linears={count} total={:.1} MiB",
            bytes as f64 / 1048576.0,
            total as f64 / 1048576.0
        );

        Ok(Self {
            qweight,
            scales,
            in_dim,
            out_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if x.dtype() != DType::F16 {
            candle_core::bail!("W8A16 expects F16 activation, got {:?}", x.dtype());
        }
        if x.dims().last().copied() != Some(self.in_dim) {
            candle_core::bail!(
                "W8A16 input mismatch: expected last dim {}, got {:?}",
                self.in_dim,
                x.dims()
            );
        }
        let x = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        x.apply_op1_no_bwd(self)
    }
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for CudaW8A16Linear {
    fn name(&self) -> &'static str {
        "dial-w8a16-linear"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("W8A16 linear is only available on CUDA")
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use half::f16;

        if !layout.is_contiguous() {
            candle_core::bail!("W8A16 activation must be contiguous")
        }
        let elem_count = layout.shape().elem_count();
        if elem_count % self.in_dim != 0 {
            candle_core::bail!("W8A16 activation element count is not divisible by in_dim")
        }
        let rows = elem_count / self.in_dim;
        let input_all = storage.as_cuda_slice::<f16>()?;
        let input = input_all.slice(layout.start_offset()..layout.start_offset() + elem_count);
        let mut output =
            unsafe { storage.device.alloc::<f16>(rows * self.out_dim) }.map_err(|err| {
                candle_core::Error::Msg(format!("allocating W8A16 output failed: {err}"))
            })?;

        let (kernel_name, config) = if rows <= 4 {
            (
                W8A16_DECODE_KERNEL,
                LaunchConfig {
                    grid_dim: (self.out_dim as u32, rows as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
            )
        } else {
            (
                W8A16_PREFILL_KERNEL,
                LaunchConfig {
                    grid_dim: (
                        self.out_dim.div_ceil(16) as u32,
                        rows.div_ceil(16) as u32,
                        1,
                    ),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                },
            )
        };
        let ptx = W8A16_PTX
            .get()
            .ok_or_else(|| candle_core::Error::Msg("W8A16 PTX is not initialized".into()))?;
        let function = storage
            .device
            .get_or_load_custom_func(kernel_name, W8A16_MODULE, ptx)?;
        let mut builder = function.builder();
        builder.arg(&input);
        builder.arg(&self.qweight);
        builder.arg(&self.scales);
        builder.arg(&mut output);
        let rows = rows as i32;
        let in_dim = self.in_dim as i32;
        let out_dim = self.out_dim as i32;
        builder.arg(&rows);
        builder.arg(&in_dim);
        builder.arg(&out_dim);
        unsafe { builder.launch(config) }.map_err(|err| {
            candle_core::Error::Msg(format!("launching W8A16 kernel failed: {err}"))
        })?;

        let mut dims = layout.shape().dims().to_vec();
        *dims.last_mut().expect("W8A16 input rank checked") = self.out_dim;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, storage.device.clone()),
            candle_core::Shape::from(dims),
        ))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn cuda_w8a16_matches_dense_for_decode_and_prefill() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let in_dim = 64;
        let out_dim = 37;
        let weights = (0..out_dim * in_dim)
            .map(|idx| ((idx as f32 * 0.037).sin() * 0.25) + ((idx % 11) as f32 - 5.0) * 0.002)
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(weights, (out_dim, in_dim), &device)?.to_dtype(DType::F16)?;
        let dense = candle_nn::Linear::new(weight, None);
        let w8a16 = CudaW8A16Linear::from_linear("test.linear", &dense)?;

        for rows in [1usize, 9usize] {
            let input = (0..rows * in_dim)
                .map(|idx| (idx as f32 * 0.071).cos() * 0.5)
                .collect::<Vec<_>>();
            let input =
                Tensor::from_vec(input, (1, rows, in_dim), &device)?.to_dtype(DType::F16)?;
            let expected = dense.forward(&input)?.to_dtype(DType::F32)?.flatten_all()?;
            let actual = w8a16.forward(&input)?.to_dtype(DType::F32)?.flatten_all()?;
            let expected = expected.to_vec1::<f32>()?;
            let actual = actual.to_vec1::<f32>()?;
            let max_abs = expected
                .iter()
                .zip(actual.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs < 0.04,
                "rows={rows} W8A16 max_abs={max_abs} exceeds tolerance"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "CUDA microbenchmark; run explicitly on the target GPU"]
    fn benchmark_cuda_w8a16_decode() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let in_dim = 4096;
        let out_dim = 4096;
        let weights = (0..out_dim * in_dim)
            .map(|idx| ((idx as f32 * 0.013).sin() * 0.125) as f32)
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(weights, (out_dim, in_dim), &device)?.to_dtype(DType::F16)?;
        let dense = candle_nn::Linear::new(weight, None);
        let w8a16 = CudaW8A16Linear::from_linear("bench.linear", &dense)?;
        let input = (0..in_dim)
            .map(|idx| (idx as f32 * 0.017).cos() * 0.5)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input, (1, 1, in_dim), &device)?.to_dtype(DType::F16)?;

        for _ in 0..10 {
            let _ = dense.forward(&input)?;
            let _ = w8a16.forward(&input)?;
        }
        device.synchronize()?;
        let iterations = 100;
        let dense_start = Instant::now();
        for _ in 0..iterations {
            let _ = dense.forward(&input)?;
        }
        device.synchronize()?;
        let dense_ms = dense_start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        let w8a16_start = Instant::now();
        for _ in 0..iterations {
            let _ = w8a16.forward(&input)?;
        }
        device.synchronize()?;
        let w8a16_ms = w8a16_start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        println!(
            "decode linear [1,1,{in_dim}]x[{out_dim},{in_dim}]: dense_f16={dense_ms:.4}ms w8a16={w8a16_ms:.4}ms speedup={:.3}x",
            dense_ms / w8a16_ms
        );
        Ok(())
    }
}
