use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use safetensors::View;
use serde::{Deserialize, Serialize};
use std::{
    env,
    sync::OnceLock,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    sync::Mutex,
    time::{sleep_until, Instant as TokioInstant},
};

struct TransferLimiter {
    bytes_per_sec: f64,
    chunk_bytes: usize,
    next_write_at: Mutex<Instant>,
}

impl TransferLimiter {
    fn from_env() -> Option<Self> {
        let bytes_per_sec = if let Some(mbps) = parse_env_f64("SPM_TRANSFER_LIMIT_MBPS") {
            mbps * 1_000_000.0 / 8.0
        } else if let Some(kbps) = parse_env_f64("SPM_TRANSFER_LIMIT_KBPS") {
            kbps * 1_000.0 / 8.0
        } else if let Some(bytes) = parse_env_f64("SPM_TRANSFER_LIMIT_BYTES_PER_SEC") {
            bytes
        } else {
            return None;
        };

        if !bytes_per_sec.is_finite() || bytes_per_sec <= 0.0 {
            return None;
        }

        let chunk_bytes = parse_env_f64("SPM_TRANSFER_LIMIT_CHUNK_BYTES")
            .filter(|value| value.is_finite() && *value >= 1.0)
            .map(|value| value as usize)
            .unwrap_or_else(|| ((bytes_per_sec / 10.0).round() as usize).clamp(1024, 64 * 1024));

        log::info!(
            "spm transfer limit enabled: {:.3} Mbit/s, chunk={}B",
            bytes_per_sec * 8.0 / 1_000_000.0,
            chunk_bytes
        );

        Some(Self {
            bytes_per_sec,
            chunk_bytes,
            next_write_at: Mutex::new(Instant::now()),
        })
    }

    async fn wait_for_slot(&self, bytes: usize) {
        let now = Instant::now();
        let wait_until = {
            let mut next_write_at = self.next_write_at.lock().await;
            let wait_until = (*next_write_at).max(now);
            let transfer_time = Duration::from_secs_f64(bytes as f64 / self.bytes_per_sec);
            *next_write_at = wait_until + transfer_time;
            wait_until
        };

        if wait_until > now {
            sleep_until(TokioInstant::from_std(wait_until)).await;
        }
    }
}

fn parse_env_f64(name: &str) -> Option<f64> {
    env::var(name).ok()?.trim().parse::<f64>().ok()
}

fn transfer_limiter() -> Option<&'static TransferLimiter> {
    static LIMITER: OnceLock<Option<TransferLimiter>> = OnceLock::new();
    LIMITER.get_or_init(TransferLimiter::from_env).as_ref()
}

async fn write_all_limited<W>(writer: &mut W, data: &[u8]) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let Some(limiter) = transfer_limiter() else {
        writer.write_all(data).await?;
        return Ok(());
    };

    for chunk in data.chunks(limiter.chunk_bytes) {
        limiter.wait_for_slot(chunk.len()).await;
        writer.write_all(chunk).await?;
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum RawDType {
    F16,
    BF16,
    F32,
    F64,
    U8,
    U32,
    I64,
}

impl RawDType {
    fn from_dtype(dtype: DType) -> Result<Self> {
        Ok(match dtype {
            DType::F16 => Self::F16,
            DType::BF16 => Self::BF16,
            DType::F32 => Self::F32,
            DType::F64 => Self::F64,
            DType::U8 => Self::U8,
            DType::U32 => Self::U32,
            DType::I64 => Self::I64,
            other => anyhow::bail!("unsupported tensor dtype in SPM protocol: {other:?}"),
        })
    }

    fn to_dtype(self) -> DType {
        match self {
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
            Self::F32 => DType::F32,
            Self::F64 => DType::F64,
            Self::U8 => DType::U8,
            Self::U32 => DType::U32,
            Self::I64 => DType::I64,
        }
    }
}

/// 该结构体表示spm协议中的张量.
#[derive(Serialize, Debug, Deserialize)]
pub struct RawTensor {
    /// Tensor data.
    pub data: Vec<u8>,
    /// The data type as string.
    pub dtype: RawDType,
    /// The tensor shape.
    pub shape: Vec<usize>,
}

impl RawTensor {
    /// 这里写RawTensor的方法。.
    pub fn from_tensor(x: &Tensor) -> Result<Self> {
        // 获取张量的原始二进制数据。
        let data: Vec<u8> = x.data().to_vec();
        // 获取张量的数据类型并转换为紧凑枚举，避免每次传字符串。
        let dtype = RawDType::from_dtype(x.dtype())?;
        // 把 shape 变成可序列化的维度数组。
        let shape = x.shape().clone().into_dims();
        // 用上面提取的 3 个字段构造 RawTensor 实例。
        Ok(Self { data, dtype, shape })
    }

    /// 把接收到的RawTensor转换回Tensor，供后续计算使用。
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let dtype = self.dtype.to_dtype();
        Tensor::from_raw_buffer(&self.data, dtype, &self.shape, device).map_err(|e| anyhow!(e))
    }
}

#[derive(Serialize, Debug, Deserialize)]
pub struct CompactBatch {
    pub first_layer_name: String,
    pub index_pos: usize,
    pub first_block_idx: usize,
    pub num_layers: usize,
}

#[derive(Serialize, Debug, Deserialize)]
pub struct CompactRangeBatch {
    pub index_pos: usize,
    pub first_block_idx: usize,
    pub num_layers: usize,
}

/// Stateless sampling inputs supplied by the Master for one generated token.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingRequest {
    pub seed: u64,
    pub sample_index: usize,
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_context: Vec<u32>,
}

/// 工作节点的诊断信息
#[derive(Serialize, Debug, Default, Deserialize)]
pub struct WorkerInfo {
    /// 通信协议版本.
    pub version: String,
    /// Tensor数据类型.
    pub dtype: String,
    /// 操作系统.
    pub os: String,
    /// 架构.
    pub arch: String,
    /// 设备.
    pub device: String,
    /// 多GPU时的卡号.
    pub device_idx: usize,
    /// 延迟.
    pub latency: u128,
    /// One-minute load normalized by the available CPU count.
    pub cpu_usage_percent: Option<f32>,
    /// Used memory as a percentage of total memory.
    pub memory_usage_percent: Option<f32>,
    /// NPU load reported by the RKNN kernel driver when available.
    pub npu_usage_percent: Option<f32>,
    /// Highest readable thermal-zone temperature in Celsius.
    pub temperature_c: Option<f32>,
    /// Board model reported by device tree.
    pub system_model: Option<String>,
}

impl WorkerInfo {
    pub fn supports(&self, capability: &str) -> bool {
        self.version
            .split('|')
            .skip(1)
            .any(|item| item == capability)
    }
}

/// 这是SPM分布式通信协议的消息类型.
#[derive(Serialize, Debug, Deserialize)]
/// 主从之间只能发送这几种消息，分别是Hello、WorkerInfo、SingleOp、Batch和Tensor。
pub enum Message {
    /// Hello握手消息.
    Hello,
    /// Worker把自己的信息发给Master.
    WorkerInfo(WorkerInfo),
    /// 单算子推理任务（发任务）.
    SingleOp {
        // 层名字
        layer_name: String,
        // 张量数据
        x: RawTensor,
        // 位置
        index_pos: usize,
        // 块序号
        block_idx: usize,
        sampling: Option<SamplingRequest>,
    },
    /// 批量推理任务
    Batch {
        x: RawTensor,
        batch: Vec<(String, usize, usize)>,
        sampling: Option<SamplingRequest>,
    },
    /// 连续层批量推理任务，避免每个 token 重复传几十个层名字。
    CompactBatch {
        x: RawTensor,
        batch: CompactBatch,
        sampling: Option<SamplingRequest>,
    },
    /// 连续 block range 批量推理任务，只传 block 起点和层数，进一步减少字符串元数据。
    CompactRangeBatch {
        x: RawTensor,
        batch: CompactRangeBatch,
        sampling: Option<SamplingRequest>,
    },
    /// Worker->Master回传计算结果.
    Tensor { x: RawTensor, compute_us: u64 },
    /// Worker-side sampling fast path: return only the sampled token id.
    ///
    /// This avoids wrapping a four-byte token in Tensor -> RawTensor and then
    /// reconstructing a Tensor on the wire path.
    SampledToken { token: u32, compute_us: u64 },
}
/// 给Message这个枚举，实现两个实用函数。
impl Message {
    /// 创建任务消息Single_Op.
    pub fn single_op(
        layer_name: &str,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        sampling: Option<SamplingRequest>,
    ) -> Self {
        // 把 &str 转成 String（所有权转移，网络传输必备）。
        let layer_name = layer_name.to_owned();
        // 把 Tensor 转换成 RawTensor，方便序列化和网络传输。
        let x = RawTensor::from_tensor(x).expect("unsupported tensor dtype for spm single_op");
        // 构建并返回 Message::SingleOp。
        Self::SingleOp {
            layer_name,
            x,
            index_pos,
            block_idx,
            sampling,
        }
    }

    /// 创建计算结果消息
    pub fn from_tensor(x: &Tensor) -> Self {
        Self::Tensor {
            x: RawTensor::from_tensor(x).expect("unsupported tensor dtype for spm tensor"),
            compute_us: 0,
        }
    }

    /// 创建一个带计算耗时的Tensor结果消息.
    pub fn from_tensor_with_compute(x: &Tensor, compute_us: u64) -> Self {
        Self::Tensor {
            x: RawTensor::from_tensor(x).expect("unsupported tensor dtype for spm tensor"),
            compute_us,
        }
    }

    pub fn from_raw_tensor_with_compute(x: RawTensor, compute_us: u64) -> Self {
        Self::Tensor { x, compute_us }
    }

    pub fn sampled_token_with_compute(token: u32, compute_us: u64) -> Self {
        Self::SampledToken { token, compute_us }
    }

    /// 创建批量计算任务消息
    pub fn from_batch(
        x: &Tensor,
        batch: Vec<(String, usize, usize)>,
        sampling: Option<SamplingRequest>,
    ) -> Self {
        Self::Batch {
            x: RawTensor::from_tensor(x).expect("unsupported tensor dtype for spm batch"),
            batch,
            sampling,
        }
    }

    pub fn from_compact_batch(
        x: &Tensor,
        batch: CompactBatch,
        sampling: Option<SamplingRequest>,
    ) -> Self {
        Self::CompactBatch {
            x: RawTensor::from_tensor(x).expect("unsupported tensor dtype for spm compact batch"),
            batch,
            sampling,
        }
    }

    pub fn from_compact_range_batch(
        x: &Tensor,
        batch: CompactRangeBatch,
        sampling: Option<SamplingRequest>,
    ) -> Self {
        Self::CompactRangeBatch {
            x: RawTensor::from_tensor(x)
                .expect("unsupported tensor dtype for spm compact range batch"),
            batch,
            sampling,
        }
    }

    /// 把Message消息->二进制字节数组.
    fn to_bytes(&self) -> Result<Vec<u8>> {
        bitcode::serialize(self).map_err(|e| anyhow!(e))
    }

    /// 收到网络二进制->还原成Message消息.
    fn from_bytes(raw: &[u8]) -> Result<Self> {
        bitcode::deserialize(raw).map_err(|e| anyhow!(e))
    }

    /// 从网络流里读取一条完整的消息，并复用调用方提供的缓冲区。
    ///
    /// Persistent Master/Worker connections invoke this once per generated token,
    /// so reusing the allocation avoids a fresh Vec allocation on every RPC.
    pub async fn from_reader_with_buffer<R>(
        reader: &mut R,
        buffer: &mut Vec<u8>,
    ) -> Result<(usize, Self)>
    where
        R: AsyncReadExt + Unpin,
    {
        // 读魔数，校验协议。
        let magic = reader.read_u32().await?;
        if magic != super::PROTO_MAGIC {
            return Err(anyhow!("invalid magic value: {magic}"));
        }
        // 读消息长度。
        let req_size = reader.read_u32().await?;
        if req_size > super::MESSAGE_MAX_SIZE {
            return Err(anyhow!("request size {req_size} > MESSAGE_MAX_SIZE"));
        }

        let req_size = req_size as usize;
        buffer.resize(req_size, 0);
        reader.read_exact(&mut buffer[..req_size]).await?;
        Ok((req_size, Self::from_bytes(&buffer[..req_size])?))
    }

    /// Convenience wrapper for one-shot connections.
    pub async fn from_reader<R>(reader: &mut R) -> Result<(usize, Self)>
    where
        R: AsyncReadExt + Unpin,
    {
        let mut buffer = Vec::new();
        Self::from_reader_with_buffer(reader, &mut buffer).await
    }

    /// 把消息序列化成二进制->按照[魔数+长度+数据]的协议格式->发送到网络流
    pub async fn to_writer<W>(&self, writer: &mut W) -> Result<usize>
    where
        W: AsyncWriteExt + Unpin,
    {
        // 把消息序列化成二进制。
        let req = self.to_bytes()?;
        let req_size = req.len() as u32;
        // 安全校验。
        if req_size > super::MESSAGE_MAX_SIZE {
            return Err(anyhow!("request size {req_size} > MESSAGE_MAX_SIZE"));
        }
        // 只合并 8 字节 header，避免大 tensor 消息为了单次写入又拷贝一整份 body。
        let mut header = [0_u8; 8];
        header[..4].copy_from_slice(&super::PROTO_MAGIC.to_be_bytes());
        header[4..].copy_from_slice(&req_size.to_be_bytes());
        write_all_limited(writer, &header).await?;
        write_all_limited(writer, &req).await?;
        // 返回总字节数。
        Ok(8 + req.len())
    }
}


#[cfg(test)]
mod tests {
    use super::Message;

    #[test]
    fn sampled_token_roundtrip() {
        let encoded = Message::sampled_token_with_compute(12345, 6789)
            .to_bytes()
            .unwrap();
        match Message::from_bytes(&encoded).unwrap() {
            Message::SampledToken { token, compute_us } => {
                assert_eq!(token, 12345);
                assert_eq!(compute_us, 6789);
            }
            other => panic!("unexpected decoded message: {other:?}"),
        }
    }
}
