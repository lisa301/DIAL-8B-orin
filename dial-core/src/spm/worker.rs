use std::{
    collections::HashMap,
    fs,
    net::SocketAddr,
    process::Command,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use super::{Context, Forwarder, Message, WorkerInfo};
use crate::models::{
    llama3::{Cache, Config},
    Generator,
};

use anyhow::Result;
use candle_core::{DType, Device};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

/// 每执行5次推理/操作，就输出一次worker统计日志。
const NUM_OPS_TO_STATS: usize = 5;

#[derive(Default)]
pub(super) struct WorkerMetrics {
    pub(super) cpu_usage_percent: Option<f32>,
    pub(super) memory_usage_percent: Option<f32>,
    pub(super) npu_usage_percent: Option<f32>,
    pub(super) temperature_c: Option<f32>,
    pub(super) system_model: Option<String>,
}

#[derive(Clone, Copy)]
struct CpuTimes {
    total: u64,
    idle: u64,
}

static PREVIOUS_CPU_TIMES: OnceLock<Mutex<Option<CpuTimes>>> = OnceLock::new();

fn read_cpu_times() -> Option<CpuTimes> {
    let text = fs::read_to_string("/proc/stat").ok()?;
    let mut fields = text.lines().next()?.split_whitespace();
    if fields.next()? != "cpu" {
        return None;
    }
    let values = fields
        .filter_map(|value| value.parse::<u64>().ok())
        .collect::<Vec<_>>();
    if values.len() < 4 {
        return None;
    }
    let idle = values[3].saturating_add(values.get(4).copied().unwrap_or(0));
    Some(CpuTimes {
        total: values.iter().copied().sum(),
        idle,
    })
}

fn cpu_usage_percent() -> Option<f32> {
    let current = read_cpu_times()?;
    let sample = PREVIOUS_CPU_TIMES.get_or_init(|| Mutex::new(None));
    let mut previous = sample.lock().ok()?;
    let result = previous.and_then(|old| {
        let total_delta = current.total.saturating_sub(old.total);
        let idle_delta = current.idle.saturating_sub(old.idle);
        (total_delta > 0).then(|| {
            ((total_delta.saturating_sub(idle_delta)) as f32 / total_delta as f32 * 100.0)
                .clamp(0.0, 100.0)
        })
    });
    *previous = Some(current);
    result
}

fn percentage_values(text: &str) -> Vec<f32> {
    text.split('%')
        .filter_map(|segment| {
            segment
                .split(|character: char| !character.is_ascii_digit() && character != '.')
                .filter(|value| !value.is_empty())
                .next_back()?
                .parse::<f32>()
                .ok()
        })
        .filter(|value| *value >= 0.0 && *value <= 100.0)
        .collect()
}

fn read_scaled_percentage(path: &str) -> Option<f32> {
    let raw = fs::read_to_string(path).ok()?;
    let value = raw
        .trim()
        .split(|character: char| !character.is_ascii_digit() && character != '.')
        .find_map(|value| value.parse::<f32>().ok())?;
    let scaled = if value > 100.0 { value / 10.0 } else { value };
    (scaled >= 0.0).then(|| scaled.clamp(0.0, 100.0))
}

fn value_after_marker(text: &str, marker: &str) -> Option<f32> {
    let start = text.rfind(marker)? + marker.len();
    text[start..]
        .split(|character: char| !character.is_ascii_digit() && character != '.')
        .find_map(|value| value.parse::<f32>().ok())
}

fn tegrastats_metrics() -> (Option<f32>, Option<f32>) {
    // tegrastats runs continuously. Coreutils timeout bounds one short sample
    // so the topology endpoint cannot leave a monitoring process behind.
    let output = Command::new("timeout")
        .args(["0.6s", "tegrastats", "--interval", "100"])
        .output();
    let Ok(output) = output else {
        return (None, None);
    };
    let text = String::from_utf8_lossy(&output.stdout);
    if text.trim().is_empty() {
        return (None, None);
    }
    let usage =
        value_after_marker(&text, "GR3D_FREQ ").or_else(|| value_after_marker(&text, "GR3D_PCI "));
    let temperature = value_after_marker(&text, "GPU@");
    (usage, temperature)
}

fn nvidia_smi_metrics(device_idx: usize) -> (Option<f32>, Option<f32>) {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            &device_idx.to_string(),
        ])
        .output();
    let Ok(output) = output else {
        return (None, None);
    };
    if !output.status.success() {
        return (None, None);
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut values = text.trim().split(',').map(str::trim);
    let usage = values.next().and_then(|value| value.parse::<f32>().ok());
    let temperature = values.next().and_then(|value| value.parse::<f32>().ok());
    (usage, temperature)
}

pub(super) fn collect_worker_metrics(device_idx: usize) -> WorkerMetrics {
    let cpu_usage_percent = cpu_usage_percent().or_else(|| {
        fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|text| text.split_whitespace().next()?.parse::<f32>().ok())
            .map(|load| {
                let cpu_count = std::thread::available_parallelism()
                    .map(|count| count.get())
                    .unwrap_or(1) as f32;
                (load / cpu_count * 100.0).clamp(0.0, 100.0)
            })
    });

    let memory_usage_percent = fs::read_to_string("/proc/meminfo").ok().and_then(|text| {
        let mut total_kib = None;
        let mut available_kib = None;
        for line in text.lines() {
            let mut fields = line.split_whitespace();
            match fields.next() {
                Some("MemTotal:") => total_kib = fields.next()?.parse::<f32>().ok(),
                Some("MemAvailable:") => available_kib = fields.next()?.parse::<f32>().ok(),
                _ => {}
            }
        }
        let total = total_kib?;
        let available = available_kib?;
        (total > 0.0).then(|| ((total - available) / total * 100.0).clamp(0.0, 100.0))
    });

    let npu_usage = fs::read_to_string("/sys/kernel/debug/rknpu/load")
        .ok()
        .and_then(|text| percentage_values(&text).into_iter().reduce(f32::max));
    let jetson_gpu_usage = [
        "/sys/devices/gpu.0/load",
        "/sys/devices/platform/17000000.gpu/load",
        "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/load",
        "/sys/devices/platform/17000000.ga10b/devfreq/17000000.ga10b/load",
        "/sys/class/devfreq/17000000.gpu/load",
        "/sys/class/devfreq/17000000.ga10b/load",
    ]
    .iter()
    .find_map(|path| read_scaled_percentage(path));
    let (tegrastats_usage, tegrastats_temperature) = if npu_usage.is_none() {
        tegrastats_metrics()
    } else {
        (None, None)
    };
    let (nvidia_gpu_usage, nvidia_temperature) =
        if npu_usage.is_none() && jetson_gpu_usage.is_none() && tegrastats_usage.is_none() {
            nvidia_smi_metrics(device_idx)
        } else {
            (None, None)
        };
    // Kept under the protocol's original NPU field for compatibility. The UI
    // labels it GPU for CUDA nodes and NPU for RKNN nodes.
    let npu_usage_percent = npu_usage.or_else(|| {
        [jetson_gpu_usage, tegrastats_usage, nvidia_gpu_usage]
            .into_iter()
            .flatten()
            .reduce(f32::max)
    });

    let mut temperatures = Vec::new();
    for index in 0..16 {
        let path = format!("/sys/class/thermal/thermal_zone{index}/temp");
        if let Ok(raw) = fs::read_to_string(path) {
            if let Ok(value) = raw.trim().parse::<f32>() {
                let celsius = if value > 1000.0 {
                    value / 1000.0
                } else {
                    value
                };
                if (-40.0..=150.0).contains(&celsius) {
                    temperatures.push(celsius);
                }
            }
        }
    }
    if let Some(value) = tegrastats_temperature.or(nvidia_temperature) {
        temperatures.push(value);
    }
    let temperature_c = temperatures.into_iter().reduce(f32::max);

    let system_model = fs::read("/proc/device-tree/model").ok().and_then(|bytes| {
        let value = String::from_utf8_lossy(&bytes)
            .trim_matches(char::from(0))
            .trim()
            .to_string();
        (!value.is_empty()).then_some(value)
    });

    WorkerMetrics {
        cpu_usage_percent,
        memory_usage_percent,
        npu_usage_percent,
        temperature_c,
        system_model,
    }
}

/// 一个独立的工作节点.
#[derive(Clone)]
struct WorkerContext<F> {
    device: Device,
    device_idx: usize,
    dtype: DType,
    blocks: Arc<HashMap<String, Box<F>>>,
    block_names_by_idx: Arc<Vec<Option<String>>>,
    cache: Cache,
}
/// AI推理工作节点的方法，用于向master报告自己的状态和性能指标。
impl<F: Forwarder> WorkerContext<F> {
    ///创建WorkInfo结构，发送给主节点（master）.
    fn to_info(&self, latency: u128) -> WorkerInfo {
        let metrics = collect_worker_metrics(self.device_idx);
        let mut version = format!(
            "{}|compact_batch_v1|compact_range_batch_v1",
            env!("CARGO_PKG_VERSION")
        );
        if crate::models::qwen3_vl::worker_gguf_output_head_enabled() {
            version.push_str("|gguf_output_head_v1");
        }
        if crate::models::qwen3_vl::worker_gguf_sample_token_enabled() {
            version.push_str("|gguf_sample_token_v1");
        }
        WorkerInfo {
            version,
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            device: if self.device.is_cuda() {
                "cuda".to_string()
            } else if self.device.is_metal() {
                "metal".to_string()
            } else {
                "cpu".to_string()
            },
            device_idx: self.device_idx,
            latency,
            dtype: format!("{:?}", self.dtype),
            cpu_usage_percent: metrics.cpu_usage_percent,
            memory_usage_percent: metrics.memory_usage_percent,
            npu_usage_percent: metrics.npu_usage_percent,
            temperature_c: metrics.temperature_c,
            system_model: metrics.system_model,
        }
    }

    fn expand_compact_range_batch(
        &self,
        index_pos: usize,
        first_block_idx: usize,
        num_layers: usize,
    ) -> Result<Vec<(String, usize, usize)>> {
        let mut ops = Vec::with_capacity(num_layers);
        for offset in 0..num_layers {
            let block_idx = first_block_idx + offset;
            let Some(Some(layer_name)) = self.block_names_by_idx.get(block_idx) else {
                bail!("compact range references missing local block index {block_idx}");
            };
            ops.push((layer_name.clone(), index_pos, block_idx));
        }
        Ok(ops)
    }

    /// Create a copy of self with new kv-cache.
    /// 复制自己+换新自己->给客户端使用
    fn get_client_context(&self) -> Self {
        WorkerContext {
            device: self.device.clone(),
            device_idx: self.device_idx,
            dtype: self.dtype,
            blocks: self.blocks.clone(),
            block_names_by_idx: self.block_names_by_idx.clone(),
            cache: self.cache.as_new(),
        }
    }
}

/// 定义结构体——工作节点.
pub struct Worker<G: Generator> {
    listener: TcpListener,
    context: Arc<WorkerContext<G::Shardable>>,
}
/// 给Worker这个结构体实现方法
impl<G: Generator + 'static> Worker<G> {
    /// 判断是否开启传输跟踪日志
    fn transfer_trace_enabled() -> bool {
        matches!(
            std::env::var("SPM_TRACE_TRANSFER").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        )
    }
    /// 生成操作统计摘要
    fn ops_summary(ops: &[(String, usize, usize)]) -> String {
        let first = ops.first().map(|(name, _, _)| name.as_str()).unwrap_or("-");
        let last = ops.last().map(|(name, _, _)| name.as_str()).unwrap_or("-");
        format!("ops={} first={} last={}", ops.len(), first, last)
    }

    fn expand_compact_batch(
        first_layer_name: String,
        index_pos: usize,
        first_block_idx: usize,
        num_layers: usize,
    ) -> Result<Vec<(String, usize, usize)>> {
        let split_at = first_layer_name
            .rfind(|c: char| !c.is_ascii_digit())
            .map(|idx| idx + 1)
            .ok_or_else(|| anyhow!("compact batch layer name has no numeric suffix"))?;
        if split_at >= first_layer_name.len() {
            bail!("compact batch layer name has empty numeric suffix");
        }
        let (prefix, start) = first_layer_name.split_at(split_at);
        let start = start.parse::<usize>()?;
        let mut ops = Vec::with_capacity(num_layers);
        for offset in 0..num_layers {
            ops.push((
                format!("{}{}", prefix, start + offset),
                index_pos,
                first_block_idx + offset,
            ));
        }
        Ok(ops)
    }

    fn layer_index_from_name(layer_name: &str) -> Option<usize> {
        let split_at = layer_name.rfind(|c: char| !c.is_ascii_digit())? + 1;
        layer_name.get(split_at..)?.parse::<usize>().ok()
    }

    fn expand_layer_spec(spec: &str) -> Result<Vec<String>> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Ok(Vec::new());
        }
        let Some(dash_idx) = spec.rfind('-') else {
            return Ok(vec![spec.to_string()]);
        };
        let Ok(stop) = spec[dash_idx + 1..].parse::<usize>() else {
            return Ok(vec![spec.to_string()]);
        };
        let prefix_with_start = &spec[..dash_idx];
        let Some(split_at) = prefix_with_start
            .rfind(|c: char| !c.is_ascii_digit())
            .map(|idx| idx + 1)
        else {
            return Ok(vec![spec.to_string()]);
        };
        if split_at >= prefix_with_start.len() {
            return Ok(vec![spec.to_string()]);
        }
        let base = &prefix_with_start[..split_at];
        let start = prefix_with_start[split_at..].parse::<usize>()?;
        if stop < start {
            bail!("invalid layer range {spec}: stop < start");
        }
        Ok((start..=stop).map(|idx| format!("{base}{idx}")).collect())
    }

    fn extra_worker_layers(cfg: &Config) -> Result<Vec<String>> {
        if matches!(
            std::env::var("SPM_WORKER_LOAD_ALL_LAYERS").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        ) {
            return Ok((0..cfg.num_hidden_layers)
                .map(|idx| format!("model.language_model.layers.{idx}"))
                .collect());
        }
        let Some(raw) = std::env::var("SPM_WORKER_EXTRA_LAYERS").ok() else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for spec in raw.split(',') {
            out.extend(Self::expand_layer_spec(spec)?);
        }
        Ok(out)
    }

    fn auto_shadow_worker_layers(layer_names: &[String]) -> Vec<String> {
        if !matches!(
            std::env::var("SPM_WORKER_AUTO_SHADOW_LAYERS")
                .ok()
                .as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        ) {
            return Vec::new();
        }

        let Some((first_remote_idx, first_remote_name)) = layer_names
            .iter()
            .filter_map(|name| Self::layer_index_from_name(name).map(|idx| (idx, name)))
            .min_by_key(|(idx, _)| *idx)
        else {
            return Vec::new();
        };
        if first_remote_idx == 0 {
            return Vec::new();
        }

        let Some(split_at) = first_remote_name
            .rfind(|c: char| !c.is_ascii_digit())
            .map(|idx| idx + 1)
        else {
            return Vec::new();
        };
        let base = &first_remote_name[..split_at];
        (0..first_remote_idx)
            .map(|idx| format!("{base}{idx}"))
            .collect()
    }

    /// Create a new Worker from the context.
    /// 整个AI worker最核心的初始化函数！
    /// 创建并启动一个AI推理工作节点的服务，加载模型、绑定端口、准备就绪等待客户端请求。
    pub async fn new(ctx: Context) -> Result<Self> {
        /// 获取工作节点名称
        let worker_name = if let Some(name) = &ctx.args.name {
            name.to_string()
        } else {
            return Err(anyhow!("no --name provided for worker"));
        };
        /// 获取节点的拓扑（模型层分配）
        let worker_topology = if let Some(node) = ctx.topology.get(&worker_name) {
            node
        } else if ctx.args.auto_plan_profile.is_some() {
            return Err(anyhow!(
                "worker '{}' is absent from the automatic plan; do not start this worker or enable it in the profile",
                worker_name
            ));
        } else if !ctx.topology.is_empty() {
            let first = ctx.topology.keys().next().unwrap();
            log::warn!(
                "topology for worker name '{}' not found, using '{}'",
                &worker_name,
                first
            );
            ctx.topology.get(first).unwrap()
        } else {
            return Err(anyhow!(
                "could not find topology for {worker_name} and topology file is empty"
            ));
        };
        let mut layer_names = worker_topology.layers.clone();
        for layer_name in Self::auto_shadow_worker_layers(&layer_names) {
            if !layer_names.contains(&layer_name) {
                log::info!("worker auto shadow layer enabled: {}", layer_name);
                layer_names.push(layer_name);
            }
        }
        for layer_name in Self::extra_worker_layers(&ctx.config)? {
            if !layer_names.contains(&layer_name) {
                log::info!("worker extra layer enabled: {}", layer_name);
                layer_names.push(layer_name);
            }
        }

        /// 加载模型层（blocks)
        let mut blocks = HashMap::new();
        // 在这显示加载了哪些块
        for block_layer_name in &layer_names {
            log::info!("loading {} ...", &block_layer_name);

            let block = G::Shardable::load(
                block_layer_name.to_string(),
                ctx.var_builder.pp(block_layer_name),
                &ctx.config,
            )?;
            blocks.insert(block_layer_name.to_string(), block);
        }
        let owns_final_layer = layer_names.iter().any(|layer_name| {
            Self::layer_index_from_name(layer_name)
                == Some(ctx.config.num_hidden_layers.saturating_sub(1))
        });
        crate::models::qwen3_vl::load_worker_gguf_output_head(&ctx.config, owns_final_layer)?;
        if let Some((linear_count, weight_bytes)) = crate::models::qwen3_vl::worker_w8a16_summary()
        {
            log::info!(
                "worker W8A16 load complete: layers={} quantized_linears={} weight_memory={:.1} MiB",
                layer_names.len(),
                linear_count,
                weight_bytes as f64 / 1048576.0
            );
        }
        if let Some((linear_count, weight_bytes, prefill_count, prefill_bytes)) =
            crate::models::qwen3_vl::worker_gguf_summary()
        {
            log::info!(
                "worker GGUF load complete: layers={} quantized_linears={} weight_memory={:.1} MiB fp16_prefill_linears={} fp16_prefill_memory={:.1} MiB",
                layer_names.len(),
                linear_count,
                weight_bytes as f64 / 1048576.0,
                prefill_count,
                prefill_bytes as f64 / 1048576.0
            );
        }

        let blocks = Arc::new(blocks);
        let mut block_names_by_idx = vec![None; ctx.config.num_hidden_layers];
        for layer_name in blocks.keys() {
            if let Some(idx) = Self::layer_index_from_name(layer_name) {
                if let Some(slot) = block_names_by_idx.get_mut(idx) {
                    *slot = Some(layer_name.clone());
                }
            }
        }
        /// 绑定TCP端口，启动监听
        let listener = TcpListener::bind(&ctx.args.address).await?;
        /// 打印启动日志
        log::info!(
            "listening on {} (mem:{}) ...",
            &ctx.args.address,
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        let cache = ctx.cache;
        let device = ctx.device;
        let dtype = ctx.dtype;
        let device_idx = ctx.args.device;
        /// 组装Worker 实例
        let context = WorkerContext {
            device,
            device_idx,
            dtype,
            blocks,
            block_names_by_idx: Arc::new(block_names_by_idx),
            cache,
        };
        /// 返回worker实例
        Ok(Self {
            listener,
            context: Arc::new(context),
        })
    }

    /// Read a message from the socket and return elapsed time, message size and message.
    /// 异步网络工具函数，从socket读取消息并返回经过的时间、消息大小和消息。
    async fn read_message_timed<R>(mut socket: R) -> Result<(Duration, usize, Message)>
    where
        // 这个函数能处理任何异步可读的数据流（TCP、管道、文件等）。
        R: AsyncReadExt + Unpin,
    {
        let start = Instant::now();
        // 异步读取消息。
        let (size, message) = Message::from_reader(&mut socket).await?;
        let latency = start.elapsed();

        // 返回结果。
        Ok((latency, size, message))
    }

    /// Write a message to the socket and return the elapsed time with written size.
    /// 异步网络工具函数，将消息写入socket并返回经过的时间与写入的大小。
    async fn write_message_timed<W>(mut socket: W, message: Message) -> Result<(Duration, usize)>
    where
        W: AsyncWriteExt + Unpin,
    {
        let start = Instant::now();
        let size = message.to_writer(&mut socket).await?;
        let latency = start.elapsed();

        Ok((latency, size))
    }

    /// Main loop handling communication with the master.
    /// 这个函数是整个Worker工作节点的开头，专门负责和Master建立连接、握手、验证身份
    /// 是分布式AI服务的安全＋通信入口
    async fn handle_master_client(
        mut socket: TcpStream,
        client: SocketAddr,
        base_context: Arc<WorkerContext<G::Shardable>>,
    ) -> Result<()> {
        if let Err(e) = socket.set_nodelay(true) {
            log::warn!("[{}] failed to set TCP_NODELAY: {}", &client, e);
        }
        // 读取Master发来的第一条消息，必须是Hello握手包
        let (latency, _size, hello) = Self::read_message_timed(&mut socket).await?;
        if !matches!(hello, Message::Hello) {
            return Err(anyhow!(
                "[{}] unpexpected message instead of hello: {:?}",
                &client,
                hello
            ));
        }

        // 发送worker信息给Master
        if let Err(e) = Self::write_message_timed(
            &mut socket,
            Message::WorkerInfo(base_context.to_info(latency.as_millis())),
        )
        .await
        //异步发送
        {
            return Err(anyhow!("[{}] could not send worker info: {:?}", &client, e));
        }

        let mut msg_idx = 0;
        let mut avg_ops = 0;
        let mut avg_write = 0;
        let mut avg_read = 0;
        let mut client_context: Option<WorkerContext<G::Shardable>> = None;

        // 持续读取消息
        while let Ok((read_time, read_size, op_message)) =
            Self::read_message_timed(&mut socket).await
        {
            // Status probes stop after the Hello handshake. Allocate a fresh
            // KV cache only when this connection sends its first inference op.
            let context = client_context.get_or_insert_with(|| base_context.get_client_context());
            let req_start = Instant::now();
            let compute_start = Instant::now();

            /// 记录请求开始时间
            let (x, ops, sampling) = match op_message {
                /// 单操作请求
                Message::SingleOp {
                    layer_name,
                    x,
                    index_pos,
                    block_idx,
                    sampling,
                } => (x, vec![(layer_name, index_pos, block_idx)], sampling),
                /// 批量操作请求
                Message::Batch { x, batch, sampling } => (x, batch, sampling),
                Message::CompactBatch { x, batch, sampling } => (
                    x,
                    Self::expand_compact_batch(
                        batch.first_layer_name,
                        batch.index_pos,
                        batch.first_block_idx,
                        batch.num_layers,
                    )?,
                    sampling,
                ),
                Message::CompactRangeBatch { x, batch, sampling } => (
                    x,
                    context.expand_compact_range_batch(
                        batch.index_pos,
                        batch.first_block_idx,
                        batch.num_layers,
                    )?,
                    sampling,
                ),
                _ => {
                    return Err(anyhow!(
                        "[{}] unhandled message in loop: {:?}",
                        &client,
                        op_message
                    ));
                }
            };
            let ops_summary = Self::ops_summary(&ops);
            let final_request_block_idx = ops.last().map(|(_, _, block_idx)| *block_idx);

            // （新增）这里避免使用 `unwrap()`：
            // 为什么要加：一旦出现协议/数据不一致或 shape 错误，`unwrap()` 会直接 panic 把 worker 进程干掉；
            // 改成返回带上下文的错误，让 master 能拿到错误信息并继续运行/重试。
            //
            // 解码张量并统计耗时
            let decode_start = Instant::now();
            let mut x = x
                .to_tensor(&context.device)
                .map_err(|e| anyhow!("[{}] could not decode tensor: {e}", &client))?;
            let decode_time = decode_start.elapsed();
            /// 本次要执行的模型层数
            let num_ops = ops.len();

            // 遍历所有要执行的模型层
            for (layer_name, index_pos, block_idx) in ops {
                // 根据模型层名获取模型层
                if let Some(block) = context.blocks.get(&layer_name) {
                    // （新增）同样避免 `unwrap()`：把 layer/index_pos/block_idx 打进错误里，方便定位是哪一层/哪一步出错。
                    // forward 前向传播
                    x = block
                        .forward(&x, index_pos, block_idx, &mut context.cache)
                        .await
                        .map_err(|e| {
                            anyhow!(
                                "[{}] forward failed for {} (index_pos={}, block_idx={}): {e}",
                                &client,
                                layer_name,
                                index_pos,
                                block_idx
                            )
                        })?;
                } else {
                    return Err(anyhow!("could not find layer {}", &layer_name));
                }
            }

            if let Some(logits) = crate::models::qwen3_vl::maybe_forward_worker_gguf_output_head(
                &x,
                final_request_block_idx,
                sampling.as_ref(),
            )
            .map_err(|e| anyhow!("[{}] worker GGUF output head failed: {e}", &client))?
            {
                x = logits;
            }

            // 发送推理结果张量。RawTensor 转换会触发 device-to-host 拷贝/同步，也属于远端侧耗时。
            let response_tensor = super::RawTensor::from_tensor(&x)
                .map_err(|e| anyhow!("[{}] could not encode response tensor: {e}", &client))?;
            let compute_us = compute_start
                .elapsed()
                .as_micros()
                .min(u128::from(u64::MAX)) as u64;
            let elaps_ops = Duration::from_micros(compute_us);
            // 异步发送结果。
            match Self::write_message_timed(
                &mut socket,
                Message::from_raw_tensor_with_compute(response_tensor, compute_us),
            )
            .await
            {
                Ok((elaps_write, written)) => {
                    // 发送成功：统计性能+打印日志。
                    let ops_per_sec = (num_ops as f64 / elaps_ops.as_secs_f64()) as usize;
                    let write_bytes_per_sec = (written as f64 / elaps_write.as_secs_f64()) as usize;
                    let read_bytes_per_sec = (read_size as f64 / read_time.as_secs_f64()) as usize;
                    // 累计平均值
                    avg_ops += ops_per_sec;
                    avg_write += write_bytes_per_sec;
                    avg_read += read_bytes_per_sec;
                    // 打印详细的日志。
                    if Self::transfer_trace_enabled() {
                        log::info!(
                            "[transfer][worker {} msg={}] {} read={}B/{:.3}ms decode={:.3}ms compute={:.3}ms write={}B/{:.3}ms total={:.3}ms",
                            &client,
                            msg_idx,
                            ops_summary,
                            read_size,
                            read_time.as_secs_f64() * 1000.0,
                            decode_time.as_secs_f64() * 1000.0,
                            elaps_ops.as_secs_f64() * 1000.0,
                            written,
                            elaps_write.as_secs_f64() * 1000.0,
                            req_start.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(e) => {
                    return Err(anyhow!(
                        "[{}] could not send response tensor: {:?}",
                        &client,
                        e
                    ));
                }
            }

            // 每处理 N 条消息，计算并打印一次统计，避免刷屏 stdout。
            if msg_idx % NUM_OPS_TO_STATS == 0 {
                let avg_ops_ps = avg_ops / NUM_OPS_TO_STATS;
                let avg_read_h =
                    human_bytes::human_bytes(avg_read as f64 / NUM_OPS_TO_STATS as f64);
                let avg_write_h =
                    human_bytes::human_bytes(avg_write as f64 / NUM_OPS_TO_STATS as f64);
                log::info!(
                    "ops={}/s read={}/s write={}/s",
                    avg_ops_ps,
                    avg_read_h,
                    avg_write_h
                );
                avg_ops = 0;
                avg_write = 0;
                avg_read = 0;
            }
            msg_idx += 1;
        }

        Ok(())
    }

    /// 运行工作节点服务器的accept循环，等待并处理来自主节点的连接请求。
    pub async fn run(&mut self) -> Result<()> {
        while let Ok((socket, client)) = self.listener.accept().await {
            log::info!("{} connected", &client);
            let context = self.context.clone();
            tokio::spawn(async move {
                /// 处理连接，出错只打印日志，不崩溃整个服务
                if let Err(e) = Self::handle_master_client(socket, client, context).await {
                    log::error!("{}", e);
                }
            });
        }

        Ok(())
    }
}
