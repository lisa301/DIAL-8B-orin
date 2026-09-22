use std::{
    fmt::{Debug, Display},
    path::PathBuf,
};

use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{
    models::{
        llama3::{Cache, Config, LlamaConfig},
        qwen3_vl::Qwen3VlConfig,
    },
    utils, Args,
};

#[cfg(feature = "master")]
mod api;
#[cfg(feature = "master")]
mod master;

mod client;
mod planner;
mod proto;
mod topology;
mod worker;

#[cfg(feature = "master")]
pub use master::*;

pub use client::*;
pub use planner::*;
pub use proto::*;
pub use topology::*;
pub use worker::*;

/// Determines if we run in master or worker mode.
#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum Mode {
    #[default]
    Master,
    Worker,
}

/// 所有模块共用这份数据
#[derive(Clone)]
pub struct Context {
    pub args: Args,
    pub dtype: DType,
    pub topology: Topology,
    pub data_path: PathBuf,
    pub device: Device,
    pub config: Config,
    pub cache: Cache,
    pub var_builder: VarBuilder<'static>,
}

impl Context {
    /// 是整个框架的入口初始化函数
    pub fn from_args(args: Args) -> Result<Self> {
        let dtype: DType = match args.dtype.as_deref() {
            Some("f16") => DType::F16,
            Some("bf16") => DType::BF16,
            Some("f32") => DType::F32,
            Some(dtype) => bail!("unsupported dtype {dtype}"),
            None => DType::F16,
        };
        // 选择运行设备
        let device = utils::get_inference_device(args.cpu, args.device)
            .map_err(|e| anyhow!("can't attach to device: {:?}", e))?;

        // 打印启动日志
        log::info!(
            "[{:?}] dtype={:?} device={:?} mem={}",
            args.mode,
            &dtype,
            &device,
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );
        // 模型路径+读取config.json
        let data_path = PathBuf::from(&args.model);
        let config_filename = data_path.join("config.json");

        let raw = std::fs::read(&config_filename)
            .map_err(|e| anyhow!("can't read {}: {:?}", config_filename.display(), e))?;
        let v: serde_json::Value = serde_json::from_slice(&raw)
            .map_err(|e| anyhow!("can't parse {}: {:?}", config_filename.display(), e))?;
        let is_qwen3_vl = matches!(
            v.get("model_type").and_then(|m| m.as_str()),
            Some("qwen3_vl")
        );
        let mut config: Config = if is_qwen3_vl {
            Qwen3VlConfig::from_path(&config_filename)?.text()
        } else {
            LlamaConfig::from_path(&config_filename)?.into_config()
        };
        if is_qwen3_vl {
            crate::models::qwen3_vl::configure_worker_gguf_from_args(
                args.worker_quantized_gguf.as_deref(),
                args.worker_gguf_fp16_prefill,
                args.worker_gguf_output_head,
                args.worker_gguf_sample_token,
                matches!(&args.mode, Mode::Worker),
                &device,
                dtype,
                &config,
            )?;
            crate::models::qwen3_vl::configure_qkv_rknn_from_args(&args)?;
            crate::models::qwen3_vl::configure_mlp_rknn_from_args(&args)?;
            crate::models::qwen3_vl::configure_local_q8_from_args(args.local_linear_q8);
            crate::models::qwen3_vl::configure_worker_w8a16_from_args(
                args.worker_w8a16,
                matches!(&args.mode, Mode::Worker),
                &device,
                dtype,
            );
        }
        // 强制注意力层用f32计算，限制KV缓存最大长度
        config.attn_f32 = args.attn_f32;
        if args.kv_cache_max_len > 0 && config.max_seq_len > args.kv_cache_max_len {
            log::info!(
                "capping max_seq_len from {} to {}",
                config.max_seq_len,
                args.kv_cache_max_len
            );
            config.max_seq_len = args.kv_cache_max_len;
        }
        // Preserve the static topology unless automatic planning is explicitly enabled.
        let topology = if let Some(profile_path) = args.auto_plan_profile.as_deref() {
            let layer_prefix = if is_qwen3_vl {
                "model.language_model.layers"
            } else {
                "model.layers"
            };
            let planned = AutoPlanner::from_path(profile_path)?.plan_with_algorithm(
                &config,
                layer_prefix,
                args.auto_plan_algorithm,
            )?;
            log::info!("{}", planned.summary());
            if matches!(&args.mode, Mode::Master) {
                if let Some(output_path) = args.auto_plan_output.as_deref() {
                    planned.write_report(output_path)?;
                }
            }
            planned.topology
        } else {
            Topology::from_path(&args.topology)?
        };
        // 读取模型权重文件
        let model_tensors_index: PathBuf = data_path.join("model.safetensors.index.json");
        let var_builder =
            utils::load_var_builder_from_index(model_tensors_index, dtype, device.clone())?;
        // 创建KV缓存
        let cache = Cache::new(true, dtype, &config, &device)?;
        // 把所有东西打包返回Context
        Ok(Context {
            args,
            dtype,
            topology,
            data_path,
            device,
            config,
            cache,
            var_builder,
        })
    }
}

/// 定义了模型层如何运行的统一接口.
#[async_trait]
/// trait 定义和约束
pub trait Forwarder: Debug + Send + Sync + Display {
    /// 从模型权重文件中，加载某一层的参数。
    fn load(name: String, vb: VarBuilder, cfg: &Config) -> Result<Box<Self>>
    where
        Self: Sized;

    /// 执行一次前向推理.
    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor>;

    /// 可变版本的前向推理.
    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor>;

    /// 批量推理的接口.
    async fn forward_batch(
        &mut self,
        _x: &Tensor,
        _batch: Vec<(String, usize, usize)>,
        _cache: &mut Cache,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    /// 获取层名.
    fn layer_name(&self) -> &str;

    /// Release optional local MLP RKNN contexts when another RKNN model needs NPU memory.
    fn release_mlp_rknn(&self, _reason: &str) -> usize {
        0
    }

    /// 获取层的唯一标识符（默认为 "local"）.
    fn ident(&self) -> &str {
        "local"
    }
}
