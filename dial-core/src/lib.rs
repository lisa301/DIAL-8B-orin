//! This is the core library where all spm logic is implemented.
#[macro_use]
extern crate anyhow;

use spm::{Mode, PlannerAlgorithm};

use clap::Parser;

pub mod models;
pub mod spm;
pub mod utils;

#[derive(Clone, Parser, Default, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// GPU device index.
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    #[arg(long, default_value_t, value_enum)]
    pub mode: Mode,

    /// Worker name.
    #[arg(long, default_value = "worker0")]
    pub name: Option<String>,

    /// Binding address and port for workers.
    #[arg(long, default_value = "0.0.0.0:10128")]
    pub address: String,

    /// Enable OpenAI compatible chat completion API.
    #[arg(long)]
    pub api: Option<String>,

    /// （新增）作为“客户端”连接到一个已启动的 API 服务端（例如 http://127.0.0.1:8082）。
    ///
    /// 为什么要加：之前用 curl 需要手写 JSON，请求/解析都不方便；
    /// 加了这个参数后，`spm-cli` 可以直接在命令行里输入问题并打印模型回复（纯文本）。
    #[arg(long)]
    pub api_client: Option<String>,

    /// （新增）配合 `--api-client`：发送一条消息并退出（单次提问）。
    #[arg(long)]
    pub ask: Option<String>,

    /// （新增）配合 `--api-client`：启动交互式对话（REPL），像聊天一样连续提问。
    #[arg(long, default_value_t = false)]
    pub repl: bool,

    /// （新增）配合 `--api-client`：服务端流式输出（边生成边显示）。
    #[arg(long, default_value_t = true)]
    pub stream: bool,

    /// （新增）配合 `--api-client`：在命令行额外打印服务端返回的性能指标（如 ttft_s/total_s）。
    /// 默认不打印，避免影响“只要回复文本”的使用体验。
    #[arg(long, default_value_t = true)]
    pub metrics: bool,

    /// （新增）限制 KV-cache/rope 的最大上下文长度（默认 4096）。
    ///
    /// 为什么要加：在 GPU 上使用旋转 KV-cache 时需要预分配固定窗口，
    /// 过大的 max_len 会导致显存暴涨/ OOM；限制到 4K 既能加速 decode
    /// 也能避免占用过大显存。
    #[arg(long, default_value_t = 4096)]
    pub kv_cache_max_len: usize,

    /// （新增）限制多模态图片送入视觉编码器前的最大边长（像素）。
    ///
    /// 为什么要加：Qwen3-VL 的视觉编码（ViT）在 CPU 上对 token 数非常敏感（复杂度近似 O(N^2)）。
    /// 默认如果把图片缩放到 768（48x48 patches），TTFT 很容易到几十秒；把它降到 384（24x24）
    /// 往往能带来数量级的提速。
    #[arg(long)]
    pub vision_max_side: Option<u32>,

    /// （新增）禁止把小图放大（只缩小，不上采样）。
    ///
    /// 为什么要加：上采样不会增加细节，但会显著增加视觉 token 数与 TTFT。
    #[arg(long, default_value_t = false)]
    pub vision_no_upscale: bool,

    /// （新增）把多模态图片强制缩放/补边到固定的正方形边长（像素），用于需要静态输入 shape 的硬件后端（如 BM1684）。
    ///
    /// - 输出尺寸会对齐到 `patch_size * spatial_merge_size` 的整数倍，避免后续 encode() 再 crop。
    /// - 默认会保持宽高比并做 letterbox（用 0.5 灰填充，归一化后约为 0），不会裁剪内容。
    /// - 与 `--vision-max-side` 不同：它是“强制固定”，适合导出 ONNX/bmodel。
    #[arg(long, default_value = "448")]
    pub vision_fixed_side: Option<u32>,

    ///   使用 RKNN 运行视觉编码器（ViT+merger），指定 .rknn 模型路径。
    #[arg(long)]
    pub vision_rknn: Option<String>,

    /// RKNN runtime 库路径（librknnrt.so），不填则自动搜索。
    #[arg(long)]
    pub vision_rknn_lib: Option<String>,

    ///   使用 RKNN chunk 运行 Qwen3-VL 文本路径（目录下应包含按层切块的 .rknn）。
    ///
    /// - decode：加载 `text_decode` chunk（必需）。
    /// - prefill：默认不使用；需显式开启 `--text-rknn-prefill`。
    #[arg(long)]
    pub text_rknn_dir: Option<String>,

    /// （新增）文本 RKNN chunk 的 runtime 库路径（librknnrt.so），不填则自动搜索。
    #[arg(long)]
    pub text_rknn_lib: Option<String>,

    /// 使用 RKNN 运行本地文本注意力中的无状态 QKV 投影子图。
    ///
    /// 这条路径不接管 KV cache，不接管 attention softmax，只替换 `q_proj/k_proj/v_proj`
    /// 三个线性层，适合作为比完整 text_rknn chunk 更稳定的 NPU+CPU 拆分方案。
    #[arg(long)]
    pub text_qkv_rknn_dir: Option<String>,

    /// 文本 QKV RKNN 子图的 runtime 库路径（librknnrt.so），不填则自动搜索。
    #[arg(long)]
    pub text_qkv_rknn_lib: Option<String>,

    /// 使用 RKNN 运行本地文本 MLP 子图（gate/up + SiLU + down）。
    ///
    /// 这条路径只接管 decode 阶段 shape=[1,1,hidden] 的 MLP，attention/KV cache 仍由宿主运行。
    #[arg(long)]
    pub text_mlp_rknn_dir: Option<String>,

    /// 文本 MLP RKNN 子图的 runtime 库路径（librknnrt.so），不填则自动搜索。
    #[arg(long)]
    pub text_mlp_rknn_lib: Option<String>,

    /// 显式启用 text RKNN prefill。
    ///
    /// 适合 RK3588 这类 NPU+CPU 部署：prefill 走完整的 RKNN 文本 chunk，decode 再按
    /// `--text-decode-mode` 选择 NPU 前缀或纯 CPU 软件尾部。默认关闭，避免在 prefill
    /// chunk 未准备完整时污染 KV/hidden。
    #[arg(long, default_value_t = false)]
    pub text_rknn_prefill: bool,

    /// （新增）文本 decode 执行模式（仅影响 decode 阶段）：
    /// - auto：有 text_rknn 则自动使用（全覆盖=全NPU，前缀覆盖=NPU+CPU），无则纯软件路径。
    /// - full-npu：decode 必须全层走 RKNN（需要完整 decode chunk 覆盖）。
    /// - npu-cpu：decode 先走 RKNN 前缀层，再走本地/分布式剩余层；保留 `npu-gpu` 旧别名。
    /// - cpu-only：decode 强制不走 RKNN，全部走本地/分布式层；保留 `cpu-gpu` 旧别名。
    #[arg(long, default_value = "auto")]
    pub text_decode_mode: String,

    /// （新增）配合 `--api-client`：附带本地图片文件，用于 Qwen3-VL 这类多模态模型的图片理解。
    /// 实现方式：客户端会把图片转成 base64，并按 OpenAI 风格的 `image_base64` part 发送。
    #[arg(long)]
    pub image: Option<String>,

    /// 配合 `--api-client`：附带本地视频文件。服务端默认按 Qwen3-VL 官方规则采样。
    #[arg(long)]
    pub video: Option<String>,

    /// 单个原始视频请求允许的最大二进制大小。超限会明确拒绝，不会截断视频。
    #[arg(long, default_value_t = 268_435_456)]
    pub video_max_bytes: usize,

    /// 官方视频采样后的最大帧数（默认 768）。
    #[arg(long, default_value_t = 768)]
    pub video_max_frames: usize,

    /// 官方视频采样帧率（默认 2 FPS）。设置为 0 时按最大帧数均匀采样。
    #[arg(long, default_value_t = 2.0)]
    pub video_fps: f64,

    /// 视频采样的最小帧数（默认 4）。
    #[arg(long, default_value_t = 4)]
    pub video_min_frames: usize,

    /// 显式关闭官方采样，处理全部原始帧；长视频会非常慢。
    #[arg(long, default_value_t = false)]
    pub video_no_sample: bool,

    /// 视觉时间 patch 的批大小。增大可提升吞吐，但会增加内存占用。
    #[arg(long, default_value_t = 4)]
    pub video_batch_size: usize,

    /// 视频采样帧送入视觉编码器前的最大边长。仅缩放空间尺寸，不减少时间帧。
    #[arg(long, default_value_t = 128)]
    pub video_max_side: u32,

    /// Llama3 model data path.
    #[arg(
        long,
        default_value = "/home/firefly/Documents/Qwen3-VL-8B-Instruct"
    )]
    pub model: String,

    /// Topology file.
    #[arg(
        long,
        default_value = "/home/firefly/Documents/Dial_llama/topology_qwen3vl.yml"
    )]
    pub topology: String,

    /// Replace the static topology with a profile-guided automatic plan.
    #[arg(long)]
    pub auto_plan_profile: Option<String>,

    /// Planning objective used with --auto-plan-profile.
    #[arg(long, value_enum, default_value_t = PlannerAlgorithm::Dial)]
    pub auto_plan_algorithm: PlannerAlgorithm,

    /// Write the selected automatic plan and cost breakdown as JSON (master only).
    #[arg(long)]
    pub auto_plan_output: Option<String>,

    /// The initial prompt.
    #[arg(long, default_value = "")]
    pub prompt: String,
    /// The system prompt.
    #[arg(long, default_value = "You are a helpful AI assistant.")]
    pub system_prompt: String,
    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,
    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 2048)]
    pub sample_len: usize,
    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f64,
    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    pub top_p: Option<f64>,
    /// Only sample among the top K samples.
    #[arg(long)]
    pub top_k: Option<usize>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    pub repeat_last_n: usize,
    /// Use different dtype than f16
    #[arg(long)]
    pub dtype: Option<String>,
    /// Force attention matmul/softmax to use f32 (more stable, slower).
    #[arg(long, default_value_t = false)]
    pub attn_f32: bool,
    /// Run the Qwen3-VL lm_head through the CPU q8 implementation.
    /// Enabled by default for the RK3588 master; pass `true` or `false` for A/B tests.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub lm_head_q8: bool,
    /// Enable q8 quantization for local Qwen3-VL transformer linear layers. Use `--local-linear-q8 false` to disable.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub local_linear_q8: bool,
    /// Quantize Qwen3-VL transformer linear weights to row-wise int8 on CUDA workers.
    /// Activations, outputs, KV cache, and network tensors remain F16.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    pub worker_w8a16: bool,
    /// Use Ollama/llama.cpp GGUF quantized transformer weights on CUDA workers.
    /// Only transformer layers assigned to this worker are loaded from the file.
    #[arg(long)]
    pub worker_quantized_gguf: Option<String>,
    /// Use fused FP16 projections for GGUF prefill while retaining GGUF for decode.
    /// This dequantizes a second, resident copy of the assigned projection weights.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    pub worker_gguf_fp16_prefill: bool,
    /// Run final RMSNorm and the quantized GGUF output projection on the Worker
    /// that owns the final transformer layer, returning F16 logits to the Master.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    pub worker_gguf_output_head: bool,
    /// Sample on the GGUF output-head Worker and return only one U32 token id.
    /// Requires --worker-gguf-output-head and removes full-vocabulary logits transfers.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    pub worker_gguf_sample_token: bool,
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_plan_algorithm_names_match_experiment_reports() {
        let latency = Args::try_parse_from([
            "dial-core",
            "--auto-plan-profile",
            "profile.yml",
            "--auto-plan-algorithm",
            "edgeshard-latency",
        ])
        .unwrap();
        assert_eq!(
            latency.auto_plan_algorithm,
            PlannerAlgorithm::EdgeShardLatency
        );
        assert_eq!(
            serde_json::to_value(latency.auto_plan_algorithm).unwrap(),
            serde_json::json!("edgeshard-latency")
        );

        let throughput = Args::try_parse_from([
            "dial-core",
            "--auto-plan-profile",
            "profile.yml",
            "--auto-plan-algorithm",
            "edgeshard-throughput",
        ])
        .unwrap();
        assert_eq!(
            throughput.auto_plan_algorithm,
            PlannerAlgorithm::EdgeShardThroughput
        );
        assert_eq!(
            serde_json::to_value(throughput.auto_plan_algorithm).unwrap(),
            serde_json::json!("edgeshard-throughput")
        );
    }

    #[test]
    fn lm_head_q8_is_boolean_and_enabled_by_default() {
        let default_args = Args::try_parse_from(["dial-core"]).unwrap();
        assert!(default_args.lm_head_q8);

        let q8_args = Args::try_parse_from(["dial-core", "--lm-head-q8", "true"]).unwrap();
        assert!(q8_args.lm_head_q8);

        let f16_args = Args::try_parse_from(["dial-core", "--lm-head-q8", "false"]).unwrap();
        assert!(!f16_args.lm_head_q8);
    }

    #[test]
    fn worker_w8a16_is_boolean_and_disabled_by_default() {
        let default_args = Args::try_parse_from(["dial-core"]).unwrap();
        assert!(!default_args.worker_w8a16);

        let enabled = Args::try_parse_from(["dial-core", "--worker-w8a16", "true"]).unwrap();
        assert!(enabled.worker_w8a16);

        let disabled = Args::try_parse_from(["dial-core", "--worker-w8a16", "false"]).unwrap();
        assert!(!disabled.worker_w8a16);
    }

    #[test]
    fn worker_gguf_fp16_prefill_is_boolean_and_disabled_by_default() {
        let default_args = Args::try_parse_from(["dial-core"]).unwrap();
        assert!(!default_args.worker_gguf_fp16_prefill);

        let enabled =
            Args::try_parse_from(["dial-core", "--worker-gguf-fp16-prefill", "true"]).unwrap();
        assert!(enabled.worker_gguf_fp16_prefill);

        let disabled =
            Args::try_parse_from(["dial-core", "--worker-gguf-fp16-prefill", "false"]).unwrap();
        assert!(!disabled.worker_gguf_fp16_prefill);
    }

    #[test]
    fn worker_gguf_output_head_is_boolean_and_disabled_by_default() {
        let default_args = Args::try_parse_from(["dial-core"]).unwrap();
        assert!(!default_args.worker_gguf_output_head);

        let enabled =
            Args::try_parse_from(["dial-core", "--worker-gguf-output-head", "true"]).unwrap();
        assert!(enabled.worker_gguf_output_head);

        let disabled =
            Args::try_parse_from(["dial-core", "--worker-gguf-output-head", "false"]).unwrap();
        assert!(!disabled.worker_gguf_output_head);
    }

    #[test]
    fn worker_gguf_sample_token_is_boolean_and_disabled_by_default() {
        let default_args = Args::try_parse_from(["dial-core"]).unwrap();
        assert!(!default_args.worker_gguf_sample_token);

        let enabled =
            Args::try_parse_from(["dial-core", "--worker-gguf-sample-token", "true"]).unwrap();
        assert!(enabled.worker_gguf_sample_token);

        let disabled =
            Args::try_parse_from(["dial-core", "--worker-gguf-sample-token", "false"]).unwrap();
        assert!(!disabled.worker_gguf_sample_token);
    }

    #[test]
    fn worker_quantized_gguf_accepts_a_path() {
        let default_args = Args::try_parse_from(["dial-core"]).unwrap();
        assert!(default_args.worker_quantized_gguf.is_none());

        let args = Args::try_parse_from([
            "dial-core",
            "--worker-quantized-gguf",
            "/models/qwen3-vl-q4_k_m.gguf",
        ])
        .unwrap();
        assert_eq!(
            args.worker_quantized_gguf.as_deref(),
            Some("/models/qwen3-vl-q4_k_m.gguf")
        );
    }
}
