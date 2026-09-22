use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

fn default_rope_theta() -> f32 {
    10_000.0
}

#[derive(Debug, Clone, Deserialize)]
/// 定义公共结构体Qwen3VlConfig：Qwen3-VL模型的*顶层总配置*
pub struct Qwen3VlConfig {
    /// 模型类型
    pub model_type: String,
    /// 模型架构
    pub architectures: Option<Vec<String>>,

    /// 该配置在部分模型 checkpoint 中是顶层字段.
    #[serde(default)]
    /// 是否将词嵌入与输出投影绑定
    pub tie_word_embeddings: bool,
    /// 图像特殊token的ID
    pub image_token_id: u32,
    pub video_token_id: Option<u32>,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    /// 文本分支的配置
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
}

#[derive(Debug, Clone, Deserialize)]
/// 定义公共结构体 TextConfig
pub struct TextConfig {
    pub hidden_size: usize,
    /// 隐藏层
    pub intermediate_size: usize,
    /// 中间层
    pub vocab_size: usize,
    /// 词表大小
    pub num_hidden_layers: usize,
    /// 隐藏层数量
    pub num_attention_heads: usize,
    /// 注意力头数
    pub num_key_value_heads: usize,
    /// 键值头数
    pub rms_norm_eps: f64,
    /// RMS归一化
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// rope参数
    pub bos_token_id: Option<u32>,
    /// 开始token的ID
    pub eos_token_id: Option<u32>,
    /// 结束token的ID
    pub max_position_embeddings: usize,
    /// 最大位置嵌入

    /// Some checkpoints omit `lm_head.weight` and tie output projection to embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    // 是否将词嵌入与输出投影绑定
}

#[derive(Debug, Clone, Deserialize)]
/// 定义公共结构体 VisionConfig
pub struct VisionConfig {
    pub depth: usize,
    /// 深度
    pub hidden_size: usize,
    /// 隐藏层
    pub intermediate_size: usize,
    /// 中间层
    pub num_heads: usize,
    /// 注意力头数
    pub in_channels: usize,
    /// 输入通道数
    pub patch_size: usize,
    /// 空间patch大小
    pub temporal_patch_size: usize,
    /// 时间patch大小
    pub num_position_embeddings: usize,
    /// 位置编码数量
    pub spatial_merge_size: usize,
    /// 空间融合尺寸
    pub out_hidden_size: usize,
    /// 输出隐藏层
    pub deepstack_visual_indexes: Option<Vec<usize>>,
    // DeepStack视觉层索引
}
///  为 Qwen3VlConfig 结构体实现方法
impl Qwen3VlConfig {
    /// 定义关联方法：类似静态方法，无需结构体实例即可调用
    // 功能：从文件路径加载并解析配置
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading configuration from {}", path.display());
        let data =
            std::fs::read(path).map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))
    }
    /// 定义【实例方法】（必须通过 Qwen3VlConfig 实例调用）
    // 功能：把多模态配置 转换为 Llama3 文本模型专用配置
    pub fn text(&self) -> crate::models::llama3::Config {
        crate::models::llama3::Config {
            hidden_size: self.text_config.hidden_size,
            intermediate_size: self.text_config.intermediate_size,
            vocab_size: self.text_config.vocab_size,
            num_hidden_layers: self.text_config.num_hidden_layers,
            num_attention_heads: self.text_config.num_attention_heads,
            num_key_value_heads: self.text_config.num_key_value_heads,
            rms_norm_eps: self.text_config.rms_norm_eps,
            rope_theta: self.text_config.rope_theta,
            bos_token_id: self.text_config.bos_token_id,
            eos_token_id: self.text_config.eos_token_id,
            max_seq_len: self.text_config.max_position_embeddings,
            attn_f32: false,
        }
    }
}
