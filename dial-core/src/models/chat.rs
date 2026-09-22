use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// The role of a message in a chat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageRole {
    /// System prompt.
    #[serde(alias = "system")]
    System,
    /// User prompt.
    #[serde(alias = "user")]
    User,
    /// Assistant response.
    #[serde(alias = "assistant")]
    Assistant,
}

/// 自定义 MessageRole 的打印样式
impl std::fmt::Display for MessageRole {
    /// 定义必须实现的方法
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// 写入格式化输出
        write!(
            f,
            "{}",
            match self {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
            }
        )
    }
}

/// Message 结构，表示一条完整的聊天格式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message role.
    pub role: MessageRole,
    /// Messagae content.
    pub content: MessageContent,
}

///  MessageContent 这个类型要同时兼容两类输入：纯文本和多段多模态内容.
/// - Legacy format: `"content": "hello"` 旧格式是字符串
/// - Multi-part format (OpenAI-style): `"content": [{"type":"text","text":"hello"}, ...]`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
/// 定义公开枚举 MessageContent，是消息内容的统一抽象
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
/// 定义 ContentPart 枚举，表示多段内容的单个片段
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },

    /// Base64-encoded image payload (raw base64, without data URL prefix).
    #[serde(rename = "image_base64")]
    ImageBase64 {
        /// Optional mime type, e.g. "image/png" or "image/jpeg".
        #[serde(default)]
        media_type: Option<String>,
        /// Base64 data, without prefix.
        data: String,
    },

    /// OpenAI-style image URL part. This also supports data URLs such as
    /// "data:image/png;base64,....".
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },

    /// Base64-encoded video payload (raw base64, without data URL prefix).
    #[serde(rename = "video_base64")]
    VideoBase64 {
        /// Optional mime type, e.g. "video/mp4" or "video/webm".
        #[serde(default)]
        media_type: Option<String>,
        /// Base64 data, without prefix.
        data: String,
    },

    /// Video URL part. This build accepts data URLs only so inference remains local.
    #[serde(rename = "video_url")]
    VideoUrl { video_url: VideoUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoUrl {
    pub url: String,
}
/// 给定一个 MessageContent结构体添加实用方法
impl MessageContent {
    /// Returns a plain-text representation.
    /// Errors if the content contains non-text parts.
    /// 读取当前实例，不修改
    pub fn to_text_strict(&self) -> Result<String> {
        match self {
            MessageContent::Text(s) => Ok(s.clone()),
            /// 纯文本，直接克隆文本，返回成功
            MessageContent::Parts(parts) => {
                /// 文本 + 图片，进入处理
                let mut out = String::new();
                /// 创建一个空字符串，用来拼接文本
                for part in parts {
                    match part {
                        ContentPart::Text { text } => out.push_str(text),
                        /// 如果是文本片段，追加到输出字符串
                        ContentPart::ImageBase64 { .. }
                        | ContentPart::ImageUrl { .. }
                        | ContentPart::VideoBase64 { .. }
                        | ContentPart::VideoUrl { .. } => {
                            bail!("non-text content (image/video) is not supported by this model")
                        }
                    }
                }
                Ok(out)
            }
        }
    }
    /// 判断是否包含图片或视频
    pub fn has_image(&self) -> bool {
        match self {
            MessageContent::Text(_) => false, // 纯文本，返回false
            MessageContent::Parts(parts) => parts.iter().any(|p| {
                matches!(
                    p,
                    ContentPart::ImageBase64 { .. } | ContentPart::ImageUrl { .. }
                )
            }),
        }
    }

    pub fn has_video(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts.iter().any(|p| {
                matches!(
                    p,
                    ContentPart::VideoBase64 { .. } | ContentPart::VideoUrl { .. }
                )
            }),
        }
    }

    pub fn is_multimodal(&self) -> bool {
        self.has_image() || self.has_video()
    }
}
/// 给定一个 Message结构体添加实用方法
impl Message {
    /// 创建系统消息
    pub fn system(content: String) -> Self {
        Self {
            role: MessageRole::System,
            content: MessageContent::Text(content),
        }
    }

    /// 创建用户消息.
    pub fn user(content: String) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::Text(content),
        }
    }

    /// 创建助手消息.
    pub fn assistant(content: String) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: MessageContent::Text(content),
        }
    }
    /// 是否多模态
    pub fn is_multimodal(&self) -> bool {
        self.content.is_multimodal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_native_video_parts_as_multimodal_content() {
        let message: Message = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [
                {
                    "type": "video_base64",
                    "media_type": "video/mp4",
                    "data": "AAAA"
                },
                {"type": "text", "text": "What happens?"}
            ]
        }))
        .unwrap();

        assert!(message.content.has_video());
        assert!(!message.content.has_image());
        assert!(message.is_multimodal());
        assert!(message.content.to_text_strict().is_err());
    }

    #[test]
    fn deserializes_video_data_urls() {
        let message: Message = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [{
                "type": "video_url",
                "video_url": {"url": "data:video/webm;base64,AAAA"}
            }]
        }))
        .unwrap();

        assert!(message.content.has_video());
    }
}
