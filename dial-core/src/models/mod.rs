pub mod chat;
pub mod llama3;
pub mod qwen3_vl;

use crate::spm::{Context, Forwarder};

use anyhow::Result;
use async_trait::async_trait;
use chat::Message;

/// Token 结构体.
pub struct Token {
    /// 定义id.
    pub id: u32,
    /// 解析后的文本片段.
    pub text: Option<String>,
    /// Token流结束标记.
    pub is_end_of_stream: bool,
}
/// 为token实现Display trait.
impl std::fmt::Display for Token {
    ///  Display 唯一要求实现的方法
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            if let Some(text) = &self.text {
                text.clone()
            } else {
                // 无文本时显示token ID
                format!("<token {}>", self.id)
            }
        )
    }
}

/// 一个模型必须实现这个trait,才能被dial使用.
#[async_trait]
/// 定义公共trait.
pub trait Generator {
    /// This associated type determines which part of the model can be sharded.
    type Shardable: Forwarder;

    /// The model name.
    const MODEL_NAME: &'static str;

    /// Load the model from the context.
    async fn load(context: Context) -> Result<Box<Self>>;

    /// Add a message to the chat.
    fn add_message(&mut self, message: Message) -> Result<()>;
    /// Clear chat history.
    fn reset(&mut self) -> Result<()>;

    /// Return the next token.
    async fn next_token(&mut self, index: usize) -> Result<Token>;
    /// Return the number of generated tokens so far.
    fn generated_tokens(&self) -> usize;
}
