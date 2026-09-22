//! 这是 spm 通信协议模块
// 定义2个核心常量：魔数、最大消息大小
// 对外公开所有消息相关的东西
const PROTO_MAGIC: u32 = 0x104F4C7;

/// spm protocol message max size.
const MESSAGE_MAX_SIZE: u32 = 512 * 1024 * 1024;

mod message;
// 声明子模块

pub use message::*;
// 把 message 子模块里的所有东西全部公开暴露出去
