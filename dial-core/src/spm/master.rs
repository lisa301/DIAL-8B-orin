use std::io::Write;

use crate::models::{chat::Message, Generator};

use super::{api, reset_distributed_profile, Context};

use anyhow::Result;

/// A master connects to, communicates with and orchestrates the workers.
pub struct Master<G> {
    pub ctx: Context,
    pub model: Box<G>,
}

//给泛型Master结构体实现方法
impl<G: Generator + Send + Sync + 'static> Master<G> {
    /// 异步创建并初始化Master主节点。
    pub async fn new(ctx: Context) -> Result<Self> {
        let model = G::load(ctx.clone()).await?;
        Ok(Self { ctx, model })
    }
    // 整个程序的入口
    pub async fn run(mut self) -> Result<()> {
        if self.ctx.args.api.is_some() {
            // 如果命令行带 --api 参数，就启动HTTP接口服务。
            api::start(self).await?;
        } else {
            // CLI模式，添加系统提示词+用户提示词。
            self.model
                .add_message(Message::system(self.ctx.args.system_prompt.clone()))?;
            self.model
                .add_message(Message::user(self.ctx.args.prompt.clone()))?;

            // 生成回复并输出到终端
            self.generate(|data| {
                if data.is_empty() {
                    println!();
                } else {
                    print!("{data}")
                }
                std::io::stdout().flush().unwrap();
            })
            .await?;
        }

        Ok(())
    }

    /// 重置整个主节点
    pub fn reset(&mut self) -> Result<()> {
        reset_distributed_profile();
        self.model.reset()
    }

    /// 逐一生成token，并通过stream函数实时输出。
    pub async fn generate<S>(&mut self, mut stream: S) -> Result<()>
    where
        S: FnMut(&str),
    {
        /// 打印日志
        log::info!(
            "starting the inference loop (mem={})\n\n",
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        log::debug!("  ctx.args.sample_len = {}", self.ctx.args.sample_len);

        stream(&self.ctx.args.prompt);

        let mut start_gen = std::time::Instant::now();

        for index in 0..self.ctx.args.sample_len {
            if index == 1 {
                // record start time again since the first token is the warmup
                start_gen = std::time::Instant::now()
            }
            /// 生成下一个词/字
            let token = self.model.next_token(index).await?;
            /// 如果生成结束，停止循环；否则把生成的token实时输出。
            if token.is_end_of_stream {
                break;
            } else {
                stream(&token.to_string());
                // Yield to let HTTP streaming tasks flush chunked responses promptly.
                tokio::task::yield_now().await;
            }
        }

        // 输出结束标记
        stream("");

        let dt = start_gen.elapsed();
        let generated = self.model.generated_tokens();
        /// 打印性能统计
        log::info!(
            "{} tokens generated ({} token/s) - mem={}",
            generated,
            (generated - 1) as f64 / dt.as_secs_f64(),
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        Ok(())
    }
}
