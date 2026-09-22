use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{linear_no_bias as linear, Embedding, Linear, Module, RmsNorm};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use tokenizers::Tokenizer;

use crate::{
    models::{chat::Message, Generator, Token},
    spm::{Context, Forwarder},
};

use super::{transformer::Transformer, History};

/// Default end of stream token if not found in configuration.
const DEFAULT_EOS_TOKEN: &str = "</s>";

/// Load the tokenizer and return the first tokens from the prompt in context.
fn load_tokenizer(ctx: &Context) -> Result<(Tokenizer, Option<u32>)> {
    let tokenizer_filename = ctx.data_path.join("tokenizer.json");

    log::info!("loading tokenizer from {}", tokenizer_filename.display());

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let eos_token_id = ctx
        .config
        .eos_token_id
        .or_else(|| tokenizer.token_to_id(DEFAULT_EOS_TOKEN));

    Ok((tokenizer, eos_token_id))
}

/// Create the logit sampling logic from the context.
fn create_logits_processor(ctx: &Context) -> LogitsProcessor {
    let temperature = ctx.args.temperature;
    let sampling = if temperature <= 0. {
        Sampling::ArgMax
    } else {
        match (ctx.args.top_k, ctx.args.top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    };
    LogitsProcessor::from_sampling(ctx.args.seed, sampling)
}

/// LLama main class.
pub struct LLama {
    ctx: Context,
    // transformer模块前后均有分词器，前面有词嵌入层，后面有归一化层和线性层。
    tokenizer: Tokenizer, // 分词器
    embedding: Embedding, // 词嵌入层
    eos_token_id: Option<u32>,
    index_pos: usize,
    generated: usize,

    blocks: Vec<Box<dyn Forwarder>>,

    ln_f: RmsNorm,   // 归一化层
    lm_head: Linear, // 线性层

    logits_processor: LogitsProcessor,

    history: History,
    tokens: Vec<u32>,
}

impl LLama {
    // 每次前向传播生成一个token
    async fn forward(&mut self, x: &Tensor, idx: usize) -> Result<Tensor> {
        let (_batch_size, seq_len) = x.dims2()?;
        // log::info!("Input tensor shape: {:?}", x.shape());
        let mut x = self.embedding.forward(x)?;
        // log::info!("Input tensor shape after embedding: {:?}", x.shape());

        let num_blocks = self.blocks.len(); // 这里获取模型有多少个transformer模块，来保证全部进行推理
                                            // log::info!("num_blocks: {:?}", num_blocks);
        let mut block_idx = 0; // 每次生成一个token就要执行一次forward,所以块索引会归零

        // log::info!("X = {}", &x);

        while block_idx < num_blocks {
            let curr_block_id = self.blocks[block_idx].ident().to_owned();
            if curr_block_id == "local" {
                // log::info!("x={:?} idx={idx} block={block_idx}", x.shape());

                // do not batch local inferences
                x = self.blocks[block_idx]
                    .forward_mut(&x, idx, block_idx, &mut self.ctx.cache)
                    .await
                    .map_err(|e| {
                        anyhow!("error in forward operation of local block {block_idx}: {e}")
                    })?;

                block_idx += 1;
            } else {
                // collect all contiguous layers running on the same worker
                let mut batch = vec![];
                let first = block_idx;
                while block_idx < num_blocks && self.blocks[block_idx].ident() == curr_block_id {
                    batch.push((
                        self.blocks[block_idx].layer_name().to_string(),
                        idx,
                        block_idx,
                    ));
                    block_idx += 1;
                }

                x = self.blocks[first]
                    .forward_batch(&x, batch, &mut self.ctx.cache)
                    .await
                    .map_err(|e| {
                        anyhow!("error in forward batch operation for block {block_idx}: {e}")
                    })?;
            }

            // log::info!("{}.forward(X) -> {}", &curr_block_id, &x);
        }

        // log::info!("layer normalization (ln_f), tensor shape: {:?}", x.shape());
        let x = self // 归一化层
            .ln_f
            .forward(&x)
            .map_err(|e| anyhow!("error in ln_f.forward: {e}"))?;
        // log::info!("ln_f shape: {:?}", self.ln_f.shape());
        // log::info!("After layer normalization (ln_f), tensor shape: {:?}", x.shape());

        let x = x // 切片
            .i((.., seq_len - 1, ..))
            .map_err(|e| anyhow!("error in x.i: {e}"))?
            .contiguous()
            .map_err(|e| anyhow!("error in x.i.contiguous: {e}"))?;
        // log::info!("After qie pian, tensor shape: {:?}", x.shape());

        let logits = self // 线性层
            .lm_head
            .forward(&x)
            .map_err(|e| anyhow!("error in lm_head.forward: {e}"))?;
        // log::info!("lm_head shape: {:?}", self.lm_head.shape());
        // log::info!("After layer lm_head, tensor shape: {:?}", logits.shape());
        // log::info!("After layer to_dtype, tensor shape: {:?}", logits);

        logits // 改变数据类型
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("error converting logits: {e}"))
    }

    fn start_dialog_prompt(&mut self) -> Result<()> {
        // make sure we start clean
        self.tokens.clear();
        self.ctx.cache.clear();
        self.index_pos = 0;

        // log::debug!("generating history tokens ...");

        // generate raw from history
        let dialog = self.history.encode_dialog_to_prompt()?; // 将对话历史进行编码

        // log::debug!("dialog={}", &dialog);

        // tokenize raw  // 将对话历史编码的进行分词作为初始的tokens进行输入。
        self.tokens = self
            .tokenizer
            .encode(dialog, false) // do not add special tokens as we already added them
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();

        // log::debug!("encoded={:?}", &self.tokens);

        // log::debug!("history tokens: {}", self.tokens.len());

        Ok(())
    }
}

#[async_trait]
impl Generator for LLama {
    type Shardable = Transformer;

    const MODEL_NAME: &'static str = "llama3";

    /// Load this model from the context.
    async fn load(ctx: Context) -> Result<Box<Self>> {
        log::info!("loading embeddings ...");
        let embedding: Embedding = candle_nn::embedding(
            ctx.config.vocab_size,
            ctx.config.hidden_size,
            ctx.var_builder.pp("model.embed_tokens"),
        )?;

        log::info!("loading lm_head ...");
        let lm_head = linear(
            ctx.config.hidden_size,
            ctx.config.vocab_size,
            ctx.var_builder.pp("lm_head"),
        )?;

        log::info!("loading model.norm ...");
        let ln_f = candle_nn::rms_norm(
            ctx.config.hidden_size,
            ctx.config.rms_norm_eps,
            ctx.var_builder.pp("model.norm"),
        )?;

        log::info!("loading {} blocks ...", ctx.config.num_hidden_layers);

        let mut blocks: Vec<Box<dyn Forwarder>> = vec![];
        let mut worker_connections = crate::spm::ClientPool::default();

        for i in 0..ctx.config.num_hidden_layers {
            let block_layer_name = format!("model.layers.{i}");
            if let Some((node_name, node)) = ctx.topology.get_node_for_layer(&block_layer_name) {
                log::debug!("node {node_name} will serve {}", &block_layer_name);
                let client = worker_connections
                    .client_for_layer(ctx.device.clone(), &node.host, &block_layer_name)
                    .await?;
                blocks.push(Box::new(client));
            } else {
                log::debug!("{} will be served locally", &block_layer_name);
                blocks.push(Transformer::load(
                    block_layer_name.clone(),
                    ctx.var_builder.pp(&block_layer_name),
                    &ctx.config,
                )?);
            }
        }

        for block in &blocks {
            log::info!("  {}", block)
        }

        let (tokenizer, eos_token_id) = load_tokenizer(&ctx)?;
        let tokens = vec![];
        let history = History::new();

        let logits_processor = create_logits_processor(&ctx);
        let index_pos = 0;

        log::info!(
            "model loaded - mem={}",
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        let generated = 0;

        Ok(Box::new(Self {
            tokenizer,
            tokens,
            generated,
            history,
            eos_token_id,
            index_pos,
            ctx,
            embedding,
            blocks,
            ln_f,
            lm_head,
            logits_processor,
        }))
    }

    /// Add a message to the chat history.
    fn add_message(&mut self, message: Message) -> Result<()> {
        self.history.push(message);
        Ok(())
    }

    /// Reset the chat pipeline state.
    fn reset(&mut self) -> Result<()> {
        self.tokens.clear();
        self.history.clear();
        self.ctx.cache.clear();
        self.index_pos = 0;
        self.generated = 0;
        Ok(())
    }

    /// Return the next token.
    async fn next_token(&mut self, index: usize) -> Result<Token> {
        // 记录调试信息，显示当前调用的函数和传入的索引
        // log::trace!("model.next_token({index})");
        // log::info!("Starting next_token function with index: {}", index);

        // 第一次使用聊天记录预填充令牌。
        if self.generated == 0 {
            self.start_dialog_prompt()?;
        }
        // 获取当前已有的 token 数量，其实可以使用self.generated代替吧
        let num_tokens = self.tokens.len();
        // log::info!("Number of existing tokens: {}", num_tokens);
        // // 根据是否使用 KV 缓存和索引值，确定上下文大小和上下文索引
        let (context_size, context_index) = if self.ctx.cache.with_kv_cache() && index > 0 {
            (1, self.index_pos)
        } else {
            (num_tokens, 0)
        };

        // 计算上下文偏移量
        let context_offset = num_tokens.saturating_sub(context_size);
        // log::info!("Context offset: {}", context_offset);
        // 获取上下文 token
        let context_tokens = &self.tokens[context_offset..];
        // log::info!("Context tokens: {:?}", context_tokens);

        let num_context_tokens = context_tokens.len();
        // log::info!("Number of context tokens: {}", num_context_tokens);

        // 这个才是在执行前向传播时的输入，在forward内部再执行embedding
        let input = Tensor::new(context_tokens, &self.ctx.device)?
            .unsqueeze(0)
            .map_err(|e| anyhow!("error squeezing context tokens: {e}"))?;
        // log::info!("Input tensor shape: {:?}", input.shape());

        // log::info!("input={:?} context_index={context_index}", input.shape());

        // 调用LLama的forward函数，会获得logits,是一个张量类型的
        // 每次调用会获得一个经过线性层处理之后的张量  得到的结果是【1，128256】的张量
        let logits = self
            .forward(&input, context_index)
            .await
            .map_err(|e| anyhow!("error in model.forward: {e}"))?;
        // log::info!("Logits tensor shape after forward pass: {:?}", logits.shape());

        // 对 logits 张量进行挤压操作，去除批次维度，并记录挤压后的形状信息
        let logits = logits
            .squeeze(0)
            .map_err(|e| anyhow!("error squeezing logits: {e}"))?;
        // log::info!("Logits tensor shape after squeezing: {:?}", logits.shape());

        // 如果 repeat_penalty 不等于 1，则应用重复惩罚，并记录相关信息
        let logits = if self.ctx.args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = num_tokens.saturating_sub(self.ctx.args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.ctx.args.repeat_penalty,
                &self.tokens[start_at..],
            )?
        };
        // log::info!("Logits tensor shape after applying repeat penalty: {:?}", logits.shape());

        self.index_pos += num_context_tokens;
        // log::info!("Updated index position: {}", self.index_pos);

        // 使用 logits_processor 对 logits 进行采样，得到下一个 token，并记录该信息
        // 在这里才会生成一个token,该token有两个流向，一个是作为Token进行返回，用作下一轮的输入
        // 另一个是加入到self.tokens，在其它地方进行异步输出
        let next_token = self
            .logits_processor
            .sample(&logits)
            .map_err(|e| anyhow!("error sampling logits {logits}: {e}"))?;

        // log::info!("next_token: {next_token}");

        // 更新 self.generated 和 self.index_pos，并记录相关信息
        self.generated += 1; // 每生成一个token,该变量加1
                             // log::info!("Total generated tokens: {}", self.generated);

        //  // 将下一个 token 添加到 self.tokens 列表中，并记录该信息，这里的token使用的是ID记录
        self.tokens.push(next_token);
        let text = match self.tokenizer.decode(&[next_token], false) {
            // 使用分词器解码token
            Ok(s) => Some(s),
            Err(e) => {
                log::error!("could not decode token {next_token}: {e}");
                None
            }
        };
        log::info!("text : {:?}", text);

        // 该函数的返回结果是一个Token结构体，包括token的ID、文本和是否为流的结束标记
        Ok(Token {
            id: next_token,
            text: match self.tokenizer.decode(&[next_token], false) {
                // 使用分词器解码token
                Ok(s) => Some(s),
                Err(e) => {
                    log::error!("could not decode token {next_token}: {e}");
                    None
                }
            },
            is_end_of_stream: Some(next_token) == self.eos_token_id,
        })
    }

    /// Return the number of generated tokens so far.
    fn generated_tokens(&self) -> usize {
        self.generated
    }
}
