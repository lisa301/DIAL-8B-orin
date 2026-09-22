use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::Config;

/// Abstraction over cosine and sine tables, kv-caching and attention masking.
#[derive(Debug, Clone)]
pub struct Cache {
    cos: Tensor,
    sin: Tensor,

    masks: HashMap<usize, Tensor>,
    use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,

    device: Device,
    max_seq_len: usize,
}

impl Cache {
    /// Creates a new cache instance with the provided configuration.
    /// Set `use_kv_cache` to false to disable kv-caching.
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        let max_seq_len = config.max_seq_len;
        // precompute freqs_cis
        let n_elem = config.hidden_size / config.num_attention_heads;

        log::debug!("cache::n_elem = {n_elem}");

        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
            .collect();

        let theta = Tensor::new(theta.as_slice(), device)?;

        log::debug!("cache::theta = {}", &theta);

        let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        log::debug!("cache::idx_theta = {}", &idx_theta);

        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        log::debug!("cache::cos = {}", &cos);
        log::debug!("cache::sin = {}", &sin);

        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            device: device.clone(),
            cos,
            sin,
            max_seq_len,
        })
    }

    /// Return true if kv-caching is enabled.
    pub fn with_kv_cache(&self) -> bool {
        self.use_kv_cache
    }

    /// Return the cached cosine value for the given position and sequence length.
    pub fn cosine(&self, index_pos: usize, seq_len: usize) -> Result<Tensor> {
        self.cos.narrow(0, index_pos, seq_len)
    }

    /// Return the cached sine value for the given position and sequence length.
    pub fn sine(&self, index_pos: usize, seq_len: usize) -> Result<Tensor> {
        self.sin.narrow(0, index_pos, seq_len)
    }

    /// Get the attention mask for the given sequence length.
    pub fn mask(&mut self, seq_len: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&seq_len) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?;
            self.masks.insert(seq_len, mask.clone());
            Ok(mask)
        }
    }

    /// Process the input k and v by either generating their cache entry or applying a previously cached one.
    /// 这个操作是要更新缓存的
    pub fn process_kv(
        &mut self,
        block_idx: usize,
        index_pos: usize,
        mut k: Tensor,
        mut v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if self.use_kv_cache {
            // （新增）当一个“新请求/新对话”的 prefill 开始时，index_pos 会回到 0。
            //
            // 为什么要加：在分布式推理时，master 和 worker 之间是长连接，worker 侧的 KV-cache
            // 如果不在新请求开始时重置，就会把“上一次请求的 k/v”继续 append，导致注意力矩阵的
            // key_len 变大（例如变成 164），但 mask 仍按本次 seq_len（例如 24）生成，从而触发
            // `cannot broadcast [24, 24] to [1, 32, 24, 164]` 这类崩溃。
            //
            // 实现：index_pos==0 时直接覆盖该 block 的 kv 条目，而不是拼接旧缓存。
            if index_pos == 0 {
                self.kvs[block_idx] = Some((k.clone(), v.clone()));
                return Ok((k, v));
            }

            // if this block_idx in cache
            if let Some((cache_k, cache_v)) = &self.kvs[block_idx] {
                // update cache entry
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[2];
                if k_seq_len > self.max_seq_len {
                    k = k
                        .narrow(2, k_seq_len - self.max_seq_len, self.max_seq_len)?
                        .contiguous()?
                }
                let v_seq_len = v.dims()[2];
                if v_seq_len > self.max_seq_len {
                    v = v
                        .narrow(2, v_seq_len - self.max_seq_len, self.max_seq_len)?
                        .contiguous()?
                }
            }
            // set entry for this block
            self.kvs[block_idx] = Some((k.clone(), v.clone()))
        }
        Ok((k, v))
    }

    /// Decode fast path: update an existing preallocated KV cache in-place.
    ///
    /// The generic `process_kv` path grows cache with `cat + contiguous` on every decode token,
    /// which repeatedly copies the whole historical KV tensor. During single-token decode we
    /// preallocate up to `max_seq_len`, write the new token with `slice_set`, and return a
    /// narrowed view for attention. `slice_set` is backed by a device copy on both CPU and
    /// CUDA, so the same allocation-free decode path is used on the Orin worker.
    pub fn process_kv_decode_in_place(
        &mut self,
        block_idx: usize,
        index_pos: usize,
        k: Tensor,
        v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if !self.use_kv_cache || index_pos == 0 || index_pos >= self.max_seq_len {
            return self.process_kv(block_idx, index_pos, k, v);
        }

        let dims = k.dims();
        if dims.len() != 4 || dims[2] != 1 {
            return self.process_kv(block_idx, index_pos, k, v);
        }
        let [b_sz, kv_heads, _, head_dim]: [usize; 4] = dims.try_into().map_err(|_| {
            candle_core::Error::Msg(format!("unexpected kv rank for decode cache: {dims:?}"))
        })?;
        let dst_pos = index_pos.min(self.max_seq_len.saturating_sub(1));
        let active_len = index_pos.saturating_add(1).min(self.max_seq_len);

        let needs_alloc = match &self.kvs[block_idx] {
            Some((cache_k, cache_v)) => {
                cache_k.dims() != [b_sz, kv_heads, self.max_seq_len, head_dim]
                    || cache_v.dims() != [b_sz, kv_heads, self.max_seq_len, head_dim]
                    || !cache_k.is_contiguous()
                    || !cache_v.is_contiguous()
                    || cache_k.dtype() != k.dtype()
                    || cache_v.dtype() != v.dtype()
            }
            None => true,
        };

        if needs_alloc {
            let cache_k = Tensor::zeros(
                (b_sz, kv_heads, self.max_seq_len, head_dim),
                k.dtype(),
                k.device(),
            )?;
            let cache_v = Tensor::zeros(
                (b_sz, kv_heads, self.max_seq_len, head_dim),
                v.dtype(),
                v.device(),
            )?;
            if index_pos > 0 {
                if let Some((old_k, old_v)) = self.kvs[block_idx].take() {
                    let old_len = old_k.dims()[2].min(dst_pos);
                    if old_len > 0 {
                        cache_k.slice_set(&old_k.narrow(2, 0, old_len)?.contiguous()?, 2, 0)?;
                        cache_v.slice_set(&old_v.narrow(2, 0, old_len)?.contiguous()?, 2, 0)?;
                    }
                }
            }
            self.kvs[block_idx] = Some((cache_k, cache_v));
        }

        let (cache_k, cache_v) = self.kvs[block_idx]
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("missing preallocated kv cache".into()))?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        cache_k.slice_set(&k, 2, dst_pos)?;
        cache_v.slice_set(&v, 2, dst_pos)?;

        let start = index_pos.saturating_add(1).saturating_sub(self.max_seq_len);
        let out_k = cache_k.narrow(2, start, active_len)?;
        let out_v = cache_v.narrow(2, start, active_len)?;
        Ok((out_k, out_v))
    }

    /// Return a copy of this cache with the same state but new kv table.
    pub fn as_new(&self) -> Self {
        let mut copy = self.clone();
        copy.clear();
        copy
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.masks.clear();
        self.kvs = vec![None; self.kvs.len()];
    }

    /// Clone KV tensors for a given layer if present.
    pub fn kv_clone(&self, block_idx: usize) -> Option<(Tensor, Tensor)> {
        self.kvs
            .get(block_idx)
            .and_then(|entry| entry.as_ref().map(|(k, v)| (k.clone(), v.clone())))
    }

    /// Overwrite KV tensors for a given layer.
    pub fn set_kv(&mut self, block_idx: usize, mut k: Tensor, mut v: Tensor) -> Result<()> {
        if block_idx >= self.kvs.len() {
            candle_core::bail!(
                "cache layer index out of range: {} >= {}",
                block_idx,
                self.kvs.len()
            );
        }
        let k_seq_len = k.dims()[2];
        if k_seq_len > self.max_seq_len {
            k = k
                .narrow(2, k_seq_len - self.max_seq_len, self.max_seq_len)?
                .contiguous()?;
        }
        let v_seq_len = v.dims()[2];
        if v_seq_len > self.max_seq_len {
            v = v
                .narrow(2, v_seq_len - self.max_seq_len, self.max_seq_len)?
                .contiguous()?;
        }
        self.kvs[block_idx] = Some((k, v));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn test_config(num_layers: usize, max_seq_len: usize) -> Config {
        Config {
            hidden_size: 32,
            intermediate_size: 64,
            vocab_size: 128,
            num_hidden_layers: num_layers,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            bos_token_id: None,
            eos_token_id: None,
            max_seq_len,
            attn_f32: false,
        }
    }

    #[test]
    fn process_kv_replaces_on_index_pos_zero() -> Result<()> {
        let device = Device::Cpu;
        let cfg = test_config(2, 64);
        let mut cache = Cache::new(true, DType::F32, &cfg, &device)?;

        // （新增）回归测试：先模拟一次“续写”导致 cache 已存在；
        // 再模拟一次“新请求 prefill（index_pos=0）”，验证不会 append 旧缓存。
        //
        // Shapes: (b, kv_heads, seq, head_dim)
        let k1 = Tensor::zeros((1, 4, 10, 8), DType::F32, &device)?;
        let v1 = Tensor::zeros((1, 4, 10, 8), DType::F32, &device)?;
        let (k, _v) = cache.process_kv(0, 1, k1, v1)?;
        assert_eq!(k.dims()[2], 10);

        // A new prefill should replace instead of append.
        let k2 = Tensor::zeros((1, 4, 7, 8), DType::F32, &device)?;
        let v2 = Tensor::zeros((1, 4, 7, 8), DType::F32, &device)?;
        let (k, _v) = cache.process_kv(0, 0, k2, v2)?;
        assert_eq!(k.dims()[2], 7);

        Ok(())
    }

    fn assert_decode_cache_is_preallocated(device: &Device) -> Result<()> {
        let cfg = test_config(1, 8);
        let mut cache = Cache::new(true, DType::F32, &cfg, device)?;

        // Prefill creates the normal short cache. The first decode call must migrate it once
        // into fixed-capacity device storage; subsequent tokens only update one position.
        let prefill_k = Tensor::from_vec(
            (0..16).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 1, 2, 8),
            device,
        )?;
        let prefill_v = (&prefill_k + 100f64)?;
        cache.process_kv(0, 0, prefill_k, prefill_v)?;

        for index_pos in 2..5 {
            let token_k = Tensor::full(index_pos as f32, (1, 1, 1, 8), device)?;
            let token_v = Tensor::full((index_pos + 100) as f32, (1, 1, 1, 8), device)?;
            let (active_k, active_v) =
                cache.process_kv_decode_in_place(0, index_pos, token_k, token_v)?;

            assert_eq!(active_k.dims(), [1, 1, index_pos + 1, 8]);
            assert_eq!(active_v.dims(), [1, 1, index_pos + 1, 8]);
            let (storage_k, storage_v) = cache.kvs[0].as_ref().unwrap();
            assert_eq!(storage_k.dims(), [1, 1, cfg.max_seq_len, 8]);
            assert_eq!(storage_v.dims(), [1, 1, cfg.max_seq_len, 8]);
        }

        let (active_k, active_v) = cache.process_kv_decode_in_place(
            0,
            5,
            Tensor::full(5f32, (1, 1, 1, 8), device)?,
            Tensor::full(105f32, (1, 1, 1, 8), device)?,
        )?;
        assert_eq!(active_k.flatten_all()?.to_vec1::<f32>()?[16], 2f32);
        assert_eq!(active_k.flatten_all()?.to_vec1::<f32>()?[40], 5f32);
        assert_eq!(active_v.flatten_all()?.to_vec1::<f32>()?[40], 105f32);
        Ok(())
    }

    #[test]
    fn decode_cache_is_preallocated_on_cpu() -> Result<()> {
        assert_decode_cache_is_preallocated(&Device::Cpu)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn decode_cache_is_preallocated_on_cuda() -> Result<()> {
        assert_decode_cache_is_preallocated(&Device::new_cuda(0)?)
    }
}
