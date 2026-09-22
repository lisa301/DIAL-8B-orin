//! Qwen3-VL vision encoder (ViT-like) and projector (merger).
//!
//! This is a pragmatic implementation aimed at the Qwen3-VL safetensors layout:
//! - `model.visual.patch_embed.proj.(weight|bias)`
//! - `model.visual.pos_embed.weight`
//! - `model.visual.blocks.{i}.(norm1|attn|norm2|mlp).*`
//! - `model.visual.merger.*` (projects to language hidden size)

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    Activation, Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder,
};

use super::VisionConfig;

fn rotate_half_vision(x: &Tensor) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    let last = *dims
        .last()
        .ok_or_else(|| candle_core::Error::msg("rotate_half_vision: empty dims"))?;
    if last % 2 != 0 {
        return Err(candle_core::Error::msg(format!(
            "rotate_half_vision: last dim {} is not even",
            last
        )));
    }
    let x1 = x.narrow(D::Minus1, 0, last / 2)?;
    let x2 = x.narrow(D::Minus1, last / 2, last / 2)?;
    let x2 = x2.neg()?;
    Tensor::cat(&[&x2, &x1], D::Minus1)
}

fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> candle_core::Result<(Tensor, Tensor)> {
    let orig_q_dtype = q.dtype();
    let orig_k_dtype = k.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    // q/k shape: (batch, heads, seq, head_dim)
    // cos/sin shape: (seq, head_dim) -> broadcast to (1, 1, seq, head_dim)
    let cos = cos
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(DType::F32)?
        .broadcast_as(q.shape().dims())?;
    let sin = sin
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(DType::F32)?
        .broadcast_as(q.shape().dims())?;
    let q_embed = ((&q * &cos)? + (rotate_half_vision(&q)? * &sin)?)?.to_dtype(orig_q_dtype)?;
    let k_embed = ((&k * &cos)? + (rotate_half_vision(&k)? * &sin)?)?.to_dtype(orig_k_dtype)?;
    Ok((q_embed, k_embed))
}

#[derive(Debug, Clone)]
struct VitAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    attn_f32: bool,
}

impl VitAttention {
    fn load(
        vb: VarBuilder,
        hidden: usize,
        num_heads: usize,
        attn_f32: bool,
    ) -> candle_core::Result<Self> {
        let head_dim = hidden / num_heads;
        Ok(Self {
            qkv: candle_nn::linear(hidden, 3 * hidden, vb.pp("qkv"))?,
            proj: candle_nn::linear(hidden, hidden, vb.pp("proj"))?,
            num_heads,
            head_dim,
            attn_f32,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        let qkv = self.qkv.forward(x)?; // (b, seq, 3*hidden)
        let qkv = qkv.reshape((b, seq, 3, self.num_heads, self.head_dim))?;
        let q = qkv
            .narrow(2, 0, 1)?
            .squeeze(2)?
            .transpose(1, 2)?
            .contiguous()?; // (b, h, seq, d)
        let k = qkv
            .narrow(2, 1, 1)?
            .squeeze(2)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = qkv
            .narrow(2, 2, 1)?
            .squeeze(2)?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin)?;

        let y = {
            let in_dtype = q.dtype();
            let compute_dtype = if self.attn_f32 { DType::F32 } else { in_dtype };
            let q = if compute_dtype == in_dtype {
                q
            } else {
                q.to_dtype(compute_dtype)?
            };
            let k = if compute_dtype == in_dtype {
                k
            } else {
                k.to_dtype(compute_dtype)?
            };
            let v = if compute_dtype == in_dtype {
                v
            } else {
                v.to_dtype(compute_dtype)?
            };
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            let y = att.matmul(&v.contiguous()?)?;
            if compute_dtype == in_dtype {
                y
            } else {
                y.to_dtype(in_dtype)?
            }
        };

        let y = y.transpose(1, 2)?.reshape((b, seq, hidden))?;
        self.proj.forward(&y)
    }
}

#[derive(Debug, Clone)]
struct VitMlp {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl VitMlp {
    fn load_with_dims(
        vb: VarBuilder,
        hidden: usize,
        intermediate: usize,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(hidden, intermediate, vb.pp("linear_fc1"))?,
            fc2: candle_nn::linear(intermediate, hidden, vb.pp("linear_fc2"))?,
            act: Activation::GeluPytorchTanh,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.fc2.forward(&self.act.forward(&self.fc1.forward(x)?)?)
    }
}

#[derive(Debug, Clone)]
struct VitBlock {
    norm1: LayerNorm,
    attn: VitAttention,
    norm2: LayerNorm,
    mlp: VitMlp,
}

impl VitBlock {
    fn load(vb: VarBuilder, cfg: &VisionConfig, attn_f32: bool) -> candle_core::Result<Self> {
        Ok(Self {
            norm1: candle_nn::layer_norm(cfg.hidden_size, 1e-5, vb.pp("norm1"))?,
            attn: VitAttention::load(vb.pp("attn"), cfg.hidden_size, cfg.num_heads, attn_f32)?,
            norm2: candle_nn::layer_norm(cfg.hidden_size, 1e-5, vb.pp("norm2"))?,
            mlp: VitMlp::load_with_dims(vb.pp("mlp"), cfg.hidden_size, cfg.intermediate_size)?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let x = (self.attn.forward(&self.norm1.forward(x)?, cos, sin)? + x)?;
        let x = (self.mlp.forward(&self.norm2.forward(&x)?)? + &x)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct Merger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    act: Activation,
    spatial_merge_size: usize,
}

impl Merger {
    fn load(
        vb: VarBuilder,
        token_hidden: usize,
        spatial_merge_size: usize,
        out_hidden: usize,
    ) -> candle_core::Result<Self> {
        let merged_hidden = token_hidden * spatial_merge_size * spatial_merge_size;
        Ok(Self {
            // Note: the Qwen3-VL safetensors layout stores merger.norm over the per-patch hidden
            // (before spatial merging), while the fc layers operate on merged hidden.
            norm: candle_nn::layer_norm(token_hidden, 1e-5, vb.pp("norm"))?,
            fc1: candle_nn::linear(merged_hidden, merged_hidden, vb.pp("linear_fc1"))?,
            fc2: candle_nn::linear(merged_hidden, out_hidden, vb.pp("linear_fc2"))?,
            act: Activation::GeluPytorchTanh,
            spatial_merge_size,
        })
    }

    fn spatial_merge(&self, x: &Tensor, hp: usize, wp: usize) -> candle_core::Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        if seq != hp * wp {
            return Err(candle_core::Error::msg(format!(
                "spatial_merge: seq {} != hp*wp {}*{}",
                seq, hp, wp
            )));
        }
        let m = self.spatial_merge_size;
        if m == 1 {
            return Ok(x.clone());
        }
        if hp % m != 0 || wp % m != 0 {
            return Err(candle_core::Error::msg(format!(
                "spatial_merge: patch grid {}x{} not divisible by merge size {}",
                hp, wp, m
            )));
        }

        // (b, seq, hidden) -> (b, hp, wp, hidden)
        let x = x.contiguous()?.reshape((b, hp, wp, hidden))?;
        // (b, hp, wp, hidden) -> (b, hp/m, m, wp/m, m, hidden)
        let x = x.reshape((b, hp / m, m, wp / m, m, hidden))?;
        // (b, hp/m, wp/m, m, m, hidden)
        let x = x.transpose(2, 3)?.contiguous()?;
        // (b, (hp/m)*(wp/m), hidden*m*m)
        x.reshape((b, (hp / m) * (wp / m), hidden * m * m))
    }

    fn forward(&self, x: &Tensor, hp: usize, wp: usize) -> candle_core::Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = self.spatial_merge(&x, hp, wp)?;
        let x = self.fc1.forward(&x)?;
        let x = self.act.forward(&x)?;
        self.fc2.forward(&x)
    }
}

#[derive(Debug, Clone)]
struct DeepMerger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    act: Activation,
    spatial_merge_size: usize,
}

impl DeepMerger {
    fn load(
        vb: VarBuilder,
        merged_hidden: usize,
        spatial_merge_size: usize,
        out_hidden: usize,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            // Deepstack merger applies norm after spatial merge, so norm dim = merged_hidden.
            norm: candle_nn::layer_norm(merged_hidden, 1e-5, vb.pp("norm"))?,
            fc1: candle_nn::linear(merged_hidden, merged_hidden, vb.pp("linear_fc1"))?,
            fc2: candle_nn::linear(merged_hidden, out_hidden, vb.pp("linear_fc2"))?,
            act: Activation::GeluPytorchTanh,
            spatial_merge_size,
        })
    }

    fn spatial_merge(&self, x: &Tensor, hp: usize, wp: usize) -> candle_core::Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        if seq != hp * wp {
            return Err(candle_core::Error::msg(format!(
                "deepstack spatial_merge: seq {} != hp*wp {}*{}",
                seq, hp, wp
            )));
        }
        let m = self.spatial_merge_size;
        if m == 1 {
            return Ok(x.clone());
        }
        if hp % m != 0 || wp % m != 0 {
            return Err(candle_core::Error::msg(format!(
                "deepstack spatial_merge: patch grid {}x{} not divisible by merge size {}",
                hp, wp, m
            )));
        }

        let x = x.contiguous()?.reshape((b, hp, wp, hidden))?;
        let x = x.reshape((b, hp / m, m, wp / m, m, hidden))?;
        let x = x.transpose(2, 3)?.contiguous()?;
        x.reshape((b, (hp / m) * (wp / m), hidden * m * m))
    }

    fn forward(&self, x: &Tensor, hp: usize, wp: usize) -> candle_core::Result<Tensor> {
        let x = self.spatial_merge(x, hp, wp)?;
        let x = self.norm.forward(&x)?;
        let x = self.fc1.forward(&x)?;
        let x = self.act.forward(&x)?;
        self.fc2.forward(&x)
    }
}

/// Vision encoder + merger producing language-hidden-size embeddings.
#[derive(Debug, Clone)]
pub struct VisionEncoder {
    cfg: VisionConfig,
    patch: Conv2d,
    pos_embed: Embedding,
    blocks: Vec<VitBlock>,
    merger: Merger,
    deepstack: Vec<(usize, DeepMerger)>,
    pos_cache: Arc<Mutex<HashMap<(usize, usize), Tensor>>>,
    rope_cache: Arc<Mutex<HashMap<(usize, usize, usize), (Tensor, Tensor)>>>,
}

static VISION_NORM_LOGGED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub struct VisionOutputs {
    pub embeds: Tensor,
    pub deepstack: Vec<Tensor>,
}

pub fn image_to_tensor(rgb: &[u8], h: usize, w: usize, device: &Device) -> Result<Tensor> {
    if rgb.len() != h * w * 3 {
        bail!(
            "invalid rgb buffer, expected {} bytes got {}",
            h * w * 3,
            rgb.len()
        );
    }

    let mode = std::env::var("SPM_VISION_NORM").unwrap_or_else(|_| "half".to_string());
    if !VISION_NORM_LOGGED.swap(true, Ordering::Relaxed) {
        log::info!("vision norm mode: {}", mode);
    }
    let (mean, std) = match mode.as_str() {
        "clip" => (
            [0.481_454_66_f32, 0.457_827_5_f32, 0.408_210_73_f32],
            [0.268_629_54_f32, 0.261_302_58_f32, 0.275_777_11_f32],
        ),
        "none" => ([0.0_f32, 0.0_f32, 0.0_f32], [1.0_f32, 1.0_f32, 1.0_f32]),
        _ => ([0.5_f32, 0.5_f32, 0.5_f32], [0.5_f32, 0.5_f32, 0.5_f32]),
    };

    let mut data = Vec::with_capacity(h * w * 3);
    for px in rgb.chunks_exact(3) {
        let r = px[0] as f32 / 255.0;
        let g = px[1] as f32 / 255.0;
        let b = px[2] as f32 / 255.0;
        data.push((r - mean[0]) / std[0]);
        data.push((g - mean[1]) / std[1]);
        data.push((b - mean[2]) / std[2]);
    }
    let t = Tensor::from_vec(data, (h, w, 3), device)?;
    let t = t.transpose(0, 2)?.transpose(1, 2)?;
    Ok(t.unsqueeze(0)?)
}

impl VisionEncoder {
    /// （新增）对外暴露必要的视觉配置参数，避免在上层直接访问私有字段。
    /// 为什么要加：图片缩放需要知道 patch_size / spatial_merge_size / num_position_embeddings 才能
    /// 按 Qwen3-VL 的 pipeline 做网格对齐与像素上下限约束。
    pub fn patch_size(&self) -> usize {
        self.cfg.patch_size
    }

    pub fn spatial_merge_size(&self) -> usize {
        self.cfg.spatial_merge_size
    }

    pub fn num_position_embeddings(&self) -> usize {
        self.cfg.num_position_embeddings
    }

    pub fn load(cfg: VisionConfig, vb: VarBuilder, attn_f32: bool) -> Result<Self> {
        if let Some(idx) = cfg.deepstack_visual_indexes.as_ref() {
            log::info!("vision_config.deepstack_visual_indexes: {:?}", idx);
        }
        // Patch embed weights are stored as [out, in, temporal, kh, kw] for video support.
        // For images, we fold the temporal dimension into channels and run a 2D conv with
        // in_channels = in_channels * temporal_patch_size by duplicating the image frames.
        let patch_vb = vb.pp("model.visual.patch_embed.proj");
        let w5 = patch_vb.get(
            (
                cfg.hidden_size,
                cfg.in_channels,
                cfg.temporal_patch_size,
                cfg.patch_size,
                cfg.patch_size,
            ),
            "weight",
        )?;
        let b = patch_vb.get(cfg.hidden_size, "bias")?;
        let w = w5.reshape((
            cfg.hidden_size,
            cfg.in_channels * cfg.temporal_patch_size,
            cfg.patch_size,
            cfg.patch_size,
        ))?;

        let mut conv_cfg = Conv2dConfig::default();
        conv_cfg.stride = cfg.patch_size;
        let patch = Conv2d::new(w, Some(b), conv_cfg);

        let pos_embed = candle_nn::embedding(
            cfg.num_position_embeddings,
            cfg.hidden_size,
            vb.pp("model.visual.pos_embed"),
        )?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(VitBlock::load(
                vb.pp(&format!("model.visual.blocks.{i}")),
                &cfg,
                attn_f32,
            )?);
        }

        let merger = Merger::load(
            vb.pp("model.visual.merger"),
            cfg.hidden_size,
            cfg.spatial_merge_size,
            cfg.out_hidden_size,
        )?;

        let mut deepstack = Vec::new();
        if let Some(indexes) = cfg.deepstack_visual_indexes.as_ref() {
            let merged_hidden = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
            for (i, idx) in indexes.iter().enumerate() {
                let dm = DeepMerger::load(
                    vb.pp(&format!("model.visual.deepstack_merger_list.{i}")),
                    merged_hidden,
                    cfg.spatial_merge_size,
                    cfg.out_hidden_size,
                )?;
                deepstack.push((*idx, dm));
            }
        }

        Ok(Self {
            cfg,
            patch,
            pos_embed,
            blocks,
            merger,
            deepstack,
            pos_cache: Arc::new(Mutex::new(HashMap::new())),
            rope_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn vision_rotary_embeddings(
        &self,
        hp: usize,
        wp: usize,
        head_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let key = (hp, wp, head_dim);
        if let Ok(cache) = self.rope_cache.lock() {
            if let Some((cos, sin)) = cache.get(&key) {
                return Ok((cos.clone(), sin.clone()));
            }
        }

        if head_dim % 2 != 0 {
            bail!("vision head_dim {} must be divisible by 2", head_dim);
        }

        // Match HF Qwen3-VL vision rotary:
        // - VisionRotaryEmbedding(dim=head_dim/2)
        // - inv_freq is built from arange(0, dim, 2), so length = head_dim / 4? No:
        //   for head_dim=72, dim=36, inv_freq length=18, and after combining row/col
        //   plus duplication we need a final width of 72.
        let rotary_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / 10_000f32.powf(i as f32 / rotary_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, inv_freq_len, device)?.to_dtype(DType::F32)?;

        let rows: Vec<u32> = (0..hp)
            .flat_map(|r| std::iter::repeat(r as u32).take(wp))
            .collect();
        let cols: Vec<u32> = (0..hp).flat_map(|_| (0..wp).map(|c| c as u32)).collect();

        let row = Tensor::from_vec(rows, hp * wp, device)?
            .to_dtype(DType::F32)?
            .reshape((hp * wp, 1))?;
        let col = Tensor::from_vec(cols, hp * wp, device)?
            .to_dtype(DType::F32)?
            .reshape((hp * wp, 1))?;
        let inv = inv_freq.reshape((1, inv_freq.elem_count()))?;
        let row_freqs = row.matmul(&inv)?;
        let col_freqs = col.matmul(&inv)?;
        let rotary = Tensor::cat(&[&row_freqs, &col_freqs], D::Minus1)?;
        let emb = Tensor::cat(&[&rotary, &rotary], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;

        if let Ok(mut cache) = self.rope_cache.lock() {
            cache.insert(key, (cos.clone(), sin.clone()));
        }

        Ok((cos, sin))
    }

    fn clamp_and_round_hw(&self, h: u32, w: u32) -> (usize, usize) {
        // Ensure patch count doesn't exceed num_position_embeddings.
        let max_side = 768u32; // 48*16 => 2304 positions max for square images.
        let mut nh = h;
        let mut nw = w;
        if nh.max(nw) > max_side {
            let scale = max_side as f32 / nh.max(nw) as f32;
            nh = (nh as f32 * scale) as u32;
            nw = (nw as f32 * scale) as u32;
        }
        // Round down to multiples of patch size.
        let ps = self.cfg.patch_size as u32;
        nh = (nh / ps).max(1) * ps;
        nw = (nw / ps).max(1) * ps;
        (nh as usize, nw as usize)
    }

    /// Convert an RGB image tensor in `[H,W,3]` (u8 or f32) into a model input tensor `[1,3,H,W]` (f32),
    /// applying `(x/255 - mean)/std` with mean/std 0.5.
    pub fn image_to_tensor(
        &self,
        rgb: &[u8],
        h: usize,
        w: usize,
        device: &Device,
    ) -> Result<Tensor> {
        image_to_tensor(rgb, h, w, device)
    }

    /// Encode an image input tensor `[1,3,H,W]` into projected features plus deepstack side outputs.
    pub fn encode(&self, image: &Tensor) -> Result<VisionOutputs> {
        let image = if self.cfg.temporal_patch_size <= 1 {
            image.clone()
        } else {
            let temporal = self.cfg.temporal_patch_size;
            let frames = image.repeat((temporal, 1, 1, 1))?;
            self.pack_temporal_channels(&frames)?
        };
        self.encode_packed(&image)
    }

    /// Encode one consecutive temporal patch `[T,3,H,W]` without sampling frames.
    pub fn encode_video_patch(&self, frames: &Tensor) -> Result<VisionOutputs> {
        let (temporal, channels, _h, _w) = frames
            .dims4()
            .map_err(|e| anyhow!("video patch dims4 -> {e}"))?;
        if temporal != self.cfg.temporal_patch_size || channels != self.cfg.in_channels {
            bail!(
                "video patch shape must be [{},{},H,W], got {:?}",
                self.cfg.temporal_patch_size,
                self.cfg.in_channels,
                frames.dims()
            );
        }
        self.encode_video_patches(&frames.unsqueeze(0)?)
    }

    /// Encode a batch of consecutive temporal patches `[B,T,3,H,W]`.
    ///
    /// Each temporal patch remains an independent spatial attention sequence,
    /// matching Qwen3-VL's video processor while amortizing ViT kernel setup.
    pub fn encode_video_patches(&self, frames: &Tensor) -> Result<VisionOutputs> {
        let (batch, temporal, channels, h, w) = frames
            .dims5()
            .map_err(|e| anyhow!("video patches dims5 -> {e}"))?;
        if temporal != self.cfg.temporal_patch_size || channels != self.cfg.in_channels {
            bail!(
                "video patches shape must be [B,{}, {},H,W], got {:?}",
                self.cfg.temporal_patch_size,
                self.cfg.in_channels,
                frames.dims()
            );
        }
        let packed =
            frames
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, channels * temporal, h, w))?;
        self.encode_packed(&packed)
    }

    fn pack_temporal_channels(&self, frames: &Tensor) -> Result<Tensor> {
        let (temporal, channels, h, w) = frames.dims4()?;
        // Conv3d weights are flattened in C-major, then T order.
        Ok(frames
            .transpose(0, 1)?
            .contiguous()?
            .reshape((1, channels * temporal, h, w))?)
    }

    fn encode_packed(&self, image: &Tensor) -> Result<VisionOutputs> {
        let t0 = Instant::now();
        let (_b, _c, h, w) = image.dims4().map_err(|e| anyhow!("image dims4 -> {e}"))?;

        // Patch embedding.
        let t_patch = Instant::now();
        let x = self.patch.forward(image)?; // (1, hidden, h', w')

        let (_b, _hidden, hp0, wp0) = x.dims4()?;
        let patch_s = t_patch.elapsed().as_secs_f64();

        // Ensure the patch grid is compatible with spatial merging by cropping to multiples.
        let m = self.cfg.spatial_merge_size.max(1);
        if hp0 < m || wp0 < m {
            bail!(
                "image too small for spatial_merge_size {}: patch grid {}x{} (try larger image)",
                m,
                hp0,
                wp0
            );
        }
        let hp = (hp0 / m) * m;
        let wp = (wp0 / m) * m;
        let x = if hp != hp0 || wp != wp0 {
            x.narrow(2, 0, hp)?.narrow(3, 0, wp)?
        } else {
            x
        };
        let seq = hp * wp;

        if seq > self.cfg.num_position_embeddings {
            bail!(
                "too many visual tokens: {} ({}x{}), max is {} (try smaller image)",
                seq,
                hp,
                wp,
                self.cfg.num_position_embeddings
            );
        }

        let x = x.flatten_from(2)?.transpose(1, 2)?.contiguous()?; // (1, seq, hidden)

        // Add pos embeddings.
        let t_pos = Instant::now();
        //
        // 关键点：pos_embed 是按 base_side x base_side 网格展平存储的（num_position_embeddings=2304 => 48x48）。
        // 仅仅截取左上角的子网格会显著破坏位置分布（尤其是 448/672 这种非 48x48 网格），
        // 容易导致“图像损坏/噪点”的输出。这里改为对 pos_embed 做 2D resize（nearest），
        // 让任意 patch grid 都能有一致的全局位置信息。
        let pos = {
            if let Ok(cache) = self.pos_cache.lock() {
                cache.get(&(hp, wp)).cloned()
            } else {
                None
            }
        };
        let pos = match pos {
            Some(pos) => pos,
            None => {
                let base_seq = self.cfg.num_position_embeddings;
                let base_side = (base_seq as f64).sqrt() as usize;
                if base_side * base_side != base_seq {
                    bail!("num_position_embeddings {} is not a square", base_seq);
                }
                if hp > base_side || wp > base_side {
                    bail!(
                        "patch grid {}x{} exceeds base {}x{} (resize image smaller)",
                        hp,
                        wp,
                        base_side,
                        base_side
                    );
                }

                // Build full base pos table: (base_seq, hidden).
                let base_ids =
                    Tensor::arange(0u32, base_seq as u32, image.device())?.to_dtype(DType::U32)?;
                let base = self.pos_embed.forward(&base_ids)?;
                let hidden = base.dims2()?.1;

                // (base_seq, hidden) -> (1, base_side, base_side, hidden)
                let base = base.reshape((1, base_side, base_side, hidden))?;
                // (1, base_side, base_side, hidden) -> (1, hidden, base_side, base_side)
                let base = base.transpose(1, 3)?.transpose(2, 3)?;

                // Resize to target grid (nearest, candle supports on cpu/cuda/metal).
                let resized = if hp == base_side && wp == base_side {
                    base
                } else {
                    base.interpolate2d(hp, wp)?
                };

                // (1, hidden, hp, wp) -> (1, hp, wp, hidden) -> (1, seq, hidden)
                let resized = resized.transpose(1, 3)?.transpose(1, 2)?;
                let pos = resized.reshape((1, hp * wp, hidden))?;

                if let Ok(mut cache) = self.pos_cache.lock() {
                    cache.insert((hp, wp), pos.clone());
                }
                pos
            }
        };

        let mut x = (x + pos)?;
        let pos_s = t_pos.elapsed().as_secs_f64();

        // ViT blocks.
        let t_blocks = Instant::now();
        let mut deep_outputs: Vec<Tensor> = Vec::new();
        let (cos, sin) = self.vision_rotary_embeddings(
            hp,
            wp,
            self.cfg.hidden_size / self.cfg.num_heads,
            image.device(),
            x.dtype(),
        )?;
        for (i, blk) in self.blocks.iter().enumerate() {
            x = blk.forward(&x, &cos, &sin)?;
            for (idx, dm) in &self.deepstack {
                if *idx == i {
                    let d = dm.forward(&x, hp, wp)?;
                    deep_outputs.push(d);
                }
            }
        }
        let blocks_s = t_blocks.elapsed().as_secs_f64();

        // Project to language hidden size.
        let t_merger = Instant::now();
        let x = self.merger.forward(&x, hp, wp)?;
        let merger_s = t_merger.elapsed().as_secs_f64();

        let total_s = t0.elapsed().as_secs_f64();
        // 用 info 打印，方便定位 TTFT 慢到底卡在 patch/pos/blocks/merger 哪一段。
        log::info!(
            "vision encoded: input_hw={}x{} patch_grid={}x{} seq={} (patch_s={:.3} pos_s={:.3} blocks_s={:.3} merger_s={:.3} total_s={:.3})",
            h,
            w,
            hp,
            wp,
            seq
            ,patch_s
            ,pos_s
            ,blocks_s
            ,merger_s
            ,total_s
        );

        Ok(VisionOutputs {
            embeds: x,
            deepstack: deep_outputs,
        })
    }
}
