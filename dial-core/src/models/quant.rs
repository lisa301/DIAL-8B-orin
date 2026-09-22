use anyhow::{anyhow, Result};
use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{
    linear as linear_with_bias, linear_no_bias as linear_no_bias, Linear, Module, VarBuilder,
};
use regex::Regex;
use serde::Deserialize;

/// Weight quantization modes supported by this project.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightQuant {
    FP16,
    INT8,
}

impl WeightQuant {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "fp16" | "f16" => Ok(Self::FP16),
            "int8" | "i8" | "w8" => Ok(Self::INT8),
            other => Err(anyhow!("unsupported weight quant '{other}'")),
        }
    }
}

#[derive(Clone, Debug)]
struct QuantRule {
    pattern: String,
    regex: Regex,
    weight: WeightQuant,
}

/// Layer-wise quantization plan.
#[derive(Clone, Debug)]
pub struct QuantPlan {
    default: WeightQuant,
    rules: Vec<QuantRule>,
}

#[derive(Deserialize)]
struct QuantRuleFile {
    #[serde(rename = "match")]
    matcher: String,
    weight: String,
}

#[derive(Deserialize)]
struct QuantFile {
    default: Option<String>,
    layers: Option<Vec<QuantRuleFile>>,
}

impl QuantPlan {
    pub fn from_path(path: &str) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("can't read quant config {path}: {e}"))?;
        let file: QuantFile = serde_yaml::from_str(&raw)
            .map_err(|e| anyhow!("can't parse quant config {path}: {e}"))?;
        let default = match file.default {
            Some(v) => WeightQuant::from_str(&v)?,
            None => WeightQuant::FP16,
        };
        let mut rules = Vec::new();
        if let Some(items) = file.layers {
            for item in items {
                let regex = pattern_to_regex(&item.matcher)?;
                let weight = WeightQuant::from_str(&item.weight)?;
                rules.push(QuantRule {
                    pattern: item.matcher,
                    regex,
                    weight,
                });
            }
        }
        Ok(Self { default, rules })
    }

    pub fn weight_for(&self, name: &str) -> WeightQuant {
        for rule in &self.rules {
            if rule.regex.is_match(name) {
                return rule.weight;
            }
        }
        self.default
    }
}

fn pattern_to_regex(pattern: &str) -> Result<Regex> {
    let escaped = regex::escape(pattern);
    let regex_str = format!("^{}$", escaped.replace("\\*", ".*"));
    Regex::new(&regex_str).map_err(|e| anyhow!("invalid pattern {pattern}: {e}"))
}

/// A linear layer that can be either FP16/FP32 or INT8-dequantized on the fly.
#[derive(Debug, Clone)]
pub enum LinearKind {
    FP16(Linear),
    Q8(QuantLinear),
}

impl LinearKind {
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::FP16(l) => l.forward(x),
            Self::Q8(l) => l.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantLinear {
    qweight: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
    out_features: usize,
    in_features: usize,
}

impl QuantLinear {
    fn load(vb: VarBuilder, out_features: usize, in_features: usize) -> CandleResult<Self> {
        let qweight = vb.get((out_features, in_features), "weight.q8")?;
        let scales = vb.get(out_features, "weight.scale")?;
        let bias = vb.get(out_features, "bias").ok();
        Ok(Self {
            qweight,
            scales,
            bias,
            out_features,
            in_features,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let in_dtype = x.dtype();
        let q = self.qweight.to_dtype(in_dtype)?;
        let zp = Tensor::new(128f32, q.device())?
            .to_dtype(in_dtype)?
            .broadcast_as(q.shape())?;
        let q = (q - zp)?;
        let s = self
            .scales
            .to_dtype(in_dtype)?
            .reshape((self.out_features, 1))?
            .broadcast_as(q.shape())?;
        let w = (q * s)?;
        let wt = w.t()?;
        let mut y = match x.rank() {
            2 => x.matmul(&wt)?,
            3 => {
                let (b, s_len, hidden) = x.dims3()?;
                if hidden != self.in_features {
                    return Err(candle_core::Error::msg(format!(
                        "quant linear input mismatch: got {hidden}, expect {}",
                        self.in_features
                    )));
                }
                let x2 = x.reshape((b * s_len, hidden))?;
                let y2 = x2.matmul(&wt)?;
                y2.reshape((b, s_len, self.out_features))?
            }
            other => {
                return Err(candle_core::Error::msg(format!(
                    "quant linear unsupported rank {other}"
                )))
            }
        };
        if let Some(bias) = &self.bias {
            y = (y + bias)?;
        }
        Ok(y)
    }
}

/// Load a linear layer based on the quant plan (by full weight name).
pub fn load_linear(
    vb: VarBuilder,
    in_features: usize,
    out_features: usize,
    full_weight_name: &str,
    quant: Option<&QuantPlan>,
    with_bias: bool,
) -> CandleResult<LinearKind> {
    let want = quant.map(|q| q.weight_for(full_weight_name));
    if want == Some(WeightQuant::INT8) {
        match QuantLinear::load(vb.clone(), out_features, in_features) {
            Ok(q) => return Ok(LinearKind::Q8(q)),
            Err(e) => {
                log::warn!(
                    "quant weight not found for {} ({}), falling back to fp16",
                    full_weight_name,
                    e
                );
            }
        }
    }

    let lin = if with_bias {
        linear_with_bias(in_features, out_features, vb)?
    } else {
        linear_no_bias(in_features, out_features, vb)?
    };
    Ok(LinearKind::FP16(lin))
}
