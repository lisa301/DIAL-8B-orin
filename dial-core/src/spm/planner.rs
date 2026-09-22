use std::{collections::HashMap, fs, ops::Range, path::Path};

use anyhow::Result;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::models::llama3::Config;

use super::{Node, Topology};

const MIB: f64 = 1024.0 * 1024.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum PlannerAlgorithm {
    /// DIAL's TTFT/TPOT/energy-aware objective.
    #[default]
    #[serde(rename = "dial")]
    #[value(name = "dial")]
    Dial,
    /// EdgeShard Algorithm 1 objective, constrained to executable continuous shards.
    #[serde(rename = "edgeshard-latency")]
    #[value(name = "edgeshard-latency")]
    EdgeShardLatency,
    /// EdgeShard Algorithm 2 bottleneck objective, adapted to DIAL's star transport.
    #[serde(rename = "edgeshard-throughput")]
    #[value(name = "edgeshard-throughput")]
    EdgeShardThroughput,
}

impl PlannerAlgorithm {
    fn score_definition(self) -> &'static str {
        match self {
            Self::Dial => "risk_weighted_mean_and_tail_ttft_tpot_energy_and_remote_workers",
            Self::EdgeShardLatency => "estimated_end_to_end_request_latency_ms",
            Self::EdgeShardThroughput => "estimated_pipeline_bottleneck_ms_per_request",
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_dtype_bytes() -> usize {
    2
}

fn default_prompt_tokens() -> usize {
    512
}

fn default_output_tokens() -> usize {
    128
}

fn default_ttft_weight() -> f64 {
    0.6
}

fn default_tpot_weight() -> f64 {
    0.4
}

fn default_min_devices() -> usize {
    1
}

/// A profile metric can be uniform across all layers or measured per layer.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum LayerMetric {
    Uniform(f64),
    PerLayer(Vec<f64>),
}

impl Default for LayerMetric {
    fn default() -> Self {
        Self::Uniform(0.0)
    }
}

impl LayerMetric {
    fn validate(&self, field: &str, num_layers: usize) -> Result<()> {
        match self {
            Self::Uniform(value) => {
                if !value.is_finite() || *value < 0.0 {
                    bail!("{field} must be finite and non-negative, got {value}");
                }
            }
            Self::PerLayer(values) => {
                if values.len() != num_layers {
                    bail!(
                        "{field} has {} values but the model has {num_layers} layers",
                        values.len()
                    );
                }
                if values
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                {
                    bail!("{field} values must be finite and non-negative");
                }
            }
        }
        Ok(())
    }

    fn sum(&self, range: Range<usize>) -> f64 {
        match self {
            Self::Uniform(value) => *value * range.len() as f64,
            Self::PerLayer(values) => values[range].iter().sum(),
        }
    }

    fn value_at(&self, layer: usize) -> f64 {
        match self {
            Self::Uniform(value) => *value,
            Self::PerLayer(values) => values[layer],
        }
    }

    fn validate_not_below(
        &self,
        baseline: &Self,
        field: &str,
        baseline_field: &str,
        num_layers: usize,
    ) -> Result<()> {
        for layer in 0..num_layers {
            let value = self.value_at(layer);
            let baseline_value = baseline.value_at(layer);
            if value < baseline_value {
                bail!(
                    "{field} at layer {layer} ({value}) must be no smaller than {baseline_field} ({baseline_value})"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannerRole {
    Master,
    Worker,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PlannerDevice {
    pub name: String,
    pub role: PlannerRole,
    pub host: Option<String>,
    pub description: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub usable_memory_mb: f64,
    #[serde(default)]
    pub fixed_memory_mb: f64,
    pub layer_memory_mb: LayerMetric,
    pub prefill_ms: LayerMetric,
    pub decode_ms: LayerMetric,
    /// Optional per-layer P95 profile. Missing values fall back to the mean profile.
    #[serde(default)]
    pub prefill_p95_ms: Option<LayerMetric>,
    /// Optional per-layer P95 profile. Missing values fall back to the mean profile.
    #[serde(default)]
    pub decode_p95_ms: Option<LayerMetric>,
    #[serde(default)]
    pub prefill_energy_mj: LayerMetric,
    #[serde(default)]
    pub decode_energy_mj: LayerMetric,
    /// Master-to-worker round-trip latency. Ignored for the master.
    #[serde(default)]
    pub rtt_ms: f64,
    /// P95 round-trip latency used by DIAL's conservative tail scenario.
    pub rtt_p95_ms: Option<f64>,
    /// Measured master-to-worker throughput. Ignored for the master.
    pub bandwidth_mbps: Option<f64>,
    /// P05 throughput used by DIAL's conservative tail scenario.
    pub bandwidth_p05_mbps: Option<f64>,
    /// Serialization, protocol, and dispatch overhead for one round trip.
    #[serde(default)]
    pub protocol_ms: f64,
    /// P95 serialization, protocol, and dispatch overhead.
    pub protocol_p95_ms: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PlannerObjective {
    #[serde(default = "default_prompt_tokens")]
    pub prompt_tokens: usize,
    #[serde(default = "default_output_tokens")]
    pub output_tokens: usize,
    #[serde(default = "default_dtype_bytes")]
    pub dtype_bytes: usize,
    /// Context reserved in the KV cache. Zero uses the runtime max sequence length.
    #[serde(default)]
    pub kv_context_tokens: usize,
    #[serde(default = "default_ttft_weight")]
    pub ttft_weight: f64,
    #[serde(default = "default_tpot_weight")]
    pub tpot_weight: f64,
    /// Convex weight in [0, 1] for the conservative P95/P05 tail scenario.
    #[serde(default)]
    pub risk_weight: f64,
    #[serde(default)]
    pub energy_weight: f64,
    /// Optional regularizer for operational complexity, applied per remote worker used.
    #[serde(default)]
    pub remote_device_penalty: f64,
    /// Lower bound used for device-subset ablations. Normal planning should keep this at one.
    #[serde(default = "default_min_devices")]
    pub min_devices: usize,
    /// Optional upper bound. Omit it to allow every enabled device.
    pub max_devices: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PlannerProfile {
    #[serde(default)]
    pub version: u32,
    pub objective: PlannerObjective,
    pub devices: Vec<PlannerDevice>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PlanCost {
    pub ttft_ms: f64,
    pub tpot_ms: f64,
    pub tail_ttft_ms: f64,
    pub tail_tpot_ms: f64,
    pub request_energy_mj: f64,
    pub score: f64,
}

impl PlanCost {
    fn add(&self, other: &Self) -> Self {
        Self {
            ttft_ms: self.ttft_ms + other.ttft_ms,
            tpot_ms: self.tpot_ms + other.tpot_ms,
            tail_ttft_ms: self.tail_ttft_ms + other.tail_ttft_ms,
            tail_tpot_ms: self.tail_tpot_ms + other.tail_tpot_ms,
            request_energy_mj: self.request_energy_mj + other.request_energy_mj,
            score: self.score + other.score,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PlannedSegment {
    pub device: String,
    pub role: String,
    pub start_layer: usize,
    pub end_layer: usize,
    pub num_layers: usize,
    pub estimated_memory_mb: f64,
    pub estimated_ttft_ms: f64,
    pub estimated_tpot_ms: f64,
    pub estimated_tail_ttft_ms: f64,
    pub estimated_tail_tpot_ms: f64,
}

#[derive(Clone, Debug)]
struct SearchSegment {
    device_idx: usize,
    start_layer: usize,
    end_layer: usize,
    memory_mb: f64,
    cost: PlanCost,
    request_compute_ms: f64,
    request_network_ms: f64,
}

#[derive(Clone, Debug)]
struct SearchLabel {
    cost: PlanCost,
    segments: Vec<SearchSegment>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PlanReport {
    pub algorithm: PlannerAlgorithm,
    pub score_definition: String,
    pub implementation_note: String,
    pub profile_version: u32,
    pub model_layers: usize,
    pub hidden_size: usize,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub risk_weight: f64,
    pub selected_device_count: usize,
    pub selected_remote_worker_count: usize,
    pub cost: PlanCost,
    pub segments: Vec<PlannedSegment>,
    pub states_considered: usize,
    pub feasible_transitions: usize,
    pub rejected_for_memory: usize,
}

#[derive(Debug)]
pub struct PlannedTopology {
    pub topology: Topology,
    pub report: PlanReport,
}

impl PlannedTopology {
    pub fn summary(&self) -> String {
        let layout = self
            .report
            .segments
            .iter()
            .map(|segment| {
                format!(
                    "{}:{}-{}",
                    segment.device, segment.start_layer, segment.end_layer
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        format!(
            "auto plan {:?} selected {} device(s), score={:.3}, mean_ttft={:.3}ms, mean_tpot={:.3}ms, tail_ttft={:.3}ms, tail_tpot={:.3}ms: {}",
            self.report.algorithm,
            self.report.selected_device_count,
            self.report.cost.score,
            self.report.cost.ttft_ms,
            self.report.cost.tpot_ms,
            self.report.cost.tail_ttft_ms,
            self.report.cost.tail_tpot_ms,
            layout
        )
    }

    pub fn write_report(&self, path: &str) -> Result<()> {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| {
                    anyhow!(
                        "can't create auto-plan output directory {}: {e}",
                        parent.display()
                    )
                })?;
            }
        }
        let bytes = serde_json::to_vec_pretty(&self.report)?;
        fs::write(path, bytes)
            .map_err(|e| anyhow!("can't write auto-plan report {}: {e}", path.display()))?;
        log::info!("wrote automatic plan report to {}", path.display());
        Ok(())
    }
}

pub struct AutoPlanner {
    profile: PlannerProfile,
}

impl AutoPlanner {
    pub fn from_path(path: &str) -> Result<Self> {
        log::info!("loading automatic topology profile from {path}");
        let raw = fs::read_to_string(path)
            .map_err(|e| anyhow!("can't read automatic topology profile {path}: {e}"))?;
        let profile = serde_yaml::from_str(&raw)
            .map_err(|e| anyhow!("can't parse automatic topology profile {path}: {e}"))?;
        Ok(Self { profile })
    }

    #[cfg(test)]
    fn from_profile(profile: PlannerProfile) -> Self {
        Self { profile }
    }

    pub fn plan(&self, config: &Config, layer_prefix: &str) -> Result<PlannedTopology> {
        self.plan_with_algorithm(config, layer_prefix, PlannerAlgorithm::Dial)
    }

    pub fn plan_with_algorithm(
        &self,
        config: &Config,
        layer_prefix: &str,
        algorithm: PlannerAlgorithm,
    ) -> Result<PlannedTopology> {
        let devices = self.validate(config)?;
        let objective = &self.profile.objective;
        let num_layers = config.num_hidden_layers;
        let device_count = devices.len();
        let max_devices = objective
            .max_devices
            .unwrap_or(device_count)
            .clamp(1, device_count);
        let min_devices = objective.min_devices;
        let kv_context = if objective.kv_context_tokens == 0 {
            config.max_seq_len
        } else {
            objective.kv_context_tokens.min(config.max_seq_len)
        };
        let head_dim = config.hidden_size / config.num_attention_heads;
        let kv_mb_per_layer =
            (2usize * config.num_key_value_heads * head_dim * kv_context * objective.dtype_bytes)
                as f64
                / MIB;

        // dp[end_layer][device_mask][last_device] stores the exact best scalar-cost
        // layout for that state. A device appears at most once, so every selected
        // device owns one continuous segment and unselected devices stay unused.
        let states_per_layer = 1usize << device_count;
        let mut dp =
            vec![vec![vec![None::<SearchLabel>; device_count]; states_per_layer]; num_layers + 1];
        let mut states_considered = 0usize;
        let mut feasible_transitions = 0usize;
        let mut rejected_for_memory = 0usize;

        for device_idx in 0..device_count {
            for end in 1..=num_layers {
                let Some(segment) =
                    self.segment_cost(&devices, device_idx, 0..end, config, kv_mb_per_layer)?
                else {
                    rejected_for_memory += 1;
                    continue;
                };
                feasible_transitions += 1;
                let cost = self.initial_cost(&segment, algorithm);
                dp[end][1 << device_idx][device_idx] = Some(SearchLabel {
                    cost,
                    segments: vec![segment],
                });
            }
        }

        for end in 1..num_layers {
            for mask in 1..states_per_layer {
                if mask.count_ones() as usize >= max_devices {
                    continue;
                }
                for last_device in 0..device_count {
                    let Some(label) = dp[end][mask][last_device].clone() else {
                        continue;
                    };
                    states_considered += 1;
                    for next_device in 0..device_count {
                        if mask & (1 << next_device) != 0 {
                            continue;
                        }
                        for next_end in end + 1..=num_layers {
                            let Some(segment) = self.segment_cost(
                                &devices,
                                next_device,
                                end..next_end,
                                config,
                                kv_mb_per_layer,
                            )?
                            else {
                                rejected_for_memory += 1;
                                continue;
                            };
                            feasible_transitions += 1;
                            let candidate_cost = self.extend_cost(&label.cost, &segment, algorithm);
                            let next_mask = mask | (1 << next_device);
                            let slot = &mut dp[next_end][next_mask][next_device];
                            let replace = slot
                                .as_ref()
                                .map(|current| {
                                    candidate_cost.score < current.cost.score
                                        || (candidate_cost.score == current.cost.score
                                            && candidate_cost.ttft_ms < current.cost.ttft_ms)
                                })
                                .unwrap_or(true);
                            if replace {
                                let mut segments = label.segments.clone();
                                segments.push(segment);
                                *slot = Some(SearchLabel {
                                    cost: candidate_cost,
                                    segments,
                                });
                            }
                        }
                    }
                }
            }
        }

        let mut best: Option<SearchLabel> = None;
        for mask in 1..states_per_layer {
            let selected_devices = mask.count_ones() as usize;
            if selected_devices < min_devices || selected_devices > max_devices {
                continue;
            }
            for last_device in 0..device_count {
                let Some(candidate) = dp[num_layers][mask][last_device].clone() else {
                    continue;
                };
                let replace = best
                    .as_ref()
                    .map(|current| {
                        candidate.cost.score < current.cost.score
                            || (candidate.cost.score == current.cost.score
                                && candidate.segments.len() < current.segments.len())
                    })
                    .unwrap_or(true);
                if replace {
                    best = Some(candidate);
                }
            }
        }

        let best = best.ok_or_else(|| {
            anyhow!(
                "automatic planner found no feasible placement for {num_layers} layers; check device memory profiles"
            )
        })?;
        self.build_result(
            devices,
            best,
            config,
            layer_prefix,
            algorithm,
            states_considered,
            feasible_transitions,
            rejected_for_memory,
        )
    }

    fn initial_cost(&self, segment: &SearchSegment, algorithm: PlannerAlgorithm) -> PlanCost {
        let mut cost = segment.cost.clone();
        cost.score = match algorithm {
            PlannerAlgorithm::Dial => segment.cost.score,
            PlannerAlgorithm::EdgeShardLatency => {
                segment.request_compute_ms + segment.request_network_ms
            }
            PlannerAlgorithm::EdgeShardThroughput => {
                segment.request_compute_ms.max(segment.request_network_ms)
            }
        };
        cost
    }

    fn extend_cost(
        &self,
        current: &PlanCost,
        segment: &SearchSegment,
        algorithm: PlannerAlgorithm,
    ) -> PlanCost {
        let mut cost = current.add(&segment.cost);
        cost.score = match algorithm {
            PlannerAlgorithm::Dial => current.score + segment.cost.score,
            PlannerAlgorithm::EdgeShardLatency => {
                current.score + segment.request_compute_ms + segment.request_network_ms
            }
            PlannerAlgorithm::EdgeShardThroughput => current
                .score
                .max(segment.request_compute_ms)
                .max(segment.request_network_ms),
        };
        cost
    }

    fn validate<'a>(&'a self, config: &Config) -> Result<Vec<&'a PlannerDevice>> {
        let objective = &self.profile.objective;
        if objective.prompt_tokens == 0 {
            bail!("objective.prompt_tokens must be greater than zero");
        }
        if objective.output_tokens == 0 {
            bail!("objective.output_tokens must be greater than zero");
        }
        if objective.dtype_bytes == 0 {
            bail!("objective.dtype_bytes must be greater than zero");
        }
        if objective.min_devices == 0 {
            bail!("objective.min_devices must be greater than zero");
        }
        if matches!(objective.max_devices, Some(0)) {
            bail!("objective.max_devices must be greater than zero when provided");
        }
        if config.num_attention_heads == 0 || config.hidden_size % config.num_attention_heads != 0 {
            bail!(
                "model hidden_size {} is not divisible by num_attention_heads {}",
                config.hidden_size,
                config.num_attention_heads
            );
        }
        for (name, value) in [
            ("ttft_weight", objective.ttft_weight),
            ("tpot_weight", objective.tpot_weight),
            ("risk_weight", objective.risk_weight),
            ("energy_weight", objective.energy_weight),
            ("remote_device_penalty", objective.remote_device_penalty),
        ] {
            if !value.is_finite() || value < 0.0 {
                bail!("objective.{name} must be finite and non-negative");
            }
        }
        if objective.risk_weight > 1.0 {
            bail!("objective.risk_weight must be between 0 and 1");
        }
        if objective.ttft_weight == 0.0
            && objective.tpot_weight == 0.0
            && objective.energy_weight == 0.0
        {
            bail!("at least one planning objective weight must be positive");
        }

        let devices = self
            .profile
            .devices
            .iter()
            .filter(|device| device.enabled)
            .collect::<Vec<_>>();
        if devices.is_empty() {
            bail!("automatic planner profile has no enabled devices");
        }
        if devices.len() > 12 {
            bail!("automatic planner supports at most 12 enabled devices");
        }
        let effective_max_devices = objective
            .max_devices
            .unwrap_or(devices.len())
            .min(devices.len());
        if objective.min_devices > effective_max_devices {
            bail!(
                "objective.min_devices ({}) exceeds the effective max_devices ({effective_max_devices})",
                objective.min_devices
            );
        }
        let master_count = devices
            .iter()
            .filter(|device| device.role == PlannerRole::Master)
            .count();
        if master_count != 1 {
            bail!("automatic planner requires exactly one enabled master, found {master_count}");
        }
        let mut names = std::collections::HashSet::new();
        let mut worker_hosts = std::collections::HashSet::new();
        for device in &devices {
            if device.name.trim().is_empty() {
                bail!("planner device name cannot be empty");
            }
            if !names.insert(device.name.as_str()) {
                bail!("duplicate planner device name '{}'", device.name);
            }
            if !device.usable_memory_mb.is_finite() || device.usable_memory_mb <= 0.0 {
                bail!("device {} usable_memory_mb must be positive", device.name);
            }
            if !device.fixed_memory_mb.is_finite()
                || device.fixed_memory_mb < 0.0
                || device.fixed_memory_mb > device.usable_memory_mb
            {
                bail!("device {} has invalid fixed_memory_mb", device.name);
            }
            device.layer_memory_mb.validate(
                &format!("devices.{}.layer_memory_mb", device.name),
                config.num_hidden_layers,
            )?;
            device.prefill_ms.validate(
                &format!("devices.{}.prefill_ms", device.name),
                config.num_hidden_layers,
            )?;
            device.decode_ms.validate(
                &format!("devices.{}.decode_ms", device.name),
                config.num_hidden_layers,
            )?;
            for (tail, mean, tail_name, mean_name) in [
                (
                    device.prefill_p95_ms.as_ref(),
                    &device.prefill_ms,
                    "prefill_p95_ms",
                    "prefill_ms",
                ),
                (
                    device.decode_p95_ms.as_ref(),
                    &device.decode_ms,
                    "decode_p95_ms",
                    "decode_ms",
                ),
            ] {
                if let Some(tail) = tail {
                    let field = format!("devices.{}.{tail_name}", device.name);
                    tail.validate(&field, config.num_hidden_layers)?;
                    tail.validate_not_below(
                        mean,
                        &field,
                        &format!("devices.{}.{mean_name}", device.name),
                        config.num_hidden_layers,
                    )?;
                }
            }
            device.prefill_energy_mj.validate(
                &format!("devices.{}.prefill_energy_mj", device.name),
                config.num_hidden_layers,
            )?;
            device.decode_energy_mj.validate(
                &format!("devices.{}.decode_energy_mj", device.name),
                config.num_hidden_layers,
            )?;
            if device.role == PlannerRole::Worker {
                if device.host.as_deref().unwrap_or("").is_empty() {
                    bail!("worker {} requires a host", device.name);
                }
                if !worker_hosts.insert(device.host.as_deref().unwrap()) {
                    bail!(
                        "worker {} reuses host {}; each planned worker must have a unique endpoint",
                        device.name,
                        device.host.as_deref().unwrap()
                    );
                }
                let bandwidth = device
                    .bandwidth_mbps
                    .ok_or_else(|| anyhow!("worker {} requires bandwidth_mbps", device.name))?;
                if !bandwidth.is_finite() || bandwidth <= 0.0 {
                    bail!("worker {} bandwidth_mbps must be positive", device.name);
                }
                if !device.rtt_ms.is_finite()
                    || device.rtt_ms < 0.0
                    || !device.protocol_ms.is_finite()
                    || device.protocol_ms < 0.0
                {
                    bail!("worker {} has invalid network timings", device.name);
                }
                let rtt_p95 = device.rtt_p95_ms.unwrap_or(device.rtt_ms);
                let protocol_p95 = device.protocol_p95_ms.unwrap_or(device.protocol_ms);
                let bandwidth_p05 = device.bandwidth_p05_mbps.unwrap_or(bandwidth);
                if !rtt_p95.is_finite()
                    || rtt_p95 < device.rtt_ms
                    || !protocol_p95.is_finite()
                    || protocol_p95 < device.protocol_ms
                {
                    bail!(
                        "worker {} tail network latencies must be finite and no smaller than their mean values",
                        device.name
                    );
                }
                if !bandwidth_p05.is_finite() || bandwidth_p05 <= 0.0 || bandwidth_p05 > bandwidth {
                    bail!(
                        "worker {} bandwidth_p05_mbps must be positive and no greater than bandwidth_mbps",
                        device.name
                    );
                }
            }
        }
        Ok(devices)
    }

    fn segment_cost(
        &self,
        devices: &[&PlannerDevice],
        device_idx: usize,
        range: Range<usize>,
        config: &Config,
        kv_mb_per_layer: f64,
    ) -> Result<Option<SearchSegment>> {
        let device = devices[device_idx];
        let objective = &self.profile.objective;
        let memory_mb = device.fixed_memory_mb
            + device.layer_memory_mb.sum(range.clone())
            + kv_mb_per_layer * range.len() as f64;
        if memory_mb > device.usable_memory_mb {
            return Ok(None);
        }

        let prefill_compute = device.prefill_ms.sum(range.clone());
        let decode_compute = device.decode_ms.sum(range.clone());
        let tail_prefill_compute = device
            .prefill_p95_ms
            .as_ref()
            .unwrap_or(&device.prefill_ms)
            .sum(range.clone());
        let tail_decode_compute = device
            .decode_p95_ms
            .as_ref()
            .unwrap_or(&device.decode_ms)
            .sum(range.clone());
        let request_compute_ms = prefill_compute + decode_compute * objective.output_tokens as f64;
        let mut ttft_ms = prefill_compute + decode_compute;
        let mut tpot_ms = decode_compute;
        let mut tail_ttft_ms = tail_prefill_compute + tail_decode_compute;
        let mut tail_tpot_ms = tail_decode_compute;
        let mut request_network_ms = 0.0;
        let prefill_energy = device.prefill_energy_mj.sum(range.clone());
        let decode_energy = device.decode_energy_mj.sum(range.clone());
        let request_energy_mj = prefill_energy + decode_energy * objective.output_tokens as f64;

        if device.role == PlannerRole::Worker {
            let bandwidth = device.bandwidth_mbps.unwrap();
            let tail_bandwidth = device.bandwidth_p05_mbps.unwrap_or(bandwidth);
            let tail_rtt = device.rtt_p95_ms.unwrap_or(device.rtt_ms);
            let tail_protocol = device.protocol_p95_ms.unwrap_or(device.protocol_ms);
            // The current DIAL protocol returns each remote segment to the master.
            let prefill_bytes = 2.0
                * objective.prompt_tokens as f64
                * config.hidden_size as f64
                * objective.dtype_bytes as f64;
            let decode_bytes = 2.0 * config.hidden_size as f64 * objective.dtype_bytes as f64;
            let prefill_network =
                device.rtt_ms + device.protocol_ms + prefill_bytes * 8.0 / (bandwidth * 1000.0);
            let decode_network =
                device.rtt_ms + device.protocol_ms + decode_bytes * 8.0 / (bandwidth * 1000.0);
            let tail_prefill_network =
                tail_rtt + tail_protocol + prefill_bytes * 8.0 / (tail_bandwidth * 1000.0);
            let tail_decode_network =
                tail_rtt + tail_protocol + decode_bytes * 8.0 / (tail_bandwidth * 1000.0);
            request_network_ms = prefill_network + decode_network * objective.output_tokens as f64;
            ttft_ms += prefill_network + decode_network;
            tpot_ms += decode_network;
            tail_ttft_ms += tail_prefill_network + tail_decode_network;
            tail_tpot_ms += tail_decode_network;
        }

        let mean_latency_score = objective.ttft_weight * ttft_ms + objective.tpot_weight * tpot_ms;
        let tail_latency_score =
            objective.ttft_weight * tail_ttft_ms + objective.tpot_weight * tail_tpot_ms;
        let score = (1.0 - objective.risk_weight) * mean_latency_score
            + objective.risk_weight * tail_latency_score
            + objective.energy_weight * request_energy_mj
            + if device.role == PlannerRole::Worker {
                objective.remote_device_penalty
            } else {
                0.0
            };
        Ok(Some(SearchSegment {
            device_idx,
            start_layer: range.start,
            end_layer: range.end - 1,
            memory_mb,
            request_compute_ms,
            request_network_ms,
            cost: PlanCost {
                ttft_ms,
                tpot_ms,
                tail_ttft_ms,
                tail_tpot_ms,
                request_energy_mj,
                score,
            },
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_result(
        &self,
        devices: Vec<&PlannerDevice>,
        best: SearchLabel,
        config: &Config,
        layer_prefix: &str,
        algorithm: PlannerAlgorithm,
        states_considered: usize,
        feasible_transitions: usize,
        rejected_for_memory: usize,
    ) -> Result<PlannedTopology> {
        let mut nodes = HashMap::new();
        for device in &devices {
            if device.role == PlannerRole::Worker {
                nodes.insert(
                    device.name.clone(),
                    Node {
                        host: device.host.clone().unwrap(),
                        description: device.description.clone(),
                        layers: Vec::new(),
                    },
                );
            }
        }

        let mut segments = Vec::new();
        let mut selected_remote_worker_count = 0usize;
        for segment in &best.segments {
            let device = devices[segment.device_idx];
            if device.role == PlannerRole::Worker {
                selected_remote_worker_count += 1;
                let node = nodes.get_mut(&device.name).unwrap();
                node.layers.extend(
                    (segment.start_layer..=segment.end_layer)
                        .map(|layer| format!("{layer_prefix}.{layer}")),
                );
            }
            segments.push(PlannedSegment {
                device: device.name.clone(),
                role: match device.role {
                    PlannerRole::Master => "master".to_string(),
                    PlannerRole::Worker => "worker".to_string(),
                },
                start_layer: segment.start_layer,
                end_layer: segment.end_layer,
                num_layers: segment.end_layer - segment.start_layer + 1,
                estimated_memory_mb: segment.memory_mb,
                estimated_ttft_ms: segment.cost.ttft_ms,
                estimated_tpot_ms: segment.cost.tpot_ms,
                estimated_tail_ttft_ms: segment.cost.tail_ttft_ms,
                estimated_tail_tpot_ms: segment.cost.tail_tpot_ms,
            });
        }

        Ok(PlannedTopology {
            topology: Topology::from_nodes(nodes),
            report: PlanReport {
                algorithm,
                score_definition: algorithm.score_definition().to_string(),
                implementation_note: match algorithm {
                    PlannerAlgorithm::Dial => {
                        "DIAL risk-aware phase planner using mean and conservative P95/P05 profiles over the measured master-worker-master path"
                    }
                    PlannerAlgorithm::EdgeShardLatency => {
                        "Independent EdgeShard latency-objective reimplementation with continuous shards, executed over DIAL's master-worker-master path"
                    }
                    PlannerAlgorithm::EdgeShardThroughput => {
                        "Independent EdgeShard bottleneck-objective reimplementation, executed over DIAL's master-worker-master path without EdgeShard No-bubbles scheduling"
                    }
                }
                .to_string(),
                profile_version: self.profile.version,
                model_layers: config.num_hidden_layers,
                hidden_size: config.hidden_size,
                prompt_tokens: self.profile.objective.prompt_tokens,
                output_tokens: self.profile.objective.output_tokens,
                risk_weight: self.profile.objective.risk_weight,
                selected_device_count: best.segments.len(),
                selected_remote_worker_count,
                cost: best.cost,
                segments,
                states_considered,
                feasible_transitions,
                rejected_for_memory,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(layers: usize) -> Config {
        Config {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: layers,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            max_seq_len: 1024,
            attn_f32: false,
        }
    }

    fn device(name: &str, role: PlannerRole, speed: f64, memory: f64) -> PlannerDevice {
        PlannerDevice {
            name: name.to_string(),
            role,
            host: (role == PlannerRole::Worker).then(|| format!("{name}:10128")),
            description: None,
            enabled: true,
            usable_memory_mb: memory,
            fixed_memory_mb: 100.0,
            layer_memory_mb: LayerMetric::Uniform(100.0),
            prefill_ms: LayerMetric::Uniform(speed),
            decode_ms: LayerMetric::Uniform(speed),
            prefill_p95_ms: None,
            decode_p95_ms: None,
            prefill_energy_mj: LayerMetric::Uniform(0.0),
            decode_energy_mj: LayerMetric::Uniform(0.0),
            rtt_ms: if role == PlannerRole::Worker {
                0.2
            } else {
                0.0
            },
            rtt_p95_ms: None,
            bandwidth_mbps: (role == PlannerRole::Worker).then_some(1000.0),
            bandwidth_p05_mbps: None,
            protocol_ms: 0.0,
            protocol_p95_ms: None,
        }
    }

    fn objective() -> PlannerObjective {
        PlannerObjective {
            prompt_tokens: 128,
            output_tokens: 32,
            dtype_bytes: 2,
            kv_context_tokens: 128,
            ttft_weight: 0.6,
            tpot_weight: 0.4,
            risk_weight: 0.0,
            energy_weight: 0.0,
            remote_device_penalty: 0.0,
            min_devices: 1,
            max_devices: None,
        }
    }

    #[test]
    fn selects_only_fast_orin_when_it_can_hold_every_layer() {
        let planner = AutoPlanner::from_profile(PlannerProfile {
            version: 1,
            objective: objective(),
            devices: vec![
                device("master", PlannerRole::Master, 20.0, 5000.0),
                device("rk1", PlannerRole::Worker, 20.0, 5000.0),
                device("rk2", PlannerRole::Worker, 20.0, 5000.0),
                device("orin", PlannerRole::Worker, 1.0, 5000.0),
            ],
        });
        let result = planner.plan(&config(8), "model.layers").unwrap();
        assert_eq!(result.report.selected_device_count, 1);
        assert_eq!(result.report.segments[0].device, "orin");
        assert_eq!(result.report.segments[0].start_layer, 0);
        assert_eq!(result.report.segments[0].end_layer, 7);
        assert!(result.topology["rk1"].layers.is_empty());
        assert!(result.topology["rk2"].layers.is_empty());
        assert_eq!(result.topology["orin"].layers.len(), 8);
    }

    #[test]
    fn uses_optional_rk_workers_when_orin_cannot_hold_every_layer() {
        let planner = AutoPlanner::from_profile(PlannerProfile {
            version: 1,
            objective: objective(),
            devices: vec![
                device("master", PlannerRole::Master, 30.0, 250.0),
                device("rk1", PlannerRole::Worker, 10.0, 450.0),
                device("rk2", PlannerRole::Worker, 11.0, 450.0),
                device("orin", PlannerRole::Worker, 1.0, 450.0),
            ],
        });
        let result = planner.plan(&config(8), "model.layers").unwrap();
        assert_eq!(result.report.selected_device_count, 3);
        assert_eq!(
            result
                .report
                .segments
                .iter()
                .map(|segment| segment.num_layers)
                .sum::<usize>(),
            8
        );
        assert!(result
            .report
            .segments
            .iter()
            .any(|segment| segment.device == "orin"));
        assert_eq!(
            result
                .report
                .segments
                .iter()
                .filter(|segment| segment.device.starts_with("rk"))
                .count(),
            2
        );
    }

    #[test]
    fn max_devices_is_respected_without_requiring_all_devices() {
        let mut objective = objective();
        objective.max_devices = Some(2);
        let planner = AutoPlanner::from_profile(PlannerProfile {
            version: 1,
            objective,
            devices: vec![
                device("master", PlannerRole::Master, 20.0, 5000.0),
                device("rk1", PlannerRole::Worker, 10.0, 5000.0),
                device("rk2", PlannerRole::Worker, 10.0, 5000.0),
                device("orin", PlannerRole::Worker, 1.0, 5000.0),
            ],
        });
        let result = planner.plan(&config(8), "model.layers").unwrap();
        assert!(result.report.selected_device_count <= 2);
        assert_eq!(result.report.selected_device_count, 1);
    }

    #[test]
    fn min_devices_supports_the_no_subset_ablation() {
        let mut forced_objective = objective();
        forced_objective.min_devices = 4;
        forced_objective.max_devices = Some(4);
        let planner = AutoPlanner::from_profile(PlannerProfile {
            version: 2,
            objective: forced_objective,
            devices: vec![
                device("master", PlannerRole::Master, 1.0, 10_000.0),
                device("rk1", PlannerRole::Worker, 1.0, 10_000.0),
                device("rk2", PlannerRole::Worker, 1.0, 10_000.0),
                device("orin", PlannerRole::Worker, 1.0, 10_000.0),
            ],
        });

        let result = planner.plan(&config(8), "model.layers").unwrap();
        assert_eq!(result.report.selected_device_count, 4);
        assert!(result
            .report
            .segments
            .iter()
            .all(|segment| segment.num_layers > 0));
    }

    #[test]
    fn risk_weight_avoids_a_fast_but_unstable_worker() {
        let master = device("master", PlannerRole::Master, 2.0, 10_000.0);
        let mut unstable = device("unstable", PlannerRole::Worker, 1.0, 10_000.0);
        unstable.rtt_ms = 0.0;
        unstable.bandwidth_mbps = Some(1_000_000.0);
        unstable.prefill_p95_ms = Some(LayerMetric::Uniform(10.0));
        unstable.decode_p95_ms = Some(LayerMetric::Uniform(10.0));

        let devices = vec![master, unstable];
        let mean_result = AutoPlanner::from_profile(PlannerProfile {
            version: 2,
            objective: objective(),
            devices: devices.clone(),
        })
        .plan(&config(8), "model.layers")
        .unwrap();
        assert_eq!(mean_result.report.segments[0].device, "unstable");

        let mut robust_objective = objective();
        robust_objective.risk_weight = 1.0;
        let robust_result = AutoPlanner::from_profile(PlannerProfile {
            version: 2,
            objective: robust_objective,
            devices,
        })
        .plan(&config(8), "model.layers")
        .unwrap();

        assert_eq!(robust_result.report.segments[0].device, "master");
        assert!(
            robust_result.report.cost.tail_ttft_ms < mean_result.report.cost.tail_ttft_ms,
            "the robust plan must improve the conservative tail scenario"
        );
    }

    #[test]
    fn rejects_a_tail_profile_below_its_mean() {
        let mut worker = device("worker", PlannerRole::Worker, 2.0, 10_000.0);
        worker.prefill_p95_ms = Some(LayerMetric::Uniform(1.0));
        let planner = AutoPlanner::from_profile(PlannerProfile {
            version: 2,
            objective: objective(),
            devices: vec![device("master", PlannerRole::Master, 2.0, 10_000.0), worker],
        });

        let error = planner.plan(&config(8), "model.layers").unwrap_err();
        assert!(error.to_string().contains("must be no smaller than"));
    }

    #[test]
    fn edgeshard_latency_prefers_one_fast_segment_for_a_single_request() {
        let profile = PlannerProfile {
            version: 1,
            objective: objective(),
            devices: vec![
                device("master", PlannerRole::Master, 1.0, 10_000.0),
                device("worker", PlannerRole::Worker, 2.0, 10_000.0),
            ],
        };
        let result = AutoPlanner::from_profile(profile)
            .plan_with_algorithm(
                &config(8),
                "model.layers",
                PlannerAlgorithm::EdgeShardLatency,
            )
            .unwrap();

        assert_eq!(result.report.algorithm, PlannerAlgorithm::EdgeShardLatency);
        assert_eq!(result.report.selected_device_count, 1);
        assert_eq!(result.report.segments[0].device, "master");
        assert_eq!(
            result.report.score_definition,
            "estimated_end_to_end_request_latency_ms"
        );
    }

    #[test]
    fn edgeshard_throughput_balances_the_pipeline_bottleneck() {
        let mut worker = device("worker", PlannerRole::Worker, 1.0, 10_000.0);
        worker.rtt_ms = 0.0;
        worker.bandwidth_mbps = Some(1_000_000.0);
        let profile = PlannerProfile {
            version: 1,
            objective: objective(),
            devices: vec![device("master", PlannerRole::Master, 1.0, 10_000.0), worker],
        };
        let result = AutoPlanner::from_profile(profile)
            .plan_with_algorithm(
                &config(8),
                "model.layers",
                PlannerAlgorithm::EdgeShardThroughput,
            )
            .unwrap();

        assert_eq!(
            result.report.algorithm,
            PlannerAlgorithm::EdgeShardThroughput
        );
        assert_eq!(result.report.selected_device_count, 2);
        assert_eq!(
            result
                .report
                .segments
                .iter()
                .map(|segment| segment.num_layers)
                .sum::<usize>(),
            8
        );
        assert!(result.report.cost.score < 8.0 * 33.0);
    }

    #[test]
    fn rejects_a_profile_without_exactly_one_master() {
        let planner = AutoPlanner::from_profile(PlannerProfile {
            version: 1,
            objective: objective(),
            devices: vec![device("orin", PlannerRole::Worker, 1.0, 5000.0)],
        });
        let error = planner.plan(&config(8), "model.layers").unwrap_err();
        assert!(error.to_string().contains("exactly one enabled master"));
    }

    #[test]
    fn version_one_profile_without_tail_fields_remains_compatible() {
        let raw = r#"
version: 1
objective:
  prompt_tokens: 128
  output_tokens: 32
devices:
  - name: master
    role: master
    usable_memory_mb: 10000
    layer_memory_mb: 100
    prefill_ms: 2
    decode_ms: 1
"#;
        let profile: PlannerProfile = serde_yaml::from_str(raw).unwrap();
        assert_eq!(profile.objective.risk_weight, 0.0);
        assert_eq!(profile.objective.min_devices, 1);

        let result = AutoPlanner::from_profile(profile)
            .plan(&config(8), "model.layers")
            .unwrap();
        assert_eq!(result.report.cost.tail_ttft_ms, result.report.cost.ttft_ms);
        assert_eq!(result.report.cost.tail_tpot_ms, result.report.cost.tpot_ms);
    }

    #[test]
    fn four_device_example_profile_parses_and_covers_all_layers() {
        let raw = include_str!("../../../docs/auto_plan_4devices.example.yml");
        let profile: PlannerProfile = serde_yaml::from_str(raw).unwrap();
        let planner = AutoPlanner::from_profile(profile);
        let result = planner
            .plan(&config(36), "model.language_model.layers")
            .unwrap();
        assert_eq!(
            result
                .report
                .segments
                .iter()
                .map(|segment| segment.num_layers)
                .sum::<usize>(),
            36
        );
        assert!(result.report.selected_device_count <= 4);
        assert_eq!(result.topology.len(), 3);
        assert_eq!(result.report.selected_device_count, 1);
        assert_eq!(result.report.segments[0].device, "orin-worker");
        assert!(result.report.risk_weight > 0.0);
        assert!(result.report.cost.tail_ttft_ms >= result.report.cost.ttft_ms);
    }
}
