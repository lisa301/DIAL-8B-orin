use std::collections::HashMap;

use anyhow::Result;
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
/// 定义一个全局、只编译一次的正则表达式
lazy_static! {
    static ref LAYER_RANGE_PARSER: Regex = Regex::new(r"(?m)^(.+[^\d])(\d+)-(\d+)$").unwrap();
}

/// A single node (worker).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    /// Address and port of the worker.
    pub host: String,
    /// Optional descriptioon.
    pub description: Option<String>,
    /// Layers hosted by this worker. Range expressions are supported.
    pub layers: Vec<String>,
}

impl Node {
    /// 判断当前这个Worker节点，是否负责运行某一层模型.
    pub fn is_layer_owner(&self, full_layer_name: &str) -> bool {
        for prefix in &self.layers {
            if full_layer_name.starts_with(&format!("{}.", prefix)) {
                return true;
            }
        }
        false
    }
}

/// 结构体定义，The topology is a worker-name -> worker-info map.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Topology(HashMap<String, Node>);

/// 工作节点使用（给topology这个结构体写方法）
impl Topology {
    pub(crate) fn from_nodes(nodes: HashMap<String, Node>) -> Self {
        Self(nodes)
    }

    /// Load the topology from a yaml file.
    pub fn from_path(path: &str) -> Result<Self> {
        Self::from_path_impl(path, true)
    }

    pub(crate) fn from_path_silent(path: &str) -> Result<Self> {
        Self::from_path_impl(path, false)
    }

    fn from_path_impl(path: &str, log_loading: bool) -> Result<Self> {
        if log_loading {
            log::info!("loading topology from {}", path);
        }
        //// 从topology.yaml文件读取文件内容，解析成topology结构体
        let mut topology: Self = serde_yaml::from_str(&std::fs::read_to_string(path)?)
            .map_err(|e| anyhow!("can't read {path}: {e}"))?;

        // 检查范围表达式
        for (_worker_name, node) in topology.iter_mut() {
            let mut layers = vec![];
            // 创建一个空列表，用来存放展开后的层名称。
            for layer_name in &node.layers {
                // 遍历这个节点负责的所有层。
                // 使用正则表达式检查层名称是否包含范围表达式，如果匹配成功，提取出基名称、起始编号和结束编号。
                if let Some(caps) = LAYER_RANGE_PARSER.captures_iter(layer_name).next() {
                    let base = caps.get(1).unwrap().as_str().to_string();
                    // 前缀
                    let start = caps.get(2).unwrap().as_str().to_string().parse::<usize>()?;
                    // 起始编号
                    let stop = caps.get(3).unwrap().as_str().to_string().parse::<usize>()?;
                    // 结束编号；验证范围表达式是否合法，结束编号必须大于起始编号。
                    if stop <= start {
                        return Err(anyhow!(
                            "invalid range expression {layer_name}, end must be > start"
                        ));
                    }
                    // 遍历起始编号到结束编号，生成完整的层名称，并添加到列表中。
                    for n in start..=stop {
                        layers.push(format!("{}{}", base, n));
                    }
                // 如果层名称不包含范围表达式，直接添加到列表中。
                } else {
                    layers.push(layer_name.to_string());
                }
            }
            // 把原来的范围表达式替换成展开后的真实层列表。
            node.layers = layers;
        }

        Ok(topology)
    }

    /// 返回指定层的服务节点，如果未找到则返回 None。
    pub fn get_node_for_layer(&self, layer_name: &str) -> Option<(&str, &Node)> {
        for (node_name, node) in &self.0 {
            // 遍历拓扑结构中的每个节点，检查它是否负责服务指定的层。(self.0就是topology的hashmap)
            for node_layer_name in &node.layers {
                // 遍历当前节点负责的所有层，检查是否有与指定层名称匹配的层。
                if layer_name == node_layer_name {
                    return Some((node_name, node));
                }
            }
        }
        None
    }
}
/// 实现 Deref 和 DerefMut trait，使得 Topology 可以像 HashMap 一样被访问和修改。
impl std::ops::Deref for Topology {
    type Target = HashMap<String, Node>;
    fn deref(&self) -> &HashMap<String, Node> {
        &self.0
    }
}

impl std::ops::DerefMut for Topology {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
