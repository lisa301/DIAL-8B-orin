//! This is the spm command line utility.

use dial_core::{
    spm::{Context, Master, Mode, Worker},
    Args,
};

use anyhow::{anyhow, Result};
use base64::Engine;
use clap::Parser;
use std::{fs, io::Write, path::Path};
use tokio::io::{AsyncBufReadExt, BufReader};

fn fmt_metric(v: Option<f64>, unit: &str) -> String {
    v.map(|v| format!("{v:.3}{unit}"))
        .unwrap_or_else(|| "null".to_string())
}

fn print_metrics(
    ttft: Option<f64>,
    total: Option<f64>,
    tps: Option<f64>,
    decode_tps: Option<f64>,
    dist_overhead: Option<f64>,
    remote_compute: Option<f64>,
    remote_requests: Option<usize>,
) {
    eprintln!(
        "[metrics] ttft_s={} total_s={} tps={} decode_tps={} dist_overhead_s={} remote_compute_s={} remote_requests={}",
        fmt_metric(ttft,"s"),
        fmt_metric(total,"s"),
        fmt_metric(tps,"toks/s"),
        fmt_metric(decode_tps,"toks/s"),
        fmt_metric(dist_overhead,"s"),
        fmt_metric(remote_compute,"s"),
        remote_requests
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string()),
    );
}

fn display_model_name(model_dir: &str) -> String {
    let name = Path::new(model_dir)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model_dir)
        .to_ascii_lowercase();

    if name.contains("qwen3-vl-8b") {
        "qwen3-vl-8B".to_string()
    } else if name.contains("qwen3-vl-2b") {
        "qwen3-vl-2B".to_string()
    } else if name.contains("llama-3-8b") || name.contains("llama3-8b") {
        "llama3-8B".to_string()
    } else {
        name
    }
}

fn print_model_header(model_dir: &str) {
    println!("[choose_model]:{}", display_model_name(model_dir));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectedModel {
    Qwen3Vl,
    Llama3,
}

/// （新增）根据图片扩展名推断 mime 类型。
///
/// 为什么要加：Qwen3-VL 的 `image_base64` part 允许携带 `media_type`；
/// 这能帮助服务端/模型更准确地理解图片格式（png/jpg/webp...）。
fn guess_media_type(path: &str) -> Option<String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mime = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        _ => return None,
    };
    Some(mime.to_string())
}

/// （新增）把本地图片读出来并转成 `image_base64` 这个多模态输入 part。
///
/// 为什么要加：用户希望“命令行直接问图”，而不是手写一大坨 JSON + base64；
/// 这个函数把“读文件 + base64 编码 + 组装 ContentPart”封装起来。
fn image_part_from_path(path: &str) -> Result<dial_core::models::chat::ContentPart> {
    let bytes = fs::read(path).map_err(|e| anyhow!("can't read image {}: {e}", path))?;
    let data = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(dial_core::models::chat::ContentPart::ImageBase64 {
        media_type: guess_media_type(path),
        data,
    })
}

fn guess_video_media_type(path: &str) -> Option<String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mime = match ext.as_str() {
        "mp4" | "m4v" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "avi" => "video/x-msvideo",
        _ => return None,
    };
    Some(mime.to_string())
}

fn video_part_from_path(
    path: &str,
    max_bytes: usize,
) -> Result<dial_core::models::chat::ContentPart> {
    let metadata = fs::metadata(path).map_err(|e| anyhow!("can't stat video {}: {e}", path))?;
    if metadata.len() > max_bytes as u64 {
        return Err(anyhow!(
            "video {} is {} bytes, exceeding --video-max-bytes {}",
            path,
            metadata.len(),
            max_bytes
        ));
    }
    let bytes = fs::read(path).map_err(|e| anyhow!("can't read video {}: {e}", path))?;
    let data = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(dial_core::models::chat::ContentPart::VideoBase64 {
        media_type: guess_video_media_type(path),
        data,
    })
}

/// （新增）生成一个 user message：既支持纯文本，也支持“文本 + 图片”的多模态 message。
///
/// 实现了什么：当传入 `image_path` 时，构造 OpenAI 风格的 `content: [ {text}, {image_base64} ]`；
/// 不传图片时则退化为普通文本消息，兼容纯文本模型/请求。
fn user_message_with_media(
    text: String,
    image_path: Option<&str>,
    video_path: Option<&str>,
    video_max_bytes: usize,
) -> Result<dial_core::models::chat::Message> {
    if image_path.is_some() && video_path.is_some() {
        return Err(anyhow!("--image and --video cannot be used together"));
    }
    if let Some(path) = image_path {
        // Qwen3-VL 的官方/常见用法是“先给图，再提问”，因此把 image part 放在 text 前面。
        let parts = vec![
            image_part_from_path(path)?,
            dial_core::models::chat::ContentPart::Text { text },
        ];
        Ok(dial_core::models::chat::Message {
            role: dial_core::models::chat::MessageRole::User,
            content: dial_core::models::chat::MessageContent::Parts(parts),
        })
    } else if let Some(path) = video_path {
        let parts = vec![
            video_part_from_path(path, video_max_bytes)?,
            dial_core::models::chat::ContentPart::Text { text },
        ];
        Ok(dial_core::models::chat::Message {
            role: dial_core::models::chat::MessageRole::User,
            content: dial_core::models::chat::MessageContent::Parts(parts),
        })
    } else {
        Ok(dial_core::models::chat::Message::user(text))
    }
}

/// （新增）把 `--api-client` 的输入统一规范成 base URL。
///
/// 为什么要加：用户可能传 `127.0.0.1:8082` / `http://127.0.0.1:8082/` 等各种形式；
/// 统一后拼接 `/api/v1/chat/completions` 更稳妥。
fn normalize_api_base(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    if s.is_empty() {
        return s;
    }
    if !s.starts_with("http://") && !s.starts_with("https://") {
        s = format!("http://{s}");
    }
    s.trim_end_matches('/').to_string() //把末尾的/去掉
}

/// Read an OpenAI-style SSE stream and print delta tokens to stdout as they arrive.
/// Returns the full assistant content and optional metrics when provided by the server.
async fn consume_sse_stream(
    mut resp: reqwest::Response,
    print_deltas: bool,
) -> Result<(
    String,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<usize>,
)> {
    let mut buf = String::new();
    let mut out = String::new();
    let mut ttft_s: Option<f64> = None;
    let mut total_s: Option<f64> = None;
    let mut tokens_per_second: Option<f64> = None;
    let mut decode_tokens_per_second: Option<f64> = None;
    let mut distributed_overhead_s: Option<f64> = None;
    let mut remote_compute_s: Option<f64> = None;
    let mut remote_requests: Option<usize> = None;

    loop {
        let chunk = resp.chunk().await?;
        let Some(chunk) = chunk else { break };
        let s = std::str::from_utf8(&chunk)
            .map_err(|e| anyhow!("invalid utf-8 in SSE response: {e}"))?;
        buf.push_str(s);

        while let Some(idx) = buf.find("\n\n") {
            let event = buf[..idx].to_string();
            buf.drain(..idx + 2);

            for line in event.lines() {
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();

                if data == "[DONE]" {
                    if print_deltas {
                        println!();
                    }
                    return Ok((
                        out,
                        ttft_s,
                        total_s,
                        tokens_per_second,
                        decode_tokens_per_second,
                        distributed_overhead_s,
                        remote_compute_s,
                        remote_requests,
                    ));
                }

                // Normal OpenAI streaming chunk.
                if data.starts_with('{') {
                    let v: serde_json::Value =
                        serde_json::from_str(data).map_err(|e| anyhow!("bad SSE json: {e}"))?;

                    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                        return Err(anyhow!("{err}"));
                    }

                    if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                        if !c.is_empty() {
                            out.push_str(c);
                            if print_deltas {
                                print!("{c}");
                                std::io::stdout().flush().ok();
                            }
                        }
                    }

                    // Our server optionally attaches metrics on the final chunk.
                    if ttft_s.is_none() {
                        ttft_s = v.get("ttft_s").and_then(|x| x.as_f64());
                    }
                    if total_s.is_none() {
                        total_s = v.get("total_s").and_then(|x| x.as_f64());
                    }
                    if tokens_per_second.is_none() {
                        tokens_per_second = v.get("tokens_per_second").and_then(|x| x.as_f64());
                    }
                    if decode_tokens_per_second.is_none() {
                        decode_tokens_per_second =
                            v.get("decode_tokens_per_second").and_then(|x| x.as_f64());
                    }
                    if distributed_overhead_s.is_none() {
                        distributed_overhead_s =
                            v.get("distributed_overhead_s").and_then(|x| x.as_f64());
                    }
                    if remote_compute_s.is_none() {
                        remote_compute_s = v.get("remote_compute_s").and_then(|x| x.as_f64());
                    }
                    if remote_requests.is_none() {
                        remote_requests = v
                            .get("remote_requests")
                            .and_then(|x| x.as_u64())
                            .map(|v| v as usize);
                    }
                }
            }
        }
    }

    // Stream ended without a [DONE] marker (still return what we got).
    if print_deltas {
        println!();
    }
    Ok((
        out,
        ttft_s,
        total_s,
        tokens_per_second,
        decode_tokens_per_second,
        distributed_overhead_s,
        remote_compute_s,
        remote_requests,
    ))
}

/// （新增）`spm-cli` 的“简易客户端模式”：
/// - 连接到已运行的 `--api` 服务端
/// - 自动拼 JSON、发送请求
/// - 只在终端打印 assistant 的纯文本内容（不输出整段 JSON）
///
/// 为什么要加：让第二个端口（API 服务）用起来更像“聊天”，而不是每次写 curl + JSON。
async fn run_api_client(args: Args) -> Result<()> {
    let base = args
        .api_client
        .as_deref()
        .map(normalize_api_base)
        .unwrap_or_default();
    if base.is_empty() {
        return Err(anyhow!("--api-client is empty"));
    }

    let url = format!("{base}/api/v1/chat/completions");
    let http = reqwest::Client::new();

    let mut messages: Vec<dial_core::models::chat::Message> = vec![];
    if !args.system_prompt.is_empty() {
        // 把 system_prompt 放到会话历史里，服务端按 OpenAI 兼容格式处理。
        messages.push(dial_core::models::chat::Message::system(
            args.system_prompt.clone(),
        ));
    }

    // Single-shot mode: --ask, else fallback to --prompt, else read stdin once.
    if !args.repl {
        let ask = args
            .ask
            .clone()
            .or_else(|| (!args.prompt.is_empty()).then(|| args.prompt.clone()));

        let ask = match ask {
            Some(s) => s,
            None => {
                let mut buf = String::new();
                BufReader::new(tokio::io::stdin())
                    .read_line(&mut buf)
                    .await?;
                buf.trim().to_string()
            }
        };

        if ask.is_empty() {
            return Err(anyhow!(
                "no prompt provided; use --ask, --prompt, or pipe text into stdin"
            ));
        }

        // 单次提问：支持纯文本或“文本+图片”（由 --image 控制）。
        messages.push(user_message_with_media(
            ask,
            args.image.as_deref(),
            args.video.as_deref(),
            args.video_max_bytes,
        )?);

        if args.stream {
            let resp = http
                .post(url)
                .json(&serde_json::json!({ "messages": messages, "stream": true }))
                .send()
                .await?
                .error_for_status()?;
            print_model_header(&args.model);
            let (
                _content,
                ttft,
                total,
                tps,
                decode_tps,
                dist_overhead,
                remote_compute,
                remote_requests,
            ) = consume_sse_stream(resp, true).await?;
            if args.metrics {
                print_metrics(
                    ttft,
                    total,
                    tps,
                    decode_tps,
                    dist_overhead,
                    remote_compute,
                    remote_requests,
                );
            }
        }
        return Ok(());
    }

    // REPL mode.
    let mut stdin = BufReader::new(tokio::io::stdin());
    let mut line = String::new();

    if !args.prompt.is_empty() {
        // 可选：启动 REPL 前先发一条初始 prompt（也支持 --image）。
        messages.push(user_message_with_media(
            args.prompt.clone(),
            args.image.as_deref(),
            args.video.as_deref(),
            args.video_max_bytes,
        )?);
        if args.stream {
            let resp = http
                .post(&url)
                .json(&serde_json::json!({ "messages": messages, "stream": true }))
                .send()
                .await?
                .error_for_status()?;
            print_model_header(&args.model);
            let (
                content,
                ttft,
                total,
                tps,
                decode_tps,
                dist_overhead,
                remote_compute,
                remote_requests,
            ) = consume_sse_stream(resp, true).await?;
            messages.push(dial_core::models::chat::Message::assistant(content));
            if args.metrics {
                print_metrics(
                    ttft,
                    total,
                    tps,
                    decode_tps,
                    dist_overhead,
                    remote_compute,
                    remote_requests,
                );
            }
        }
    }

    loop {
        line.clear();
        // 交互提示符写到 stderr，方便把 stdout 重定向保存模型回复内容。
        eprint!("你> ");
        let n = stdin.read_line(&mut line).await?;
        if n == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/q" | "/quit" | "/exit") {
            break;
        }

        // （新增）REPL 图片快捷指令：`/img path/to.png 你要问的问题`
        // 为什么要加：在 REPL 里临时换图片更方便，不需要退出重启进程/改 --image。
        let (user_text, img_path, video_path) = if let Some(rest) = input.strip_prefix("/img ") {
            let rest = rest.trim();
            let mut it = rest.splitn(2, char::is_whitespace);
            let path = it.next().unwrap_or("").trim();
            let question = it.next().unwrap_or("").trim();
            if path.is_empty() || question.is_empty() {
                println!("用法: /img <图片路径> <问题>");
                continue;
            }
            (question.to_string(), Some(path.to_string()), None)
        } else if let Some(rest) = input.strip_prefix("/video ") {
            let rest = rest.trim();
            let mut it = rest.splitn(2, char::is_whitespace);
            let path = it.next().unwrap_or("").trim();
            let question = it.next().unwrap_or("").trim();
            if path.is_empty() || question.is_empty() {
                println!("用法: /video <视频路径> <问题>");
                continue;
            }
            (question.to_string(), None, Some(path.to_string()))
        } else {
            (input.to_string(), None, None)
        };

        // 行内 /img 或 /video 优先；普通文本才复用启动参数中的媒体。
        let (selected_image, selected_video) = if img_path.is_some() || video_path.is_some() {
            (img_path.as_deref(), video_path.as_deref())
        } else {
            (args.image.as_deref(), args.video.as_deref())
        };
        messages.push(user_message_with_media(
            user_text,
            selected_image,
            selected_video,
            args.video_max_bytes,
        )?);
        if args.stream {
            let resp = http
                .post(&url)
                .json(&serde_json::json!({ "messages": messages, "stream": true }))
                .send()
                .await?
                .error_for_status()?;
            print_model_header(&args.model);
            let (
                content,
                ttft,
                total,
                tps,
                decode_tps,
                dist_overhead,
                remote_compute,
                remote_requests,
            ) = consume_sse_stream(resp, true).await?;
            messages.push(dial_core::models::chat::Message::assistant(content));
            if args.metrics {
                print_metrics(
                    ttft,
                    total,
                    tps,
                    decode_tps,
                    dist_overhead,
                    remote_compute,
                    remote_requests,
                );
            }
        }
    }

    Ok(())
}

fn detect_model_type(model_dir: &str) -> Result<SelectedModel> {
    let config_path = Path::new(model_dir).join("config.json");
    let raw =
        fs::read(&config_path).map_err(|e| anyhow!("can't read {}: {e}", config_path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|e| anyhow!("can't parse {}: {e}", config_path.display()))?;
    Ok(match v.get("model_type") {
        Some(serde_json::Value::String(s)) if s == "qwen3_vl" => SelectedModel::Qwen3Vl,
        _ => SelectedModel::Llama3,
    })
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn env_value_is_off(value: &str) -> bool {
    let value = value.trim().to_ascii_lowercase();
    matches!(
        value.as_str(),
        "0" | "false" | "no" | "off" | "disable" | "disabled"
    )
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn parse_cpu_affinity_spec(spec: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in spec
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if let Some((start, end)) = part.split_once('-') {
            let (Ok(start), Ok(end)) = (start.trim().parse::<usize>(), end.trim().parse::<usize>())
            else {
                continue;
            };
            if start <= end {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    cpus
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn default_cpu_affinity_spec() -> Option<String> {
    let cpus = std::thread::available_parallelism().ok()?.get();
    if cpus >= 8 {
        Some("4-7".to_string())
    } else {
        None
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn set_thread_affinity(tid: libc::pid_t, cpus: &[usize]) -> std::io::Result<()> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for cpu in cpus {
            libc::CPU_SET(*cpu, &mut set);
        }
        let rc = libc::sched_setaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if rc == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn apply_cpu_affinity() {
    if std::env::var("DIAL_DISABLE_CPU_AFFINITY")
        .map(|value| !env_value_is_off(&value))
        .unwrap_or(false)
    {
        log::info!("cpu affinity disabled by DIAL_DISABLE_CPU_AFFINITY");
        return;
    }

    let spec = match std::env::var("DIAL_CPU_AFFINITY") {
        Ok(value) if env_value_is_off(&value) => {
            log::info!("cpu affinity disabled by DIAL_CPU_AFFINITY={value}");
            return;
        }
        Ok(value) => value,
        Err(_) => match default_cpu_affinity_spec() {
            Some(value) => value,
            None => return,
        },
    };

    let cpus = parse_cpu_affinity_spec(&spec);
    if cpus.is_empty() {
        log::warn!("invalid DIAL_CPU_AFFINITY={spec}; cpu affinity not changed");
        return;
    }

    let mut applied = 0usize;
    let mut last_error = None;
    if let Ok(entries) = fs::read_dir("/proc/self/task") {
        for entry in entries.flatten() {
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            let Ok(tid) = name.parse::<libc::pid_t>() else {
                continue;
            };
            match set_thread_affinity(tid, &cpus) {
                Ok(()) => applied += 1,
                Err(e) => last_error = Some(e),
            }
        }
    } else if let Err(e) = set_thread_affinity(0, &cpus) {
        last_error = Some(e);
    } else {
        applied = 1;
    }

    if applied > 0 {
        log::info!("cpu affinity set to {spec} for {applied} thread(s)");
    } else if let Some(error) = last_error {
        log::warn!("failed to set cpu affinity {spec}: {error}");
    } else {
        log::warn!("failed to set cpu affinity {spec}: no target threads found");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // parse command line
    let args = Args::parse();

    // （新增）客户端模式：不加载本地模型、不启动 master/worker，只负责调用 API 并打印纯文本回复。
    if args.api_client.is_some() {
        return run_api_client(args).await;
    }

    #[cfg(target_arch = "aarch64")]
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        // RK3588 has 4 big Cortex-A76 cores plus 4 small A55 cores. Candle/GEMM defaults to
        // all CPUs, which often slows single-token decode by scheduling matmul work on A55 cores.
        std::env::set_var("RAYON_NUM_THREADS", "4");
    }

    let selected_model = detect_model_type(&args.model)?;

    // setup logging
    if std::env::var_os("RUST_LOG").is_none() {
        // set `RUST_LOG=debug` to see debug logs
        std::env::set_var("RUST_LOG", "info,tokenizers=error,actix_server=warn");
    }

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_module_path(false)
        .format_target(false)
        .init();

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    apply_cpu_affinity();

    // setup context
    let ctx = Context::from_args(args)?;
    log::info!("selected model: {:?}", selected_model);

    // run either in master or worker mode depending on command line
    let ret = match (ctx.args.mode.clone(), selected_model) {
        (Mode::Master, SelectedModel::Qwen3Vl) => {
            Master::<dial_core::models::qwen3_vl::Qwen3Vl>::new(ctx)
                .await?
                .run()
                .await
        }
        (Mode::Worker, SelectedModel::Qwen3Vl) => {
            Worker::<dial_core::models::qwen3_vl::Qwen3Vl>::new(ctx)
                .await?
                .run()
                .await
        }
        (Mode::Master, SelectedModel::Llama3) => {
            Master::<dial_core::models::llama3::LLama>::new(ctx)
                .await?
                .run()
                .await
        }
        (Mode::Worker, SelectedModel::Llama3) => {
            Worker::<dial_core::models::llama3::LLama>::new(ctx)
                .await?
                .run()
                .await
        }
    };

    if ret.is_err() {
        // we were possibly streaming text, add a newline before reporting the error
        println!();
        return ret;
    }

    Ok(())
}
