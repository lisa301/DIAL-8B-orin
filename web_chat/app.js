"use strict";

const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

const elements = {
  connection: document.querySelector("#connectionStatus"),
  connectionText: document.querySelector("#connectionText"),
  connectionLatency: document.querySelector("#connectionLatency"),
  imageInput: document.querySelector("#imageInput"),
  dropZone: document.querySelector("#dropZone"),
  imagePreview: document.querySelector("#imagePreview"),
  previewImage: document.querySelector("#previewImage"),
  imageName: document.querySelector("#imageName"),
  imageMeta: document.querySelector("#imageMeta"),
  clearImageButton: document.querySelector("#clearImageButton"),
  promptInput: document.querySelector("#promptInput"),
  characterCount: document.querySelector("#characterCount"),
  askButton: document.querySelector("#askButton"),
  stopButton: document.querySelector("#stopButton"),
  answerState: document.querySelector("#answerState"),
  emptyText: document.querySelector("#emptyText"),
  answerContent: document.querySelector("#answerContent"),
  copyButton: document.querySelector("#copyButton"),
  metrics: document.querySelector("#metrics"),
  metricTtft: document.querySelector("#metricTtft"),
  metricTotal: document.querySelector("#metricTotal"),
  metricTps: document.querySelector("#metricTps"),
  metricRemote: document.querySelector("#metricRemote"),
  toast: document.querySelector("#toast")
};

let selectedFile = null;
let selectedDataUrl = "";
let previewObjectUrl = "";
let activeController = null;
let answerText = "";
let toastTimer = null;

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 2800);
}

function formatBytes(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function updateSubmitState() {
  const promptReady = elements.promptInput.value.trim().length > 0;
  elements.askButton.disabled = !selectedFile || !promptReady || Boolean(activeController);
  elements.characterCount.textContent = String(elements.promptInput.value.length);
}

function resetMetrics() {
  elements.metrics.hidden = true;
  elements.metricTtft.textContent = "--";
  elements.metricTotal.textContent = "--";
  elements.metricTps.textContent = "--";
  elements.metricRemote.textContent = "--";
}

function resetAnswer() {
  answerText = "";
  elements.answerContent.textContent = "";
  elements.answerContent.hidden = true;
  elements.answerContent.classList.remove("streaming");
  elements.answerState.hidden = false;
  elements.answerState.className = "answer-state empty";
  elements.emptyText.textContent = "模型回答将在这里实时显示";
  elements.copyButton.disabled = true;
  resetMetrics();
}

function clearImage() {
  selectedFile = null;
  selectedDataUrl = "";
  elements.imageInput.value = "";
  if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
  previewObjectUrl = "";
  elements.previewImage.removeAttribute("src");
  elements.imagePreview.hidden = true;
  elements.dropZone.hidden = false;
  elements.clearImageButton.disabled = true;
  updateSubmitState();
}

function loadImage(file) {
  const validTypes = ["image/png", "image/jpeg", "image/webp", "image/bmp"];
  if (!file || !validTypes.includes(file.type)) {
    showToast("请选择 PNG、JPG、WEBP 或 BMP 图片");
    return;
  }
  if (file.size > MAX_IMAGE_BYTES) {
    showToast("图片不能超过 20 MB");
    return;
  }

  selectedFile = file;
  if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
  previewObjectUrl = URL.createObjectURL(file);
  elements.previewImage.src = previewObjectUrl;
  elements.imageName.textContent = file.name;
  elements.imageMeta.textContent = formatBytes(file.size);
  elements.dropZone.hidden = true;
  elements.imagePreview.hidden = false;
  elements.clearImageButton.disabled = false;

  const reader = new FileReader();
  reader.addEventListener("load", () => {
    selectedDataUrl = String(reader.result || "");
    updateSubmitState();
  });
  reader.addEventListener("error", () => showToast("读取图片失败"));
  reader.readAsDataURL(file);
  updateSubmitState();
}

function setLoading(loading) {
  if (loading) {
    elements.answerState.hidden = false;
    elements.answerState.className = "answer-state loading";
    elements.emptyText.textContent = "正在分析图片";
    elements.answerContent.hidden = true;
    elements.askButton.hidden = true;
    elements.stopButton.hidden = false;
  } else {
    elements.askButton.hidden = false;
    elements.stopButton.hidden = true;
  }
  updateSubmitState();
}

function showStreamContent() {
  elements.answerState.hidden = true;
  elements.answerContent.hidden = false;
  elements.answerContent.classList.add("streaming");
  elements.answerContent.textContent = answerText;
  elements.copyButton.disabled = answerText.length === 0;
}

function applyMetrics(payload) {
  if (payload.total_s == null) return;
  elements.metrics.hidden = false;
  elements.metricTtft.textContent = payload.ttft_s == null ? "--" : `${payload.ttft_s.toFixed(2)} s`;
  elements.metricTotal.textContent = `${payload.total_s.toFixed(2)} s`;
  elements.metricTps.textContent = payload.tokens_per_second == null
    ? "--" : `${payload.tokens_per_second.toFixed(2)} token/s`;
  elements.metricRemote.textContent = payload.remote_requests == null
    ? "--" : String(payload.remote_requests);
}

function consumeEventPayload(rawPayload) {
  if (!rawPayload || rawPayload === "[DONE]") return;
  const payload = JSON.parse(rawPayload);
  if (payload.error) throw new Error(payload.error);

  const chunk = payload.choices?.[0]?.delta?.content;
  if (chunk) {
    answerText += chunk;
    showStreamContent();
  }
  applyMetrics(payload);
}

async function readEventStream(response) {
  if (!response.body) throw new Error("浏览器不支持流式响应");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const event of events) {
      for (const line of event.split("\n")) {
        if (line.startsWith("data:")) consumeEventPayload(line.slice(5).trim());
      }
    }
    if (done) break;
  }
}

async function askQuestion() {
  const prompt = elements.promptInput.value.trim();
  if (!selectedFile || !selectedDataUrl || !prompt) return;

  activeController = new AbortController();
  resetMetrics();
  answerText = "";
  setLoading(true);

  const base64 = selectedDataUrl.slice(selectedDataUrl.indexOf(",") + 1);
  const requestBody = {
    stream: true,
    messages: [{
      role: "user",
      content: [
        {type: "image_base64", media_type: selectedFile.type, data: base64},
        {type: "text", text: prompt}
      ]
    }]
  };

  try {
    const response = await fetch("/api/v1/chat/completions", {
      method: "POST",
      headers: {"Content-Type": "application/json", "Accept": "text/event-stream"},
      body: JSON.stringify(requestBody),
      signal: activeController.signal
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `请求失败（HTTP ${response.status}）`);
    }
    await readEventStream(response);
    if (!answerText) throw new Error("模型没有返回内容");
    elements.answerContent.classList.remove("streaming");
  } catch (error) {
    if (error.name === "AbortError") {
      if (!answerText) {
        elements.answerState.hidden = false;
        elements.answerState.className = "answer-state empty";
        elements.emptyText.textContent = "已停止接收回答";
      }
      elements.answerContent.classList.remove("streaming");
    } else {
      elements.answerState.hidden = false;
      elements.answerState.className = "answer-state empty";
      elements.emptyText.textContent = `调用失败：${error.message}`;
      elements.answerContent.classList.remove("streaming");
      showToast(`调用失败：${error.message}`);
    }
  } finally {
    activeController = null;
    setLoading(false);
    elements.copyButton.disabled = answerText.length === 0;
  }
}

async function checkConnection() {
  const started = performance.now();
  try {
    const response = await fetch("/api/v1/topology", {cache: "no-store"});
    if (!response.ok) throw new Error(String(response.status));
    const data = await response.json();
    const elapsed = Math.round(performance.now() - started);
    const onlineWorkers = (data.workers || []).filter(worker => worker.online).length;
    elements.connection.className = "connection online";
    elements.connectionText.textContent = `Master 在线 · ${onlineWorkers}/${(data.workers || []).length} Worker`;
    elements.connectionLatency.textContent = `${elapsed} ms`;
  } catch (_) {
    elements.connection.className = "connection error";
    elements.connectionText.textContent = "Master 不可用";
    elements.connectionLatency.textContent = "";
  }
}

elements.dropZone.addEventListener("click", () => elements.imageInput.click());
elements.imageInput.addEventListener("change", event => loadImage(event.target.files?.[0]));
elements.clearImageButton.addEventListener("click", clearImage);
elements.promptInput.addEventListener("input", () => {
  if (elements.promptInput.value.length > 1000) {
    elements.promptInput.value = elements.promptInput.value.slice(0, 1000);
  }
  updateSubmitState();
});
elements.promptInput.addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !elements.askButton.disabled) {
    askQuestion();
  }
});
elements.askButton.addEventListener("click", askQuestion);
elements.stopButton.addEventListener("click", () => activeController?.abort());
elements.copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(answerText);
    showToast("回答已复制");
  } catch (_) {
    showToast("浏览器未授予剪贴板权限");
  }
});

document.querySelectorAll("[data-prompt]").forEach(button => {
  button.addEventListener("click", () => {
    elements.promptInput.value = button.dataset.prompt || "";
    elements.promptInput.focus();
    updateSubmitState();
  });
});

["dragenter", "dragover"].forEach(type => {
  elements.dropZone.addEventListener(type, event => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach(type => {
  elements.dropZone.addEventListener(type, event => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
  });
});
elements.dropZone.addEventListener("drop", event => loadImage(event.dataTransfer.files?.[0]));

resetAnswer();
updateSubmitState();
checkConnection();
window.setInterval(checkConnection, 5000);
