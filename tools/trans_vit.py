import torch
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from transformers import AutoModel, AutoProcessor

# ------------------------------------------------
# 1. 模型路径
# ------------------------------------------------

model_path = "/home/seaway/sdb/ljl/model/Qwen3-VL-8B-Instruct"

print("Loading model...")

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="cpu"
).eval()

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

visual = model.visual

print("Vision model loaded")

# ------------------------------------------------
# 2. 构造测试图像
# ------------------------------------------------

img = Image.new("RGB", (448, 448), (128, 128, 128))

inputs = processor(
    text="<image>",
    images=img,
    return_tensors="pt"
)

hidden_states = inputs["pixel_values"]
grid_thw = inputs["image_grid_thw"]

print("hidden_states:", hidden_states.shape)
print("grid_thw:", grid_thw)

# ------------------------------------------------
# 3. PyTorch forward
# ------------------------------------------------

with torch.no_grad():

    torch_out = visual(
        hidden_states=hidden_states,
        grid_thw=grid_thw
    )

    torch_out = torch_out.last_hidden_state

print("pytorch output:", torch_out.shape)

# ------------------------------------------------
# 4. Wrapper (用于 ONNX 导出)
# ------------------------------------------------

class VisionWrapper(torch.nn.Module):

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, hidden_states, grid_thw):

        out = self.visual(
            hidden_states=hidden_states,
            grid_thw=grid_thw
        )

        return out.last_hidden_state


wrapper = VisionWrapper(visual).eval()

# ------------------------------------------------
# 5. 导出 ONNX
# ------------------------------------------------

onnx_path = "qwen3_vit.onnx"

print("Exporting ONNX...")

torch.onnx.export(
    wrapper,
    (hidden_states, grid_thw),
    onnx_path,
    input_names=["hidden_states", "grid_thw"],
    output_names=["vision_embeds"],
    opset_version=17,
    do_constant_folding=True
)

print("ONNX exported:", onnx_path)

# ------------------------------------------------
# 6. ONNXRuntime 推理验证
# ------------------------------------------------

print("Running ONNXRuntime check...")

session = ort.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]
)

ort_inputs = {
    "hidden_states": hidden_states.numpy(),
    "grid_thw": grid_thw.numpy()
}

ort_out = session.run(None, ort_inputs)

print("ONNX output:", ort_out[0].shape)

# ------------------------------------------------
# 7. 数值误差检查
# ------------------------------------------------

diff = np.max(np.abs(torch_out.numpy() - ort_out[0]))

print("max diff:", diff)

print("Done!")