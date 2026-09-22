# RKLLM Runtime Smoke Test

当前机器是 x86-64，`librkllmrt.so` 只有 aarch64/armhf，所以不能在本机直接运行 `.rkllm`。需要先交叉编译测试程序，再放到 RK3588 板子上跑。

## 1. 构建

```bash
cd /home/seaway/sdb/ljl/Dial_llama
bash tools/build_rkllm_hidden_smoke.sh
```

输出目录：

```text
/home/seaway/sdb/ljl/Dial_llama/build/rkllm_hidden_smoke_aarch64
```

## 2. 拷到 RK3588

把下面两个东西放到板子同一个目录：

```text
/home/seaway/sdb/ljl/Dial_llama/build/rkllm_hidden_smoke_aarch64
/home/seaway/sdb/ljl/model/qwen3vl_text_prefix_l00_l01_w8a8_calib2.rkllm
```

## 3. 在板子运行

```bash
cd rkllm_hidden_smoke_aarch64
chmod +x run_prefix_l00_l01_smoke.sh

./run_prefix_l00_l01_smoke.sh /path/to/qwen3vl_text_prefix_l00_l01_w8a8_calib2.rkllm
```

如果成功，最后会打印：

```text
RKLLM hidden smoke OK
```

这只说明 `.rkllm`、`librkllmrt.so` 和 NPU runtime 能跑通。要接入 `dial-cli` 还需要确认 `GET_LAST_HIDDEN_LAYER` 返回的是 layer 1 后的原始 hidden，而不是带 final norm 的 hidden；否则 0-1 层 RKLLM 不能接 GPU 的第 2 层。
