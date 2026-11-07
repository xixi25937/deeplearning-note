# 手写 Vision Transformer 与 INT8 动态量化

基于 PyTorch 从零实现 ViT，在 CIFAR-10 上完成训练，再手动实现线性层的 INT8 权重量化与输入动态量化。
完整测试集评估中，权重序列化大小由 **10.301 MiB 降至 2.708 MiB**，准确率由 **83.77% 降至 83.69%**。

## 模型配置

| 配置 | 数值 |
|---|---|
| 数据集 | CIFAR-10，10 类 |
| 输入 | RGB，32 × 32 |
| Patch | 4 × 4，共 64 个 patch token，另加 1 个 CLS token |
| 特征维度 | 192 |
| Transformer Block | 6 |
| 每个 Block 的注意力头数 | 3，每头维度 64 |
| MLP | 192 → 768 → 192 |
| Dropout | 0.1 |
| 原始参数量 | 2,693,578 |

训练使用 AdamW，100 个 epoch，batch size 128，学习率 0.001，weight decay 0.05，预热 5 个 epoch 后使用余弦学习率调度，标签平滑系数 0.1。

## 量化实现

`quantize.py` 中包含三个部分：

- `quantize_weight()`：根据最大绝对值计算缩放系数，将浮点张量对称量化至 `[-127, 127]`，保存为 INT8。
- `quantize_linear()`：动态量化输入，用 INT32 矩阵乘法累加，乘回输入与权重的缩放系数，再加上浮点偏置。
- `QuantizedLinear`：提前量化权重，并以冻结的 `nn.Parameter` 保存权重、缩放系数和偏置，支持 `state_dict` 保存与加载。

模型中的 25 个线性层均被替换，包括每个 Block 的 QKV 映射、注意力输出投影、两个 MLP 线性层，以及最终分类头。
PatchEmbed 卷积、LayerNorm、位置编码与注意力分数等仍使用浮点计算。

当前实现使用通用 INT32 矩阵乘法，适合验证量化计算和存储收益。实际推理速度取决于底层整数运算内核；本项目中的代码尚未接入 oneDNN 或 CUDA 的优化 INT8 路径。

## 实验结果

评估使用完整 CIFAR-10 测试集 10,000 张图片，CPU 单线程，batch size 128，`eval()` 与 `inference_mode()`，沿用训练代码的测试预处理，无随机数据增强。

| 指标 | FP32 基线 | 手写量化 |
|---|---:|---:|
| 测试准确率 | 83.77% | 83.69% |
| 正确预测数 | 8377 / 10000 | 8369 / 10000 |
| state_dict 序列化大小 | 10.301 MiB | 2.708 MiB |

权重大小减少约 **73.7%**，缩小约 **3.8 倍**，准确率下降 **0.08 个百分点**。MiB = 1,048,576 字节；大小均通过 `torch.save(model.state_dict(), BytesIO())` 统计。

训练时保存的最佳准确率为 83.79%，使用 CUDA BF16 评估；上表使用 CPU FP32 重新评估后的 83.77% 作为统一对照。
原始 checkpoint 的 SHA256：`1f97df1530fc03eb985247cefe70dd2d12968f858525a467ecbe36a95227ac21`。

记录环境：Intel Core i7-13650HX、Windows、Python 3.10.20、PyTorch 2.12.0.dev20260408+cu128、torchvision 0.27.0.dev20260407+cu128。量化测试在 CPU 上进行，训练需要 CUDA。

## 文件说明

| 文件 | 用途 |
|---|---|
| `vit.py` | 手写 PatchEmbed、多头注意力、MLP、Block 与 ViT |
| `train.py` | CIFAR-10 数据读取、预处理和训练 |
| `baseline.py` | 测量原始权重的准确率、大小和 CPU 前向耗时 |
| `quantize.py` | 手写量化、线性层替换和完整测试集评估 |

## 运行

准备包含 PyTorch 和 torchvision 的 Python 环境，从仓库根目录进入项目：

```bash
cd newtorch/vit
```

数据目录为 `newtorch/work/data/cifar-10-batches-py`。数据读取使用 `download=False`，运行前需确保 CIFAR-10 数据已在该目录中。

1. 在支持 CUDA 的环境中训练：

   ```bash
   python train.py --model vit --epochs 100 --batch_size 128 --out runs/vit
   ```

2. 测量 FP32 基线，生成后续对比所需的 `runs/vit/baseline_fp32.json`：

   ```bash
   python baseline.py --threads 1
   ```

3. 在 CPU 上量化并评估完整测试集：

   ```bash
   python quantize.py
   ```

训练权重与本地评估输出保存在 `runs/`，由 Git 忽略。复现实验需先训练获得 `runs/vit/best.pt`，或在本地放入与上述配置一致的 checkpoint，再依次运行基线和量化评估。

查看张量形状时，可调用 `model(x, print_shapes=True)`；普通调用默认关闭形状打印。
