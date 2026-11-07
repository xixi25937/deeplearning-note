import torch

def quantize_weight(w):

    max_abs = w.abs().max()

    scale = max_abs / 127.0

    if scale == 0:
        return torch.zeros_like(w, dtype=torch.int8), scale

    q = torch.round(w / scale)
    q = q.clamp(-127, 127).to(torch.int8)
    return q, scale

def quantize_linear(x, w_int8, w_scale, bias):
    # 量化输入的x
    x_int8, x_scale = quantize_weight(x)

    #输入的整数和权重做矩阵乘法

    out_int32 = torch.matmul(x_int8.to(torch.int32), w_int8.to(torch.int32).T)
    # 乘法结果是整数，需要乘以权重的scale来恢复浮点数

    out_float = out_int32.float() * w_scale * x_scale
    # 最后加上偏置
    if bias is not None:
        out_float += bias
    return out_float

class QuantizedLinear(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.w_int8, self.w_scale = quantize_weight(linear.weight.detach())

        self.w_int8 = torch.nn.Parameter(self.w_int8, requires_grad=False)
        self.w_scale = torch.nn.Parameter(self.w_scale, requires_grad=False)

        self.bias = None

        if linear.bias is not None:
            self.bias = torch.nn.Parameter(
            linear.bias.detach().clone(),
            requires_grad=False,
            )

    def forward(self, x):
        return quantize_linear(x, self.w_int8, self.w_scale, self.bias)

if __name__ == "__main__":
    import io
    import json
    from pathlib import Path

    from train import build_model, load_cifar, preprocess

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parent
    model, img_size = build_model("vit")

    checkpoint = torch.load(
        root / "runs/vit/best.pt", map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(checkpoint["model"])

    for block in model.blocks:
        block.attn.qkv = QuantizedLinear(block.attn.qkv)
        block.attn.proj = QuantizedLinear(block.attn.proj)
        block.mlp.fc1 = QuantizedLinear(block.mlp.fc1)
        block.mlp.fc2 = QuantizedLinear(block.mlp.fc2)

    model.head = QuantizedLinear(model.head)

    model.eval()
    x_test, y_test = load_cifar(
        False, "cpu", root=str(root.parent / "work/data")
    )
    assert len(x_test) == len(y_test) == 10000
    batch_size = 128
    correct = 0
    print("开始评估：CPU 单线程，CIFAR-10 测试集 10000 张，batch=128", flush=True)

    with torch.inference_mode():
        for start in range(0, len(x_test), batch_size):
            images = x_test[start:start + batch_size]
            labels = y_test[start:start + batch_size]
            images = preprocess(images, img_size, train=False)
            logits = model(images)
            assert logits.shape == (len(labels), 10)
            assert torch.isfinite(logits).all(), "模型输出包含非有限数值"
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()

            seen = min(start + batch_size, len(x_test))
            if (start // batch_size + 1) % 10 == 0 or seen == len(x_test):
                print(f"已评估 {seen}/{len(x_test)}", flush=True)

    accuracy = correct / len(y_test)

    # 与 FP32 基线使用相同的 state_dict 序列化方式统计大小。
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mib = buffer.tell() / 1024**2
    baseline = json.loads(
        (root / "runs/vit/baseline_fp32.json").read_text(encoding="utf-8")
    )

    print(f"量化前准确率：{baseline['test_accuracy']:.2%}")
    print(f"量化后准确率：{accuracy:.2%}（{correct}/{len(y_test)}）")
    print(f"准确率变化：{(accuracy - baseline['test_accuracy']) * 100:+.2f} 个百分点")
    print(f"量化前权重大小：{baseline['state_dict_bytes'] / 1024**2:.3f} MiB")
    print(f"量化后权重大小：{size_mib:.3f} MiB")
