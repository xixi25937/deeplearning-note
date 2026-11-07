"""记录当前权重的 CPU FP32 基线：python baseline.py"""
import argparse
import hashlib
import io
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch

from train import build_model, load_cifar, preprocess


def measure(model, x, warmup=10, iterations=50):
    for _ in range(warmup):
        model(x)
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        model(x)
        durations.append(time.perf_counter() - start)
    return {
        "batch_size": len(x),
        "warmup": warmup,
        "iterations": iterations,
        "median_batch_ms": statistics.median(durations) * 1000,
        "images_per_second": len(x) * iterations / sum(durations),
    }


@torch.inference_mode()
def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=root / "runs/vit/best.pt")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads 必须大于 0")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    checkpoint_path = args.ckpt.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["args"]["model"] != "vit":
        raise ValueError("这份基线脚本用于当前 CIFAR-10 小 ViT")
    model, img_size = build_model("vit")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.float().eval()

    x, y = load_cifar(False, "cpu", root=str(root.parent / "work/data"))
    assert len(x) == len(y) == 10000, "必须评估完整的 CIFAR-10 测试集"
    x = preprocess(x, img_size, train=False)
    batch_size = 128
    correct = 0
    print("正在评估 CIFAR-10 测试集全部 10000 张图片（CPU FP32）...", flush=True)
    for start in range(0, len(x), batch_size):
        logits = model(x[start:start + batch_size])
        assert logits.shape == (len(y[start:start + batch_size]), 10)
        assert torch.isfinite(logits).all(), "输出包含非有限数值"
        correct += (logits.argmax(1) == y[start:start + batch_size]).sum().item()
        if (start // batch_size + 1) % 20 == 0:
            print(f"  已评估 {min(start + batch_size, len(x))}/10000", flush=True)

    print("正在计时：batch=1 和 batch=128，各预热 10 次、测量 50 次...", flush=True)
    timings = [measure(model, x[:size]) for size in (1, batch_size)]
    state_buffer = io.BytesIO()
    torch.save(model.state_dict(), state_buffer)
    parameters = sum(p.numel() for p in model.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert parameter_bytes == parameters * 4
    result = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_epoch_zero_based": checkpoint["epoch"],
        "checkpoint_reported_test_accuracy": checkpoint["test_acc"],
        "device": "cpu",
        "dtype": "float32",
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "model": checkpoint["args"]["model"],
        "input_shape_chw": [3, img_size, img_size],
        "parameters": parameters,
        "parameter_bytes": parameter_bytes,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "state_dict_bytes": state_buffer.tell(),
        "dataset": "CIFAR-10 test",
        "test_samples": len(y),
        "correct": correct,
        "test_accuracy": correct / len(y),
        "evaluation_batch_size": batch_size,
        "preprocessing": "train.preprocess(train=False), RGB / 255, ImageNet mean/std",
        "timing_scope": "model forward only; preprocessed CPU inputs; eval + inference_mode; no autocast",
        "timings": timings,
    }
    output = checkpoint_path.parent / "baseline_fp32.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# ViT 量化前基线（CPU FP32）",
        "",
        f"记录时间：{result['recorded_at']}",
        f"权重：`{checkpoint_path}`",
        f"SHA256：`{result['checkpoint_sha256']}`",
        "",
        "| 指标 | 实测结果 |",
        "|---|---:|",
        f"| CIFAR-10 测试准确率 | {result['test_accuracy']:.2%}（{correct}/10000） |",
        f"| 参数量 | {parameters:,} |",
        f"| 参数占用 | {parameter_bytes / 1024**2:.3f} MiB |",
        f"| 原始 best.pt 文件 | {result['checkpoint_bytes'] / 1024**2:.3f} MiB |",
        f"| 仅 state_dict 序列化大小 | {result['state_dict_bytes'] / 1024**2:.3f} MiB |",
    ]
    for timing in timings:
        size = timing["batch_size"]
        report.extend([
            f"| batch={size} 整批推理耗时（中位数） | {timing['median_batch_ms']:.3f} ms |",
            f"| batch={size} 吞吐量（总图片数 / 总耗时） | {timing['images_per_second']:.2f} 张/秒 |",
        ])
    report.extend([
        "",
        "模型：32×32 输入，4×4 patch，192 维，6 个 Block，3 个注意力头，10 类。",
        f"环境：{result['platform']}；Python {result['python']}；PyTorch {result['pytorch']}；CPU 线程数 {args.threads}，interop 线程数 1。",
        f"处理器：{result['processor']}。",
        "",
        "评估沿用训练代码的测试预处理，无数据增强。模型使用 eval + inference_mode，关闭形状打印和自动混合精度。",
        "计时只包括模型前向计算，不包括数据读取、预处理、权重加载和文件保存；各 batch 预热 10 次，测量 50 次。MiB = 1,048,576 字节。",
        "后续量化应在相同设备、线程数、batch、预处理和计时条件下对比；文件大小应采用相同 state_dict 保存方式。速度会随电脑负载波动。",
        f"训练时 best.pt 记录的准确率为 {checkpoint['test_acc']:.2%}（CUDA BF16 评估），与本次 CPU FP32 的实测值分开记录。",
        "",
        "复现（在 newtorch/vit 目录下，使用已安装 PyTorch 的 Python 环境）：`python baseline.py --threads 1`。",
        "",
    ])
    output.with_suffix(".md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\n原始指标已保存：{output}")


if __name__ == "__main__":
    main()
