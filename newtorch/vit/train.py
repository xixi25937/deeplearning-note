"""
CIFAR-10 上训练 / 微调分类模型

用法（在 newtorch/vit 目录下）：
  python train.py --model vit                        第 1 周：小 ViT 从零训练
  python train.py --model resnet                     第 1 周：ResNet-18 从零训练，作对照
  python train.py --model deit --epochs 10 --lr 1e-4 第 2 周：DeiT-Tiny 加载 ImageNet 预训练权重后微调
  python train.py --model deit --keep_rate 0.7 --epochs 10 --lr 1e-4 --out runs/deit_kr0.7
                                                     第 3 周：带剪枝微调（EViT 的做法）

输出在 runs/<名字>/：log.csv（每个 epoch 的 loss 和准确率）、best.pt（测试准确率最高的模型）

整个 CIFAR-10 只有 150MB，直接放进显存，数据增强也在 GPU 上做，不用 DataLoader。
"""
import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets

from vit import VisionTransformer

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # ImageNet 均值方差，DeiT 预训练时用的就是这组
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_cifar(train, device, root="../work/data"):
    ds = datasets.CIFAR10(root=root, train=train, download=False)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous().to(device)  # uint8 [N, 3, 32, 32]
    y = torch.tensor(ds.targets, device=device)
    return x, y


def augment(x):
    # 随机水平翻转 + 四周补 4 像素后随机裁回 32x32，每张图各自随机
    B = len(x)
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip.view(B, 1, 1, 1), x.flip(3), x)
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    r = torch.randint(0, 9, (B, 1), device=x.device) + torch.arange(32, device=x.device)  # [B, 32] 每张图取哪些行
    c = torch.randint(0, 9, (B, 1), device=x.device) + torch.arange(32, device=x.device)
    return x[torch.arange(B, device=x.device).view(B, 1, 1), :, r.view(B, 32, 1), c.view(B, 1, 32)].permute(0, 3, 1, 2)


def preprocess(x, img_size, train):
    x = x.float() / 255
    if train:
        x = augment(x)
    if img_size != 32:
        # DeiT 的位置编码是按 224x224（14x14 个 patch）学的，所以把 32x32 放大到 224
        x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def build_model(name):
    if name == "vit":
        # CIFAR-10 从零训练的小 ViT：32x32 图，4x4 patch -> 64 个 token
        return VisionTransformer(img_H=32, img_W=32, Patch_H=4, Patch_W=4, num_classes=10,
                                 embed_dim=192, depth=6, num_heads=3), 32
    if name == "resnet":
        # CIFAR 版 ResNet-18：第一层改成 3x3 卷积、去掉 maxpool，否则 32x32 的图一上来就被缩到 8x8
        m = torchvision.models.resnet18(num_classes=10)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        return m, 32
    if name == "deit":
        import timm
        m = VisionTransformer(num_classes=10, embed_dim=192, depth=12, num_heads=3)  # DeiT-Tiny 结构
        sd = timm.create_model("deit_tiny_patch16_224", pretrained=True).state_dict()
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}  # 分类头 1000 类换成 10 类，重新学
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert set(missing) == {"head.weight", "head.bias"} and not unexpected, (missing, unexpected)
        return m, 224
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, x, y, img_size, batch_size=500):
    model.eval()
    correct = 0
    for i in range(0, len(x), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(preprocess(x[i:i + batch_size], img_size, train=False))
        correct += (logits.argmax(1) == y[i:i + batch_size]).sum().item()
    model.train()
    return correct / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["vit", "resnet", "deit"], default="vit")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=5, help="预热 epoch 数")
    p.add_argument("--keep_rate", type=float, default=1.0, help="< 1 时在第 4/7/10 层剪枝，只对 deit 有意义")
    p.add_argument("--max_steps", type=int, default=None, help="调试用，训练这么多步就停")
    p.add_argument("--out", default=None, help="默认 runs/<model>")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out = args.out or f"runs/{args.model}"

    torch.manual_seed(args.seed)
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    xtr, ytr = load_cifar(True, device)
    xte, yte = load_cifar(False, device)

    model, img_size = build_model(args.model)
    model = model.to(device)
    if args.keep_rate < 1:
        model.set_keep_rate(args.keep_rate)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, input {img_size}x{img_size}")

    # ponytail: 所有参数都做 weight decay；DeiT 原文跳过 norm/bias/pos_embed，想对比再拆参数组
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(xtr) // args.batch_size
    total = args.epochs * steps_per_epoch
    warm = args.warmup * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    log_file = open(os.path.join(args.out, "log.csv"), "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["epoch", "train_loss", "train_acc", "test_acc", "lr", "sec"])
    best, step = 0.0, 0

    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(len(xtr), device=device)
        loss_sum = correct = seen = 0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            x, y = preprocess(xtr[idx], img_size, train=True), ytr[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
            step += 1
            if args.max_steps and step >= args.max_steps:
                break

        acc = evaluate(model, xte, yte, img_size)
        dt = time.time() - t0
        log.writerow([epoch, f"{loss_sum / seen:.4f}", f"{correct / seen:.4f}", f"{acc:.4f}",
                      f"{sched.get_last_lr()[0]:.2e}", f"{dt:.1f}"])
        log_file.flush()
        print(f"epoch {epoch:3d} | loss {loss_sum / seen:.4f} | train {correct / seen:.2%} | test {acc:.2%} | {dt:.0f}s")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "test_acc": acc},
                       os.path.join(args.out, "best.pt"))
        if args.max_steps and step >= args.max_steps:
            break

    log_file.close()
    print(f"done. best test acc {best:.2%}, saved to {args.out}/best.pt")


if __name__ == "__main__":
    main()
