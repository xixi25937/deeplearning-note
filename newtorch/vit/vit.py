import math
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """切patch,展平，映射成token向量"""
    def __init__(self, img_H = 224, img_W = 224, Patch_H = 16, Patch_W = 16, in_chans = 3, embed_dim = 768):
        super().__init__()
        self.img_H = img_H
        self.img_W = img_W
        self.Patch_H = Patch_H
        self.Patch_W = Patch_W
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        #强制要求整除
        if img_H % Patch_H != 0 or img_W % Patch_W != 0:
            raise ValueError("图片的高和宽必须分别能被 patch 的高和宽整除")

        #卷积来切成patch
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(Patch_H, Patch_W),
            stride=(Patch_H, Patch_W)
        ) #[批次大小, 输出通道数, 输出高度, 输出宽度]

        self.num_patches = (img_H // Patch_H) * (img_W // Patch_W)

    def forward(self, x, print_shapes=False):
        B, C, H, W = x.shape
        assert H == self.img_H and W == self.img_W
        # [B, C, H, W] -> [B, embed_dim, H/Patch_H, W/Patch_W]
        x = self.proj(x)
        if print_shapes:
            print(f"  PatchEmbed 卷积 [B, D, 网格高, 网格宽]: {tuple(x.shape)}")
        # need [B, patch数量, embed_dim]
        # 先降维再换序
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        if print_shapes:
            print(f"  展平 patch 网格 [B, D, patch数]: {tuple(x.shape)}")
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        if print_shapes:
            print(f"  PatchEmbed 输出 [B, patch数, D]: {tuple(x.shape)}")
        return x





class MultiHeadAttention(nn.Module):
    """多头注意力"""
    #输入：[B, num_patches, embed_dim] 即 [B, N, D]
    def __init__(self, embed_dim, num_heads, dropout = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.dropout = nn.Dropout(dropout)
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 缩放因子，防止点积过大导致 softmax 梯度消失
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, print_shapes=False):
        B, N, D = x.shape
        qkv = self.qkv(x)
        if print_shapes:
            print(f"  QKV 线性映射 [B, N, 3D]: {tuple(qkv.shape)}")
        q, k, v = qkv.chunk(3, dim=-1)  # 分成 Q、K、V
        if print_shapes:
            print(f"  分开 Q/K/V，各自 [B, N, D]: {tuple(q.shape)} / {tuple(k.shape)} / {tuple(v.shape)}")
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        if print_shapes:
            print(f"  拆分注意力头 Q/K/V，各自 [B, 头数, N, 每头维度]: {tuple(q.shape)} / {tuple(k.shape)} / {tuple(v.shape)}")

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        if print_shapes:
            print(f"  注意力分数 Q @ K转置 [B, 头数, N, N]: {tuple(attn.shape)}")
        attn = attn.softmax(dim = -1)
        attn = self.dropout(attn)
        if print_shapes:
            print(f"  Softmax + Dropout 后的注意力权重: {tuple(attn.shape)}")
        out = attn @ v  # [B, num_heads, N, head_dim]
        if print_shapes:
            print(f"  注意力权重 @ V [B, 头数, N, 每头维度]: {tuple(out.shape)}")
        out = out.transpose(1, 2).reshape(B, N, D)
        if print_shapes:
            print(f"  合并注意力头 [B, N, D]: {tuple(out.shape)}")
        out = self.proj(out)
        if print_shapes:
            print(f"  注意力输出投影: {tuple(out.shape)}")
        return out


class MLP(nn.Module):
    """全连接层"""
    def __init__(self, embed_dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, print_shapes=False):
        x = self.fc1(x)
        if print_shapes:
            print(f"  MLP fc1 扩展特征: {tuple(x.shape)}")
        x = self.act(x)
        x = self.dropout(x)
        if print_shapes:
            print(f"  MLP GELU + Dropout: {tuple(x.shape)}")
        x = self.fc2(x)
        x = self.dropout(x)
        if print_shapes:
            print(f"  MLP fc2 + Dropout 恢复特征维度: {tuple(x.shape)}")
        return x


class Block(nn.Module):
    """Transformer block: Multi-Head Attention + MLP + LayerNorm + Residual"""
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, hidden_dim, dropout)

    def forward(self, x, print_shapes=False):
        attn_input = self.norm1(x)
        if print_shapes:
            print(f"  LayerNorm 1: {tuple(attn_input.shape)}")
        x = x + self.attn(attn_input, print_shapes=print_shapes)  # 残差连接
        if print_shapes:
            print(f"  注意力残差相加: {tuple(x.shape)}")
        mlp_input = self.norm2(x)
        if print_shapes:
            print(f"  LayerNorm 2: {tuple(mlp_input.shape)}")
        x = x + self.mlp(mlp_input, print_shapes=print_shapes)   # 残差连接
        if print_shapes:
            print(f"  MLP 残差相加: {tuple(x.shape)}")
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer (ViT)"""
    def __init__(self, img_H = 224, img_W = 224, Patch_H = 16, Patch_W = 16,
                 in_chans = 3, num_classes = 1000, embed_dim = 768, depth = 12,
                 num_heads = 12, mlp_ratio = 4.0, dropout = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_H, img_W, Patch_H, Patch_W, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 分类token
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))  # 位置
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, int(embed_dim * mlp_ratio), dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, print_shapes=False):
        """print_shapes=True 时打印各阶段的张量形状，普通训练默认关闭。"""
        B = x.shape[0]
        if print_shapes:
            print(f"输入图片 [B, C, H, W]: {tuple(x.shape)}")
        x = self.patch_embed(x, print_shapes=print_shapes)

        cls = self.cls_token.expand(B, -1, -1) #复制cls_token到每个样本
        if print_shapes:
            print(f"CLS token 展开到每张图片: {tuple(cls.shape)}")
        x = torch.cat([cls, x], dim =1) #拼接
        if print_shapes:
            print(f"拼接 CLS [B, N, D]: {tuple(x.shape)}")

        x = x + self.pos_embed
        x = self.pos_drop(x)
        if print_shapes:
            print(f"位置编码（沿 batch 广播）: {tuple(self.pos_embed.shape)}")
            print(f"加位置编码 + Dropout: {tuple(x.shape)}")

        for i, blk in enumerate(self.blocks, start=1):
            if print_shapes:
                print(f"\nBlock {i}/{len(self.blocks)} 输入: {tuple(x.shape)}")
            x = blk(x, print_shapes=print_shapes)

        x = self.norm(x)
        if print_shapes:
            print(f"\n最终 LayerNorm: {tuple(x.shape)}")
        x = x[:, 0] #取出CLS token
        if print_shapes:
            print(f"取出 CLS [B, D]: {tuple(x.shape)}")
        x = self.head(x)
        if print_shapes:
            print(f"分类头输出 [B, 类别数]: {tuple(x.shape)}")
        return x


if __name__ == "__main__":
    model = VisionTransformer(img_H=32, img_W=32, Patch_H=4, Patch_W=4, embed_dim=192, num_heads=3, num_classes=10
                              , depth=6, mlp_ratio=4.0, dropout=0.1)
    x = torch.randn(2, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 10)
