"""
代码 ↔ 论文 Decoder 结构 对照

论文 Decoder 结构图:
┌─────────────────────────────────────────┐
│  全连接层 (LM Head)                      │ ← self.head
├─────────────────────────────────────────┤
│  Add & Norm                             │ ← self.norm
├─────────────────────────────────────────┤
│  逐位前馈网络 (Feed Forward)             │ ← MLP 类
│  ├─ Linear(d_model → 4*d_model)         │
│  ├─ GELU                                │
│  └─ Linear(4*d_model → d_model)         │
├─────────────────────────────────────────┤
│  Add & Norm                             │ ← Block 类中的 norm2 + 残差
├─────────────────────────────────────────┤
│  多头注意力 (Cross-Attention)            │ ← 论文有，GPT 已删除
│  (Query 来自 Decoder, Key/Value 来自    │    因为没有 Encoder 了
│   Encoder 输出)                         │
├─────────────────────────────────────────┤
│  Add & Norm                             │ ← Block 类中的 norm1 + 残差
├─────────────────────────────────────────┤
│  掩蔽多头注意力 (Masked Self-Attention)  │ ← CausalSelfAttention 类
│  ├─ Linear → Q, K, V                    │
│  ├─ 多头拆分 (n_heads)                   │
│  ├─ Q @ K.T / sqrt(d_k)                 │
│  ├─ 掩码 (下三角，挡住未来)               │
│  ├─ Softmax                             │
│  └─ @ V → 合并多头 → Linear              │
├─────────────────────────────────────────┤
│  嵌入层 (Embedding) + 位置编码           │ ← token_emb + pos_emb
│  (Token Embedding + Position Encoding)  │
└─────────────────────────────────────────┘
           ↑
         目标 (Target / Input tokens)

GPT 的简化:
- 去掉了 Cross-Attention 层（因为没有 Encoder）
- 只保留 Masked Self-Attention + Feed Forward
- 位置编码用可学习 Embedding（不是正弦函数）
- Add & Norm 改为 Pre-Norm（先 Norm 再进子层，论文是 Post-Norm）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    对应论文: 掩蔽多头注意力 (Masked Multi-Head Attention)
    
    论文结构:
        ┌─────────────────────────┐
        │  掩蔽多头注意力           │
        │  ├─ Linear → Q, K, V    │
        │  ├─ 拆成 n_heads        │
        │  ├─ Q @ K.T / sqrt(d_k) │
        │  ├─ 掩码 (下三角)        │  ← 关键！挡住未来信息
        │  ├─ Softmax             │
        │  └─ @ V → 合并 → Linear  │
        └─────────────────────────┘
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear: d_model → 3*d_model (Q, K, V 一起算)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask: 下三角矩阵，1 表示允许看，0 表示挡住
        # 这就是论文中的 "掩蔽" (Mask)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024)
        )
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Step 1: Linear → Q, K, V
        # [B, T, d_model] → [B, T, 3*d_model]
        qkv = self.qkv(x)
        
        # Step 2: 拆成多头
        # [B, T, 3*d_model] → [B, T, 3, n_heads, d_k] → [3, B, n_heads, T, d_k]
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Step 3: Scaled Dot-Product Attention
        # Q @ K.T: [B, n_heads, T, d_k] @ [B, n_heads, d_k, T] → [B, n_heads, T, T]
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Step 4: 应用掩码（关键！）
        # 下三角为 1（保留），上三角为 0（设为 -inf，softmax 后变 0）
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Step 5: Softmax 得到注意力权重
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Step 6: @ V 加权求和，合并多头
        # [B, n_heads, T, T] @ [B, n_heads, T, d_k] → [B, n_heads, T, d_k]
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        
        # Step 7: 输出投影
        return self.proj(out)


class MLP(nn.Module):
    """
    对应论文: 逐位前馈网络 (Position-wise Feed-Forward Network)
    
    论文结构:
        ┌─────────────────────────┐
        │  逐位前馈网络             │
        │  ├─ Linear(d_model→4*d) │
        │  ├─ ReLU (代码用 GELU)   │
        │  └─ Linear(4*d→d_model) │
        └─────────────────────────┘
    """
    def __init__(self, d_model, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


class Block(nn.Module):
    """
    对应论文: Decoder Layer 的一个 Block
    
    论文结构 (原始是 Post-Norm):
        ┌─────────────────────────┐
        │  x = x + Attn(x)        │  ← 先算注意力，再 Norm，再加残差
        │  x = LayerNorm(x)       │
        │  x = x + MLP(x)         │
        │  x = LayerNorm(x)       │
        └─────────────────────────┘
    
    代码实现 (Pre-Norm，GPT 标准):
        ┌─────────────────────────┐
        │  x = x + Attn(LN(x))    │  ← 先 Norm，再算注意力，再加残差
        │  x = x + MLP(LN(x))     │
        └─────────────────────────┘
    """
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)
    
    def forward(self, x):
        # Pre-Norm: 先 LayerNorm，再进子层，最后加残差
        x = x + self.attn(self.norm1(x))   # 对应论文: Add & Norm + Masked Self-Attention
        x = x + self.mlp(self.norm2(x))    # 对应论文: Add & Norm + Feed Forward
        return x


class GPT(nn.Module):
    """
    对应论文: 完整 Decoder Stack（但已简化）
    
    论文结构:
        ┌─────────────────────────┐
        │  全连接层 (输出概率)      │  ← self.head
        ├─────────────────────────┤
        │  Add & Norm             │  ← self.norm
        ├─────────────────────────┤
        │  [Block × N]            │  ← self.blocks
        │  ├─ Masked Self-Attn    │
        │  ├─ Add & Norm          │
        │  ├─ Cross-Attention     │  ← GPT 已删除！
        │  ├─ Add & Norm          │
        │  ├─ Feed Forward        │
        │  └─ Add & Norm          │
        ├─────────────────────────┤
        │  Embedding + 位置编码    │  ← token_emb + pos_emb
        └─────────────────────────┘
               ↑
             目标序列
    
    GPT 简化:
        - 删除 Cross-Attention（没有 Encoder）
        - 位置编码用可学习 Embedding（不是正弦）
        - Post-Norm 改为 Pre-Norm
        - Weight Tying（head 和 token_emb 共享权重）
    """
    def __init__(
        self,
        vocab_size=50257,
        d_model=256,
        n_heads=8,
        n_layers=6,
        mlp_ratio=4,
        dropout=0.1,
        max_seq_len=1024,
    ):
        super().__init__()
        
        # 嵌入层: Token ID → 向量
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # 位置编码: 位置索引 → 向量（可学习，非正弦）
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # N 个 Block 堆叠
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        
        # 最终 LayerNorm
        self.norm = nn.LayerNorm(d_model)
        
        # 全连接层: d_model → vocab_size（输出概率）
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying: 输入嵌入和输出层共享权重
        self.head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x):
        B, T = x.shape
        
        # Step 1: 嵌入层 + 位置编码
        # [B, T] → [B, T, d_model]
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        x = self.dropout(tok + pos)
        
        # Step 2: N 个 Block
        for block in self.blocks:
            x = block(x)
        
        # Step 3: 最终 LayerNorm
        x = self.norm(x)
        
        # Step 4: 全连接层 → 输出概率
        # [B, T, d_model] → [B, T, vocab_size]
        logits = self.head(x)
        
        return logits


# 打印对照表
if __name__ == "__main__":
    print(__doc__)
    print("\n模型结构对照完成！")
    print("关键差异: GPT 删除了 Cross-Attention，因为只有 Decoder，没有 Encoder 输出可以 attend。")
