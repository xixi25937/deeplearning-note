"""
模型结构对照：代码 vs Transformer 论文
"""

# ============================================================
# 论文中的 Transformer 架构（原始 Encoder-Decoder）
# ============================================================
#
# Encoder:
#   Input Embedding + Positional Encoding
#   for _ in range(N):
#       x = x + MultiHeadAttention(LayerNorm(x))      # Self-Attention
#       x = x + FeedForward(LayerNorm(x))
#
# Decoder:
#   Output Embedding + Positional Encoding
#   for _ in range(N):
#       x = x + MaskedMultiHeadAttention(LayerNorm(x)) # Masked Self-Attention
#       x = x + MultiHeadAttention(LayerNorm(x), encoder_output)  # Cross-Attention
#       x = x + FeedForward(LayerNorm(x))
#   Linear + Softmax
#
# ============================================================
# GPT 的简化（只有 Decoder，去掉 Cross-Attention）
# ============================================================
#
#   Token Embedding + Position Embedding
#   for _ in range(N):
#       x = x + CausalSelfAttention(LayerNorm(x))   # 只有 Self-Attention，带 causal mask
#       x = x + MLP(LayerNorm(x))
#   LayerNorm
#   Linear to vocab_size
#
# ============================================================
# 代码对应关系
# ============================================================

class CodeToPaper:
    """
    代码中的每个类 ↔ 论文中的组件
    """
    
    # 1. CausalSelfAttention 类
    #    对应论文: Multi-Head Attention + Causal Mask
    #    关键操作:
    #      - Linear(d_model, 3*d_model)  → Q, K, V 投影
    #      - reshape 到 n_heads           → 多头拆分
    #      - Q @ K.T / sqrt(d_k)          → Scaled Dot-Product Attention
    #      - masked_fill (下三角)          → Causal Mask（防止看未来）
    #      - softmax                      → Attention 权重
    #      - @ V                          → 加权求和
    #      - Linear(d_model, d_model)     → 输出投影
    
    # 2. MLP 类
    #    对应论文: Position-wise Feed-Forward Network
    #    结构: Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model)
    #    论文公式: FFN(x) = max(0, xW1 + b1)W2 + b2
    #    代码用 GELU 替代 ReLU（GPT 标准）
    
    # 3. Block 类
    #    对应论文: 一个 Decoder Layer
    #    结构:
    #      x = x + CausalSelfAttention(LayerNorm(x))   ← Self-Attention 子层
    #      x = x + MLP(LayerNorm(x))                   ← FFN 子层
    #    这叫 Pre-Norm（先 Norm 再进子层），原始论文是 Post-Norm
    
    # 4. GPT 类
    #    对应论文: 完整 Decoder Stack
    #    组件:
    #      - token_emb: Token Embedding（查表）
    #      - pos_emb: Position Embedding（可学习，非正弦）
    #      - blocks: N 个 Block 堆叠
    #      - norm: 最后的 LayerNorm
    #      - head: Linear(d_model → vocab_size)
    
    pass


# ============================================================
# 数据流 Shape 变化
# ============================================================

"""
输入: token IDs [B, T]  (B=batch_size, T=seq_len)

Step 1: Embedding
    token_emb: [B, T] → [B, T, d_model]
    pos_emb:   [T]    → [T, d_model] → 广播到 [B, T, d_model]
    x = token_emb + pos_emb  → [B, T, d_model]

Step 2: N 个 Block（每个 Block 内部）
    输入: [B, T, d_model]
    
    Self-Attention 子层:
        LayerNorm: [B, T, d_model] → [B, T, d_model]
        QKV Linear: [B, T, d_model] → [B, T, 3*d_model]
        reshape: [B, T, 3*d_model] → [B, T, 3, n_heads, d_k] → [3, B, n_heads, T, d_k]
        Q @ K.T: [B, n_heads, T, d_k] @ [B, n_heads, d_k, T] → [B, n_heads, T, T]
        causal mask: [B, n_heads, T, T] → 下三角保留，上三角 -inf
        softmax: [B, n_heads, T, T]
        @ V: [B, n_heads, T, T] @ [B, n_heads, T, d_k] → [B, n_heads, T, d_k]
        reshape: [B, n_heads, T, d_k] → [B, T, d_model]
        输出投影: [B, T, d_model] → [B, T, d_model]
        残差: x + attn_output → [B, T, d_model]
    
    MLP 子层:
        LayerNorm: [B, T, d_model]
        Linear(d_model→4*d_model): [B, T, d_model] → [B, T, 4*d_model]
        GELU: [B, T, 4*d_model]
        Linear(4*d_model→d_model): [B, T, 4*d_model] → [B, T, d_model]
        残差: x + mlp_output → [B, T, d_model]
    
    输出: [B, T, d_model] → 传给下一个 Block

Step 3: 最终输出
    LayerNorm: [B, T, d_model]
    LM Head (Linear d_model→vocab_size): [B, T, d_model] → [B, T, vocab_size]
    
输出: logits [B, T, vocab_size]
"""

# ============================================================
# 当前配置的具体数字
# ============================================================

config = {
    "vocab_size": 50257,   # GPT-2 tokenizer 的词表大小
    "d_model": 256,        # 隐藏维度（论文用 512，GPT-3 用 12288）
    "n_heads": 8,          # 注意力头数（论文用 8，GPT-3 用 96）
    "n_layers": 6,         # Block 数量（论文用 6，GPT-3 用 96）
    "d_k": 256 // 8,       # 每个头的维度 = 32
    "mlp_hidden": 256 * 4, # MLP 隐藏层 = 1024（4倍比例）
    "max_seq_len": 1024,   # 最大序列长度
}

# 参数量计算:
# - token_emb: 50257 * 256 = 12,865,792
# - pos_emb: 1024 * 256 = 262,144
# - 每个 Block:
#     attention QKV: 256 * 768 = 196,608
#     attention proj: 256 * 256 = 65,536
#     mlp fc1: 256 * 1024 = 262,144
#     mlp fc2: 1024 * 256 = 262,144
#     norm1 + norm2: 2 * (256*2) = 1,024
#     小计: ~787,456
# - 6 个 Block: 6 * 787,456 = 4,724,736
# - final norm: 512
# - head (weight tied): 0（共享 token_emb）
# 总计: 12,865,792 + 262,144 + 4,724,736 + 512 ≈ 17,853,184 ≈ 17.87M

print("对照完成。看代码时对照上面的注释，每个类、每行操作都对应论文中的组件。")
