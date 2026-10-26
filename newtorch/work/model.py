import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """标准 Causal Self-Attention（GPT 用）"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask: 下三角为 1，上三角为 0
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024)
        )
    
    def forward(self, x):
        B, T, C = x.shape
        
        # QKV: [B, T, 3*C] -> [B, T, 3, n_heads, d_k] -> [3, B, n_heads, T, d_k]
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores: [B, n_heads, T, d_k] @ [B, n_heads, d_k, T] -> [B, n_heads, T, T]
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply causal mask
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Output: [B, n_heads, T, d_k] -> [B, T, C]
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    """标准 FFN: Linear -> GELU -> Linear"""
    def __init__(self, d_model, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


class Block(nn.Module):
    """标准 Pre-Norm Transformer Block"""
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """标准 GPT 模型"""
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
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: 输入 embedding 和输出层共享权重
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
        
        # Token + Position embedding
        tok = self.token_emb(x)  # [B, T, d_model]
        pos = self.pos_emb(torch.arange(T, device=x.device))  # [T, d_model]
        x = self.dropout(tok + pos)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.norm(x)
        logits = self.head(x)  # [B, T, vocab_size]
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成"""
        self.eval()
        for _ in range(max_new_tokens):
            # 取最后 max_seq_len 个 token（防止超出位置编码范围）
            idx_cond = idx[:, -self.pos_emb.weight.shape[0]:]
            
            logits = self(idx_cond)  # [B, T, vocab_size]
            logits = logits[:, -1, :] / temperature  # 只取最后一个位置
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


# ==================== 测试 ====================
if __name__ == "__main__":
    # 测试模型
    model = GPT(
        vocab_size=50257,
        d_model=256,
        n_heads=8,
        n_layers=6,
        mlp_ratio=4,
        dropout=0.1,
    )
    
    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randint(0, 50257, (2, 128))
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # 测试生成
    prompt = torch.randint(0, 50257, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
