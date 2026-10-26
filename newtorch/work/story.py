import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class StoryDataset(Dataset):
    """
    TinyStories 数据集
    把多个故事拼接成连续 token 流，然后切成固定长度的块
    """
    def __init__(self, parquet_path, tokenizer_name="gpt2", seq_len=512, max_stories=None):
        self.seq_len = seq_len
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        # 加载数据
        df = pd.read_parquet(parquet_path)
        stories = df['text'].tolist()
        
        if max_stories:
            stories = stories[:max_stories]
        
        # 编码所有故事，用 <|endoftext|> 分隔
        print(f"Tokenizing {len(stories)} stories...")
        all_tokens = []
        for story in stories:
            tokens = self.tokenizer.encode(story)
            all_tokens.extend(tokens)
            all_tokens.append(self.tokenizer.eot_token)  # 故事结束符
        
        self.data = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Total tokens: {len(self.data):,}")
        
        # 计算样本数：总 token 数 // seq_len
        self.n_samples = len(self.data) // seq_len
        print(f"Samples (seq_len={seq_len}): {self.n_samples}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 取连续的一段 token
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 是为了构造目标（shift）
        
        chunk = self.data[start:end]
        
        # 输入是前 seq_len 个，目标是后 seq_len 个（预测下一个 token）
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y


def get_dataloader(parquet_path, batch_size=32, seq_len=512, max_stories=None, shuffle=True):
    """
    获取 DataLoader
    """
    dataset = StoryDataset(parquet_path, seq_len=seq_len, max_stories=max_stories)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )


# ==================== 测试 ====================
if __name__ == "__main__":
    # 测试数据加载
    path = "./data/tinystories/validation.parquet"
    
    print("Testing StoryDataset...")
    dataset = StoryDataset(path, seq_len=128, max_stories=100)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # 取一个样本看看
    x, y = dataset[0]
    print(f"\nSample shape: x={x.shape}, y={y.shape}")
    print(f"x[:20]: {x[:20].tolist()}")
    print(f"y[:20]: {y[:20].tolist()}")
    
    # 解码看看内容
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"\nDecoded x[:50]: {tokenizer.decode(x[:50].tolist())}")
    print(f"Decoded y[:50]: {tokenizer.decode(y[:50].tolist())}")
    
    # 测试 DataLoader
    print("\nTesting DataLoader...")
    loader = get_dataloader(path, batch_size=4, seq_len=128, max_stories=100)
    
    for batch_x, batch_y in loader:
        print(f"Batch shape: x={batch_x.shape}, y={batch_y.shape}")
        break
