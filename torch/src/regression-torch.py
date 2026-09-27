import numpy as np
import torch
from torch.utils import data

from torch import nn
from d2l import torch as d2l

true_w = torch.tensor([2.0, -3.0, 4.0])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    """构建一个PyTorch数据迭代器"""
    # 将输入 features 和答案 labels 按样本一一配对，封装成 dataset，方便按索引访问
    dataset = data.TensorDataset(*data_arrays)
    # 按 batch_size 分成一批一批读取；训练时通常会打乱每轮数据顺序
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

batch_size = 10
# 为这份模拟数据构建批量读取器；本例中它准备作为训练数据使用
data_iter = load_array((features, labels), batch_size)

# 读取第一批数据；这一步只是取数据，还没有开始训练
next(iter(data_iter))

net = nn.Sequential(nn.Linear(3, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for x, y in data_iter:
        l = loss(net(x), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print(f'epoch {epoch+1}, loss {float(l):f}')