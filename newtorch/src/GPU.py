import torch
from torch import nn

print(torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:1'))
print(torch.cuda.device_count())

def try_gpu(i = 0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device('cpu')]


print(try_gpu(), try_gpu(10), try_all_gpus())

x = torch.tensor([1, 2, 3])
print(x.device)  #默认是cpu

X = torch.ones(2, 3, device = try_gpu())
print(X)

Z = X.cuda(0)
print(X)
print(Z)

#神经网络与GPU
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device = try_gpu())  #把一个对象移动到指定的设备，或者转换到指定的数据类

print(net(X))

print(net[0].weight.data.device)