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
