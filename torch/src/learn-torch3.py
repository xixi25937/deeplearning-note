import torch
import torch.nn.functional as F
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
layer(torch.FloatTensor([[1, 2, 3, 4, 5]]))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

Y = net(torch.rand(4, 8))
Y.mean() #所有元素的均值

#带参数的图层

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))
    
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.biaas.data
        return F.relu(linear)
        
dense = MyLinear(5, 3)
print(dense.weight)
