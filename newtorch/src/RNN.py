import torch
from d2l import torch as d2l

#说明,$\mathbf{X}_t\mathbf{W}_{xh} + \mathbf{H}_{t-1}\mathbf{W}_{hh}$ 
#等价于把 $\mathbf{X}_t$ 和 $\mathbf{H}_{t-1}$ 沿列方向拼接
#再乘上 $\mathbf{W}_{xh}$ 和 $\mathbf{W}_{hh}$ 沿行方向拼接后的矩阵。
X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))

torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))