import torch 
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

num_inputs = 784
num_outputs = 10

w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) #前两个参数高斯随机
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

#定义网络的前向传播
def net(X):
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)

y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y]) #输出y_hat中第0行第0列和第1行第2列的元素

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y)), y]) # 返回每一行对应真实标签概率的负对数

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) #返回每一行最大元素的索引
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) #返回预测正确的数量

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() #将模型设置为评估模式
    metric = d2l.Accumulator(2) # 正确预测数、预测总数
    for x, y in data_iter:
        metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]
    
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
print(evaluate_accuracy(net, test_iter)) #输出在测试集上的精度