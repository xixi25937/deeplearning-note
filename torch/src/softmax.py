import torch 
from IPython import display
from d2l import torch as d2l
import matplotlib.pyplot as plt

num_inputs = 784
num_outputs = 10

w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) #前两个参数高斯随机
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X - X.max(dim=1, keepdim=True).values)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

#定义网络的前向传播
def net(X):
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)

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
        self.data = [0.0] * n #当n = 3 时， 相当于 self.data = [0.0, 0.0, 0.0]

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x) #推理
        l = loss(y_hat, y) #计算损失
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad() #清理梯度
            l.backward()    #计算梯度
            updater.step()  #更新参数
            metric.add(
                float(l) * len(y), accuracy(y_hat, y),
                y.size().numel()
            )   
        else:
            l.sum().backward()
            updater(x.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
    
# class Animator:
#     def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None
#                  xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:' ), 
#                  nrows=1, ncols=1, figsize=(3.5, 2.5)):
#         if legend is None:
#             legend =  []
#         d2l.use_svg_display()
#         self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
#         if nrows*ncols ==1:
#             self.axes = [self.axes,]
#         self.config_axes = lambda: d2l.set_axes(self.axes[0],xlabel, ylabel, 
#                                                 xlim, ylim, xscale, yscale, legend)
#         self.x, self.y, self.fmts = None, None, fmts
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
       # animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        print(f'epoch {epoch + 1}, loss {train_loss:.4f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
        
def predict_ch3(net, test_iter, n=6):
    """预测标签（定义在第3章）"""
    for x, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        x[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n]
    )

if __name__ == '__main__':
    batch_size = 256
    # 直接运行本文件时才加载数据并执行 softmax 回归示例。
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    print(evaluate_accuracy(net, test_iter))

    lr = 0.1

    def updater(batch_size):
        return d2l.sgd([w, b], lr, batch_size)

    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    predict_ch3(net, test_iter)
    plt.show()
