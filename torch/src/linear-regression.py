import random 
import torch 
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlabel:', labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
#d2l.plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) #打乱顺序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for x , y in data_iter(batch_size, features, labels):
    print(x, '\n', y)
    break

w = torch.normal(0, 0.01, size = (2 , 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

def linreg(x , w , b):
    return torch.matmul(x, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) **2 /2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
lr = 0.000001
num_epochs = 3000000
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y) #x y 的小批量损失
        l.sum().backward() #小批量损失的梯度
        sgd([w, b], lr, batch_size)  #使用参数的梯度更新参数
    
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}, loss {float(train_l.mean()):.4f}')

print(f"w的估计差值: {true_w - w.reshape(true_w.shape)}")
print(f"b的估计差值: {true_b - b}")    