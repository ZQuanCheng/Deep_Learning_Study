import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline # 展示高清图


# 一个 Inception 块
class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Conv2d(in_channels, 24, kernel_size=1)
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)



# 搭建神经网络
class CNN(nn.Module):
        def __init__(self):
            ''' 搭建神经网络各层 '''
            super(CNN, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                Inception(in_channels=10),
                nn.Conv2d(88, 20, kernel_size=5), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                Inception(in_channels=20),
                nn.Flatten(),
                nn.Linear(1408, 10)
            )

        def forward(self, x):
            y = self.net(x)
            return y






if __name__ == '__main__':
    # 展示高清图
    # 之前已导入库from matplotlib_inline import backend_inline
    backend_inline.set_matplotlib_formats('svg')

    # 制作数据集
    # 数据集转换参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081)
    ])
    # 下载训练集与测试集
    train_Data = datasets.MNIST(
        root='D:/Pycharm/Py_Projects/DNN/dataset/mnist/',  # 下载路径
        train=True,  # 是 train 集
        download=True,  # 如果该路径没有该数据集，就下载
        transform=transform  # 数据集转换参数
    )
    test_Data = datasets.MNIST(
        root='D:/Pycharm/Py_Projects/DNN/dataset/mnist/',  # 下载路径
        train=False,  # 是 test 集
        download=True,  # 如果该路径没有该数据集，就下载
        transform=transform  # 数据集转换参数
    )

    # 批次加载器
    train_loader = DataLoader(dataset=train_Data, shuffle=True, batch_size=128)
    test_loader = DataLoader(dataset=test_Data, shuffle=False, batch_size=128)

    # 查看网络结构
    X = torch.rand(size=(1, 1, 28, 28))
    for layer in CNN().net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)

    # 创建子类的实例，并搬到 GPU 上
    model = CNN().to('cuda:0')

    # 损失函数的选择
    loss_fn = nn.CrossEntropyLoss()  # 自带 softmax 激活函数

    # 优化算法的选择
    learning_rate = 0.1  # 设置学习率
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    # 训练网络
    epochs = 10
    losses = []  # 记录损失函数变化的列表

    for epoch in range(epochs):
        for (x, y) in train_loader:  # 获取小批次的 x 与 y
            x, y = x.to('cuda:0'), y.to('cuda:0')  # 由于数据集内部进不去，只能在循环的过程中取出一部分样本，就立即将之搬到 GPU 上。
            Pred = model(x)  # 一次前向传播（小批量）
            loss = loss_fn(Pred, y)  # 计算损失函数
            losses.append(loss.item())  # 记录损失函数的变化
            optimizer.zero_grad()  # 清理上一轮滞留的梯度
            loss.backward()  # 一次反向传播
            optimizer.step()  # 优化内部参数

    Fig = plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.show()



    # 测试网络
    correct = 0
    total = 0
    with torch.no_grad():  # 该局部关闭梯度计算功能
        for (x, y) in test_loader:  # 获取小批次的 x 与 y
            x, y = x.to('cuda:0'), y.to('cuda:0')
            Pred = model(x)  # 一次前向传播（小批量）
            _, predicted = torch.max(Pred.data, dim=1)
            correct += torch.sum((predicted == y))
            total += y.size(0)
    print(f'测试集精准度: {100 * correct / total} %')






















