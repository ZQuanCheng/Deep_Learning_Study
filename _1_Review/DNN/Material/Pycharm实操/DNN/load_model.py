import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline # 展示高清图


if __name__ == '__main__':
    # 展示高清图
    # 之前已导入库from matplotlib_inline import backend_inline
    backend_inline.set_matplotlib_formats('svg')

    # 生成数据集
    X1 = torch.rand(10000, 1)  # 输入特征 1
    X2 = torch.rand(10000, 1)  # 输入特征 2
    X3 = torch.rand(10000, 1)  # 输入特征 3
    print(X1.shape, X2.shape, X3.shape)  # torch.Size([10000, 1]) torch.Size([10000, 1]) torch.Size([10000, 1])

    Y1 = ((X1 + X2 + X3) < 1).float()  # 输出特征 1
    Y2 = ((1 < (X1 + X2 + X3)) & ((X1 + X2 + X3) < 2)).float()  # 输出特征 2
    Y3 = ((X1 + X2 + X3) > 2).float()  # 输出特征 3
    print(Y1.shape, Y2.shape, Y3.shape)  # torch.Size([10000, 1]) torch.Size([10000, 1]) torch.Size([10000, 1])

    Data = torch.cat([X1, X2, X3, Y1, Y2, Y3], axis=1)  # 整合数据集; cat就是numpy中的concatenate
    print(Data.type())  # torch.FloatTensor

    Data = Data.to('cuda:0')  # 把数据集搬到 GPU 上
    print(Data.type())  # torch.cuda.FloatTensor
    print(Data.shape)  # torch.Size([10000, 6])

    # 划分训练集与测试集
    train_size = int(len(Data) * 0.7)  # 训练集的样本数量
    test_size = len(Data) - train_size  # 测试集的样本数量
    Data = Data[torch.randperm(Data.size(0)), :]  # 打乱样本的顺序
    train_Data = Data[:train_size, :]  # 训练集样本
    test_Data = Data[train_size:, :]  # 测试集样本
    print(train_Data.shape, test_Data.shape)  # torch.Size([7000, 6]) torch.Size([3000, 6])

    # print(Data)


    # 把模型赋给新网络
    new_model = torch.load('model.pth')

    # 测试网络
    # 给测试集划分输入与输出
    X = test_Data[:, :3]  # 前 3 列为输入特征
    Y = test_Data[:, -3:]  # 后 3 列为输出特征
    with torch.no_grad():  # 该局部关闭梯度计算功能
        Pred = new_model(X)  # 用新模型进行一次前向传播
        Pred[:, torch.argmax(Pred, axis=1)] = 1
        Pred[Pred != 1] = 0
        correct = torch.sum((Pred == Y).all(1))  # 预测正确的样本
        total = Y.size(0)  # 全部的样本数量
        print(f'测试集精准度: {100*correct/total} %')

