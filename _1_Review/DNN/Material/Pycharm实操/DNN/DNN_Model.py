import torch.nn as nn

class DNN(nn.Module):
    def __init__(self):
        ''' 搭建神经网络各层 '''
        super(DNN, self).__init__()
        self.net = nn.Sequential(    # 按顺序搭建各层
            nn.Linear(3, 5), nn.ReLU(),  # 第 1 层：全连接层
            nn.Linear(5, 5), nn.ReLU(),  # 第 2 层：全连接层
            nn.Linear(5, 5), nn.ReLU(),  # 第 3 层：全连接层
            nn.Linear(5, 3)              # 第 4 层：全连接层
        )

    def forward(self, x):
        ''' 前向传播 '''
        y = self.net(x)  # x 即输入数据
        return y         # y 即输出数据

