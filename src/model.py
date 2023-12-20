"""
model.py
author: Yuxuan Zhou
date: 2023/12/20
"""

import torch.nn as nn


class LanguageIdentificationModel(nn.Module):
    # 初始化函数，定义模型的基本结构
    def __init__(self):
        # 继承父类的初始化方法
        super(LanguageIdentificationModel, self).__init__()

        # 定义神经网络结构
        # 第一个卷积层，输入为(1, 800, 80)，输出为(32, 400, 40)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        # 第二个卷积层，输入为(32, 400, 40)，输出为(64, 200, 20)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        # LSTM层，输入为(200, 64 * 20)，输出为(200, 128)
        self.lstm = nn.LSTM(input_size=64 * 20, hidden_size=128, num_layers=2, batch_first=True)
        # 全连接层，输入为(200, 128)，输出为(2)
        self.fc = nn.Linear(128, 2)

    # 定义前向传播过程
    # x的维度为(batch_size, 1, 800, 80)
    def forward(self, x):
        # (batch_size, 32, 400, 40)
        x = self.conv1(x)
        # (batch_size, 64, 200, 20)
        x = self.conv2(x)
        # (batch_size, 200, 64 * 20)
        x = x.view(x.size(0), x.size(2), -1)
        # (batch_size, 200, 128)
        x, _ = self.lstm(x)
        # (batch_size, 2)
        x = self.fc(x[:, -1, :])
        return x
