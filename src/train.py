"""
train.py
author: Yuxuan Zhou
date: 2023/12/20
"""

import time
import torch
import torch.optim as optim


def fit(model, train_data, valid_data, epochs, learning_rate, device):
    # 设置随机种子
    # torch.manual_seed(3407)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_start = time.time()

    # 训练模型
    for epoch in range(epochs):
        start = time.time()

        # 训练模式
        model.train()

        # 初始化训练集和验证集的损失和准确率
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0

        for i, data in enumerate(train_data):
            # 将数据加载到device中
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            loss.backward()
            # 更新参数
            optimizer.step()

        with torch.no_grad():
            # 验证模式
            model.eval()

            for i, data in enumerate(train_data):
                # 将数据加载到device中
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 前向传播
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                # 计算损失
                loss = criterion(outputs, labels)
                train_loss += loss.item()
                # 计算准确率
                _, preds = torch.max(outputs, 1)
                train_acc += sum(preds == labels.data).detach()

            for i, data in enumerate(valid_data):
                # 将数据加载到device中
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 前向传播
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                # 计算损失
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                # 计算准确率
                _, preds = torch.max(outputs, 1)
                valid_acc += sum(preds == labels.data).detach()

        end = time.time()

        # 计算训练集和验证集的损失和准确率
        train_loss = train_loss / len(train_data.dataset)
        valid_loss = valid_loss / len(valid_data.dataset)
        train_acc = train_acc / len(train_data.dataset)
        valid_acc = valid_acc / len(valid_data.dataset)

        print('Epoch: {} \t'
              'Training Loss: {:.6f} \t'
              'Validation Loss: {:.6f} \t'
              'Training Acc: {:.6f} \t'
              'Validation Acc: {:.6f} \t'
              'Time: {:.6f}s'.format(epoch + 1, train_loss, valid_loss, train_acc, valid_acc, end - start))

    total_end = time.time()
    print('Total Time: {:.6f}s'.format(total_end - total_start))
