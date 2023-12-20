"""
dataloader.py
author: Yuxuan Zhou
date: 2023/12/20
"""

import glob
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# 读取数据集
def load_data():
    zh_path = glob.glob("../train_data/language_0/*.npy")
    en_path = glob.glob("../train_data/language_1/*.npy")

    # 梅尔频谱
    mel_spectrogram = []
    # 语种标签
    labels = []

    for path in zh_path:
        mel_spectrogram.append(np.load(path))
    labels += [0] * len(zh_path)

    for path in en_path:
        mel_spectrogram.append(np.load(path))
    labels += [1] * len(en_path)

    # mel_spectrogram是一个list，其中每个元素是一个numpy array，代表一条梅尔频谱语音数据，维度为(x, 80)
    # labels是一个list，其中每个元素是一个int，代表对应的语种标签，0为中文，1为英文
    return mel_spectrogram, labels


# 将梅尔频谱的维度统一至(uniform_len, 80)
def uniform(mel_spectrogram, uniform_len=800):
    for i in range(len(mel_spectrogram)):
        # 如果梅尔频谱的长度小于uniform_len，则在后面补[0, 0, ..., 0]
        if mel_spectrogram[i].shape[0] < uniform_len:
            mel_spectrogram[i] = np.concatenate((mel_spectrogram[i], np.zeros(
                (uniform_len - mel_spectrogram[i].shape[0], 80))), axis=0)
        # 如果梅尔频谱的长度大于uniform_len，则截断后面的部分
        else:
            mel_spectrogram[i] = mel_spectrogram[i][:uniform_len, :]


# 将数据集分为训练集和验证集，并加载为DataLoader
def split_data(mel_spectrogram, labels, valid_size, batch_size, random_state):
    # 将数据集划分为训练集和验证集
    x_train, x_valid, y_train, y_valid = train_test_split(mel_spectrogram,
                                                          labels,
                                                          test_size=valid_size,
                                                          random_state=random_state)
    # 将训练集和验证集从list转换为numpy array
    x_train, x_valid, y_train, y_valid = np.array(x_train), np.array(x_valid), np.array(y_train), np.array(y_valid)

    # 将训练集和验证集的数据类型转换为tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_valid = torch.tensor(x_valid, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_valid = torch.tensor(y_valid, dtype=torch.int64)

    # 将训练集和验证集的特征和标签组合
    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)

    # 使用DataLoader加载数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_loader, valid_loader
