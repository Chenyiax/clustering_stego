import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, alpha):
        super(EncoderBlock, self).__init__()
        self.alpha = alpha
        self.fc1 = nn.Linear(alpha, 512, bias=False)
        self.conv1 = BasicBlock(1, 32)
        self.conv2 = BasicBlock(32, 16)
        self.conv3 = BasicBlock(16, 32)
        self.conv4 = BasicBlock(32, 16)
        self.fc2 = nn.Linear(16 * 512, 2048, bias=False)
        self.fc3 = nn.Linear(2048, alpha, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, self.alpha)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 16 * 512)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x.view(-1, self.alpha)


class Encoder(nn.Module):
    def __init__(self, alpha=16):
        super(Encoder, self).__init__()
        self.block0 = EncoderBlock(alpha)
        self.block1 = EncoderBlock(alpha)

    def forward(self, x, m):
        '''
        :param x: 模型初始化的原始参数
        :param m: 秘密信息
        :return:  含有秘密信息的参数
        '''
        # 根据秘密信息，将参数分成两堆
        x_0 = x[m == 0]
        x_1 = x[m == 1]

        # 一堆嵌0, 一堆嵌1
        x_0 = self.block0(x_0)
        x_1 = self.block1(x_1)

        y = torch.empty_like(x)
        # 将嵌完秘密信息的参数，按秘密信息进行排列
        y[m == 0] = x_0
        y[m == 1] = x_1
        return y


class Decoder(nn.Module):
    def __init__(self, alpha=16):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(alpha, 512)
        self.conv1 = BasicBlock(1, 32)
        self.conv2 = BasicBlock(32, 16)
        self.conv3 = BasicBlock(16, 32)
        self.conv4 = BasicBlock(32, 16)

        self.fc2 = nn.Linear(512 * 16, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(-1, 1, 512)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 512 * 16)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.sigmoid(x)
        return x.view(-1)


class EncoderBlock_v2(nn.Module):
    def __init__(self, alpha):
        super(EncoderBlock_v2, self).__init__()
        self.alpha = alpha
        self.fc1 = nn.Linear(alpha, 512, bias=False)
        self.conv1 = BasicBlock(1, 32)
        self.conv2 = BasicBlock(32, 16)
        self.fc2 = nn.Linear(16 * 512, 2048, bias=False)
        self.fc3 = nn.Linear(2048, alpha, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, self.alpha)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 16 * 512)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x.view(-1, self.alpha)


class Encoder_v2(nn.Module):
    def __init__(self, alpha=16):
        super(Encoder_v2, self).__init__()
        self.block0 = EncoderBlock(alpha)
        self.block1 = EncoderBlock(alpha)

    def forward(self, x, m):
        '''
        :param x: 模型初始化的原始参数
        :param m: 秘密信息
        :return:  含有秘密信息的参数
        '''
        # 根据秘密信息，将参数分成两堆
        x_0 = x[m == 0]
        x_1 = x[m == 1]

        # 一堆嵌0, 一堆嵌1
        x_0 = self.block0(x_0)
        x_1 = self.block1(x_1)

        y = torch.empty_like(x)
        # 将嵌完秘密信息的参数，按秘密信息进行排列
        y[m == 0] = x_0
        y[m == 1] = x_1
        return y


class Decoder_v2(nn.Module):
    def __init__(self, alpha=16):
        super(Decoder_v2, self).__init__()
        self.fc1 = nn.Linear(alpha, 512)
        self.conv1 = BasicBlock(1, 32)
        self.conv2 = BasicBlock(32, 16)

        self.fc2 = nn.Linear(512 * 16, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(-1, 1, 512)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 512 * 16)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.sigmoid(x)
        return x.view(-1)
