import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from math import log

def one_hot_encode(board):
    data = torch.zeros(1, 16, 4, 4)
    for i in range(4):
        for j in range(4):
            if board[i,j] == 0:
                data[0, 0, i, j] = 1
            else:
                data[0, int(log(board[i, j], 2)), i, j] = 1
    return data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(16, 128, (1, 4))
        self.conv2 = nn.Conv2d(16, 128, (4, 1))
        self.conv3 = nn.Conv2d(16, 128, (1, 2))
        self.conv4 = nn.Conv2d(16, 128, (2, 1))
        self.conv5 = nn.Conv2d(16, 128, 2)
        self.conv6 = nn.Conv2d(16, 128, 3)
        self.conv7 = nn.Conv2d(16, 128, 4)
        self.fc1 = nn.Linear(5888, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = f.relu(self.conv1(x))
        x2 = f.relu(self.conv2(x))
        x3 = f.relu(self.conv3(x))
        x4 = f.relu(self.conv4(x))
        x5 = f.relu(self.conv5(x))
        x6 = f.relu(self.conv6(x))
        x7 = f.relu(self.conv7(x))
        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0], -1)
        x3 = x3.view(x3.size()[0], -1)
        x4 = x4.view(x4.size()[0], -1)
        x5 = x5.view(x5.size()[0], -1)
        x6 = x6.view(x5.size()[0], -1)
        x7 = x7.view(x5.size()[0], -1)
        x = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=1)
        x = f.relu(nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x))
        x = self.dropout(self.fc1(x))
        x = f.relu(nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x))
        x = self.dropout(self.fc2(x))
        x = f.relu(nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x))
        x = self.fc3(x)
        return x

    def find_direction(self, board):
        inputs = one_hot_encode(board)
        output = self(inputs)
        _, predict = torch.max(output.data, -1)
        return int(predict)

