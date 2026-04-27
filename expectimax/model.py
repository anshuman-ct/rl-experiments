# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpectimaxCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2)

        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        # x: (B, 1, 4, 4)

        x = F.relu(self.conv1(x))   # (B,64,3,3)
        x = F.relu(self.conv2(x))   # (B,128,2,2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)