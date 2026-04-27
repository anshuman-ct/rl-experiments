import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(16, 128, kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(16, 128, kernel_size=(2, 1))

        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv12 = nn.Conv2d(128, 128, kernel_size=(2, 1))
        self.conv21 = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv22 = nn.Conv2d(128, 128, kernel_size=(2, 1))

        first = 4 * 3 * 128 * 2
        second = (2 * 4 * 128 * 2) + (3 * 3 * 128 * 2)

        self.fc = nn.Sequential(
            nn.Linear(first + second, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = x.to(config.DEVICE)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))

        x11 = F.relu(self.conv11(x1))
        x12 = F.relu(self.conv12(x1))
        x21 = F.relu(self.conv21(x2))
        x22 = F.relu(self.conv22(x2))

        def flat(t):
            return t.view(t.size(0), -1)

        concat = torch.cat([
            flat(x1), flat(x2),
            flat(x11), flat(x12),
            flat(x21), flat(x22)
        ], dim=1)

        return self.fc(concat)