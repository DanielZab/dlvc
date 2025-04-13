from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch


class YourCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_net = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.conv_net(x)
        assert x.shape[2:] == (4,4), f"Wrong shape {x.shape}"
        x = torch.flatten(x, 1)
        x = self.fc_net(x)
        assert (torch.sum(x, dim=1) < 1.001).all() and (torch.sum(x, dim=1) > 0.999).all(), "Softmax not working"
        return x

    