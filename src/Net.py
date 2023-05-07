import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='same')

        self.rel = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same')

    def forward(self, sample):
        """

        :param x: torch.Tensor: (BATCH_SIZE, 28, 28, 1)
        :return: y1, y2: torch.Tensor: (BATCH_SIZE, 28, 28, 1)
        """

        sample = sample.permute(0, 3, 1, 2)  # (BATCH_SIZE, 1, 28, 28)

        x = self.conv1(sample)  # (BATCH_SIZE, 16, 28, 28)
        x = self.rel(x)
        x = self.conv2(x)  # (BATCH_SIZE, 32, 28, 28)
        x = self.rel(x)
        x = self.conv3(x)  # (BATCH_SIZE, 64, 28, 28)
        x = self.rel(x)
        x = self.conv4(x)  # (BATCH_SIZE, 2, 28, 28)

        y2 = sample - x

        y1 = x.permute(0, 2, 3, 1)  # (BATCH_SIZE, 28, 28, 1)
        y2 = y2.permute(0, 2, 3, 1)  # (BATCH_SIZE, 28, 28, 1)

        return y1, y2
