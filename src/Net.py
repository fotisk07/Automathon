import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    # Autoencoder structure
    def __init__(self):
        super().__init__()
        self.rel = nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='same')  # (BATCH_SIZE, 16, 28, 28)

        # (BATCH_SIZE, 16, 14, 14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')  # (BATCH_SIZE, 32, 14, 14)

        # (BATCH_SIZE, 32, 7, 7)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')  # (BATCH_SIZE, 64, 7, 7)

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')  # (BATCH_SIZE, 1, 7, 7)

        # (BATCH_SIZE, 1, 14, 14)
        self.up1 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=1, padding='same')  # (BATCH_SIZE, 64, 14, 14)

        # (BATCH_SIZE, 64, 28, 28)
        self.up2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding='same')  # (BATCH_SIZE, 32, 28, 28)

        self.conv7 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=3, stride=1, padding='same')  # (BATCH_SIZE, 16, 28, 28)

    def forward(self, sample):
        """

        :param x: torch.Tensor: (BATCH_SIZE, 28, 28, 1)
        :return: y1, y2: torch.Tensor: (BATCH_SIZE, 28, 28, 1)
        """

        sample = sample.permute(0, 3, 1, 2)  # (BATCH_SIZE, 1, 28, 28)

        x = self.conv1(sample)  # (BATCH_SIZE, 16, 28, 28)
        x = self.pool1(x)
        x = self.rel(x)
        x = self.conv2(x)  # (BATCH_SIZE, 32, 28, 28)
        x = self.pool2(x)  # (BATCH_SIZE, 32, 14, 14)
        x = self.rel(x)
        x = self.conv3(x)  # (BATCH_SIZE, 64, 28, 28)
        x = self.rel(x)
        x = self.conv4(x)  # (BATCH_SIZE, 128, 28, 28)
        x = self.rel(x)
        x = self.up1(x)  # (BATCH_SIZE, 128, 14, 14)
        x = self.rel(x)
        x = self.conv5(x)  # (BATCH_SIZE, 64, 14, 14)
        x = self.rel(x)
        x = self.up2(x)  # (BATCH_SIZE, 64, 28, 28)
        x = self.rel(x)
        x = self.conv6(x)  # (BATCH_SIZE, 32, 28, 28)
        x = self.rel(x)
        x = self.conv7(x)  # (BATCH_SIZE, 1, 112, 112)

        y2 = sample - x

        y1 = x.permute(0, 2, 3, 1)  # (BATCH_SIZE, 28, 28, 1)
        y2 = y2.permute(0, 2, 3, 1)  # (BATCH_SIZE, 28, 28, 1)

        return y1, y2
