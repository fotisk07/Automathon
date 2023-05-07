from typing import Any
import pytorch_lightning as pl
from torch import optim
from torch import nn
import torch
from src.Net import Net


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Net()
        self.loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y1, y2 = batch.split(1, 3)

        y1_pred, y2_pred = self.model(x)
        loss = self.loss(y1_pred, y1) + self.loss(y2_pred, y2)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y1, y2 = batch.split(1, 3)
        y1_pred, y2_pred = self.model(x)
        loss = self.loss(y1_pred, y1) + self.loss(y2_pred, y2)

        self.log("valid/loss", loss)

    def forward(self, batch):
        # used by the testing script
        x = batch

        y1_pred, y2_pred = self.model(x)

        return torch.cat([y1_pred, y2_pred], dim=3)
