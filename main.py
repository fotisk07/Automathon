from src.data import get_data
from src.Net import Net
from src.LitModel import LitModel
from src.submit import create_preds

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import yaml
from pytorch_lightning.utilities import model_summary
from torchinfo import summary


# Config import and logger setup
config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
wandb_logger = WandbLogger(name='Basic Architecture', project='Automathon')

# Data import
train_loader, valid_loader, test_loader = get_data(config)

# Model Setup
network = Net()
summary(network, input_size=(16, 28, 28, 1))

model = LitModel(network)

# Training
train = pl.Trainer(max_epochs=10, logger=wandb_logger)
train.fit(model, train_dataloaders=train_loader,
          val_dataloaders=valid_loader)


# Testing
create_preds(config, network, test_loader)
