from src.data import get_data
from src.LitModel import LitModel
from src.submit import create_preds

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import yaml


train = True

# Config import and logger setup
config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)


# Data import
train_loader, valid_loader, test_loader = get_data(config)

# Model Setup
model = LitModel()

if train:
    wandb_logger = WandbLogger(
        name='Custom Loss', project='Automathon')
    train = pl.Trainer(max_epochs=15, logger=wandb_logger)
    train.fit(model, train_dataloaders=train_loader,
              val_dataloaders=valid_loader)

else:
    # Testing
    create_preds(config, test_loader)
