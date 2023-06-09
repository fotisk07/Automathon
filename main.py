from src.data import get_data
from src.LitModel import LitModel
from src.submit import create_preds

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import yaml
from cliconfig import make_config


# Config import and logger setup
config = make_config('configs/default.yaml')

# Data import
train_loader, valid_loader, test_loader = get_data(config)

# Model Setup
model = LitModel()


if config['train'] == True:
    wandb_logger = WandbLogger(
        name=config["experiment_name"], project='Automathon')
    train = pl.Trainer(max_epochs=config["epochs"], logger=wandb_logger)
    train.fit(model, train_dataloaders=train_loader,
              val_dataloaders=valid_loader)

else:
    # Testing
    create_preds(config, test_loader)
