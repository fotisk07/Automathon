from src.LitModel import LitModel
import torch
import numpy as np
import pytorch_lightning as pl


def create_preds(config, network, data_loader):
    print("Loading up the model")

    model = LitModel.load_from_checkpoint(
        checkpoint_path=config["checkpoint_path"], model=network)

    print("Done loading the model")
    print("Testing...")

    model = LitModel.load_from_checkpoint(
        checkpoint_path="Automathon/18lmvoly/checkpoints/epoch=9-step=4380.ckpt", model=network)
    trainer = pl.Trainer()
    predictions = trainer.predict(model, dataloaders=data_loader)

    predictions = [pred.detach().numpy() for pred in predictions]
    predictions = np.array(predictions)
    predictions = np.reshape(predictions, (-1, 28, 28, 2))

    print("Done testing")

    np.save("predictions.npy", predictions)
