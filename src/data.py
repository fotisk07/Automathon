import numpy as np
from torch.utils.data import Dataset
import torch


class customDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.dataset[index, :, :, :1]
        y1 = self.dataset[index, :, :, 1:2]
        y2 = self.dataset[index, :, :, 2:]

        x = torch.from_numpy(x).float()
        y1 = torch.from_numpy(y1).float()
        y2 = torch.from_numpy(y2).float()

        out = torch.cat([x, y1, y2], dim=2)
        return out


def get_data(config):
    # Load data
    original_train = np.load(config["train_path"])
    original_test = np.load(config["test_path"])

    # Reshape data (N, 28, 28, 1) -> (n, BATCH_SIZE, 28, 28, 1)

    np.random.shuffle(original_train)

    dataset_train = original_train[:7000]
    dataset_valid = original_train[7000:]

    train_dataset = customDataset(dataset_train)
    valid_dataset = customDataset(dataset_valid)
    test_dataset = customDataset(original_test)

    params = {'batch_size': config["batch_size"],
              'shuffle': config["shuffle"], 'num_workers': config["num_workers"]}

    params_val = {'batch_size': config["batch_size"],
                  'shuffle': False, 'num_workers': config["num_workers"]}

    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, **params_val)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    return train_loader, valid_loader, test_loader
