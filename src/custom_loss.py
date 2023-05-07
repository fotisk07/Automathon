import torch
import torch.nn as nn

def loss(y1hat, y2hat, y1, y2):
    #y1 shape (BATCH_SIZE, 28, 28, 1)
    #y2 shape (BATCH_SIZE, 28, 28, 1)
    #y1hat shape (BATCH_SIZE, 28, 28, 1)
    #y2hat shape (BATCH_SIZE, 28, 28, 1)

    l1 = torch.abs((y1-y1hat)).sum(axis=[1,2,3]) + torch.abs((y2-y2hat)).sum(axis=[1,2,3])
    l2 = torch.abs((y1-y2hat)).sum(axis=[1,2,3]) + torch.abs((y2-y1hat)).sum(axis=[1,2,3])

    L1 = torch.stack([l1,l2], dim=1)
    return L1.min(axis=1)[0].mean()
