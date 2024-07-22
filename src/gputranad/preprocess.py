"""Preprocessing module."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .models import N_WINDOWS

NUM_EPOCHS = 5
BATCH_SIZE = 128


def convert_to_windows(data):
    """Convert a 2D numpy array to a 3D tensor with overlapping windows.

    Args:
        data (np.array): 2D Numpy array to convert.

    Returns:
        torch.Tensor: 3D tensor with overlapping windows.

    """
    windows = []
    for i, g in enumerate(data):
        if i >= N_WINDOWS:
            w = data[i - N_WINDOWS : i]
        else:
            w = torch.cat([data[0].repeat(N_WINDOWS - i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)


def setup(filename, for_training=True):
    """Prepare data for training or inference.

    Args:
        filename (str): File path.
        for_training (bool, optional): Decides batch size. Defaults to True.

    Returns:
        DataLoader: DataLoader for training or inference.

    """
    raw_data = np.load(filename)
    data = convert_to_windows(torch.DoubleTensor(raw_data))
    dataset = TensorDataset(data, data)
    return DataLoader(dataset, batch_size=BATCH_SIZE if for_training else raw_data.shape[0])


def predict(model, dataloader):
    """Predict using trained Model.

    Args:
        model (nn.Module): TranAD model.
        dataloader (DataLoader): Data to predict.

    Returns:
        tuple: (loss, predictions).

    """
    mse_loss = nn.MSELoss(reduction="none")
    for d, _ in dataloader:
        window = d.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, dataloader.batch_size, model.n_feats)
        z = model(window, elem)
        if isinstance(z, tuple):
            z = z[1]
    loss = mse_loss(z, elem)[0]
    return loss.detach().numpy(), z.detach().numpy()[0]


# %%
