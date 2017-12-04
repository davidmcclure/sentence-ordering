

import os
import torch


def checkpoint(root, key, model, epoch):
    """Save model checkpoint.
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'{key}.{epoch}.pt')
    torch.save(model, path)
