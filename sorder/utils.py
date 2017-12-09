

import numpy as np

import os
import torch

from torch.nn.utils.rnn import pack_padded_sequence


def checkpoint(root, key, model, epoch):
    """Save model checkpoint.
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'{key}.{epoch}.pt')
    torch.save(model, path)


def pack(tensor, sizes, batch_first=True):
    """Pack padded tensors, provide reorder indexes.
    """
    # Get indexes for sorted sizes.
    size_sort = np.argsort(sizes)[::-1].tolist()

    # Sort the tensor / sizes.
    tensor = tensor[torch.LongTensor(size_sort)].type(ftype)
    sizes = np.array(sizes)[size_sort].tolist()

    packed = pack_padded_sequence(Variable(tensor), sizes, batch_first)

    return packed, size_sort
