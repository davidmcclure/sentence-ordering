

import numpy as np

import os
import torch

from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from .cuda import ftype, itype


def checkpoint(root, key, model, epoch):
    """Save model checkpoint.
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'{key}.{epoch}.pt')
    torch.save(model, path)


def pad(tensor, size):
    """Zero-pad a tensor to given length on the right.
    """
    if len(tensor) >= size:
        return tensor[:size], size

    pad_size = size - tensor.size(0)

    padding = Variable(torch.zeros(pad_size, *tensor.size()[1:]))

    return torch.cat([tensor, padding]), tensor.size(0)


def pack(tensor, sizes, ttype=ftype, batch_first=True):
    """Pack padded tensors, provide reorder indexes.
    """
    # Get indexes for sorted sizes.
    size_sort = np.argsort(sizes)[::-1].tolist()

    # Sort the tensor by size.
    tensor = tensor[torch.LongTensor(size_sort)].type(ttype)

    # Sort sizes descending.
    sizes = np.array(sizes)[size_sort].tolist()

    tensor = pack_padded_sequence(tensor, sizes, batch_first)

    # Indexes to restore original order.
    reorder = torch.LongTensor(np.argsort(size_sort)).type(itype)

    return tensor, reorder


def pad_and_pack(tensors, size, *args, **kwargs):
    """Pad a list of tensors to a given length, pack.

    Args:
        tensors (list): Variable-length tensors.
    """
    padded, sizes = zip(*[pad(t, size) for t in tensors])

    return pack(torch.stack(padded), sizes, *args, **kwargs)
