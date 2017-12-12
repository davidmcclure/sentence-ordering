

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


def pad(variable, size):
    """Zero-pad a variable to given length on the right.

    Args:
        variable (Variable)
        size (int)

    Returns: padded variable, size
    """
    # Truncate long inputs.
    variable = variable[:size]

    var_size = variable.size(0)

    # If too short, pad to length.
    if var_size < size:
        padding = torch.zeros(size - var_size, *variable.size()[1:])
        variable = torch.cat([variable, Variable(padding)])

    return variable, var_size


def pack(batch, sizes, batch_first=True):
    """Pack padded variables, provide reorder indexes.

    Args:
        batch (Variable)
        sizes (list[int])

    Returns: packed sequence, reorder indexes
    """
    # Get indexes for sorted sizes.
    size_sort = np.argsort(sizes)[::-1].tolist()

    # Sort the tensor by size.
    batch = batch[torch.LongTensor(size_sort)]

    # Sort sizes descending.
    sizes = np.array(sizes)[size_sort].tolist()

    batch = pack_padded_sequence(batch, sizes, batch_first)

    # Indexes to restore original order.
    reorder = torch.LongTensor(np.argsort(size_sort))

    return batch, reorder


def pad_and_pack(variables, size, *args, **kwargs):
    """Pad a list of tensors to a given length, pack.

    Args:
        tensors (list): Variable-length tensors.
    """
    padded, sizes = zip(*[pad(v, size) for v in variables])

    return pack(torch.stack(padded), sizes, *args, **kwargs)
