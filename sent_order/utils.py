

import numpy as np

import os
import random
import torch
import scandir
import re

from collections import OrderedDict

from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from .cuda import ftype, itype


def checkpoint(root, key, model, epoch=0):
    """Save model checkpoint.
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'{key}.{epoch}.bin')
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

    # Original data size.
    var_size = variable.size(0)

    # If too short, pad to length.
    if var_size < size:

        padding = variable.data.new(size-var_size, *variable.size()[1:])
        padding = padding.zero_()

        variable = torch.cat([variable, Variable(padding)])

    return variable, var_size


def pad_and_stack(variables, size):
    """Pad a batch of variables

    Args:
        variables (list of Variable)
        size (int)

    Returns: stacked tensor, sizes
    """
    padded, sizes = zip(*[pad(v, size) for v in variables])

    return torch.stack(padded), sizes


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
    batch = batch[torch.LongTensor(size_sort).type(itype)]

    # Sort sizes descending.
    sizes = np.array(sizes)[size_sort].tolist()

    batch = pack_padded_sequence(batch, sizes, batch_first)

    # Indexes to restore original order.
    reorder = torch.LongTensor(np.argsort(size_sort)).type(itype)

    return batch, reorder


def pad_and_pack(variables, pad_size):
    """Pad a list of tensors to a given length, pack.

    Args:
        tensors (list): Variable-length tensors.
    """
    padded, sizes = pad_and_stack(variables, pad_size)

    return pack(padded, sizes)


def sort_by_key(d, desc=False):
    """Sort dictionary by key.
    """
    items = sorted(d.items(), key=lambda x: x[0], reverse=desc)

    return OrderedDict(items)


def scan_paths(root, pattern=None):
    """Scan paths by regex.
    """
    for root, dirs, files in scandir.walk(root, followlinks=True):
        for name in files:

            # Match the pattern.
            if not pattern or re.search(pattern, name):
                yield os.path.join(root, name)
