

import numpy as np

import os
import torch

from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


def checkpoint(root, key, model, epoch):
    """Save model checkpoint.
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'{key}.{epoch}.pt')
    torch.save(model, path)


def pack(tensor, sizes, ttype, batch_first=True):
    """Pack padded tensors, provide reorder indexes.
    """
    # Get indexes for sorted sizes.
    size_sort = np.argsort(sizes)[::-1].tolist()

    # Sort the tensor by size.
    tensor = tensor[torch.LongTensor(size_sort)].type(ttype)
    tensor = Variable(tensor)

    # Sort sizes descending.
    sizes = np.array(sizes)[size_sort].tolist()

    tensor = pack_padded_sequence(tensor, sizes, batch_first)

    return tensor, size_sort


def pad(tensor, length):
    """Zero-pad a tensor to given length on the right.
    """
    pad_size = length - tensor.size(0)

    padding = tensor.new(pad_size, *tensor.size()[1:]).zero_()

    return torch.cat([tensor, padding])
