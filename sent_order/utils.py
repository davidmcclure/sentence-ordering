

import numpy as np
import re
import torch

from itertools import groupby
from functools import reduce

from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .cuda import itype


def parse_int(text):
    """Parse an integer out of a string.
    """
    matches = re.findall('[0-9]+', text)
    return int(matches[0]) if matches else None


def remove_consec_dupes(seq):
    """Remove consecutive duplicates in a list.
    [1,1,2,2,3,3] -> [1,2,3]
    """
    return [x[0] for x in groupby(seq)]


# TODO: Test.
def regroup_indexes(seq, size_fn):
    """Given a sequence A that contains items of variable size, provide a list
    of indexes that, when iterated as pairs, will slice a flat sequence B into
    groups with sizes that correspond to the items in A.

    Args:
        seq (iterable)
        size_fn (func): Provides size of an individual item in `seq`.

    Returns: list<int>
    """
    return reduce(lambda ix, i: (*ix, ix[-1] + size_fn(i)), seq, (0,))


def parse_int(text):
    """Parse an integer out of a string.
    """
    matches = re.findall('[0-9]+', text)
    return int(matches[0]) if matches else None


def pad_right_and_stack(xs, pad_size=None):
    """Pad and stack a list of variable-length seqs.

    Args:
        xs (list<tensor>)
        pad_size (int)

    Returns: stacked xs, sizes
    """
    # Default to max seq size.
    if not pad_size:
        pad_size = max([len(x) for x in xs])

    padded, sizes = [], []
    for x in xs:

        px = F.pad(x, (0, pad_size-len(x)))
        padded.append(px)

        size = min(pad_size, len(x))
        sizes.append(size)

    return torch.stack(padded), sizes


def pack(x, sizes, batch_first=True):
    """Pack padded variables, provide reorder indexes.

    Args:
        batch (Variable)
        sizes (list[int])

    Returns: packed sequence, reorder indexes
    """
    # Get indexes for sorted sizes.
    size_sort = np.argsort(sizes)[::-1].tolist()

    # Sort tensor by size.
    x = x[torch.LongTensor(size_sort).type(itype)]

    # Sort sizes descending.
    sizes = np.array(sizes)[size_sort].tolist()

    # Pack the sequences.
    x = pack_padded_sequence(x, sizes, batch_first)

    # Indexes to restore original order.
    reorder = torch.LongTensor(np.argsort(size_sort)).type(itype)

    return x, reorder
