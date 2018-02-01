

import torch
import os

from contextlib import contextmanager


if torch.cuda.is_available():
    itype = torch.cuda.LongTensor
    ftype = torch.cuda.FloatTensor

else:
    itype = torch.LongTensor
    ftype = torch.FloatTensor


@contextmanager
def gpu(device):
    """If CUDA is available, use a given GPU.
    """
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            yield

    else: yield
