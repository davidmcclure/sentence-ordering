

import torch
import os


CUDA = bool(os.environ.get('CUDA'))


if CUDA:
    itype = torch.cuda.LongTensor
    ftype = torch.cuda.FloatTensor

else:
    itype = torch.LongTensor
    ftype = torch.FloatTensor
