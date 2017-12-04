

import torch
import os


CUDA = bool(os.environ.get('CUDA'))

ftype = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

itype = torch.cuda.LongTensor if CUDA else torch.LongTensor
