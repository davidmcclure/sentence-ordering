

import torch


if torch.cuda.is_available():
    itype = torch.cuda.LongTensor
    ftype = torch.cuda.FloatTensor

else:
    itype = torch.LongTensor
    ftype = torch.FloatTensor
