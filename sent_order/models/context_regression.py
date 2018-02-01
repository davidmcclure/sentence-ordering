

import attr
import torch

from torchtext.vocab import Vectors

from cached_property import cached_property


@attr.s
class LazyVectors:

    name = attr.ib()

    unk_idx = 1

    @cached_property
    def loader(self):
        return Vectors(self.name)

    @cached_property
    def dim(self):
        return self.loader.dim

    @cached_property
    def vectors(self):
        return torch.cat([
            torch.zeros((2, self.dim)),
            self.loader.vectors,
        ])

    def stoi(self, s):
        idx = self.loader.stoi.get(s)
        return idx + 2 if idx else self.unk_idx

    def itos(self, i):
        return self.loader.itos[i - 2]


@attr.s
class Sentence:
    tokens = attr.ib()
