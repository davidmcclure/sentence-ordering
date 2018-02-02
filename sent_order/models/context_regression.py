

import attr
import torch
import os
import ujson

from torchtext.vocab import Vectors

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

from cached_property import cached_property
from glob import glob
from itertools import islice
from tqdm import tqdm


@attr.s
class LazyVectors:

    name = attr.ib(default='glove.840B.300d.txt')

    unk_idx = 1

    @cached_property
    def loader(self):
        return Vectors(self.name)

    @cached_property
    def vectors(self):
        return torch.cat([
            torch.zeros((2, self.loader.dim)),
            self.loader.vectors,
        ])

    def stoi(self, s):
        idx = self.loader.stoi.get(s)
        return idx + 2 if idx else self.unk_idx

    def itos(self, i):
        return self.loader.itos[i - 2]


VECTORS = LazyVectors()


@attr.s
class Sentence:

    tokens = attr.ib()

    @cached_property
    def indexes(self):
        return [VECTORS.stoi(s) for s in self.tokens]


@attr.s
class Paragraph:

    sents = attr.ib()

    @classmethod
    def read_arxiv(cls, path):
        """Wrap parsed arXiv abstracts as paragraphs.
        """
        for path in glob(os.path.join(path, '*.json')):
            for line in open(path):
                yield cls.from_arxiv_json(line)

    @classmethod
    def from_arxiv_json(cls, line):
        """Parse JSON, take tokens.
        """
        json = ujson.loads(line.strip())

        return cls([
            Sentence(s['token'])
            for s in json['sentences']
        ])

    def indexes(self, perm=None):
        perm = perm or range(len(self.sents))
        return [ti for si in perm for ti in self.sents[si].indexes]


class Corpus:

    def __init__(self, path, skim=None):
        """Load grafs into memory.
        """
        reader = Paragraph.read_arxiv(path)

        if skim:
            reader = islice(reader, skim)

        self.grafs = list(tqdm(reader, total=skim))
