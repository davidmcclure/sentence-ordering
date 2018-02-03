

import attr
import torch
import os
import ujson

from torchtext.vocab import Vectors

from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

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

    def index_var_1d(self, perm=None, pad=None):
        """Token indexes, flattened to 1d series.
        """
        perm = perm or range(len(self.sents))

        idx = [ti for si in perm for ti in self.sents[si].indexes]
        idx = Variable(torch.LongTensor(idx))

        if pad:
            idx = F.pad(idx, (0, pad-len(idx)))

        return idx


class Corpus:

    def __init__(self, path, skim=None):
        """Load grafs into memory.
        """
        reader = Paragraph.read_arxiv(path)

        if skim:
            reader = islice(reader, skim)

        self.grafs = list(tqdm(reader, total=skim))


class Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.embeddings = nn.Embedding(
            VECTORS.vectors.shape[0],
            VECTORS.vectors.shape[1],
        )

        self.embeddings.weight.data.copy_(VECTORS.vectors)

        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, 100, (n, VECTORS.vectors.shape[1]))
            for n in (3, 4, 5)
        ])

        self.dropout = nn.Dropout()

        self.out = nn.Linear(300, 5)

    def forward(self, x):

        embeds = self.embeddings(x)

        x = embeds.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)

        x = self.out(x)

        return F.log_softmax(x, dim=1)
