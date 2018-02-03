

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
    def read_arxiv(cls, path, scount=None):
        """Wrap parsed arXiv abstracts as paragraphs.
        """
        for path in glob(os.path.join(path, '*.json')):
            for line in open(path):

                graf = cls.from_arxiv_json(line)

                # Filter by sentence count.
                if not scount or len(graf) == scount:
                    yield graf

    @classmethod
    def from_arxiv_json(cls, line):
        """Parse JSON, take tokens.
        """
        json = ujson.loads(line.strip())

        return cls([
            Sentence(s['token'])
            for s in json['sentences']
        ])

    def __len__(self):
        return len(self.sents)

    def index_var_1d(self, perm=None, pad=None):
        """Token indexes, flattened to 1d series.
        """
        perm = perm or range(len(self.sents))

        idx = [ti for si in perm for ti in self.sents[si].indexes]
        idx = Variable(torch.LongTensor(idx))

        if pad:
            idx = F.pad(idx, (0, pad-len(idx)))

        return idx

    def index_var_2d(self, pad=50):
        """Token indexes, flattened to 1d series.
        """
        idx = []
        for sent in self.sents:
            sidx = Variable(torch.LongTensor(sent.indexes))
            sidx = F.pad(sidx, (0, pad-len(sidx)))
            idx.append(sidx)

        return torch.stack(idx)


class Corpus:

    def __init__(self, path, skim=None, scount=None):
        """Load grafs into memory.
        """
        reader = Paragraph.read_arxiv(path, scount)

        if skim:
            reader = islice(reader, skim)

        self.grafs = list(tqdm(reader, total=skim))


class Regressor(nn.Module):

    def __init__(self):

        super().__init__()

        self.embeddings = nn.Embedding(
            VECTORS.vectors.shape[0],
            VECTORS.vectors.shape[1],
        )

        self.embeddings.weight.data.copy_(VECTORS.vectors)

        self.convs = nn.ModuleList([
            nn.Conv3d(1, 500, (1, n, VECTORS.vectors.shape[1]))
            for n in range(1, 10)
        ])

        self.dropout = nn.Dropout()

        self.out = nn.Linear(9*500*6, 5)

    def forward(self, x):

        # batch | in channel | sent | word | embed
        x = self.embeddings(x).unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(4) for conv in self.convs]

        gx = [F.max_pool2d(xi, xi.shape[-2:]).view(1, -1) for xi in x]
        gx = torch.cat(gx, 1)

        sx = [F.max_pool2d(xi, (1, xi.shape[-1])).view(1, -1) for xi in x]
        sx = torch.cat(sx, 1)

        x = torch.cat([gx, sx], 1)

        x = self.dropout(x)

        return self.out(x)
