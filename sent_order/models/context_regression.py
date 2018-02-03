

import attr
import torch
import os
import random
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


@attr.s
class Batch:

    grafs = attr.ib()

    def index_var_2d(self, *args, **kwargs):
        """Stack graf index tensors.
        """
        return torch.stack([
            g.index_var_2d(*args, **kwargs)
            for g in self.grafs
        ])


class Corpus:

    def __init__(self, path, skim=None, scount=None):
        """Load grafs into memory.
        """
        reader = Paragraph.read_arxiv(path, scount)

        if skim:
            reader = islice(reader, skim)

        self.grafs = list(tqdm(reader, total=skim))

    def random_batch(self, size):
        """Query random batch.
        """
        return Batch(random.sample(self.grafs, size))


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

        x = self.embeddings(x).unsqueeze(1)

        convs = [F.relu(conv(x)).squeeze(4) for conv in self.convs]

        gc = [
            F.max_pool2d(c, c.shape[-2:]).view(len(x), -1)
            for c in convs
        ]

        sc = [
            F.max_pool2d(c, (1, c.shape[-1])).view(len(x), -1)
            for c in convs
        ]

        gc = torch.cat(gc, 1)
        sc = torch.cat(sc, 1)

        x = torch.cat([gc, sc], 1)

        x = self.dropout(x)

        return self.out(x)


class Model:

    def __init__(self, *args, **kwargs):
        self.corpus = Corpus(*args, **kwargs)

    @cached_property
    def regressor(self):
        return Regressor()

    def train(self, epochs=10, epoch_size=100, lr=1e-4, batch_size=10):
        """Train for N epochs.
        """
        self.regressor.train(True)

        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=lr)

        for epoch in range(epochs):

            print(f'\nEpoch {epoch}')

            epoch_loss = 0
            for _ in tqdm(range(epoch_size)):

                optimizer.zero_grad()

                batch = self.corpus.random_batch(batch_size)

                yt, yp = self.train_batch(batch)

                loss = ((yt-yp)**2).mean()
                loss.backward()

                optimizer.step()

                epoch_loss += loss.data[0]

            print('Loss: %f' % (epoch_loss / epoch_size))

    def train_batch(self, batch):
        """Shuffle, predict.
        """
        x = batch.index_var_2d()

        perms = torch.stack([
            torch.randperm(x.shape[1])
            for _ in range(len(x))
        ])

        x = torch.stack([xi[perm] for xi, perm in zip(x, perms)])

        yt = Variable(perms.float() / (x.shape[1]-1))

        yp = self.regressor(x)

        return yt, yp
