

import numpy as np

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
from scipy.stats import kendalltau
from tqdm import tqdm

from ..cuda import ftype, itype


class LazyVectors:

    unk_idx = 1

    def __init__(self, name='glove.840B.300d.txt'):
        self.name = name
        self.set_vocab([])

    @cached_property
    def loader(self):
        return Vectors(self.name)

    def set_vocab(self, vocab):
        self._itos = [v for v in vocab if v in self.loader.stoi]
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    @cached_property
    def weights(self):
        weights = torch.stack([
            self.loader.vectors[self.loader.stoi[s]]
            for s in self._itos
        ])

        return torch.cat([torch.zeros((2, self.loader.dim)), weights])

    def stoi(self, s):
        idx = self._stoi.get(s)
        return idx + 2 if idx else self.unk_idx


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
        idx = Variable(torch.LongTensor(idx)).type(itype)

        if pad:
            idx = F.pad(idx, (0, pad-len(idx)))

        return idx

    def index_var_2d(self, pad=50):
        """Token indexes, flattened to 1d series.
        """
        idx = []
        for sent in self.sents:
            sidx = Variable(torch.LongTensor(sent.indexes)).type(itype)
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

    def vocab(self):
        """Build vocab list.
        """
        vocab = set()

        for graf in self.grafs:
            for sent in graf.sents:
                vocab.update(sent.tokens)

        return list(vocab)


class Regressor(nn.Module):

    def __init__(self):

        super().__init__()

        self.embeddings = nn.Embedding(
            VECTORS.weights.shape[0],
            VECTORS.weights.shape[1],
        )

        self.embeddings.weight.data.copy_(VECTORS.weights)

        self.convs = nn.ModuleList([
            nn.Conv3d(1, 5000, (1, n, VECTORS.weights.shape[1]))
            for n in range(1, 6)
        ])

        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(5*5000*6, 500)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 5)

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

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class Model:

    def __init__(self, *args, **kwargs):
        self.corpus = Corpus(*args, **kwargs)
        VECTORS.set_vocab(self.corpus.vocab())

    @cached_property
    def regressor(self):
        """Initialize regressor.
        """
        reg = Regressor()

        if torch.cuda.is_available():
            reg.cuda()

        return reg

    def train(self, epochs=10, epoch_size=10, lr=1e-4, batch_size=10):
        """Train for N epochs.
        """
        self.regressor.train(True)

        params = [p for p in self.regressor.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(params, lr=lr)

        for epoch in range(epochs):

            print(f'\nEpoch {epoch}')

            epoch_loss = 0
            kts = []
            for _ in tqdm(range(epoch_size)):

                optimizer.zero_grad()

                batch = self.corpus.random_batch(batch_size)

                yt, yp = self.train_batch(batch)

                loss = ((yt-yp)**2).mean(1).mean()
                loss.backward()

                optimizer.step()

                epoch_loss += loss.data[0]

                for t, p in zip(yt, yp):
                    t = np.argsort(list(t.data))
                    p = np.argsort(list(p.data))
                    kt = kendalltau(t, p)
                    kts.append(kt.correlation)

            print('Loss: %f' % (epoch_loss / epoch_size))
            print('KT: %f' % np.mean(kts))
            print('PMR: %f' % (kts.count(1) / len(kts)))

    def train_batch(self, batch):
        """Shuffle, predict.
        """
        x = batch.index_var_2d()

        perms = torch.stack([
            torch.randperm(x.shape[1]).type(itype)
            for _ in range(len(x))
        ])

        x = torch.stack([xi[perm] for xi, perm in zip(x, perms)])

        yt = Variable(perms.float() / (x.shape[1]-1)).type(ftype)

        yp = self.regressor(x)

        return yt, yp
