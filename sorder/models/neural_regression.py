

import attr
import os
import ujson
import random
import click

import numpy as np

from glob import glob
from itertools import islice
from tqdm import tqdm
from cached_property import cached_property
from gensim.models import KeyedVectors

import torch
from torch import nn
from torch.autograd import Variable

from sorder.vectors import LazyVectors
from sorder.cuda import CUDA, ftype, itype


vectors = LazyVectors.read()


def read_abstracts(path, maxlen):
    """Parse abstract JSON lines.
    """
    for path in glob(os.path.join(path, '*.json')):
        with open(path) as fh:
            for line in fh:
                json = ujson.loads(line.strip())
                if len(json['sentences']) < maxlen:
                    yield Abstract.from_json(json)


@attr.s
class Sentence:

    tokens = attr.ib()

    def tensor(self, dim=300, pad=50):
        """Stack word vectors, padding zeros on left.
        """
        x = [vectors[t] for t in self.tokens if t in vectors]
        x += [np.zeros(dim)] * pad
        x = x[:pad]
        x = list(reversed(x))
        x = np.array(x)
        x = torch.from_numpy(x)
        x = x.float()
        return x


@attr.s
class Abstract:

    sentences = attr.ib()

    @classmethod
    def from_json(cls, json):
        """Pull out raw token series.
        """
        return cls([Sentence(s['token']) for s in json['sentences']])

    def tensor(self):
        """Stack sentence tensors.
        """
        tensors = [s.tensor() for s in self.sentences]
        return torch.stack(tensors)


@attr.s
class AbstractBatch:

    abstracts = attr.ib()

    def tensor(self):
        """Stack abstract tensors.
        """
        tensors = [a.tensor() for a in self.abstracts]
        return torch.cat(tensors)


class Corpus:

    def __init__(self, path, skim=None, maxlen=10):
        """Load abstracts into memory.
        """
        reader = read_abstracts(path, maxlen)

        if skim:
            reader = islice(reader, skim)

        self.abstracts = list(tqdm(reader, total=skim))

    def random_batches(self, n, size):
        for _ in tqdm(range(n)):
            yield random.sample(self.abstracts, size)


class SentenceEncoder(nn.Module):

    def __init__(self, lstm_dim=512):
        super().__init__()
        self.lstm = nn.LSTM(300, lstm_dim, batch_first=True)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x)
        return hn.squeeze()

    def encode_batch(self, batch):
        """Encode sentences in an abstract batch.

        Args:
            batch (list of Abstract)

        Yields: Unpacked tensors for each abstract.
        """
        # Combine sentences into single batch.
        x = torch.cat([a.tensor() for a in batch])
        x = Variable(x).type(ftype)

        y = self(x)

        # Unpack into separate tensor for each abstract.
        start = 0
        for a in batch:
            yield y[start:start+len(a.sentences)]
            start += len(a.sentences)

    def batch_xy(self, batch):
        """Given a batch, encode sentences and make x/y training pairs.

        Args:
            batch (list of Abstract)
        """
        x = []
        y = []
        for a in self.encode_batch(batch):
            for i in range(len(a)):
                x.append(a[i])
                y.append(i / (len(a)-1))

        x = torch.stack(x)
        y = Variable(torch.FloatTensor(y)).type(ftype)

        return x, y


class Regressor(nn.Module):

    def __init__(self, input_dim=512):
        super().__init__()
        self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        y = self.out(x)
        return y.squeeze()


@click.command()
@click.argument('train_path', type=click.Path())
@click.option('--train_skim', type=int, default=10000)
@click.option('--lr', type=float, default=1e-4)
@click.option('--epochs', type=int, default=50)
@click.option('--epoch_size', type=int, default=100)
@click.option('--batch_size', type=int, default=10)
def main(train_path, train_skim, lr, epochs, epoch_size, batch_size):

    torch.manual_seed(1)

    train = Corpus(train_path, train_skim)

    encoder = SentenceEncoder()
    regressor = Regressor()

    params = list(encoder.parameters()) + list(regressor.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    criterion = nn.MSELoss()

    if CUDA:
        encoder = encoder.cuda()
        regressor = regressor.cuda()
        criterion = criterion.cuda()

    first_loss = None
    for epoch in range(epochs):

        print(f'\nEpoch {epoch}')

        epoch_loss = 0
        for batch in train.random_batches(epoch_size, batch_size):

            optimizer.zero_grad()

            x, y = encoder.batch_xy(batch)
            y_pred = regressor(x)

            loss = criterion(y_pred, y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.data[0]

        if not first_loss:
            first_loss = epoch_loss

        epoch_loss /= first_loss
        print(epoch_loss)


if __name__ == '__main__':
    main()
