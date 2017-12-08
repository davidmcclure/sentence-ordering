

import numpy as np

import os
import click
import torch
import attr
import random
import ujson

from tqdm import tqdm
from itertools import islice
from glob import glob
from scipy import stats

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from sorder.cuda import CUDA, ftype, itype
from sorder.utils import checkpoint
from sorder.vectors import LazyVectors


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

    def xy(self):
        """Generate x,y pairs.
        """
        for i, s in enumerate(self.sentences):
            y = i / (len(self.sentences)-1)
            yield s.tensor(), y


@attr.s
class Batch:

    abstracts = attr.ib()

    def tensor(self):
        """Stack abstract tensors.
        """
        tensors = [a.tensor() for a in self.abstracts]
        return torch.cat(tensors)

    def xy(self):
        """Generate x, y pairs.
        """
        for ab in self.abstracts:
            yield from ab.xy()

    def xy_tensors(self):
        """Generate x + y tensors
        """
        x, y = zip(*self.xy())

        x = Variable(torch.stack(x)).type(ftype)
        y = Variable(torch.FloatTensor(y)).type(ftype)

        return x, y


class Corpus:

    def __init__(self, path, skim=None, maxlen=10):
        """Load abstracts into memory.
        """
        reader = read_abstracts(path, maxlen)

        if skim:
            reader = islice(reader, skim)

        self.abstracts = list(tqdm(reader, total=skim))

    def random_batch(self, size):
        return Batch(random.sample(self.abstracts, size))


class Model(nn.Module):

    def __init__(self, lstm_dim):
        super().__init__()
        self.lstm = nn.LSTM(300, lstm_dim, batch_first=True)
        self.out = nn.Linear(lstm_dim, 1)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x)
        return self.out(hn.squeeze()).squeeze()


@click.group()
def cli():
    pass


@cli.command()
@click.argument('train_path', type=click.Path())
@click.argument('model_path', type=click.Path())
@click.option('--train_skim', type=int, default=10000)
@click.option('--lr', type=float, default=1e-4)
@click.option('--epochs', type=int, default=100)
@click.option('--epoch_size', type=int, default=100)
@click.option('--batch_size', type=int, default=10)
@click.option('--lstm_dim', type=int, default=1024)
def train(train_path, model_path, train_skim, lr, epochs, epoch_size,
    batch_size, lstm_dim):
    """Train model.
    """
    train = Corpus(train_path, train_skim)

    model = Model(lstm_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_func = nn.L1Loss()

    if CUDA:
        model = model.cuda()
        loss_func = loss_func.cuda()

    first_loss = None
    for epoch in range(epochs):

        print(f'\nEpoch {epoch}')

        epoch_loss = 0
        for _ in tqdm(range(epoch_size)):

            optimizer.zero_grad()

            batch = train.random_batch(batch_size)

            x, y = batch.xy_tensors()

            y_pred = model(x)

            loss = loss_func(y_pred, y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.data[0]

        checkpoint(model_path, 'model', model, epoch)
        print(epoch_loss / epoch_size)


@cli.command()
@click.argument('model_path', type=click.Path())
@click.argument('test_path', type=click.Path())
@click.option('--test_skim', type=int, default=10000)
@click.option('--map_source', default='cpu')
@click.option('--map_target', default='cpu')
def predict(model_path, test_path, test_skim, map_source, map_target):
    """Predict on dev / test.
    """
    model = torch.load(
        model_path,
        map_location={map_source: map_target},
    )

    test = Corpus(test_path, test_skim)

    kts = []
    correct = 0
    for ab in tqdm(test.abstracts):

        batch = Batch([ab])
        x, y = batch.xy_tensors()

        preds = model(x).sort()[1].data.tolist()

        kt, _ = stats.kendalltau(preds, range(len(preds)))
        kts.append(kt)

        if kt == 1:
            correct += 1

    print(sum(kts) / len(kts))
    print(correct / len(test.abstracts))


if __name__ == '__main__':
    cli()
