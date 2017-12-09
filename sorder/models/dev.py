

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
from boltons.iterutils import pairwise
from scipy import stats

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn import functional as F

from sorder.cuda import CUDA, ftype, itype
from sorder.vectors import LazyVectors
from sorder.utils import checkpoint, pack


vectors = LazyVectors.read()


def read_abstracts(path, maxlen):
    """Parse abstract JSON lines.
    """
    for path in glob(os.path.join(path, '*.json')):
        with open(path) as fh:
            for line in fh:

                # Parse JSON.
                abstract = Abstract.from_line(line)

                # Filter by length.
                if len(abstract.sentences) < maxlen:
                    yield abstract


@attr.s
class Sentence:

    tokens = attr.ib()

    def tensor(self, dim=300, pad=500):
        """Stack word vectors, padding zeros on left.
        """
        # Map words to embeddings.
        x = [
            vectors[t] if t in vectors else np.zeros(dim)
            for t in self.tokens
        ]

        # Pad zeros.
        x += [np.zeros(dim)] * pad
        x = x[:pad]
        x = np.array(x)
        x = torch.from_numpy(x)
        x = x.float()

        return x, len(self.tokens)


@attr.s
class Abstract:

    sentences = attr.ib()

    @classmethod
    def from_line(cls, line):
        """Parse JSON, take tokens.
        """
        json = ujson.loads(line.strip())

        return cls([
            Sentence(s['token'])
            for s in json['sentences']
        ])

    def context_tensor(self):
        """Make a tensor with all words.
        """
        sents = sorted(self.sentences, key=lambda x: random.random())

        tokens = [t for s in sents for t in s.tokens]

        return Sentence(tokens).tensor()

    def tensors_iter(self):
        """Provide context tensor, then sentences.
        """
        yield self.context_tensor()

        for sent in self.sentences:
            yield sent.tensor()

    def clip_random(self):
        """Take the last N sentences, where N is random.
        """
        if len(self.sentences) > 2:
            start = random.randint(0, len(self.sentences)-2)
            self.sentences = self.sentences[start:]


@attr.s
class Batch:

    abstracts = attr.ib()

    def tensors_iter(self):
        """Provide tensors for all sentences.
        """
        for ab in self.abstracts:
            ab.clip_random()
            yield from ab.tensors_iter()

    def packed_sentence_tensor(self):
        """Stack sentence tensors for all abstracts.
        """
        tensors, sizes = zip(*self.tensors_iter())

        return pack(torch.stack(tensors), sizes, ftype)


class Corpus:

    def __init__(self, path, skim=None, maxlen=10):
        """Load abstracts into memory.
        """
        reader = read_abstracts(path, maxlen)

        if skim:
            reader = islice(reader, skim)

        self.abstracts = list(tqdm(reader, total=skim))

    def random_batch(self, size):
        """Query random batch.
        """
        return Batch(random.sample(self.abstracts, size))


class Encoder(nn.Module):

    def __init__(self, lstm_dim):
        super().__init__()
        self.lstm = nn.LSTM(300, lstm_dim, batch_first=True,
            bidirectional=True)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x)
        return hn[-1].squeeze()

    def encode_batch(self, batch):
        """Encode sentences in a batch, then regroup by abstract.
        """
        x, size_sort = batch.packed_sentence_tensor()

        reorder = torch.LongTensor(np.argsort(size_sort)).type(itype)

        y = self(x)[reorder]

        start = 0
        for ab in batch.abstracts:
            end = start + len(ab.sentences) + 1
            yield y[start:end]
            start = end

    def batch_xy(self, batch):
        """Encode sentences, generate positive and negative pairs.
        """
        x = []
        y = []
        for ab in self.encode_batch(batch):

            if len(ab) < 3:
                continue

            context = ab[0]
            s1 = ab[1]
            sn = random.choice(ab[2:])

            # First.
            x.append(torch.cat([context, s1]))
            y.append(1)

            # Not first.
            x.append(torch.cat([context, sn]))
            y.append(0)

        x = torch.stack(x)
        y = Variable(torch.FloatTensor(y)).type(ftype)

        return x, y


class Regressor(nn.Module):

    def __init__(self, input_dim, lin_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, lin_dim)
        self.lin2 = nn.Linear(lin_dim, lin_dim)
        self.out = nn.Linear(lin_dim, 1)

    def forward(self, x):
        y = F.relu(self.lin1(x))
        y = F.relu(self.lin2(y))
        y = F.sigmoid(self.out(y))
        return y.squeeze()


@click.group()
def cli():
    pass


@cli.command()
@click.argument('train_path', type=click.Path())
@click.argument('model_path', type=click.Path())
@click.option('--train_skim', type=int, default=10000)
@click.option('--lr', type=float, default=1e-3)
@click.option('--epochs', type=int, default=1000)
@click.option('--epoch_size', type=int, default=100)
@click.option('--batch_size', type=int, default=10)
@click.option('--lstm_dim', type=int, default=500)
@click.option('--lin_dim', type=int, default=300)
def train(train_path, model_path, train_skim, lr, epochs,
    epoch_size, batch_size, lstm_dim, lin_dim):
    """Train model.
    """
    train = Corpus(train_path, train_skim)

    m1 = Encoder(lstm_dim)
    m2 = Regressor(lstm_dim*2, lin_dim)

    params = list(m1.parameters()) + list(m2.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    loss_func = nn.BCELoss()

    if CUDA:
        m1 = m1.cuda()
        m2 = m2.cuda()

    for epoch in range(epochs):

        print(f'\nEpoch {epoch}')

        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        for _ in tqdm(range(epoch_size)):

            optimizer.zero_grad()

            batch = train.random_batch(batch_size)

            x, y = m1.batch_xy(batch)

            y_pred = m2(x)

            loss = loss_func(y_pred, y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.data[0]
            epoch_correct += (y_pred.round() == y).sum().data[0]
            epoch_total += len(y)

        print(epoch_loss / epoch_size)
        print(epoch_correct / epoch_total)


if __name__ == '__main__':
    cli()
