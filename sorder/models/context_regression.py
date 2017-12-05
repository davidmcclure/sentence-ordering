

import click
import torch
import os

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm
from scipy import stats

from sorder.cuda import CUDA, ftype, itype
from sorder.abstracts import Corpus
from sorder.utils import checkpoint


def pad_2d_left(x, maxlen):
    """Pad left-size zeros on a 2d variable.
    """
    pad_dim = x.data.shape[1]
    pad_len = maxlen-len(x)

    zeros = torch.zeros(pad_len, pad_dim)
    zeros = Variable(zeros).type(ftype)

    return torch.cat([zeros, x])


class SentenceEncoder(nn.Module):

    def __init__(self, lstm_dim=1000):
        super().__init__()
        self.lstm = nn.LSTM(300, lstm_dim, batch_first=True)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x)
        return hn.squeeze()

    def encode_batch(self, batch):
        """List of Abstract -> list of (sent, dim) tensors.

        Args:
            batch (list of Abstract)

        Yields: Unpacked (sent, dim) tensors for each abstract.
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


class ShuffledContextEncoder(nn.Module):

    def __init__(self, input_dim=1000, lstm_dim=2000):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_dim, batch_first=True)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x)
        return hn.squeeze()

    def encode_batch(self, batch, maxlen=10):
        """List of (sent, dim) -> (abstract, dim).

        Args:
            batch (list of (sent x dim))

        Yields: (abstract, dim)
        """
        x = []
        for sents in batch:

            # Shuffle sentences.
            shuffle = torch.randperm(len(sents)).type(itype)
            sents = sents[shuffle]

            # Pad length.
            sents = pad_2d_left(sents, maxlen)

            x.append(sents)

        return self(torch.stack(x))

    def batch_xy(self, batch):
        """List of (sent, dim) -> x/y regression pairs.
        """
        contexts = self.encode_batch(batch)

        x = []
        y = []
        for sents, ctx in zip(batch, contexts):
            for i in range(len(sents)):
                x.append(torch.cat([ctx, sents[i]]))
                y.append(i / (len(sents)-1))

        x = torch.stack(x)
        y = Variable(torch.FloatTensor(y)).type(ftype)

        return x, y


class Regressor(nn.Module):

    def __init__(self, input_dim=3000):
        super().__init__()
        self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        y = self.out(x)
        return y.squeeze()


@click.group()
def cli():
    pass


@cli.command()
@click.argument('train_path', type=click.Path())
@click.option('--train_skim', type=int, default=10000)
@click.option('--lr', type=float, default=1e-4)
@click.option('--epochs', type=int, default=100)
@click.option('--epoch_size', type=int, default=100)
@click.option('--batch_size', type=int, default=10)
def train(train_path, train_skim, lr, epochs, epoch_size, batch_size):
    """Train model.
    """
    train = Corpus(train_path, train_skim)

    s_encoder = SentenceEncoder()
    c_encoder = ShuffledContextEncoder()
    regressor = Regressor()

    params = (
        list(s_encoder.parameters()) +
        list(c_encoder.parameters()) +
        list(regressor.parameters())
    )

    optimizer = torch.optim.Adam(params, lr=lr)

    loss_func = nn.L1Loss()

    if CUDA:
        s_encoder = s_encoder.cuda()
        c_encoder = c_encoder.cuda()
        regressor = regressor.cuda()

    first_loss = None
    for epoch in range(epochs):

        print(f'\nEpoch {epoch}')

        epoch_loss = 0
        for _ in tqdm(range(epoch_size)):

            optimizer.zero_grad()

            batch = train.random_batch(batch_size)
            batch = list(s_encoder.encode_batch(batch))

            x, y = c_encoder.batch_xy(batch)
            y_pred = regressor(x)

            loss = loss_func(y_pred, y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.data[0]

        epoch_loss /= epoch_size

        if not first_loss:
            first_loss = epoch_loss

        print(epoch_loss)
        print(epoch_loss / first_loss)


if __name__ == '__main__':
    cli()
