

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


class SentenceEncoder(nn.Module):

    def __init__(self, lstm_dim, lstm_num_layers):
        super().__init__()
        self.lstm = nn.LSTM(300, lstm_dim, lstm_num_layers, batch_first=True)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x)
        return hn[-1].squeeze()

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

    def __init__(self, input_dim, lin_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, lin_dim)
        self.lin2 = nn.Linear(lin_dim, lin_dim)
        self.out = nn.Linear(lin_dim, 1)

    def forward(self, x):
        y = F.relu(self.lin1(x))
        y = F.relu(self.lin2(y))
        y = self.out(y)
        return y.squeeze()


@click.group()
def cli():
    pass


@cli.command()
@click.argument('train_path', type=click.Path())
@click.argument('model_path', type=click.Path())
@click.option('--train_skim', type=int, default=10000)
@click.option('--lr', type=float, default=1e-4)
@click.option('--epochs', type=int, default=50)
@click.option('--epoch_size', type=int, default=100)
@click.option('--batch_size', type=int, default=10)
@click.option('--lstm_dim', type=int, default=512)
@click.option('--lstm_num_layers', type=int, default=4)
@click.option('--lin_dim', type=int, default=512)
def train(train_path, model_path, train_skim, lr, epochs, epoch_size,
    batch_size, lstm_dim, lstm_num_layers, lin_dim):
    """Train model.
    """
    train = Corpus(train_path, train_skim)

    encoder = SentenceEncoder(lstm_dim, lstm_num_layers)
    regressor = Regressor(lstm_dim, lin_dim)

    params = list(encoder.parameters()) + list(regressor.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    loss_func = nn.L1Loss()

    if CUDA:
        encoder = encoder.cuda()
        regressor = regressor.cuda()
        loss_func = loss_func.cuda()

    first_loss = None
    for epoch in range(epochs):

        print(f'\nEpoch {epoch}')

        epoch_loss = 0
        for _ in tqdm(range(epoch_size)):

            optimizer.zero_grad()

            batch = train.random_batch(batch_size)

            x, y = encoder.batch_xy(batch)
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

        checkpoint(model_path, 'encoder', encoder, epoch)
        checkpoint(model_path, 'regressor', regressor, epoch)


@cli.command()
@click.argument('encoder_path', type=click.Path())
@click.argument('regressor_path', type=click.Path())
@click.argument('test_path', type=click.Path())
@click.option('--test_skim', type=int, default=10000)
@click.option('--map_source', default='cuda:0')
@click.option('--map_target', default='cpu')
def predict(encoder_path, regressor_path, test_path, test_skim,
    map_source, map_target):
    """Predict on dev / test.
    """
    encoder = torch.load(
        encoder_path,
        map_location={map_source: map_target},
    )

    regressor = torch.load(
        regressor_path,
        map_location={map_source: map_target},
    )

    test = Corpus(test_path, test_skim)

    kts = []
    correct = 0
    for ab in tqdm(test.abstracts):

        sents = Variable(ab.tensor()).type(ftype)
        sents = encoder(sents)

        preds = regressor(sents).sort()[1].data.tolist()

        kt, _ = stats.kendalltau(preds, range(len(preds)))
        kts.append(kt)

        if kt == 1:
            correct += 1

    print(sum(kts) / len(kts))
    print(correct / len(test.abstracts))


if __name__ == '__main__':
    cli()
