

import numpy as np

import os
import click
import torch
import attr
import random
import ujson
import math

from tqdm import tqdm
from itertools import islice
from glob import glob
from boltons.iterutils import pairwise, chunked_iter
from scipy import stats

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch.nn import functional as F

from sorder.utils import checkpoint, pad_and_pack, random_subseq
from sorder.vectors import LazyVectors
from sorder.cuda import CUDA, ftype, itype


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

    def tensor(self, dim=300):
        """Stack word vectors.
        """
        x = [
            vectors[t] if t in vectors else np.zeros(dim)
            for t in self.tokens
        ]

        x = np.array(x)
        x = torch.from_numpy(x)
        x = x.float()

        return x


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


@attr.s
class Batch:

    abstracts = attr.ib()

    def packed_sentence_tensor(self, size=50):
        """Pack sentence tensors.
        """
        sents = [
            Variable(s.tensor()).type(ftype)
            for a in self.abstracts
            for s in a.sentences
        ]

        return pad_and_pack(sents, size)

    def unpack_sentences(self, encoded):
        """Unpack encoded sentences.
        """
        start = 0
        for ab in self.abstracts:
            end = start + len(ab.sentences)
            yield encoded[start:end]
            start = end


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

    def __init__(self, input_dim, lstm_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_dim, batch_first=True,
            bidirectional=True)

    def forward(self, x, reorder):
        _, (hn, cn) = self.lstm(x)
        # Cat forward + backward hidden layers.
        out = hn.transpose(0, 1).contiguous().view(hn.data.shape[1], -1)
        return out[reorder]


class Regressor(nn.Module):

    def __init__(self, input_dim, lin_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, lin_dim)
        self.lin2 = nn.Linear(lin_dim, lin_dim)
        self.lin3 = nn.Linear(lin_dim, lin_dim)
        self.lin4 = nn.Linear(lin_dim, lin_dim)
        self.lin5 = nn.Linear(lin_dim, lin_dim)
        self.out = nn.Linear(lin_dim, 1)

    def forward(self, x):
        y = F.relu(self.lin1(x))
        y = F.relu(self.lin2(y))
        y = F.relu(self.lin3(y))
        y = F.relu(self.lin4(y))
        y = F.relu(self.lin5(y))
        y = self.out(y)
        return y.squeeze()


def train_batch(batch, sent_encoder, graf_encoder, regressor):
    """Train the batch.
    """
    x, reorder = batch.packed_sentence_tensor()

    # Encode sentences.
    sents = sent_encoder(x, reorder)

    # Generate x / y pairs.
    examples = []
    for ab in batch.unpack_sentences(sents):

        # Random middle window.
        size = random.randint(2, len(ab))
        i1 = random.randint(0, len(ab)-size)
        i2 = i1 + size
        window = ab[i1:i2]

        # Left and right sentences.
        zeros = Variable(ab[0].data.clone().zero_())
        lsent = ab[i1-1] if i1 else zeros
        rsent = ab[i2] if i2 < len(ab)-1 else zeros

        for i in range(len(window)):

            # Shuffle window.
            perm = torch.randperm(len(window)).type(itype)
            shuffled_window = window[perm]

            # Left, right, size, sentence.
            size = Variable(torch.FloatTensor([len(window)])).type(ftype)
            sent = torch.cat([lsent, rsent, size, window[i]])

            # 0 <-> 1
            y = i / (len(window)-1)

            # Graf, sentence, length, position.
            examples.append((shuffled_window, sent, y))

    windows, sents, ys = zip(*examples)

    # Encode contexts.
    windows, reorder = pad_and_pack(windows, 10)
    windows = graf_encoder(windows, reorder)

    # Stack [window, sentence].
    x = torch.stack([
        torch.cat([window, sent])
        for window, sent in zip(windows, sents)
    ])

    y = Variable(torch.FloatTensor(ys)).type(ftype)

    return y, regressor(x)


def train(train_path, model_path, train_skim, lr, epochs, epoch_size,
    batch_size, lstm_dim, lin_dim):
    """Train model.
    """
    train = Corpus(train_path, train_skim)

    sent_encoder = Encoder(300, lstm_dim)
    graf_encoder = Encoder(2*lstm_dim, lstm_dim)
    regressor = Regressor(8*lstm_dim+1, lin_dim)

    params = (
        list(sent_encoder.parameters()) +
        list(graf_encoder.parameters()) +
        list(regressor.parameters())
    )

    optimizer = torch.optim.Adam(params, lr=lr)

    loss_func = nn.L1Loss()

    if CUDA:
        sent_encoder = sent_encoder.cuda()
        graf_encoder = graf_encoder.cuda()
        regressor = regressor.cuda()

    for epoch in range(epochs):

        print(f'\nEpoch {epoch}')

        epoch_loss, correct, total = 0, 0, 0

        for _ in tqdm(range(epoch_size)):

            optimizer.zero_grad()

            batch = train.random_batch(batch_size)

            y, y_pred = train_batch(batch, sent_encoder, \
                    graf_encoder, regressor)

            loss = loss_func(y_pred, y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.data[0]

            # Check selection accuracy.
            start = 0
            for end in range(1, len(y)):

                if y[end].data[0] == 0:

                    pred = y_pred[start:end].data

                    max_idx = np.argmax(pred)
                    min_idx = np.argmin(pred)

                    max_d = 1-pred[max_idx]
                    min_d = pred[min_idx]

                    # If we'd make a correct selection.
                    if (
                        max_d < min_d and max_idx == len(pred)-1 or
                        min_d < max_d and min_idx == 0
                    ):
                        correct += 1

                    total += 1

                    start = end

        print(epoch_loss / epoch_size)
        print(correct / total)
