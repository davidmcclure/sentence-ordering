

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
from boltons.iterutils import pairwise, chunked_iter
from scipy import stats

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch.nn import functional as F

from sent_order.vectors import LazyVectors
from sent_order.cuda import ftype, itype
from sent_order.utils import checkpoint, pad_and_pack, pad_and_stack
from sent_order.perms import sample_uniform_perms


vectors = LazyVectors.read()


@attr.s
class Sentence:

    tokens = attr.ib()

    def tensor(self):
        """Stack word vectors.
        """
        x = [
            vectors[t] if t in vectors else np.zeros(vectors.dim)
            for t in self.tokens
        ]

        x = np.array(x)
        x = torch.from_numpy(x)
        x = x.float()

        return x


@attr.s
class Paragraph:

    sentences = attr.ib()

    @classmethod
    def read_arxiv(cls, path, size=None):
        """Wrap parsed arXiv abstracts as paragraphs.
        """
        for path in glob(os.path.join(path, '*.json')):
            for line in open(path):

                graf = cls.from_arxiv_json(line)

                if not size or len(graf.sentences) == size:
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

    def sentence_variables(self):
        """Gather sentence tensors.
        """
        for s in self.sentences:
            yield Variable(s.tensor()).type(ftype)


@attr.s
class Batch:

    grafs = attr.ib()

    def sentence_variables(self):
        """Pack sentence tensors.
        """
        for g in self.grafs:
            yield from g.sentence_variables()

    def unpack_sentences(self, encoded):
        """Unpack encoded sentences.
        """
        start = 0
        for ab in self.grafs:
            end = start + len(ab.sentences)
            yield encoded[start:end]
            start = end

    def shuffle(self):
        """Shuffle sentences in all grafs.
        """
        for ab in self.grafs:
            random.shuffle(ab.sentences)


class Corpus:

    def __init__(self, path, count=None, size=None):
        """Load grafs into memory.
        """
        reader = Paragraph.read_arxiv(path, size)

        if count:
            reader = islice(reader, count)

        self.grafs = list(tqdm(reader, total=count))

    def random_batch(self, size):
        """Query random batch.
        """
        return Batch(random.sample(self.grafs, size))

    def batches(self, size):
        """Iterate all batches.
        """
        for grafs in chunked_iter(self.grafs, size):
            yield Batch(grafs)


class SentenceEncoder(nn.Module):

    def __init__(self, embed_dim, lstm_dim):
        """Initialize the LSTM.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            embed_dim,
            lstm_dim,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x, pad_size=30):
        """Encode word embeddings as single sentence vector.

        Args:
            x (list of Variable): Encoded sentences for each graf.
        """
        # Pad, pack, encode.
        x, reorder = pad_and_pack(x, pad_size)
        _, (hn, _) = self.lstm(x)

        # Cat forward + backward hidden layers.
        out = hn.transpose(0, 1).contiguous().view(hn.data.shape[1], -1)

        return out[reorder]


class Regressor(nn.Module):

    def __init__(self, lstm_dim, lin_dim):
        """Initialize LSTM, linear layers.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            lstm_dim,
            lstm_dim,
            bidirectional=True,
            batch_first=True,
        )

        self.lin1 = nn.Linear(7000, lin_dim)
        self.lin2 = nn.Linear(lin_dim, lin_dim)
        self.lin3 = nn.Linear(lin_dim, lin_dim)
        self.lin4 = nn.Linear(lin_dim, lin_dim)
        self.lin5 = nn.Linear(lin_dim, lin_dim)
        self.out = nn.Linear(lin_dim, 1)

    def forward(self, x, pad_size=5):
        """Encode sentences as a single paragraph vector, predict KT.
        """
        # Pad, pack, encode.
        packed, reorder = pad_and_pack(x, pad_size)
        _, (hn, _) = self.lstm(packed)

        # Cat forward + backward hidden layers.
        y = hn.transpose(0, 1).contiguous().view(hn.data.shape[1], -1)
        y = y[reorder]

        sents = torch.stack([t.view(5000) for t in x])

        y = torch.cat([y, sents], 1)

        y = F.relu(self.lin1(y))
        y = F.relu(self.lin2(y))
        y = F.relu(self.lin3(y))
        y = F.relu(self.lin4(y))
        y = F.relu(self.lin5(y))
        y = self.out(y)

        return y.squeeze()


def train_batch(batch, sent_encoder, regressor):
    """Train the batch.
    """
    # Encode sentences.
    sents = batch.sentence_variables()
    sents = sent_encoder(sents)

    # Generate x / y pairs.
    x, y = [], []
    for ab in batch.unpack_sentences(sents):

        perms, kts = sample_uniform_perms(len(ab))

        # Squeeze middle KTS towards 0.
        kts = kts**3

        for perm, kt in zip(perms, kts):

            perm = torch.LongTensor(perm).type(itype)

            x.append(ab[perm])
            y.append(kt)

    y = Variable(torch.FloatTensor(y)).type(ftype)

    return regressor(x), y


def train(train_path, model_path, train_skim, lr, epochs, epoch_size,
    batch_size, lstm_dim, lin_dim):
    """Train model.
    """
    train = Corpus(train_path, train_skim, 5)

    sent_encoder = SentenceEncoder(300, lstm_dim)
    regressor = Regressor(2*lstm_dim, lin_dim)

    params = (
        list(sent_encoder.parameters()) +
        list(regressor.parameters())
    )

    optimizer = torch.optim.Adam(params, lr=lr)

    loss_func = nn.MSELoss()

    if torch.cuda.is_available():
        sent_encoder = sent_encoder.cuda()
        regressor = regressor.cuda()

    for epoch in range(epochs):

        print(f'\nEpoch {epoch}')

        epoch_loss = 0
        for _ in tqdm(range(epoch_size)):

            optimizer.zero_grad()

            batch = train.random_batch(batch_size)

            y_pred, y = train_batch(batch, sent_encoder, regressor)

            loss = loss_func(y_pred, y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.data[0]

        checkpoint(model_path, 'sent_encoder', sent_encoder, epoch)
        checkpoint(model_path, 'regressor', regressor, epoch)

        print(epoch_loss / epoch_size)
