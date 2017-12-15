

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
        _, (hn, _) = self.lstm(x)
        # Cat forward + backward hidden layers.
        out = hn.transpose(0, 1).contiguous().view(hn.data.shape[1], -1)
        return out[reorder]


class Classifier(nn.Module):

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


def train_batch(batch, s_encoder, r_encoder, classifier):
    """Train the batch.
    """
    x, reorder = batch.packed_sentence_tensor()

    # Encode sentences.
    sents = s_encoder(x, reorder)

    # Generate x / y pairs.
    examples = []
    for ab in batch.unpack_sentences(sents):

        split = random.randint(0, len(ab)-2)
        right = ab[split:]

        # Previous sentence (zeros if first).
        prev_sent = (
            Variable(torch.zeros(ab.data.shape[1])).type(ftype)
            if split == 0 else ab[split-1]
        )

        for i in range(len(right)):

            # Previous -> candidate.
            sent = torch.cat([prev_sent, right[i]])

            # Shuffle right.
            perm = torch.randperm(len(right)).type(itype)
            shuffled_right = right[perm]

            # 0 <-> 1
            y = i / (len(right)-1)

            examples.append((sent, shuffled_right, y))

    sents, rights, ys = zip(*examples)

    # Encode rights.
    rights, reorder = pad_and_pack(rights, 10)
    rights = r_encoder(rights, reorder)

    # <sent, right>
    x = zip(sents, rights)
    x = list(map(torch.cat, x))
    x = torch.stack(x)

    y = Variable(torch.FloatTensor(ys)).type(ftype)

    return classifier(x), y


def train(train_path, model_path, train_skim, lr, epochs, epoch_size,
    batch_size, lstm_dim, lin_dim):
    """Train model.
    """
    train = Corpus(train_path, train_skim)

    s_encoder = Encoder(300, lstm_dim)
    r_encoder = Encoder(2*lstm_dim, lstm_dim)
    classifier = Classifier(6*lstm_dim, lin_dim)

    params = (
        list(s_encoder.parameters()) +
        list(r_encoder.parameters()) +
        list(classifier.parameters())
    )

    optimizer = torch.optim.Adam(params, lr=lr)

    loss_func = nn.L1Loss()

    if CUDA:
        s_encoder = s_encoder.cuda()
        r_encoder = r_encoder.cuda()
        classifier = classifier.cuda()

    for epoch in range(epochs):

        print(f'\nEpoch {epoch}')

        epoch_loss, correct, total = 0, 0, 0

        for _ in tqdm(range(epoch_size)):

            optimizer.zero_grad()

            batch = train.random_batch(batch_size)

            y_pred, y = train_batch(batch, s_encoder, r_encoder, classifier)

            loss = loss_func(y_pred, y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.data[0]

            # EVAL

            start = 0
            for end in range(1, len(y)):

                if y[end].data[0] == 0:

                    pred = y_pred[start:end].data.tolist()

                    if np.argmin(pred) == 0:
                        correct += 1

                    total += 1

                    start = end

        print(epoch_loss / epoch_size)
        print(correct / total)
