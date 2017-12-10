

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

    order = attr.ib()
    tokens = attr.ib()

    def tensor(self, dim=300, pad=50):
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
            Sentence(i, s['token'])
            for i, s in enumerate(json['sentences'])
        ])


@attr.s
class Batch:

    abstracts = attr.ib()

    def sentence_tensor_iter(self):
        """Generate (tensor, size) pairs for each sentence.
        """
        for ab in self.abstracts:
            for sent in ab.sentences:
                yield sent.tensor()

    def packed_sentence_tensor(self):
        """Stack sentence tensors for all abstracts.
        """
        tensors, sizes = zip(*self.sentence_tensor_iter())

        return pack(torch.stack(tensors), sizes, ftype)

    def shuffle(self):
        """Shuffle sentences in all abstracts.
        """
        for ab in self.abstracts:
            random.shuffle(ab.sentences)


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

    def batches(self, size):
        """Iterate all batches.
        """
        for abstracts in chunked_iter(self.abstracts, size):
            yield Batch(abstracts)


class SentenceEncoder(nn.Module):

    def __init__(self, lstm_dim):
        super().__init__()
        self.lstm = nn.LSTM(300, lstm_dim, batch_first=True,
            bidirectional=True)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x)
        return cn[-1].squeeze()

    def encode_batch(self, batch):
        """Encode sentences in a batch, then regroup by abstract.
        """
        x, size_sort = batch.packed_sentence_tensor()

        reorder = torch.LongTensor(np.argsort(size_sort)).type(itype)

        y = self(x)[reorder]

        start = 0
        for ab in batch.abstracts:
            end = start + len(ab.sentences)
            yield y[start:end]
            start = end

    def batch_xy(self, batch):
        """Encode sentences, generate positive and negative pairs.
        """
        x = []
        y = []
        for ab in self.encode_batch(batch):
            for i in range(0, len(ab)-2):

                s1, s2 = ab[i], ab[i+1]

                # Pick random sentence that isn't next.
                s3r = list(range(len(ab)))
                s3r.remove(i)
                s3r.remove(i+1)

                s3 = ab[random.choice(s3r)]

                context = ab.mean(0)

                # Next.
                x.append(torch.cat([context, s1, s2]))
                y.append(1)

                # Not next.
                x.append(torch.cat([context, s1, s3]))
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


class BeamSearch:

    def __init__(self, sents, model, beam_size=1000):
        """Initialize containers.

        Args:
            sents (FloatTensor): Tensor of encoded sentences.
        """
        self.sents = sents
        self.model = model

        self.beam_size = beam_size

        self.beam = [((i,), 0) for i in range(len(self.sents))]

    def transition_score(self, i1, i2):
        """Score a sentence transition.
        """
        x = torch.cat([
            self.sents.mean(0),
            self.sents[i1],
            self.sents[i2],
        ])

        return self.model(x).data[0]

    def step(self):
        """Expand, score, prune.
        """
        new_beam = []

        for path, score in self.beam:
            for i in range(len(self.sents)):

                if i in path:
                    continue

                new_score = score + self.transition_score(path[-1], i)

                new_path = (*path, i)

                new_beam.append((new_path, new_score))

        new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)

        self.beam = new_beam[:self.beam_size]

    def search(self):
        """Probe, return top sequence.
        """
        for _ in range(len(self.sents)-1):
            self.step()

        return self.beam[0][0]


def train(train_path, model_path, train_skim, lr, epochs, epoch_size,
    batch_size, lstm_dim, lin_dim):
    """Train model.
    """
    train = Corpus(train_path, train_skim)

    m1 = SentenceEncoder(lstm_dim)
    m2 = Regressor(lstm_dim*3, lin_dim)

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

        checkpoint(model_path, 'm1', m1, epoch)
        checkpoint(model_path, 'm2', m2, epoch)

        print(epoch_loss / epoch_size)
        print(epoch_correct / epoch_total)


def predict(test_path, m1_path, m2_path, test_skim, map_source, map_target):
    """Predict order.
    """
    test = Corpus(test_path, test_skim)

    m1 = torch.load(m1_path, map_location={map_source: map_target})
    m2 = torch.load(m2_path, map_location={map_source: map_target})

    kts = []
    correct = 0
    for batch in test.batches(10):

        batch.shuffle()

        encoded = m1.encode_batch(batch)

        for ab, sents in tqdm(zip(batch.abstracts, encoded)):

            gold = np.argsort([s.order for s in ab.sentences])
            pred = BeamSearch(sents, m2).search()

            kt, _ = stats.kendalltau(gold, pred)
            kts.append(kt)

            if kt == 1:
                correct += 1

    print(sum(kts) / len(kts))
    print(correct / len(kts))
