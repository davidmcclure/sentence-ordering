

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

from sorder.utils import checkpoint, pad_and_pack
from sorder.vectors import LazyVectors
from sorder.cuda import ftype, itype


vectors = LazyVectors.read()


def read_abstracts(path):
    """Parse abstract JSON lines.
    """
    for path in glob(os.path.join(path, '*.json')):
        with open(path) as fh:
            for line in fh:
                yield Abstract.from_line(line)


@attr.s
class Sentence:

    position = attr.ib()
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
            Sentence(i, s['token'])
            for i, s in enumerate(json['sentences'])
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

    def shuffle(self):
        """Shuffle sentences in all abstracts.
        """
        for ab in self.abstracts:
            random.shuffle(ab.sentences)


class Corpus:

    def __init__(self, path, skim=None):
        """Load abstracts into memory.
        """
        reader = read_abstracts(path)

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
        self.out = nn.Linear(lin_dim, 2)

    def forward(self, x):
        y = F.relu(self.lin1(x))
        y = F.relu(self.lin2(y))
        y = F.relu(self.lin3(y))
        y = F.relu(self.lin4(y))
        y = F.relu(self.lin5(y))
        y = F.log_softmax(self.out(y))
        return y.squeeze()


def train_batch(batch, s_encoder, classifier):
    """Train the batch.
    """
    x, reorder = batch.packed_sentence_tensor()

    # Encode sentences.
    sents = s_encoder(x, reorder)

    # Generate x / y pairs.
    x, y = [], []
    for ab in batch.unpack_sentences(sents):
        for s1, s2 in pairwise(ab):

            x.append(torch.cat([s1, s2]))
            y.append(0)

            x.append(torch.cat([s2, s1]))
            y.append(1)

    x = torch.stack(x)
    y = Variable(torch.LongTensor(y)).type(itype)

    return classifier(x), y


def train(train_path, model_path, train_skim, lr, epochs, epoch_size,
    batch_size, lstm_dim, lin_dim):
    """Train model.
    """
    train = Corpus(train_path, train_skim)

    s_encoder = Encoder(300, lstm_dim)
    classifier = Classifier(4*lstm_dim, lin_dim)

    params = (
        list(s_encoder.parameters()) +
        list(classifier.parameters())
    )

    optimizer = torch.optim.Adam(params, lr=lr)

    loss_func = nn.NLLLoss()

    if torch.cuda.is_available():
        s_encoder = s_encoder.cuda()
        classifier = classifier.cuda()

    for epoch in range(epochs):

        print(f'\nEpoch {epoch}')

        epoch_loss, c, t = 0, 0, 0

        for _ in tqdm(range(epoch_size)):

            optimizer.zero_grad()

            batch = train.random_batch(batch_size)

            y_pred, y = train_batch(batch, s_encoder, classifier)

            loss = loss_func(y_pred, y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.data[0]

            # EVAL

            matches = (
                np.argmax(y_pred.data.tolist(), 1) ==
                np.array(y.data.tolist())
            )

            c += matches.sum()
            t += len(matches)

        checkpoint(model_path, 's_encoder', s_encoder, epoch)
        checkpoint(model_path, 'classifier', classifier, epoch)

        print(epoch_loss / epoch_size)
        print(c / t)


def beam_search(ab, classifier, beam_size=100):
    """Beam search.
    """
    beam = [((i,), 0) for i in range(len(ab))]

    for _ in range(len(ab)-1):

        new_beam = []

        # Get new path candidates.
        for path, score in beam:
            for i in range(len(ab)):
                if i not in path:
                    new_beam.append(((*path, i), score))

        # Get input tensors from final two sents.
        x = torch.stack([
            torch.cat([ab[p[-2]], ab[p[-1]]])
            for p, _ in new_beam
        ])

        x = x.type(ftype)

        y = classifier(x)

        # Update scores.
        new_beam = [
            (path, score + new_score.data[0])
            for (path, score), new_score in zip(new_beam, y)
        ]

        # Sort by score.
        new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)

        # Keep N highest scoring paths.
        beam = new_beam[:beam_size]

    return beam[0][0]


def predict(test_path, s_encoder_path, classifier_path, gp_path, test_skim,
    map_source, map_target):
    """Predict order.
    """
    test = Corpus(test_path, test_skim)

    s_encoder = torch.load(
        s_encoder_path,
        map_location={map_source: map_target},
    )

    classifier = torch.load(
        classifier_path,
        map_location={map_source: map_target},
    )

    gps = []
    for i, batch in enumerate(tqdm(test.batches(100))):

        batch.shuffle()

        # Encode sentence batch.
        sent_batch, reorder = batch.packed_sentence_tensor()
        sent_batch = s_encoder(sent_batch, reorder)

        # Re-group by abstract.
        unpacked = batch.unpack_sentences(sent_batch)

        for ab, sents in zip(batch.abstracts, unpacked):

            gold = [s.position for s in ab.sentences]

            pred = beam_search(sents, classifier)
            pred = np.argsort(pred).tolist()

            gps.append((gold, pred))

        # TODO|dev
        if i % 100 == 0:
            with open(gp_path, 'w') as fh:
                ujson.dump(gps, fh)

    with open(gp_path, 'w') as fh:
        ujson.dump(gps, fh)
