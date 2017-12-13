

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
from sorder.utils import checkpoint, pad_and_pack


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
        y = F.sigmoid(self.out(y))
        return y.squeeze()


def train_batch(batch, sent_encoder, left_encoder, right_encoder, classifier):
    """Train the batch.
    """
    x, reorder = batch.packed_sentence_tensor()

    # Encode sentences.
    sents = sent_encoder(x, reorder)

    # Generate positive / negative examples.
    examples = []
    for ab in batch.unpack_sentences(sents):
        for i in range(len(ab)-1):

            right = ab[i:]

            # If first step, empdy left context.
            left = (
                Variable(torch.zeros(1, ab.data.shape[1])).type(ftype)
                if i == 0 else ab[:i]
            )

            # Shuffle right.
            perm = torch.randperm(len(right)).type(itype)
            shuffled_right = right[perm]

            first = right[0]
            other = random.choice(right[1:])

            # First / not-first.
            examples.append((left, shuffled_right, first, 1))
            examples.append((left, shuffled_right, other, 0))

    lefts, rights, candidates, ys = zip(*examples)

    # Encode lefts.
    lefts, reorder = pad_and_pack(lefts, 10)
    lefts = left_encoder(lefts, reorder)

    # Encode rights.
    rights, reorder = pad_and_pack(rights, 10)
    rights = right_encoder(rights, reorder)

    # Cat (left, right, candidate).
    x = torch.stack([
        torch.cat([left, right, candidate])
        for left, right, candidate in zip(lefts, rights, candidates)
    ])

    y = Variable(torch.FloatTensor(ys)).type(ftype)

    return y, classifier(x)


def train(train_path, model_path, train_skim, lr, epochs, epoch_size,
    batch_size, lstm_dim, lin_dim):
    """Train model.
    """
    train = Corpus(train_path, train_skim)

    sent_encoder = Encoder(300, lstm_dim)
    left_encoder = Encoder(2*lstm_dim, lstm_dim)
    right_encoder = Encoder(2*lstm_dim, lstm_dim)
    classifier = Classifier(6*lstm_dim, lin_dim)

    params = (
        list(sent_encoder.parameters()) +
        list(left_encoder.parameters()) +
        list(right_encoder.parameters()) +
        list(classifier.parameters())
    )

    optimizer = torch.optim.Adam(params, lr=lr)

    loss_func = nn.BCELoss()

    if CUDA:
        sent_encoder = sent_encoder.cuda()
        left_encoder = left_encoder.cuda()
        right_encoder = right_encoder.cuda()
        classifier = classifier.cuda()

    for epoch in range(epochs):

        print(f'\nEpoch {epoch}')

        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        for _ in tqdm(range(epoch_size)):

            optimizer.zero_grad()

            batch = train.random_batch(batch_size)

            y, y_pred = train_batch(batch, sent_encoder, left_encoder,
                    right_encoder, classifier)

            loss = loss_func(y_pred, y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.data[0]
            epoch_correct += (y_pred.round() == y).sum().data[0]
            epoch_total += len(y)

        checkpoint(model_path, 'sent_encoder', sent_encoder, epoch)
        checkpoint(model_path, 'left_encoder', left_encoder, epoch)
        checkpoint(model_path, 'right_encoder', right_encoder, epoch)
        checkpoint(model_path, 'classifier', classifier, epoch)

        print(epoch_loss / epoch_size)
        print(epoch_correct / epoch_total)


def greedy_order(sents, left_encoder, right_encoder, classifier):
    """Order greedy.
    """
    order = []

    while len(order) < len(sents):

        # Left sentences.
        if len(order) == 0:
            left = Variable(torch.zeros(1, sents.data.shape[1])).type(ftype)

        else:
            left = sents[torch.LongTensor(order).type(itype)]

        # Right sentences.
        right_idx = [i for i in range(len(sents)) if i not in order]
        candidates = sents[torch.LongTensor(right_idx).type(itype)]

        # Encode left.
        left, reorder = pad_and_pack([left], 10)
        left = left_encoder(left, reorder)

        # Encode right.
        right, reorder = pad_and_pack([candidates], 10)
        right = right_encoder(right, reorder)

        # Cat (left, right, candidate).
        x = torch.stack([
            torch.cat([left[0], right[0], candidate])
            for candidate in candidates
        ])

        pred = right_idx.pop(np.argmax(classifier(x).data.tolist()))
        order.append(pred)

    return order


def predict(test_path, sent_encoder_path, left_encoder_path,
    right_encoder_path, classifier_path, test_skim, map_source, map_target):
    """Predict order.
    """
    test = Corpus(test_path, test_skim)

    dmap = {map_source: map_target}

    sent_encoder = torch.load(sent_encoder_path, dmap)
    left_encoder = torch.load(left_encoder_path, dmap)
    right_encoder = torch.load(right_encoder_path, dmap)
    classifier = torch.load(classifier_path, dmap)

    kts = []
    for batch in tqdm(test.batches(10)):

        batch.shuffle()

        # Encode sentences.
        x, reorder = batch.packed_sentence_tensor()
        encoded = sent_encoder(x, reorder)

        unpacked = batch.unpack_sentences(encoded)

        for ab, sents in zip(batch.abstracts, unpacked):

            pred = greedy_order(sents, left_encoder, right_encoder, classifier)
            gold = np.argsort([s.position for s in ab.sentences])

            kt, _ = stats.kendalltau(gold, pred)
            kts.append(kt)

    print(sum(kts) / len(kts))
    print(kts.count(1) / len(kts))
