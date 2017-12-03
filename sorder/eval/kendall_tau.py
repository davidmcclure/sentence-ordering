

import numpy as np

import attr
import os
import click
import torch
import ujson

from gensim.models import KeyedVectors
from boltons.iterutils import pairwise, chunked_iter
from tqdm import tqdm
from glob import glob
from itertools import islice
from scipy import stats
from itertools import permutations

from sklearn.metrics import classification_report, accuracy_score

from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


ftype = torch.FloatTensor


def load_vectors(path):
    print('Loading vectors...')
    global vectors
    vectors = KeyedVectors.load(path)


def read_abstracts(path):
    for path in glob(os.path.join(path, '*.json')):
        with open(path) as fh:
            for line in fh:
                raw = ujson.loads(line.strip())
                yield Abstract.from_raw(raw)


class Corpus:

    def __init__(self, path, skim):
        reader = islice(read_abstracts(path), skim)
        self.abstracts = list(tqdm(reader, total=skim))

    def batches(self, size):
        for chunk in chunked_iter(tqdm(self.abstracts), size):
            yield AbstractBatch(chunk)


@attr.s
class Abstract:

    sentences = attr.ib()

    @classmethod
    def from_raw(cls, raw):
        return cls([Sentence(s['token']) for s in raw['sentences']])

    def tensor(self):
        tensors = [s.tensor() for s in self.sentences]
        return torch.stack(tensors)


@attr.s
class Sentence:

    tokens = attr.ib()

    def tensor(self, dim=300, pad=50):
        x = [vectors[t] for t in self.tokens if t in vectors]
        x += [np.zeros(dim)] * pad
        x = x[:pad]
        x = list(reversed(x))
        x = np.array(x)
        x = torch.from_numpy(x)
        x = x.float()
        return x


class SentenceEncoder(nn.Module):

    def __init__(self, lstm_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(300, lstm_dim)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x.transpose(0, 1))
        return hn


class Model(nn.Module):

    def __init__(self, input_dim=128, lstm_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_dim)
        self.out = nn.Linear(lstm_dim, 1)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x.transpose(0, 1))
        y = F.sigmoid(self.out(hn))
        return y.squeeze()


@click.command()
@click.argument('sent_encoder_path', type=click.Path(), default='data/sent-encoder.pt')
@click.argument('model_path', type=click.Path(), default='data/model.pt')
@click.argument('test_path', type=click.Path(), default='data/dev.json')
@click.argument('vectors_path', type=click.Path(), default='data/vectors/vectors.bin')
@click.option('--test_skim', type=int, default=1000)
def main(sent_encoder_path, model_path, test_path, vectors_path, test_skim):

    load_vectors(vectors_path)

    sent_encoder = torch.load(
        sent_encoder_path,
        map_location={'cuda:0': 'cpu'}
    )

    model = torch.load(
        model_path,
        map_location={'cuda:0': 'cpu'}
    )

    test = Corpus(test_path, test_skim)

    kts = []
    c, t = 0, 0
    for ab in tqdm(test.abstracts):

        if len(ab.sentences) > 5:
            continue

        sents = Variable(ab.tensor())
        sents = sent_encoder(sents).squeeze()

        pad_dim = sents.data.shape[1]
        pad_len = 10-len(sents)
        zeros = Variable(torch.zeros(pad_len, pad_dim)).type(ftype)

        perms = list(permutations(range(len(sents))))

        x = torch.stack([
            torch.cat([zeros, sents[torch.LongTensor(perm)]])
            for perm in perms
        ])

        preds = model(x)

        pred = perms[np.argmax(preds.data.tolist())]

        kt, _ = stats.kendalltau(range(len(ab.sentences)), pred)
        kts.append(kt)

        if kt == 1: c += 1
        t += 1

    print(sum(kts) / len(kts))
    print(c/t)


if __name__ == '__main__':
    main()
