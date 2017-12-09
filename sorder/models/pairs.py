

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

    def tensor(self, dim=300, pad=50):
        """Stack word vectors, padding zeros on left.
        """
        # Get word tensors and length.
        x = [vectors[t] for t in self.tokens if t in vectors]
        size = min(len(x), pad)

        # Pad zeros.
        x += [np.zeros(dim)] * pad
        x = x[:pad]
        x = np.array(x)
        x = torch.from_numpy(x)
        x = x.float()
        return x, size


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

    def sentence_tensor_iter(self):
        """Generate (tensor, size) pairs for each sentence.
        """
        for ab in self.abstracts:
            for sent in ab.sentences:

                tensor, size = sent.tensor()

                # Discard all-OOV sentences.
                if size:
                    yield tensor, size

    def packed_sentence_tensor(self):
        """Stack sentence tensors for all abstracts.
        """
        tensors, sizes = zip(*self.sentence_tensor_iter())

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


@click.group()
def cli():
    pass


@cli.command()
@click.argument('train_path', type=click.Path())
@click.option('--train_skim', type=int, default=10000)
@click.option('--epochs', type=int, default=1000)
@click.option('--epoch_size', type=int, default=100)
@click.option('--batch_size', type=int, default=10)
def train(train_path, train_skim, epochs, epoch_size, batch_size):
    """Train model.
    """
    train = Corpus(train_path, train_skim)

    for epoch in range(epochs):

        print(f'\nEpoch {epoch}')

        for _ in tqdm(range(epoch_size)):

            batch = train.random_batch(batch_size)
            print(batch)

            # encode sentences in batch
            # generate pos/neg xy pairs
            # predict 0/1


if __name__ == '__main__':
    cli()
