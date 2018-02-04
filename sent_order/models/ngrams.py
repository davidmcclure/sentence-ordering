

import numpy as np

import attr
import torch
import os
import random
import ujson

from torchtext.vocab import Vectors

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from cached_property import cached_property
from glob import glob
from itertools import islice
from scipy.stats import kendalltau
from tqdm import tqdm

from ..cuda import ftype, itype


def pad_and_stack(xs, pad_size):
    """Pad and stack a list of variable-length seqs.

    Args:
        xs (list[Variable])
        pad_size (int)

    Returns: stacked xs, sizes
    """
    padded, sizes = [], []
    for x in xs:

        px = F.pad(x, (0, pad_size-len(x)))
        padded.append(px)

        size = min(pad_size, len(x))
        sizes.append(size)

    return torch.stack(padded), sizes


def pack(x, sizes, batch_first=True):
    """Pack padded variables, provide reorder indexes.

    Args:
        batch (Variable)
        sizes (list[int])

    Returns: packed sequence, reorder indexes
    """
    size_sort = np.argsort(sizes)[::-1].tolist()

    # Sort x and sizes by size descending.
    x = x[torch.LongTensor(size_sort).type(itype)]
    sizes = np.array(sizes)[size_sort].tolist()

    # Indexes to restore original order.
    reorder = torch.LongTensor(np.argsort(size_sort)).type(itype)

    # Pack the sequence.
    packed = pack_padded_sequence(x, sizes, batch_first)

    return packed, reorder


class LazyVectors:

    unk_idx = 1

    def __init__(self, name='glove.840B.300d.txt'):
        self.name = name

    @cached_property
    def loader(self):
        return Vectors(self.name)

    def set_vocab(self, vocab):
        """Set corpus vocab.
        """
        # Intersect with model vocab.
        self.vocab = [v for v in vocab if v in self.loader.stoi]

        # Map string -> intersected index.
        self._stoi = {s: i for i, s in enumerate(self.vocab)}

    @cached_property
    def weights(self):
        """Build weights tensor for embedding layer.
        """
        # Select vectors for vocab words.
        weights = torch.stack([
            self.loader.vectors[self.loader.stoi[s]]
            for s in self.vocab
        ])

        # Padding + UNK zeros rows.
        return torch.cat([
            torch.zeros((2, self.loader.dim)),
            weights,
        ])

    def stoi(self, s):
        """Map string -> embedding index.
        """
        idx = self._stoi.get(s)
        return idx + 2 if idx else self.unk_idx


VECTORS = LazyVectors()


@attr.s
class Sentence:

    tokens = attr.ib()

    def index_var(self):
        idx = [VECTORS.stoi(s) for s in self.tokens]
        idx = torch.LongTensor(idx)
        idx = Variable(idx).type(itype)
        return idx


@attr.s
class Paragraph:

    sents = attr.ib()

    def __len__(self):
        return len(self.sents)

    @classmethod
    def read_arxiv(cls, path, scount=None):
        """Wrap parsed arXiv abstracts as paragraphs.
        """
        for path in glob(os.path.join(path, '*.json')):
            for line in open(path):

                graf = cls.from_arxiv_json(line)

                # Filter by sentence count.
                if not scount or len(graf) == scount:
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


@attr.s
class Batch:

    grafs = attr.ib()

    def index_vars(self):
        return [s.index_var() for g in self.grafs for s in g.sents]


class Corpus:

    def __init__(self, path, skim=None, scount=None):
        """Load grafs into memory.
        """
        reader = Paragraph.read_arxiv(path, scount)

        if skim:
            reader = islice(reader, skim)

        self.grafs = list(tqdm(reader, total=skim))

    def random_batch(self, size):
        """Query random batch.
        """
        return Batch(random.sample(self.grafs, size))

    def vocab(self):
        """Build vocab list.
        """
        vocab = set()

        for graf in self.grafs:
            for sent in graf.sents:
                vocab.update(sent.tokens)

        return list(vocab)


class Encoder(nn.Module):

    def __init__(self):
        """Initialize embeddings + LSTM.
        """
        super().__init__()

        self.embeddings = nn.Embedding(
            VECTORS.weights.shape[0],
            VECTORS.weights.shape[1],
        )

        self.embeddings.weight.data.copy_(VECTORS.weights)

        self.lstm = nn.LSTM(300, 500, batch_first=True)

    def forward(self, x, pad_size=50):
        """Encode sentences.
        """
        x, sizes = pad_and_stack(x, pad_size)

        x = self.embeddings(x)

        x, reorder = pack(x, sizes)

        _, (hn, _) = self.lstm(x)

        return hn[0][reorder]
