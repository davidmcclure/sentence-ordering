

import torch
import os
import ujson
import attr
import random
import re

import numpy as np

from torchtext.vocab import Vectors
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F

from cached_property import cached_property
from glob import glob
from boltons.iterutils import chunked_iter
from tqdm import tqdm
from itertools import islice
from scipy import stats

from sent_order.cuda import itype, ftype


def parse_int(text):
    """Parse an integer out of a string.
    """
    matches = re.findall('[0-9]+', text)
    return int(matches[0]) if matches else None


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
class Token:

    text = attr.ib()
    document_id = attr.ib()
    coref_id = attr.ib()


class Document:

    def __init__(self, tokens):
        self.tokens = tokens

    def __repr__(self):
        return 'Document<%d tokens>' % len(self.tokens)


class GoldFile:

    def __init__(self, path):
        self.path = path

    def lines(self):
        """Split lines into cols. Skip comments / blank lines.
        """
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith('#'):
                    yield line.split()

    def tokens(self):
        """Generate tokens.
        """
        open_tag = None
        for line in self.lines():

            digit = parse_int(line[-1])

            if digit is not None and line[-1].startswith('('):
                open_tag = digit

            yield Token(line[3], int(line[1]), open_tag)

            if line[-1].endswith(')'):
                open_tag = None

    def documents(self):
        """Group tokens by document.
        """
        tokens = []

        for token in self.tokens():

            if not tokens or token.document_id == tokens[-1].document_id:
                tokens.append(token)

            else:
                yield Document(tokens)
                tokens = [token]
