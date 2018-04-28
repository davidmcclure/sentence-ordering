

import torch
import os
import ujson
import attr
import random
import re
import glob

import numpy as np

from torchtext.vocab import Vectors
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F

from collections import UserList
from cached_property import cached_property
from glob import glob
from boltons.iterutils import chunked_iter, windowed_iter
from collections import defaultdict
from tqdm import tqdm
from itertools import islice
from scipy import stats

from sent_order.cuda import itype, ftype
from sent_order.utils import scan_paths


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


class Document(UserList):

    def __repr__(self):
        return 'Document<%d tokens>' % len(self)

    def token_idx_tensor(self):
        idx = [VECTORS.stoi(t.text) for t in self]
        idx = torch.LongTensor(idx).type(itype)
        return idx


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
        groups = defaultdict(list)

        for token in self.tokens():
            groups[token.document_id].append(token)

        for tokens in groups.values():
            yield Document(tokens)


class Corpus:

    @classmethod
    def from_files(cls, root):
        """Load from globbed gold files.
        """
        docs = []
        for path in tqdm(scan_paths(root, 'gold_conll$')):
            gf = GoldFile(path)
            docs += list(gf.documents())

        return cls(docs)

    def __init__(self, documents):
        self.documents = documents

    def vocab(self):
        """Build vocab list.
        """
        vocab = set()

        for doc in self.documents:
            vocab.update([t.text for t in doc])

        return vocab


class SpanAttention(nn.Module):

    def __init__(self, state_dim, hidden_dim):

        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states):
        return F.softmax(self.score(states).squeeze(), dim=0)


class SpanScorer(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, g):
        return self.score(g)


class DocEncoder(nn.Module):

    def __init__(self, input_dim, lstm_dim):

        super().__init__()

        weights = VECTORS.weights()

        self.embeddings = nn.Embedding(
            weights.shape[0],
            weights.shape[1],
        )

        self.embeddings.weight.data.copy_(weights)

        self.lstm = nn.LSTM(
            input_dim,
            lstm_dim,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout()

        self.span_attention = SpanAttention(lstm_dim*2, 150)
        self.span_scorer = SpanScorer(lstm_dim*2*2 + input_dim + 1, 150)

    def forward(self, doc):
        """Pad, pack, encode, reorder.

        Args:
            batch (Batch)
        """
        x = doc.token_idx_tensor()

        # TODO: Batch?
        x = x.unsqueeze(0)

        embeds = self.embeddings(x)
        # TODO: Char CNN.

        x, _ = self.lstm(embeds)
        x = self.dropout(x)

        # Generate and encode spans.
        spans = []
        for n in range(1, 11):
            for w in windowed_iter(range(len(doc)), n):

                i1, i2 = w[0], w[-1]
                tokens = embeds[0][i1:i2+1]
                states = x[0][i1:i2+1]

                # Attend over raw word embeddings.
                attn = self.span_attention(states).view(-1, 1)
                attn = sum(tokens * attn)

                # Include span size.
                size = torch.FloatTensor([n])

                g = torch.cat([states[0], states[-1], attn, size])
                spans.append((i1, i2, g))

        g = torch.stack([s[2] for s in spans])
        sm = self.span_scorer(g)

        print(sm)

        # score spans
        # skim top-scoring
        # generate pairs
        # get pair embeddings
        # score pairs, softmax
