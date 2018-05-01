

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

from datetime import datetime as dt
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

    def __init__(self, tokens):
        self.tokens = tokens

    def __repr__(self):
        return 'Document<%d tokens>' % len(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def token_idx_tensor(self):
        idx = [VECTORS.stoi(t.text) for t in self.tokens]
        idx = torch.LongTensor(idx).type(itype)
        return idx

    @cached_property
    def coref_id_to_mentions(self):
        """Map coref id -> list of (start, end) token indexes.
        """
        id_to_idx = defaultdict(list)

        for i, token in enumerate(self.tokens):

            if not token.coref_id:
                continue

            spans = id_to_idx[token.coref_id]

            if len(spans) and spans[-1][-1] == i-1:
                spans[-1].append(i)

            else:
                spans.append([i])

        return {
            cid: [(s[0], s[-1]) for s in spans]
            for cid, spans in id_to_idx.items()
        }

    @cached_property
    def mentions(self):
        """Set of gold mention spans.
        """
        return set([
            mention for cluster in self.coref_id_to_mentions.values()
            for mention in cluster
        ])

    @cached_property
    def antecedents(self):
        """Map span (start, end) -> list of (start, end) of antecedents.
        """
        return {
            span: set(spans[:i+1])
            for _, spans in self.coref_id_to_mentions.items()
            for i, span in enumerate(spans[1:])
        }

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

        # TODO: Randomly truncate up to 50 sents.
        for tokens in groups.values():
            yield Document(tokens[:500])


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
            vocab.update([t.text for t in doc.tokens])

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
            # nn.ReLU(),
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
            # nn.ReLU(),
        )

    def forward(self, g):
        return self.score(g)


class PairScorer(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.score(x).view(-1)


class Coref(nn.Module):

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

        state_dim = lstm_dim * 2
        span_dim = state_dim * 2 + input_dim + 1
        pair_dim = span_dim * 3

        self.span_attention = SpanAttention(state_dim, 150)
        self.span_scorer = SpanScorer(span_dim, 150)
        self.pair_scorer = PairScorer(pair_dim, 150)

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
        for n in range(1, 5):
            for w in windowed_iter(range(len(doc)), n):

                i1, i2 = w[0], w[-1]
                tokens = embeds[0][i1:i2+1]
                states = x[0][i1:i2+1]

                # Attend over raw word embeddings.
                attn = self.span_attention(states).view(-1, 1)
                attn = sum(tokens * attn)

                # Include span size.
                size = torch.FloatTensor([n]).type(ftype)

                g = torch.cat([states[0], states[-1], attn, size])
                spans.append((i1, i2, g))

        # Score spans.
        g = torch.stack([s[2] for s in spans])
        sm = self.span_scorer(g).squeeze()

        # Sort spans by unary score.
        span_sm = sorted(zip(spans, sm), key=lambda p: p[1], reverse=True)

        # Take top lambda*T spans, keeping score.
        # TODO: Skip overlapping spans.
        spans = [(i1, i2, g, sm) for (i1, i2, g), sm in span_sm]
        spans = spans[:round(len(doc)*0.4)]

        # Sort spans by start index.
        spans = sorted(spans, key=lambda s: s[0])

        # Get pairwise `sa` scores in bulk.
        x = []
        for ix, i in enumerate(spans):
            for j in spans[:ix]:
                gi, gj = i[2], j[2]
                x.append(torch.cat([gi, gj, gi*gj]))

        x = torch.stack(x)
        sa = self.pair_scorer(x)

        # Get combined `sij` scores.
        c = 0
        for ix, i in enumerate(spans):

            # TODO: Just consider K antecedents.
            j_sa = []
            for j in spans[:ix]:
                j_sa.append((j, sa[c]))
                c += 1

            # Antecedents + 0 for epsilon.
            sij = [(i[-1] + j[-1] + sa).view(1) for j, sa in j_sa]
            sij = torch.cat([*sij, torch.FloatTensor([0]).type(ftype)])

            # Get distribution over possible antecedents.
            pred = F.softmax(sij, dim=0)

            yield (
                (i[0], i[1]), # i
                [(j[0], j[1]) for j, _ in j_sa], # y(i)
                pred, # distribution over y(i)
            )


class Trainer:

    def __init__(self, train_root, lr=1e-4):

        self.train_corpus = Corpus.from_files(train_root)

        VECTORS.set_vocab(self.train_corpus.vocab())

        self.model = Coref(300, 200)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def train_epoch(self, epoch):

        print(f'\nEpoch {epoch}')

        self.model.train()

        epoch_loss = 0
        correct = 0
        for doc in tqdm(random.sample(self.train_corpus.documents, 100)):

            loss = []
            for i, yi, pred in self.model(doc):

                gold_span_idxs = doc.antecedents.get(i, [])

                gold_pred_idxs = [
                    j for j, span in enumerate(yi)
                    if span in gold_span_idxs
                ]

                if not gold_pred_idxs:
                    gold_pred_idxs = [len(pred)-1]

                loss.append(sum([pred[i] for i in gold_pred_idxs]).log())

                for ix in gold_pred_idxs:
                    if ix != len(pred)-1 and ix == pred.argmax().item():
                        correct += 1

            loss = sum(loss) / len(loss) * -1
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()

        print('Loss: %f' % epoch_loss)
        print('Correct: %d' % correct)

    def train(self, epochs=10):
        for epoch in range(epochs):
            self.train_epoch(epoch)
