

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


class Document:

    def __init__(self, tokens):
        self.tokens = tokens

    def __repr__(self):
        return 'Document<%d tokens>' % len(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def token_index_tensor(self):
        idx = [VECTORS.stoi(t.text) for t in self.tokens]
        idx = torch.LongTensor(idx).type(itype)
        return idx

    @cached_property
    def coref_id_to_tokens(self):
        """Map coref id -> token indexes, grouped by mention.
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

        return id_to_idx

    @cached_property
    def coref_id_to_spans(self):
        """Map coref id -> list of (start, end) token indexes.
        """
        return {
            cid: [(s[0], s[-1]) for s in spans]
            for cid, spans in self.coref_id_to_tokens.items()
        }

    @cached_property
    def span_to_antecedents(self):
        """Map span (start, end) -> list of (start, end) of antecedents.
        """
        return {
            span: set(spans[:i+1])
            for _, spans in self.coref_id_to_spans.items()
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

        for tokens in groups.values():
            # TODO: Randomly truncate up to 50 sents.
            yield Document(tokens[:500])


class Corpus:

    @classmethod
    def from_files(cls, root):
        """Load from globbed gold files.
        """
        docs = []
        for path in tqdm(scan_paths(root, 'gold_conll$')):
            gold = GoldFile(path)
            docs += list(gold.documents())

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


class FFNN(nn.Module):

    def __init__(self, input_dim, hidden_dim=150):

        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.score(x)


class SpanAttention(FFNN):

    def forward(self, x):
        return F.softmax(self.score(x).squeeze(), dim=0)


class Coref(nn.Module):

    def __init__(self, input_dim, lstm_dim):

        super().__init__()

        weights = VECTORS.weights()

        self.embeddings = nn.Embedding(
            weights.shape[0],
            weights.shape[1],
        )

        self.embeddings.weight.data.copy_(weights)
        self.embeddings.weight.requires_grad = False

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

        self.span_attention = SpanAttention(state_dim)
        self.span_scorer = FFNN(span_dim)
        self.pair_scorer = FFNN(pair_dim)

    def forward(self, doc):
        """Pad, pack, encode, reorder.

        Args:
            batch (Batch)
        """
        x = doc.token_index_tensor()
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
                size = torch.FloatTensor([n]).type(ftype)

                g = torch.cat([states[0], states[-1], attn, size])
                spans.append((i1, i2, g))

        # Score spans.
        g = torch.stack([s[2] for s in spans])
        sm = self.span_scorer(g).squeeze()

        # Sort spans by unary score.
        span_sm = sorted(zip(spans, sm), key=lambda p: p[1], reverse=True)
        spans = [(*span, sm) for span, sm in span_sm]

        # Remove overlapping spans.
        nonoverlapping = []
        taken = set()
        for s in spans:
            indexes = range(s[0], s[1]+1)
            takens = [i in taken for i in indexes]
            if len(set(takens)) == 1 or (takens[0] == takens[-1] == False):
                nonoverlapping.append(s)
                taken.update(indexes)

        # Take top lambda*T spans, keeping score.
        spans = nonoverlapping[:round(len(doc)*0.4)]

        # Sort spans by start index.
        spans = sorted(spans, key=lambda s: s[0])

        # Get pairwise `sa` scores in bulk.
        x = []
        for ix, i in enumerate(spans):
            for j in spans[:ix]:
                gi, gj = i[2], j[2]
                # TODO: Speaker / genre features.
                x.append(torch.cat([gi, gj, gi*gj]))

        x = torch.stack(x)
        sa = self.pair_scorer(x).view(-1)

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

            yield (
                (i[0], i[1]), # i
                [(j[0], j[1]) for j, _ in j_sa], # y(i)
                sij, # Scores for each y(i)
            )


class Trainer:

    def __init__(self, train_root, lr=1e-3):

        self.train_corpus = Corpus.from_files(train_root)

        VECTORS.set_vocab(self.train_corpus.vocab())

        self.model = Coref(300, 200)

        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam(params, lr=lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def train_epoch(self, epoch, batch_size=100):

        print(f'\nEpoch {epoch}')

        self.model.train()

        docs = random.sample(self.train_corpus.documents, batch_size)

        epoch_loss, correct, total = 0, 0, 0
        for doc in tqdm(docs):

            self.optimizer.zero_grad()

            total += len(doc.span_to_antecedents)

            losses = []
            for i, yi, sij in self.model(doc):

                gold_span_idxs = doc.span_to_antecedents.get(i, [])

                gold_pred_idxs = [
                    j for j, span in enumerate(yi)
                    if span in gold_span_idxs
                ]

                if not gold_pred_idxs:
                    gold_pred_idxs = [len(sij)-1]

                # Get distribution over possible antecedents.
                pred = F.softmax(sij, dim=0)

                p = sum([pred[i] for i in gold_pred_idxs])

                if p < 1:
                    losses.append(p.log())

                for ix in gold_pred_idxs:
                    if ix != len(pred)-1 and ix == pred.argmax().item():
                        correct += 1

            if losses:

                loss = sum(losses) / len(losses) * -1
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                epoch_loss += loss.item()

        print('Loss: %f' % epoch_loss)
        if total: print('Correct: %f' % (correct/total))

    def train(self, epochs=10, *args, **kwargs):
        for epoch in range(epochs):
            self.train_epoch(epoch, *args, **kwargs)
