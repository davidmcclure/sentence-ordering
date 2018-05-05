

import attr
import os
import re

from collections import defaultdict
from itertools import islice
from boltons.iterutils import pairwise, chunked, windowed
from tqdm import tqdm
from cached_property import cached_property
from glob import glob

import torch
from torchtext.vocab import Vectors
from torch import nn
from torch.nn import functional as F

from ..cuda import itype


def parse_int(text):
    """Parse an integer out of a string.
    """
    matches = re.findall('[0-9]+', text)
    return int(matches[0]) if matches else None


@attr.s
class Token:

    text = attr.ib()
    document_id = attr.ib()
    doc_index = attr.ib()
    sent_index = attr.ib()
    coref_id = attr.ib()


class Document:

    def __init__(self, tokens):
        self.tokens = tokens

    def __repr__(self):
        return 'Document<%d tokens>' % len(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def token_texts(self):
        return [t.text for t in self.tokens]

    @cached_property
    def sent_start_indexes(self):
        return [i for i, t in enumerate(self.tokens) if t.sent_index == 0]

    def sents(self):
        for i1, i2 in pairwise(self.sent_start_indexes + [len(self)]):
            yield self.tokens[i1:i2]


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
        for i, line in enumerate(self.lines()):

            digit = parse_int(line[-1])

            if digit is not None and line[-1].startswith('('):
                open_tag = digit

            yield Token(
                text=line[3],
                document_id=int(line[1]),
                doc_index=i,
                sent_index=int(line[2]),
                coref_id=open_tag,
            )

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
    def from_files(cls, root, skim=None):
        """Load from gold files.
        """
        pattern = os.path.join(root, '**/*gold_conll')

        paths = glob(pattern, recursive=True)[:skim]

        docs = []
        for path in tqdm(paths):
            docs += list(GoldFile(path).documents())

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


class WordEmbedding(nn.Embedding):

    def __init__(self, vocab, path='glove.840B.300d.txt'):
        """Set vocab, map s->i.
        """
        loader = Vectors(path)

        # Map string -> intersected index.
        self.vocab = [v for v in vocab if v in loader.stoi]
        self._stoi = {s: i for i, s in enumerate(self.vocab)}

        super().__init__(len(self.vocab)+2, loader.dim)

        # Select vectors for vocab words.
        weights = torch.stack([
            loader.vectors[loader.stoi[s]]
            for s in self.vocab
        ])

        # Padding + UNK zeros rows.
        weights = torch.cat([
            torch.zeros((2, loader.dim)),
            weights,
        ])

        # Copy in pretrained weights.
        self.weight.data.copy_(weights)

    def stoi(self, s):
        """Map string -> embedding index. <UNK> --> 1.
        """
        idx = self._stoi.get(s)
        return idx + 2 if idx is not None else 1

    def tokens_to_idx(self, tokens):
        """Given a list of tokens, map to embedding indexes.
        """
        return torch.LongTensor([self.stoi(t) for t in tokens]).type(itype)


class DocEncoder(nn.Module):

    def __init__(self, vocab, lstm_dim=200, lstm_num_layers=1):

        super().__init__()

        self.embeddings = WordEmbedding(vocab)

        self.lstm = nn.LSTM(
            self.embeddings.weight.shape[1],
            lstm_dim,
            bidirectional=True,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout()

    def forward(self, tokens):
        """BiLSTM over document tokens.
        """
        x = self.embeddings.tokens_to_idx(tokens)
        x = x.unsqueeze(0)

        embeds = self.embeddings(x)
        embeds = self.dropout(embeds)

        states, _ = self.lstm(embeds)
        states = self.dropout(states)

        return embeds.squeeze(), states.squeeze()


class SpanAttention(nn.Module):

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
        return F.softmax(self.score(x).squeeze(), dim=0)


class Span:

    def __init__(self, tokens, g):
        self.tokens = tokens
        self.g = g

    def __repr__(self):
        texts = ' '.join([t.text for t in self.tokens])
        return f'Span<{texts}, {self.g.shape}>'


class SpanEncoder(nn.Module):

    def __init__(self, state_dim=400):
        super().__init__()
        self.attention = SpanAttention(state_dim)

    def forward(self, doc, embeds, states):

        spans = []
        for n in range(1, 11):
            for tokens in windowed(doc.tokens, n):

                i1 = tokens[0].doc_index
                i2 = tokens[-1].doc_index

                span_embeds = embeds[i1:i2+1]
                span_states = states[i1:i2+1]

                attn = self.attention(span_states)
                attn = sum(span_embeds * attn.view(-1, 1))

                # TODO: Embedded span size phi.
                g = torch.cat([span_states[0], span_states[-1], attn])
                spans.append(Span(tokens, g))

        return spans


class Coref(nn.Module):

    def __init__(self, vocab):

        super().__init__()

        self.encode_doc = DocEncoder(vocab)
        self.encode_spans = SpanEncoder()

    def forward(self, doc):

        # LSTM over tokens.
        embeds, states = self.encode_doc(doc.token_texts())

        spans = self.encode_spans(doc, embeds, states)

        return spans

        # spans = self.score_spans(spans)
        # spans = self.prune_spans(spans)
