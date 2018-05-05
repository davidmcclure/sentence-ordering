

import os
import attr
import re

from collections import defaultdict
from itertools import islice, groupby
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


def remove_consec_dupes(seq):
    """Remove consecutive duplicates in a list.
    [1,1,2,2,3,3] -> [1,2,3]
    """
    return [x[0] for x in groupby(seq)]


@attr.s
class Token:
    text = attr.ib()
    document_id = attr.ib()
    doc_index = attr.ib()
    sent_index = attr.ib()
    clusters = attr.ib()


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


# TODO: Nested tags? v4/data/train/data/english/annotations/bn/cnn/01/cnn_0102.v4_gold_conll
# *** v4/data/train/data/english/annotations/bn/pri/01/pri_0103.v4_gold_conll

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

    # TODO: Test the cluster parsing.
    def tokens(self):
        """Generate tokens.
        """
        open_clusters = set()
        for i, line in enumerate(self.lines()):

            clusters = open_clusters.copy()

            parts = [p for p in line[-1].split('|') if p != '-']

            for part in parts:

                cid = parse_int(part)

                # Open: (5
                if re.match('^\(\d+$', part):
                    clusters.add(cid)
                    open_clusters.add(cid)

                # Close: 5)
                elif re.match('^\d+\)$', part):
                    open_clusters.remove(cid)

                # Solo: (5)
                elif re.match('^\((\d+)\)$', part):
                    clusters.add(cid)

            yield Token(
                text=line[3],
                document_id=int(line[1]),
                doc_index=i,
                sent_index=int(line[2]),
                clusters=clusters,
            )

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

    def __init__(self, vocab, lstm_dim, lstm_num_layers=1):

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

    @property
    def embed_dim(self):
        return self.embeddings.weight.shape[1]

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


class Scorer(nn.Module):

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


class SpanAttention(Scorer):
    pass


class Span:

    def __init__(self, tokens, g):
        self.tokens = tokens
        self.g = g
        self.score = None

    def __repr__(self):
        return 'Span<%d, %d, %s, %s, %f>' % (
            self.start_idx,
            self.end_idx,
            ' '.join([t.text for t in self.tokens]),
            self.g.shape,
            self.score,
        )

    @cached_property
    def start_idx(self):
        return self.tokens[0].doc_index

    @cached_property
    def end_idx(self):
        return self.tokens[-1].doc_index


class SpanEncoder(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.attention = SpanAttention(state_dim)

    def forward(self, doc, embeds, states):

        # Get raw attention scores in bulk.
        attns = self.attention(states)
        print(attns.shape)

        spans = []
        for n in range(1, 11):
            for tokens in windowed(doc.tokens, n):

                i1 = tokens[0].doc_index
                i2 = tokens[-1].doc_index
                print(i1, i2)

                span_embeds = embeds[i1:i2+1]
                span_states = states[i1:i2+1]
                span_attns = attns[i1:i2+1]

                print(span_attns.shape)

                # Softmax over attention scores for span.
                attn = F.softmax(span_attns.squeeze(), dim=0)
                attn = sum(span_embeds * attn.view(-1, 1))

                # TODO: Embedded span size phi.
                g = torch.cat([span_states[0], span_states[-1], attn])
                spans.append(Span(tokens, g))

        return spans


class SpanScorer(Scorer):

    def forward(self, spans):

        x = torch.stack([s.g for s in spans])
        scores = self.score(x).squeeze()

        # Set scores on spans.
        # TODO: Can we tolist() here?
        for span, sm in zip(spans, scores.tolist()):
            span.score = sm

        return spans


def prune_spans(spans, T, lbda=0.4):
    """Sort by score; remove overlaps, skim lamda*T; sort by start idx.
    """
    # Sory by unary score, descending.
    spans = sorted(spans, key=lambda s: s.score, reverse=True)

    # Remove spans that overlap with higher-scoring spans.
    taken_idxs = set()
    pruned = []
    for span in spans:

        span_idxs = [t.doc_index for t in span.tokens]
        slots = [idx in taken_idxs for idx in span_idxs]
        slots = remove_consec_dupes(slots)

        if len(slots) == 1 or slots == [False, True, False]:
            pruned.append(span)
            taken_idxs.update(span_idxs)

    # Take top lambda*T.
    pruned = pruned[:round(T*lbda)]

    # Order by left doc index.
    pruned = sorted(pruned, key=lambda s: s.tokens[0].doc_index)

    return pruned


class Coref(nn.Module):

    def __init__(self, vocab, lstm_dim=200):

        super().__init__()

        self.encode_doc = DocEncoder(vocab, lstm_dim)

        # LSTM forward + back.
        state_dim = lstm_dim * 2

        # Left + right LSTM states, head attention.
        g_dim = state_dim * 2 + self.encode_doc.embed_dim

        self.encode_spans = SpanEncoder(state_dim)
        self.score_spans = SpanScorer(g_dim)

    def forward(self, doc):

        # LSTM over tokens.
        embeds, states = self.encode_doc(doc.token_texts())

        spans = self.encode_spans(doc, embeds, states)
        spans = self.score_spans(spans)
        spans = prune_spans(spans, len(doc))

        # score pairs
        # yield distributions over antecedents

        return spans
