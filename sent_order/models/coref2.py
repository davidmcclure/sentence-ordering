

import os
import attr
import re
import random

import numpy as np

from collections import defaultdict
from itertools import islice, groupby
from functools import reduce
from boltons.iterutils import pairwise, chunked, windowed
from tqdm import tqdm
from cached_property import cached_property
from glob import glob

import torch
from torchtext.vocab import Vectors
from torch import nn, optim
from torch.nn import functional as F

from ..cuda import itype, ftype


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


# TODO: Test.
def regroup_indexes(seq, size_fn):
    """Given a sequence A that contains items of variable size, provide a list
    of indexes that, when iterated as pairs, will slice a flat sequence B into
    groups with sizes that correspond to the items in A.

    Args:
        seq (iterable)
        size_fn (func): Provides size of an individual item in `seq`.

    Returns: list<int>
    """
    return reduce(lambda ix, i: (*ix, ix[-1] + size_fn(i)), seq, (0,))


@attr.s
class Token:
    doc_slug = attr.ib()
    doc_part = attr.ib()
    text = attr.ib()
    sent_index = attr.ib()
    clusters = attr.ib()


class Document:

    # TODO: Test the cluster parsing.
    # v4/data/train/data/english/annotations/bn/pri/01/pri_0103.v4_gold_conll
    @classmethod
    def from_lines(cls, lines):
        """Parse tokens.
        """
        tokens = []

        open_clusters = set()
        for i, line in enumerate(lines):

            clusters = open_clusters.copy()

            parts = [p for p in line[-1].split('|') if p != '-']

            for part in parts:

                cid = parse_int(part)

                # Open: (5
                if re.match('^\(\d+$', part):
                    clusters.add(cid)
                    open_clusters.add(cid)

                # Close: 5)
                elif re.match('^\d+\)$', part) and cid in open_clusters:
                    open_clusters.remove(cid)

                # Solo: (5)
                elif re.match('^\((\d+)\)$', part):
                    clusters.add(cid)

            tokens.append(Token(
                doc_slug=line[0],
                doc_part=int(line[1]),
                text=line[3],
                sent_index=int(line[2]),
                clusters=clusters,
            ))

        return cls(tokens)

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

    def truncate_sents_random(self, max_sents=50):
        """Randomly truncate sents up to N, return new instance.
        """
        sents = list(self.sents())

        count = random.randint(1, min(len(sents), max_sents))
        start = random.randint(0, len(sents)-count)

        # Slice out random sentence window.
        new_sents = sents[start:start+count]

        # Flatten out tokens.
        tokens = [t for sent in new_sents for t in sent]

        return self.__class__(tokens)

    @cached_property
    def coref_id_to_index_range(self):
        """Map coref id -> token indexes, grouped by mention.
        """
        id_idx = defaultdict(list)

        for i, token in enumerate(self.tokens):
            for cid in token.clusters:

                spans = id_idx[cid]

                if len(spans) and spans[-1][-1] == i-1:
                    spans[-1].append(i)

                else:
                    spans.append([i])

        return id_idx

    @cached_property
    def coref_id_to_i1i2(self):
        """Map coref id -> (start, end) span indexes.
        """
        return {
            cid: [(s[0], s[-1]) for s in spans]
            for cid, spans in self.coref_id_to_index_range.items()
        }

    @cached_property
    def i1i2_to_ant_i1i2(self):
        """Map span (start, end) -> list of (start, end) of antecedents.
        """
        return {
            span: set(spans[:i+1])
            for _, spans in self.coref_id_to_i1i2.items()
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

    def doc_id_lines(self):
        """Group lines by document.
        """
        return groupby(self.lines(), lambda line: int(line[1]))

    def documents(self):
        """Parse lines -> tokens, generate documents.
        """
        for _, lines in self.doc_id_lines():
            yield Document.from_lines(lines)


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

    # TODO: Turian embeddings.
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
        # TODO: Char CNN.
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


@attr.s(frozen=True, repr=False)
class Span:

    # Parent document reference.
    doc = attr.ib()

    # Left / right token indexes.
    i1 = attr.ib()
    i2 = attr.ib()

    # Span embedding tensor.
    g = attr.ib()

    # Unary mention score, as tensor.
    sm = attr.ib(default=None)

    # List of candidate antecedent spans.
    yi = attr.ib(default=None)

    # Pairwise scores for each yi.
    sij = attr.ib(default=None)

    def __repr__(self):
        text = ' '.join([t.text for t in self.tokens])
        return 'Span<%d, %d, "%s">' % (self.i1, self.i2, text)

    @cached_property
    def tokens(self):
        return self.doc.tokens[self.i1:self.i2+1]

    @cached_property
    def i1i2(self):
        return (self.i1, self.i2)

    def sij_gold_indexes(self):
        """Get indexes of gold antecedents in the sij score tensor.
        """
        ant_i1i2s = self.doc.i1i2_to_ant_i1i2.get(self.i1i2, [])

        sij_idx = [
            i for i, span in enumerate(self.yi)
            if span.i1i2 in ant_i1i2s
        ]

        if not sij_idx:
            sij_idx = [len(self.sij)-1]

        return sij_idx


class SpanScorer(nn.Module):

    def __init__(self, state_dim, gi_dim):
        super().__init__()
        self.attention = Scorer(state_dim)
        self.sm = Scorer(gi_dim)

    def forward(self, doc, embeds, states):
        """Generate spans, attend over LSTM states, form encodings.
        """
        # Get raw attention scores over LSTM states.
        attns = self.attention(states)

        # Slice out spans, build encodings.
        spans = []
        for n in range(1, 11):
            for w in windowed(range(len(doc.tokens)), n):

                i1, i2 = w[0], w[-1]

                span_embeds = embeds[i1:i2+1]
                span_states = states[i1:i2+1]
                span_attns = attns[i1:i2+1]

                # Softmax over attention scores for span.
                attn = F.softmax(span_attns.squeeze(), dim=0)
                attn = sum(span_embeds * attn.view(-1, 1))

                # Left LSTM + right LSTM + attention.
                # TODO: Embedded span size phi.
                g = torch.cat([span_states[0], span_states[-1], attn])

                spans.append(Span(doc, i1, i2, g))

        x = torch.stack([s.g for s in spans])

        # Unary scores for spans.
        scores = self.sm(x).squeeze()

        # Set scores on spans.
        # TODO: Scalar sm_sort, for perf?
        spans = [
            attr.evolve(span, sm=sm)
            for span, sm in zip(spans, scores)
        ]

        return spans


def prune_spans(spans, T, lbda=0.4):
    """Sort by score; remove overlaps, skim lamda*T; sort by start idx.
    """
    # Sory by unary score, descending.
    spans = sorted(spans, key=lambda s: s.sm, reverse=True)

    # Remove spans that overlap with higher-scoring spans.
    taken_idxs = set()
    pruned = []
    for span in spans:

        span_idxs = range(span.i1, span.i2+1)
        slots = [idx in taken_idxs for idx in span_idxs]
        slots = remove_consec_dupes(slots)

        # Allow spans that overlap with nothing, or spans that are "supersets"
        # of previously-accepted spans - start to left, end to right.
        if len(slots) == 1 or slots == [False, True, False]:
            pruned.append(span)
            taken_idxs.update(span_idxs)

    # Take top lambda*T.
    pruned = pruned[:round(T*lbda)]

    # Order by left doc index.
    pruned = sorted(pruned, key=lambda s: s.i1)

    return pruned


class PairScorer(Scorer):

    def forward(self, spans):
        """Map span -> candidate antecedents, score pairs.
        """
        # Take up to K antecedents.
        # TODO: Don't include spans that overlap?
        spans = [
            attr.evolve(span, yi=spans[ix-250:ix])
            for ix, span in enumerate(spans)
        ]

        # Form pair embeddings.
        # TODO: Distance / speaker embeds.
        pairs = [
            torch.cat([i.g, j.g, i.g*j.g])
            for i in spans for j in i.yi
        ]

        if not pairs:
            raise RuntimeError('No candidate pairs after pruning.')

        # Get pairwise `sa` scores.
        scores = self.score(torch.stack(pairs)).view(-1)

        sa_idx = regroup_indexes(spans, lambda s: len(s.yi))

        spans_sij = []
        for span, (i1, i2) in zip(spans, pairwise(sa_idx)):

            # Build composite `sij` scores.
            sij = [
                (span.sm + yi.sm + sa).view(1)
                for yi, sa in zip(span.yi, scores[i1:i2])
            ]

            # Add epsilon score 0 as last element.
            epsilon = torch.FloatTensor([0]).type(ftype)
            sij = torch.cat([*sij, epsilon])

            spans_sij.append(attr.evolve(span, sij=sij))

        return spans_sij


class Coref(nn.Module):

    def __init__(self, vocab, lstm_dim=200):

        super().__init__()

        self.encode_doc = DocEncoder(vocab, lstm_dim)

        # LSTM forward + back.
        state_dim = lstm_dim * 2

        # Left + right LSTM states, head attention.
        gi_dim = state_dim * 2 + self.encode_doc.embed_dim

        # i, j, i*j
        gij_dim = gi_dim * 3

        self.score_spans = SpanScorer(state_dim, gi_dim)
        self.score_pairs = PairScorer(gij_dim)

    def forward(self, doc):

        # LSTM over tokens.
        embeds, states = self.encode_doc(doc.token_texts())

        spans = self.score_spans(doc, embeds, states)
        spans = prune_spans(spans, len(doc))

        return self.score_pairs(spans)


class Trainer:

    def __init__(self, train_root, lr=1e-3):

        self.train_corpus = Corpus.from_files(train_root)

        self.model = Coref(self.train_corpus.vocab())

        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam(params, lr=lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self, epochs=10, *args, **kwargs):
        for epoch in range(epochs):
            self.train_epoch(epoch, *args, **kwargs)

    def train_epoch(self, epoch, batch_size=100):

        print(f'\nEpoch {epoch}')

        self.model.train()

        docs = random.sample(self.train_corpus.documents, batch_size)

        epoch_loss = []
        for doc in tqdm(docs):

            # Handle errors when model over-prunes.
            try:
                epoch_loss.append(self.train_doc(doc))
            except RuntimeError as e:
                print(e)

        print('Loss: %f' % np.mean(epoch_loss))

    def train_doc(self, doc):

        # Train on random sub-docs.
        doc = doc.truncate_sents_random()

        self.optimizer.zero_grad()

        losses = []
        for span in self.model(doc):

            # Softmax over possible antecedents.
            pred = F.softmax(span.sij, dim=0)

            # Get indexes of correct antecedents.
            yt = span.sij_gold_indexes()

            # Sum mass assigned to correct antecedents.
            p = sum([pred[i] for i in yt])

            losses.append(p.log())

            yp = pred.argmax().item()
            if yp != len(pred)-1 and yp in yt:
                print(span, span.yi[yp])

        loss = sum(losses) * -1
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

        return loss.item()
