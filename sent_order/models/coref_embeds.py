

import os
import attr
import re
import random

import numpy as np

from itertools import groupby, combinations
from boltons.iterutils import pairwise, chunked, windowed
from tqdm import tqdm
from cached_property import cached_property
from glob import glob
from collections import defaultdict

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


def pad_right_and_stack(xs, pad_size=None):
    """Pad and stack a list of variable-length seqs.

    Args:
        xs (list[Variable])
        pad_size (int)

    Returns: stacked xs, sizes
    """
    # Default to max seq size.
    if not pad_size:
        pad_size = max([len(x) for x in xs])

    padded, sizes = [], []
    for x in xs:

        px = F.pad(x, (0, pad_size-len(x)))
        padded.append(px)

        size = min(pad_size, len(x))
        sizes.append(size)

    return torch.stack(padded), sizes


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

    def truncate_sents_random(self, max_sents=10):
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
    def coref_id_to_pred_index(self):
        """Map CONLL tag id -> index, by order of appearance.
        """
        kv = sorted(self.coref_id_to_i1i2.items(), key=lambda x: (x[1][0]))
        return {k: i for i, k in enumerate([k for k, _ in kv])}

    def y_true(self, out_dim):
        """Word -> ascending cluster id.
        """
        preds = []
        for t in self.tokens:

            pred = torch.zeros(out_dim).type(ftype)

            for cid in t.clusters:
                pred_ix = self.coref_id_to_pred_index[cid]+1
                if pred_ix < out_dim:
                    pred[pred_ix] = 1

            if not t.clusters:
                pred[0] = 1

            preds.append(pred)

        return torch.stack(preds)


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
        return groupby(self.lines(), lambda line: (line[0], line[1]))

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

    @classmethod
    def from_combined_file(cls, path):
        """Load from merged gold files.
        """
        docs = GoldFile(path).documents()
        return cls(list(docs))

    def __init__(self, documents):
        self.documents = documents

    def vocab(self):
        """Build vocab list.
        """
        vocab = set()

        for doc in self.documents:
            vocab.update([t.text for t in doc.tokens])

        return vocab

    def training_batches(self, size, max_sents):
        """Generate batches.
        """
        # Truncate randomly.
        docs = [d.truncate_sents_random(max_sents) for d in self.documents]

        # Sort by length, chunk.
        docs = sorted(docs, key=lambda d: len(d))
        batches = chunked(docs, size)

        # Shuffle lengths.
        return sorted(batches, key=lambda x: random.random())


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


class DocEmbedder(nn.Module):

    def __init__(self, vocab, lstm_dim=500, hidden_dim=200, out_dim=10,
        lstm_num_layers=1):

        super().__init__()

        self.out_dim = out_dim

        self.embeddings = WordEmbedding(vocab)

        self.lstm = nn.LSTM(
            self.embeddings.weight.shape[1],
            lstm_dim,
            bidirectional=True,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (n, lstm_dim*2))
            for n in (2, 3, 4, 5)
        ])

        self.dropout = nn.Dropout()

        self.embed = nn.Sequential(
            nn.Linear(lstm_dim*2+400+1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
        )

    @property
    def embed_dim(self):
        return self.embeddings.weight.shape[1]

    def forward(self, docs):
        """BiLSTM over document tokens.
        """
        x, _ = pad_right_and_stack([
            self.embeddings.tokens_to_idx(tokens)
            for tokens in docs
        ])

        x = self.embeddings(x)
        x = self.dropout(x)

        x, (hn, _) = self.lstm(x)
        x = self.dropout(x)

        cx = x.unsqueeze(1)
        cx = [F.relu(conv(cx)).squeeze(3) for conv in self.convs]
        cx = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in cx]
        cx = torch.cat(cx, 1)

        cx = cx.unsqueeze(1).expand(x.shape[0], x.shape[1], cx.shape[1])
        x = torch.cat([x, cx], 2)

        # Token positions.
        pos = torch.arange(0, x.shape[1]).type(ftype)
        pos = pos.expand(x.shape[0], x.shape[1]).unsqueeze(2)
        x = torch.cat([x, pos], dim=2)

        return self.embed(x)

    def train_batch(self, docs, out_dim=10):
        """Predict token classes.
        """
        tokens = [d.token_texts() for d in docs]

        yps = self(tokens)

        yt, yp = [], []
        for i, doc in enumerate(docs):

            yti = doc.y_true(self.out_dim)
            ypi = yps[i][:len(yti)]

            yt.append(yti)
            yp.append(ypi)

        yp = torch.cat(yp, dim=0)
        yt = torch.cat(yt, dim=0)

        return yp, yt


class Trainer:

    def __init__(self, train_path, dev_path, lr=1e-3,
        batch_size=10, max_sents=5):

        self.batch_size = batch_size
        self.max_sents = max_sents

        self.train_corpus = Corpus.from_combined_file(train_path)
        self.dev_corpus = Corpus.from_combined_file(dev_path)

        self.model = DocEmbedder(self.train_corpus.vocab())

        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam(params, lr=lr)

        if torch.cuda.is_available():
            self.model.cuda()

        self.dev_batches = self.dev_corpus.training_batches(
            self.batch_size, self.max_sents)

    def train(self, epochs=10, *args, **kwargs):
        for epoch in range(epochs):
            self.train_epoch(epoch, *args, **kwargs)

    def train_epoch(self, epoch):
        """Train a single epoch.
        """
        print(f'\nEpoch {epoch}')

        self.model.train()

        batches = self.train_corpus.training_batches(
            self.batch_size, self.max_sents)

        epoch_loss = []
        for docs in tqdm(batches):

            try:

                self.optimizer.zero_grad()

                yp, yt = self.model.train_batch(docs)

                loss = F.binary_cross_entropy(yp, yt)
                loss.backward()

                self.optimizer.step()

                epoch_loss.append(loss.item())

            except RuntimeError as e:
                print(e)

        print('Loss: %f' % np.mean(epoch_loss))
    #     print('Dev loss: %f' % self.dev_loss())
    #
    # def dev_loss(self):
    #     """Get dev loss.
    #     """
    #     self.model.eval()
    #
    #     losses = []
    #     for docs in tqdm(self.dev_batches):
    #
    #         try:
    #
    #             x1, x2, y = self.model.embed_training_pairs(docs)
    #             loss = F.cosine_embedding_loss(x1, x2, y)
    #             losses.append(loss.item())
    #
    #         except RuntimeError as e:
    #             print(e)
    #
    #     return np.mean(losses)
