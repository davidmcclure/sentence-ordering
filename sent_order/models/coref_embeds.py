

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

    def __init__(self, vocab, lstm_dim=500, hidden_dim=200, embed_dim=50,
        lstm_num_layers=1):

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

        self.embed = nn.Sequential(
            nn.Linear(lstm_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim),
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

        x, _ = self.lstm(x)
        x = self.dropout(x)

        return self.embed(x)

    def embed_training_pairs(self, docs, skim=1000):
        """Build (token1, token2, coref flag) pairs.
        """
        tokens = [d.token_texts() for d in docs]

        embeds = self(tokens)

        # Get positive / negative examples.
        x1, x2, y = [], [], []
        for doc, tokens in zip(docs, embeds):

            # Get indexes of tokens in / not in clusters.
            cidx, eidx = [], []
            for i, t in enumerate(doc.tokens):
                if t.clusters:
                    cidx.append(i)
                else:
                    eidx.append(i)

            # Clamp empty indexes to size of coref indexes.
            if len(eidx) > len(cidx):
                eidx = random.sample(eidx, len(cidx))

            whitelist = cidx + eidx

            # Take pairs, select at most 1000 random.
            pairs = list(combinations(whitelist, 2))
            pairs = random.sample(pairs, min(skim, len(pairs)))

            for i1, i2 in pairs:

                c1 = doc.tokens[i1].clusters
                c2 = doc.tokens[i2].clusters

                # Connected if both have coref tag, or both empty.
                coref = (bool(set.intersection(c1, c2)) or
                    (not c1 and not c2))

                x1.append(tokens[i1])
                x2.append(tokens[i2])
                y.append(1 if coref else -1)

        x1 = torch.stack(x1)
        x2 = torch.stack(x2)
        y = torch.FloatTensor(y).type(ftype)

        return x1, x2, y


class Trainer:

    def __init__(self, train_path, lr=1e-3):

        self.train_corpus = Corpus.from_combined_file(train_path)

        self.model = DocEmbedder(self.train_corpus.vocab())

        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam(params, lr=lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self, epochs=10, *args, **kwargs):
        for epoch in range(epochs):
            self.train_epoch(epoch, *args, **kwargs)

    def train_epoch(self, epoch, batch_size=10, max_sents=10):
        """Train a single epoch.
        """
        print(f'\nEpoch {epoch}')

        self.model.train()

        batches = self.train_corpus.training_batches(batch_size, max_sents)

        epoch_loss = []
        for docs in tqdm(batches):

            try:
                loss = self.train_batch(docs)
                epoch_loss.append(loss)
            except Exception as e:
                print(e, docs)

        print('Loss: %f' % np.mean(epoch_loss))

    def train_batch(self, docs):
        """Embed docs, generate pairs.
        """
        self.optimizer.zero_grad()

        x1, x2, y = self.model.embed_training_pairs(docs)

        loss = F.cosine_embedding_loss(x1, x2, y)
        loss.backward()

        self.optimizer.step()

        return loss.item()
