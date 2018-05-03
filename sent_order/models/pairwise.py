

import attr
import re
import numpy as np

from collections import defaultdict
from cached_property import cached_property
from tqdm import tqdm
from boltons.iterutils import pairwise, chunked
from itertools import islice

import torch
from torchtext.vocab import Vectors
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F

from ..cuda import itype, ftype
from ..utils import scan_paths


def parse_int(text):
    """Parse an integer out of a string.
    """
    matches = re.findall('[0-9]+', text)
    return int(matches[0]) if matches else None


def pad_right_and_stack(xs, pad_size=None):
    """Pad and stack a list of variable-length seqs.

    Args:
        xs (list[Tensor])
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


def pack(x, sizes, batch_first=True):
    """Pack padded variables, provide reorder indexes.

    Args:
        batch (Variable)
        sizes (list[int])

    Returns: packed sequence, reorder indexes
    """
    # Get indexes for sorted sizes.
    size_sort = np.argsort(sizes)[::-1].tolist()

    # Sort tensor by size.
    x = x[torch.LongTensor(size_sort).type(itype)]

    # Sort sizes descending.
    sizes = np.array(sizes)[size_sort].tolist()

    # Pack the sequences.
    x = pack_padded_sequence(x, sizes, batch_first)

    # Indexes to restore original order.
    reorder = torch.LongTensor(np.argsort(size_sort)).type(itype)

    return x, reorder


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
        paths = islice(scan_paths(root, 'gold_conll$'), skim)

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

    def pairs(self):
        """Generate sentence pairs.
        """
        for doc in self.documents:
            for s1, s2 in pairwise(doc.sents()):
                yield [t.text for t in s1], [t.text for t in s2]

    def batches(self, size):
        """Generate batches.
        """
        return chunked(self.pairs(), size)


class Embedding(nn.Embedding):

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

    def __contains__(self, token):
        """Check if word is in vocab.
        """
        return token in self._stoi

    def stoi(self, s):
        """Map string -> embedding index.
        """
        idx = self._stoi.get(s)
        return idx + 2 if idx is not None else 1

    def tokens_to_idx(self, tokens):
        """Given a list of tokens, map to embedding indexes.
        """
        return torch.LongTensor([self.stoi(t) for t in tokens]).type(itype)


class Classifier(nn.Module):

    def __init__(self, vocab, lstm_dim, lstm_num_layers, hidden_dim):

        super().__init__()

        self.embeddings = Embedding(vocab)

        self.lstm = nn.LSTM(
            self.embeddings.weight.shape[1],
            lstm_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=lstm_num_layers,
            dropout=0.3,
        )

        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, 100, (n, lstm_dim*2))
            for n in (1, 2, 3, 4, 5)
        ])

        self.hidden = nn.Linear(500, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)

        self.dropout = nn.Dropout()

    def forward(self, pairs):
        """Given sentence pair as a single stream of tokens, predict whether
        the sentences are in order.
        """
        x, sizes = pad_right_and_stack([
            self.embeddings.tokens_to_idx(tokens)
            for tokens in pairs
        ])

        x = self.embeddings(x)
        x = self.dropout(x)

        # x, reorder = pack(x, sizes)

        x, _ = self.lstm(x)
        # hn = self.dropout(hn)

        # Cat forward + backward hidden layers.
        # x = torch.cat([hn[0,:,:], hn[1,:,:]], dim=1)
        # x = x[reorder]

        x = x.unsqueeze(1)
        # print(x.shape)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        # x = torch.stack([
        #     torch.cat([x[i1], x[i2]])
        #     for i1, i2 in chunked(range(len(x)), 2)
        # ])

        x = F.relu(self.hidden(x))
        x = F.log_softmax(self.out(x), dim=1)

        return x

    def train_batch(self, batch):
        """Generate correct / flipped pairs, predict.

        Returns: y pred, y true
        """
        x, y = [], []
        for s1, s2 in batch:

            # Correct.
            x.append(s1 + s2)
            y.append(0)

            # Reversed.
            x.append(s2 + s1)
            y.append(1)

        y = torch.LongTensor(y).type(itype)

        return self(x), y


class Trainer:

    def __init__(self, train_path, val_path, train_skim=None, val_skim=None,
        lstm_dim=500, lstm_num_layers=1, hidden_dim=200, lr=1e-3):

        self.train_corpus = Corpus.from_files(train_path, train_skim)
        self.val_corpus = Corpus.from_files(val_path, val_skim)

        vocab = set.union(
            self.train_corpus.vocab(),
            self.val_corpus.vocab(),
        )

        self.model = Classifier(vocab, lstm_dim, lstm_num_layers, hidden_dim)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self, epochs=10, batch_size=20):

        for epoch in range(epochs):

            print(f'\nEpoch {epoch}')

            batches = self.train_corpus.batches(batch_size)

            epoch_loss = []
            for i, batch in enumerate(tqdm(batches)):

                self.optimizer.zero_grad()
                self.model.train()

                yp, yt = self.model.train_batch(batch)

                loss = F.nll_loss(yp, yt)
                loss.backward()

                self.optimizer.step()

                epoch_loss.append(loss.item())

            print('Loss: %f' % np.mean(epoch_loss))
            print('Val accuracy: %f' % self.val_accuracy())

    def val_accuracy(self):

        self.model.eval()

        correct, total = 0, 0
        for batch in tqdm(self.val_corpus.batches(20)):

            yps, yts = self.model.train_batch(batch)

            for yp, yt in zip(yps, yts):
                if yp.argmax() == yt:
                    correct += 1

            total += len(yps)

        return correct / total
