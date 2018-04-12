

import torch
import os
import ujson
import attr
import random

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


def pad_and_stack(xs, pad_size=None):
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

        # Ensure length > 0.
        if len(x) == 0:
            x = x.new(1).zero_()

        if x.dim() == 1:
            padding = (0, pad_size-len(x))

        elif x.dim() == 2:
            padding = (0, 0, 0, pad_size-len(x))

        px = F.pad(x, padding)
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

    # Sort the tensor by size.
    x = x[torch.LongTensor(size_sort).type(itype)]

    # Sort sizes descending.
    sizes = np.array(sizes)[size_sort].tolist()

    x = pack_padded_sequence(x, sizes, batch_first)

    # Indexes to restore original order.
    reorder = torch.LongTensor(np.argsort(size_sort)).type(itype)

    return x, reorder


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
class Sentence:

    position = attr.ib()
    tokens = attr.ib()

    def token_idx_tensor(self):
        idx = [VECTORS.stoi(s) for s in self.tokens]
        idx = torch.LongTensor(idx)
        return idx.type(itype)


@attr.s
class Paragraph:

    sents = attr.ib()

    @classmethod
    def read_arxiv(cls, path, scount=None):
        """Wrap parsed arXiv abstracts as paragraphs.
        """
        for path in glob(os.path.join(path, '*.json')):
            for line in open(path):

                graf = cls.from_arxiv_json(line)

                # Filter by sentence count.
                if not scount or len(graf) == scount:
                    yield graf

    @classmethod
    def from_arxiv_json(cls, line):
        """Parse JSON, take tokens.
        """
        json = ujson.loads(line.strip())

        return cls([
            Sentence(pos, s['token'])
            for pos, s in enumerate(json['sentences'])
        ])

    def __len__(self):
        return len(self.sents)

    def sent_pos_tensor(self):
        pos = [sent.position / (len(self.sents)-1) for sent in self.sents]
        pos = torch.FloatTensor(pos)
        return pos.type(ftype)

    def shuffle(self):
        """Shuffle sents in-place.
        """
        random.shuffle(self.sents)


@attr.s
class Batch:

    grafs = attr.ib()

    def token_idx_tensor(self, pad=50):
        """Token indexes, flattened to 1d series.
        """
        return pad_and_stack([
            sent.token_idx_tensor()
            for graf in self.grafs
            for sent in graf.sents
        ])

    def sent_pos_tensors(self):
        """Token indexes, flattened to 1d series.
        """
        return [
            graf.sent_pos_tensor()
            for graf in self.grafs
        ]

    def repack_grafs(self, encoded):
        """Repacking encoded sentences.
        """
        start = 0
        for ab in self.grafs:
            end = start + len(ab.sents)
            yield encoded[start:end]
            start = end

    def shuffle(self):
        """Shuffle all grafs in-place.
        """
        for graf in self.grafs:
            graf.shuffle()


class Corpus:

    def __init__(self, path, skim=None, scount=None):
        """Load grafs into memory.
        """
        reader = Paragraph.read_arxiv(path, scount)

        if skim:
            reader = islice(reader, skim)

        self.grafs = list(tqdm(reader, total=skim))

    def vocab(self):
        """Build vocab list.
        """
        vocab = set()

        for graf in self.grafs:
            for sent in graf.sents:
                vocab.update(sent.tokens)

        return vocab

    def batches(self, size):
        """Generate batches.
        """
        return [Batch(grafs) for grafs in chunked_iter(self.grafs, size)]


class SentEncoder(nn.Module):

    def __init__(self, input_dim, lstm_dim):
        """Initialize the LSTM.
        """
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

    def forward(self, batch):
        """Pad, pack, encode, reorder.

        Args:
            batch (Batch)
        """
        x, sizes = batch.token_idx_tensor()

        x = self.embeddings(x)
        x = self.dropout(x)

        x, reorder = pack(x, sizes)

        _, (hn, _) = self.lstm(x)
        hn = self.dropout(hn)

        # Cat forward + backward hidden layers.
        out = torch.cat([hn[0,:,:], hn[1,:,:]], dim=1)

        return out[reorder]


class GrafEncoder(nn.Module):

    def __init__(self, input_dim, lstm_dim):
        """Initialize the LSTM.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            lstm_dim,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout()

    def forward(self, x):
        """Pad, pack, encode, reorder.

        Args:
            grafs (list of Variable): Encoded sentences for each graf.
        """
        x, reorder = pack(*pad_and_stack(x))

        _, (hn, _) = self.lstm(x)
        hn = self.dropout(hn)

        # Cat forward + backward hidden layers.
        out = torch.cat([hn[0,:,:], hn[1,:,:]], dim=1)

        return out[reorder]


class Regressor(nn.Module):

    def __init__(self, input_dim, lin_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim, lin_dim)
        self.out = nn.Linear(lin_dim, 1)

    def forward(self, x):
        y = F.relu(self.lin(x))
        return self.out(y)


class Model(nn.Module):

    def __init__(self, se_dim=500, ge_dim=500, lin_dim=200):

        super().__init__()

        self.sent_encoder = SentEncoder(VECTORS.loader.dim, se_dim)
        self.graf_encoder = GrafEncoder(se_dim*2, ge_dim)
        self.regressor = Regressor(se_dim*2 + ge_dim*2, lin_dim)

    def forward(self, batch):
        """Given a set of shuffled paragraphs, predict orderings.
        """
        # Batch-encode sents.
        sents = self.sent_encoder(batch)

        # Batch-encode grafs.
        grafs = list(batch.repack_grafs(sents))
        grafs = self.graf_encoder(grafs)

        x = torch.stack([
            torch.cat([graf, sent], dim=0)
            for graf, sents in zip(grafs, batch.repack_grafs(sents))
            for sent in sents
        ])

        y = self.regressor(x)

        return [graf.squeeze() for graf in batch.repack_grafs(y)]


class Trainer:

    def __init__(self, train_path, val_path, train_skim=None, val_skim=None,
        lr=1e-3, *args, **kwargs):

        self.train_corpus = Corpus(train_path, train_skim)
        self.val_corpus = Corpus(val_path, val_skim)

        vocab = set.union(
            self.train_corpus.vocab(),
            self.val_corpus.vocab(),
        )

        VECTORS.set_vocab(vocab)

        self.model = Model(*args, **kwargs)

        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam(params, lr=lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self, epochs=10, batch_size=20, eval_every=1000):

        for epoch in range(epochs):

            print(f'\nEpoch {epoch}')

            batches = self.train_corpus.batches(batch_size)

            epoch_loss = []
            for i, batch in enumerate(tqdm(batches)):

                self.model.train()
                batch.shuffle()
                self.optimizer.zero_grad()

                yt = torch.cat(batch.sent_pos_tensors())
                yp = torch.cat(self.model(batch))

                loss = F.mse_loss(yp, yt)
                loss.backward()

                self.optimizer.step()

                epoch_loss.append(loss.item())

                if i % eval_every == 0:
                    print('Val KT: %f' % self.val_mean_kt())

            print('Loss: %f' % np.mean(epoch_loss))
            print('Val KT: %f' % self.val_mean_kt())

    def val_mean_kt(self):

        self.model.eval()

        kts = []
        for batch in tqdm(self.val_corpus.batches(20)):

            batch.shuffle()

            yts = batch.sent_pos_tensors()
            yps = self.model(batch)

            for yt, yp in zip(yts, yps):
                yt = np.argsort(yt.tolist())
                yp = np.argsort(yp.tolist())
                kt, _ = stats.kendalltau(yt, yp)
                kts.append(kt)

        print(yts[0], yps[0])

        return np.mean(kts)
