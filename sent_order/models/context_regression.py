

import torch
import os
import ujson
import attr

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

from sent_order.cuda import itype


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

    # Sort the tensor by size.
    x = x[torch.LongTensor(size_sort).type(itype)]

    # Sort sizes descending.
    sizes = np.array(sizes)[size_sort].tolist()

    x = pack_padded_sequence(x, sizes, batch_first)

    # Indexes to restore original order.
    reorder = torch.LongTensor(np.argsort(size_sort)).type(itype)

    return x, reorder


def read_abstracts(path):
    """Parse abstract JSON lines.
    """
    for path in glob(os.path.join(path, '*.json')):
        with open(path) as fh:
            for line in fh:
                yield Abstract.from_line(line)


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

    tokens = attr.ib()

    def index_var(self):
        idx = [VECTORS.stoi(s) for s in self.tokens]
        idx = torch.LongTensor(idx)
        return Variable(idx).type(itype)


@attr.s
class Paragraph:

    sents = attr.ib()

    @classmethod
    def read_arxiv(cls, path, scount=None):
        """Wrap parsed arXiv abstrcts as paragraphs.
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
            Sentence(s['token'])
            for s in json['sentences']
        ])

    def __len__(self):
        return len(self.sents)


@attr.s
class Batch:

    grafs = attr.ib()

    def index_var(self, pad=50):
        """Token indexes, flattened to 1d series.
        """
        return pad_and_stack([
            sent.index_var()
            for graf in self.grafs
            for sent in graf.sents
        ])


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

        return list(vocab)

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

    def forward(self, batch):
        """Pad, pack, encode, reorder.

        Args:
            contexts (list of Variable): Encoded sentences for each graf.
        """
        x, sizes = batch.index_var()

        x = self.embeddings(x)

        x, reorder = pack(x, sizes)

        _, (hn, _) = self.lstm(x)

        # Cat forward + backward hidden layers.
        out = hn.transpose(0, 1).contiguous().view(hn.data.shape[1], -1)

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
        # self.graf_encoder = Encoder(se_dim, ge_dim)
        # self.regressor = Regressor(ge_dim, lin_dim)


class Trainer:

    def __init__(self, train_path, skim=None, lr=1e-3, *args, **kwargs):

        self.train_corpus = Corpus(train_path, skim)

        VECTORS.set_vocab(self.train_corpus.vocab())

        self.model = Model(*args, **kwargs)

        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam(params, lr=lr)

        # TODO: CUDA

    def train(self, epochs=10, batch_size=20):

        for epoch in range(epochs):

            print(f'\nEpoch {epoch}')

            self.model.train()

            epoch_loss = []
            for batch in self.train_corpus.batches(batch_size):

                self.optimizer.zero_grad()

                yt, yp = self.train_batch(batch)

                loss = F.mse_loss(yp, yt)
                loss.backward()

                self.optimizer.step()

                epoch_loss.append(loss.data[0])

            print('Loss: %f' % np.mean(epoch_loss))
            # TODO: eval

    def train_batch(self, batch):
        sents = self.model.sent_encoder(batch)
        print(sents)
        # get list of word index tensors for sents
        # encode sents
        # regroup by graf
        # make training pairs
