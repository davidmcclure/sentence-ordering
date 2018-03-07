

import torch
import os
import ujson
import attr

from torchtext.vocab import Vectors
from torch import nn

from cached_property import cached_property
from glob import glob
from boltons.iterutils import chunked_iter
from tqdm import tqdm
from itertools import islice


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

    @cached_property
    def indexes(self):
        return [VECTORS.stoi(s) for s in self.tokens]


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

    # def index_var_1d(self, perm=None, pad=None):
        # """Token indexes, flattened to 1d series.
        # """
        # perm = perm or range(len(self.sents))

        # idx = [ti for si in perm for ti in self.sents[si].indexes]
        # idx = Variable(torch.LongTensor(idx)).type(itype)

        # if pad:
            # idx = F.pad(idx, (0, pad-len(idx)))

        # return idx

    # def index_var_2d(self, pad=50):
        # """Token indexes, flattened to 1d series.
        # """
        # idx = []
        # for sent in self.sents:
            # sidx = Variable(torch.LongTensor(sent.indexes)).type(itype)
            # sidx = F.pad(sidx, (0, pad-len(sidx)))
            # idx.append(sidx)

        # return torch.stack(idx)


@attr.s
class Batch:

    grafs = attr.ib()

    # def index_var_2d(self, *args, **kwargs):
        # """Stack graf index tensors.
        # """
        # return torch.stack([
            # g.index_var_2d(*args, **kwargs)
            # for g in self.grafs
        # ])


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


class Encoder(nn.Module):

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

    def forward(self, x, pad_size):
        """Pad, pack, encode, reorder.

        Args:
            contexts (list of Variable): Encoded sentences for each graf.
        """
        # Pad, pack, encode.
        x, reorder = pad_and_pack(x, pad_size)
        _, (hn, _) = self.lstm(x)

        # Cat forward + backward hidden layers.
        # TODO: Is this wrong?
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

        self.sent_encoder = Encoder(VECTORS.loader.dim, se_dim)
        self.graf_encoder = Encoder(se_dim, ge_dim)
        self.regressor = Regressor(ge_dim, lin_dim)


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
            for batch in self.train_corpus.batches():

                self.optimizer.zero_grad()

                yt, yp = self.train_batch(batch)

                loss = F.mse_loss(yp, yt)
                loss.backward()

                self.optimizer.step()

                epoch_loss.append(loss.data[0])

            print('Loss: %f' % np.mean(epoch_loss))
            # TODO: eval
