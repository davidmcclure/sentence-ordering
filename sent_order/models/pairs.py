

import random
import numpy as np

from tqdm import tqdm
from boltons.iterutils import chunked

import torch
from torch.nn import functional as F
from torch import nn, optim

from ..cuda import CUDA, itype
from ..embeds import WordEmbedding
from ..conll import Corpus
from .. import utils


class Classifier(nn.Module):

    def __init__(self, vocab, lstm_dim, hidden_dim, lstm_num_layers=1):

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

        self.predict = nn.Sequential(
            nn.Linear(lstm_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.LogSoftmax(1),
        )

    def forward(self, pairs):
        """Encode document tokens, predict.
        """
        x, sizes = utils.pad_right_and_stack([
            self.embeddings.tokens_to_idx(s1 + s2)
            for s1, s2 in pairs
        ])

        x = self.embeddings(x)
        x = self.dropout(x)

        x, reorder = utils.pack(x, sizes)

        _, (hn, _) = self.lstm(x)
        hn = self.dropout(hn)

        # Cat forward + backward hidden layers.
        x = torch.cat([hn[0,:,:], hn[1,:,:]], dim=1)
        x = x[reorder]

        return self.predict(x)

    def train_batch(self, pairs):
        """Generate correct / reversed examples, predict.

        Returns: y pred, y true
        """
        x, y = [], []
        for s1, s2 in pairs:

            # Correct.
            x.append((s1, s2))
            y.append(0)

            # Reversed.
            x.append((s2, s1))
            y.append(1)

        y = torch.LongTensor(y).type(itype)

        return self(x), y


class Trainer:

    def __init__(self, train_path, dev_path, lr=1e-3, lstm_dim=500,
        hidden_dim=200, batch_size=20):

        self.batch_size = batch_size

        self.train_corpus = Corpus.from_combined_file(train_path)
        self.dev_corpus = Corpus.from_combined_file(dev_path)

        vocab = set.union(
            self.train_corpus.vocab(),
            self.dev_corpus.vocab())

        self.model = Classifier(vocab, lstm_dim, hidden_dim)

        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam(params, lr=lr)

        if CUDA:
            self.model.cuda()

    def train(self, epochs=10):
        for epoch in range(epochs):
            self.train_epoch(epoch)

    def train_epoch(self, epoch,):
        """Train a single epoch.
        """
        print(f'\nEpoch {epoch}')

        self.model.train()

        pairs = self.train_corpus.sent_pairs()
        batches = chunked(pairs, self.batch_size)

        epoch_loss = []
        for batch in tqdm(batches):

            self.optimizer.zero_grad()

            yp, yt = self.model.train_batch(batch)

            loss = F.nll_loss(yp, yt)
            loss.backward()

            self.optimizer.step()

            epoch_loss.append(loss.item())

        print('Loss: %f' % np.mean(epoch_loss))
        print('Dev accuracy: %f' % self.dev_accuracy())

    def dev_accuracy(self):
        """Predict dev pairs.
        """
        self.model.eval()

        pairs = self.dev_corpus.sent_pairs()
        batches = chunked(pairs, self.batch_size)

        correct, total = 0, 0
        for batch in tqdm(batches):

            yps, yts = self.model.train_batch(batch)

            for yp, yt in zip(yps, yts):
                if yp.argmax() == yt:
                    correct += 1

            total += len(yps)

        return correct / total
