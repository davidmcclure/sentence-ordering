

import numpy as np

import os
import attr
import csv

from wordfreq import top_n_list
from cached_property import cached_property
from itertools import islice
from gensim.models import KeyedVectors


VECTORS_PATH = os.path.join(os.path.dirname(__file__), 'data/vectors.bin')


@attr.s
class LazyVectors:

    path = attr.ib()

    @classmethod
    def read(cls):
        return cls(VECTORS_PATH)

    @cached_property
    def model(self):
        return KeyedVectors.load(self.path)

    @cached_property
    def vocab(self):
        """Model vocabulary (possibly a subset)
        """
        return list(self.model.vocab.keys())

    @cached_property
    def vocab_index(self):
        """Token -> index.
        """
        return {t: i for i, t in enumerate(self.vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab) + 2

    @property
    def vector_dim(self):
        return self.model.vector_size

    @cached_property
    def syn0(self):
        """Slice out embedding matrix for cropped vocab.
        """
        indexes = [self.model.vocab[t].index for t in self.vocab]

        return self.model.syn0[indexes]

    def build_weights(self):
        """Prepend a zeros row for <UNK>.
        """
        zeros = np.zeros(self.vector_dim)

        # Padding, <UNK>, vocab.
        return np.vstack([zeros, zeros, self.syn0])

    def weights_index(self, token):
        """Get the index of a word in the weights matrix.
        """
        return (
            # Since paddig / <UKN> are 0-1.
            self.vocab_index[token.lower()] + 2
            if token in self.vocab_index else 1
        )
