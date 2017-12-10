

import os
import attr

from cached_property import cached_property
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

    def __getitem__(self, key):
        return self.model[key]

    def __contains__(self, key):
        return key in self.model
