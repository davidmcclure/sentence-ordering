

import attr
import os
import ujson

import numpy as np

from glob import glob
from itertools import islice
from tqdm import tqdm


def read_abstracts(path):
    """Parse abstract JSON lines.
    """
    for path in glob(os.path.join(path, '*.json')):
        with open(path) as fh:
            for line in fh:
                json = ujson.loads(line.strip())
                yield Abstract.from_json(json)


@attr.s
class Sentence:

    tokens = attr.ib()

    def tensor(self, dim=300, pad=50):
        """Stack word vectors, padding zeros on left.
        """
        x = [vectors[t] for t in self.tokens if t in vectors]
        x += [np.zeros(dim)] * pad
        x = x[:pad]
        x = list(reversed(x))
        x = np.array(x)
        x = torch.from_numpy(x)
        x = x.float()
        return x


@attr.s
class Abstract:

    sentences = attr.ib()

    @classmethod
    def from_json(cls, json):
        """Pull out raw token series.
        """
        return cls([Sentence(s['token']) for s in json['sentences']])

    def tensor(self):
        """Stack sentence tensors.
        """
        tensors = [s.tensor() for s in self.sentences]
        return torch.stack(tensors)


class Corpus:

    def __init__(self, path, skim=None):
        """Load abstracts into memory.
        """
        reader = read_abstracts(path)

        if skim:
            reader = islice(reader, skim)

        self.abstracts = list(tqdm(reader, total=skim))
