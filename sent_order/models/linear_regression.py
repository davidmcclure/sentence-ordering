

import numpy as np

import os
import ujson
import attr

from collections import Counter, defaultdict
from itertools import islice
from boltons.iterutils import windowed
from tqdm import tqdm_notebook
from glob import glob

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


def read_abstracts(path):
    """Parse abstract JSON lines.
    """
    for path in glob(os.path.join(path, '*.json')):
        with open(path) as fh:
            for line in fh:
                yield Abstract.from_line(line)


@attr.s
class Sentence:

    tokens = attr.ib()

    def ngrams(self, n=1):
        for ng in windowed(self.tokens, n):
            yield '_'.join(ng)

    def ngram_counts(self, vocab, maxn=3):
        for n in range(1, maxn+1):
            counts = Counter(self.ngrams(n))
            for k, v in counts.items():
                if k in vocab:
                    yield f'_{k}', v

    def word_count(self):
        return len(self.tokens)

    def _features(self, vocab):
        yield from self.ngram_counts(vocab)
        yield 'word_count', self.word_count()

    def features(self, vocab):
        return dict(self._features(vocab))


@attr.s
class Abstract:

    sentences = attr.ib()

    @classmethod
    def from_line(cls, line):
        """Parse JSON, take tokens.
        """
        json = ujson.loads(line.strip())

        return cls([
            Sentence(s['token'])
            for s in json['sentences']
        ])

    def xy(self, vocab):
        for i, sent in enumerate(self.sentences):
            x = sent.features(vocab)
            y = i / (len(self.sentences)-1)
            yield x, y


class Corpus:

    def __init__(self, path, skim=None):
        """Load abstracts into memory.
        """
        reader = read_abstracts(path)

        if skim:
            reader = islice(reader, skim)

        self.abstracts = list(tqdm_notebook(reader, total=skim))

    def xy(self, vocab):
        for abstract in tqdm_notebook(self.abstracts):
            yield from abstract.xy(vocab)

    def ngram_counts(self, n):
        counts = defaultdict(lambda: 0)
        for ab in tqdm_notebook(self.abstracts):
            for sent in ab.sentences:
                for ngram in sent.ngrams(n):
                    counts[ngram] += 1
        return Counter(counts)

    def most_common_ngrams(self, n, depth):
        counts = self.ngram_counts(n)
        return set([k for k, _ in counts.most_common(depth)])
