

import os
import attr
import re
import random

import numpy as np

from itertools import groupby
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

    def truncate_sents_random(self, max_sents=50):
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
