

import attr
import re
import random
import os

from collections import defaultdict
from itertools import groupby
from boltons.iterutils import pairwise, chunked
from glob import glob
from cached_property import cached_property
from tqdm import tqdm
from textblob import TextBlob

from . import utils


@attr.s
class Token:
    doc_slug = attr.ib()
    doc_part = attr.ib()
    text = attr.ib()
    sent_index = attr.ib()
    clusters = attr.ib()


class Document:

    @classmethod
    def from_text(cls, doc_slug, doc_part, text):
        """Tokenize raw text input.
        """
        tokens = []
        for i, sent in enumerate(TextBlob(text).sentences):
            for token in sent.tokens:

                tokens.append(Token(
                    doc_slug=doc_slug,
                    doc_part=doc_part,
                    text=str(token),
                    sent_index=i,
                    clusters=set(),
                ))

        return cls(tokens)

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

                cid = utils.parse_int(part)

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

    @cached_property
    def coref_id_to_index_range(self):
        """Map coref id -> token indexes, grouped by mention.
        """
        id_idx = defaultdict(list)

        for i, token in enumerate(self.tokens):
            for cid in token.clusters:

                spans = id_idx[cid]

                if len(spans) and spans[-1][-1] == i-1:
                    spans[-1].append(i)

                else:
                    spans.append([i])

        return id_idx

    @cached_property
    def coref_id_to_i1i2(self):
        """Map coref id -> (start, end) span indexes.
        """
        return {
            cid: [(s[0], s[-1]) for s in spans]
            for cid, spans in self.coref_id_to_index_range.items()
        }

    @cached_property
    def i1i2_to_ant_i1i2(self):
        """Map span (start, end) -> list of (start, end) of antecedents.
        """
        return {
            span: set(spans[:i+1])
            for _, spans in self.coref_id_to_i1i2.items()
            for i, span in enumerate(spans[1:])
        }

    def to_conll_format(self, clusters):
        """Generate CONLL output format.
        """
        tags = [[] for _ in range(len(self))]

        for cid, cluster in enumerate(clusters):
            for i1, i2 in cluster:

                if i1 == i2:
                    tags[i1].append(f'({cid})')

                else:
                    tags[i1].append(f'({cid}')
                    tags[i2].append(f'{cid})')

        tags = ['|'.join(t) if t else '-' for t in tags]

        token_tag = zip(self.tokens, tags)

        return conll_tpl.render(doc=self, token_tag=token_tag)


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

    def sent_pair_tokens(self):
        """Generate sentence pairs.
        """
        for doc in self.documents:
            for s1, s2 in pairwise(doc.sents()):
                yield [t.text for t in s1], [t.text for t in s2]
