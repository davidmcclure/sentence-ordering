

import os
import attr
import re
import random

import numpy as np
import networkx as nx

from collections import defaultdict
from itertools import islice, groupby
from boltons.iterutils import pairwise, chunked, windowed
from tqdm import tqdm
from cached_property import cached_property
from jinja2 import Environment, PackageLoader
from subprocess import Popen, PIPE

import torch
from torch import nn, optim
from torch.nn import functional as F

from .. import utils
from ..embeds import WordEmbedding
from ..conll import Corpus
from ..cuda import itype, ftype, CUDA


jinja_env = Environment(loader=PackageLoader('sent_order', 'templates'))
conll_tpl = jinja_env.get_template('conll.txt')


class DistanceEmbedding(nn.Embedding):

    def __init__(self, embed_dim=20):
        self.bins = (1, 2, 3, 4, 8, 16, 32, 64)
        return super().__init__(len(self.bins)+1, embed_dim)

    def dtoi(self, d):
        return sum([True for i in self.bins if d >= i])

    def forward(self, ds):
        """Map distances to indexes, embed.
        """
        idx = torch.LongTensor([self.dtoi(d) for d in ds]).type(itype)
        return super().forward(idx)


class DocEncoder(nn.Module):

    def __init__(self, vocab, lstm_dim, lstm_num_layers=1):

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

    @property
    def embed_dim(self):
        return self.embeddings.weight.shape[1]

    def forward(self, tokens):
        """BiLSTM over document tokens.
        """
        # TODO: Char CNN.
        x = self.embeddings.tokens_to_idx(tokens)
        x = x.unsqueeze(0)

        embeds = self.embeddings(x)
        embeds = self.dropout(embeds)

        states, _ = self.lstm(embeds)
        states = self.dropout(states)

        return embeds.squeeze(), states.squeeze()


class Scorer(nn.Module):

    def __init__(self, input_dim, hidden_dim=150):

        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.score(x)


@attr.s(frozen=True, repr=False)
class Span:

    # Parent document reference.
    doc = attr.ib()

    # Left / right token indexes.
    i1 = attr.ib()
    i2 = attr.ib()

    # Span embedding tensor.
    g = attr.ib(default=None)

    # Unary mention score, as tensor.
    sm = attr.ib(default=None)

    # List of candidate antecedent spans.
    yi = attr.ib(default=None)

    # Pairwise scores for each yi.
    sij = attr.ib(default=None)

    def __repr__(self):
        text = ' '.join([t.text for t in self.tokens])
        return 'Span<%d, %d, "%s">' % (self.i1, self.i2, text)

    @cached_property
    def tokens(self):
        return self.doc.tokens[self.i1:self.i2+1]

    @cached_property
    def i1i2(self):
        return (self.i1, self.i2)

    def sij_gold_indexes(self):
        """Get indexes of gold antecedents in the sij score tensor.
        """
        ant_i1i2s = self.doc.i1i2_to_ant_i1i2.get(self.i1i2, [])

        sij_idx = [
            i for i, span in enumerate(self.yi)
            if span.i1i2 in ant_i1i2s
        ]

        if not sij_idx:
            sij_idx = [len(self.sij)-1]

        return sij_idx


class SpanScorer(nn.Module):

    def __init__(self, state_dim, width_dim, gi_dim):
        super().__init__()
        self.attention = Scorer(state_dim)
        self.width_embeddings = DistanceEmbedding(width_dim)
        self.sm = Scorer(gi_dim)

    def forward(self, doc, embeds, states):
        """Generate spans, attend over LSTM states, form encodings.
        """
        # Get raw attention scores over LSTM states.
        attns = self.attention(states)

        # Slice out spans, build encodings.
        spans, gs = [], []
        for n in range(1, 11):
            for w in windowed(range(len(doc.tokens)), n):

                i1, i2 = w[0], w[-1]

                span_embeds = embeds[i1:i2+1]
                span_states = states[i1:i2+1]
                span_attns = attns[i1:i2+1]

                # Softmax over attention scores for span.
                attn = F.softmax(span_attns.squeeze(), dim=0)
                attn = sum(span_embeds * attn.view(-1, 1))

                # Left LSTM + right LSTM + attention + width.
                g = torch.cat([span_states[0], span_states[-1], attn])

                spans.append(Span(doc, i1, i2))
                gs.append(g)

        gs = torch.stack(gs)

        # Cat on width embeddings.
        widths = self.width_embeddings([s.i2-s.i1+1 for s in spans])
        gs = torch.cat([gs, widths], dim=1)

        # Unary scores for spans.
        sms = self.sm(gs).squeeze()

        # Set scores on spans.
        # TODO: Scalar sm_sort, for perf?
        spans = [
            attr.evolve(span, g=g, sm=sm)
            for span, g, sm in zip(spans, gs, sms)
        ]

        return spans


def prune_spans(spans, T, lbda=0.4):
    """Sort by score; remove overlaps, skim lamda*T; sort by start idx.
    """
    # Sory by unary score, descending.
    spans = sorted(spans, key=lambda s: s.sm, reverse=True)

    # Remove spans that overlap with higher-scoring spans.
    taken_idxs = set()
    pruned = []
    for span in spans:

        span_idxs = range(span.i1, span.i2+1)
        slots = [idx in taken_idxs for idx in span_idxs]
        slots = utils.remove_consec_dupes(slots)

        # Allow spans that overlap with nothing, or spans that are "supersets"
        # of previously-accepted spans - start to left, end to right.
        if len(slots) == 1 or slots == [False, True, False]:
            pruned.append(span)
            taken_idxs.update(span_idxs)

    # Take top lambda*T.
    pruned = pruned[:round(T*lbda)]

    # Order by left doc index.
    # TODO: Sort by i2 too?
    pruned = sorted(pruned, key=lambda s: s.i1)

    return pruned


class PairScorer(nn.Module):

    def __init__(self, dist_dim, gij_dim):
        super().__init__()
        self.dist_embeddings = DistanceEmbedding(dist_dim)
        self.sa = Scorer(gij_dim)

    def forward(self, spans):
        """Map span -> candidate antecedents, score pairs.
        """
        # Take up to K antecedents.
        spans = [
            attr.evolve(span, yi=spans[max(0, ix-250):ix])
            for ix, span in enumerate(spans)
        ]

        pairs, dists = [], []
        for i in spans:
            for j in i.yi:
                pairs.append(torch.cat([i.g, j.g, i.g*j.g]))
                dists.append(i.i1 - j.i1)

        if not pairs:
            raise RuntimeError('No candidate pairs after pruning.')

        pairs = torch.stack(pairs)

        # Cat on distance embeddings.
        dists = self.dist_embeddings(dists)
        pairs = torch.cat([pairs, dists], dim=1)

        # Get pairwise `sa` scores.
        # TODO: Batch for big eval inputs?
        sas = self.sa(pairs).view(-1)

        # Get indexes to re-group scores by i.
        sa_idx = utils.regroup_indexes(spans, lambda s: len(s.yi))

        spans_sij = []
        for span, (i1, i2) in zip(spans, pairwise(sa_idx)):

            # Build composite `sij` scores.
            sij = [
                (span.sm + yi.sm + sa).view(1)
                for yi, sa in zip(span.yi, sas[i1:i2])
            ]

            # Add epsilon score 0 as last element.
            epsilon = torch.FloatTensor([0]).type(ftype)
            sij = torch.cat([*sij, epsilon])

            spans_sij.append(attr.evolve(span, sij=sij))

        return spans_sij


class Coref(nn.Module):

    def __init__(self, vocab, lstm_dim=200, size_dim=20):

        super().__init__()

        self.encode_doc = DocEncoder(vocab, lstm_dim)

        # LSTM forward + back.
        state_dim = lstm_dim * 2

        # Left + right LSTM states, head attention, width.
        gi_dim = state_dim * 2 + self.encode_doc.embed_dim + size_dim

        # i, j, i*j, dist
        gij_dim = gi_dim * 3 + size_dim

        self.score_spans = SpanScorer(state_dim, size_dim, gi_dim)
        self.score_pairs = PairScorer(size_dim, gij_dim)

    def forward(self, doc):
        """Generate spans with yi / sij.
        """
        # LSTM over tokens.
        embeds, states = self.encode_doc(doc.token_texts())

        spans = self.score_spans(doc, embeds, states)
        spans = prune_spans(spans, len(doc))

        return self.score_pairs(spans)

    def predict(self, doc):
        """Given a doc, generate a set of grouped (i1,i2) mention clusters.
        """
        graph = nx.Graph()
        for span in self(doc):

            max_idx = span.sij.argmax().item()

            if max_idx < len(span.sij)-1:
                graph.add_edge(span.i1i2, span.yi[max_idx].i1i2)

        return list(nx.connected_components(graph))

    def dump_conll_preds(self, docs, path):
        """Write CONLL prediction file.
        """
        outputs = []
        for doc in tqdm(docs):
            clusters = self.predict(doc)
            outputs.append(doc.to_conll_format(clusters))

        with open(path, 'w') as fh:
            print('\n'.join(outputs), file=fh)


class Trainer:

    def __init__(self, train_path, dev_path, eval_script, pred_dir,
        checkpoint_dir, lr=1e-3):

        self.eval_script = eval_script
        self.pred_dir = pred_dir
        self.checkpoint_dir = checkpoint_dir
        self.dev_path = dev_path

        self.train_corpus = Corpus.from_combined_file(train_path)
        self.dev_corpus = Corpus.from_combined_file(dev_path)

        self.model = Coref(self.train_corpus.vocab())

        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam(params, lr=lr)

        if CUDA:
            self.model.cuda()

    def train(self, epochs=10, *args, **kwargs):
        for epoch in range(epochs):
            self.train_epoch(epoch, *args, **kwargs)

    def train_epoch(self, epoch, batch_size=100, eval_every=100):
        """Train a single epoch.
        """
        print(f'\nEpoch {epoch}')

        self.model.train()

        epoch_loss = []
        for i, doc in enumerate(tqdm(self.train_corpus.documents)):

            # Handle errors when model over-prunes.
            try:
                epoch_loss.append(self.train_doc(doc))
            except RuntimeError as e:
                print(e)

        print('Loss: %f' % np.mean(epoch_loss))

        self.checkpoint(epoch)
        print(self.eval_dev(epoch))

    def train_doc(self, doc):
        """Train a single doc.
        """
        # Train on random sub-docs.
        doc = doc.truncate_sents_random()

        self.optimizer.zero_grad()

        losses = []
        for span in self.model(doc):

            # Softmax over possible antecedents.
            pred = F.softmax(span.sij, dim=0)

            # Get indexes of correct antecedents.
            yt = span.sij_gold_indexes()

            # Sum mass assigned to correct antecedents.
            p = sum([pred[i] for i in yt])

            losses.append(p.log())

            # Print correct predictions.
            yp = pred.argmax().item()
            if yp != len(pred)-1 and yp in yt:
                print(span, span.yi[yp])

        loss = sum(losses) * -1
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

        return loss.item()

    def eval_dev(self, epoch, lines=11):
        """Dump dev predictions, eval.
        """
        path = os.path.join(self.pred_dir, f'dev.{epoch}.conll')

        self.model.dump_conll_preds(self.dev_corpus.documents, path)

        p = Popen([self.eval_script, 'all', self.dev_path, path], stdout=PIPE)
        output, err = p.communicate()

        return '\n'.join(output.decode().splitlines()[-lines:])

    def checkpoint(self, epoch):
        """Save model.
        """
        path = os.path.join(self.checkpoint_dir, f'coref.{epoch}.bin')
        torch.save(self.model, path)
