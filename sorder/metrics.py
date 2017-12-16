

import numpy as np

import attr
import warnings

from cached_property import cached_property
from collections import defaultdict, Counter, OrderedDict
from scipy import stats

from .utils import sort_by_key


warnings.simplefilter("ignore")


class Metrics:

    def __init__(self, gold_pred):
        self.gold_pred = gold_pred

    @cached_property
    def len_counts(self):
        """Sentence count -> count.
        """
        lengths = map(len, [g for g, _ in self.gold_pred])
        return Counter(list(lengths))

    def perfect_order_pct_by_len(self, max_len=10):
        """Percent perfect order by sentence count.
        """
        perfect = Counter()

        for gold, pred in self.gold_pred:
            if gold == pred:
                perfect[len(gold)] += 1

        return sort_by_key({
            slen: perfect.get(slen, 0) / self.len_counts[slen]
            for slen in self.len_counts
            if slen <= max_len
        })

    def overall_perfect_order_pct(self):
        """Percent perfect order overall.
        """
        perfect = sum([1 for g, p in self.gold_pred if g==p])
        return perfect / len(self.gold_pred)

    @cached_property
    def kts_by_len(self):
        """Kendall's tau by sentence count.
        """
        kts = defaultdict(list)

        for gold, pred in self.gold_pred:
            kt, _ = stats.kendalltau(gold, pred)
            kts[len(gold)].append(kt)

        return kts

    def avg_kt_by_len(self, max_len=10):
        """Average KT for each sentence count.
        """
        return sort_by_key({
            slen: sum(kts) / len(kts)
            for slen, kts in self.kts_by_len.items()
            if slen <= max_len
        })

    @cached_property
    def all_kts(self):
        """Set of all KT scores.
        """
        return [kt for kts in self.kts_by_len.values() for kt in kts]

    def overall_kt(self):
        """Overall average KT.
        """
        return sum(self.all_kts) / len(self.all_kts)

    def positional_accuracy_pct_by_len(self, max_len=10):
        """Percentage of sentences in the right slot.
        """
        correct = Counter()

        for gold, pred in self.gold_pred:
            correct[len(gold)] += np.equal(gold, pred).sum()

        return sort_by_key({
            slen: correct.get(slen, 0) / (self.len_counts[slen] * slen)
            for slen in self.len_counts
            if slen <= max_len
        })

    def overall_positional_accuracy_pct(self):
        """Positional accuracy for all sentences.
        """
        c, t = 0, 0

        for gold, pred in self.gold_pred:
            c += np.equal(gold, pred).sum()
            t += len(gold)

        return c / t
