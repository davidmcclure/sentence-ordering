

import numpy as np

import attr

from cached_property import cached_property
from collections import defaultdict, Counter

from scipy import stats


@attr.s
class Metrics:

    gold_pred = attr.ib()

    @cached_property
    def len_counts(self):
        """Sentence count -> count.
        """
        lengths = map(len, [g for g, _ in self.gold_pred])
        return Counter(list(lengths))

    @cached_property
    def kts_by_len(self):
        """Kendall's tau by sentence count.
        """
        kts = defaultdict(list)

        for gold, pred in self.gold_pred:
            kt, _ = stats.kendalltau(gold, pred)
            kts[len(gold)].append(kt)

        return kts

    @cached_property
    def avg_kt_by_len(self):
        """Average KT for each sentence count.
        """
        return {
            count: sum(kts) / len(kts)
            for count, kts in self.kts_by_len.items()
        }

    @cached_property
    def all_kts(self):
        """Set of all KT scores.
        """
        return [kt for kts in self.kts_by_len.values() for kt in kts]

    @cached_property
    def overall_kt(self):
        """Overall average KT.
        """
        return sum(self.all_kts) / len(self.all_kts)

    @cached_property
    def perfect_order_pct_by_len(self):
        """Percent perfect order by sentence count.
        """
        perfect = Counter()

        for gold, pred in self.gold_pred:
            if gold == pred:
                perfect[len(gold)] += 1

        return {
            slen: perfect.get(slen, 0) / self.len_counts[slen]
            for slen in self.len_counts
        }

    @cached_property
    def overall_perfect_order_pct(self):
        """Percent perfect order overall.
        """
        perfect = sum([1 for g, p in self.gold_pred if g==p])
        return perfect / len(self.gold_pred)
