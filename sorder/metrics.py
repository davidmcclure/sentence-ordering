

import attr

from collections import defaultdict
from cached_property import cached_property


@attr.s
class Metrics:

    gold_pred = attr.ib()

    @cached_property
    def kts_by_sent_count(self):
        """Kendall's tau by sentence length.
        """
        kts = defaultdict(list)

        for gold, pred in self.gold_pred:
            kt, _ = stats.kendalltau(gold, pred)
            kts[len(gold)].append(kt)

        return kts

    @cached_property
    def avg_kt_by_sent_count(self):
        """Average KT for each sentence count.
        """
        return {
            count: sum(kts) / len(kts)
            for count, kts in self.kts_by_sent_count
        }

    @cached_property
    def all_kts(self):
        """Set of all KT scores.
        """
        by_count = self.kts_by_sent_count.values()

        return [kt for kts in by_count for kt in kts]

    @cached_property
    def avg_kt(self):
        """Overall average KT.
        """
        return sum(self.all_kts) / len(self.all_kts)
