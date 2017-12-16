

import attr

from collections import defaultdict


@attr.s
class Metrics:

    gold_pred = attr.ib()

    def kts_by_sent_count(self):
        """Kendall's tau by sentence length.
        """
        kts = defaultdict(list)

        for gold, pred in self.gold_pred:
            kt, _ = stats.kendalltau(gold, pred)
            kts[len(gold)].append(kt)

        return kts
