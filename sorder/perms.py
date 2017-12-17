

import numpy as np
import random

from rpy2.robjects.packages import importr
from rpy2.robjects import r

per_mallows = importr('PerMallows')


class RDist:

    def __init__(self, gc_count=1000):
        self.gc_count = gc_count
        self.calls = 0

    def __call__(self, n, size, dist):
        """Draw a random permutation as a given KT distance.
        """
        res = per_mallows.rdist(n, size, dist)

        # 0-index
        res = np.array(res) - 1
        self._gc()

        return res

    def _gc(self):
        """rdist() leaks, so force GC in R every N calls.
        """
        self.calls += 1
        if self.calls % self.gc_count == 0:
            r('gc()')


rdist = RDist()


def max_perm_dist(size):
    """Maximum KT distance for sequence of given size.
    """
    return int(per_mallows.maxi_dist(size)[0])


def sample_perms(size, dist, n=10):
    """Drawn N permutations at a uniformly selected KT distance.

    Args:
        size (int): Seq length.
        dist (float): 0-1, where 0 is correct order, 1 is max perm dist.
    """
    dist = max_perm_dist(size) * dist

    return rdist(n, size, dist)
