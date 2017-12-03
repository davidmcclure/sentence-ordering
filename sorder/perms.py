

import numpy as np

from rpy2.robjects.packages import importr
from rpy2.robjects import r

per_mallows = importr('PerMallows')


class PermAtDist:

    def __init__(self, gc_count=1000):
        self.gc_count = gc_count
        self.calls = 0

    def __call__(self, size, dist):
        """Draw a random permutation as a given KT distance.
        """
        perm = per_mallows.rdist(1, size, dist)
        # 0-index
        perm = np.array(list(perm), dtype=int) - 1
        self._gc()
        return perm.tolist()

    def _gc(self):
        """rdist() leaks, so force GC in R every N calls.
        """
        self.calls += 1
        if self.calls % self.gc_count == 0:
            r('gc()')


perm_at_dist = PermAtDist()


def max_perm_dist(size):
    """Maximum KT distance for sequence of given size.
    """
    return int(per_mallows.maxi_dist(size)[0])


def sample_uniform_perms(size, n=10):
    """Sample N perms, uniformly distributed across the (-1, 1) KT interval.
    """
    max_dist = max_perm_dist(size)
    dists = np.linspace(0, max_dist, n, dtype=int)
    return [perm_at_dist(size, int(d)) for d in dists]
