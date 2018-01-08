

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


def sample_uniform_perms(size, maxn=10):
    """Sample N perms, uniformly distributed across the (-1, 1) KT interval.
    """
    max_dist = max_perm_dist(size)

    # At most, 1 sample for each possible distance.
    n = min(maxn, max_dist+1)

    dists = np.linspace(0, max_dist, n, dtype=int)

    perms = [
        rdist(1, size, int(d))[0].astype(int)
        for d in dists
    ]

    return perms, dists / max_dist


def random_perm_at_dist(size, dist):
    """Generate a random permutation at a given swap distance.
    """
    perm = list(range(size))

    # Left indexes of correctly-ordered pairs.
    ordered = set(range(size-1))

    for _ in range(dist):

        i1 = random.sample(ordered, 1)[0]
        i2 = i1 + 1

        perm[i1], perm[i2] = perm[i2], perm[i1]

        ordered.remove(i1)

        if i2+1 < len(perm) and perm[i2] < perm[i2+1]:
            ordered.add(i2)

        if i1 > 0 and perm[i1-1] < perm[i1]:
            ordered.add(i1-1)

    return perm
