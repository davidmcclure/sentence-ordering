

import numpy as np

from rpy2.robjects.packages import importr

per_mallows = importr('PerMallows')


def perm_at_dist(size, dist):
    """Draw a random permutation as a given KT distance.
    """
    perm = per_mallows.rdist(1, size, dist)
    return list(map(int, perm))


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
