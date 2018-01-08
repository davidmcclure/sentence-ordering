

import numpy as np
import random


def max_perm_dist(size):
    """Maximum KT distance for sequence of given size.
    """
    return int(size * (size-1) / 2)


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


def sample_uniform_perms(size, maxn=10):
    """Sample N perms, uniformly distributed across the (-1, 1) KT interval.
    """
    max_dist = max_perm_dist(size)

    # At most, 1 sample for each possible distance.
    n = min(maxn, max_dist+1)

    dists = np.linspace(0, max_dist, n, dtype=int)

    perms = [
        random_perm_at_dist(size, int(d))
        for d in dists
    ]

    kts = ((1 - dists / max_dist) * 2) - 1

    return perms, kts
