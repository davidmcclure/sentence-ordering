

import re

from itertools import groupby
from functools import reduce


def parse_int(text):
    """Parse an integer out of a string.
    """
    matches = re.findall('[0-9]+', text)
    return int(matches[0]) if matches else None


def remove_consec_dupes(seq):
    """Remove consecutive duplicates in a list.
    [1,1,2,2,3,3] -> [1,2,3]
    """
    return [x[0] for x in groupby(seq)]


# TODO: Test.
def regroup_indexes(seq, size_fn):
    """Given a sequence A that contains items of variable size, provide a list
    of indexes that, when iterated as pairs, will slice a flat sequence B into
    groups with sizes that correspond to the items in A.

    Args:
        seq (iterable)
        size_fn (func): Provides size of an individual item in `seq`.

    Returns: list<int>
    """
    return reduce(lambda ix, i: (*ix, ix[-1] + size_fn(i)), seq, (0,))
