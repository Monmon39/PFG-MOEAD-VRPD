# metric/cmetric.py

from typing import List
from pfg_moead_vrpd_ver2.model.solution import Solution


def c_metric(A: List[Solution], B: List[Solution]) -> float:
    """
    C(A, B):
    Percentage of solutions in B dominated by at least one solution in A
    """

    if not A or not B:
        return 0.0

    dominated = 0

    for sb in B:
        if any(sa.dominates(sb) for sa in A):
            dominated += 1

    return dominated / len(B)
