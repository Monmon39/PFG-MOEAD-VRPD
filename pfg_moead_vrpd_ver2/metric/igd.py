import math
from typing import List
from pfg_moead_vrpd_ver2.model.solution import Solution


def igd(approx_pf: List[Solution], true_pf: List[Solution]) -> float:
    """
    IGD = (1 / |PF*|) * sum_{x in PF*} min_{y in PF} distance(x, y)

    approx_pf : Pareto front thu được
    true_pf   : Pareto front tham chiếu (union hoặc best-known)
    """
    if not approx_pf or not true_pf:
        return float("inf")

    total = 0.0
    for t in true_pf:
        best = float("inf")
        for a in approx_pf:
            d = math.sqrt(
                (t.makespan - a.makespan) ** 2 +
                (t.carbonEmission - a.carbonEmission) ** 2
            )
            if d < best:
                best = d
        total += best

    return total / len(true_pf)
