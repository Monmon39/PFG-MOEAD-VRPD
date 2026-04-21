# metric/hv.py

from typing import List, Tuple
from pfg_moead_vrpd_ver2.model.solution import Solution


def hypervolume(
    pf: List[Solution],
    ref_point: Tuple[float, float] | None = None
) -> float:
    """
    Hypervolume for 2-objective MINIMIZATION
    Objectives:
        f1 = makespan
        f2 = carbonEmission
    """

    if not pf:
        return 0.0

    # Pareto filter (safety)
    front = []
    for s in pf:
        if not any(t is not s and t.dominates(s) for t in pf):
            front.append(s)

    if not front:
        return 0.0

    # Reference point
    if ref_point is None:
        max_m = max(s.makespan for s in front)
        max_c = max(s.carbonEmission for s in front)
        ref_point = (max_m * 1.1, max_c * 1.1)

    # Sort by makespan ASC
    front.sort(key=lambda s: s.makespan)

    hv = 0.0
    prev_m = ref_point[0]

    # Accumulate rectangles (right → left)
    for s in reversed(front):
        width = prev_m - s.makespan
        height = ref_point[1] - s.carbonEmission

        if width > 0 and height > 0:
            hv += width * height

        prev_m = s.makespan

    return hv
