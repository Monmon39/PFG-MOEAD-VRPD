import random
from typing import List, Dict, Tuple
from pfg_moead_vrpd_ver2.model.solution import Solution


# BUILD PFG
def build_pfg(ep: List[Solution], grid_size: int = 10) -> List[Solution]:

    if len(ep) < 2:
        return [s.copy() for s in ep]

    # f_min, f_max
    f1_vals = [s.makespan for s in ep]
    f2_vals = [s.carbonEmission for s in ep]

    f1_min, f1_max = min(f1_vals), max(f1_vals)
    f2_min, f2_max = min(f2_vals), max(f2_vals)

    # Tính độ rộng lưới d_j
    # Thêm 1e-9 để tránh lỗi chia cho 0 hoặc làm tròn sai số
    dx = (f1_max - f1_min) / grid_size + 1e-9
    dy = (f2_max - f2_min) / grid_size + 1e-9

    # Gán chỉ số lưới cho từng nghiệm trong EP
    # solution_grids lưu tọa độ (gx, gy) của từng nghiệm tương ứng với index trong ep
    solution_grids = []
    for s in ep:
        gx = int((s.makespan - f1_min) / dx)
        gy = int((s.carbonEmission - f2_min) / dy)
        solution_grids.append((min(grid_size - 1, gx), min(grid_size - 1, gy)))

    pfg_indices = set()

    # Tìm gmin theo từng dải mục tiêu

    # Quét cho mục tiêu f1
    # Xét từng dải của mục tiêu f2
    for j in range(grid_size):
        s_indices = [idx for idx, grid in enumerate(solution_grids) if grid[1] == j]

        if s_indices:
            gmin = min(solution_grids[idx][0] for idx in s_indices)
            for idx in s_indices:
                if solution_grids[idx][0] == gmin:
                    pfg_indices.add(idx)

    # Quét cho mục tiêu f2
    # Xét từng dải của mục tiêu f1
    for j in range(grid_size):
        s_indices = [idx for idx, grid in enumerate(solution_grids) if grid[0] == j]

        if s_indices:
            gmin = min(solution_grids[idx][1] for idx in s_indices)
            for idx in s_indices:
                if solution_grids[idx][1] == gmin:
                    pfg_indices.add(idx)

    pfg = [ep[idx].copy() for idx in pfg_indices]

    return pfg


# SELECT FROM PFG
def pfg_select(ep: List[Solution],
               target_size: int,
               rnd: random.Random) -> List[Solution]:

    if not ep:
        return []

    if len(ep) <= target_size:
        return [s.copy() for s in ep]

    return [s.copy() for s in rnd.sample(ep, target_size)]

# SAMPLE ONE SOLUTION
def sample_from_pfg(pfg_pool: List[Solution],
                    rnd: random.Random) -> Solution:
    """
    Sampling strategy:
    - 50% best makespan
    - 50% best carbon
    """
    if not pfg_pool:
        return None

    if len(pfg_pool) == 1:
        return pfg_pool[0].copy()

    if rnd.random() < 0.5:
        best = min(pfg_pool, key=lambda s: s.makespan)
    else:
        best = min(pfg_pool, key=lambda s: s.carbonEmission)

    return best.copy()