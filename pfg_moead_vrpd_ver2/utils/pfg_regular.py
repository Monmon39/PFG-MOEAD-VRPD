import random
from typing import List
from pfg_moead_vrpd_ver2.model.solution import Solution


def build_pfg_regular(ep: List[Solution], grid_size: int = 10) -> List[Solution]:
    """
    Phiên bản Regular Grid: Chia không gian mục tiêu thành lưới grid_size x grid_size.
    Mỗi ô lưới chỉ chọn tối đa một nghiệm đại diện, bỏ qua quy tắc Boundary Selection.
    """
    if len(ep) < 2:
        return [s.copy() for s in ep]

    # Xác định f_min, f_max để định nghĩa không gian lưới
    f1_vals = [s.makespan for s in ep]
    f2_vals = [s.carbonEmission for s in ep]

    f1_min, f1_max = min(f1_vals), max(f1_vals)
    f2_min, f2_max = min(f2_vals), max(f2_vals)

    # Tính độ rộng lưới (dx, dy)
    # Thêm 1e-9 để tránh lỗi
    dx = (f1_max - f1_min) / grid_size + 1e-9
    dy = (f2_max - f2_min) / grid_size + 1e-9

    # Gom nhóm các nghiệm vào các ô (cells)
    # Sử dụng dictionary với key là tọa độ ô (gx, gy)
    grid_cells = {}

    for s in ep:
        # Tính toán tọa độ ô mà nghiệm s thuộc về
        gx = int((s.makespan - f1_min) / dx)
        gy = int((s.carbonEmission - f2_min) / dy)

        # Đảm bảo index không vượt quá giới hạn [0, grid_size - 1]
        cell_coords = (min(grid_size - 1, gx), min(grid_size - 1, gy))

        # Chiến thuật "Regular": Nếu ô này chưa có nghiệm, thì nhận nghiệm này làm đại diện.

        if cell_coords not in grid_cells:
            grid_cells[cell_coords] = s.copy()

    # 4. Trả về danh sách các nghiệm đại diện từ các ô lưới
    return list(grid_cells.values())


def pfg_select(ep: List[Solution],
               target_size: int,
               rnd: random.Random) -> List[Solution]:
    if not ep:
        return []

    if len(ep) <= target_size:
        return [s.copy() for s in ep]

    return [s.copy() for s in rnd.sample(ep, target_size)]


def sample_from_pfg(pfg_pool: List[Solution],
                    rnd: random.Random) -> Solution:
    if not pfg_pool:
        return None
    return rnd.choice(pfg_pool).copy()