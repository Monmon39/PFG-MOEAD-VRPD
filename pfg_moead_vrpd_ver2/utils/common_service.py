# utils/common_service.py

import csv
import random
from typing import Dict, List

from pfg_moead_vrpd.model.customer import Customer
from pfg_moead_vrpd.model.solution import Solution


# LOAD DATA
def load_customers_v2(path: str) -> Dict[int, Customer]:
    customers: Dict[int, Customer] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if not row:
                continue

            cid = int(row[0])
            customers[cid] = Customer(
                cid=cid,
                x=float(row[1]),
                y=float(row[2]),
                demand=float(row[3]),
                ready_time=float(row[5]),
                due_time=float(row[6]),
                service_time=float(row[4]),
                drone_serve=bool(int(row[9])),
                time=0.0
            )

    return customers



# RANDOM INITIAL SOLUTION (CORRECT VRP-D)
def random_solution(customers: Dict[int, Customer],
                    num_trucks: int,
                    drone_ratio: float,
                    rnd: random.Random) -> Solution:
    sol = Solution()

    # ---- permutation (exclude depot) ----
    perm = [cid for cid in customers if cid != 0]
    rnd.shuffle(perm)

    # ---- split permutation to trucks ----
    routes = [[] for _ in range(num_trucks)]
    for i, cid in enumerate(perm):
        routes[i % num_trucks].append(cid)

    sol.truckRoutes = [[0] + r + [0] for r in routes]

    # ---- drone sets PER TRUCK ----
    sol.droneCustomers = []

    for route in sol.truckRoutes:
        candidates = [
            cid for cid in route
            if cid != 0 and customers[cid].drone_serve
        ]

        k = int(len(candidates) * drone_ratio)
        drone_set = (
            set(rnd.sample(candidates, k)) if k > 0 else set()
        )
        sol.droneCustomers.append(drone_set)

    return sol


# PARETO FILTER
def pareto_filter(population: List[Solution]) -> List[Solution]:
    ep = []
    for s in population:
        if not any(t is not s and t.dominates(s) for t in population):
            ep.append(s)
    return ep


# SAVE FULL PARETO (DEBUG)
def save_pareto(pareto: List[Solution], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("makespan,carbon,truckRoutes,droneCustomers\n")
        for s in pareto:
            f.write(
                f"{s.makespan},{s.carbonEmission},"
                f"{s.truckRoutes},{[sorted(ds) for ds in s.droneCustomers]}\n"
            )
