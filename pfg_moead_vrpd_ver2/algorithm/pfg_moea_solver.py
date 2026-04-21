import random
import time
import numpy as np
from typing import List, Dict, Set

from pfg_moead_vrpd_ver2.model.solution import Solution
from pfg_moead_vrpd_ver2.model.customer import Customer
from pfg_moead_vrpd_ver2.model.evaluator import Evaluator
from pfg_moead_vrpd_ver2.utils.pfg import build_pfg


class PFGMOEASolver:
    """
    PFG-MOEA (PURE)
    - Parent selection: PFG (External Population)
    - Environmental selection: Pareto + Knee removal
    - No decomposition (khác MOEA/D)
    """

    def __init__(
        self,
        pop_size: int,
        max_time: float,
        num_trucks: int,
        customers: Dict[int, Customer],
        seed: int = 1
    ):
        self.pop_size = pop_size
        self.max_time = max_time
        self.num_trucks = num_trucks
        self.customers = customers

        self.rnd = random.Random(seed)

        self.PC = 0.9
        self.PM_SWAP = 0.1
        self.PM_FLIP = 0.1

        self.population: List[Solution] = []
        self.external_pop: List[Solution] = []
        self.pfg_pool: List[Solution] = []

    # MAIN
    def run(self) -> List[Solution]:

        start = time.time()

        self.population = []
        self.external_pop = []

        self._init_population()

        for s in self.population:
            self._update_external(s)

        generation = 0

        while time.time() - start < self.max_time:

            # Build PFG từ external population
            ep = build_pfg(self.external_pop)
            self.pfg_pool = ep

            # OFFSPRING
            offspring = []

            while len(offspring) < self.pop_size:

                if len(self.pfg_pool) < 2:
                    parents = self.population
                else:
                    parents = self.pfg_pool

                p1 = self.rnd.choice(parents)
                p2 = self.rnd.choice(parents)

                if self.rnd.random() > self.PC:
                    continue

                perm1 = self._flatten(p1.truckRoutes)
                perm2 = self._flatten(p2.truckRoutes)

                child_perm = self._order_crossover(perm1, perm2)

                if self.rnd.random() < self.PM_SWAP:
                    self._swap_mutate(child_perm)

                child = Solution()
                child.truckRoutes = self._split(child_perm)

                # Drone crossover
                drones = []
                for t in range(self.num_trucks):
                    d = self._uniform_crossover_drone(
                        p1.droneCustomers[t],
                        p2.droneCustomers[t],
                        child.truckRoutes[t]
                    )
                    self._flip_mutate_drone(d, child.truckRoutes[t])
                    drones.append(d)

                child.droneCustomers = drones
                child.normalize()
                self._enforce_no_adjacent_drones(child)

                # Local search
                self._local_search(child)

                # Evaluate
                Evaluator.evaluate(child, self.customers)

                offspring.append(child)

            # Combine
            combined = self.population + offspring

            # Estimate z*
            F = np.array([self._obj(s) for s in combined])
            z_star = np.min(F, axis=0)

            # Estimate z_nad (1/3 sampling)
            sample = self.rnd.sample(combined, max(2, self.pop_size // 3))
            sf = self._fast_nondominated_sort(sample)[0]
            z_nad = np.max([self._obj(s) for s in sf], axis=0)

            # Environmental Selection
            self.population = self._environmental_selection(
                combined, z_star, z_nad
            )

            # Update external
            for s in self.population:
                self._update_external(s)

            generation += 1

        self.generations = generation
        return self.external_pop[:]

    # OBJECTIVE
    def _obj(self, s):
        return (s.makespan, s.carbonEmission)

    # INIT
    def _init_population(self):

        cust = [c for c in self.customers if c != 0]

        for _ in range(self.pop_size):

            perm = cust[:]
            self.rnd.shuffle(perm)

            sol = Solution()
            sol.truckRoutes = self._split(perm)

            drones = []
            for route in sol.truckRoutes:
                dset = set()
                for cid in route[1:-1]:
                    if self.customers[cid].drone_serve and self.rnd.random() < 0.4:
                        dset.add(cid)
                drones.append(dset)

            sol.droneCustomers = drones
            sol.normalize()

            Evaluator.evaluate(sol, self.customers)

            self.population.append(sol)

    # NON-DOMINATED SORT
    def _fast_nondominated_sort(self, pop):

        S, n = {}, {}
        fronts = [[]]

        for p in pop:
            S[p], n[p] = [], 0

            for q in pop:
                if p.dominates(q):
                    S[p].append(q)
                elif q.dominates(p):
                    n[p] += 1

            if n[p] == 0:
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        nxt.append(q)
            i += 1
            fronts.append(nxt)

        fronts.pop()
        return fronts

    # ENVIRONMENTAL SELECTION
    def _environmental_selection(self, pop, z_star, z_nad):

        fronts = self._fast_nondominated_sort(pop)
        new_pop = []

        for front in fronts:

            if len(new_pop) + len(front) <= self.pop_size:
                new_pop.extend(front)
            else:

                while len(new_pop) + len(front) > self.pop_size:

                    worst = self._knee(front, z_star, z_nad)
                    front.remove(worst)

                new_pop.extend(front)
                break

        return new_pop[:self.pop_size]

    # KNEE POINT
    def _knee(self, sols, z_star, z_nad):

        worst, worst_d = None, -1

        for s in sols:
            f1, f2 = self._obj(s)

            f1n = (f1 - z_star[0]) / max(1e-9, z_nad[0] - z_star[0])
            f2n = (f2 - z_star[1]) / max(1e-9, z_nad[1] - z_star[1])

            d = abs(f1n + f2n - 1)

            if d > worst_d:
                worst_d = d
                worst = s

        return worst

    # OPERATORS
    def _order_crossover(self, p1, p2):

        n = len(p1)
        child = [-1] * n

        a, b = sorted(self.rnd.sample(range(n), 2))
        child[a:b+1] = p1[a:b+1]

        used = set(child[a:b+1])
        pos = (b + 1) % n

        for g in p2:
            if g not in used:
                child[pos] = g
                pos = (pos + 1) % n

        return child

    def _swap_mutate(self, perm):
        i, j = self.rnd.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]

    def _uniform_crossover_drone(self, d1, d2, route):
        child = set()
        for cid in route[1:-1]:
            if not self.customers[cid].drone_serve:
                continue
            if cid in d1 and cid in d2:
                child.add(cid)
            elif cid in d1 or cid in d2:
                if self.rnd.random() < 0.6:
                    child.add(cid)
        return child

    def _flip_mutate_drone(self, dset, route):
        for cid in route[1:-1]:
            if not self.customers[cid].drone_serve:
                continue
            if self.rnd.random() < self.PM_FLIP:
                if cid in dset:
                    dset.remove(cid)
                else:
                    dset.add(cid)

    # HELPERS
    def _flatten(self, routes):
        return [c for r in routes for c in r if c != 0]

    def _split(self, seq):

        k = self.num_trucks
        n = len(seq)

        if n == 0:
            return [[0, 0] for _ in range(k)]

        cuts = sorted(self.rnd.sample(range(1, n), k - 1)) if n >= k else []

        routes, prev = [], 0

        for c in cuts:
            routes.append([0] + seq[prev:c] + [0])
            prev = c

        routes.append([0] + seq[prev:] + [0])

        while len(routes) < k:
            routes.append([0, 0])

        return routes

    def _update_external(self, cand):

        for e in self.external_pop:
            if e.dominates(cand):
                return

        self.external_pop = [
            e for e in self.external_pop if not cand.dominates(e)
        ]

        self.external_pop.append(cand.copy())

    def _enforce_no_adjacent_drones(self, sol):

        for t, route in enumerate(sol.truckRoutes):
            dset = set(sol.droneCustomers[t])
            prev = False

            for cid in route[1:-1]:
                if cid in dset:
                    if prev:
                        dset.remove(cid)
                        prev = False
                    else:
                        prev = True
                else:
                    prev = False

            sol.droneCustomers[t] = dset

    # LOCAL SEARCH
    def _local_search(self, sol):

        for route in sol.truckRoutes:

            if len(route) <= 4:
                continue

            improved = True

            while improved:
                improved = False

                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):

                        before = self._dist(route[i-1], route[i]) + \
                                 self._dist(route[j], route[j+1])

                        after = self._dist(route[i-1], route[j]) + \
                                self._dist(route[i], route[j+1])

                        if after < before:
                            route[i:j+1] = reversed(route[i:j+1])
                            improved = True
                            break
                    if improved:
                        break

        self._enforce_no_adjacent_drones(sol)

    def _dist(self, i, j):
        return Evaluator._dist(self.customers, i, j)