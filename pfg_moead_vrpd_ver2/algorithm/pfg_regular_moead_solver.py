import math
import random
import time
from typing import Dict, List, Set

from pfg_moead_vrpd_ver2.model.solution import Solution
from pfg_moead_vrpd_ver2.model.customer import Customer
from pfg_moead_vrpd_ver2.model.evaluator import Evaluator
from pfg_moead_vrpd_ver2.utils.pfg_regular import build_pfg_regular, sample_from_pfg


class RegularGridMOEADSolver:
    """
    PFG-MOEA/D for VRP-D
    - Parent 1: MOEA/D neighborhood
    - Parent 2: PFG (External Population)
    - Multi-truck
    - One drone per truck
    - Stop by maximum computational time
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

        self.T = max(2, int(round(pop_size * 0.1)))
        self.delta = 0.9

        self.crossover_prob = 0.9
        self.mutation_swap_prob = 0.10
        self.mutation_flip_prob = 0.10

        self.weights: List[List[float]] = []
        self.neighborhood: List[List[int]] = []

        self.population: List[Solution] = []
        self.population_perms: List[List[int]] = []
        self.population_drones: List[List[Set[int]]] = []

        self.ideal_point = [float("inf"), float("inf")]
        self.external_pop: List[Solution] = []
        self.pfg_pool: List[Solution] = []

        self._init_weights()
        self._init_neighborhood()

    # MAIN LOOP
    def run(self) -> List[Solution]:
        start_time = time.time()

        self.external_pop = []
        self.population = []
        self.population_perms = []
        self.population_drones = []

        # ===== Initialization =====
        self._init_population()
        self._update_ideal_point()

        for s in self.population:
            self._update_external_population(s)
        generation = 0
        # ===== Evolution Loop =====
        while time.time() - start_time < self.max_time:

            ep = build_pfg_regular(self.external_pop)
            self.pfg_pool = ep

            for i in range(self.pop_size):
                if time.time() - start_time >= self.max_time:
                    break

                # Parent 1: MOEA/D
                p1 = self._select_from_neighborhood(i)

                # crossover probability
                if self.rnd.random() > self.crossover_prob:
                    continue

                perm1 = self.population_perms[p1]
                drone1 = self.population_drones[p1]

                # Parent 2: PFG
                pfg_parent = sample_from_pfg(self.pfg_pool, self.rnd)
                perm2 = self._flatten(pfg_parent.truckRoutes)
                drone2 = pfg_parent.droneCustomers

                # Order crossover
                child_perm = self._order_crossover_safe(perm1, perm2)

                if self.rnd.random() < self.mutation_swap_prob:
                    self._swap_mutate(child_perm)

                # Build child
                child = Solution()
                child.truckRoutes = self._split(child_perm, self.num_trucks)

                # Drone crossover + mutation
                child_drones: List[Set[int]] = []
                for t in range(self.num_trucks):
                    d = self._uniform_crossover_drone(
                        drone1[t], drone2[t], child.truckRoutes[t]
                    )
                    self._flip_mutate_drone(d, child.truckRoutes[t])
                    child_drones.append(d)

                child.droneCustomers = child_drones
                child.normalize()
                self._enforce_no_adjacent_drones(child)

                self._local_search(child)

                # Evaluate
                Evaluator.evaluate(child, self.customers)

                self._update_ideal(child)
                self._update_neighborhood(i, child)
                self._update_external_population(child)
            generation += 1
        self.generations = generation
        return self.external_pop[:]

    # INITIALIZATION
    def _init_weights(self):
        for i in range(self.pop_size):
            w1 = i / max(1, self.pop_size - 1)
            self.weights.append([w1, 1.0 - w1])

    def _init_neighborhood(self):
        for i in range(self.pop_size):
            wi = self.weights[i]
            dist = []
            for j in range(self.pop_size):
                wj = self.weights[j]
                d = math.hypot(wi[0] - wj[0], wi[1] - wj[1])
                dist.append((d, j))
            dist.sort(key=lambda x: x[0])
            self.neighborhood.append([j for _, j in dist[:self.T]])

    def _init_population(self):

        cust_list = [cid for cid in self.customers if cid != 0]

        for _ in range(self.pop_size):

            perm = cust_list[:]
            self.rnd.shuffle(perm)

            sol = Solution()
            sol.truckRoutes = self._split(perm, self.num_trucks)

            drones: List[Set[int]] = []

            for route in sol.truckRoutes:
                dset = set()

                for i in range(1, len(route) - 1):
                    cid = route[i]

                    if not self.customers[cid].drone_serve:
                        continue

                    if self.rnd.random() < 0.4:
                        dset.add(cid)

                drones.append(dset)

            sol.droneCustomers = drones
            sol.normalize()

            Evaluator.evaluate(sol, self.customers)

            self.population.append(sol)
            self.population_perms.append(perm[:])
            self.population_drones.append([set(ds) for ds in drones])

    # CORE
    def _select_from_neighborhood(self, i: int) -> int:
        if self.rnd.random() < self.delta:
            neigh = self.neighborhood[i]
            return neigh[self.rnd.randrange(len(neigh))]
        return self.rnd.randrange(self.pop_size)

    def _update_neighborhood(self, i: int, child: Solution):
        for idx in self.neighborhood[i]:

            f_child = self._tchebycheff(child, self.weights[idx])
            f_old = self._tchebycheff(self.population[idx], self.weights[idx])

            # === PFG allows relaxed dominance ===
            if (
                f_child < f_old
                or (
                    child.makespan < self.population[idx].makespan
                    and child.carbonEmission
                    <= self.population[idx].carbonEmission * 1.1
                )
            ):
                c = child.copy()
                c.normalize()
                self._enforce_no_adjacent_drones(c)

                self.population[idx] = c
                self.population_perms[idx] = self._flatten(c.truckRoutes)
                self.population_drones[idx] = [set(ds) for ds in c.droneCustomers]

    # OPERATORS
    def _order_crossover_safe(self, p1: List[int], p2: List[int]) -> List[int]:
        gene_set = set(p1)
        p2 = [g for g in p2 if g in gene_set]

        if len(p2) != len(p1):
            missing = list(gene_set - set(p2))
            self.rnd.shuffle(missing)
            p2.extend(missing)

        n = len(p1)
        child = [-1] * n

        a, b = sorted(self.rnd.sample(range(n), 2))
        child[a:b + 1] = p1[a:b + 1]
        used = set(child[a:b + 1])

        pos = (b + 1) % n
        for g in p2:
            if g not in used:
                child[pos] = g
                pos = (pos + 1) % n

        return child

    def _swap_mutate(self, perm: List[int]):
        i, j = self.rnd.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]

    def _uniform_crossover_drone(
        self, d1: Set[int], d2: Set[int], route: List[int]
    ) -> Set[int]:
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

    def _flip_mutate_drone(self, dset: Set[int], route: List[int]):
        for cid in route[1:-1]:
            if not self.customers[cid].drone_serve:
                continue
            if self.rnd.random() < self.mutation_flip_prob * 2:
                if cid in dset:
                    dset.remove(cid)
                else:
                    dset.add(cid)

    # HELPERS
    def _split(self, seq: List[int], k: int) -> List[List[int]]:
        """
        Random breakpoint split:
        - Keep permutation order
        - Randomly choose k-1 cut points
        - Always return exactly k routes
        """

        n = len(seq)

        if k <= 0:
            return []

        # Nếu không có khách
        if n == 0:
            return [[0, 0] for _ in range(k)]

        # Nếu số khách <= số truck
        if n <= k:
            routes = []
            for i in range(k):
                if i < n:
                    routes.append([0, seq[i], 0])
                else:
                    routes.append([0, 0])
            return routes

        # ===== RANDOM CUTS =====
        cuts = sorted(self.rnd.sample(range(1, n), k - 1))

        routes = []
        prev = 0

        for cut in cuts:
            part = seq[prev:cut]
            routes.append([0] + part + [0])
            prev = cut

        # phần cuối
        routes.append([0] + seq[prev:] + [0])

        return routes

    def _flatten(self, routes: List[List[int]]) -> List[int]:
        return [cid for r in routes for cid in r if cid != 0]

    def _tchebycheff(self, s: Solution, w: List[float]) -> float:
        return max(
            w[0] * abs(s.makespan - self.ideal_point[0]),
            w[1] * abs(s.carbonEmission - self.ideal_point[1])
        )

    def _update_ideal_point(self):
        for s in self.population:
            self._update_ideal(s)

    def _update_ideal(self, s: Solution):
        self.ideal_point[0] = min(self.ideal_point[0], s.makespan)
        self.ideal_point[1] = min(self.ideal_point[1], s.carbonEmission)

    def _update_external_population(self, cand: Solution):
        for e in self.external_pop:
            if e.dominates(cand):
                return
        self.external_pop = [e for e in self.external_pop if not cand.dominates(e)]
        self.external_pop.append(cand.copy())

    def _enforce_no_adjacent_drones(self, sol: Solution):
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

    # ================= LOCAL SEARCH (2-opt) =================

    def _dist(self, i: int, j: int) -> float:
        return Evaluator._dist(self.customers, i, j)

    def _two_opt_delta(self, route: List[int], i: int, k: int) -> float:
        a = route[i - 1]
        b = route[i]
        c = route[k]
        d = route[k + 1]

        before = self._dist(a, b) + self._dist(c, d)
        after = self._dist(a, c) + self._dist(b, d)

        return after - before

    def _reverse_sublist(self, route: List[int], i: int, k: int):
        route[i:k+1] = reversed(route[i:k+1])

    def _local_search(self, sol: Solution):
        for route in sol.truckRoutes:
            if len(route) <= 4:
                continue

            improved = True
            while improved:
                improved = False
                n = len(route)

                for i in range(1, n - 2):
                    for k in range(i + 1, n - 1):

                        delta = self._two_opt_delta(route, i, k)

                        if delta < -1e-6:
                            self._reverse_sublist(route, i, k)
                            improved = True
                            break

                    if improved:
                        break

        # Re-fix drone constraint after route change
        self._enforce_no_adjacent_drones(sol)
