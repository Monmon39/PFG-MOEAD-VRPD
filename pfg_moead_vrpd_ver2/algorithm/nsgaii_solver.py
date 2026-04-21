import random
import time
from typing import Dict, List, Set

from pfg_moead_vrpd_ver2.model.solution import Solution
from pfg_moead_vrpd_ver2.model.customer import Customer
from pfg_moead_vrpd_ver2.model.evaluator import Evaluator


class NSGA2Solver:
    """
    NSGA-II for VRP-D
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

        self.crossover_prob = 0.9
        self.mutation_swap_prob = 0.10
        self.mutation_flip_prob = 0.10

        self.population: List[Solution] = []
        self.population_perms: List[List[int]] = []
        self.population_drones: List[List[Set[int]]] = []


    # MAIN LOOP

    def run(self) -> List[Solution]:
        start_time = time.time()
        self.population = []
        self.population_perms = []
        self.population_drones = []

        self._init_population()

        fronts = self._fast_non_dominated_sort()
        for f in fronts:
            self._crowding_distance(f)
        generation = 0
        while time.time() - start_time < self.max_time:
            offspring = []
            offspring_perms = []
            offspring_drones = []

            while len(offspring) < self.pop_size  and time.time() - start_time < self.max_time:

                p1 = self._tournament()
                p2 = self._tournament()

                # crossover probability
                if self.rnd.random() > self.crossover_prob:
                    continue
                perm1 = self.population_perms[p1]
                perm2 = self.population_perms[p2]

                drone1 = self.population_drones[p1]
                drone2 = self.population_drones[p2]

                child_perm = self._order_crossover_safe(perm1, perm2)

                if self.rnd.random() < self.mutation_swap_prob:
                    self._swap_mutate(child_perm)

                child = Solution()
                child.truckRoutes = self._split(child_perm, self.num_trucks)

                child_drones = []
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

                Evaluator.evaluate(child, self.customers)

                offspring.append(child)
                offspring_perms.append(child_perm[:])
                offspring_drones.append([set(ds) for ds in child_drones])

            self.population.extend(offspring)
            self.population_perms.extend(offspring_perms)
            self.population_drones.extend(offspring_drones)

            self._survivor_selection()
            generation += 1
        self.generations = generation
        return self._get_pareto_front()

    # INITIALIZATION

    def _init_population(self):

        cust_list = [cid for cid in self.customers if cid != 0]

        for _ in range(self.pop_size):
            perm = cust_list[:]
            self.rnd.shuffle(perm)

            sol = Solution()
            sol.truckRoutes = self._split(perm, self.num_trucks)

            drones = []
            for route in sol.truckRoutes:
                dset = set()
                for cid in route[1:-1]:
                    if self.customers[cid].drone_serve and self.rnd.random() < 0.3:
                        dset.add(cid)
                drones.append(dset)

            sol.droneCustomers = drones
            sol.normalize()
            self._enforce_no_adjacent_drones(sol)

            Evaluator.evaluate(sol, self.customers)

            self.population.append(sol)
            self.population_perms.append(perm[:])
            self.population_drones.append([set(ds) for ds in drones])

    # NSGA-II CORE

    def _tournament(self) -> int:
        i = self.rnd.randrange(len(self.population))
        j = self.rnd.randrange(len(self.population))

        a = self.population[i]
        b = self.population[j]

        if a.rank < b.rank:
            return i
        if b.rank < a.rank:
            return j

        return i if a.crowdingDistance > b.crowdingDistance else j

    def _survivor_selection(self):

        fronts = self._fast_non_dominated_sort()

        new_pop = []
        new_perms = []
        new_drones = []

        for front in fronts:

            self._crowding_distance(front)

            if len(new_pop) + len(front) <= self.pop_size:

                for s in front:
                    idx = self.population.index(s)
                    new_pop.append(s)
                    new_perms.append(self.population_perms[idx])
                    new_drones.append(self.population_drones[idx])

            else:
                front.sort(key=lambda s: s.crowdingDistance, reverse=True)
                remain = self.pop_size - len(new_pop)

                for s in front[:remain]:
                    idx = self.population.index(s)
                    new_pop.append(s)
                    new_perms.append(self.population_perms[idx])
                    new_drones.append(self.population_drones[idx])
                break

        self.population = new_pop
        self.population_perms = new_perms
        self.population_drones = new_drones

    def _fast_non_dominated_sort(self):

        fronts = [[]]

        for p in self.population:
            p.dom_count = 0
            p.dom_set = []

            for q in self.population:
                if p.dominates(q):
                    p.dom_set.append(q)
                elif q.dominates(p):
                    p.dom_count += 1

            if p.dom_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dom_set:
                    q.dom_count -= 1
                    if q.dom_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _crowding_distance(self, front):

        if not front:
            return

        for s in front:
            s.crowdingDistance = 0.0

        for obj in ["makespan", "carbonEmission"]:

            front.sort(key=lambda s: getattr(s, obj))

            front[0].crowdingDistance = float("inf")
            front[-1].crowdingDistance = float("inf")

            min_val = getattr(front[0], obj)
            max_val = getattr(front[-1], obj)

            if max_val == min_val:
                continue

            for i in range(1, len(front) - 1):
                prev_val = getattr(front[i - 1], obj)
                next_val = getattr(front[i + 1], obj)

                front[i].crowdingDistance += (
                    (next_val - prev_val) / (max_val - min_val)
                )

    def _get_pareto_front(self):
        return [s for s in self.population if s.rank == 0]

    # OPERATORS & LOCAL SEARCH (GIỮ NGUYÊN NHƯ MOEA/D)

    def _order_crossover_safe(self, p1, p2):
        n = len(p1)
        child = [-1] * n
        a, b = sorted(self.rnd.sample(range(n), 2))
        child[a:b + 1] = p1[a:b + 1]
        used = set(child[a:b + 1])
        pos = (b + 1) % n
        for gene in p2:
            if gene not in used:
                child[pos] = gene
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
                if self.rnd.random() < 0.5:
                    child.add(cid)
        return child

    def _flip_mutate_drone(self, dset, route):
        for cid in route[1:-1]:
            if not self.customers[cid].drone_serve:
                continue
            if self.rnd.random() < self.mutation_flip_prob:
                if cid in dset:
                    dset.remove(cid)
                else:
                    dset.add(cid)

    def _split(self, seq, k):
        n = len(seq)
        if n <= k:
            routes = []
            for i in range(k):
                if i < n:
                    routes.append([0, seq[i], 0])
                else:
                    routes.append([0, 0])
            return routes

        cuts = sorted(self.rnd.sample(range(1, n), k - 1))
        routes = []
        prev = 0
        for cut in cuts:
            routes.append([0] + seq[prev:cut] + [0])
            prev = cut
        routes.append([0] + seq[prev:] + [0])
        return routes

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

    def _local_search(self, sol):

        for route in sol.truckRoutes:
            if len(route) <= 4:
                continue

            improved = True
            while improved:
                improved = False
                n = len(route)

                for i in range(1, n - 2):
                    for k in range(i + 1, n - 1):

                        a, b = route[i - 1], route[i]
                        c, d = route[k], route[k + 1]

                        before = Evaluator._dist(self.customers, a, b) + \
                                 Evaluator._dist(self.customers, c, d)

                        after = Evaluator._dist(self.customers, a, c) + \
                                Evaluator._dist(self.customers, b, d)

                        if after - before < -1e-6:
                            route[i:k + 1] = reversed(route[i:k + 1])
                            improved = True
                            break
                    if improved:
                        break

        self._enforce_no_adjacent_drones(sol)