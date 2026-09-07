import random
import time
from typing import Dict, List, Set
from collections import Counter
import numpy as np
from scipy.spatial.distance import cdist

from model.solution import Solution
from model.customer import Customer
from model.evaluator import Evaluator


class NSGA3Solver:
    """
    NSGA-III for VRP-D
    - Multi-truck
    - One drone per truck
    - Stop by maximum computational time
    - Uses Reference-point based non-dominated sorting
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

        # Generate Reference Points (nobj = 2 for makespan and carbonEmission)
        self.V = self._generate_reference_points(self.pop_size, 2)


    def run(self) -> List[Solution]:
        start_time = time.time()
        self.population = []
        self.population_perms = []
        self.population_drones = []

        self._init_population()
        self._fast_non_dominated_sort() 
        
        generation = 0
        while time.time() - start_time < self.max_time:
            offspring = []
            offspring_perms = []
            offspring_drones = []

            while len(offspring) < self.pop_size and time.time() - start_time < self.max_time:

                p1 = self._tournament()
                p2 = self._tournament()

                
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


    @staticmethod
    def _generate_reference_points(npop, nvar):
        from itertools import combinations
        import math
        
        def combination(n, m):
            return math.comb(n, m)
            
        h1 = 0
        while combination(h1 + nvar, nvar - 1) <= npop:
            h1 += 1
        points = np.array(list(combinations(np.arange(1, h1 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
        points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
        if h1 < nvar:
            h2 = 0
            while combination(h1 + nvar - 1, nvar - 1) + combination(h2 + nvar, nvar - 1) <= npop:
                h2 += 1
            if h2 > 0:
                temp_points = np.array(list(combinations(np.arange(1, h2 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
                temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
                temp_points = temp_points / 2 + 1 / (2 * nvar)
                points = np.concatenate((points, temp_points), axis=0)
        return points

    def _tournament(self) -> int:
        i = self.rnd.randrange(len(self.population))
        j = self.rnd.randrange(len(self.population))

        a = self.population[i]
        b = self.population[j]

        if a.rank < b.rank:
            return i
        if b.rank < a.rank:
            return j
        return i if self.rnd.random() < 0.5 else j

    def _survivor_selection(self):
        fronts = self._fast_non_dominated_sort()

        new_pop = []
        new_perms = []
        new_drones = []

        pop_map = {id(s): i for i, s in enumerate(self.population)}
        
        S_t = []

        for front in fronts:
            if len(new_pop) + len(front) <= self.pop_size:
                for s in front:
                    idx = pop_map[id(s)]
                    new_pop.append(s)
                    new_perms.append(self.population_perms[idx])
                    new_drones.append(self.population_drones[idx])
                    S_t.append(s)
            else:
                K = self.pop_size - len(new_pop)
                
                last_front = front
                last_front_perms = [self.population_perms[pop_map[id(s)]] for s in front]
                last_front_drones = [self.population_drones[pop_map[id(s)]] for s in front]
                
                for s in front:
                    S_t.append(s)

                sel_sols, sel_perms, sel_drones = self._environmental_selection(
                    S_t, last_front, last_front_perms, last_front_drones, len(new_pop), K
                )
                
                new_pop.extend(sel_sols)
                new_perms.extend(sel_perms)
                new_drones.extend(sel_drones)
                break

        self.population = new_pop
        self.population_perms = new_perms
        self.population_drones = new_drones

    def _environmental_selection(self, S_t, last_front, last_front_perms, last_front_drones, num_already_selected, K):
        nobj = 2
        
        # 1. Translate objectives
        objs = np.array([[s.makespan, s.carbonEmission] for s in S_t])
        zmin = np.min(objs, axis=0)
        t_objs = objs - zmin
        
        # 2. Extreme points
        w = 1e-6 + np.eye(nobj)
        extreme = np.zeros(nobj, dtype=int)
        for i in range(nobj):
            extreme[i] = np.argmin(np.max(t_objs / w[i], axis=1))
            
        # 3. Intercepts (Hyperplane)
        try:
            hyperplane = np.linalg.solve(t_objs[extreme], np.ones(nobj))
            a = 1 / hyperplane
            if np.any(np.isnan(a)) or np.any(a <= 0):
                a = np.max(t_objs, axis=0)
        except np.linalg.LinAlgError:
            a = np.max(t_objs, axis=0)
            
        a[a == 0] = 1e-6
        t_objs /= a
        
        # 4. Association
        dist_matrix = cdist(t_objs, self.V, 'cosine')
        cosine = 1 - dist_matrix
        norm_t_objs = np.linalg.norm(t_objs, axis=1, keepdims=True)
        distance = norm_t_objs * np.sqrt(np.clip(1 - cosine ** 2, 0.0, 1.0))
        
        association = np.argmin(distance, axis=1)
        dis = np.min(distance, axis=1)
        
        # 5. Niching
        nv = len(self.V)
        rho = np.zeros(nv, dtype=int)
        counts = Counter(association[:num_already_selected])
        for key, val in counts.items():
            rho[key] = val
            
        choose = np.full(len(last_front), False)
        v_choose = np.full(nv, True)
        
        while np.sum(choose) < K:
            temp = np.where(v_choose)[0]
            if len(temp) == 0:
                break
                
            jmin = np.where(rho[temp] == np.min(rho[temp]))[0]
            j = temp[self.rnd.choice(jmin)]
            
            I = np.where((~choose) & (association[num_already_selected:] == j))[0]
            
            if I.size > 0:
                if rho[j] == 0:
                    s = np.argmin(dis[num_already_selected + I])
                else:
                    s = self.rnd.randrange(I.size)
                choose[I[s]] = True
                rho[j] += 1
            else:
                v_choose[j] = False
                
        chosen_indices = np.where(choose)[0]

        if len(chosen_indices) < K:
            remaining = np.where(~choose)[0]
            fallback_chosen = self.rnd.sample(list(remaining), K - len(chosen_indices))
            chosen_indices = np.concatenate([chosen_indices, fallback_chosen])

        return ([last_front[i] for i in chosen_indices],
                [last_front_perms[i] for i in chosen_indices],
                [last_front_drones[i] for i in chosen_indices])

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

    def _get_pareto_front(self):
        return [s for s in self.population if s.rank == 0]


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