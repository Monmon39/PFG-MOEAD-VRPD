import random
import time
import math
from typing import Dict, List, Set
from collections import Counter
import numpy as np

# from pfg_moead_vrpd_ver2.model.solution import Solution
# from pfg_moead_vrpd_ver2.model.customer import Customer
# from pfg_moead_vrpd_ver2.model.evaluator import Evaluator

from model.solution import Solution
from model.customer import Customer
from model.evaluator import Evaluator


class ALNSMOSolver:
    """
    Multi-truck, One drone per truck
    Makespan và Carbon Emission.
    Max_time.
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

        self.population: List[Solution] = []
        
        self.operators = ['op_swap_intra', 'op_relocate_inter', 'op_swap_inter', 'op_drone_flip', 'op_2opt']
        self.weights = {op: 1.0 for op in self.operators}
        self.scores = {op: 0.0 for op in self.operators}
        self.usage = {op: 0 for op in self.operators}
        
        self.theta_1 = 1.0  
        self.theta_2 = 0.5  
        self.epsilon = 0.3  
        self.generations = 0

    def run(self) -> List[Solution]:
        start_time = time.time()
        
        self._init_population()
        
        while time.time() - start_time < self.max_time:
            self.generations += 1
            
            fronts = self._fast_non_dominated_sort(self.population)
            self._assign_crowding_distance(fronts)
            
            offspring_pop = []

            while len(offspring_pop) < self.pop_size and time.time() - start_time < self.max_time:

                parent = self._tournament(self.population)
                
                chosen_op = self._roulette_wheel_selection()
                self.usage[chosen_op] += 1
                
                child = self._apply_operator(chosen_op, parent)
                
                self._enforce_no_adjacent_drones(child)
                Evaluator.evaluate(child, self.customers)
                
                self._update_operator_score(chosen_op, parent, child)
                
                if self._metropolis_accept(parent, child, self.generations):
                    offspring_pop.append(child)
                else:
                    offspring_pop.append(self._clone_solution(parent))
            
            self._update_weights()
            
            combined_pop = self.population + offspring_pop
            self.population = self._survivor_selection(combined_pop, self.pop_size)

        fronts = self._fast_non_dominated_sort(self.population)
        return fronts[0] if fronts else []

    def _apply_operator(self, op: str, parent: Solution) -> Solution:
        child = self._clone_solution(parent)
        
        if op == 'op_swap_intra':
            # Đổi chỗ 2 khách hàng trong cùng 1 tuyến xe tải
            t = self.rnd.randrange(self.num_trucks)
            route = child.truckRoutes[t]
            if len(route) > 3:
                i, j = self.rnd.sample(range(1, len(route) - 1), 2)
                route[i], route[j] = route[j], route[i]

        elif op == 'op_relocate_inter':
            # Rút 1 khách hàng từ xe này cắm sang xe khác
            if self.num_trucks > 1:
                t1, t2 = self.rnd.sample(range(self.num_trucks), 2)
                r1, r2 = child.truckRoutes[t1], child.truckRoutes[t2]
                if len(r1) > 2:
                    idx1 = self.rnd.randrange(1, len(r1) - 1)
                    cust = r1.pop(idx1)
                    idx2 = self.rnd.randrange(1, len(r2))
                    r2.insert(idx2, cust)

        elif op == 'op_swap_inter':
            # Đổi chỗ 2 khách hàng giữa 2 xe tải khác nhau
            if self.num_trucks > 1:
                t1, t2 = self.rnd.sample(range(self.num_trucks), 2)
                r1, r2 = child.truckRoutes[t1], child.truckRoutes[t2]
                if len(r1) > 2 and len(r2) > 2:
                    idx1 = self.rnd.randrange(1, len(r1) - 1)
                    idx2 = self.rnd.randrange(1, len(r2) - 1)
                    r1[idx1], r2[idx2] = r2[idx2], r1[idx1]

        elif op == 'op_drone_flip':
            # Random thay đổi quyết định giao bằng Drone (Bật/Tắt)
            t = self.rnd.randrange(self.num_trucks)
            route = child.truckRoutes[t]
            dset = child.droneCustomers[t]
            if len(route) > 2:
                cid = self.rnd.choice(route[1:-1])
                if self.customers[cid].drone_serve:
                    if cid in dset:
                        dset.remove(cid)
                    else:
                        dset.add(cid)

        elif op == 'op_2opt':
            # Đảo ngược một đoạn lộ trình (Local Search cơ bản)
            t = self.rnd.randrange(self.num_trucks)
            route = child.truckRoutes[t]
            if len(route) > 4:
                i, j = sorted(self.rnd.sample(range(1, len(route) - 1), 2))
                route[i:j+1] = reversed(route[i:j+1])
                
        return child

    def _roulette_wheel_selection(self) -> str:
        total = sum(self.weights.values())
        pick = self.rnd.uniform(0, total)
        current = 0
        for op, w in self.weights.items():
            current += w
            if current >= pick:
                return op
        return self.operators[-1]

    def _update_operator_score(self, op: str, parent: Solution, child: Solution):
        if child.dominates(parent):
            self.scores[op] += self.theta_1
        elif not parent.dominates(child):
            self.scores[op] += self.theta_2

    def _update_weights(self):
        for op in self.operators:
            if self.usage[op] > 0:
                self.weights[op] = (1 - self.epsilon) * self.weights[op] + \
                                   self.epsilon * (self.scores[op] / self.usage[op])
            self.scores[op] = 0
            self.usage[op] = 0

    def _metropolis_accept(self, parent: Solution, child: Solution, current_iter: int) -> bool:

        if child.dominates(parent) or not parent.dominates(child):
            return True
            
        df1 = (child.makespan - parent.makespan) / (parent.makespan + 1e-6)
        df2 = (child.carbonEmission - parent.carbonEmission) / (parent.carbonEmission + 1e-6)
        
        # Exponential penalty
        exponent = - (df1 + df2) * current_iter * max(1, child.rank)
        
        if exponent < -20: return False
        return self.rnd.random() < math.exp(exponent)

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
            self._enforce_no_adjacent_drones(sol)
            Evaluator.evaluate(sol, self.customers)
            self.population.append(sol)

    def _split(self, seq, k):
        n = len(seq)
        if n <= k:
            routes = []
            for i in range(k):
                routes.append([0, seq[i], 0] if i < n else [0, 0])
            return routes

        cuts = sorted(self.rnd.sample(range(1, n), k - 1))
        routes = []
        prev = 0
        for cut in cuts:
            routes.append([0] + seq[prev:cut] + [0])
            prev = cut
        routes.append([0] + seq[prev:] + [0])
        return routes

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

    def _clone_solution(self, sol: Solution) -> Solution:
        new_sol = Solution()
        new_sol.truckRoutes = [r[:] for r in sol.truckRoutes]
        new_sol.droneCustomers = [set(d) for d in sol.droneCustomers]
        new_sol.makespan = sol.makespan
        new_sol.carbonEmission = sol.carbonEmission
        new_sol.rank = sol.rank
        return new_sol

    def _fast_non_dominated_sort(self, pop: List[Solution]):
        fronts = [[]]
        for p in pop:
            p.dom_count = 0
            p.dom_set = []
            for q in pop:
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

    def _assign_crowding_distance(self, fronts: List[List[Solution]]):
        for front in fronts:
            l = len(front)
            for f in front:
                f.crowding_distance = 0.0
            if l <= 2:
                for f in front:
                    f.crowding_distance = float('inf')
                continue
            
            # Theo Makespan
            front.sort(key=lambda x: x.makespan)
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            m_min, m_max = front[0].makespan, front[-1].makespan
            if m_max > m_min:
                for i in range(1, l - 1):
                    front[i].crowding_distance += (front[i+1].makespan - front[i-1].makespan) / (m_max - m_min)
            
            # Theo Carbon
            front.sort(key=lambda x: x.carbonEmission)
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            c_min, c_max = front[0].carbonEmission, front[-1].carbonEmission
            if c_max > c_min:
                for i in range(1, l - 1):
                    front[i].crowding_distance += (front[i+1].carbonEmission - front[i-1].carbonEmission) / (c_max - c_min)

    def _tournament(self, pop: List[Solution]) -> Solution:
        a, b = self.rnd.sample(pop, 2)
        if a.rank < b.rank:
            return a
        elif b.rank < a.rank:
            return b
        elif getattr(a, 'crowding_distance', 0) > getattr(b, 'crowding_distance', 0):
            return a
        elif getattr(b, 'crowding_distance', 0) > getattr(a, 'crowding_distance', 0):
            return b
        return a if self.rnd.random() < 0.5 else b

    def _survivor_selection(self, combined_pop: List[Solution], k: int) -> List[Solution]:
        fronts = self._fast_non_dominated_sort(combined_pop)
        self._assign_crowding_distance(fronts)
        
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= k:
                new_pop.extend(front)
            else:
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                needed = k - len(new_pop)
                new_pop.extend(front[:needed])
                break
        return new_pop