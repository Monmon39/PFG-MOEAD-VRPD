# model/evaluator.py

from typing import Dict, List, Set, Tuple
from .solution import Solution
from .customer import Customer
from .constraints import Constraints


class Evaluator:
    """
    VRP-D Evaluator – PARALLEL TRUCK–DRONE
    - Drone-served customers are REMOVED from truck route
    - Truck and drone operate in parallel
    - Objectives: makespan & carbon emission
    """

    TRUCK_SPEED = 35.0
    DRONE_SPEED = 50.0
    DRONE_ENDURANCE = 30.0

    TRUCK_CAPACITY = 1300.0
    DRONE_CAPACITY = 10.0

    S_LAUNCH = 1.0
    S_RECOVER = 1.0

    TRUCK_CO2 = 1.2603
    DRONE_PGFER = 3.773e-4
    DRONE_AER = 3.3333

    SERVICE_LEVEL_TARGET = 0.8

    # ---- softened penalties (controlled)
    PENALTY_ENDURANCE = 50.0
    PENALTY_SERVICE = 150.0

    _DIST_CACHE: Dict[Tuple[int, int], float] = {}

    # =====================================================
    @staticmethod
    def _dist(customers: Dict[int, Customer], i: int, j: int) -> float:
        key = (i, j)
        if key not in Evaluator._DIST_CACHE:
            Evaluator._DIST_CACHE[key] = customers[i].distance_to(customers[j])
        return Evaluator._DIST_CACHE[key]

    # =====================================================
    @staticmethod
    def evaluate(sol: Solution, customers: Dict[int, Customer]) -> None:

        total_emission = 0.0
        truck_finish_times: List[float] = []
        drone_finish_times: List[float] = []

        accounted: Set[int] = set()
        sum_service_level = 0.0
        endurance_violation = 0.0

        dist = Evaluator._dist

        # ================= SIMULATION =====================
        for ti, route in enumerate(sol.truckRoutes):

            drone_set = sol.droneCustomers[ti]

            # ----- build truck route (REMOVE drone customers)
            truck_stops = [0]
            for cid in route[1:-1]:
                if cid not in drone_set:
                    truck_stops.append(cid)
            truck_stops.append(0)

            # ----- build drone jobs per arc
            drone_jobs = {}
            rlen = len(route)

            for pos in range(1, rlen - 1):
                cid = route[pos]
                if cid not in drone_set:
                    continue

                launch = next(
                    route[p] for p in range(pos - 1, -1, -1)
                    if route[p] == 0 or route[p] not in drone_set
                )
                ret = next(
                    route[p] for p in range(pos + 1, rlen)
                    if route[p] == 0 or route[p] not in drone_set
                )

                drone_jobs.setdefault((launch, ret), []).append(cid)

            # ================= TIME SIM =====================
            truck_time = 0.0
            drone_time = 0.0

            for i in range(len(truck_stops) - 1):
                prev = truck_stops[i]
                nxt = truck_stops[i + 1]

                leg_dist = dist(customers, prev, nxt)
                travel_time = leg_dist / Evaluator.TRUCK_SPEED

                # ---- truck travel
                truck_time += travel_time
                total_emission += Evaluator.TRUCK_CO2 * leg_dist

                # ---- truck service
                if nxt != 0:
                    cus = customers[nxt]
                    if nxt not in accounted:
                        sum_service_level += Evaluator.service_level(truck_time, cus)
                        accounted.add(nxt)
                    truck_time += cus.service_time

                # ---- drone parallel jobs
                arc_jobs = drone_jobs.get((prev, nxt))
                if arc_jobs:
                    launch_time = truck_time - travel_time + Evaluator.S_LAUNCH

                    for cid in arc_jobs:
                        cus = customers[cid]

                        fly1 = dist(customers, prev, cid) / Evaluator.DRONE_SPEED
                        fly2 = dist(customers, cid, nxt) / Evaluator.DRONE_SPEED
                        serve = cus.service_time

                        flight_time = fly1 + serve + fly2
                        if flight_time > Evaluator.DRONE_ENDURANCE:
                            endurance_violation += (
                                flight_time - Evaluator.DRONE_ENDURANCE
                            )

                        finish = (
                            launch_time
                            + flight_time
                            + Evaluator.S_RECOVER
                        )
                        drone_time = max(drone_time, finish)

                        arrival = launch_time + fly1
                        if cid not in accounted:
                            sum_service_level += Evaluator.service_level(arrival, cus)
                            accounted.add(cid)

                        drone_dist = (
                            dist(customers, prev, cid)
                            + dist(customers, cid, nxt)
                        )
                        total_emission += (
                            Evaluator.DRONE_PGFER
                            * Evaluator.DRONE_AER
                            * drone_dist
                        )

            truck_finish_times.append(truck_time)
            drone_finish_times.append(drone_time)

        # ================= OBJECTIVES =====================
        makespan = max(
            max(truck_finish_times, default=0.0),
            max(drone_finish_times, default=0.0)
        )

        all_customers = {
            cid
            for r in sol.truckRoutes
            for cid in r
            if cid != 0
        }

        avg_service = (
            sum_service_level / len(all_customers)
            if all_customers else 1.0
        )

        # ================= PENALTIES ======================
        makespan += endurance_violation * Evaluator.PENALTY_ENDURANCE

        if avg_service < Evaluator.SERVICE_LEVEL_TARGET:
            makespan += (
                Evaluator.SERVICE_LEVEL_TARGET - avg_service
            ) * Evaluator.PENALTY_SERVICE

        # ================= CONSTRAINTS ===================
        feasible = Constraints.check_all(
            sol,
            customers,
            Evaluator.TRUCK_CAPACITY,
            Evaluator.DRONE_CAPACITY
        )

        sol.makespan = round(makespan, 2)
        sol.carbonEmission = round(total_emission, 2)
        sol.feasible = feasible
        sol.serviceLevel = round(avg_service, 3)

    # =====================================================
    @staticmethod
    def service_level(arrival: float, c: Customer) -> float:
        """
        Soft service level:
        - On time        : 1.0
        - Late delivery  : linear decay
        - Very late      : 0.0
        """

        if arrival < c.ready_time:
            return 0.0

        if arrival <= c.due_time:
            return 1.0

        lateness = arrival - c.due_time

        # soft decay window (10 time units)
        return max(0.0, 1.0 - lateness / 10.0)
