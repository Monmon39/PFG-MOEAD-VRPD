# model/solution.py

from typing import List, Set


class Solution:
    """
    Solution encoding for VRP-D (Truck – Drone)

    Encoding:
    - truckRoutes[i]: route of truck i (start & end with depot 0)
    - droneCustomers[i]: customers served by drone of truck i

    IMPORTANT:
    - truckRoutes ALWAYS contain all customers in visiting order
    - droneCustomers indicates which customers are served by drone
    - Removing drone customers from truck path is DONE IN EVALUATOR,
      NOT in the encoding.
    """

    def __init__(self):
        # Decision variables
        self.truckRoutes: List[List[int]] = []
        self.droneCustomers: List[Set[int]] = []

        # Objective values (MIN)
        self.makespan: float = float("inf")
        self.carbonEmission: float = float("inf")

        # Constraint handling
        self.penalty: float = 0.0          # aggregated penalty (for soft constraints)
        self.isFeasible: bool = True       # HARD constraint feasibility flag

        # optional: store violated constraint IDs (C1, C2, ...)
        self.violatedConstraints: Set[str] = set()

        self.rank: int = 0
        self.crowdingDistance: float = 0.0

    # Normalize encoding
    def normalize(self):
        """
        Ensure:
        - len(truckRoutes) == len(droneCustomers)
        - each truck route starts & ends with depot (0)
        - DO NOT remove drone customers from truckRoutes
          (this must be handled by evaluator)
        """

        # --- match number of trucks
        while len(self.droneCustomers) < len(self.truckRoutes):
            self.droneCustomers.append(set())

        if len(self.droneCustomers) > len(self.truckRoutes):
            self.droneCustomers = self.droneCustomers[:len(self.truckRoutes)]

        # --- normalize each route
        for i, route in enumerate(self.truckRoutes):

            # empty route → depot-depot
            if not route:
                self.truckRoutes[i] = [0, 0]
                continue

            # ensure start depot
            if route[0] != 0:
                route.insert(0, 0)

            # ensure end depot
            if route[-1] != 0:
                route.append(0)

    # Constraint utilities
    def reset_constraints(self):
        self.penalty = 0.0
        self.isFeasible = True
        self.violatedConstraints.clear()

    def add_violation(self, cid: str, penalty: float = 0.0):
        """
        Register a constraint violation.
        cid     : constraint ID (e.g., 'C1', 'C7')
        penalty : penalty amount (if soft constraint)
        """
        self.isFeasible = False
        self.violatedConstraints.add(cid)
        self.penalty += penalty

    # Copy
    def copy(self) -> "Solution":
        s = Solution()

        s.truckRoutes = [r[:] for r in self.truckRoutes]
        s.droneCustomers = [set(ds) for ds in self.droneCustomers]

        s.makespan = self.makespan
        s.carbonEmission = self.carbonEmission

        s.penalty = self.penalty
        s.isFeasible = self.isFeasible
        s.violatedConstraints = set(self.violatedConstraints)

        s.rank = self.rank
        s.crowdingDistance = self.crowdingDistance

        return s

    # Pareto dominance (minimize objectives)
    def dominates(self, other: "Solution") -> bool:
        """
        Constraint-aware dominance:
        - Feasible dominates infeasible
        - If both feasible → Pareto dominance
        """
        if other is None:
            return False

        if self.isFeasible and not other.isFeasible:
            return True
        if not self.isFeasible and other.isFeasible:
            return False
        if not self.isFeasible and not other.isFeasible:
            return self.penalty < other.penalty

        # both feasible
        no_worse = (
            self.makespan <= other.makespan
            and self.carbonEmission <= other.carbonEmission
        )
        strictly_better = (
            self.makespan < other.makespan
            or self.carbonEmission < other.carbonEmission
        )

        return no_worse and strictly_better

    # String
    def __str__(self):
        lines = [
            f"Makespan = {self.makespan:.2f}",
            f"Carbon   = {self.carbonEmission:.2f}",
            f"Feasible = {self.isFeasible}",
        ]

        if not self.isFeasible:
            lines.append(
                f"Violated constraints: {sorted(self.violatedConstraints)}"
            )

        for i, route in enumerate(self.truckRoutes):
            lines.append(f"Truck {i + 1}: {route}")
            if i < len(self.droneCustomers):
                lines.append(
                    f"Drone {i + 1}: {sorted(self.droneCustomers[i])}"
                )

        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()
