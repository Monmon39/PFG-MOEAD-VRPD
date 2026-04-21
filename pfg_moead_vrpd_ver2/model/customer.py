# model/customer.py

class Customer:
    """
    Customer model for VRP-D
    Compatible with CSV dataset and solvers
    """

    def __init__(
        self,
        cid: int,
        x: float,
        y: float,
        demand: float,
        ready_time: float,
        due_time: float,
        service_time: float,
        drone_serve: bool,
        time: float = 0.0,
    ):
        self.id = cid
        self.x = x
        self.y = y
        self.demand = demand

        # Time window
        self.ready_time = ready_time
        self.due_time = due_time

        # Service time
        self.service_time = service_time

        # Drone eligibility
        self.drone_serve = drone_serve

        # Extra time attribute (drone / delay / release time)
        self.time = time

    # Distance (Manhattan)
    def distance_to(self, other: "Customer") -> float:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __str__(self):
        return f"C[{self.id}](x={self.x}, y={self.y})"

    def __repr__(self):
        return self.__str__()
