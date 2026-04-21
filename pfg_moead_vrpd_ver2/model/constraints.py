# model/constraints.py

from typing import Dict
from .solution import Solution
from .customer import Customer


class Constraints:
    """
    HARD constraints for VRP-D (Truck–Drone).
    Any violation => infeasible solution.
    """

    # ======================================================
    @staticmethod
    def truck_capacity(
        sol: Solution,
        customers: Dict[int, Customer],
        truck_capacity: float
    ) -> bool:
        """
        C1 – Truck capacity constraint:
        Tổng demand các khách do truck phục vụ không vượt quá tải trọng truck.
        """
        for t, route in enumerate(sol.truckRoutes):
            load = 0.0
            drone_set = sol.droneCustomers[t]

            for cid in route:
                if cid != 0 and cid not in drone_set:
                    load += customers[cid].demand

            if load > truck_capacity:
                return False
        return True

    # ======================================================
    @staticmethod
    def drone_capacity(
        sol: Solution,
        customers: Dict[int, Customer],
        drone_capacity: float
    ) -> bool:
        """
        C2 – Drone payload constraint:
        Demand của mỗi khách drone không vượt quá tải drone.
        """
        for drone_set in sol.droneCustomers:
            for cid in drone_set:
                if customers[cid].demand > drone_capacity:
                    return False
        return True

    # ======================================================
    @staticmethod
    def unique_service(sol: Solution) -> bool:
        """
        C3 – Unique service constraint:
        Mỗi khách chỉ được phục vụ đúng một lần (truck hoặc drone).
        """
        served = set()

        for t, route in enumerate(sol.truckRoutes):
            for cid in route:
                if cid == 0:
                    continue
                if cid in served:
                    return False
                served.add(cid)

            for cid in sol.droneCustomers[t]:
                if cid in served:
                    return False
                served.add(cid)

        return True

    # ======================================================
    @staticmethod
    def drone_only_if_on_route(sol: Solution) -> bool:
        """
        C4 – Drone-on-route constraint:
        Khách do drone phục vụ phải xuất hiện trong route truck tương ứng.
        """
        for t, route in enumerate(sol.truckRoutes):
            route_set = set(route)
            for cid in sol.droneCustomers[t]:
                if cid not in route_set:
                    return False
        return True

    # ======================================================
    @staticmethod
    def non_empty_route(sol: Solution) -> bool:
        """
        C5 – Valid route constraint:
        Route truck phải bắt đầu và kết thúc tại depot (0).
        """
        for route in sol.truckRoutes:
            if len(route) < 2:
                return False
            if route[0] != 0 or route[-1] != 0:
                return False
        return True

    # ======================================================
    @staticmethod
    def max_one_drone_per_truck(sol: Solution) -> bool:
        """
        C6 – Single drone per truck:
        Mỗi truck chỉ có đúng một tập droneCustomers (đã đảm bảo bởi encoding).
        """
        return len(sol.droneCustomers) == len(sol.truckRoutes)

    # ======================================================
    @staticmethod
    def drone_customer_eligibility(
        sol: Solution,
        customers: Dict[int, Customer]
    ) -> bool:
        """
        C7 – Drone eligibility constraint:
        Drone chỉ được phục vụ khách có drone_serve = True.
        """
        for drone_set in sol.droneCustomers:
            for cid in drone_set:
                if not customers[cid].drone_serve:
                    return False
        return True

    # ======================================================
    @staticmethod
    def no_duplicate_drone_customers(sol: Solution) -> bool:
        """
        C8 – No duplicate drone service:
        Một khách không được drone phục vụ nhiều hơn một lần.
        """
        seen = set()
        for drone_set in sol.droneCustomers:
            for cid in drone_set:
                if cid in seen:
                    return False
                seen.add(cid)
        return True

    # ======================================================
    @staticmethod
    def one_drone_flight_one_customer(sol: Solution) -> bool:
        """
        C9 – One-customer-per-flight constraint:
        Drone mỗi truck chỉ phục vụ tập khách rời rạc (không lặp, không chuỗi).
        → Ánh xạ bằng việc không cho phép thứ tự / lặp.
        """
        for drone_set in sol.droneCustomers:
            if len(drone_set) != len(set(drone_set)):
                return False
        return True

    # ======================================================
    @staticmethod
    def drone_not_serving_depot(sol: Solution) -> bool:
        """
        C10 – Drone depot constraint:
        Drone không được phục vụ depot (0).
        """
        for drone_set in sol.droneCustomers:
            if 0 in drone_set:
                return False
        return True

    # ======================================================
    @staticmethod
    def truck_route_no_duplicate(sol: Solution) -> bool:
        """
        C11 – Truck no-duplicate constraint:
        Trong route truck không được lặp khách.
        """
        for route in sol.truckRoutes:
            seen = set()
            for cid in route:
                if cid == 0:
                    continue
                if cid in seen:
                    return False
                seen.add(cid)
        return True

    # ======================================================
    @staticmethod
    def all_customers_served(
        sol: Solution,
        customers: Dict[int, Customer]
    ) -> bool:
        """
        C12 – Complete service constraint:
        Tất cả khách (trừ depot) phải được phục vụ.
        """
        served = set()

        for t, route in enumerate(sol.truckRoutes):
            for cid in route:
                if cid != 0:
                    served.add(cid)
            for cid in sol.droneCustomers[t]:
                served.add(cid)

        for cid in customers:
            if cid != 0 and cid not in served:
                return False

        return True

    # ======================================================
    @staticmethod
    def drone_subset_of_customers(
        sol: Solution,
        customers: Dict[int, Customer]
    ) -> bool:
        """
        C13 – Drone subset constraint:
        Drone chỉ được phục vụ các khách tồn tại trong tập khách.
        """
        all_customers = set(customers.keys())
        for drone_set in sol.droneCustomers:
            if not drone_set.issubset(all_customers):
                return False
        return True

    # ======================================================
    @staticmethod
    def truck_subset_of_customers(
        sol: Solution,
        customers: Dict[int, Customer]
    ) -> bool:
        """
        C14 – Truck subset constraint:
        Truck chỉ được đi qua các khách tồn tại trong tập khách.
        """
        all_customers = set(customers.keys())
        for route in sol.truckRoutes:
            for cid in route:
                if cid not in all_customers:
                    return False
        return True

    # ======================================================
    @staticmethod
    def no_empty_solution(sol: Solution) -> bool:
        """
        C15 – Non-empty solution constraint:
        Phải có ít nhất một khách được phục vụ.
        """
        for route in sol.truckRoutes:
            if len(route) > 2:
                return True
        return False

    # ======================================================
    @staticmethod
    def encoding_consistency(sol: Solution) -> bool:
        """
        C16 – Encoding consistency constraint:
        Số truckRoutes phải khớp với số droneCustomers.
        """
        return len(sol.truckRoutes) == len(sol.droneCustomers)

    # ======================================================
    @staticmethod
    def check_all(
        sol: Solution,
        customers: Dict[int, Customer],
        truck_capacity: float,
        drone_capacity: float
    ) -> bool:
        """
        Aggregate all HARD constraints (C1 – C16).
        """
        return (
            Constraints.truck_capacity(sol, customers, truck_capacity)
            and Constraints.drone_capacity(sol, customers, drone_capacity)
            and Constraints.unique_service(sol)
            and Constraints.drone_only_if_on_route(sol)
            and Constraints.non_empty_route(sol)
            and Constraints.max_one_drone_per_truck(sol)
            and Constraints.drone_customer_eligibility(sol, customers)
            and Constraints.no_duplicate_drone_customers(sol)
            and Constraints.one_drone_flight_one_customer(sol)
            and Constraints.drone_not_serving_depot(sol)
            and Constraints.truck_route_no_duplicate(sol)
            and Constraints.all_customers_served(sol, customers)
            and Constraints.drone_subset_of_customers(sol, customers)
            and Constraints.truck_subset_of_customers(sol, customers)
            and Constraints.no_empty_solution(sol)
            and Constraints.encoding_consistency(sol)
        )
