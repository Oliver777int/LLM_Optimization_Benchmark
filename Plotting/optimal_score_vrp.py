"""Capacited Vehicles Routing Problem (CVRP). Source: https://developers.google.com/optimization/routing/vrp"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
#from vrp import *

def create_data_model(vrp):
    """Stores the data for the problem."""
    data = {}
    data["demands"] = vrp.DELIVERY_REQUIREMENTS
    data["vehicle_capacities"] = list(vrp.VEHICLES.values())
    data["num_vehicles"] = len(vrp.VEHICLES)
    data["depot"] = 0
    data["distance_matrix"] = vrp.cost_matrix
    return data

def get_total_load(data, manager, routing, solution):
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            index = solution.Value(routing.NextVar(index))
        total_load += route_load
    return total_load

def get_optimal_score(objective, vrp):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model(vrp)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    optimal_cost = solution.ObjectiveValue()
    if objective == "min":
        return optimal_cost
    else:
        optimal_revenue = vrp.r * get_total_load(data, manager, routing, solution) - optimal_cost
        return optimal_revenue


if __name__ == "__main__":
    objective = "max"
    delivery_requirements = [0, 2, 2, 4, 6, 2, 4, 8, 2]
    #vrp =VehicleRoutingProblem(NUMBER_OF_NODES=9, SIZE=100, VEHICLES={"a":20, "b":16}, DELIVERY_REQUIREMENTS=delivery_requirements, SEED=1, weight_factor=0, revenue_per_container=100)
    #print(get_optimal_score(objective, vrp))