"""Simple Travelling Salesperson Problem (TSP) between cities. Source: https://developers.google.com/optimization/routing/tsp"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from Optimization_Benchmarks.TravelingSalesmanProblem import TSPInstance

def create_data_model(tsp: TSPInstance):
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = tsp.get_distance_matrix()
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()} miles")
    index = routing.Start(0)
    plan_output = "Route for vehicle 0:\n"
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += f" {manager.IndexToNode(index)} ->"
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += f" {manager.IndexToNode(index)}\n"
    print(plan_output)
    plan_output += f"Route distance: {route_distance}miles\n"


def OptimalTSPScore(number_of_nodes: str, size: int, seed: str) -> int:
    """Entry point of the program."""
    # Instantiate the data problem.
    tsp = TSPInstance(number_of_nodes, size, seed)
    data = create_data_model(tsp)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    #if solution:
    #    print_solution(manager, routing, solution)
    
    return solution.ObjectiveValue()


if __name__ == "__main__":
    OptimalTSPScore(7, 100, 41)