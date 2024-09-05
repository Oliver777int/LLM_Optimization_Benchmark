import random
import math
import json
import re
from typing import Dict, Any, Union, Optional
from Optimization_Benchmarks.Base.ProblemSolution import ProblemSolution

class VRPInstance:
    def __init__(self, NUMBER_OF_NODES, SIZE, VEHICLES, DELIVERY_REQUIREMENTS, weight_factor, revenue_per_container, SEED):
        self.NUMBER_OF_NODES = NUMBER_OF_NODES
        self.SIZE = SIZE
        self.VEHICLES = VEHICLES
        self.DELIVERY_REQUIREMENTS = DELIVERY_REQUIREMENTS
        self.weight_factor = weight_factor
        self.r = revenue_per_container
        self.SEED = SEED
        
        self.graph = self.generate_graph()
        self.nodes = list(self.graph.keys())
        self.coordinates_string = self.get_coordinates()
        self.cost_matrix = self.generate_cost_matrix()

    def generate_graph(self):
        random.seed(self.SEED)
        if self.NUMBER_OF_NODES <= 0 or self.SIZE <= 0:
            raise ValueError("Number of nodes and size must be positive integers.")
        graph = {}
        for node in range(0, self.NUMBER_OF_NODES):
            x_coordinate = random.randint(-self.SIZE, self.SIZE)
            y_coordinate = random.randint(-self.SIZE, self.SIZE)
            graph[node] = (x_coordinate, y_coordinate)
        random.seed(None)
        return graph
    
    def generate_cost_matrix(self):
        dist_matrix = [[0.0] * self.NUMBER_OF_NODES for _ in range(self.NUMBER_OF_NODES)]
        for i in range(self.NUMBER_OF_NODES):
            for j in range(self.NUMBER_OF_NODES):
                dist_matrix[i][j] = round(self.distance(self.graph[i], self.graph[j]))
    
    def get_coordinates(self):
        result_string = ""
        for n, c in self.graph.items():
            result_string += f"{n}: {c}, "
        return result_string[:-2]
    
    def distance(self, node1, node2):
        x1, y1 = node1
        x2, y2 = node2
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


class VRPSolutionCostMinimization(ProblemSolution):
    problem_instance: VRPInstance
    delivery_plan: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"delivery_plan: {self.delivery_plan}\nCost: {self.score}"
    
    def evaluate(self, response: str) -> Union[float, None]:
        """
        Given a response string from the LLM, evaluate the solution and return the score. If the solution is not valid/feasible, return None.
        """
        self.delivery_plan = self.parse(response)
        
        # Return null if parsing failed.
        if not self.delivery_plan:
            return

        # If solution is valid compute its score.
        if self.isValid(response):
            return self.compute_cost()
    
    def distance(self, node1, node2) -> float:
        x1, y1 = self.problem_instance.graph[node1]
        x2, y2 = self.problem_instance.graph[node2]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def compute_cost(self) -> int:
        total_cost = 0
        for vehicle in self.delivery_plan["vehicles"]:
            deliveries = vehicle["deliveries"]
            current_load = sum(deliveries.values())
            current_node = 0
            for delivery_node, weight in deliveries.items():
                delivery_node = int(re.sub(r'\D', '', delivery_node))
                total_cost += self.distance(current_node, delivery_node) * (1 + current_load)
                current_load -= weight
                current_node = delivery_node
            total_cost += self.distance(current_node, 0) * (1 + current_load)
        return int(total_cost)
    
    def parse(self, response: str) -> Union[Dict[str, Any], None]:
        response = response.lower()
        # Find the contents of the json using markdown
        response = response.replace("\\","")
        start_marker = "```json"
        end_marker = "```"
        start_index = response.find(start_marker) + len(start_marker)
        response = response[start_index:]
        end_index = response.find(end_marker)
        if start_index == -1 or end_index == -1:
            return None
        parenthesis_index = response.find('{')
        response = response[parenthesis_index:end_index].strip() # Extracts the json without "delivery_plan = "

        # Try to turn the parsed response into a json (delivery_plan)
        try:
            response = response.replace("'", '"')
            response = json.loads(response)
            return response
        except json.JSONDecodeError:
            self.feasibility_violation = "Incorrect response format"
            return None
    
    def isValid(self) -> bool:
        try:
            feasible = self.format_feasibility()
            if feasible:
                feasible = self.capacity_feasibility()
        except Exception as e:
            self.feasibility_violation = str(e)
            return False
        return feasible
    
    def format_feasibility(self) -> bool:
        if not isinstance(self.delivery_plan, dict) or 'vehicles' not in self.delivery_plan:
            self.feasibility_violation = "Format violation 1"
            return False

        required_keys = {"vehicle_name", "deliveries"}
        for vehicle in self.delivery_plan["vehicles"]:
            try:
                if not set(vehicle.keys()) == required_keys:
                    self.feasibility_violation = "Delivery plan keys format violation"
                    return False
                
                if not isinstance(vehicle["vehicle_name"],str):
                    self.feasibility_violation = "Vehicle id format violation"
                    return False
                
                for delivery in vehicle["deliveries"].items():
                    if not isinstance(delivery, tuple) or not isinstance(delivery[0],str) or not isinstance(delivery[1],int):
                        self.feasibility_violation = "Deliveries format violation"
                        return False 
                    
            except Exception as e:
                self.feasibility_violation = "Format violation 2" + str(e)
                return False
        return True

    def capacity_feasibility(self) -> bool:
        delivery_requirements = self.problem_instance.DELIVERY_REQUIREMENTS.copy()
        for vehicle in self.delivery_plan["vehicles"]:
            delivery_amount = sum([delivery for delivery in vehicle['deliveries'].values()])
            if delivery_amount > self.problem_instance.VEHICLES[vehicle["vehicle_name"]]:
                self.feasibility_violation = f'Vehicle {vehicle["vehicle_name"]} maximum load capacity violated'
                return False
            
            for delivery, amount in vehicle['deliveries'].items():
                    delivery_node = int(delivery[-1])
                    delivery_requirements[delivery_node] -= amount

        # Requires all deliveries to be fulfilled.
        if any(delivery_requirements):
            self.feasibility_violation = "Not all deliveries are fulfilled"
            return False

        if len(self.delivery_plan["vehicles"]) != len(self.VEHICLES):
            self.feasibility_violation = "Vehicles can only be used for a single route. The provided solution uses the same vehicle more than once."
            return False
        return True


class VRPSolutionRevenueMaximization(VRPSolutionCostMinimization):
    """
    This solution class is almost identical to the Cost Minimization VRP Solution, except the score is now the revenue of all vehicles. The Revenue is computed
    using: Revenue = number_of_deliveries * costant - total_cost_of_delivery. This solution class also accepts solutions which do not visit all customers, i.e,
    it is much less constrained compared to total cost minimization.
    """
    tsp_problem_instance: VRPInstance
    delivery_plan: Dict[str, Any] = None
    total_containers_delivered: Optional[int] = None

    def __str__(self) -> str:
        return f"delivery_plan: {self.delivery_plan}\nRevenue: {self.score}"
    
    def evaluate(self, response: str) -> Union[float, None]:
        """
        Given a response string from the LLM, evaluate the solution and return the score. If the solution is not valid/feasible, return None.
        """
        self.delivery_plan = self.parse(response)
        
        # Return null if parsing failed.
        if not self.delivery_plan:
            return

        # If solution is valid compute its score.
        if self.isValid(response):
            return int(self.compute_revenue())

    def isValid(self) -> bool:
        try:
            feasible = self.format_feasibility()
            if feasible:
                feasible = self.capacity_feasibility()
        except Exception as e:
            self.feasibility_violation = str(e)
            return False
        return feasible

    def compute_revenue(self) -> float:
        self.revenue = self.tsp_problem_instance.r * self.total_containers_delivered - self.compute_cost()

    def capacity_feasibility(self) -> bool:
        delivery_requirements = self.problem_instance.DELIVERY_REQUIREMENTS.copy()
        for vehicle in self.delivery_plan["vehicles"]:
            delivery_amount = sum([delivery for delivery in vehicle['deliveries'].values()])
            if delivery_amount > self.problem_instance.VEHICLES[vehicle["vehicle_name"]]:
                self.feasibility_violation = f'Vehicle {vehicle["vehicle_name"]} maximum load capacity violated'
                return False
            
            for delivery, amount in vehicle['deliveries'].items():
                    delivery_node = int(delivery[-1])

                    if delivery_requirements[delivery_node] >= amount:
                        self.total_containers_delivered += amount

                    delivery_requirements[delivery_node] -= amount
        
        if len(self.delivery_plan["vehicles"]) != len(self.VEHICLES):
            self.feasibility_violation = "Vehicles can only be used for a single route. The provided solution uses the same vehicle more than once."
            return False
        return True
