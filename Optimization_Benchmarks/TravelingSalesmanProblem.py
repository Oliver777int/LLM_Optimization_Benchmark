import random
import math

from Optimization_Benchmarks.Base import ProblemSolution
from typing import List, Dict, Tuple, Optional, Union

class TSPInstance():
    def __init__(self, NUMBER_OF_NODES, SIZE, SEED) -> None:
        #super().__init__()
        self.NUMBER_OF_NODES = NUMBER_OF_NODES
        self.SIZE = SIZE
        self.SEED = SEED

        # Generates a graph as a dict where keys are the nodes and values are the coordinates.
        random.seed(SEED)
        self.graph = self.generate_graph()
        random.seed(None)
        self.nodes = list(self.graph.keys())
        self.coordinates = self.get_coordinates()

    def generate_graph(self) -> Dict[int, Tuple[int, int]]:
        """
        Creates a graph that defines the traveling salesman problem instance. NUMBER_OF_NODES nodes are scattered over the
        grid of size [-SIZE, SIZE]. The coordinates of these nodes are then stored in the graph dict.
        """
        if self.NUMBER_OF_NODES <= 0 or self.SIZE <= 0:
            raise ValueError("Number of nodes and size must be positive integers.")
        
        graph = {}
        for node in range(0, self.NUMBER_OF_NODES):
            x_coordinate = random.randint(-self.SIZE, self.SIZE)
            y_coordinate = random.randint(-self.SIZE, self.SIZE)
            graph[node] = (x_coordinate, y_coordinate)
        return graph

    def get_coordinates(self) -> str:
        """ Gets the coordinates as a string to be put in the meta-prompt"""
        result_string = ""
        for n, c in self.graph.items():
            result_string += f"{n}: {c}, "
        return result_string[:-2]
    
    def distance(self, node1, node2) -> float:
        """Calculates the Euclidian distance between two nodes."""
        x1, y1 = node1
        x2, y2 = node2
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def nearest_neighbor(self, start_node: int) -> List[int]:
        """
        Computes the nearest neighbor solution given a specified start node. This method is used as a comparison evaluation only.
        """
        unvisited_nodes = self.nodes.copy()
        current_node = start_node
        path = [current_node]
        unvisited_nodes.remove(current_node)

        while unvisited_nodes:
            nearest_node = min(unvisited_nodes, key=lambda node: self.distance(self.tsp_graph[current_node], self.tsp_graph[node]))
            path.append(nearest_node)
            unvisited_nodes.remove(nearest_node)
            current_node = nearest_node
        return path
    
    def get_distance_matrix(self) -> List[List[float]]:
        """
        Gets the distance matrix of the TSP instance.
        """
        dist_matrix = [[0.0] * self.NUMBER_OF_NODES for _ in range(self.NUMBER_OF_NODES)]
        for i in range(self.NUMBER_OF_NODES):
            for j in range(self.NUMBER_OF_NODES):
                dist_matrix[i][j] = round(self.distance(self.graph[i], self.graph[j]), 4)
        return dist_matrix

class TSPSolution(ProblemSolution):
    path: Optional[List[int]] = None
    problem_instance: TSPInstance

    def __str__(self) -> str:
        return f"<trace> {','.join(map(str, self.path))} </trace> length: {self.score}"

    def evaluate(self, response: str) -> Union[float, None]:
        """
        Given a response string from the LLM, evaluate the solution and return the score. If the solution is not valid/feasible, return None.
        """
        if self.isValid(response):
            return self.compute_score()
    
    def isValid(self, response: str) -> bool:
        trace_len = len("<trace>")
        idx_1 = response.rfind("<trace>")
        if idx_1 == -1:# or response.count("<trace>") > 1:
            return False
        idx_2 = response.rfind("</trace>", idx_1 + trace_len)
        if idx_2 == -1:# or response.count("</trace>") > 1:
            return False
        response = response[idx_1:idx_2 + trace_len]
        path = response[idx_1+trace_len:idx_2].split(',')
        try:
            self.path = [int(i) for i in path]
            if len(self.path) < 2:
                return False
            if len(self.path) == self.problem_instance.NUMBER_OF_NODES+1:
                    self.path = self.path[:-1]
            if len(self.path) != self.problem_instance.NUMBER_OF_NODES:
                return False
            for i in self.path:
                if i < 0 or i >= self.problem_instance.NUMBER_OF_NODES:
                    return False
        except ValueError:
            return False
        return True
    
    def compute_score(self) -> float:
        total_length = 0
        graph = self.problem_instance.graph
        for i in range(len(self.path) - 1):
            current_node = self.path[i]
            next_node = self.path[i + 1]

            current_coord = graph[current_node]
            next_coord = graph[next_node]

            distance = math.sqrt((next_coord[0] - current_coord[0]) ** 2 + (next_coord[1] - current_coord[1]) ** 2)
            total_length += distance

        # Add the length of traveling from the last node back to the first
        last_node = self.path[-1]
        first_node = self.path[0]
        last_coord = graph[last_node]
        first_coord = graph[first_node]
        return round(total_length + math.sqrt((first_coord[0] - last_coord[0]) ** 2 + (first_coord[1] - last_coord[1]) ** 2)) 
    
    def random_init(self):
        self.path = list(self.problem_instance.nodes)
        random.shuffle(self.path)
        self.score = self.compute_score()
        return self
