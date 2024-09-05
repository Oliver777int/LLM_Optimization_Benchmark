import random
from typing import List
from Optimization_Benchmarks.Base.ProblemSolution import ProblemSolution

class SolutionScoreHistory:
    def __init__(self, maximize_score: bool, max_solutions: int, shuffling: bool, SEED=None):
        self.maximize_score = maximize_score
        self.max_solutions = max_solutions
        self.shuffling = shuffling
        self.SEED = SEED

        self.solution_history = []
        self.score_history = set()
        random.seed(self.SEED)
        
    def append_list(self, solutions: List[ProblemSolution]) -> None:
        for solution in solutions:
            self.append(solution)

    def append(self, solution: ProblemSolution):
        score = int(solution.score)
        if score not in self.score_history:
            self.solution_history.append(solution)
            self.score_history.add(solution.score)
            self.sort_history()
            if self.shuffling:
                self.shuffle_history()

    def sort_history(self):
        sorted_history = sorted(self.solution_history, key=lambda x: x.score, reverse=not self.maximize_score) # Sort history based on Score
        self.solution_history = sorted_history[-self.max_solutions:] # Keep the top n solutions where n = max_solutions.

    def shuffle_history(self):
        random.shuffle(self.solution_history)
    
    def get_best_scored_solution(self):
        self.sort_history()
        return self.solution_history[-1]
    
    def get_solution_history(self) -> str:
        solution_score_history_as_string = ""
        for solution in self.solution_history:
            solution_score_history_as_string += str(solution) + "\n"
        return solution_score_history_as_string


