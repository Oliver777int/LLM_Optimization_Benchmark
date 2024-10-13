"""
This Script Plots the optimality gap of multiple Traveling salesman experiments of different number of nodes. To run this script,
make sure to first run run_tsp.py for some different number of nodes, example: n = 7, 10, 15. Make sure the results are stored as json in the results folder.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Plotting.optimal_score_tsp import OptimalTSPScore
from Plotting.optimal_score_vrp import OptimalVRPScore
from Plotting.generate_tsp_table import create_table
from Optimization_Benchmarks.TravelingSalesmanProblem import TSPSolution, TSPInstance
from Optimization_Benchmarks.VehicleRoutingProblem import VRPInstance

def get_optimal_score(problem_name, data, node, size, seed) -> int:
    if problem_name == "tsp":
        return OptimalTSPScore(node, size, seed)
    if problem_name == "vrp":
        vrp = VRPInstance(node, size, data.get("vehicles"), data.get("delivery_requirements"), 0, data.get("revenue_per_container"), seed)
        return OptimalVRPScore(data.get("objective"), vrp)
    raise ValueError("Can only plot results of tsp and vrp")

def Import_Data(folder_path: str, model: str = "", problem_name: str = ""):
    nodes, mean_optimality_gaps, sem_optimality_gaps, mean_feasibility_rates, sem_feasibility_rates = [], [], [], [], []

    for filename in os.listdir(folder_path):
        if problem_name not in filename or model not in filename:
            continue
        
        with open(os.path.join(folder_path, filename), 'r') as file:
            data = json.load(file)
        
        node = int(data.get("nodes"))
        nodes.append(node)
        runs = data.get("runs")
        seeds = data.get("seeds")
        size = data.get("size") or 100
        max_iterations = data.get("max_iterations")

        optimality_gaps = []
        feasibility_rates = []
        for idx, run in enumerate(runs):
            scores_i = []
            feasible = 0
            successful_iterations = set()
            
            for iteration in run:
                successful_iterations.add(iteration.get("iteration"))
                # Extracts the score of run i and iteration j.
                score_i_j = iteration.get("score")
                scores_i.append(score_i_j)
            feasibility_rates.append(100 * len(successful_iterations)/(max(successful_iterations)+1))
            min_score = min(scores_i)
            optimal_score = get_optimal_score(problem_name, data, node, size, seeds[idx])
            optimality_gap = 100 * (min_score - optimal_score)/optimal_score
            optimality_gaps.append(optimality_gap)

        mean_feasibility_rates.append(np.mean(feasibility_rates))
        sem_feasibility_rates.append(np.std(feasibility_rates) / np.sqrt(len(feasibility_rates)))
        mean_optimality_gaps.append(np.mean(optimality_gaps))
        sem_optimality_gaps.append(np.std(optimality_gaps) / np.sqrt(len(optimality_gaps)))

        file_info = filename.replace('.json', '').split('_')
        print(f"Mean Optimal Score of {int(file_info[1].replace('node',''))} node {problem_name} using {file_info[3]}: {round(np.mean(optimality_gaps),3)} % +- {round(np.std(optimality_gaps) / np.sqrt(len(optimality_gaps)),2)} and Feasibility rate was {np.mean(feasibility_rates)} +- {np.std(feasibility_rates) / np.sqrt(len(feasibility_rates))}")
    return nodes, mean_optimality_gaps, sem_optimality_gaps, mean_feasibility_rates, sem_feasibility_rates

def get_models(results_path) -> set:
    models = set()
    for filename in os.listdir(results_path):
        try:
            models.add(filename.split('_')[3].replace(".json", ""))
        except:
            pass
    return models

def get_seed_averaged_nearest_neighbour_opt_gap(nodes, size, seeds):
    avg_nn_opt_gaps = []
    for seed in seeds:
        tsp = TSPInstance(nodes, size, seed)
        optimal = OptimalTSPScore(nodes, size, seed)
        nn_scores = []
        for start_node in range(nodes):
            nn_sol = TSPSolution(problem_instance=tsp, path=tsp.nearest_neighbor(start_node))
            nn_scores.append(nn_sol.compute_score())
        avg_nn_opt_gap = (np.mean(nn_scores) - optimal) / optimal
        avg_nn_opt_gaps.append(avg_nn_opt_gap)
    return np.mean(avg_nn_opt_gaps)

def generate_tsp_plot(results_path: str) -> None:
    # Plot Nearest neighbour and Random.
    n_values = [7, 10, 15]
    random_mean = [8.0, 26.2, 69.3]
    random_std = [3.2, 2.4, 9.3]
    nn_mean = [6.93, 10.9, 19.0]
    nn_std = [2.67, 5.9, 8.95]

    plt.errorbar(n_values, random_mean, yerr=random_std, label='Random', marker='o',capsize=5, capthick=1.5)
    plt.errorbar(n_values, nn_mean, yerr=nn_std, label='Nearest Neighbor (NN)', marker='o',capsize=5, capthick=1.5)

    # Plot LLM results
    new_opt_gaps = []
    for model in get_models(results_path):
        nodes, mean_optimality_gaps, sem_optimality_gaps, _, _ = Import_Data(results_path, model=model, problem_name="tsp")
        plt.errorbar(nodes, mean_optimality_gaps, yerr=sem_optimality_gaps, label=model, marker='o', capsize=5, capthick=1.5)
        new_opt_gaps.append(mean_optimality_gaps) # Used for creating the table

    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Optimality Gap (%)')
    plt.title('Traveling Salesman - Optimality Gap Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_path}/tsp_optimality_gap.png")

    # Return if no tsp experiments exists
    if not nodes:
        plt.clf()
        return
    
    if __name__ == "__main__":
        plt.show()
    
    plt.clf()
    for model in get_models(results_path):
        nodes, _, _, mean_feasibility, sem_feasibility = Import_Data(results_path, model=model, problem_name="tsp")
        plt.errorbar(nodes, mean_feasibility, yerr=sem_feasibility, label=model, marker='o', capsize=5, capthick=1.5)
    
    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Feasibility rate (%)')
    plt.title('Feasibility rate Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_path}/tsp_feasibility.png")

    create_table(results_path, list(get_models(results_path)), new_opt_gaps)
    if __name__ == "__main__":
        plt.show()

def generate_vrp_plot(results_path: str) -> None:
    # Plot Nearest neighbour and Random.
    n_values = [5, 7, 9]

    # Plot LLM results
    new_opt_gaps = []
    for model in get_models(results_path):
        nodes, mean_optimality_gaps, sem_optimality_gaps, _, _ = Import_Data(results_path, model=model, problem_name="vrp")
        plt.errorbar(nodes, mean_optimality_gaps, yerr=sem_optimality_gaps, label=model, marker='o', capsize=5, capthick=1.5)
        new_opt_gaps.append(mean_optimality_gaps) # Used for creating the table

    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Optimality Gap (%)')
    plt.title('Vehicle routing - Optimality Gap Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_path}/vrp_optimality_gap.png")

    # Return if no vrp experiments exists
    if not nodes:
        plt.clf()
        return
    
    if __name__ == "__main__":
        plt.show()
    
    plt.clf()
    for model in get_models(results_path):
        nodes, _, _, mean_feasibility, sem_feasibility = Import_Data(results_path, model=model, problem_name="vrp")
        plt.errorbar(nodes, mean_feasibility, yerr=sem_feasibility, label=model, marker='o', capsize=5, capthick=1.5)
    
    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Feasibility rate (%)')
    plt.title('Feasibility rate Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_path}/vrp_feasibility.png")

if __name__ == "__main__":
    results_path = '../results_vrp'
    generate_tsp_plot(results_path)
    generate_vrp_plot(results_path)
    plt.show()