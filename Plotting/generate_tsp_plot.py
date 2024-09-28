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

from Plotting.optimal_score_tsp import OptimalTSPScore  # If running from outside the Plotting
from Optimization_Benchmarks.TravelingSalesmanProblem import TSPSolution, TSPInstance

def Import_Data(folder_path: str, model: str = ""):
    nodes, mean_optimality_gaps, std_optimality_gaps = [], [], []

    for filename in os.listdir(folder_path):
        if "tsp" not in filename or model not in filename:
            continue
        
        with open(os.path.join(folder_path, filename), 'r') as file:
            data = json.load(file)
        
        node = int(data.get("nodes"))
        nodes.append(node)
        runs = data.get("runs")
        seeds = data.get("seeds")

        optimality_gaps = []
        for idx, run in enumerate(runs):
            scores_i = []
            for iteration in run:
                # Extracts the score of run i and iteration j.
                score_i_j = iteration.get("score")
                scores_i.append(score_i_j)
            min_score = min(scores_i)
            optimal_score = OptimalTSPScore(node, 100, seeds[idx])
            optimality_gap = 100 * (min_score - optimal_score)/optimal_score
            optimality_gaps.append(optimality_gap)

        mean_optimality_gaps.append(np.mean(optimality_gaps))
        std_optimality_gaps.append(np.std(optimality_gaps))

        file_info = filename.replace('.json', '').split('_')
        print(f"Mean Optimal Score of {int(file_info[1].replace('node',''))} node tsp using {file_info[3]}: {round(np.mean(optimality_gaps),3)} %")
    return nodes, mean_optimality_gaps, std_optimality_gaps

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

def generate_plot(results_path: str) -> None:
    # Plot Nearest neighbour and Random.
    n_values = [7, 10, 15]
    random_mean = [8.0, 26.2, 69.3]
    random_std = [3.2, 2.4, 9.3]
    nn_mean = [6.93, 10.9, 19.0]
    nn_std = [2.67, 5.9, 8.95]

    plt.errorbar(n_values, random_mean, yerr=random_std, label='Random', marker='o',capsize=5, capthick=1.5)
    plt.errorbar(n_values, nn_mean, yerr=nn_std, label='Nearest Neighbor (NN)', marker='o',capsize=5, capthick=1.5)

    # Plot LLM results
    for model in get_models(results_path):
        nodes, mean_optimality_gaps, std_optimality_gaps = Import_Data(results_path, model=model)
        plt.errorbar(nodes, mean_optimality_gaps, yerr=std_optimality_gaps, label='Mixtral 8x7B', marker='o', capsize=5, capthick=1.5)

    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Optimality Gap (%)')
    plt.title('Optimality Gap Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_path}/tsp_results.png")

if __name__ == "__main__":
    results_path = '../results'
    generate_plot(results_path)
    plt.show()