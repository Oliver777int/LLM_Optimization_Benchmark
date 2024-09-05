"""
This Script Plots the optimality gap of multiple Traveling salesman experiments of different number of nodes. To run this script,
make sure to first run run_tsp.py for some different number of nodes, example: n = 7, 10, 15. Make sure the results are stored as json in the results folder.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from optimal_score_tsp import OptimalTSPScore
import os
import json
results_path = '../results'

def Import_Data(folder_path: str):
    nodes = []
    mean_optimal_scores = []
    std_optimal_scores = []
    seeds = []
    for filename in os.listdir(folder_path):
        print(f"File: {filename}")
        if "tsp" not in filename:
            print(f"Passing {filename}")
            continue

        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        nodes.append(int(data.get("nodes")))
        seeds.append(data.get("seed"))
        runs = data.get("runs")
        scores = []
        for run in runs:
            scores_i = []
            for iteration in run:
                # Extracts the score of run i and iteration j.
                score_i_j = iteration.get("score")
                scores_i.append(score_i_j)
            min_score = min(scores_i)
            scores.append(min_score)
        mean_optimal_score = np.mean(scores)
        mean_optimal_scores.append(mean_optimal_score)
        std_optimal_score = np.std(scores)
        std_optimal_scores.append(std_optimal_score)
        print(f"Mean Optimal Score: {mean_optimal_score}")
    return nodes, mean_optimal_scores, std_optimal_scores, seeds

# Import Experiment data
nodes, mean_optimal_scores, std_optimal_scores, seeds = Import_Data(results_path)

# Compute Optimal Score to get Optimality gap. This is done for each problem instance.
mean_optimality_gaps = []
std_optimality_gaps = []
for idx, node in enumerate(nodes):
    optimal_solution = OptimalTSPScore(node, 100, seeds[idx])
    mean_optimality_gap = (mean_optimal_scores[idx] - optimal_solution)/optimal_solution
    mean_optimality_gaps.append(mean_optimality_gap)
    std_optimality_gaps.append(std_optimal_scores[idx] / mean_optimal_scores[idx])

# Plotting

#plt.errorbar(n_values, random_mean, yerr=random_std, label='Random', marker='o',capsize=5, capthick=1.5)
plt.errorbar(nodes, mean_optimality_gaps, yerr=std_optimality_gaps, label='LLM Optimization', marker='o', capsize=5, capthick=1.5)
#plt.errorbar(n_values, nn_mean, yerr=nn_std, label='Nearest Neighbor (NN)', marker='o',capsize=5, capthick=1.5)

plt.xlabel('Number of Nodes (n)')
plt.ylabel('Optimality Gap (%)')
plt.title('Optimality Gap Comparison')
plt.legend()
plt.grid(True)
plt.show()