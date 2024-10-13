import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union

def create_table(result_path: str = None, model_name: Union[str, List[str]] = None, result_data: Union[List[float], List[List[float]]] = []):
    # Data
    models = ["LLAMA 3.2 90B", "LLAMA 3.1 70B", "LLAMA 3.1 8B", "Mixtral 8x7b"]
    n_values = [7, 10, 15]  # Different values of n
    optimality_gaps = np.array([[0.35, 12.365, 38.437], [0.35, 12.238, 43.677], [1.431, 12.541, 44.545], [0.709, 17.235, 54.58]])  # Generate some random optimality gap values for demo

    # Add new data if provided
    if model_name:
        if isinstance(model_name, list):
            for model in model_name:
                models.append(model)
        else:
            models.append(model_name)
    if result_data:
        if isinstance(result_data[0], list):
            for result in result_data:
                if len(result) == 3:
                    new_data = np.array(result)
                    optimality_gaps = np.vstack([optimality_gaps, new_data])
        else:
            if len(result_data) == 3:
                new_data = np.array(result_data)
                optimality_gaps = np.vstack([optimality_gaps, new_data])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(14, 4))
    plt.subplots_adjust(left=0.2)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    plt.title("LLM Optimality Gap for different values of n", fontsize=14)

    # Create the table
    table_data = [["{:.2f}%".format(optimality_gaps[i, j]) for j in range(len(n_values))] for i in range(len(models))]
    table = plt.table(cellText=table_data,
                    rowLabels=models,
                    colLabels=[f"n = {n}" for n in n_values],
                    cellLoc='center',
                    loc='center')

    # Style adjustments
    table.scale(1, 2)
    table.set_fontsize(12)

    # Light grey cells and turning best value bold.
    for i in range(len(models)):
        cell = table[(i + 1, -1)]
        cell.set_facecolor("#f0f0f0")

    for j in range(len(n_values)):
        cell = table[(0, j)]
        cell.set_facecolor("#f0f0f0")

    for j in range(len(n_values)):
        column_values = optimality_gaps[:, j]
        min_row_idx = np.argmin(column_values)
        cell = table[(min_row_idx + 1, j)]
        cell.set_text_props(weight='bold')
    
    plt.savefig(f"{result_path}/tsp_table.png")

if __name__ == "__main__":
    results_path = '../results'
    create_table(results_path, ["New model"], [100, 200, 300]) # Example input
    plt.show()