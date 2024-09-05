import logging
from pathlib import Path
import json
import yaml

from Optimization_Benchmarks.VehicleRoutingProblem import VRPInstance, VRPSolutionCostMinimization, VRPSolutionRevenueMaximization
from Optimization_Benchmarks.SolutionScoreHistory import SolutionScoreHistory
from Optimizer import Optimize

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)
CONFIG_PATH = Path("config.yaml")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)


    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    experiment_dict = config["experiment_dict"]
    logger.info(f"Running {len(experiment_dict)} experiments.")

    for i, experiment in enumerate(experiment_dict):
        settings = json.dumps(config["optimization_settings"], indent=4)
        hyperparameters = json.dumps(config["optimization_hyperparameters"], indent=4)
        logger.info(f"Experiment {i+1}\nOptimization Settings:\n{settings}\nHyperparameters:\n{hyperparameters}")
        
        # Optimization settings
        VERBOSE = config["optimization_settings"]["verbose"]
        RUNS = config["optimization_settings"]["runs"]
        SAVE_RESULTS = config["optimization_settings"]["save_results"]
        SAVE_META_PROMPT = config["optimization_settings"]["save_meta_prompt"]
        OPTIMAL_COST = config["optimization_settings"]["optimal_cost"]
        PAST_SOLUTIONS = config["optimization_settings"]["past_solutions"]

        # Optimization hyperparameters
        MAX_LIST_LENGTH = config["optimization_hyperparameters"]["max_list_length"]
        SHUFFLING = config["optimization_hyperparameters"]["shuffling"]
        ITERATION_STEPS = config["optimization_hyperparameters"]["iteration_steps"]
        MODEL = config["optimization_hyperparameters"]["model"]
        TEMPERATURE = config["optimization_hyperparameters"]["temperature"]
        MAX_TOKENS = config["optimization_hyperparameters"]["max_tokens"]
        MAX_REFLECTIONS = experiment["i"]

        # Vehicle Routing Problem - Parameters
        SEED = config["vehicle_routing_problem_parameters"]["seed"]
        NUMBER_OF_NODES = experiment["nodes"]
        DELIVERY_REQUIREMENTS = experiment["delivery_demands"]
        VEHICLES = config["vehicle_routing_problem_parameters"]["vehicles"]
        SIZE = config["vehicle_routing_problem_parameters"]["size"]
        INCLUDE_ROUTING = config["vehicle_routing_problem_parameters"]["include_routing"]
        WEIGHT_FACTOR = config["vehicle_routing_problem_parameters"]["weight_factor"]
        REVENUE_PER_CONTAINER = config["vehicle_routing_problem_parameters"]["revenue_per_container"]
        OBJECTIVE = config["vehicle_routing_problem_parameters"]["objective"]

        json_path = RESULT_DIR / f"vrp_{OBJECTIVE}_{experiment['n']-1}_{PAST_SOLUTIONS}_{MAX_REFLECTIONS}_{RUNS}_{MODEL}.json"
        # Result dictionary saved to json if SAVE_RESULTS = True
        results = {"model": MODEL,
                "seed": SEED,
                "revenue_per_container": REVENUE_PER_CONTAINER,
                "routing": INCLUDE_ROUTING,
                "nodes": NUMBER_OF_NODES,
                "number_of_runs": RUNS,
                "temperature": TEMPERATURE, 
                "max_tokens": MAX_TOKENS,
                "shuffling": SHUFFLING, 
                "max_list_length": MAX_LIST_LENGTH,
                "iteration_steps": ITERATION_STEPS,
                "optimal_score": OPTIMAL_COST,
                "vehicles": VEHICLES,
                "delivery_requirements": DELIVERY_REQUIREMENTS,
                "max_reflections": MAX_REFLECTIONS,
                "meta_prompt": None,
                "runs":[]
        }

        # Meta prompt dictionary saved to json if SAVE_META_PROMPT = True
        meta_prompts = {
            "seed": SEED,
            "nodes": NUMBER_OF_NODES,
            "max_list_length": MAX_LIST_LENGTH,
            "optimal_cost": OPTIMAL_COST,
            "optimal_route": [],
            "meta_prompt": []
        }

        # Initializes a Transport mission planning problem
        vrp = VehicleRoutingProblem(NUMBER_OF_NODES=NUMBER_OF_NODES, 
                                    SIZE=SIZE, VEHICLES=VEHICLES, 
                                    DELIVERY_REQUIREMENTS=DELIVERY_REQUIREMENTS, 
                                    weight_factor=WEIGHT_FACTOR,
                                    revenue_per_container=REVENUE_PER_CONTAINER,
                                    SEED=SEED)

        for run in range(RUNS):
            print(f"---------------- RUN {run+1} ------------------")
            solution_score_history = Solution_Score_History(problem_instance=vrp, max_solutions=MAX_LIST_LENGTH, shuffling=SHUFFLING)
            #initial_lengths = list(solution_score_history.length_history)

            # (1) Initialize the optimizer
            optimize = Optimize_VRP_Maximize_Revenue(vrp, 
                                    solution_score_history,
                                    max_tokens=MAX_TOKENS,
                                    temperature=TEMPERATURE,
                                    max_iterations=ITERATION_STEPS, 
                                    max_reflections=MAX_REFLECTIONS, 
                                    optimal_score = OPTIMAL_COST, 
                                    include_routing=INCLUDE_ROUTING,
                                    VERBOSE=VERBOSE)
            
            # (2) Run the optimizer
            optimize(run)

            if VERBOSE:
                print(optimize.meta_prompt)
            result = optimize.results
            results["runs"].append(result)
            results["meta_prompt"] = optimize.meta_prompt

            if SAVE_RESULTS:
                with open(json_path, "w") as f:
                    json.dump(results, f)

            if SAVE_META_PROMPT:
                with open(RESULT_DIR / f"meta_prompts", "w") as f:
                    json.dump(meta_prompts, f)
        if VERBOSE:
            print(results)
