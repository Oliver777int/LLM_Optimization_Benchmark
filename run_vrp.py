import logging
from pathlib import Path
import json
import yaml

from Optimization_Benchmarks.VehicleRoutingProblem import VRPInstance, VRPSolutionCostMinimization, VRPSolutionRevenueMaximization
from Optimization_Benchmarks.SolutionScoreHistory import SolutionScoreHistory
from Plotting.optimal_score_vrp import OptimalVRPScore
from Plotting.generate_plot import generate_vrp_plot
from Optimizer import Optimize
from Prompts.vehicle_routing_prompt_template import get_vrp_meta_prompt_template

RESULT_DIR = Path("results_vrp")
RESULT_DIR.mkdir(exist_ok=True)
CONFIG_PATH = Path("config.yaml")

def Configure_Logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)
    return logger

if __name__ == "__main__":
    logger = Configure_Logger()

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    sweep_configurations = config.get("vehicle_routing_configuration_sweep")
    logger.info(f"Running {len(sweep_configurations)} experiments.")

    for i, current_config in enumerate(sweep_configurations or range(1)):
        
        settings = json.dumps(config["optimization_settings"], indent=4)
        hyperparameters = json.dumps(config["optimization_hyperparameters"], indent=4)
        logger.info(f"Experiment {i+1}\nOptimization Settings:\n{settings}\nHyperparameters:\n{hyperparameters}")
        
        # Optimization settings
        VERBOSE = config["optimization_settings"]["verbose"]
        RUNS = config["optimization_settings"]["runs"]
        SAVE_RESULTS = config["optimization_settings"]["save_results"]
        SAVE_META_PROMPT = config["optimization_settings"]["save_meta_prompt"]
        DELAY = config["optimization_settings"]["delay"]

        # Optimization hyperparameters
        MAX_LIST_LENGTH = config["optimization_hyperparameters"]["max_list_length"]
        SHUFFLING = config["optimization_hyperparameters"]["shuffling"]
        ITERATION_STEPS = config["optimization_hyperparameters"]["iteration_steps"]
        MODEL = config["optimization_hyperparameters"]["model"]
        TEMPERATURE = config["optimization_hyperparameters"]["temperature"]
        MAX_TOKENS = config["optimization_hyperparameters"]["max_tokens"]
        INIT_SOL_PER_STEP = config["optimization_hyperparameters"]["init_sol_per_step"]
        if current_config:
            MAX_REFLECTIONS = current_config.get("i") or config["optimization_settings"]["number_of_reflections"]

        # Vehicle Routing Problem - Variable parameters
        if current_config:
            SEEDS = current_config.get("seeds") or config["vehicle_routing_problem_parameters"]["seeds"]
            NUMBER_OF_NODES = current_config.get("number_of_nodes") or config["vehicle_routing_problem_parameters"]["number_of_nodes"]
            DELIVERY_REQUIREMENTS = current_config.get("delivery_demands") or config["vehicle_routing_problem_parameters"]["delivery_demands"]
        
        # Vehicle Routing Problem - Constant parameters 
        SIZE = config["vehicle_routing_problem_parameters"]["size"]
        VEHICLES = config["vehicle_routing_problem_parameters"]["vehicles"]
        WEIGHT_FACTOR = config["vehicle_routing_problem_parameters"]["weight_factor"]
        REVENUE_PER_CONTAINER = config["vehicle_routing_problem_parameters"]["revenue_per_container"]
        OBJECTIVE = config["vehicle_routing_problem_parameters"]["objective"]

        json_path = RESULT_DIR / f"vrp-{OBJECTIVE}_{str(NUMBER_OF_NODES).zfill(3)}node_{RUNS}runs_{MODEL}.json"
        # Result dictionary saved to json if SAVE_RESULTS = True
        results = {"model": MODEL,
                "seeds": SEEDS,
                "revenue_per_container": REVENUE_PER_CONTAINER,
                "objective": OBJECTIVE,
                "nodes": NUMBER_OF_NODES,
                "number_of_runs": RUNS,
                "temperature": TEMPERATURE, 
                "max_tokens": MAX_TOKENS,
                "shuffling": SHUFFLING, 
                "max_list_length": MAX_LIST_LENGTH,
                "max_iterations": ITERATION_STEPS,
                "vehicles": VEHICLES,
                "delivery_requirements": DELIVERY_REQUIREMENTS,
                "init_sol_per_step": INIT_SOL_PER_STEP,
                "delay": DELAY,
                "verbose": VERBOSE,
                "max_reflections": MAX_REFLECTIONS,
                "meta_prompt": None,
                "runs":[]
        }

        # Meta prompt dictionary saved to json if SAVE_META_PROMPT = True
        meta_prompts = {
            "seed": SEEDS,
            "nodes": NUMBER_OF_NODES,
            "meta_prompt": []
        }
        
        logger.info(f"Looping through the following seeds: {SEEDS[0:RUNS]}")
        for run in range(RUNS):
            SEED = SEEDS[run]
            logger.info(f"Seed {SEED}")
            logger.info(f"Run {run+1}")
            
             # Initializes a Transport mission planning problem
            vrp = VRPInstance(
                NUMBER_OF_NODES=NUMBER_OF_NODES, 
                SIZE=SIZE, VEHICLES=VEHICLES, 
                DELIVERY_REQUIREMENTS=DELIVERY_REQUIREMENTS, 
                weight_factor=WEIGHT_FACTOR,
                revenue_per_container=REVENUE_PER_CONTAINER,
                SEED=SEED
            )
            
            results["optimal_score"] = OptimalVRPScore(OBJECTIVE, vrp)
            print(f"Optimal score for this run is: {results['optimal_score']}")

            # Prompt templates
            VRP_META_PROMPT_TEMPLATE = get_vrp_meta_prompt_template(vrp)
            print(VRP_META_PROMPT_TEMPLATE)
            solution_score_history = SolutionScoreHistory(maximize_score=False, max_solutions=MAX_LIST_LENGTH, shuffling=SHUFFLING, SEED=SEED)

            # (1) Initialize the optimizer.
            solution_classes = {"min": VRPSolutionCostMinimization, "max": VRPSolutionRevenueMaximization}
            optimize = Optimize(vrp, solution_classes[OBJECTIVE], solution_score_history, VRP_META_PROMPT_TEMPLATE, config=results)

            # (2) Run the optimizer.
            optimize(run)

            # (3) Store the results.
            if VERBOSE:
                print(optimize.meta_prompt)
            result = optimize.results
            results["runs"].append(result)
            results["meta_prompt"] = optimize.meta_prompt
            meta_prompts["meta_prompt"].append(optimize.meta_prompt)

            if SAVE_RESULTS:
                with open(json_path, "w") as f:
                    json.dump(results, f)

            if SAVE_META_PROMPT:
                with open(RESULT_DIR / f"meta_prompts.json", "w") as f:
                    json.dump(meta_prompts, f)
        if VERBOSE:
            print(results)

    generate_vrp_plot(RESULT_DIR)