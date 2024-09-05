# LLM_Optimization_Benchmark
A Codebase for Benchmarking the optimization proficiency of Large Language Models, including traveling salesman and vehicle routing optimization. 

# Implementation details
This implementation uses Groq cloud for generating responses by LLMs. This requires an account on Groq Cloud along with an api key stored in a .env file as GROQ_AP_KEY = <YOUR_API_KEY>

# Usage
1. Clone the repository.
2. Create an environment for installing dependencies (for example using venv -> python -m venv myenv).
3. Install dependencies using: pip install -r requirements.txt
4. Set Configurations in the config.yaml file.
5. Run either traveling salesman optimization or vehicle routing optimization using run_tsp.py or run_vrp.py
6. Results are saved to a results directory which can be plotted using for example plot_tsp.py or plot_vrp.py located in the Plotting directory.

# Note.
Additional optimization benchmarks can be created using the abstract class called ProblemSolution which requires
1. a __str__ method representing the solution as a string.
2. an evaluate method that computes the score of a response by the LLM. (May have to include parsing of the LLM output).
3. A prompt outlining the Optimization problem.

Please note that this project is still being updated and that all functionalities are not available for VRP Optimization yet.
