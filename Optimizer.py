from groq import Groq
from groq.types.chat import ChatCompletion
import os
import time
from dotenv import load_dotenv
from Optimization_Benchmarks.SolutionScoreHistory import SolutionScoreHistory
from Optimization_Benchmarks.Base.ProblemSolution import ProblemSolution
from typing import Union, Any

load_dotenv()

class Optimize:
    def __init__(self, problem_instance: Any, solution_class: ProblemSolution, solution_score_history: SolutionScoreHistory, meta_prompt_template: str, config: dict,
                 reflection_prompt: str=None, formatation_prompt: str=None):
        # Classes and instances
        self.problem_instance = problem_instance
        self.solution_class = solution_class
        self.solution_score_history = solution_score_history

        # Prompts and Prompt templates. Reflection and formattation are optional and not used by default.
        self.meta_prompt_template = meta_prompt_template
        self.reflection_prompt_template = reflection_prompt
        self.formatation_prompt = formatation_prompt

        # Iteration Hyperparameters.
        self.max_iterations = config.get("max_iterations")
        self.max_reflections = config.get("max_reflections")
        self.optimal_score = config.get("optimal_score")
        self.verbose = config.get("verbose")
        self.delay = config.get("delay") or 0
        self.init_sol_per_step = config.get("init_sol_per_step")
        # LLM settings.
        self.model = config.get("model")
        self.max_tokens = config.get("max_tokens") 
        self.temperature = config.get("temperature")

        # Initialize Groq Client.
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Generate Meta Prompt
        self.meta_prompt = self.meta_prompt_template.format(solution_score_history=solution_score_history.get_solution_history())
        if self.verbose:
            print(self.meta_prompt)

        # Collecting results
        self.results = []

    def __call__(self, run):
        for i in range(self.max_iterations):
            print(f"Run {run+1}, Iteration {i+1}")
            
            sol_per_step = self.init_sol_per_step
            if i >= 5:
                sol_per_step = 1
            
            for j in range(sol_per_step):
                score = None
                self.feasibility_violation = None
                self.messages = []

                response = self.generate(self.meta_prompt)

                # If formatation prompt is given, make a second call to the LLM to specify the output format.
                # Should only be used for models where output format cannot be specified.
                if self.formatation_prompt:
                    response = self.generate(self.formatation_prompt)
                
                # Evaluation is performed inside the solution class.
                solution_instance = self.solution_class(problem_instance=self.problem_instance)
                score = solution_instance.evaluate(response)
                solution_instance.score = score
                if self.verbose:
                    print(f"Score: {score}")
                
                # If Score is None -> Reflect.
                if not score:
                    solution_instance.score = self.reflect(solution_instance)
                
                if score:
                    self.solution_score_history.append(solution_instance)
                    self.meta_prompt = self.meta_prompt_template.format(solution_score_history=self.solution_score_history.get_solution_history())

                    # Early stopping if optimal solution was found.
                    if score == self.optimal_score:
                        result = {"iteration": i, "solution": str(solution_instance), "score": score, "feasibility_violation": None, "optimal": True}
                        self.results.append(result)
                        return
                
                    result = {"iteration": i, "solution": str(solution_instance), "score": score, "optimal": False}
                    self.results.append(result)

                time.sleep(self.delay)

    def request(self) -> ChatCompletion:
        return self.client.chat.completions.create(
                messages=self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def generate(self, prompt: str) -> str:
        """
        Returns a response from the LLM.
        """
        # Add the user message to the chat history.
        self.messages.append(
            {
            "role": "user",
            "content": prompt,
            }
        )

        # Get a response from the LLM.
        for attempt in range(6):
            try:
                chat_completion = self.request()
                break
            except:
                if attempt == 4:
                    print("Sleeping 10 minutes before final attempt ...")
                    time.sleep(600)
                    return ""
                else:
                    print("Sleeping 1 minute before Re-attempt ...")
                    time.sleep(60)


        response = chat_completion.choices[0].message.content

        # Adds the response to the history for reflection etc. History is reset every iteration.
        self.messages.append(
            {
            "role": "assistant",
            "content": response,
            }
        )

        if self.verbose:
            print(f"Response: {response}")

        return response
            
    def reflect(self, solution_instance: ProblemSolution) -> Union[float, None]:
        """
        If the solution could not be evaluated, retry for max_reflection iterations or until a score can be calculated.
        """
        # Return if no reflection
        if self.max_reflections == 0:
            return solution_instance.score
        
        # For i rounds, reflect and try a new solution given the history.
        for _ in range(self.max_reflections):
            if solution_instance.score is None:
                reflection_prompt = self.reflection_prompt_template.format()
                response = self.generate(reflection_prompt)
                new_score = solution_instance.evaluate(response)
                solution_instance.score = new_score

        if self.verbose:
            print(f"Reflect Score: {solution_instance.score}")
        return solution_instance.score
