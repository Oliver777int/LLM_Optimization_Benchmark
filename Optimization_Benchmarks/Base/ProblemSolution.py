from pydantic import BaseModel
from typing import Optional, Any

class ProblemSolution(BaseModel):
    problem_instance: Optional[Any] = None
    score: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
        
    def __str__(self) -> str:
        """
        String representation of solution to show to LLM in the meta prompt.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def evaluate(self, response: str) -> str:
        """
        A method that evaluates a solution given a response from the LLM. The response is the raw message from the LLM, thus parsing must be performed.

        Args:
          response (str): The response from the LLM.
          
        Return:
          Score (int): The score of the evaluated solution.
        """
        raise NotImplementedError("Subclasses should implement this method.")
