TSP_META_PROMPT_COORDINATES = """You are given a list of points with coordinates below:
{coordinates}
"""

TSP_META_PROMPT_HISTORY = """
Below are some previous traces and their lengths. Lower length is better.
{solution_score_history}

Give me a new trace that is different from all traces above and has a length lower than any of the above. The trace should traverse all points exactly once. Do not write code. The trace should start with <trace> and end with </trace>."""
