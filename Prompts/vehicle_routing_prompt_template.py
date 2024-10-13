VRP_META_PROMPT_FORMULATION ="""I need to make the following deliveries to customers located at different nodes.

{customer_demands}

Here are the location of the nodes.

{coordinates}

To make these deliveries i have {number_of_vehicles} vehicles at my desposal. These vehicles have the following name and maximum load capacity:

{vehicles}

"""

VRP_META_PROMPT_HISTORY = """Below are some previously evaluated solutions.

{solution_score_history}

Please find a better solution which minimized to the total cost of travel.

Format your output as:
```json
{{
  "vehicles": [
    {{
      "vehicle_name": "X", 
      "deliveries": {{
        "nodeA": N, 
        "nodeB": M 
      }}
    }},
    {{
      "vehicle_name": "Y", 
      "deliveries": {{
        "nodeC": P, 
        "nodeD": Q 
      }}
    }}
  ]
}}
```

Where X, Y are single character names of vehicles, A, B, C, D are integers of delivery nodes and N, M, P, Q are integer delivery amounts. Do not write code.
"""


def get_vrp_meta_prompt_template(vrp):
    customer_demands_string = '\n'.join(f"{demand} pallets to node{node+1}" for node, demand in enumerate(vrp.DELIVERY_REQUIREMENTS[1:]))
    vehicles_string = '\n'.join(f'vehicle {key}: max load {value} pallets' for key, value in vrp.VEHICLES.items())
    VRP_META_PROMPT_TEMPLATE = VRP_META_PROMPT_FORMULATION.format(
        customer_demands=customer_demands_string, 
        coordinates=vrp.coordinates_string, 
        number_of_vehicles=len(vrp.VEHICLES), 
        vehicles=vehicles_string)
    VRP_META_PROMPT_TEMPLATE += VRP_META_PROMPT_HISTORY
    return VRP_META_PROMPT_TEMPLATE