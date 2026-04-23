from typing import List, Tuple, Dict
import numpy as np
from itertools import permutations
from idea_sim.policies.sequential import seq_solve

def cen_solve(agent_choice_dict: Dict[str,List[int]],
            util_mat: np.ndarray) -> Tuple[int,List[int]]:
    agent_perm = permutations(agent_choice_dict.keys())
    return max(seq_solve(agent_order, agent_choice_dict, util_mat) for agent_order in agent_perm)
        