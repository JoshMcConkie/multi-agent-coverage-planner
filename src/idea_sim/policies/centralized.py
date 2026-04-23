from typing import List, Tuple, Dict
import numpy as np
from itertools import permutations
from idea_sim.policies.sequential import seq_solve

def cen_solve(util_mat: np.ndarray,
              agent_choice_dict: Dict[str,List[int]],
            *_) -> Tuple[int,List[int]]:
    agent_perm = permutations(agent_choice_dict.keys())
    return max(seq_solve(util_mat, agent_choice_dict,agent_order) for agent_order in agent_perm)
        