# Given full preset paths with known utility
from typing import List, Tuple, Dict
import numpy as np
from idea_sim.policies.tools import best_greedy_choice

def seq_solve(ord_agents: List,
              agent_choice_dict: Dict[str,List[int]],
              util_mat: np.ndarray) -> Tuple[int,List[int]]:
    choice_hist: List[int] = []
    for agent in ord_agents:
        score, choice_hist = best_greedy_choice(agent_choice_dict[agent],
                                                choice_hist,
                                                util_mat)
    return score, choice_hist