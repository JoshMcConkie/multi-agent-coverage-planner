# Given full preset paths with known utility
from typing import List, Tuple, Dict
import numpy as np
from idea_sim.policies.tools import best_greedy_choice

def seq_solve(util_mat: np.ndarray,
            agent_choice_dict: Dict[str,List[int]],
            ord_agents: List) -> Tuple[int,List[int]]:
    
    # print(util_mat)
    # print(agent_choice_dict)
    # print(ord_agents)
    choice_hist: List[int] = []
    for agent in ord_agents:
        score, choice_hist = best_greedy_choice(agent_choice_dict[agent],
                                                choice_hist,
                                                util_mat)
    # print(f"Score: {score}")
    return score, choice_hist