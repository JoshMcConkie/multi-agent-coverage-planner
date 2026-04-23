# Given full preset paths with known utility
from typing import List, Tuple, Dict
import numpy as np

def utility_score(choice_set,util_mat) -> Tuple[int,List[int,int]]:
    score = 0
    for row in util_mat:
        score += max(row[choice_set])
    return score, choice_set

def best_greedy_choice(options,choice_hist,util_mat)-> Tuple[int,List[int]]:
    return max((utility_score(choice_hist+[o],util_mat) for o in options))

def seq_solve(ord_agents: List,
              agent_choice_dict: Dict[str,List[int]],
              util_mat: np.ndarray) -> Tuple[int,List[int]]:
    choice_hist: List[int] = []
    for agent in ord_agents:
        score, choice_hist = best_greedy_choice(agent_choice_dict[agent],
                                                choice_hist,
                                                util_mat)
    return score, choice_hist