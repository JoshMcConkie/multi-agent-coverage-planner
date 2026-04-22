# Given full preset paths with known utility
from typing import List, Tuple

def utility_score(choice_set,util_mat) -> Tuple[int,Tuple[int,int]]:
    score = 0
    for row in util_mat:
        score += max(row[choice_set])
    return score, choice_set

def best_greedy_choice(options,choice_hist,util_mat):

    return max((utility_score(choice_hist+o,util_mat) for o in options))

def seq_solve(ord_agents,util_mat):
    choice_hist: List[int] = []
    for agent in ord_agents:
        choice_hist.append()

    # for agent in ord_agents:
    #     score, choice_hist
