from typing import List, Tuple, Dict
import numpy as np

'''

'''
def utility_score(choice_set,util_mat) -> Tuple[int,List[int,int]]:
    score = 0
    for col in range(util_mat.shape[1]):
        # print(f"util crop: {util_mat[choice_set]}")
        score += max([util_mat[choice][col] for choice in choice_set])
    return score, choice_set


def best_greedy_choice(options: List[int],choice_hist: List[int],util_mat)-> Tuple[int,List[int]]:
    '''Get next choice with greatest marginal utility.

    Args: 
        options (list): allocation options available to agent
        choice_hist (list): past selections made made (S_e in context of Nemhauser et al., 1978)
        util_mat (np.ndarray): The C (aka omega) matrix of allocations and utilities (Nemhauser et al., 1978)
    '''
    return max((utility_score(choice_hist+[o],util_mat) for o in options))