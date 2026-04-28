# Given full preset paths with known utility
from typing import List, Tuple

import numpy as np
from idea_sim.policies.tools import best_greedy_choice, path_model, update_agents
from idea_sim.objective import Objective
from idea_sim.env import GridWorld, Model

def seq_solve(model: Model, util_mat_postfix: np.ndarray = None) -> int:
    '''Sequential submodular maximization solver. Alters in place the attributes of `model`,
    including the attached agents' paths. Returns maximized utility score.

    args:
        model: a Model instance containing the utility matrix (Nemhauser et al, 1978),
        available agent allocations, etc. It is transpose from Nemhauser et al.'s design.

        util_mat_postfix: utility rows from past rounds decisions to be appended to current round

    returns:
        score: an integer, sequentially maximized objective/utility score.
        
    # 
    
    '''
    if model.agent_order is None:
        raise("ERROR: Sequential solver cannot have model.agent_order == None.")
    # print(model.util_mat)
    # print(model.agent_path_dict)
    # print(model.agent_order)
    path_ids = []

    # handling for split_seq_solve() calls, where a utility rows of
    # decisions from previous rounds are appended
    prefix_len = 0
    if util_mat_postfix is not None:
        start = model.util_mat.shape[0]
        prefix_len = util_mat_postfix.shape[0]
        path_ids = [i + start for i in range(prefix_len)]
        model.util_mat = np.append(model.util_mat, util_mat_postfix, axis=0)

    for agent in model.agent_order:
        score, path_ids = best_greedy_choice(model.agent_path_dict[agent],
                                                path_ids,
                                                model.util_mat)
    # print(f"Score: {score}")
    model.chosen_path_ids = path_ids
    update_agents(model,path_ids[prefix_len:])
    return score


def split_seq_solve(grid: GridWorld,objective: type[Objective], 
                    steps: int, agent_order, chunksize: int) -> int:
    rounds = [chunksize for s in range(steps // chunksize)] + [steps % chunksize]
    util_mat_hist = None
    path_ids_hist = []
    for chunk in rounds:
        round_model = path_model(grid, objective, chunk, agent_order)
        seq_solve(round_model,util_mat_hist)
        util_mat_hist = round_model.util_mat[round_model.chosen_path_ids]

    # print(f"Score: {score}")
    return grid.get_score()
