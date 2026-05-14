# Given full preset paths with known utility
import time

import numpy as np
from coverage_planner.policies.tools import best_greedy_choice, path_model, update_agents
from coverage_planner.env import Model, Result

def choose_seq_paths(model, prior_util_row=None):
    if model.agent_order is None:
        raise("ERROR: Sequential solver cannot have model.agent_order == None.")
    path_ids = []
    for agent in model.agent_order:
        score, path_ids = best_greedy_choice(model.agent_path_dict[agent],
                                                path_ids,
                                                model.util_mat, prior_util_row)
    paths_by_agent = update_agents(model,path_ids)
    return score, path_ids, paths_by_agent

def seq_greedy_solve(model: Model) -> Result:
    '''Sequential submodular maximization solver. Alters in place the attributes of `model`,
    including the attached agents' paths. Returns maximized utility score.

    args:
        model: a Model instance containing the utility matrix (Nemhauser et al, 1978),
        available agent allocations, etc. It is transpose from Nemhauser et al.'s design.

        util_mat_postfix: utility rows from past rounds decisions to be appended to current round

    returns:
        score: an integer, sequentially maximized objective/utility score.
    '''
    start_time = time.perf_counter()
    # print(model.util_mat)
    # print(model.agent_path_dict)
    # print(model.agent_order)

    score, path_ids, paths_by_agent = choose_seq_paths(model)
    end_time = time.perf_counter()
    result = Result(model.grid.grid,paths_by_agent,
                  score,seq_greedy_solve,
                  runtime=end_time-start_time,chosen_path_ids=path_ids,
                  metadata=None,agent_order=model.agent_order,steps=model.steps)
    model.grid.reset_grid()
    return result


def split_seq_solve(model: Model, chunksize: int) -> Result:
    start_time = time.perf_counter()
    full_rounds, remainder = divmod(model.steps,chunksize)
    rounds = [chunksize] * full_rounds
    if remainder:
        rounds.append(remainder)
    prior_coverage_state = None
    for chunk in rounds:
        round_model = path_model(model.grid, model.objective,chunk, model.agent_order)
        score, path_ids, paths_by_agent = choose_seq_paths(round_model,prior_coverage_state)
        if prior_coverage_state is None:
            prior_coverage_state = np.zeros(round_model.util_mat.shape[1])
        prior_coverage_state += round_model.util_mat[path_ids].max(axis=0)
    end_time = time.perf_counter()
    # print(f"Score: {score}")
    result = Result(model.grid.grid,paths_by_agent,
                  model.grid.get_score(),split_seq_solve,
                  runtime=end_time-start_time,chosen_path_ids=path_ids,
                  metadata={"Round-steps":rounds}, 
                  agent_order=model.agent_order,steps=model.steps,)
    model.grid.reset_grid()
    return result
