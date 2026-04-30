from typing import List, Tuple
from itertools import permutations
from tqdm import tqdm
from idea_sim.env import Model, Result
from idea_sim.policies.tools import update_agents,best_greedy_choice

def step_cen_solve(model, agent_order):
    path_ids: List[int] = []
    for agent in agent_order:
        score, path_ids = best_greedy_choice(model.agent_path_dict[agent],
                                                path_ids,
                                                model.util_mat)
    return score, path_ids, agent_order


def best_seq_greedy_solve(model: Model) -> Tuple[int,List[int]]:
    agent_perm = permutations(model.agent_path_dict.keys())
    solutions = []
    print("Centralized Solver Progress: ")
    for agent_order in tqdm(agent_perm):
        solutions.append(step_cen_solve(model, agent_order))

    score,path_ids,agent_order = max(solutions)
    paths_by_agent = update_agents(model,path_ids)
    return Result(model.grid.grid,paths_by_agent,
                  score,best_seq_greedy_solve,
                  runtime=None,chosen_path_ids=path_ids,
                  metadata=None,agent_order=agent_order)
        