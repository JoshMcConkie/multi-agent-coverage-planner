from typing import List, Tuple
from itertools import permutations
from tqdm import tqdm
from idea_sim.env import Model
from idea_sim.policies.tools import update_agents,best_greedy_choice

def step_cen_solve(model):
    path_ids: List[int] = []
    for agent in model.agent_order:
        score, path_ids = best_greedy_choice(model.agent_path_dict[agent],
                                                path_ids,
                                                model.util_mat)
    return score, path_ids, model.agent_order


def cen_solve(model: Model) -> Tuple[int,List[int]]:
    agent_perm = permutations(model.agent_path_dict.keys())
    solutions = []
    og_agent_order = model.agent_order
    print("Centralized Solver Progress: ")
    for agent_order in tqdm(agent_perm):
        model.agent_order = agent_order
        solutions.append(step_cen_solve(model))

    score,path_ids,model.agent_order = max(solutions)
    update_agents(model,path_ids)
    return score, model.agent_order
        