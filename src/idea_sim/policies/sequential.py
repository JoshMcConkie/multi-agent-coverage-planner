# Given full preset paths with known utility
from typing import List, Tuple
from idea_sim.policies.tools import best_greedy_choice, path_model, update_agents
from idea_sim.objective import Objective
from idea_sim.env import GridWorld, Model

def seq_solve(model: Model) -> Tuple[int,List[int]]:
    if model.agent_order is None:
        raise("ERROR: Sequential solver cannot have model.agent_order == None.")
    # print(model.util_mat)
    # print(model.agent_path_dict)
    # print(model.agent_order)
    path_ids: List[int] = []
    for agent in model.agent_order:
        score, path_ids = best_greedy_choice(model.agent_path_dict[agent],
                                                path_ids,
                                                model.util_mat)
    # print(f"Score: {score}")
    update_agents(model,path_ids)
    return score


def split_seq_solve(grid: GridWorld,objective: type[Objective], agent_order, steps: int, chunksize: int):
    score = 0
    chunksize = 2

    rounds = [chunksize for s in range(steps // chunksize)] + [steps % chunksize]
    for _ in rounds:
        round_model = path_model(grid, objective, steps, agent_order)
        round_score = seq_solve(round_model)
        score += round_score

    # print(f"Score: {score}")
    return score
