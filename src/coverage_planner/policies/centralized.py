import time

from itertools import permutations
from coverage_planner.env import Model, Result
from coverage_planner.policies.tools import update_agents,best_greedy_choice

def step_cen_solve(model, agent_order):
    path_ids: list[int] = []
    for agent in agent_order:
        score, path_ids = best_greedy_choice(model.agent_path_dict[agent],
                                                path_ids,
                                                model.util_mat)
    return score, path_ids, agent_order


def best_seq_greedy_solve(model: Model) -> Result:
    start_time = time.perf_counter()
    agent_perm = permutations(model.agent_path_dict.keys())
    solutions = []
    # print("\"Best\" Sequential Solver...")
    for agent_order in agent_perm:
        solutions.append(step_cen_solve(model, agent_order))

    score,path_ids,agent_order = max(solutions)
    paths_by_agent = update_agents(model,path_ids)
    end_time = time.perf_counter()
    result = Result(model.grid.grid,paths_by_agent,
                  score,best_seq_greedy_solve,
                  runtime=end_time-start_time,chosen_path_ids=path_ids,
                  metadata=None,agent_order=agent_order, steps=model.steps)
    model.grid.reset_grid()
    return result