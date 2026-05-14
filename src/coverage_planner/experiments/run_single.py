from coverage_planner.env import GridWorld, Result, EmptyResult

from coverage_planner.objective import Coverage
from coverage_planner.policies.tools import path_model
from coverage_planner.policies.centralized import best_seq_greedy_solve
from coverage_planner.policies.sequential import seq_greedy_solve, split_seq_solve

def run_single_compare(agents: list, agent_order,size: int, 
                       steps: int, chunksize: int, run_optimal: bool = True)->list[Result | EmptyResult]:
    grid = GridWorld(size)
    grid.init_agent(agents)
    model = path_model(grid,Coverage,steps,agent_order)
    seq_result = seq_greedy_solve(model)
    split_result = split_seq_solve(model, chunksize)
    if run_optimal:
        best_seq_result = best_seq_greedy_solve(model)
    else:
        best_seq_result = EmptyResult()
    return [seq_result,split_result,best_seq_result]
'''
STEPS = 6
NUM_AGENTS = 4
AGENT_ORDER = [1,2,3,4,5]
SPLIT_CHUNKSIZE = 3
WORLD_SIZE = 7

def init_world(size: int):
    grid = GridWorld(size)

    agent_1 = Agent(0,0)
    agent_2 =  Agent(grid.size-1,0)
    agent_3 =  Agent(0,grid.size-1)
    agent_4 = Agent(grid.size-1,grid.size-1)
    agent_5 = Agent(WORLD_SIZE//2,WORLD_SIZE//2)
    agents = [agent_1, agent_2, agent_3,agent_4, agent_5]
    grid.init_agent(agents[:NUM_AGENTS])
    print(grid)
    return grid

grid = init_world(WORLD_SIZE)
model = path_model(grid,Coverage,steps=STEPS,agent_order=AGENT_ORDER[:NUM_AGENTS])

seq_result = seq_greedy_solve(model)
split_result = split_seq_solve(model, chunksize=SPLIT_CHUNKSIZE)
best_seq_result = best_seq_greedy_solve(model)

compare = CompareResults([seq_result,split_result,best_seq_result])
print(compare.to_dataframe())
'''