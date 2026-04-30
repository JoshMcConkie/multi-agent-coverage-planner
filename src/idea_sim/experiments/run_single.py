from idea_sim.env import GridWorld, Agent, CompareResults
from idea_sim.objective import Coverage
from idea_sim.policies.tools import path_model
from idea_sim.policies.centralized import best_seq_greedy_solve
from idea_sim.policies.sequential import seq_greedy_solve, split_seq_solve

STEPS = 6
NUM_AGENTS = 5
AGENT_ORDER = [1,2,3,4,5]
SPLIT_CHUNKSIZE = 4
WORLD_SIZE = 9

def init_world(size: int):
    grid = GridWorld(size)

    agent_1 = Agent(0,0)
    agent_2 =  Agent(grid.size-1,0)
    agent_3 =  Agent(0,grid.size-1)
    agent_4 = Agent(0,0) # Agent(grid.size-1,grid.size-1)
    agent_5 = Agent(0,0)
    agents = [agent_1, agent_2, agent_3,agent_4, agent_5]
    grid.init_agent(agents[:NUM_AGENTS])
    print(grid)
    return grid

grid = init_world(WORLD_SIZE)
model = path_model(grid,Coverage,steps=STEPS,agent_order=AGENT_ORDER[:NUM_AGENTS])

seq_result = seq_greedy_solve(model)
# print(f"Sequential:\n{seq_result.final_grid_array}")

split_result = split_seq_solve(model, chunksize=SPLIT_CHUNKSIZE)
# print(f"Split-Sequential:\n{split_result.final_grid_array}")

best_seq_result = best_seq_greedy_solve(model)

compare = CompareResults([seq_result,split_result,best_seq_result])
print(compare.to_dataframe())
# print(f"Best Sequential:\n{best_seq_result.final_grid_array}")
# print(seq_result)
# print(split_result)
# print(best_seq_result)


# print(f"Sequential Score: {seq_result.score}")
# print(f"Split Score: {split_result.score}")
# print(f"Optimal Score: {best_seq_result.score}")
# print(f"Seq/Opt Ratio: {seq_result.score/best_seq_result.score*100}%")