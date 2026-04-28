from idea_sim.env import GridWorld, Agent
from idea_sim.objective import Coverage
from idea_sim.policies.tools import path_model
from idea_sim.policies.centralized import cen_solve
from idea_sim.policies.sequential import seq_solve, split_seq_solve

STEPS = 10
NUM_AGENTS = 3
AGENT_ORDER = [1,2,3]
SPLIT_CHUNKSIZE = 3
WORLD_SIZE = 8

def init_world(size: int):
    grid = GridWorld(size)

    agent_1 = Agent(0,0)
    agent_2 = Agent(grid.size-1,0)
    agent_3 = Agent(0,grid.size-1)
    agent_4 = Agent(grid.size-1,grid.size-1)
    agents = [agent_1, agent_2, agent_3,agent_4]
    grid.init_agent(agents[:NUM_AGENTS])
    print(grid)
    return grid

grid = init_world(WORLD_SIZE)
model = path_model(grid,Coverage,steps=STEPS,agent_order=AGENT_ORDER)
score_seq = seq_solve(model)
assert score_seq == grid.get_score()
print(f"Sequential:\n{grid}")
grid.reset()

score_split = split_seq_solve(grid, Coverage, steps=STEPS,agent_order=AGENT_ORDER, chunksize=SPLIT_CHUNKSIZE)
assert score_split == grid.get_score()
print(f"Split-Sequential:\n{grid}")
grid.reset()

model = path_model(grid,Coverage,steps=STEPS)
score_opt, agent_order = cen_solve(model)
assert score_opt == grid.get_score()

print(f"Optimal:\n{grid}")



print(f"Sequential Score: {score_seq}")
print(f"Split Score: {score_split}")
print(f"Optimal Score: {score_opt}")
print(f"Seq/Opt Ratio: {score_seq/score_opt*100}%")