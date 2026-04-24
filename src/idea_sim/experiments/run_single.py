from idea_sim.env import GridWorld, Agent
from idea_sim.objective import Coverage
from idea_sim.policies.tools import path_model
from idea_sim.policies.centralized import cen_solve
from idea_sim.policies.sequential import seq_solve

def init_world(size: int):
    grid = GridWorld(size)

    agent_1 = Agent(0,0)
    agent_2 = Agent(grid.size-1,0)
    agent_3 = Agent(0,grid.size-1)
    agent_4 = Agent(grid.size-1,grid.size-1)
    grid.init_agent([agent_1, agent_2, agent_3,agent_4])
    print(grid)
    return grid

grid = init_world(6)
model = path_model(grid,Coverage,4,[1,2,3,4])
score_seq = seq_solve(model)
print(f"Sequential:\n{grid}")

grid.reset()
model = path_model(grid,Coverage,4)
score_opt, agent_order = cen_solve(model)
print(f"Optimal:\n{grid}")

print(f"Sequential Score: {score_seq}")
print(f"Optimal Score: {score_opt}")
print(f"Ratio: {score_seq/score_opt*100}%")