import numpy as np
from typing import List, Tuple

class GridWorld:
    def __init__(self,size):
        self.size = size
        self.grid = np.zeros((size,size))
        self.agents = []

    def __str__(self):
        grid_string = f"{'\n'.join([' '.join(map(str, row)) for row in self.grid])}\n"
        path_string = '\n'+'\n'.join([f"  {agent.id}: {'->'.join(map(str,agent.path))}" for agent in self.agents])
        return grid_string+"Agent paths: "+path_string
    
    def update(self):
        for agent in self.agents:
            path = agent.path
            for i,pos in enumerate(path):
                row,col = pos
                if i+1 == len(path):
                    self.grid[row][col] = agent.id
                else:
                    self.grid[row][col] = -1 * agent.id
                    
    def init_agent(self, agent_list: List[Agent]):
        for agent in agent_list:
            try:
                assert 0 <= agent.row and 0 <= agent.col and agent.row < self.size and agent.col < self.size
            except AssertionError:
                print("Start position out of bounds.")
                return
            self.agents.append(agent)
            agent.grid = self
        self.update()
class Agent:
    _id_counter = 0

    def __init__(self,init_row, init_col):
        Agent._id_counter += 1
        self.id = Agent._id_counter
        self.row = init_row
        self.col = init_col
        self.path = [(self.row,self.col)]
        self.grid = None

    def step(self, next: Tuple[int,int]):
        self.path.append(next)
        self.grid.update()