from dataclasses import dataclass

import numpy as np
from typing import List, Tuple, Dict

class GridWorld:
    def __init__(self,size):
        self.size = size
        self.grid = np.zeros((size,size))
        self.agents: List[Agent] = []

    def __str__(self):
        grid_string = f"{'\n'.join([' '.join(map(str, row)) for row in self.grid])}\n"
        return grid_string
    
    def print_paths(self):
        path_string = '\n'+'\n'.join([f"  {agent.id}: {'->'.join(map(str,agent.path))}" for agent in self.agents])
        print(f"Agent paths: {path_string}")

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
                assert 0 <= agent.init_row and 0 <= agent.init_col and agent.init_row < self.size and agent.init_col < self.size
            except AssertionError:
                print("Start position out of bounds.")
                return
            self.agents.append(agent)
            agent.grid = self
        self.update()

    def reset(self):
        self.grid = np.zeros((self.size,self.size))
        for agent in self.agents:
            agent.reset(update_grid=False)
        self.update()

    def get_score(self):
        return np.count_nonzero(self.grid)


class Agent:
    _id_counter = 0

    def __init__(self,init_row, init_col):
        Agent._id_counter += 1
        self.id = Agent._id_counter
        self.init_row = init_row
        self.init_col = init_col
        self.path = [(self.init_row,self.init_col)]
        self.grid: GridWorld = None

    def step(self, next: Tuple[int,int]):
        self.path.append(next)
        self.grid.update()

    def reset(self, update_grid=True):
        self.path = [(self.init_row,self.init_col)]
        if update_grid:
            self.grid.update()

@dataclass
class Model:
    grid: GridWorld
    util_mat: np.ndarray
    agent_path_dict: Dict[int,list[int]]
    agent_order: List[int]
    all_paths: List[List[Tuple[int,int]]]
    chosen_path_ids: List[int] | None = None

