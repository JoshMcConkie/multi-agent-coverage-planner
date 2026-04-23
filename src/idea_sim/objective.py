from typing import Tuple, List
import numpy as np
from operator import add
from itertools import product
from idea_sim.env import GridWorld
from idea_sim.policies.sequential import seq_solve
from idea_sim.policies.centralized import cen_solve

class Objective:
    def build_util_matrix():
        print("Please define objective")
        raise NotImplementedError

class Coverage(Objective):
    @staticmethod
    def build_util_matrix(paths,coordinates):
        util_mat = np.zeros((len(paths),(len(coordinates))))
        for i,path in enumerate(paths):
            for j,c in enumerate(coordinates):
                # 1 represents coverage
                util_mat[i][j] = 1 if c in path else 0 
        return util_mat
        
def get_all_paths(grid: GridWorld,steps: int,
                  start_row: Tuple[int,int],start_col:Tuple[int,int]) -> List[Tuple[int,int]]:
    """Generate all possible path for an agent in the grid.

        #TODO: Avoid other agents starting positions.
    """
    if not grid:
        return None
    max_row = grid.grid.shape[0]
    max_col = grid.grid.shape[1]
    is_in_bounds = lambda rc: 0 <= rc[0] < max_row and 0 <= rc[1] < max_col
    try:
        assert is_in_bounds((start_row,start_col))
    except AssertionError:
        print("ERROR: Can't get paths. Start position out of bounds.")
        return None
        
    directions = [(-1,0),(0,-1),(0,1),(1,0)]
    paths = []
    
    def step(steps_rem, path):
        if not steps_rem:
            paths.append(path)
        for dir in directions:
            new_step = tuple(map(add,path[-1],dir))
            if new_step not in path and is_in_bounds(new_step):
                step(steps_rem-1,path+[new_step])

    step(steps,[(start_row,start_col)])
    return paths

def solve(grid: GridWorld,objective: type[Objective],method, steps: int, agent_order=None):
    if agent_order is None and method == seq_solve:
        raise("ERROR: Please provide agent order for sequential solver.")
    all_paths = []
    i = 0
    agent_dict = dict()
    coordinates = list(product(range(grid.size),range(grid.size)))
    for agent in grid.agents:
        solutions = get_all_paths(grid,steps,agent.row,agent.col)
        # print(f"Agent {agent.id} Solution Sample: {solutions[:5]}")
        all_paths += solutions
        agent_dict[agent.id] = [i+e for e in range(len(solutions))]
        i += len(solutions)
    util_mat = objective.build_util_matrix(all_paths,coordinates) # rows: path idx, cols: coord idx
    
    # print(f"Utililty Matrix Shape: {util_mat.shape}")
    
    final_score, final_paths = method(util_mat,agent_dict,agent_order)
    for agent in grid.agents:
        paths = agent_dict[agent.id]
        for path in final_paths:
            if path in paths:
                agent.path = all_paths[path]
    grid.update()
    print("Final Grid")
    print(grid)
    return final_score
    # Build the utility grid for each solution
    # transpose from typical example
    
    
