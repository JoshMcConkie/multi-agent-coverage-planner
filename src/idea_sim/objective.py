from typing import Tuple, List
import numpy as np
from operator import add

def get_all_paths(grid: np.ndarray,steps,start_row,start_col):
    """Generate all possible path for an agent in the grid.

        #TODO: Avoid other agents starting positions.
    """
    if not grid:
        return None
    max_row = len(grid)
    max_col = len(grid[0])
    is_in_bounds = lambda rc: 0<= rc[0] and 0<= rc[1] and rc[0] < max_row and rc[1] < max_col
    try:
        assert is_in_bounds((start_row,start_col))
    except AssertionError:
        print("ERROR: Can't get paths. Start position out of bounds.")
        return None
        
    directions = [(-1,0),(0,-1),(0,1),(1,0)]
    paths = []
    
    def step(steps_rem, path):
        if not steps:
            paths.append(path)
        for dir in directions:
            new_step = map(add,path[-1],dir)
            if new_step not in path and is_in_bounds(new_step):
                step(steps-1,path+new_step)

    step(steps,[(start_row,start_col)])
    return paths

