from itertools import product
from operator import add


from idea_sim.objective import Objective
from idea_sim.env import GridWorld, Model


def utility_score(choice_set,util_mat, prior_util_row=None) -> tuple[int,list[int]]:
    '''Return utility score for given choice set of a utility matrix.
    This assumes submodularity such that double coverage offers no 
    bonus to utility.

    '''
    score = 0
    n_choices = len(choice_set)    # original number of choices w/o prior util row
    if prior_util_row is not None:
        util_mat += prior_util_row
        choice_set += [n_choices]
    
    for col in range(util_mat.shape[1]):
        # print(f"util crop: {util_mat[choice_set]}")
        score += max([util_mat[choice][col] for choice in choice_set])
    return score, choice_set[:n_choices]


def best_greedy_choice(options: list[int],choice_hist: list[int],util_mat, prior_util_row=None)-> tuple[int,list[int]]:
    '''Get next choice with greatest marginal utility.

    Args: 
        options (list): allocation options available to agent
        choice_hist (list): past selections made made (S_e in context of Nemhauser et al., 1978)
        util_mat (np.ndarray): The C (aka omega) matrix of allocations and utilities (Nemhauser et al., 1978)
    '''
   
    return max((utility_score(choice_hist+[o],util_mat, prior_util_row) for o in options))


def get_all_paths(grid: GridWorld,steps: int,
                  start_row: tuple[int,int],start_col:tuple[int,int]) -> list[list[tuple[int,int]]]:
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
        if steps_rem == 0:
            paths.append(path)
            return
        for dir in directions:
            new_step = tuple(map(add,path[-1],dir))
            if new_step not in path and is_in_bounds(new_step):
                step(steps_rem-1,path+[new_step])

    step(steps,[(start_row,start_col)])
    return paths

def update_agents(model: Model,path_ids: list[int])-> dict[int,list[int,int]]:
    '''Append the chosen paths to the agents in the grid.
    '''
    paths_by_agent = dict()
    for agent in model.grid.agents:
        agent_path_ids = model.agent_path_dict[agent.id]
        for path_id in path_ids:
            if path_id in agent_path_ids:
                agent.path += model.all_paths[path_id][1:]
        paths_by_agent[agent.id] = agent.path
    model.grid.update()
    return paths_by_agent

def path_model(grid: GridWorld,objective: Objective, steps: int, agent_order=None):
    '''Creates `Model` object containing grid, utility matrix, available agents path, agent_order
    (if applicable), and all available paths.
    '''
    all_paths = []
    i = 0
    agent_path_dict = dict()
    coordinates = list(product(range(grid.size),range(grid.size)))

    for agent in grid.agents:
        solutions = get_all_paths(grid,steps,agent.path[-1][0],agent.path[-1][1])
        # print(f"Agent {agent.id} Solution Sample: {solutions[:5]}")
        all_paths += solutions
        agent_path_dict[agent.id] = [i+e for e in range(len(solutions))] # collect addresses of path options as found in all_paths
        i += len(solutions)
    util_mat = objective.build_util_matrix(all_paths,coordinates) # rows: path idx, cols: coord idx
    
    return Model(grid,objective,util_mat,agent_path_dict,agent_order,all_paths, steps)    