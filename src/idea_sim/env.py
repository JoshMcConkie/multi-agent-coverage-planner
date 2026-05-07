from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
from idea_sim.objective import Objective


class GridWorld:
    '''Class for path-planning allocation world with discrete points'''
    def __init__(self,size):
        self.size = size
        self.grid = np.zeros((size,size))
        self.agents: List[Agent] = []
        self.fig = None
        self.plot_count = 0

    def __str__(self):
        grid_string = f"{'\n'.join([' '.join(map(str, row)) for row in self.grid])}\n"
        return grid_string
    
    def print_paths(self):
        path_string = '\n'+'\n'.join([f"  {agent.id}: {'->'.join(map(str,agent.path))}" for agent in self.agents])
        print(f"Agent paths: {path_string}")

    def update(self):
        '''Update agent current/past positions on grid'''
        for agent in self.agents:
            path = agent.path
            for i,pos in enumerate(path):
                row,col = pos
                if i+1 == len(path):
                    self.grid[row][col] = agent.id
                else:
                    self.grid[row][col] = -1 * agent.id
                    
    def init_agent(self, agent_list: List[Agent]):
        '''Attach agent(s) to grid'''
        for agent in agent_list:
            try:
                assert 0 <= agent.init_row and 0 <= agent.init_col and agent.init_row < self.size and agent.init_col < self.size
            except AssertionError:
                print("Start position out of bounds.")
                return
            self.agents.append(agent)
            agent.grid = self
        self.update()

    def reset_grid(self):
        '''Reset agent positions to start and clear grid'''
        self.grid = np.zeros((self.size,self.size))
        for agent in self.agents:
            agent.reset(update_grid=False)
        self.update()

    def get_score(self):
        return np.count_nonzero(self.grid)
    
    def add_plot(self):
        if self.fig is None:
            self.fig = plt.figure()
            legend_ax = self.fig.add_subplot(2,2,1)
            legend_ax.axis("off")
            ax = self.fig.add_subplot(2,2,2+self.plot_count)
            handles, labels = ax.get_legend_handles_labels()
            legend_ax.legend(handles, labels, loc="center") 
        else:
            ax = self.fig.add_subplot(2,2,2+self.plot_count)

        self.plot_count += 1

        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect("equal")
        ax.invert_yaxis()

        for agent in self.agents:
            x = []
            y = []

            for row, col in agent.path:
                x.append(col)
                y.append(row)

            ax.plot(x, y, "o-", label=f"Agent {agent.id}")

        # ax.legend()
        
    
    def show_plots(self):
        # plt.legend()
        plt.show()




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

@dataclass(frozen=True)
class Model:
    grid: GridWorld
    objective: Objective
    util_mat: np.ndarray
    agent_path_dict: Dict[int,list[int]]
    agent_order: List[int]
    all_paths: List[List[Tuple[int,int]]]
    steps: int

@dataclass(frozen=True)
class EmptyResult:
    def summary_dict(self)->dict:
        return dict()

@dataclass(frozen=True)
class Result:
    final_grid_array: np.ndarray
    paths_by_agent: Dict[int,List[Tuple[int,int]]]
    score: int
    method: callable
    runtime: any
    chosen_path_ids: List[int]
    metadata: Dict
    agent_order: List[int]
    steps: int

    def get_rows(self):
        method_name = (
            self.method.__name__ if callable(self.method)
            else str(self.method)
        )
        rows = [
            ("Method", method_name),
            ("Score", self.score),
            ("Runtime", self.runtime),
            # ("Agent Order", self.agent_order),
            ("Steps", self.steps),
            ("Metadata", self.metadata)
        ]
        return rows
    
    def summary_dict(self)->dict:
        return {
            "method": self.method.__name__ if callable(self.method) else str(self.method),
            "score": self.score,
            "runtime": self.runtime,
            "num_agents": len(self.agent_order)
        }

    def __str__(self):
        rows = self.get_rows()
        
        left_width = max(len(str(label)) for label, _ in rows)
        right_width = max(len(str(value)) for _, value in rows)

        line = f"+-{'-' * left_width}-+-{'-' * right_width}-+"

        table = [line]
        for label, value in rows:
            table.append(
                f"| {str(label).ljust(left_width)} | {str(value).ljust(right_width)} |"
            )
        table.append(line)

        return "\n".join(table)

