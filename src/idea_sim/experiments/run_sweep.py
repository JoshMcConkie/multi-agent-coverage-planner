from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from idea_sim.env import GridWorld, Agent
from idea_sim.experiments.run_single import run_single_compare
from idea_sim.metrics import CompareSweep
from idea_sim.objective import Coverage
from idea_sim.policies.tools import path_model
from idea_sim.policies.centralized import best_seq_greedy_solve
from idea_sim.policies.sequential import seq_greedy_solve, split_seq_solve

from tqdm import tqdm


#=====All agents start at same location======
def agents_stacked(num_agents, size, steps, chunksize):
    full_results = []
    iterations = 0
    for row in tqdm(range(size)):
        for c in range(size):
            Agent._id_counter = 0
            agents = [Agent(row, c) for _ in range(num_agents)]
            agent_order = [agent.id for agent in agents]
            single_result = run_single_compare(agents, agent_order, size,
                            steps, chunksize)
            full_results += single_result
            iterations += 1
    
    return CompareSweep(full_results), iterations
# print("=====Sweep Running: All agents start at same location=====")
# results, i = agents_stacked(num_agents=3,size=9,
#                         steps=6,chunksize=3)

# print(results.summary())
# print(f"Iterations: {i}")

# Checking chunk size

rows = []
num_agents=4
max_size=8
# steps=7
for steps in range(1,max_size+1):
    for n in range(1,num_agents+1):
        chunksize_results = dict()
        for chunk in range(1,steps+1):
            results, j = agents_stacked(n,max_size,
                                steps,chunksize=chunk)
            value = results.summary().loc[split_seq_solve.__name__,("score","min_ratio_to_baseline")]

            rows.append({
                "agents": n,
                "steps": steps,
                "chunksize": chunk,
                "min_ratio_to_baseline": value
            })

df = (
    pd.Series(chunksize_results, name="min_ratio_to_baseline")
    .rename_axis("chunksize")
    .reset_index()
)

plt.scatter(df["chunksize"],df["min_ratio_to_baseline"], marker="x",label="min ratio to baseline")
plt.xlabel("chunksize")
plt.title("Split-Seq Relative performance by chunksize")
plt.legend()
plt.savefig(
    f"results/same_start/same_start__split_perf_by_chunk__agents_{num_agents}__grid_{max_size}x{max_size}__steps_{steps}.png"
)

#=====Four