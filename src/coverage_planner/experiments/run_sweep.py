from coverage_planner.env import Agent, EmptyResult
from coverage_planner.experiments import storage
from coverage_planner.experiments.run_single import run_single_compare
from coverage_planner.metrics import CompareSweep
from coverage_planner.policies.sequential import seq_greedy_solve, split_seq_solve

from tqdm.auto import tqdm

#=====All agents start at same location======
def agents_stacked(num_agents, size, steps, chunksize, run_optimal=True):
    full_results = []
    per_start = []
    for row in range(size):
        for c in range(size):
            Agent._id_counter = 0
            agents = [Agent(row, c) for _ in range(num_agents)]
            agent_order = [agent.id for agent in agents]
            single_result = run_single_compare(agents, agent_order, size,
                            steps, chunksize, run_optimal)
            full_results += single_result
            per_start.append((row, c, single_result))

    return CompareSweep(full_results), per_start

# Checking chunk size
rows = []
raw_rows = []

num_agents = 7
max_size = 8

total_runs = num_agents * max_size * (max_size + 1) // 2

SOLVE_OPTIMAL = False
SWEEP_NAME = "same_start"

with tqdm(total=total_runs, desc="Running same-start sweep") as pbar:
    for steps in range(1, max_size + 1):
        for n in range(1, num_agents + 1):
            for chunk in range(1, steps + 1):
                results, per_start = agents_stacked(
                    n,
                    max_size,
                    steps,
                    chunksize=chunk,
                    run_optimal=SOLVE_OPTIMAL
                )

                for start_row, start_col, single in per_start:
                    for res in single:
                        if isinstance(res, EmptyResult):
                            continue
                        method_name = (
                            res.method.__name__
                            if callable(res.method) else str(res.method)
                        )
                        raw_rows.append({
                            "agents": n,
                            "steps": steps,
                            "chunksize": chunk,
                            "start_row": start_row,
                            "start_col": start_col,
                            "method": method_name,
                            "score": float(res.score),
                            "runtime": float(res.runtime),
                        })

                summary = results.summary(reference_method=seq_greedy_solve.__name__,
                                          include_baseline=SOLVE_OPTIMAL)                

                split_ratio_to_seq_min_score = summary.loc[
                    split_seq_solve.__name__,
                    ("score", "min_ratio_to_reference"),
                ]
                split_ratio_to_seq_mean_score = summary.loc[
                    split_seq_solve.__name__,
                    ("score", "mean_ratio_to_reference"),
                ]

                split_ratio_to_seq_max_runtime = summary.loc[
                    split_seq_solve.__name__,
                    ("runtime", "max_ratio_to_reference"),
                ]
                split_ratio_to_seq_mean_runtime = summary.loc[
                    split_seq_solve.__name__,
                    ("runtime", "mean_ratio_to_reference"),
                ]
               
                new_row = {
                    "agents": n,
                    "steps": steps,
                    "chunksize": chunk,
                    "split_ratio_to_seq_min": split_ratio_to_seq_min_score,
                    "split_ratio_to_seq_mean": split_ratio_to_seq_mean_score,
                    "split_ratio_to_seq_max_runtime": split_ratio_to_seq_max_runtime,
                    "split_ratio_to_seq_mean_runtime": split_ratio_to_seq_mean_runtime,

                }
                

                if SOLVE_OPTIMAL:
                    seq_ratio_to_baseline_max_runtime = summary.loc[
                        seq_greedy_solve.__name__,
                        ("runtime", "max_ratio_to_baseline"),
                    ]
                    
                    seq_ratio_to_baseline_min_score = summary.loc[
                        seq_greedy_solve.__name__,
                        ("score", "min_ratio_to_baseline"),
                    ]
                    split_ratio_to_baseline_max_runtime = summary.loc[
                        split_seq_solve.__name__,
                        ("runtime", "max_ratio_to_baseline"),
                    ]
                    split_ratio_to_baseline_min_score = summary.loc[
                        split_seq_solve.__name__,
                        ("score", "min_ratio_to_baseline"),
                    ]
                    new_row |= {
                    "split_ratio_to_baseline": split_ratio_to_baseline_min_score,
                    "seq_ratio_to_baseline": seq_ratio_to_baseline_min_score,
                    "split_ratio_to_baseline_runtime": split_ratio_to_baseline_max_runtime,
                    "seq_ratio_to_baseline_runtime": seq_ratio_to_baseline_max_runtime,
                    }
                rows.append(new_row)     
                pbar.set_postfix({
                    "agents": n,
                    "steps": steps,
                    "chunk": chunk,
                })
                pbar.update(1)

print(
    f"Persisting {len(rows)} aggregate rows and "
    f"{len(raw_rows)} raw result rows to {storage.DEFAULT_DB_PATH}..."
)

with storage.connect() as conn:
    storage.init_schema(conn)
    sweep_id = storage.create_sweep(
        conn,
        name=SWEEP_NAME,
        grid_size=max_size,
        max_agents=num_agents,
        max_steps=max_size,
        solve_optimal=SOLVE_OPTIMAL,
    )
    storage.insert_sweep_rows(conn, sweep_id, rows)
    storage.insert_sweep_results(conn, sweep_id, raw_rows)

print(f"Done. Persisted sweep_id={sweep_id} to {storage.DEFAULT_DB_PATH}")
