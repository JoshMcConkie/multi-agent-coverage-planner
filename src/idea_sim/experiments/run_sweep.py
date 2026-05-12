from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, FuncNorm
import numpy as np
import pandas as pd

from idea_sim.env import Agent
from idea_sim.experiments.run_single import run_single_compare
from idea_sim.metrics import CompareSweep
from idea_sim.policies.sequential import seq_greedy_solve, split_seq_solve

from tqdm.auto import tqdm

from pathlib import Path

#=====All agents start at same location======
def agents_stacked(num_agents, size, steps, chunksize, run_optimal=True):
    full_results = []
    iterations = 0
    for row in range(size):
        for c in range(size):
            Agent._id_counter = 0
            agents = [Agent(row, c) for _ in range(num_agents)]
            agent_order = [agent.id for agent in agents]
            single_result = run_single_compare(agents, agent_order, size,
                            steps, chunksize, run_optimal)
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

num_agents = 7
max_size = 8

total_runs = num_agents * max_size * (max_size + 1) // 2

SOLVE_OPTIMAL = False

with tqdm(total=total_runs, desc="Running same-start sweep") as pbar:
    for steps in range(1, max_size + 1):
        for n in range(1, num_agents + 1):
            for chunk in range(1, steps + 1):
                results, _ = agents_stacked(
                    n,
                    max_size,
                    steps,
                    chunksize=chunk,
                    run_optimal=SOLVE_OPTIMAL
                )

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

print("Creating charts...")

df = pd.DataFrame(rows)

output_dir = Path("results/same_start")
output_dir.mkdir(parents=True, exist_ok=True)


def centered_ratio_norm(series_list, center=1.0, min_dev=0.05):
    """
    Build a TwoSlopeNorm centered at 1.0.

    min_dev prevents vmin == vcenter == vmax when all ratios are exactly 1.
    """
    values = pd.concat(
        [s.dropna() for s in series_list],
        ignore_index=True,
    )

    if values.empty:
        max_dev = min_dev
    else:
        max_dev = np.nanmax(np.abs(values.to_numpy() - center))
        max_dev = max(max_dev, min_dev)

    return TwoSlopeNorm(
        vmin=max(0.5,center - max_dev),
        vcenter=center,
        vmax=min(3,center + max_dev),
    )
def centered_ratio_log_norm(series_list, center=1.0, min_factor=1.05, max_factor=3.0):
    """
    Build a logarithmic norm centered at `center`.

    Equal multiplicative changes around center get equal color distance:
        0.5 and 2.0 are equally far from 1.0.

    All plotted values must be positive.
    """
    if center <= 0:
        raise ValueError("center must be positive for logarithmic normalization")

    values = pd.concat(
        [s.dropna() for s in series_list],
        ignore_index=True,
    )

    values = values[np.isfinite(values) & (values > 0)]

    if values.empty:
        max_log_dev = np.log(min_factor)
    else:
        log_devs = np.abs(np.log(values.to_numpy() / center))
        max_log_dev = np.nanmax(log_devs)
        max_log_dev = max(max_log_dev, np.log(min_factor))

    max_log_dev = min(max_log_dev, np.log(max_factor))

    vmin = center * np.exp(-max_log_dev)
    vmax = center * np.exp(max_log_dev)

    return FuncNorm(
        (
            lambda x: np.log(x / center),
            lambda y: center * np.exp(y),
        ),
        vmin=vmin,
        vmax=vmax,
    )


score_norm = centered_ratio_log_norm([
    df["split_ratio_to_seq_min"],
    df["split_ratio_to_seq_mean"],
])

runtime_norm = centered_ratio_log_norm([
    df["split_ratio_to_seq_max_runtime"],
    df["split_ratio_to_seq_mean_runtime"],
])

score_cmap = plt.get_cmap("RdYlGn").copy()
score_cmap.set_bad(color="lightgray")

runtime_cmap = plt.get_cmap("RdYlGn_r").copy()
runtime_cmap.set_bad(color="lightgray")


def make_scale_note(norm):
    return (
        f"Colors use a log-centered scale at 1.00; "
        f"displayed range clipped to [{norm.vmin:.2f}, {norm.vmax:.2f}]."
    )


def add_centered_colorbar(fig, ax, im, norm, label):
    cbar = fig.colorbar(im, ax=ax, label=label)

    ticks = [norm.vmin, 1.0, norm.vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([
        f"{norm.vmin:.2f}",
        "1.00",
        f"{norm.vmax:.2f}",
    ])

    return cbar


def plot_ratio_heatmap(
    df,
    *,
    agents,
    value_col,
    title,
    colorbar_label,
    filename,
    cmap,
    norm,
    save_dir,
):
    df_n = df[df["agents"] == agents]

    heat = df_n.pivot(
        index="steps",
        columns="chunksize",
        values=value_col,
    )

    masked_heat = np.ma.masked_invalid(heat.to_numpy(dtype=float))

    fig, ax = plt.subplots()

    im = ax.imshow(
        masked_heat,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=norm,
    )

    ax.set_xlabel("chunksize")
    ax.set_ylabel("steps")
    ax.set_title(title)

    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns)

    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)

    add_centered_colorbar(fig, ax, im, norm, colorbar_label)

    fig.text(
        0.5,
        0.01,
        make_scale_note(norm),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    plt.savefig(
        save_dir / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


grid_dir = output_dir / f"grid_{max_size}x{max_size}"
grid_dir.mkdir(parents=True, exist_ok=True)

score_label = "Score ratio (split / greedy)\n1 = equal, >1 split better, <1 split worse"
runtime_label = "Runtime ratio (split / greedy)\n1 = equal, >1 split slower, <1 split faster"

for n in range(1, num_agents + 1):
    plot_ratio_heatmap(
        df,
        agents=n,
        value_col="split_ratio_to_seq_min",
        title=f"Split vs Greedy Min Performance by Chunksize — agents={n}",
        colorbar_label=score_label,
        filename=(
            f"same_start__split_min_perf_heatmap"
            f"__agents_{n}"
            f"__grid_{max_size}x{max_size}.png"
        ),
        cmap=score_cmap,
        norm=score_norm,
        save_dir=grid_dir,
    )

    plot_ratio_heatmap(
        df,
        agents=n,
        value_col="split_ratio_to_seq_mean",
        title=f"Split vs Greedy Mean Performance by Chunksize — agents={n}",
        colorbar_label=score_label,
        filename=(
            f"same_start__split_mean_perf_heatmap"
            f"__agents_{n}"
            f"__grid_{max_size}x{max_size}.png"
        ),
        cmap=score_cmap,
        norm=score_norm,
        save_dir=grid_dir,
    )

    plot_ratio_heatmap(
        df,
        agents=n,
        value_col="split_ratio_to_seq_max_runtime",
        title=f"Split vs Greedy Max Runtime by Chunksize — agents={n}",
        colorbar_label=runtime_label,
        filename=(
            f"same_start__split_max_runtime_heatmap"
            f"__agents_{n}"
            f"__grid_{max_size}x{max_size}.png"
        ),
        cmap=runtime_cmap,
        norm=runtime_norm,
        save_dir=grid_dir,
    )

    plot_ratio_heatmap(
        df,
        agents=n,
        value_col="split_ratio_to_seq_mean_runtime",
        title=f"Split vs Greedy Mean Runtime by Chunksize — agents={n}",
        colorbar_label=runtime_label,
        filename=(
            f"same_start__split_mean_runtime_heatmap"
            f"__agents_{n}"
            f"__grid_{max_size}x{max_size}.png"
        ),
        cmap=runtime_cmap,
        norm=runtime_norm,
        save_dir=grid_dir,
    )

print("Done.")