"""Rendering helpers for persisted sweep data.

Two top-level entry points:
    - render_heatmaps(df, meta, output_root)  -> per-agent ratio heatmaps.
    - render_scatter(raw_df, meta, ...)       -> score-vs-runtime scatter
                                                 of methods relative to a
                                                 reference method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import FuncNorm, TwoSlopeNorm


SCATTER_SERIES_COLUMNS: tuple[str, ...] = (
    "method",
    "chunksize",
    "agents",
    "steps",
)


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
        vmin=max(0.5, center - max_dev),
        vcenter=center,
        vmax=min(3, center + max_dev),
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


def render_heatmaps(
    df: pd.DataFrame,
    meta: dict,
    output_root: Path = Path("results"),
) -> Path:
    """Render the standard 4-heatmap family for every agent count in the sweep."""
    grid_size = int(meta["grid_size"])
    max_agents = int(meta["max_agents"])
    name = meta["name"]

    grid_dir = output_root / name / f"grid_{grid_size}x{grid_size}"
    grid_dir.mkdir(parents=True, exist_ok=True)

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

    score_label = "Score ratio (split / greedy)\n1 = equal, >1 split better, <1 split worse"
    runtime_label = "Runtime ratio (split / greedy)\n1 = equal, >1 split slower, <1 split faster"

    for n in range(1, max_agents + 1):
        plot_ratio_heatmap(
            df,
            agents=n,
            value_col="split_ratio_to_seq_min",
            title=f"Split vs Greedy Min Performance by Chunksize — agents={n}",
            colorbar_label=score_label,
            filename=(
                f"{name}__split_min_perf_heatmap"
                f"__agents_{n}"
                f"__grid_{grid_size}x{grid_size}.png"
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
                f"{name}__split_mean_perf_heatmap"
                f"__agents_{n}"
                f"__grid_{grid_size}x{grid_size}.png"
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
                f"{name}__split_max_runtime_heatmap"
                f"__agents_{n}"
                f"__grid_{grid_size}x{grid_size}.png"
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
                f"{name}__split_mean_runtime_heatmap"
                f"__agents_{n}"
                f"__grid_{grid_size}x{grid_size}.png"
            ),
            cmap=runtime_cmap,
            norm=runtime_norm,
            save_dir=grid_dir,
        )

    return grid_dir


def _filters_suffix(filters: dict | None) -> str:
    if not filters:
        return ""
    parts = [f"{k}_{v}" for k, v in sorted(filters.items())]
    return "__" + "_".join(parts)


def _apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, value in filters.items():
        if col not in df.columns:
            raise ValueError(
                f"Cannot filter on unknown column {col!r}. "
                f"Available columns: {list(df.columns)}"
            )
        mask &= df[col] == value
    return df[mask]


def _ratio_axis_bounds(values: np.ndarray, *, max_factor: float = 10.0) -> tuple[float, float]:
    """Symmetric log bounds around 1.0, clipped to [1/max_factor, max_factor]."""
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return (1.0 / max_factor, max_factor)

    log_dev = np.max(np.abs(np.log(finite)))
    log_dev = min(log_dev, np.log(max_factor))
    factor = float(np.exp(log_dev)) if log_dev > 0 else 1.05
    return (1.0 / factor, factor)


def render_scatter(
    raw_df: pd.DataFrame,
    meta: dict,
    *,
    series_by: str = "method",
    reference_method: str = "seq_greedy_solve",
    output_root: Path = Path("results"),
    filters: dict | None = None,
) -> Path:
    """Render a score-vs-runtime scatter with each non-reference result
    plotted as a ratio against the reference method run on the same
    `(agents, steps, chunksize, start_row, start_col)` cell.

    Args:
        raw_df: per-method, per-start results (from `storage.load_sweep_raw_df`).
        meta: the sweep header dict (used for output path and titles).
        series_by: column controlling color/marker. One of
            {"method", "chunksize", "agents", "steps"}.
        reference_method: method used as the denominator for ratios.
        output_root: parent directory under which the plot is saved.
        filters: optional dict of column->value to slice `raw_df` before plotting
            (e.g. `{"agents": 3}`).

    Returns:
        The directory the plot was written to.
    """
    if series_by not in SCATTER_SERIES_COLUMNS:
        raise ValueError(
            f"series_by={series_by!r} not supported. "
            f"Choose one of {SCATTER_SERIES_COLUMNS}."
        )

    if raw_df.empty:
        raise ValueError("raw_df is empty; nothing to plot.")

    df = _apply_filters(raw_df, filters)
    if df.empty:
        raise ValueError(f"No rows left after applying filters={filters!r}.")

    methods_present = set(df["method"].unique())
    if reference_method not in methods_present:
        raise ValueError(
            f"reference_method={reference_method!r} not found in raw data. "
            f"Available methods: {sorted(methods_present)}"
        )

    join_keys = ["agents", "steps", "chunksize", "start_row", "start_col"]

    reference = (
        df[df["method"] == reference_method]
        .rename(columns={"score": "ref_score", "runtime": "ref_runtime"})
        [join_keys + ["ref_score", "ref_runtime"]]
    )

    non_ref = df[df["method"] != reference_method].copy()
    if non_ref.empty:
        raise ValueError(
            f"After filtering, only the reference method "
            f"({reference_method!r}) is present; nothing to plot."
        )

    merged = non_ref.merge(reference, on=join_keys, how="inner")

    merged["runtime_ratio"] = merged["runtime"] / merged["ref_runtime"]
    merged["score_ratio"] = merged["score"] / merged["ref_score"]

    merged = merged[
        np.isfinite(merged["runtime_ratio"])
        & np.isfinite(merged["score_ratio"])
        & (merged["runtime_ratio"] > 0)
        & (merged["score_ratio"] > 0)
    ]

    if merged.empty:
        raise ValueError(
            "No finite, positive ratios left to plot (check for zero "
            "scores or runtimes in the reference method)."
        )

    fig, ax = plt.subplots(figsize=(7, 6))

    series_values = sorted(merged[series_by].unique(), key=lambda v: (str(type(v)), v))
    cmap = plt.get_cmap("tab10")

    for i, value in enumerate(series_values):
        subset = merged[merged[series_by] == value]
        ax.scatter(
            subset["runtime_ratio"],
            subset["score_ratio"],
            label=f"{series_by}={value}",
            color=cmap(i % cmap.N),
            alpha=0.7,
            edgecolors="none",
            s=28,
        )

    ax.axhline(1.0, linestyle="--", linewidth=0.8, color="gray")
    ax.axvline(1.0, linestyle="--", linewidth=0.8, color="gray")
    ax.scatter(
        [1.0], [1.0],
        marker="x", color="black", s=60, zorder=5,
        label=f"{reference_method} (reference)",
    )

    x_min, x_max = _ratio_axis_bounds(merged["runtime_ratio"].to_numpy())
    y_min, y_max = _ratio_axis_bounds(merged["score_ratio"].to_numpy())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel(f"Runtime ratio (method / {reference_method})")
    ax.set_ylabel(f"Score ratio (method / {reference_method})")

    title_bits = [
        f"Score vs Runtime ({meta['name']}, "
        f"grid {meta['grid_size']}x{meta['grid_size']})",
        f"colored by {series_by}",
    ]
    if filters:
        filt_str = ", ".join(f"{k}={v}" for k, v in sorted(filters.items()))
        title_bits.append(f"filters: {filt_str}")
    ax.set_title("\n".join(title_bits))

    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    grid_size = int(meta["grid_size"])
    grid_dir = output_root / meta["name"] / f"grid_{grid_size}x{grid_size}"
    grid_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{meta['name']}__scatter__seriesby_{series_by}"
        f"{_filters_suffix(filters)}"
        f"__grid_{grid_size}x{grid_size}.png"
    )

    fig.tight_layout()
    fig.savefig(grid_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return grid_dir
