"""CLI dispatcher for rendering plots from a persisted sweep.

Examples:
    python -m coverage_planner.experiments.plot_sweep heatmap
    python -m coverage_planner.experiments.plot_sweep scatter --series-by method
    python -m coverage_planner.experiments.plot_sweep scatter --series-by chunksize \\
        --sweep-id 3
"""

from __future__ import annotations

import argparse

from coverage_planner.experiments import plotting, storage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render plots from a persisted sweep in results/sweeps.db.",
    )
    parser.add_argument(
        "kind",
        choices=["heatmap", "scatter"],
        help="Plot kind to render.",
    )
    parser.add_argument(
        "--sweep-id",
        type=int,
        default=None,
        help="Specific sweep_id to plot. Defaults to the latest matching --name.",
    )
    parser.add_argument(
        "--name",
        default="same_start",
        help="Sweep name to look up when --sweep-id is not given.",
    )
    parser.add_argument(
        "--series-by",
        default="method",
        choices=list(plotting.SCATTER_SERIES_COLUMNS),
        help="Column controlling color in scatter plots.",
    )
    parser.add_argument(
        "--reference-method",
        default="seq_greedy_solve",
        help="Reference method used as denominator for scatter ratios.",
    )

    args = parser.parse_args()

    with storage.connect() as conn:
        if args.kind == "heatmap":
            sweep_id, meta, df = storage.load_sweep_df(
                conn, sweep_id=args.sweep_id, name=args.name,
            )
            print(
                f"Loaded sweep_id={sweep_id} (name={meta['name']}, "
                f"grid={meta['grid_size']}x{meta['grid_size']}, "
                f"rows={len(df)})"
            )
            print("Rendering heatmaps...")
            out_dir = plotting.render_heatmaps(df, meta)
        else:
            sweep_id, meta, raw_df = storage.load_sweep_raw_df(
                conn, sweep_id=args.sweep_id, name=args.name,
            )
            print(
                f"Loaded sweep_id={sweep_id} (name={meta['name']}, "
                f"grid={meta['grid_size']}x{meta['grid_size']}, "
                f"raw_rows={len(raw_df)})"
            )
            print(f"Rendering scatter (series_by={args.series_by})...")
            out_dir = plotting.render_scatter(
                raw_df,
                meta,
                series_by=args.series_by,
                reference_method=args.reference_method,
            )

    print(f"Done. Wrote plots for sweep_id={sweep_id} to {out_dir}")


if __name__ == "__main__":
    main()
