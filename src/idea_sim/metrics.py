
from dataclasses import dataclass


import pandas as pd

from idea_sim.env import Result
from idea_sim.policies.centralized import best_seq_greedy_solve


@dataclass
class CompareResults:
    results: list[Result]
    def summary(self):
        pass

    def to_dataframe(self):
        return pd.DataFrame([r.summary_dict() for r in self.results])

    def __str__(self):
        output = self.to_dataframe()
        return output

@dataclass
class CompareSweep:
    results: list[Result]

    def summary(
        self,
        baseline_method: str = best_seq_greedy_solve.__name__,
        reference_method: str | None = None,
        include_baseline: bool = True
    ):
        df = self.to_dataframe()

        summary = df.groupby("method").agg({
            "score": ["min", "max", "mean"],
            "runtime": ["min", "max", "mean"],
        })

        def add_ratios(
            metric: str,
            stats: tuple[str, ...],
            method: str,
            label: str,
        ) -> None:
            if method not in summary.index:
                raise KeyError(f"{method=} not found in summary index")

            for stat in stats:
                denominator = summary.loc[method, (metric, stat)]

                summary[(metric, f"{stat}_ratio_to_{label}")] = (
                    summary[(metric, stat)] / denominator
                )
        if include_baseline:
            add_ratios(
                metric="score",
                stats=("min", "mean"),
                method=baseline_method,
                label="baseline",
            )

            add_ratios(
                metric="runtime",
                stats=("min", "max", "mean"),
                method=baseline_method,
                label="baseline",
            )

        if reference_method is not None:
            add_ratios(
                metric="score",
                stats=("min", "mean"),
                method=reference_method,
                label="reference",
            )

            add_ratios(
                metric="runtime",
                stats=("min", "max", "mean"),
                method=reference_method,
                label="reference",
            )

        return summary.sort_index(axis=1)   

    def to_dataframe(self):
        
        return pd.DataFrame([r.summary_dict() for r in self.results])

    def __str__(self):
        output = self.to_dataframe().__str__()
        return output