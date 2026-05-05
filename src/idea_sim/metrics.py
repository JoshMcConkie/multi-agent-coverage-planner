
from dataclasses import dataclass
from typing import List

import pandas as pd

from idea_sim.env import Result
from idea_sim.policies.centralized import best_seq_greedy_solve


@dataclass
class CompareResults:
    results: List[Result]
    def summary(self):
        pass

    def to_dataframe(self):
        return pd.DataFrame([r.summary_dict() for r in self.results])

    def __str__(self):
        output = self.to_dataframe()
        return output

@dataclass
class CompareSweep:
    results: List[Result]
    def summary(self, baseline_method: str = best_seq_greedy_solve.__name__):
        df = self.to_dataframe()

        summary = df.groupby("method").agg({
            "score": ["min", "max", "mean"],
            "runtime": ["min", "max", "mean"],
        })

        baseline = summary.loc[
            baseline_method,
            ("score", "min")
        ]

        summary[("score", "min_ratio_to_baseline")] = (
            summary[("score", "min")] / baseline
        )

        return summary

    def to_dataframe(self):
        
        return pd.DataFrame([r.summary_dict() for r in self.results])

    def __str__(self):
        output = self.to_dataframe().__str__()
        return output