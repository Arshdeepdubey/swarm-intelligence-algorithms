#!/usr/bin/env python3
"""
CLI: build the comparison table + bar chart used to back the SDP report's
conclusion ("PSO is performing better as compared to the other three
algorithms" -- Sec 5, Conclusion) from a results CSV produced by
``run_experiments.py``.

    python scripts/compare_algorithms.py --input results/results.csv \
        --output-dir results/

Produces:
    * <output-dir>/summary.csv       -- mean/std subset accuracy per (dataset, algorithm)
    * <output-dir>/comparison.png    -- grouped bar chart, one panel per dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from swarm_fs.experiment import summarize_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default="results/results.csv")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating the PNG chart.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    summary = summarize_results(df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))

    best_per_dataset = summary.loc[summary.groupby("dataset")["mean_subset_accuracy"].idxmax()]
    print("\nBest algorithm per dataset (by mean subset accuracy):")
    print(best_per_dataset[["dataset", "algorithm", "mean_subset_accuracy"]].to_string(index=False))

    if not args.no_plot:
        _plot(summary, output_dir / "comparison.png")


def _plot(summary: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted(summary["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = summary[summary["dataset"] == dataset].sort_values("algorithm")
        ax.bar(subset["algorithm"], subset["mean_subset_accuracy"], yerr=subset["std_subset_accuracy"])
        baseline = subset["mean_all_features_accuracy"].iloc[0]
        ax.axhline(baseline, color="gray", linestyle="--", label="all-features baseline")
        ax.set_title(dataset)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Subset accuracy")
        ax.legend(fontsize=8)

    fig.suptitle("SI algorithm comparison: SVM subset accuracy by dataset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
