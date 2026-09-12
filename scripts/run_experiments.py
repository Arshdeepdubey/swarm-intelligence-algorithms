#!/usr/bin/env python3
"""
CLI: reproduce the SDP report's experimental protocol.

    python scripts/run_experiments.py --n-runs 10 --output results/results.csv

By default this runs all four algorithms (PSO, ACO, ABC, Cuckoo Search) on
all four datasets (Iris, Breast Cancer, Wine, Digits) for 10 runs each,
using the population size (10) and iteration count (100) documented in the
SDP report's parameter table. That is 4 x 4 x 10 x 100 x 10 = 160,000 SVM
fits, which can take a while on the full Digits dataset -- use
``--n-runs`` / ``--max-iter`` / ``--population-size`` / ``--datasets`` /
``--algorithms`` to shrink the run for a quick smoke test.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_fs.algorithms import ALGORITHMS  # noqa: E402
from swarm_fs.datasets import available_datasets  # noqa: E402
from swarm_fs.experiment import run_all_experiments, summarize_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHMS.keys()),
        choices=list(ALGORITHMS.keys()),
        help="Which SI algorithms to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=available_datasets(),
        choices=available_datasets(),
        help="Which datasets to run against.",
    )
    parser.add_argument("--n-runs", type=int, default=10, help="Runs per (algorithm, dataset).")
    parser.add_argument("--population-size", type=int, default=10, help="Table 7: population size.")
    parser.add_argument("--max-iter", type=int, default=100, help="Table 7: iterations.")
    parser.add_argument("--alpha", type=float, default=0.99, help="Accuracy/feature-count fitness weight.")
    parser.add_argument("--kernel", type=str, default="linear", help="SVM kernel.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--output", type=str, default="results/results.csv", help="Where to write the per-run CSV."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    df = run_all_experiments(
        algorithms=args.algorithms,
        datasets=args.datasets,
        n_runs=args.n_runs,
        population_size=args.population_size,
        max_iter=args.max_iter,
        alpha=args.alpha,
        kernel=args.kernel,
        base_seed=args.seed,
        progress=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary = summarize_results(df)
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    elapsed = time.time() - start
    print(f"\nWrote {len(df)} run records to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Elapsed: {elapsed:.1f}s\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
