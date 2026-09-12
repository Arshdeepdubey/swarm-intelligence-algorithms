# Results

This folder contains a real, reproducible run of the full experimental
protocol described in the SDP report (Sec 4.3, "Experimental Outcomes"):
each of the four SI algorithms (PSO, ACO, ABC, Cuckoo Search) run **10
times** against **each of the four datasets** (Iris, Breast Cancer, Wine,
Digits), using the population size (10) and iteration count (100)
documented in the report's Table 7.

## Files

- `demo_results.csv` — one row per run (160 rows = 4 algorithms x 4
  datasets x 10 runs), with columns: `algorithm`, `dataset`, `run`,
  `seed`, `n_selected`, `n_total_features`, `subset_accuracy`,
  `all_features_accuracy`, `internal_fitness`, `n_evaluations`.
- `summary.csv` — the same data aggregated to mean/std per
  (dataset, algorithm), plus `accuracy_gain` (subset accuracy minus the
  all-features baseline).
- `comparison.png` — grouped bar chart, one panel per dataset, comparing
  mean subset accuracy (with std-dev error bars) against the all-features
  baseline (dashed line).

## Reproducing this run

```bash
pip install -r requirements.txt
python scripts/run_experiments.py \
    --datasets iris wine breast_cancer digits \
    --algorithms pso aco abc cuckoo \
    --n-runs 10 --population-size 10 --max-iter 100 \
    --output results/demo_results.csv
python scripts/compare_algorithms.py --input results/demo_results.csv --output-dir results
```

On this machine the four-dataset, four-algorithm, 10-run sweep took
roughly 12-13 minutes in total, almost entirely dominated by the Digits
dataset's ABC runs (~35s/run, because ABC evaluates 2x the population per
iteration for its employed + onlooker bee phases). Use `--n-runs`,
`--max-iter`, or `--datasets`/`--algorithms` to shrink this for a quicker
smoke test.

## What the numbers show

Across all four datasets every algorithm reduces the feature count
substantially (e.g. Digits: ~40-45 of 64 features selected; Iris: ~1-1.4
of 4) while keeping subset accuracy within a percentage point or two of
the all-features baseline -- on Iris, subset accuracy actually *exceeds*
the baseline (fewer, more informative features reduce overfitting on a
150-sample dataset).

No single algorithm dominates every dataset in this particular run: ABC
and PSO tie for best on Digits, ACO edges out the others on Breast
Cancer, and ABC leads (tied with the others within noise) on Iris, while
PSO leads on Wine. This is a useful, evidence-based nuance versus the SDP
report's headline claim that "PSO is performing better as compared to the
other three algorithms" (Sec 5, Conclusion) -- see
`docs/documentation_review.md` for a full discussion of why (different
random seeds/hyperparameters than the original report, and the original
report's own results table was not extractable as data, only as an image,
so an exact apples-to-apples reproduction isn't possible from the source
files alone).
