# T-RECS matrix factorization experiment

This experiment compares the original MF policy with a candidate that continues to recommend relevant items late in the simulation. It generates local consumption histories for the visualization and records the twenty-step narrowing diagnostic agreed for this analysis. The candidate is an opt-in configuration for review alongside the existing MF application case.

## Before and after

All policies use 500 users, 2,000 items, and 20 true topic attributes. User preferences and item attributes are independently drawn from Dirichlet(0.5). Every policy starts with the same 20 Random steps within a seed, then runs for 100 evaluation steps. Preferences stay fixed, each slate has ten distinct items, and each user consumes one new item per step. In the MF policies, LensKit retains cumulative interactions and uses 20 ALS iterations per fit, `weight=40` and `reg=0.1`.

| Policy | Latent factors | Evaluation slate |
| --- | ---: | --- |
| Baseline | 10 | 10 ranked MF items |
| Candidate | 20 | 8 ranked MF items and 2 uniform draws from the other eligible items |
| Random control | — | 10 uniform draws from the eligible items |

The baseline fit captured preferences early but its late top-ten ranking fell below Random. The candidate's pure top-ten ranking and the choice from its actual mixed slate are evaluated separately. Each comparison uses the same user's eligible pool immediately before the choice. For the choice comparison, Random is the exact expected best item out of ten uniform draws without replacement. In both development runs, the candidate beat Random on average in the pure ranking at all 100 steps. All ten sampled users beat Random on average over the final ten steps, and their mean actual choice also beat the same-pool Random expectation. The recommendation-quality cohort uses seed `130`. MF is trained on all 500 users.

| Seed | Baseline pure MF vs. Random slate | Candidate pure MF vs. Random slate | Candidate actual choice vs. Random choice |
| --- | ---: | ---: | ---: |
| 128 | 0.0314 / 0.0487 | 0.0611 / 0.0477 | 0.0763 / 0.0733 |
| 133 | 0.0360 / 0.0486 | 0.0641 / 0.0475 | 0.0848 / 0.0771 |

Table values average over the ten sampled users and the last ten evaluation steps. Match is the inner product of true user preferences and item attributes.

For consumption narrowing, count every person who has any twenty consecutive evaluation steps within one fixed set of one or two main topics. Each item's main topic is its largest true attribute. The first qualifying window gives an observed start and a confirmation on its twentieth step. A transition view can stop at confirmation while retaining the complete history for analysis of any later recovery.

| Development run | Users reaching twenty steps | Users reaching twenty steps in their own top two topics |
| --- | ---: | ---: |
| Baseline, seed 128 | 31/500 | 16/500 |
| Candidate, seed 128 | 159/500 | 109/500 |
| Baseline, seed 133 | 37/500 | 22/500 |
| Candidate, seed 133 | 139/500 | 96/500 |
| Random, seed 128 | 0/500 | 0/500 |
| Random, seed 133 | 0/500 | 0/500 |

No candidate user remained in one fixed set of one or two topics throughout the final twenty steps in either development run. Treating a confirmed episode as persistent under unlimited, visible high-match supply is a separate hypothesis. How much a larger catalog would lengthen such episodes is an open question. These are development observations about simulated consumption. The synthetic `abrupt` and `gradual` labels still describe known changes to the candidate pool, and the twenty-step count is a descriptive MF diagnostic.

## What the development controls showed

- A fixed-history comparison pointed to interaction feedback. The same MF fitted to choices from Random exposure ranked late items substantially better than when it was fitted to choices from the baseline MF policy.
- Reducing `weight` to `1` helped a late fixed-history snapshot. In the full feedback loop for seed `128` it left all ten sampled users below Random on average over the final ten steps.
- At 10 latent factors, two Random slots improved the late ranking to 4/10 wins in seed `128`. Increasing to 20 factors with that same exploration rule yielded 10/10 wins in both development seeds.
- In seed `128`, adding ten new items per evaluation step while keeping uniform exploration had little effect on the late share of choices in users' own two favorite topics. Dedicating the two Random slots to current arrivals increased that share from 51.8% in the fixed catalog to 60.7%. Only 1/500 user consumed within a fixed set of one or two main topics for all of the final twenty steps under fresh-item exposure. The arrival stream used seed `134`.
- Allowing repeats in the same fixed catalog made 444/500 users choose the exact same item in each of the last twenty steps. This serves as a stress test for repeated-item behavior.
- Auditing the first ten-step topic episode showed depletion and limited exposure together. At 874 of 893 first departures across seeds `128` and `133`, the pool still contained a same-group item with higher true utility than the actual outside-group choice. Those higher-utility items were absent from the displayed slate for that step.

## Reproduce the data locally

Use the pinned Python 3.10 environment in the [repository setup](../README.md#environment-and-installation). Run the following from the repository root in the active environment. Use separate output directories to keep comparisons and application data easy to identify.

```sh
run_dir=$(mktemp -d)
mkdir -p "$run_dir/numba" "$run_dir/mpl"
export NUMBA_CACHE_DIR="$run_dir/numba" MPLCONFIGDIR="$run_dir/mpl"
export NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1

python scripts/check_adapter.py
python scripts/check_metrics.py
python scripts/check_mf_experiment.py
python scripts/run_mf_experiment.py --seeds 128 133 --output-dir "$run_dir/comparison"
```

The runner regenerates results locally. `summary.json` reports the configuration, seed, cohort IDs, first and last ten-step recommendation quality, and narrowing counts for any twenty-step window and the final window. Each policy's `mf_seed<seed>.npz` stores `histories` (500 × 120), `slates` (500 × 120 × 10), `users` (500 × 20), `items` (20 × 2,000), the run parameters, and the cohort's per-step comparisons. The simulator checks matched startup histories, slate eligibility, unique recommendations, and the user's choice against the true score matrix. The focused check also verifies the Random expectation by enumeration and the narrowing rule on known cases. Keep the full generated files in the local output directory.

The published runner reproduced all displayed slates and consumption histories from the candidate development runs for both seeds. The twenty-step counts were also checked by directly enumerating the windows in their output histories.

## Use the candidate with the existing visualization

The original prototype used a locally generated seed `123`. Generate the candidate for that seed in its own output directory:

```sh
python scripts/run_mf_experiment.py --seeds 123 --policies candidate --output-dir "$run_dir/application"
```

The new history is at `$run_dir/application/candidate/mf_seed123.npz`. It retains the `histories`, `items`, `users`, `seed` and `iterations` keys used by the existing local MF exporter. In the original checkout, point the MF input of `scripts/export_vis_data.py` at the candidate history directory. Export `mf.json` and rebuild the prototype bundle with `prototype/build_triage_data.py` where that prototype is available. Retain the old history for comparison. The earlier RQ1/RQ2 synthetic results have their own data generator and calibration.
