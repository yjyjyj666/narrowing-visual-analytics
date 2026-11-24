"""MF application case (docs/spec.md §7): simulate, then summarize alarm timing.

For each seed, 500 users consume from 2,000 items for 20 random startup steps and
100 steps of CheckedImplicitMF recommendations, retrained after every step. The
simulation loop matches scripts/audit_behavior.py. Histories are saved to
results/mf/mf_seed<seed>.npz (not committed) and reused on later runs.

Alarms use the thresholds calibrated on synthetic stationary histories, under two
references: `startup` (index 19, alarms from 20) and `early_mf` (index 29, alarms
from 30). MF has no known change point, so the summaries describe when alarms
occur and where detectors disagree; they make no hit-rate claims.
"""
import argparse
import csv
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from alarms import DETECTORS, REFERENCES, TARGETS, alarm_table, load_thresholds  # noqa: E402
from metrics import diversity_series  # noqa: E402

OUT = REPO_ROOT / "results" / "mf"
THRESHOLDS_CSV = REPO_ROOT / "results" / "synthetic" / "synthetic_alarm_results.csv"
USERS, ITEMS, DIMENSIONS, FACTORS, SLATE = 500, 2000, 20, 10, 10
STARTUP_STEPS, STEPS = 20, 120
FIRST_TEN_MF = range(20, 30)  # indices reported as "alarms in the first ten MF steps"
BUCKET = 10


def simulate(seed, iterations):
    """One MF run; returns (histories (users, steps), actual item attributes (dims, items))."""
    from mf_checked import CheckedImplicitMF

    rng = np.random.default_rng(seed)
    users = rng.dirichlet(np.full(DIMENSIONS, 0.5), size=USERS)
    items = rng.dirichlet(np.full(DIMENSIONS, 0.5), size=ITEMS).T
    model = CheckedImplicitMF(
        num_users=USERS, num_items=ITEMS, num_latent_factors=FACTORS,
        actual_user_representation=users, actual_item_representation=items,
        num_items_per_iter=SLATE, seed=seed,
        model_params={"iterations": iterations, "rng_spec": seed})
    consumed = np.zeros((USERS, ITEMS), dtype=bool)
    choices = []
    started = time.perf_counter()
    for t in range(STEPS):
        # Random slates during startup, MF slates afterwards; interactions accumulate.
        model.run(1, startup=t < STARTUP_STEPS, reset_interactions=False,
                  repeated_items=False, no_new_items=True)
        chosen = np.asarray(model.interactions, dtype=int).copy()
        assert not consumed[np.arange(USERS), chosen].any(), "an item was consumed twice"
        consumed[np.arange(USERS), chosen] = True
        choices.append(chosen)
        # First fit after the last startup step, then refit after every MF step.
        if t >= STARTUP_STEPS - 1:
            model.train()
        if (t + 1) % 40 == 0:
            print(f"seed {seed}: {t + 1}/{STEPS} steps ({time.perf_counter() - started:.0f}s)", flush=True)
    return np.stack(choices, axis=1), items, users


def check_one_user(items, history, series, alarms, thresholds):
    """Recompute one user's series and alarms by brute force, independently of metrics.py."""
    for t in (1, 19, 29, 60, STEPS - 1):
        expected = {"cumulative": pdist(items[history[:t + 1]], "cosine").mean(),
                    "rolling10": pdist(items[history[t - 9:t + 1]], "cosine").mean() if t >= 9 else np.nan,
                    "rolling20": pdist(items[history[t - 19:t + 1]], "cosine").mean() if t >= 19 else np.nan}
        for name, value in expected.items():
            assert np.isclose(series[name][0, t], value, equal_nan=True), (name, t)
    for reference, (reference_index, start) in REFERENCES.items():
        for (detector, target), first in alarms[reference].items():
            values = series[detector][0]
            limit = (1 - thresholds[detector, target]) * values[reference_index]
            run, expected = 0, -1
            for t in range(start, STEPS):
                run = run + 1 if values[t] < limit else 0
                if run == 3:
                    expected = t
                    break
            assert first[0] == expected, (reference, detector, target)


def summarize(seed, alarms):
    """Rows for the timing summary, the timing histogram and the disagreement table."""
    summary, timing, disagreement = [], [], []
    for reference, table in alarms.items():
        start = REFERENCES[reference][1]
        for (detector, target), first in table.items():
            fired = first >= 0
            summary.append({
                "seed": seed, "reference": reference, "detector": detector, "target_false_alarm_rate": target,
                "users": len(first), "alarmed": int(fired.sum()),
                # Timing only: this does not say the switch to MF caused the alarm.
                "alarmed_first_ten_mf_steps": int(np.isin(first, FIRST_TEN_MF).sum()) if reference == "startup" else None,
                "median_alarm_index": float(np.median(first[fired])) if fired.any() else None,
            })
            for low in range(start, STEPS, BUCKET):
                high = min(low + BUCKET, STEPS) - 1
                timing.append({"seed": seed, "reference": reference, "detector": detector,
                               "target_false_alarm_rate": target, "first_index": low, "last_index": high,
                               "alarmed": int(((first >= low) & (first <= high)).sum())})
        for target in TARGETS:
            for a, b in combinations(DETECTORS, 2):
                fa, fb = table[a, target] >= 0, table[b, target] >= 0
                disagreement.append({"seed": seed, "reference": reference, "target_false_alarm_rate": target,
                                     "detector_a": a, "detector_b": b, "both": int((fa & fb).sum()),
                                     "a_only": int((fa & ~fb).sum()), "b_only": int((~fa & fb).sum()),
                                     "neither": int((~fa & ~fb).sum())})
    return summary, timing, disagreement


def write_csv(name, rows):
    path = OUT / name
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 124, 125, 126, 127])
    parser.add_argument("--iterations", type=int, default=20, help="ALS iterations per fit")
    parser.add_argument("--resimulate", action="store_true", help="ignore saved histories")
    args = parser.parse_args()
    if set(args.seeds) & set(range(10)):
        parser.error("seeds 0-9 are reserved for the synthetic experiment")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="No measurements are currently defined")
    OUT.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(THRESHOLDS_CSV)

    per_seed, all_alarms = [], []
    for seed in args.seeds:
        # Step 1: simulate, or reuse the saved histories for this seed and setting.
        path = OUT / f"mf_seed{seed}.npz"
        if path.exists() and not args.resimulate:
            data = np.load(path)
            if int(data["iterations"]) != args.iterations:
                raise SystemExit(f"{path} used {int(data['iterations'])} ALS iterations; pass --resimulate")
            histories, items = data["histories"], data["items"]
            print(f"seed {seed}: reusing {path}", flush=True)
        else:
            histories, items, users = simulate(seed, args.iterations)
            np.savez_compressed(path, histories=histories, items=items, users=users,
                                seed=seed, iterations=args.iterations)
        assert histories.shape == (USERS, STEPS)
        assert all(len(np.unique(h)) == STEPS for h in histories)

        # Step 2: diversity series and alarms under both references.
        series = diversity_series(items.T, histories)
        alarms = {ref: alarm_table(series, thresholds, ref) for ref in REFERENCES}
        check_one_user(items.T, histories[0], {k: v[:1] for k, v in series.items()},
                       {ref: {k: v[:1] for k, v in table.items()} for ref, table in alarms.items()},
                       thresholds)

        # Step 3: per-seed rows, and keep the alarms for the pooled rows.
        per_seed.append(summarize(seed, alarms)[0])
        all_alarms.append(alarms)

    # Step 4: pool all seeds' users and summarize them as one group.
    pooled_alarms = {ref: {key: np.concatenate([a[ref][key] for a in all_alarms]) for key in all_alarms[0][ref]}
                     for ref in REFERENCES}
    summary, timing, disagreement = summarize("all", pooled_alarms)
    write_csv("mf_alarm_summary_by_seed.csv", [row for rows in per_seed for row in rows])
    write_csv("mf_alarm_summary.csv", summary)
    write_csv("mf_alarm_timing.csv", timing)
    write_csv("mf_detector_disagreement.csv", disagreement)


if __name__ == "__main__":
    main()
