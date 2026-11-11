"""Calibrate and evaluate k=2 synthetic histories on separate seed groups."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import pdist

from generate_synthetic_batch import SCENARIOS, STEPS, check_batch, generate_batch

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "synthetic"

DETECTORS = ("cumulative", "rolling10", "rolling20")
TARGETS = (0.05, 0.10, 0.20)
REFERENCE_INDEX = 19
DETECTION_START = 20
USERS_PER_SEED = 500
CALIBRATION_SEEDS = range(5)
EVALUATION_SEEDS = range(5, 10)
SUMMARY_COLUMNS = (
    "detector", "target_false_alarm_rate", "theta", "calibration_false_alarm_rate",
    "evaluation_false_alarm_rate", "scenario", "users", "hits", "early_alarms",
    "misses", "mean_delay_steps_among_hits",
)


def load_seed(seed, diversity_series, *, stationary_only=False):
    items, histories, changes, favorites = generate_batch(seed, USERS_PER_SEED, 2, 40, 80)
    assert histories.shape == (USERS_PER_SEED, len(SCENARIOS), STEPS)
    assert np.all((40 <= changes) & (changes <= 80))
    check_batch(items, histories, changes, favorites)
    selected = histories[:, :1] if stationary_only else histories
    flat_scores = diversity_series(items, selected.reshape(-1, STEPS))
    scores = {name: flat_scores[name].reshape(selected.shape) for name in DETECTORS}

    # Independently check the reference and change boundaries on one history
    # from each scored scenario in every seed.
    for scenario, history in enumerate(selected[0]):
        change = changes[0]
        for step in (REFERENCE_INDEX, change - 1, change, change + 39, STEPS - 1):
            for name, window in zip(DETECTORS, (STEPS, 10, 20)):
                window_items = items[history[max(0, step - window + 1):step + 1]]
                expected = pdist(window_items, "cosine").mean()
                np.testing.assert_allclose(scores[name][0, scenario, step], expected, atol=1e-12)
    return scores, changes


def checked_alarms(values, theta, first_alarms):
    reference = values[:, REFERENCE_INDEX]
    alarms = first_alarms(values, reference, theta, DETECTION_START)
    below = values[:, DETECTION_START:] < ((1 - theta) * reference[:, None])
    runs = sliding_window_view(below, 3, axis=1).all(axis=-1)
    expected = np.where(runs.any(axis=1), runs.argmax(axis=1) + DETECTION_START + 2, -1)
    np.testing.assert_array_equal(alarms, expected)
    return alarms


def calibrate(stable, target, first_alarms):
    reference = stable[:, REFERENCE_INDEX]
    low, high = 0, 1000
    while low < high:
        middle = (low + high) // 2
        alarms = first_alarms(stable, reference, middle / 1000, DETECTION_START)
        if np.mean(alarms >= 0) <= target:
            high = middle
        else:
            low = middle + 1

    theta = low / 1000
    rate = np.mean(checked_alarms(stable, theta, first_alarms) >= 0)
    assert rate <= target
    if low:
        previous_rate = np.mean(checked_alarms(stable, (low - 1) / 1000, first_alarms) >= 0)
        assert previous_rate > target
    return theta, rate


def make_rows(evaluation, thresholds, first_alarms):
    summary_rows = []
    seed_rows = []
    for (name, target), (theta, calibration_rate) in thresholds.items():
        seed_scores = np.stack([scores[name] for _, scores, _ in evaluation])
        changes = np.stack([change for _, _, change in evaluation])
        alarms = checked_alarms(seed_scores.reshape(-1, STEPS), theta, first_alarms)
        alarms = alarms.reshape(seed_scores.shape[:-1])
        false_alarms = (alarms[:, :, 0] >= 0).sum(axis=1)

        for index, scenario in enumerate(SCENARIOS[1:], start=1):
            first = alarms[:, :, index]
            hit = (first >= changes) & (first <= changes + 39)
            early = (first >= 0) & (first < changes)
            late = first > changes + 39
            never = first < 0
            assert np.all(hit.astype(int) + early + late + never == 1)
            delay = first - changes
            prefix = (name, target, theta, calibration_rate)
            hits = int(hit.sum())
            summary_rows.append((
                *prefix, false_alarms.sum() / changes.size, scenario, changes.size,
                hits, int(early.sum()), changes.size - hits,
                float(delay[hit].mean()) if hits else None,
            ))
            for row, (seed, _, _) in enumerate(evaluation):
                seed_hits = int(hit[row].sum())
                seed_rows.append((
                    seed, *prefix, false_alarms[row] / USERS_PER_SEED, scenario,
                    USERS_PER_SEED, seed_hits, int(early[row].sum()), USERS_PER_SEED - seed_hits,
                    float(delay[row][hit[row]].mean()) if seed_hits else None,
                ))
    return summary_rows, seed_rows


def write_table(filename, header, rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_DIR / filename
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(destination)
    print(f"Saved {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_file", nargs="?", type=Path,
                        default=REPO_ROOT / "src" / "metrics.py")
    args = parser.parse_args()
    metrics_file = args.metrics_file.resolve(strict=True)
    if metrics_file.name != "metrics.py":
        parser.error("metrics_file must point to metrics.py")
    sys.path.insert(0, str(metrics_file.parent))
    from metrics import diversity_series, first_alarms

    calibration = {name: [] for name in DETECTORS}
    for seed in CALIBRATION_SEEDS:
        scores, _ = load_seed(seed, diversity_series, stationary_only=True)
        for name in DETECTORS:
            calibration[name].append(scores[name][:, 0])
        print(f"Calibration seed {seed}: verified 500 users × 3 histories; scored stationary", flush=True)
    thresholds = {
        (name, target): calibrate(np.concatenate(calibration[name]), target, first_alarms)
        for name in DETECTORS for target in TARGETS
    }

    evaluation = []
    for seed in EVALUATION_SEEDS:
        scores, changes = load_seed(seed, diversity_series)
        evaluation.append((seed, scores, changes))
        print(f"Evaluation seed {seed}: verified and scored 500 users × 3 histories", flush=True)

    summary, per_seed = make_rows(evaluation, thresholds, first_alarms)
    write_table("synthetic_alarm_results.csv", SUMMARY_COLUMNS, summary)
    write_table("synthetic_alarm_results_by_seed.csv", ("seed", *SUMMARY_COLUMNS), per_seed)


if __name__ == "__main__":
    main()
