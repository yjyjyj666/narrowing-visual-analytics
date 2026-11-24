"""H1 check: rescore the detectors with the pilot's label instead of the known change point.

The same evaluation histories (seeds 5-9, all scenarios) are scored twice with the
same calibrated thresholds and the same scoring rules (docs/spec.md §5):

- known_change: the generator's t_star; negatives are stationary histories.
- pilot_label: the label from the pilot analysis, computed from the consumed items.
  base = diversity of the items at indices 0-19; s(t) = diversity of the centred
  10-item window, indices t-5..t+4, which looks into the future. A history is
  narrowed if s(t) < 0.7 * base for 10 consecutive indices from 20 on, with onset
  at the first of them; stable if s(t) never stays below 0.85 * base for 10
  consecutive indices; otherwise ambiguous and excluded. Negatives are the
  histories labelled stable, whatever their scenario.

H1 expects Rolling-10's lead over cumulative to be larger under pilot_label,
because the centred window resembles Rolling-10.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from alarms import DETECTORS, TARGETS, alarm_table, load_thresholds  # noqa: E402
from metrics import diversity_series  # noqa: E402
from generate_synthetic_batch import SCENARIOS, STEPS, check_batch, generate_batch  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "synthetic"
EVALUATION_SEEDS = range(5, 10)
WINDOW, NARROW, STABLE, RUN = 10, 0.7, 0.85, 10
HIT_WINDOW = 39


def first_run(mask, length):
    """Index of the first entry of the first run of `length` True values per row, or -1."""
    count = np.zeros(len(mask), dtype=int)
    first = np.full(len(mask), -1)
    for i in range(mask.shape[1]):
        count = np.where(mask[:, i], count + 1, 0)
        found = (count == length) & (first < 0)
        first[found] = i - length + 1
    return first


def pilot_labels(series):
    """Pilot label and onset per history: 'narrowed', 'stable' or 'ambiguous'; onset -1 if none."""
    # The centred window t-5..t+4 is the rolling-10 window that ends at t+4.
    centred = np.full_like(series["rolling10"], np.nan)
    centred[:, WINDOW // 2:STEPS - WINDOW // 2 + 1] = series["rolling10"][:, WINDOW - 1:]
    base = series["cumulative"][:, 19]
    after = centred[:, 20:]
    # NaN comparisons are False, so the undefined tail never extends a run.
    onset = first_run(after < NARROW * base[:, None], RUN)
    onset = np.where(onset >= 0, onset + 20, -1)
    unstable = first_run(after < STABLE * base[:, None], RUN) >= 0
    labels = np.where(onset >= 0, "narrowed", np.where(unstable, "ambiguous", "stable"))
    return labels, onset, centred


def check_labels(items, histories, centred):
    """The centred window equals the brute-force diversity of items t-5..t+4."""
    for row in range(0, len(histories), 997):
        for t in (5, 20, 60, STEPS - 5):
            expected = pdist(items[histories[row, t - 5:t + 5]], "cosine").mean()
            assert np.isclose(centred[row, t], expected), (row, t)
        assert np.isnan(centred[row, :5]).all() and np.isnan(centred[row, STEPS - 4:]).all()


def score(first, positive, onset, negative):
    """Spec §5 rules with a given positive set, onset and negative set."""
    hit = positive & (first >= onset) & (first <= onset + HIT_WINDOW)
    early = positive & (first >= 0) & (first < onset)
    return {"positives": int(positive.sum()), "hits": int(hit.sum()), "early_alarms": int(early.sum()),
            "detection_rate": hit.sum() / positive.sum(),
            "mean_delay_steps_among_hits": float((first - onset)[hit].mean()) if hit.any() else None,
            "negatives": int(negative.sum()), "false_alarms": int((negative & (first >= 0)).sum()),
            "false_alarm_rate": (negative & (first >= 0)).sum() / negative.sum()}


def write_csv(name, rows):
    path = RESULTS_DIR / name
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}", flush=True)


def main():
    thresholds = load_thresholds(RESULTS_DIR / "synthetic_alarm_results.csv")
    with (RESULTS_DIR / "synthetic_alarm_results.csv").open(newline="") as file:
        published = {(r["detector"], float(r["target_false_alarm_rate"]), r["scenario"]): int(r["hits"])
                     for r in csv.DictReader(file)}

    # Step 1: regenerate the evaluation seeds and compute series, alarms and labels.
    seeds, scenarios, changes, labels, onsets, alarms = [], [], [], [], [], []
    for seed in EVALUATION_SEEDS:
        items, histories, change, favorites = generate_batch(seed, 500, 2, 40, 80)
        check_batch(items, histories, change, favorites)
        flat = histories.reshape(-1, STEPS)
        series = diversity_series(items, flat)
        label, onset, centred = pilot_labels(series)
        check_labels(items, flat, centred)
        seeds.append(np.full(len(flat), seed))
        scenarios.append(np.tile(SCENARIOS, len(histories)))
        # Stationary histories have no change point; -1 never counts as positive.
        changes.append(np.column_stack([np.full(len(histories), -1), change, change]).ravel())
        labels.append(label)
        onsets.append(onset)
        alarms.append(alarm_table(series, thresholds, "startup"))
        print(f"seed {seed}: scored and labelled {len(flat)} histories", flush=True)
    seeds, scenarios, changes, labels, onsets = map(np.concatenate, (seeds, scenarios, changes, labels, onsets))
    alarms = {key: np.concatenate([a[key] for a in alarms]) for key in alarms[0]}

    # Step 2: score both labels with identical rules, pooled and per seed.
    rows, seed_rows = [], []
    for (detector, target), first in alarms.items():
        for group, select in [("all", np.ones(len(seeds), bool))] + [(s, seeds == s) for s in EVALUATION_SEEDS]:
            for scenario in ("abrupt", "gradual", "both"):
                in_scenario = select & (np.isin(scenarios, ["abrupt", "gradual"]) if scenario == "both"
                                        else scenarios == scenario)
                variants = {
                    "known_change": score(first, in_scenario, changes, select & (scenarios == "stationary")),
                    # The pilot label pools all scenarios; `scenario` selects positives by true scenario.
                    "pilot_label": score(first, in_scenario & (labels == "narrowed"), onsets,
                                         select & (labels == "stable")),
                }
                for label_name, result in variants.items():
                    row = {"seed": group, "label": label_name, "detector": detector,
                           "target_false_alarm_rate": target, "theta": thresholds[detector, target],
                           "true_scenario": scenario, **result}
                    (rows if group == "all" else seed_rows).append(row)
                # The known-change hits must reproduce the published table.
                if group == "all" and scenario != "both":
                    assert variants["known_change"]["hits"] == published[detector, target, scenario]
    write_csv("alternate_label_results.csv", rows)
    write_csv("alternate_label_results_by_seed.csv", seed_rows)

    # Step 3: how the pilot label classifies each true scenario, and onset versus t_star.
    agreement = []
    for scenario in SCENARIOS:
        chosen = scenarios == scenario
        narrowed = chosen & (labels == "narrowed")
        offset = (onsets - changes)[narrowed] if scenario != "stationary" else np.array([])
        agreement.append({
            "true_scenario": scenario, "histories": int(chosen.sum()),
            **{f"labelled_{name}": int((chosen & (labels == name)).sum()) for name in ("narrowed", "stable", "ambiguous")},
            "median_onset_minus_t_star": float(np.median(offset)) if len(offset) else None,
            "onset_before_t_star": int((offset < 0).sum()) if len(offset) else None,
        })
    write_csv("alternate_label_agreement.csv", agreement)

    # Step 4: print the H1 comparison at every target.
    for target in TARGETS:
        rate = {(r["label"], r["detector"]): r["detection_rate"] for r in rows
                if r["target_false_alarm_rate"] == target and r["true_scenario"] == "both"}
        leads = {label: rate[label, "rolling10"] - rate[label, "cumulative"] for label in ("known_change", "pilot_label")}
        print(f"target {target:.2f}: Rolling-10 lead over cumulative "
              f"{leads['known_change']:+.1%} (known change) vs {leads['pilot_label']:+.1%} (pilot label)")


if __name__ == "__main__":
    main()
