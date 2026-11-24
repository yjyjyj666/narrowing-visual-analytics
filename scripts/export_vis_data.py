"""Export the JSON files the linked visualization reads (docs/spec.md §8).

Writes data/summary.json, data/synthetic.json (one evaluation seed) and, when the
MF histories exist, data/mf.json. --mf-dir picks the MF run: results/mf/ (from
scripts/run_mf_case.py) by default, or e.g. results/mf_candidate/candidate/ from
scripts/run_mf_experiment.py (docs/mf_pipeline_experiment.md). A small sample of
each file goes to data/sample/, which is the only part committed to Git.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from alarms import DETECTORS, REFERENCES, TARGETS, alarm_table, load_thresholds  # noqa: E402
from metrics import diversity_series, recent_vs_early  # noqa: E402
from generate_synthetic_batch import SCENARIOS, STEPS, check_batch, generate_batch  # noqa: E402

RESULTS_CSV = REPO_ROOT / "results" / "synthetic" / "synthetic_alarm_results.csv"
MF_DIR = REPO_ROOT / "results" / "mf"
INTEGER_COLUMNS = {"users", "hits", "early_alarms", "misses"}
DECIMALS = 4


def number(value):
    """A JSON number rounded to DECIMALS places, or None for NaN."""
    return None if math.isnan(value) else round(float(value), DECIMALS)


def summary_records():
    """One object per row of the synthetic results table, with typed values."""
    records = []
    with RESULTS_CSV.open(newline="") as file:
        for row in csv.DictReader(file):
            record = {}
            for key, value in row.items():
                if key in ("detector", "scenario"):
                    record[key] = value
                elif key in INTEGER_COLUMNS:
                    record[key] = int(value)
                else:
                    # A detector with no hits has an empty mean delay.
                    record[key] = float(value) if value else None
            records.append(record)
    return records


def history_records(ids, base, histories, topics, series, alarms):
    """Per-history records; `base` holds the fields that differ by data source.

    alarms: {reference: {(detector, target): first-alarm indices}}.
    """
    records = []
    for row, record_id in enumerate(ids):
        record = {"id": record_id, **base[row]}
        record["topics"] = topics[histories[row]].tolist()
        record["series"] = {name: [number(v) for v in values[row]] for name, values in series.items()}
        record["reference"] = {
            ref: {d: number(series[d][row, REFERENCES[ref][0]]) for d in DETECTORS}
            for ref in alarms
        }
        # Convert the -1 "no alarm" value of metrics.first_alarms to null.
        record["alarms"] = {
            ref: {d: {f"{t:.2f}": (int(table[d, t][row]) if table[d, t][row] >= 0 else None)
                      for t in TARGETS} for d in DETECTORS}
            for ref, table in alarms.items()
        }
        records.append(record)
    return records


def synthetic_records(seed, thresholds):
    """All users of one synthetic seed, three scenarios each (spec §3)."""
    # Step 1: regenerate the seed exactly as the calibration script does.
    items, histories, changes, favorites = generate_batch(seed, 500, 2, 40, 80)
    check_batch(items, histories, changes, favorites)
    users = len(histories)
    flat = histories.reshape(-1, STEPS)
    # Step 2: series and alarms, using the calibrated thresholds and the startup reference.
    all_series = diversity_series(items, flat)
    series = {name: all_series[name] for name in DETECTORS}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    series["recent_vs_early"] = recent_vs_early(items, flat, rng)
    alarms = {"startup": alarm_table(series, thresholds, "startup")}
    # Step 3: rows of `flat` run user-major: user 0 stationary, abrupt, gradual, user 1, ...
    ids, base = [], []
    for user in range(users):
        for scenario in SCENARIOS:
            ids.append(f"s{seed}-u{user:04d}-{scenario}")
            base.append({
                "scenario": scenario, "seed": seed, "user": user,
                "t_star": None if scenario == "stationary" else int(changes[user]),
                "favorite_topics": sorted(int(t) for t in favorites[user]),
            })
    return history_records(ids, base, flat, items.argmax(axis=1), series, alarms)


def mf_records(mf_dir, seed, thresholds):
    """All users of one MF seed, with both references (spec §7)."""
    data = np.load(mf_dir / f"mf_seed{seed}.npz")
    # T-RECS stores actual item attributes as (dimensions, items).
    items, histories = data["items"].T, data["histories"]
    all_series = diversity_series(items, histories)
    series = {name: all_series[name] for name in DETECTORS}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    series["recent_vs_early"] = recent_vs_early(items, histories, rng)
    alarms = {ref: alarm_table(series, thresholds, ref) for ref in REFERENCES}
    ids = [f"mf{seed}-u{user:04d}" for user in range(len(histories))]
    base = [{"scenario": "mf", "seed": seed, "user": user, "t_star": None}
            for user in range(len(histories))]
    return history_records(ids, base, histories, items.argmax(axis=1), series, alarms)


def write_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False guarantees that no NaN slipped past the null conversion.
    text = json.dumps(records, separators=(",", ":"), allow_nan=False)
    path.write_text(text)
    print(f"Saved {path} ({len(records)} records, {len(text) / 1e6:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-seed", type=int, default=5, help="An evaluation seed, 5-9")
    parser.add_argument("--mf-seed", type=int, default=123)
    parser.add_argument("--mf-dir", type=Path, default=MF_DIR, help="Directory holding mf_seed<seed>.npz")
    parser.add_argument("--sample-users", type=int, default=5)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()
    if args.synthetic_seed not in range(5, 10):
        parser.error("export an evaluation seed (5-9), never a calibration seed")

    thresholds = load_thresholds(RESULTS_CSV)
    files = {"summary.json": summary_records(),
             "synthetic.json": synthetic_records(args.synthetic_seed, thresholds)}
    if (args.mf_dir / f"mf_seed{args.mf_seed}.npz").exists():
        files["mf.json"] = mf_records(args.mf_dir, args.mf_seed, thresholds)
    else:
        print(f"Skipped mf.json: no mf_seed{args.mf_seed}.npz in {args.mf_dir}")

    for name, records in files.items():
        write_json(args.output / name, records)
        # The sample keeps the first few users; summary.json is small, so keep all of it.
        keep = records if name == "summary.json" else [r for r in records if r["user"] < args.sample_users]
        write_json(args.output / "sample" / name, keep)


if __name__ == "__main__":
    main()
