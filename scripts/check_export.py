"""Check exported visualization files against docs/spec.md §8 and the results tables.

Runs on data/ by default; pass data/sample to check the committed sample. Alarms are
recomputed from each record's own series with a plain loop, and the synthetic
seed's hits and false alarms are compared with the per-seed results table.
"""
import argparse
import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BY_SEED_CSV = REPO_ROOT / "results" / "synthetic" / "synthetic_alarm_results_by_seed.csv"
DETECTORS = ("cumulative", "rolling10", "rolling20")
TARGET_KEYS = ("0.05", "0.10", "0.20")
REFERENCES = {"startup": (19, 20), "early_mf": (29, 30)}
STEPS = 120
# First defined index of each series (spec §2 and the recent-versus-early ratio).
FIRST_DEFINED = {"cumulative": 1, "rolling10": 9, "rolling20": 19, "recent_vs_early": 19}


def thresholds():
    """theta per (detector, target key), read from the per-seed results table."""
    with BY_SEED_CSV.open(newline="") as file:
        return {(r["detector"], f"{float(r['target_false_alarm_rate']):.2f}"): float(r["theta"])
                for r in csv.DictReader(file)}


def first_run_end(below, start):
    """Plain-loop alarm rule: the third index of the first run of three True values."""
    run = 0
    for t in range(start, STEPS):
        run = run + 1 if below[t] else 0
        if run == 3:
            return t
    return None


def alarm_matches(exported, values, reference, start, theta):
    """The exported alarm is the one the alarm rule gives on the exported series.

    Alarms are computed before values are rounded to 4 decimals, so a value within
    rounding error of the limit may sit on either side of it. Such values count as
    "maybe below"; the exported alarm must be consistent with some choice for them.
    """
    error = 1e-4  # rounding error of the value plus that of the reference
    limit = (1 - theta) * reference
    surely = [v is not None and v < limit - error for v in values]
    maybe = [v is not None and v < limit + error for v in values]
    # No run of three sure-below values may end before the exported alarm.
    earliest = first_run_end(surely, start)
    if exported is None:
        return earliest is None
    if earliest is not None and earliest < exported:
        return False
    # The run ending at the alarm: three maybe-below values, and the value before
    # them maybe not below, or else the run would have completed one index earlier.
    run_start = exported - 2
    return (run_start >= start and all(maybe[run_start:exported + 1])
            and (run_start == start or not surely[run_start - 1]))


def check_record(record, theta, source):
    # Step 1: fields and shapes.
    assert record["scenario"] in (("stationary", "abrupt", "gradual") if source == "synthetic" else ("mf",))
    assert len(record["topics"]) == STEPS and all(0 <= t < 20 for t in record["topics"])
    assert ("favorite_topics" in record) == (source == "synthetic")
    assert (record["t_star"] is None) == (record["scenario"] in ("stationary", "mf"))
    references = ("startup",) if source == "synthetic" else ("startup", "early_mf")
    assert tuple(record["alarms"]) == references and tuple(record["reference"]) == references
    # Step 2: null exactly where a series is undefined.
    for name, first in FIRST_DEFINED.items():
        values = record["series"][name]
        assert len(values) == STEPS
        assert all(v is None for v in values[:first]) and all(v is not None for v in values[first:]), name
    # Step 3: references and alarms follow from the record's own series.
    for ref in references:
        index, start = REFERENCES[ref]
        for detector in DETECTORS:
            values = record["series"][detector]
            assert record["reference"][ref][detector] == values[index]
            for key in TARGET_KEYS:
                exported = record["alarms"][ref][detector][key]
                assert alarm_matches(exported, values, values[index], start, theta[detector, key]), \
                    (record["id"], ref, detector, key)


def check_against_table(records, theta):
    """Hits, early alarms and false alarms of a full synthetic seed match the per-seed table."""
    seed = records[0]["seed"]
    with BY_SEED_CSV.open(newline="") as file:
        table = [r for r in csv.DictReader(file) if int(r["seed"]) == seed]
    by_user = {}
    for r in records:
        by_user.setdefault(r["scenario"], []).append(r)
    for row in table:
        key = f"{float(row['target_false_alarm_rate']):.2f}"
        alarms = lambda scenario: [r["alarms"]["startup"][row["detector"]][key] for r in by_user[scenario]]  # noqa: E731
        false_rate = sum(a is not None for a in alarms("stationary")) / len(by_user["stationary"])
        assert math.isclose(false_rate, float(row["evaluation_false_alarm_rate"])), row
        firsts = alarms(row["scenario"])
        changes = [r["t_star"] for r in by_user[row["scenario"]]]
        hits = sum(a is not None and c <= a <= c + 39 for a, c in zip(firsts, changes))
        early = sum(a is not None and a < c for a, c in zip(firsts, changes))
        assert hits == int(row["hits"]) and early == int(row["early_alarms"]), row
    return len(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()
    theta = thresholds()

    summary = json.loads((args.folder / "summary.json").read_text())
    with (REPO_ROOT / "results" / "synthetic" / "synthetic_alarm_results.csv").open(newline="") as file:
        assert len(summary) == sum(1 for _ in csv.DictReader(file))
    print(f"summary.json: {len(summary)} rows match the results table")

    ids = set()
    for source in ("synthetic", "mf"):
        path = args.folder / f"{source}.json"
        if not path.exists():
            print(f"{path.name}: not present, skipped")
            continue
        records = json.loads(path.read_text())
        for record in records:
            check_record(record, theta, source)
            assert record["id"] not in ids, record["id"]
            ids.add(record["id"])
        message = f"{path.name}: {len(records)} records pass schema and alarm checks"
        if source == "synthetic" and len(records) == 1500:
            message += f"; {check_against_table(records, theta)} per-seed table rows reproduced"
        print(message)
    print("PASS")


if __name__ == "__main__":
    main()
