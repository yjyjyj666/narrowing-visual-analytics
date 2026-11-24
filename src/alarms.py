"""Calibrated thresholds and alarm references shared by the MF case, the H1 check and the export.

Definitions follow docs/spec.md: the thresholds come from the synthetic calibration
(§4), and each reference names the index whose value is the baseline and the first
index at which alarms count (§4 for synthetic histories, §7 for MF).
"""
import csv

from metrics import first_alarms

DETECTORS = ("cumulative", "rolling10", "rolling20")
TARGETS = (0.05, 0.10, 0.20)
# Reference name -> (reference index, first index at which an alarm can fire).
REFERENCES = {"startup": (19, 20), "early_mf": (29, 30)}


def load_thresholds(path):
    """Return {(detector, target): theta} from the synthetic results table."""
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    thresholds = {}
    for row in rows:
        key = (row["detector"], float(row["target_false_alarm_rate"]))
        theta = float(row["theta"])
        # The table repeats each (detector, target) once per narrowing scenario.
        if thresholds.setdefault(key, theta) != theta:
            raise ValueError(f"conflicting thresholds for {key}")
    missing = {(d, t) for d in DETECTORS for t in TARGETS} - thresholds.keys()
    if missing:
        raise ValueError(f"thresholds missing for {sorted(missing)}")
    return thresholds


def alarm_table(series, thresholds, reference):
    """First alarm index of every history for each (detector, target); -1 means none.

    series: {detector: (histories, steps)} as returned by metrics.diversity_series.
    """
    reference_index, start = REFERENCES[reference]
    return {
        (detector, target): first_alarms(
            series[detector], series[detector][:, reference_index],
            thresholds[detector, target], start)
        for detector in DETECTORS for target in TARGETS
    }
