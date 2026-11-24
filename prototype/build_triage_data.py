"""Build data/triage_data.js for prototype/triage.html from the exported JSON files.

Reads data/synthetic.json, data/mf.json and data/summary.json (written by
scripts/export_vis_data.py) and stores the series as integers (value x 10,000,
-1 for undefined) to keep the file small. The output is a script that sets
`window.TRIAGE_DATA`, so the page opens straight from disk without a server.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def encode(values):
    return [-1 if v is None else round(v * 10000) for v in values]


def compact(record):
    """The fields the prototype uses; alarms are recomputed in the page."""
    return {"id": record["id"], "sc": record["scenario"][0], "u": record["user"],
            "t": -1 if record["t_star"] is None else record["t_star"], "tp": record["topics"],
            "c": encode(record["series"]["cumulative"]), "r10": encode(record["series"]["rolling10"]),
            "r20": encode(record["series"]["rolling20"])}


def main():
    summary = json.loads((DATA / "summary.json").read_text())
    thetas = {}
    for row in summary:
        thetas.setdefault(row["detector"], {})[f"{row['target_false_alarm_rate']:.2f}"] = row["theta"]
    synthetic = json.loads((DATA / "synthetic.json").read_text())
    mf = json.loads((DATA / "mf.json").read_text())
    payload = {"thetas": thetas,
               "synthetic": {"seed": synthetic[0]["seed"], "records": [compact(r) for r in synthetic]},
               "mf": {"seed": mf[0]["seed"], "records": [compact(r) for r in mf]}}
    out = DATA / "triage_data.js"
    out.write_text("window.TRIAGE_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"Saved {out} ({len(synthetic)} synthetic and {len(mf)} MF records, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
