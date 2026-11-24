# Shared specification

This file is the single source of truth for the definitions that code, results and the visualization must share. Elaine and Jing agreed these decisions on November 11, 2026. If code and this file disagree, treat the disagreement as a bug and resolve it in a pull request; do not change a definition silently.

Each section is labeled:

- **Implemented**: agreed and already in code; the named file is the reference.
- **Agreed**: agreed but not yet implemented.
- **Proposed**: not yet agreed; confirm in review before relying on it.

## 1. Indexing and terms (Implemented)

- Every history has 120 steps. Code and data files use **0-based indices** 0–119. The interface and prose may say "step *n*", which always means index *n* − 1.
- Indices 0–19 are the reference period: random startup for MF, unrestricted consumption for synthetic histories.
- `scenario` is one of `stationary`, `abrupt`, `gradual` or `mf`.
- `detector` is one of `cumulative`, `rolling10` or `rolling20`; it names both the diversity series and the alarm built on it.
- `t_star` is the 0-based index of a known change point. It is `null` for `stationary` and `mf` in exported JSON, and −1 in NumPy arrays.

## 2. Diversity measures (Implemented: `src/metrics.py`)

- The diversity of a set of items is the mean pairwise cosine distance of their actual attribute vectors, which have 20 dimensions.
- The value at index *t* uses only items consumed at indices 0..*t*:
  - `cumulative`: all items so far; defined from index 1.
  - `rolling10` / `rolling20`: the last 10 or 20 items; defined from index 9 or 19.
- Undefined values are NaN in arrays and `null` in JSON.
- `diversity_series` expects item vectors shaped (items, dimensions). T-RECS stores actual item attributes as (dimensions, items), so transpose them first.
- **Proposed** (implemented as `metrics.recent_vs_early`): the recent-versus-early ratio at index *t* is the diversity of the last 10 items divided by the mean diversity of 10 items drawn without replacement from indices 0..*t*−10, over 20 draws. It is defined from index 19. It is a diagnostic only; no alarm uses it.

## 3. Synthetic histories (Implemented: `scripts/generate_synthetic_batch.py`)

- Each seed draws its own 2,000 items, 500 users and one `t_star` per user.
  - Item attributes and user preferences are Dirichlet(0.5) over 20 topics.
  - An item's main topic is its largest attribute dimension.
  - `t_star` is drawn uniformly from the integers 40–80.
- At every index, the user sees 10 candidates and consumes the one whose attribute vector has the largest dot product with their preference. No item is consumed twice.
- The **restricted pool** holds the items whose main topic is among the user's top *k* preferred topics. The main experiment uses *k* = 2.
  - `stationary`: candidates come from all unconsumed items.
  - `abrupt`: from `t_star` on, all 10 candidates come from the restricted pool.
  - `gradual`: from `t_star` on, each candidate comes from the restricted pool with probability min(1, (*t* − `t_star`) / 30).
- A user's three histories share preferences, items and the random stream, so they are identical before `t_star`.

## 4. Alarm rule and calibration (Implemented: `src/metrics.py`, `scripts/calibrate_synthetic_alarms.py`)

- The reference value is a detector's value at index 19.
- An alarm fires when the series stays strictly below (1 − θ) × reference for three consecutive indices, counting from index 20. The alarm index is the third index of that run.
- Only the first alarm of each history is used.
- The calibration targets are α ∈ {0.05, 0.10, 0.20}.
- For each detector and α, calibration chooses the smallest θ on a 0.001 grid whose false-alarm rate on calibration-seed `stationary` histories does not exceed α.
- The same θ applies to `abrupt` and `gradual`.

## 5. Scoring and reported quantities (Implemented: `scripts/calibrate_synthetic_alarms.py`)

The definitions below must not change after results are seen.

- **False-alarm rate**: the share of `stationary` users with any alarm at indices 20–119, counted per user.
  - Always report the realized rate on evaluation seeds next to the target α. It can differ from α and between detectors.
- **Hit**: for `abrupt` and `gradual`, the first alarm falls in [`t_star`, `t_star` + 39].
- **Early alarm**: the first alarm falls before `t_star`. It counts as a false alarm, and the user also counts as a miss.
- **Miss**: every user without a hit, including early and late alarms and no alarm.
- **Delay**: first alarm − `t_star`, averaged over hits only. Report it together with the detection rate, because a detector with fewer hits averages over a different set of users.
- Results table columns, as written to `results/synthetic/synthetic_alarm_results.csv`:
  `detector, target_false_alarm_rate, theta, calibration_false_alarm_rate, evaluation_false_alarm_rate, scenario, users, hits, early_alarms, misses, mean_delay_steps_among_hits`.
  The per-seed table adds a leading `seed` column.
- **Proposed** (implemented in `scripts/evaluate_alternate_labels.py`): the H1 check rescores the same evaluation histories with the pilot's label instead of `t_star`, using the rules above unchanged.
  - The base is the diversity at indices 0–19, and s(*t*) is the diversity of the centred window *t*−5..*t*+4.
  - *Narrowed*: s(*t*) < 0.7 × base for 10 consecutive indices from 20 on; the onset is the first of them and replaces `t_star`.
  - *Stable*: s(*t*) never stays below 0.85 × base for 10 consecutive indices. Stable histories of any scenario are the negatives.
  - Otherwise *ambiguous*, and excluded.

## 6. Seeds

| Purpose | Seeds | Status |
|---|---|---|
| Threshold calibration (`stationary` only) | 0–4 | Implemented |
| Evaluation | 5–9 | Implemented; used only for final evaluation |
| *k* pilot and development examples | 100 and above, never 0–9 | Agreed |
| MF application case | 123 | Proposed; one seed, because MF is a qualitative application (§7). 123–127 were used for the superseded original configuration |
| MF candidate development | 128 and 133 (simulation), 130 (quality cohort), 134 (arrival stream) | Used to choose the candidate configuration in §7; never used for MF results |

Report the seeds with every result.

**Open record**: the main experiment uses *k* = 2. The *k* = 1 versus *k* = 2 pilot that selected it, with its seeds and comparison, still needs to be recorded in `docs/synthetic_experiment.md`.

## 7. MF application case (Implemented: `scripts/run_mf_experiment.py`; original configuration in `scripts/run_mf_case.py`)

- **Shared settings**:
  - Preferences fixed (drift = 0).
  - Scale: 500 users, 2,000 items and 20 actual attribute dimensions.
  - Steps: 20 random startup steps, then 100 MF steps, with cumulative interactions retained, a refit after every step and no repeated consumption.
  - ALS: 20 iterations, `weight` 40 and `reg` 0.1 (the LensKit defaults).
- **Proposed: candidate configuration**, used by the current visualization data. See [docs/mf_pipeline_experiment.md](mf_pipeline_experiment.md).
  - Model: `InterleavedImplicitMF` from `src/mf_checked.py`, with 20 latent factors.
  - Each MF slate holds 8 ranked MF items and 2 items drawn uniformly from the other eligible items.
- **Original configuration**: `CheckedImplicitMF` with 10 latent factors and slates of 10 ranked items.
  - Its late top-ten ranking fell below a Random slate, so it did not act as a working recommender in the second half of the run.
  - Its histories are no longer used for MF results. They are kept only as a comparison.
- **Two references**, both reported:
  - `startup`: the value at index 19; alarms are counted from index 20.
  - `early_mf`: the value at index 29; alarms are counted from index 30.
- Under the `startup` reference, alarms at indices 20–29 are reported as **alarms in the first ten MF steps**.
  - This is a description of timing only. Do not say the switch to MF caused them.
- The `early_mf` reference for `cumulative` still includes startup items, and so does the Rolling-20 window at index 29 (indices 10–29). State this limitation when reporting it.
- MF has no known `t_star`, so no hit rate is computed.
- **Proposed: one seed.** MF is a qualitative application. It shows whether behaviors seen in the synthetic experiment appear, and whether new forms of narrowing appear. It does not estimate stable rates, so it uses one seed (123). Report MF percentages as describing that seed only.
- **Proposed**: θ for each detector and α is taken from the synthetic calibration in §4, not recalibrated on MF histories.
- **Proposed: twenty-step narrowing diagnostic.** Each item's main topic is its largest actual attribute. A user counts as narrowing if, at some point in the 100 MF steps, all items consumed in 20 consecutive steps have main topics within one fixed set of one or two topics. The first such window gives an observed start, and its 20th step is the confirmation. This is a descriptive diagnostic, not ground truth, and it is reported separately for the user's own two favorite topics.

## 8. Data files for the visualization (Implemented: `scripts/export_vis_data.py`, checked by `scripts/check_export.py`)

Generated files go in `data/` and are not committed; commit only the export script and a small sample.

- **`summary.json`**: an array with one object per row of the synthetic results table in §5, using the same field names. Missing delays are `null`.
- **`synthetic.json`**: users from one evaluation seed, all three scenarios.
- **`mf.json`**: users from one MF seed, simulated with the candidate configuration in §7.

Both per-user files are arrays of records. In this example, the strings in arrays stand for omitted values:

```json
{
  "id": "s5-u0042-abrupt",
  "scenario": "abrupt",
  "seed": 5,
  "user": 42,
  "t_star": 63,
  "favorite_topics": [3, 11],
  "topics": [7, 3, "... 120 integers: main topic of the item consumed at each index"],
  "series": {
    "cumulative": [null, 0.81, "... 120 numbers or null"],
    "rolling10": ["..."],
    "rolling20": ["..."]
  },
  "reference": { "startup": { "cumulative": 0.80, "rolling10": 0.83, "rolling20": 0.80 } },
  "alarms": { "startup": { "rolling10": { "0.05": 66, "0.10": 65, "0.20": null } } }
}
```

- `id` is unique across both files.
  - Format: `s<seed>-u<user, 4 digits>-<scenario>`.
  - The MF format, `mf<seed>-u<user, 4 digits>` such as `mf123-u0042`, is Proposed.
- `alarms` and `reference` are keyed by reference name. Synthetic records have only `startup`; MF records also have `early_mf`.
- Alarm values are 0-based indices, or `null` for no alarm. `metrics.first_alarms` returns −1 for no alarm; the export converts it.
- α keys are strings with two decimals: `"0.05"`, `"0.10"`, `"0.20"`.
- `favorite_topics` is present only for synthetic records.
- Series values are rounded to 4 decimals. Alarms are computed before rounding, so an alarm recomputed from exported values can differ when a value lies within rounding error of the limit.
- **Proposed**: `series.recent_vs_early` holds the recent-versus-early ratio from §2. The export includes it; readers must still tolerate its absence.

## 9. Changing this file

- Change definitions only through a pull request, and state in its description what changed and why.
- Ask the other member to review.
- Rerun any affected results before merging.
