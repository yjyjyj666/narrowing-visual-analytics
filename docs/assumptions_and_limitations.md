# Assumptions and limitations

This file records the assumptions behind our experiments and what each one limits. The report's limitations section should draw on it. Add an entry whenever a new setting or design choice is made, and update the status when an entry is addressed.

Each entry gives the assumption, why we made it, what it limits, the evidence, and a status:

- **Open**: the limitation could be reduced with further work.
- **Disclosed**: we accept the limitation and state it in results and the report.
- **Addressed**: further work has resolved or measured it; the entry says how.

Exact definitions live in [docs/spec.md](spec.md); this file explains their consequences.

## How the parts fit

The synthetic experiment is a **controlled experiment**. Under known types and timing of change, it measures how each detector responds: how often, how early, and how many false alarms it raises. The MF case, and any later real data, is an **application**. It has no ground truth, so it can only show whether the same behaviors appear, and help us find changes the synthetic scenarios do not cover.

## A. What counts as narrowing

**A1. Narrowing means the candidate pool shrinks to the user's favorite topics.**

- Assumption: in the synthetic histories, narrowing happens when candidates are restricted to items whose main topic is among the user's top *k* = 2 topics. User preferences never change.
- Why: this produces a known change point `t_star` and a known amount of change.
- Limits: narrowing caused by changing preferences, or by restrictions that are not topic-based, is not modeled. Results describe detection of supply-side narrowing only.
- Status: Disclosed.

**A2. Three scenarios represent the forms narrowing takes.**

- Assumption: a history is stationary, narrows abruptly, or narrows gradually over 30 steps, with one change point between indices 40 and 80.
- Why: the two forms test whether detectors differ by speed of change (RQ2).
- Limits: the generator cannot produce narrowing that later recovers, repeated or oscillating changes, or partial narrowing that stops before the restricted pool is reached. Conclusions cover only these forms.
- Evidence: the scenarios are defined in spec §3.
- Status: Open. The candidate MF histories already show one form outside these three: narrowing episodes that end before the run does (E5). The linked tool should be used on MF histories to look for others. Each form found becomes a candidate scenario or is reported as a limit.

**A3. The ground truth depends on how narrowing is labelled.**

- Assumption: the main results score detectors against the generator's `t_star`.
- Why: it is the only label that does not depend on the consumed items, and therefore on the detectors' own signal.
- Limits: a different label changes the conclusions. With the pilot's label, Rolling-10's lead over Cumulative falls from 42.5 to 4.4 percentage points at the 10% target. The two labels differ in three ways at once: positive set, negative set and onset. Which of these drives the change is not yet separated.
- Evidence: [MF case, H1 label check and equal-sample diagnostic](mf_h1_diagnostic_findings.md), §2.
- Status: Open, pending a decomposition that holds two of the three differences fixed at a time.

## B. Measures, alarms and scoring

**B1. Diversity is the mean pairwise cosine distance of actual item attributes.**

- Assumption: each item has a 20-dimensional actual attribute vector, drawn from Dirichlet(0.5) over topics, and diversity is computed on these vectors.
- Why: it is the measure used in the pilot, and it depends only on actual item attributes, not on any model.
- Limits: other diversity notions are not evaluated, such as topic entropy, catalog coverage or diversity based on latent factors. Conclusions are about cumulative, Rolling-10 and Rolling-20 windows of this one measure.
- Status: Disclosed. Topic entropy is deferred in the project plan.

**B2. One alarm rule for all detectors.**

- Assumption: an alarm fires when a series stays below (1 − θ) × its value at index 19 for 3 consecutive indices, and only the first alarm counts.
- Why: a single simple rule keeps the comparison about the measures, not the detection method.
- Limits: other change detectors, such as CUSUM, are not evaluated. Detection rates and delays are specific to this rule.
- Status: Disclosed. CUSUM is deferred in the project plan.

**B3. Thresholds are calibrated on synthetic stationary histories.**

- Assumption: θ is the smallest value on a 0.001 grid whose false-alarm rate on seeds 0–4 stays at or below the target.
- Why: it makes detectors comparable at the same false-alarm rate.
- Limits: realized rates on evaluation seeds differ from the target (8.1% to 10.5% at the 10% target), so they are always reported. The calibration holds only for histories whose diversity behaves like the synthetic stationary ones.
- Status: Disclosed.

**B4. Fixed scoring rules.**

- Assumption: a hit is a first alarm within [`t_star`, `t_star` + 39]. An earlier first alarm counts as a false alarm and a miss. Delay is averaged over hits only.
- Why: the rules were fixed before results were seen.
- Limits: a different hit window changes detection rates and delays. In particular, slower detectors would benefit from a longer window, and faster detectors from tolerance before `t_star`.
- Status: Disclosed.

## C. Synthetic generator settings

**C1. Generator parameters shape the numbers.**

- Assumption, per seed:
  - 2,000 items and 500 users, with preferences and attributes drawn from Dirichlet(0.5) over 20 topics;
  - 10 candidates per step;
  - the user consumes the candidate with the largest dot product with their preference;
  - no item is consumed twice.
- Why: these settings match the MF simulation, so the two parts are comparable.
- Limits: detection rates and delays depend on these parameters, as well as on *k* and the 30-step ramp. Real choice is noisier than an argmax.
- Status: Disclosed. See D1 for what transfers.

**C2. The choice of *k* = 2 is not documented.**

- Assumption: *k* = 2 was chosen from a pilot comparing *k* = 1 and *k* = 2 on seeds outside 0–9.
- Limits: until the pilot's seeds and comparison are recorded, readers cannot check that *k* was not tuned on evaluation data.
- Status: Open. Record the pilot in `docs/synthetic_experiment.md`.

## D. Transfer of conclusions

**D1. Only qualitative regularities transfer, not numbers.**

- Assumption: conclusions are drawn from synthetic histories with known change points.
- Limits: if real narrowing looks like our scenarios, the following can be expected to carry over:
  - the ordering of detectors;
  - the speed difference between abrupt and gradual narrowing;
  - the mechanisms, such as Cumulative's inertia from overlapping with the reference items.

  Specific detection rates and delays will not carry over, because they depend on the settings in C1. Results and the report should state conclusions as "under this kind of change, detector A responds earlier than B", not as accuracy figures for real systems.
- Status: Disclosed.

**D2. Data without ground truth cannot measure accuracy.**

- Assumption: neither MF histories nor real logs say which users narrowed, or from when.
- Limits: on such data we can check only three things: whether the synthetic behaviors appear, whether alarms can be explained from the consumption history, and whether patterns outside A2 appear. An LLM or human label would be another label definition (A3), not ground truth. At best, it could be checked on synthetic data first and used as a sensitivity analysis.
- Status: Disclosed.

## E. MF application case

**E1. The MF simulation stands in for a real recommender.**

- Assumption: T-RECS MF in the candidate configuration (spec §7, Proposed), with:
  - fixed preferences;
  - 20 random startup steps, then 100 MF steps;
  - 20 latent factors and 20 ALS iterations, with a refit after every step;
  - slates of 8 ranked MF items plus 2 items drawn uniformly from the other eligible items;
  - users who choose the slate item with the highest inner product with their actual preferences (T-RECS default, attention exponent 0).
- Why: T-RECS is the framework this project builds on. The original configuration (10 factors, 10 ranked items) ranked items worse than Random late in the run, so it did not stand in for a working recommender. The candidate ranks above Random at every MF step in the development seeds and in seed 123.
- Limits:
  - Simulated users are not real users.
  - The two random slots add diversity directly. The candidate's diversity is likely higher, and its alarms fewer, than a purely ranked slate would give. State this whenever MF alarms are reported.
  - Recommendation quality was checked on a cohort of ten users only.
  - Untrained items score zero, which affects which items are recommended early on.
  - MF uses one seed (123), so its percentages are not stable rates. Only qualitative patterns are reported. The episode pattern also appeared in development seeds 128 and 133.
- Evidence: [docs/mf_pipeline_experiment.md](mf_pipeline_experiment.md); [MF case](mf_h1_diagnostic_findings.md), §1.1–§1.2; `docs/model_audit.md` for the original configuration.
- Status: Open until both members confirm the candidate configuration.

**E2. MF thresholds come from the synthetic calibration.**

- Assumption: Proposed in spec §7, and used by the current results.
- Why: MF has no stationary histories to calibrate on.
- Limits: MF diversity is distributed differently from synthetic diversity, so MF alarm rates are not false-alarm rates and cannot be called high or low on their own. Without a control, an alarm cannot be attributed to personalization rather than to that difference.
- Status: Open. `scripts/run_mf_experiment.py` already runs a Random policy on the same seeds. Its alarm rates have not yet been computed for comparison.

**E3. The two references each have a known bias.**

- Assumption: `startup` uses index 19, the last random step. `early_mf` uses index 29, the tenth MF step.
- Limits:
  - Under `startup`, many first alarms fall in the first ten MF steps. This is reported as timing only and not attributed to the switch to MF.
  - Under `early_mf`, the reference values of Cumulative and Rolling-20 still include startup items, which keeps them high.
- Status: Disclosed, and stated wherever these results appear.

**E4. MF results depend on settings still marked Proposed.**

- Assumption: these Proposed settings are used:
  - the candidate MF configuration and the twenty-step narrowing diagnostic;
  - MF seed 123, used alone;
  - the MF thresholds in E2;
  - the MF record id format.
- Limits: the results are provisional until both members confirm these settings.
- Status: Open.

**E5. Narrowing episodes in MF end because the narrowed topics run out of items.**

- Assumption (working hypothesis): with the candidate configuration, 140 of 500 users in seed 123 spend 20 consecutive MF steps within one or two topics, but none is still narrowed in the final 20 steps. The episodes are taken to be real narrowing that ends when those topics run out of items the user would choose. Nothing is consumed twice and the catalog is fixed at 2,000 items, so such items deplete.
- Why: it explains why narrowing appears and then fades, without assuming preferences change.
- Limits:
  - The recovery may be an artifact of the fixed catalog, not something a real recommender with fresh items would show. MF results should then describe episodes, not lasting narrowing.
  - The development evidence is mixed. At most first departures, a higher-utility item from the same topic group was still in the pool but not on the slate. With new items arriving each step, only 1 of 500 users stayed narrowed to the end, but new items could appear only in the random slots.
- Evidence: [MF case](mf_h1_diagnostic_findings.md), §1.2; [docs/mf_pipeline_experiment.md](mf_pipeline_experiment.md).
- Status: Open, pending a test that separates depletion from what the slate shows.

## F. Equal-sample diagnostic

**F1. The recent-versus-early ratio uses a fixed sample size.**

- Assumption: the ratio compares 10 recent items with 10 items drawn from earlier history, averaged over 20 draws. This definition is Proposed in spec §2.
- Limits:
  - The planned reading assumed Cumulative's spread would shrink over time; it does not. Cumulative stays steady because its values overlap with the reference items, not because its sample grows.
  - The diagnostic was run on stationary histories only.
- Evidence: [MF case, H1 label check and equal-sample diagnostic](mf_h1_diagnostic_findings.md), §3.
- Status: Disclosed. The definition still needs confirmation.
