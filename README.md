# When Does Narrowing Become Visible?

Team 14 — Elaine Zhou and Jing Yu  
Visualization for Machine Learning, Fall 2026

We study how cumulative and recent consumption-diversity measures give different judgments about user histories. Our planned tool links an overview, a user-by-time matrix and individual consumption sequences. Controlled synthetic sequences will support validation; a T-RECS matrix-factorization simulation will provide an application case.

## Current status

Simulator readiness and initial model-behavior checks are complete for the tested configurations. The detector experiments and visualization interface are not yet complete. This repository currently contains the local MF adapter, reproducible checking scripts and compact audit results.

## Original research and code

We build on **T-RECS: A Simulation Tool to Study the Societal Impact of Recommender Systems**, Eli Lucherini, Matthew Sun, Amy Winecoff and Arvind Narayanan, 2021.

- Paper: https://arxiv.org/abs/2107.08959
- Original framework: https://github.com/elucherini/t-recs
- Original experiment scripts: https://github.com/sunnymatt/t-recs-experiments
- Framework revision used: `0de83a3b5039d768c86bc5f3a0cf58b2d4072920`

The upstream framework is installed as a dependency rather than copied into this repository. We do not claim authorship of T-RECS. See [THIRD_PARTY.md](THIRD_PARTY.md) and the included upstream MIT license.

## Local changes

[`src/mf_checked.py`](src/mf_checked.py) subclasses the original MF model; the upstream source is unchanged.

1. It writes learned factors back using the explicit user and item identifiers returned by LensKit. Index alignment passed in the audited upstream runs; preserving the returned order makes the assumption explicit.
2. It assigns zero latent vectors and therefore zero prediction scores to items with no training interactions, rather than retaining arbitrary random scores for them. This is a disclosed cold-start policy, not a claim that those items have zero actual value.
3. It checks that the resulting prediction scores are finite.

The simulation scripts explicitly retain cumulative interactions across calls. Behavior checks also compare ALS iteration counts; 20 iterations are the current application-case starting point, while the readiness-only script uses 5 for its execution check. None of these changes uses actual preference information to train MF.

## Environment and installation

The existing environment was tested on macOS with Python **3.10.4**, NumPy 1.23.5, pandas 1.5.3, LensKit 0.14.4 and numba 0.57.1. Do not upgrade these dependencies without testing. `requirements.txt` is a snapshot of the working environment; clean installation on another machine has not yet been verified.

Using Python 3.10 and Git:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps "git+https://github.com/elucherini/t-recs.git@0de83a3b5039d768c86bc5f3a0cf58b2d4072920"
```

The separate `--no-deps` install intentionally avoids T-RECS's historical pandas pin. It does not modify the model algorithm. Windows activation commands differ.

## Run checks

From the repository root, with the environment active:

```bash
python scripts/check_adapter.py
python scripts/check_readiness.py
python scripts/audit_behavior.py --iterations 20 --corrected-only
```

The first command is a small smoke test. The readiness script runs two original MF configurations and one Random reference. The final command uses the local adapter and runs two updated models and a frozen-model reference. To also run original-versus-adapted comparisons, omit `--corrected-only`.

Runs write JSON summaries and compressed synthetic histories under `results/`. Large arrays are ignored by Git. Runtime depends on machine and settings; the complete comparison takes several minutes, not just the smoke-test duration. A NUMBA or Matplotlib cache warning can be addressed by setting their cache directories to writable local locations.

## Checks and limitations

- Tested runs used 500 users, 2,000 items, 20 random startup steps and 100 deployment steps, with fixed preferences and no repeated consumption.
- Logs were checked for retained interactions, valid choices and factor index alignment.
- The original MF could recommend untrained items almost exclusively in its first deployment step. The explicit zero-score policy removed this effect in the tested cases.
- The model showed personalization, but later preference matching remained limited. This is an analysis case, not evidence that MF always outperforms Random or causes filter bubbles.
- No broad multi-seed inference, drift grid, full-paper reproduction or visualization evaluation is claimed.

See [docs/model_audit.md](docs/model_audit.md) for the audit details and [results/README.md](results/README.md) for the provenance of the included result files.
