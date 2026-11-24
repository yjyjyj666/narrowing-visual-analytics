# Rules for contributors and coding agents

These rules apply to both team members and to any AI coding agent working in this repository. They were agreed on November 11, 2026.

## Shared definitions

- [`docs/spec.md`](docs/spec.md) defines indexing, measures, synthetic histories, alarms, scoring, seeds, the MF case and data files. Follow it.
- Do not change a definition in code alone. If the spec is wrong or incomplete, propose the change in a pull request that edits `docs/spec.md`.
- Scoring rules are fixed before results are seen. Do not adjust them to improve a result.
- Leave anything marked **Proposed** in the spec unchanged until it is confirmed.

## Environment

- Use Python 3.10 with `requirements.txt`, and install T-RECS separately with `--no-deps`, as described in `README.md`.
- Do not upgrade, add or remove Python dependencies without a pull request that explains why. T-RECS is sensitive to dependency versions.
- Do not modify the upstream T-RECS source. Local behavior changes go in subclasses, as in `src/mf_checked.py`.
- The front-end owner manages front-end dependencies and pins their versions.

## Git

- Never push to `main`. Work on a branch named `feature/<topic>` or `fix/<topic>`.
- Merge through a pull request reviewed by the other team member.
- A pull request that edits `docs/spec.md` must say what changed and mention the other member.
- Do not add AI attribution, such as `Co-Authored-By` trailers or "Generated with" lines, or session links to commits or pull requests.
  - AI assistance is disclosed in `THIRD_PARTY.md` and in the report.

## Verification and reproducibility

- Every module needs a runnable check that compares its output with an independent computation or known cases. Put the check either in a `scripts/check_*.py` script, as with `scripts/check_metrics.py`, or as assertions inside the experiment script. Run the checks in the pinned environment before opening a pull request.
- Set every random seed explicitly, use the seed groups in the spec, and record the seeds next to the results.
- Do not commit large generated files such as `.npz` histories or full JSON exports. Commit the script that generates them, and a small sample if useful.

## Writing

- Code comments, `README.md`, `docs/spec.md` and this file are in English.
- The Chinese audit notes in `docs/` are development records and stay as they are.
- In results and reports:
  - Show the realized false-alarm rate next to the detection rate and delay.
  - Describe MF alarms in the first ten MF steps as timing only, not as caused by the switch to MF.
