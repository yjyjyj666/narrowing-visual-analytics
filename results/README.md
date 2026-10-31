# Included audit results

These compact files record local checks performed on October 31, 2026:

- `readiness_verified.json`: two original MF runs and one Random run; ALS 5 iterations, fixed preferences. The original readiness script was equivalent to `scripts/check_readiness.py`, with different output paths.
- `positive_control_verified.json`: a deliberately simple four-group sanity check. This is not a project finding; its separate development script is not included in this first upload.
- `training_depth_verified.json`: refits the same recorded histories at 5, 20 and 40 ALS iterations. The fixed-history refit script is not included in this first upload; retain this result as audit documentation, not a promise of one-command reproduction.

Behavior audit details are in `../docs/model_audit.md`. These results are limited checks, not a completed narrowing study. The 20-iteration behavior runs can be regenerated with `scripts/audit_behavior.py`. Large history arrays are deliberately excluded from this upload.
