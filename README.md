# ab-glm-abtest

A compact, production‑style A/B testing example using Binomial GLMs (**logit** and **probit**).
It demonstrates covariate adjustment, **ATE (risk difference)** and **risk ratio** via
**marginal predictions**, cluster‑robust SEs (user‑level), and a Brier score for calibration.

## Why this repo?
- Convert model coefficients into **business units** (absolute/relative lift) correctly.
- Improve **power** with covariate adjustment in randomized experiments.
- Handle **clustered sessions** with cluster‑robust SEs.

## Quick start

```bash
# 1) Install Poetry if needed
# https://python-poetry.org/docs/#installation

# 2) Install dependencies
poetry install

# 3) Run the demo script (simulates data and prints a summary)
poetry run python scripts/run_demo.py
```

Example output (numbers will vary due to simulation):

```
=== Binomial GLM (logit) for A/B test ===
Observations: 13,211    Users (clusters): 4,000
Baseline (control) p̂: 0.1287
Treated p̂:          0.1596
ATE (RD, abs. lift): +0.0309
Risk ratio (RR):     1.2403
Brier score:         0.1049
Treat coef (link):   +0.343  (cluster-robust SE: 0.067)
Note: ATE/RR computed via marginal predictions (covariate-adjusted).
```

## What’s inside
- `src/ab_glm/pipeline.py`: All core logic (simulation, GLM fit, marginal ATE/RR, Brier).
- `scripts/run_demo.py`: Minimal CLI entry to run both logit and probit.
- `tests/test_pipeline.py`: Sanity tests for numerical ranges and utilities.
- `notebooks/`: Placeholder for exploratory work.

## Interpreting results
- **ATE (RD)**: absolute lift in conversion probability (business‑friendly).
- **RR**: treated vs control probability ratio.
- **Cluster‑robust SE**: protects against intra‑user correlation (multiple sessions).
- **Brier**: lower is better (calibration + sharpness).

## Next steps
- Swap the simulator for your real experiment table.
- Add interactions (e.g., `T:country_EU`) for heterogeneity analysis.
- If rare events cause separation, consider penalized/firth logit.

## License
MIT — see `LICENSE`.
