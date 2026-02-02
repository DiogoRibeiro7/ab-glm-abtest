# Notebooks

This directory contains comprehensive Jupyter notebooks demonstrating how to use the `ab-glm-abtest` package for A/B testing analysis.

## Available Notebooks

### 1. Real-World Example (`01_real_world_example.ipynb`)
Complete walkthrough of analyzing an A/B test with:
- Data loading and quality checks
- Exploratory data analysis
- GLM fitting with cluster-robust SEs
- Business metrics calculation (ATE, Risk Ratio)
- Model diagnostics and calibration plots
- Heterogeneous treatment effects
- Business recommendations

**Use this when:** You want to see a complete analysis workflow from start to finish.

### 2. Logit vs Probit Comparison (`02_logit_vs_probit_comparison.ipynb`)
Detailed comparison of logit and probit link functions:
- Visual comparison of link functions
- Side-by-side model fitting
- Coefficient interpretation
- Treatment effect comparison
- Model fit statistics
- Robustness checks across simulations

**Use this when:** You need to understand the differences between link functions or want to run both as a sensitivity analysis.

### 3. Power Analysis & Sample Size (`03_power_analysis_sample_size.ipynb`)
Comprehensive guide to experiment planning:
- Sample size calculation formulas
- Power curves visualization
- Effect of clustering (multiple sessions per user)
- Benefits of covariate adjustment
- Simulation-based power analysis
- Interactive sample size calculator
- Test duration estimation

**Use this when:** You're planning a new experiment and need to determine sample size requirements.

## Getting Started

1. Install the package and dependencies:
```bash
poetry install
```

2. Install additional packages for notebooks:
```bash
poetry add matplotlib seaborn scikit-learn jupyter
```

3. Start Jupyter:
```bash
poetry run jupyter notebook
```

4. Open any notebook and run the cells sequentially.

## Data Requirements

All notebooks can work with either:
- Simulated data (generated automatically)
- Your own data in CSV format with columns:
  - `user_id`: Unique user identifier
  - `T`: Treatment indicator (0=control, 1=treatment)
  - `country_EU`: Binary covariate (0=non-EU, 1=EU)
  - `device_mobile`: Binary covariate (0=desktop, 1=mobile)
  - `prior_views`: Count covariate (prior engagement)
  - `y`: Binary outcome (0=no conversion, 1=conversion)

See `../examples/sample_experiment_data.csv` for an example.

## Tips

- Notebooks are designed to be self-contained and educational
- Each cell includes detailed explanations
- Visualizations help build intuition
- Code can be easily adapted for your specific use case
- Run notebooks in order (01, 02, 03) for the best learning experience