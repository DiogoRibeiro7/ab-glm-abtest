# Examples

This directory contains example data and scripts for using the `ab-glm-abtest` package.

## Files

### `sample_experiment_data.csv`
Sample A/B test dataset with 20 users and 42 sessions demonstrating the expected data format:
- Multiple sessions per user (clustered data)
- Binary treatment assignment at user level
- Pre-treatment covariates (country, device, prior engagement)
- Binary outcome (conversion)

**Columns:**
- `user_id`: Unique identifier for each user (1001-1020)
- `T`: Treatment assignment (0=control, 1=treatment)
- `country_EU`: Whether user is from EU (0=No, 1=Yes)
- `device_mobile`: Whether user is on mobile (0=Desktop, 1=Mobile)
- `prior_views`: Count of prior page views (0-8)
- `y`: Conversion outcome (0=No conversion, 1=Conversion)

### `analyze_experiment.py`
Complete Python script for analyzing A/B test data:
- Command-line interface for easy use
- Data validation and quality checks
- Randomization balance assessment
- GLM fitting with both logit and probit options
- Comprehensive results reporting
- Export capability for results

## Quick Start

### Using the Sample Data

```bash
# Analyze the sample data with default settings (logit)
python analyze_experiment.py --data sample_experiment_data.csv

# Use probit link function
python analyze_experiment.py --data sample_experiment_data.csv --link probit

# Save results to a file
python analyze_experiment.py --data sample_experiment_data.csv --output results.txt
```

### Using Your Own Data

1. Prepare your data in CSV format with the required columns
2. Ensure treatment is assigned at user level (not session level)
3. Run the analysis:

```bash
python analyze_experiment.py --data your_experiment.csv
```

### Python Script Usage

```python
import pandas as pd
from ab_glm import (
    fit_binomial_glm,
    marginal_effects_ate_and_rr,
    brier_score
)

# Load your data
df = pd.read_csv('your_experiment.csv')

# Fit GLM with cluster-robust SEs
glm, df_model, results = fit_binomial_glm(
    df,
    link="logit",
    cluster_col="user_id"
)

# Calculate treatment effects
ate, rr, p_treat, p_ctrl = marginal_effects_ate_and_rr(results, df_model)

print(f"Absolute Treatment Effect: {ate:.4f}")
print(f"Risk Ratio: {rr:.4f}")
print(f"Relative Lift: {(rr-1)*100:.2f}%")

# Model quality
predictions = results.predict(df_model)
brier = brier_score(df_model['y'].values, predictions)
print(f"Brier Score: {brier:.4f}")
```

## Data Preparation Tips

1. **Unique User IDs**: Ensure each user has a consistent ID across all their sessions
2. **Binary Coding**: Code binary variables as 0/1, not True/False or Yes/No
3. **Treatment Consistency**: Verify treatment doesn't change within users:
   ```python
   assert df.groupby('user_id')['T'].nunique().max() == 1
   ```
4. **Missing Data**: Handle missing values before analysis (the package drops NaN)
5. **Covariate Selection**: Only use pre-treatment covariates, never post-treatment

## Expected Output

Running the analysis script produces:
1. Data validation confirmation
2. Covariate balance table
3. Model coefficients with p-values
4. Business metrics (ATE, Risk Ratio, relative lift)
5. Executive summary with recommendations

Example output:
```
RESULTS SUMMARY
============================================================

Covariate-Adjusted Results:
  Control Rate: 0.156
  Treatment Rate: 0.203
  ATE (Risk Diff): 0.047
  Risk Ratio: 1.301
  Relative Lift: 30.1%
  P-value: 0.012
  Significant: Yes ✓
```

## Troubleshooting

**"Missing required columns" error**: Ensure your CSV has all required columns with exact names

**"Treatment varies within users" error**: Check that treatment is assigned at user level:
```python
df.groupby('user_id')['T'].nunique().value_counts()
```

**Singular matrix error**: This occurs when a variable has no variation (e.g., all users in treatment). Check your data balance.

**Perfect separation warning**: This can happen with small samples or when predictors perfectly predict the outcome. Consider removing problematic covariates or increasing sample size.
