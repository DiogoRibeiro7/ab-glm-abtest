# Troubleshooting Guide

This guide helps resolve common issues when using the ab-glm-abtest package for A/B testing analysis.

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [Data Preparation Errors](#data-preparation-errors)
3. [Model Fitting Problems](#model-fitting-problems)
4. [Statistical Issues](#statistical-issues)
5. [Performance Problems](#performance-problems)
6. [Interpretation Concerns](#interpretation-concerns)

## Installation Issues

### Poetry Installation Fails

**Problem:**
```
Poetry install fails with dependency conflicts
```

**Solutions:**
1. Update Poetry:
   ```bash
   poetry self update
   ```
2. Clear cache and reinstall:
   ```bash
   poetry cache clear pypi --all
   poetry install --no-cache
   ```
3. Use specific Python version:
   ```bash
   poetry env use python3.10
   poetry install
   ```

### Import Errors

**Problem:**
```python
ImportError: cannot import name 'fit_binomial_glm' from 'ab_glm'
```

**Solution:**
Ensure proper installation:
```bash
# If developing locally
pip install -e .

# Or with poetry
poetry install
```

### NumPy/Pandas Version Conflicts

**Problem:**
```
ValueError: numpy and pandas versions incompatible
```

**Solution:**
Update to compatible versions:
```bash
poetry update numpy pandas
# Or manually specify
poetry add numpy@^2.0 pandas@^2.2
```

## Data Preparation Errors

### Missing Required Columns

**Problem:**
```python
ValueError: Missing required columns: ['T', 'user_id']
```

**Solution:**
Ensure your DataFrame has all required columns:
```python
required_cols = ['user_id', 'T', 'country_EU', 'device_mobile', 'prior_views', 'y']
missing = set(required_cols) - set(df.columns)
if missing:
    print(f"Missing: {missing}")
    # Add missing columns or rename existing ones
    df = df.rename(columns={'treatment': 'T', 'outcome': 'y'})
```

### Treatment Varies Within Users

**Problem:**
```python
ValueError: Treatment varies within users! Check randomization.
```

**Diagnosis:**
```python
# Check treatment consistency
treatment_changes = df.groupby('user_id')['T'].nunique()
print(f"Users with inconsistent treatment: {(treatment_changes > 1).sum()}")

# Find problematic users
problem_users = treatment_changes[treatment_changes > 1].index
df[df['user_id'].isin(problem_users)].sort_values(['user_id', 'session_date'])
```

**Solutions:**
1. Keep first treatment assignment:
   ```python
   df['T'] = df.groupby('user_id')['T'].transform('first')
   ```
2. Remove users with inconsistent treatment:
   ```python
   consistent_users = treatment_changes[treatment_changes == 1].index
   df = df[df['user_id'].isin(consistent_users)]
   ```
3. Fix data pipeline to ensure proper randomization

### Non-Binary Variables

**Problem:**
```python
ValueError: 'T' must be binary (0/1)
```

**Solutions:**
1. Convert boolean to binary:
   ```python
   df['T'] = df['T'].astype(int)
   df['y'] = df['y'].astype(int)
   ```
2. Recode categorical:
   ```python
   df['T'] = df['treatment_group'].map({'control': 0, 'treatment': 1})
   ```
3. Check for unexpected values:
   ```python
   print(df['T'].value_counts())
   df = df[df['T'].isin([0, 1])]
   ```

### Missing Values

**Problem:**
```python
Dropped 30% of data due to missing values
```

**Diagnosis:**
```python
# Check missingness patterns
print(df.isnull().sum())
print(f"Rows with any missing: {df.isnull().any(axis=1).sum()}")

# Check if missingness is related to treatment
for col in df.columns:
    if df[col].isnull().any():
        print(f"{col} missingness by treatment:")
        print(df.groupby('T')[col].apply(lambda x: x.isnull().mean()))
```

**Solutions:**
1. Impute with median/mode:
   ```python
   df['prior_views'].fillna(df['prior_views'].median(), inplace=True)
   df['country_EU'].fillna(df['country_EU'].mode()[0], inplace=True)
   ```
2. Create missing indicator:
   ```python
   df['prior_views_missing'] = df['prior_views'].isnull().astype(int)
   df['prior_views'].fillna(0, inplace=True)
   ```
3. Use only complete cases:
   ```python
   df = df.dropna(subset=['y', 'T', 'user_id'])
   ```

## Model Fitting Problems

### Singular Matrix Error

**Problem:**
```python
numpy.linalg.LinAlgError: Singular matrix
```

**Causes:**
- Perfect collinearity between predictors
- No variation in a variable
- Perfect separation

**Diagnosis:**
```python
# Check for constant columns
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].nunique() == 1:
        print(f"{col} has no variation")

# Check correlation matrix
corr_matrix = df[['T', 'country_EU', 'device_mobile', 'prior_views']].corr()
high_corr = np.where(np.abs(corr_matrix) > 0.95)
```

**Solutions:**
1. Remove constant columns:
   ```python
   df = df.loc[:, df.nunique() > 1]
   ```
2. Remove one of highly correlated variables
3. Add small random noise (last resort):
   ```python
   df['prior_views'] += np.random.normal(0, 0.001, len(df))
   ```

### Perfect Separation Warning

**Problem:**
```
PerfectSeparationWarning: Perfect separation detected
```

**What it means:**
Some combination of predictors perfectly predicts the outcome

**Diagnosis:**
```python
# Check if any group has 0% or 100% conversion
for col in ['T', 'country_EU', 'device_mobile']:
    print(f"Conversion by {col}:")
    print(df.groupby(col)['y'].agg(['mean', 'count']))
```

**Solutions:**
1. Remove problematic predictors:
   ```python
   # If device_mobile perfectly predicts outcome
   glm, _, df_model, results = fit_binomial_glm(
       df[['user_id', 'T', 'country_EU', 'prior_views', 'y']],
       link="logit"
   )
   ```
2. Group rare categories:
   ```python
   # Combine rare values
   df['prior_views_grouped'] = pd.cut(df['prior_views'], bins=3)
   ```
3. Use penalized regression (not in package, but alternative)

### Convergence Failure

**Problem:**
```
ConvergenceWarning: Maximum iterations reached
```

**Solutions:**
1. Scale predictors:
   ```python
   df['prior_views_scaled'] = (df['prior_views'] - df['prior_views'].mean()) / df['prior_views'].std()
   ```
2. Check for outliers:
   ```python
   Q1 = df['prior_views'].quantile(0.25)
   Q3 = df['prior_views'].quantile(0.75)
   IQR = Q3 - Q1
   outliers = (df['prior_views'] < Q1 - 3*IQR) | (df['prior_views'] > Q3 + 3*IQR)
   print(f"Outliers: {outliers.sum()}")
   ```
3. Try different link function:
   ```python
   # If logit fails, try probit
   glm, _, df_model, results = fit_binomial_glm(df, link="probit")
   ```

## Statistical Issues

### Unexpectedly Large/Small Standard Errors

**Problem:**
```
Treatment SE = 0.0001 (suspiciously small)
Treatment SE = 10.5 (suspiciously large)
```

**Diagnosis:**
```python
# Check number of clusters
n_users = df['user_id'].nunique()
print(f"Number of clusters (users): {n_users}")

# Check cluster sizes
cluster_sizes = df.groupby('user_id').size()
print(f"Cluster size stats:")
print(cluster_sizes.describe())

# Check treatment balance
print(f"Treatment split: {df.groupby('user_id')['T'].first().value_counts(normalize=True)}")
```

**Solutions:**
1. Ensure sufficient clusters (>30 per group)
2. Check for data leakage:
   ```python
   # Ensure no post-treatment variables
   # Wrong: including purchase_amount when y=purchased
   ```
3. Verify cluster specification:
   ```python
   # Make sure cluster_col is correctly specified
   results = fit_binomial_glm(df, cluster_col='user_id')  # Not 'session_id'
   ```

### Unrealistic Effect Sizes

**Problem:**
```
ATE = 0.75 (75 percentage points increase seems too large)
Risk Ratio = 15.3 (unrealistic)
```

**Diagnosis:**
```python
# Check raw conversion rates
print("Raw conversion by treatment:")
print(df.groupby('T')['y'].agg(['mean', 'count', 'sum']))

# Look for data issues
print("Outcome distribution:")
print(df['y'].value_counts(normalize=True))

# Check for selection bias
user_df = df.groupby('user_id').agg({
    'T': 'first',
    'y': 'mean',
    'prior_views': 'first'
})
print("User-level stats by treatment:")
print(user_df.groupby('T').agg(['mean', 'std']))
```

**Solutions:**
1. Verify data integrity
2. Check for selection bias
3. Look for confounding time effects
4. Consider practical significance vs statistical

### P-value Inconsistencies

**Problem:**
```
Large effect but p > 0.05, or tiny effect but p < 0.001
```

**Understanding:**
- P-value depends on effect size AND sample size
- Large samples can detect tiny effects
- Small samples miss large effects

**Solutions:**
1. Report confidence intervals:
   ```python
   # More informative than p-value alone
   print(f"ATE: {ate:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
   ```
2. Calculate minimum detectable effect:
   ```python
   # What effect size can we reliably detect?
   from scipy import stats
   z_alpha = stats.norm.ppf(0.975)
   z_beta = stats.norm.ppf(0.80)
   se = results.bse['T']
   mde = (z_alpha + z_beta) * se
   print(f"Minimum detectable effect: {mde:.3f}")
   ```

## Performance Problems

### Slow Model Fitting

**Problem:**
Processing takes >1 minute for moderate data

**Diagnosis:**
```python
print(f"Data size: {df.shape}")
print(f"Users: {df['user_id'].nunique()}")
print(f"Avg sessions per user: {len(df)/df['user_id'].nunique():.1f}")
```

**Solutions:**
1. Sample for exploration:
   ```python
   # Use sample for initial analysis
   sample_users = df['user_id'].drop_duplicates().sample(n=1000)
   df_sample = df[df['user_id'].isin(sample_users)]
   ```
2. Optimize data types:
   ```python
   df['user_id'] = df['user_id'].astype('category')
   df['T'] = df['T'].astype('int8')
   ```
3. Remove unnecessary columns:
   ```python
   cols_needed = ['user_id', 'T', 'country_EU', 'device_mobile', 'prior_views', 'y']
   df = df[cols_needed]
   ```

### Memory Errors

**Problem:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
1. Process in chunks:
   ```python
   # Fit model on sample, apply to full data
   sample_results = fit_binomial_glm(df_sample)
   ```
2. Reduce precision:
   ```python
   df = df.astype({'prior_views': 'float32'})
   ```
3. Use sparse matrices for categorical variables

## Interpretation Concerns

### Different Results: Logit vs Probit

**Problem:**
Logit and probit give very different answers

**Expected differences:**
- Coefficients differ by ~1.6x (normal)
- ATE/RR should be very similar

**When to worry:**
- ATE differs by >10%
- Opposite signs
- One converges, other doesn't

**Solutions:**
1. Check for near-separation:
   ```python
   # Both should give similar results on probability scale
   results_logit = fit_binomial_glm(df, link="logit")
   results_probit = fit_binomial_glm(df, link="probit")

   ate_logit, _, _, _ = marginal_effects_ate_and_rr(results_logit[3], results_logit[2])
   ate_probit, _, _, _ = marginal_effects_ate_and_rr(results_probit[3], results_probit[2])

   print(f"Logit ATE: {ate_logit:.4f}")
   print(f"Probit ATE: {ate_probit:.4f}")
   print(f"Relative difference: {abs(ate_logit-ate_probit)/ate_logit:.1%}")
   ```
2. Use bootstrap for robustness

### Covariate Adjustment Changes Sign

**Problem:**
```
Unadjusted: +5% effect
Adjusted: -2% effect
```

**This suggests:**
- Strong confounding
- Simpson's paradox
- Incorrect model specification

**Investigation:**
```python
# Compare adjusted vs unadjusted
# Unadjusted
simple = df.groupby('T')['y'].mean()
simple_diff = simple[1] - simple[0]

# Adjusted
ate, _, _, _ = marginal_effects_ate_and_rr(results, df_model)

print(f"Unadjusted effect: {simple_diff:.3f}")
print(f"Adjusted effect: {ate:.3f}")

# Check covariate balance
covariates = ['country_EU', 'device_mobile', 'prior_views']
for cov in covariates:
    print(f"\n{cov} by treatment:")
    print(df.groupby('T')[cov].agg(['mean', 'std']))
```

**Solutions:**
1. Verify covariate relationships
2. Check for interactions:
   ```python
   # May need interaction terms
   df['T_x_mobile'] = df['T'] * df['device_mobile']
   ```
3. Report both adjusted and unadjusted

## Getting Help

### Debug Information to Collect

When seeking help, provide:

```python
import sys
import numpy as np
import pandas as pd
import statsmodels
from ab_glm import __version__

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Statsmodels: {statsmodels.__version__}")
print(f"ab-glm: {__version__}")

print(f"\nData shape: {df.shape}")
print(f"Users: {df['user_id'].nunique()}")
print(f"Treatment split: {df.groupby('user_id')['T'].first().value_counts()}")
print(f"Outcome rate: {df['y'].mean():.3f}")
print(f"Missing values: {df.isnull().sum().sum()}")
```

### Common Error Messages Reference

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| `Singular matrix` | No variation or perfect collinearity | Remove constant columns |
| `Perfect separation` | Predictor perfectly predicts outcome | Remove or group predictors |
| `Missing required columns` | Wrong column names | Rename to match expected |
| `Treatment varies within users` | Randomization issue | Use first treatment per user |
| `No rows left after dropping NA` | Too many missing values | Impute or check data source |
| `KeyError: 'T'` | Wrong coefficient name | Check `results.params.index` |

### Resources

- [GitHub Issues](https://github.com/yourusername/ab-glm-abtest/issues)
- [Stack Overflow statsmodels tag](https://stackoverflow.com/questions/tagged/statsmodels)
- [Cross Validated](https://stats.stackexchange.com/) for statistical questions