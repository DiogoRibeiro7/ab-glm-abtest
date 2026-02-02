# Assumptions and Limitations

This document outlines the statistical assumptions underlying the ab-glm-abtest package and its limitations.

## Table of Contents
1. [Statistical Assumptions](#statistical-assumptions)
2. [Model Limitations](#model-limitations)
3. [Data Requirements](#data-requirements)
4. [Validity Threats](#validity-threats)
5. [When Not to Use This Package](#when-not-to-use-this-package)
6. [Alternative Approaches](#alternative-approaches)

## Statistical Assumptions

### 1. Stable Unit Treatment Value Assumption (SUTVA)

**Assumption:** One user's treatment assignment doesn't affect another user's outcome.

**What this means:**
- No network effects (user's friends seeing treatment)
- No capacity constraints (limited inventory)
- No learning effects between users

**Violations and consequences:**
```python
# Example violation: Social network effects
# If treated users share feature with control users
# Result: Underestimate treatment effect (spillover)

# Check for violations:
# 1. Plot outcomes over time
df.groupby(['date', 'T'])['y'].mean().plot()

# 2. Check for geographic clustering
df.groupby(['region', 'T'])['y'].mean()
```

**Solutions if violated:**
- Cluster randomization by network/geography
- Time-based randomization (alternating periods)
- Increase separation between treatment and control

### 2. Random Assignment

**Assumption:** Treatment assignment is random conditional on covariates.

**Required:**
```python
# Check balance
user_df = df.groupby('user_id').first()
for col in ['country_EU', 'device_mobile', 'prior_views']:
    print(f"{col}: Control={user_df[user_df.T==0][col].mean():.3f}, "
          f"Treatment={user_df[user_df.T==1][col].mean():.3f}")
```

**Violations:**
- Self-selection into treatment
- Time-based assignment with trends
- Technical issues causing non-random exposure

**Consequences:**
- Biased treatment effect estimates
- Invalid p-values and confidence intervals

### 3. Correct Model Specification

**Assumption:** The binomial GLM with chosen link function correctly specifies the relationship.

**What's assumed:**
- Binary outcome follows binomial distribution
- Link function (logit/probit) is appropriate
- Linear relationship on link scale
- No important omitted variables

**Diagnostics:**
```python
# Check residual patterns
residuals = df_model['y'] - results.predict(df_model)
plt.scatter(results.predict(df_model), residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
```

**Violations visible as:**
- Systematic patterns in residuals
- Poor calibration (Brier score > 0.2)
- Very different results for logit vs probit

### 4. No Perfect Separation

**Assumption:** No combination of predictors perfectly predicts the outcome.

**Check for separation:**
```python
# Warning signs
for col in ['T', 'country_EU', 'device_mobile']:
    conversion_rate = df.groupby(col)['y'].mean()
    if (conversion_rate == 0).any() or (conversion_rate == 1).any():
        print(f"Warning: Perfect prediction in {col}")
```

**Consequences of violation:**
- Infinite or undefined coefficients
- Convergence failures
- Invalid standard errors

**Solutions:**
- Remove or combine sparse categories
- Use Firth's penalized likelihood
- Collect more data

### 5. Independence of Observations (Within Clusters)

**Assumption:** After accounting for clustering, observations are independent.

**What's handled automatically:**
- Multiple sessions from same user (clustered)
- Correlation within users

**What's NOT handled:**
- Time-series dependence
- Spatial correlation
- Network effects

**Diagnostic:**
```python
# Check for time trends
daily_rates = df.groupby(['date', 'T'])['y'].mean().unstack()
daily_rates.plot()
plt.title('Conversion rates over time')
```

## Model Limitations

### 1. Binary Outcomes Only

**Limited to:**
- Conversion (yes/no)
- Click (yes/no)
- Purchase (yes/no)

**NOT suitable for:**
- Revenue amounts → Use linear regression or Gamma GLM
- Count data → Use Poisson or Negative Binomial
- Time to event → Use survival analysis
- Ratings (1-5) → Use ordinal regression

### 2. Fixed Covariates

**Current implementation:**
- Fixed set of covariates
- No automated variable selection
- No interaction terms by default

**To add interactions:**
```python
# Must modify source code or create derived variables
df['T_x_mobile'] = df['T'] * df['device_mobile']
# Then include in model formula
```

### 3. No Time-Varying Effects

**Cannot handle:**
- Treatment effects that change over time
- Seasonal patterns
- Learning/fatigue effects

**Workaround:**
```python
# Analyze time periods separately
week1 = df[df['week'] == 1]
week2 = df[df['week'] == 2]
# Fit models separately and compare
```

### 4. No Heterogeneous Treatment Effects (Built-in)

**Current:**
- Single average treatment effect
- No automatic subgroup analysis

**To analyze heterogeneity:**
```python
# Manual subgroup analysis
for subgroup in ['mobile', 'desktop']:
    subset = df[df['device_type'] == subgroup]
    # Fit model and get ATE for subgroup
```

### 5. Sample Size Requirements

**Minimum requirements:**
- At least 30 users per treatment arm
- At least 100 total observations
- At least 10 conversions per group

**For reliable inference:**
- 500+ users per arm recommended
- 20+ conversions per covariate level

## Data Requirements

### Strict Requirements

1. **Binary Treatment**
   ```python
   assert set(df['T'].unique()) == {0, 1}
   ```

2. **Binary Outcome**
   ```python
   assert set(df['y'].unique()) == {0, 1}
   ```

3. **User-Level Treatment Assignment**
   ```python
   assert (df.groupby('user_id')['T'].nunique() == 1).all()
   ```

4. **No Missing Treatment or Outcome**
   ```python
   assert df[['T', 'y']].notna().all().all()
   ```

### Data Quality Assumptions

1. **No Data Leakage**
   - Covariates measured pre-treatment
   - No post-treatment variables included

2. **Accurate Tracking**
   - User IDs are consistent
   - No duplicate events
   - Complete conversion tracking

3. **Representative Sample**
   - Test period represents typical behavior
   - No major external events during test

## Validity Threats

### Internal Validity Threats

1. **Selection Bias**
   - Non-random opt-in/opt-out
   - Technical issues affecting exposure

2. **History Effects**
   - External events during test
   - Concurrent changes/tests

3. **Instrumentation**
   - Tracking changes mid-test
   - Definition changes

### External Validity Threats

1. **Novelty Effects**
   - Initial excitement wears off
   - Learning curves

2. **Hawthorne Effect**
   - Users behave differently when aware of test

3. **Seasonal/Temporal Effects**
   - Results may not generalize to other periods

4. **Population Differences**
   - Test users may differ from future users

## When Not to Use This Package

### Wrong Outcome Type

❌ **Continuous outcomes** (revenue, time on site)
```python
# Wrong: y = revenue amount
# Right: Use linear regression
from sklearn.linear_model import LinearRegression
```

❌ **Count outcomes** (number of purchases)
```python
# Wrong: y = purchase count
# Right: Use Poisson regression
import statsmodels.api as sm
model = sm.GLM(y, X, family=sm.families.Poisson())
```

❌ **Ordinal outcomes** (ratings, NPS)
```python
# Wrong: y = rating (1-5)
# Right: Use ordinal logistic regression
from statsmodels.miscmodels.ordinal_model import OrderedModel
```

### Wrong Randomization Unit

❌ **Page-level randomization**
- Same user sees both versions
- Violates independence assumption

❌ **Time-based randomization**
- Monday = control, Tuesday = treatment
- Confounded with time effects

❌ **Cluster randomization**
- Whole offices/regions assigned together
- Needs different analysis approach

### Statistical Power Issues

❌ **Rare events** (<1% base rate)
- Need massive sample sizes
- Consider different metrics

❌ **Small effects** (<1% absolute)
- May need millions of users
- Consider more sensitive metrics

❌ **High variance outcomes**
- Highly skewed distributions
- Many outliers

## Alternative Approaches

### For Different Outcome Types

**Continuous Outcomes:**
```python
# Linear regression with clustered SEs
import statsmodels.api as sm
model = sm.OLS(y, X)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df['user_id']})
```

**Count Data:**
```python
# Negative binomial for overdispersed counts
model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
```

**Time-to-Event:**
```python
# Cox proportional hazards
from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='converted')
```

### For Different Designs

**Matched Pairs:**
```python
# Paired t-test or McNemar's test
from scipy.stats import mcnemar
```

**Crossover Design:**
```python
# Mixed effects model
import statsmodels.formula.api as smf
model = smf.mixedlm("y ~ treatment + period", df, groups=df["user_id"])
```

**Sequential Testing:**
```python
# Use sequential analysis methods
# Consider alpha spending functions
```

### For Causal Inference

**Instrumental Variables:**
- When randomization is imperfect
- Use 2SLS or IV regression

**Regression Discontinuity:**
- Assignment based on threshold
- Different identification strategy

**Difference-in-Differences:**
- When you have pre/post data
- Controls for time trends

## Best Practices Summary

### ✅ DO:
- Verify randomization quality before analysis
- Check model diagnostics (Brier score, residuals)
- Report both statistical and practical significance
- Consider multiple comparisons if testing many hypotheses
- Document deviations from ideal conditions

### ❌ DON'T:
- Use with non-binary outcomes
- Ignore clustering (multiple sessions per user)
- Include post-treatment covariates
- Assume effects are homogeneous across subgroups
- Extrapolate beyond the test population/period

### 🤔 CONSIDER:
- Whether assumptions are reasonable for your context
- If violations affect your conclusions
- Alternative analyses as robustness checks
- Practical significance vs statistical significance
- Long-term vs short-term effects

## Further Reading

### Statistical Theory
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*
- Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*

### A/B Testing Practice
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments*
- Georgiev, G. Z. (2019). *Statistical Methods in Online A/B Testing*

### Violations and Solutions
- Firth, D. (1993). "Bias reduction of maximum likelihood estimates" - For separation
- White, H. (1980). "A heteroskedasticity-consistent covariance matrix estimator" - For robust SEs
- Manski, C. F. (2007). *Identification for Prediction and Decision* - For partial identification