# Interpretation Guide for A/B Test GLM Results

This guide explains how to interpret the results from Binomial GLM analysis for A/B testing, including coefficients, standard errors, and business metrics.

## Table of Contents
1. [Understanding GLM Output](#understanding-glm-output)
2. [Link Scale vs Probability Scale](#link-scale-vs-probability-scale)
3. [Interpreting Coefficients](#interpreting-coefficients)
4. [Business Metrics](#business-metrics)
5. [Statistical Significance](#statistical-significance)
6. [Model Diagnostics](#model-diagnostics)
7. [Common Pitfalls](#common-pitfalls)

## Understanding GLM Output

When you fit a Binomial GLM, you get several types of output:

```python
glm, _, df_model, results = fit_binomial_glm(df, link="logit")
```

### Components:
- **Model coefficients** (on link scale)
- **Standard errors** (cluster-robust)
- **P-values** for hypothesis testing
- **Predictions** (on probability scale)

## Link Scale vs Probability Scale

### Link Scale (Model Space)
The GLM operates in the link scale where relationships are linear:
- **Logit**: log(p/(1-p)) = β₀ + β₁T + β₂X₁ + ...
- **Probit**: Φ⁻¹(p) = β₀ + β₁T + β₂X₁ + ...

**Coefficient interpretation on link scale:**
- Logit: One-unit change in X changes log-odds by β
- Probit: One-unit change in X changes z-score by β

### Probability Scale (Business Space)
What stakeholders care about:
- Actual conversion probabilities
- Absolute differences (ATE)
- Relative differences (Risk Ratio)

**Always report results on probability scale for business decisions.**

## Interpreting Coefficients

### Treatment Coefficient (β_T)

```
Model output:
T coefficient: 0.35 (SE: 0.08, p=0.0001)
```

**Logit interpretation:**
- exp(0.35) = 1.42 → 42% increase in odds
- Positive = treatment increases conversion
- Magnitude depends on baseline rate

**Converting to probability scale:**
```python
ate, rr, p_treat, p_ctrl = marginal_effects_ate_and_rr(results, df_model)
```

### Covariate Coefficients

```
country_EU coefficient: 0.20 (SE: 0.06, p=0.001)
```

**Interpretation:**
- EU users have 22% higher odds of converting (exp(0.20) = 1.22)
- Controls for this difference when estimating treatment effect
- Makes treatment effect estimate more precise

### Intercept

```
Intercept: -2.0 (SE: 0.15)
```

**Interpretation:**
- Baseline log-odds when all predictors = 0
- Probability = 1/(1+exp(2.0)) = 0.119 (11.9%)
- Reference group: Control, non-EU, desktop, zero prior views

## Business Metrics

### Average Treatment Effect (ATE)

**Definition:** Absolute difference in probability between treatment and control

```
ATE = 0.031 (3.1 percentage points)
```

**Interpretation:**
- Treatment increases conversion by 3.1 percentage points
- If baseline is 10%, treatment achieves 13.1%
- Most intuitive metric for business stakeholders

**Calculation:**
1. Predict probabilities with everyone in treatment
2. Predict probabilities with everyone in control
3. ATE = mean(P_treatment) - mean(P_control)

### Risk Ratio (RR)

**Definition:** Ratio of treatment probability to control probability

```
Risk Ratio = 1.24
```

**Interpretation:**
- Treatment group is 1.24x as likely to convert
- 24% relative increase in conversion
- Useful for comparing across different baselines

**When to use:**
- ATE for absolute impact (revenue calculations)
- RR for relative impact (performance metrics)

### Number Needed to Treat (NNT)

**Definition:** How many users need treatment to get one additional conversion

```
NNT = 1/ATE = 1/0.031 = 32
```

**Interpretation:**
- Treat 32 users to get 1 extra conversion
- Useful for cost-benefit analysis

## Statistical Significance

### P-values

```
Treatment p-value: 0.023
```

**Interpretation:**
- Probability of seeing this effect if treatment has no real impact
- p < 0.05 → statistically significant at 5% level
- **NOT the probability that treatment works**

### Confidence Intervals

```
ATE: 0.031 [95% CI: 0.012, 0.050]
```

**Interpretation:**
- 95% confident true effect is between 1.2% and 5.0%
- Excludes zero → statistically significant
- Width indicates precision

### Cluster-Robust Standard Errors

**Why needed:**
- Multiple sessions per user violate independence
- Regular SEs underestimate uncertainty
- Cluster-robust SEs account for within-user correlation

**Impact:**
- Usually 20-50% larger than naive SEs
- More conservative (fewer false positives)
- Required for valid inference

## Model Diagnostics

### Brier Score

```
Brier Score: 0.095
```

**Interpretation:**
- Measures prediction accuracy (0=perfect, 0.25=random)
- < 0.1: Excellent
- 0.1-0.15: Good
- 0.15-0.2: Acceptable
- > 0.2: Poor

**Calculation:**
```
Brier = mean((y - p_predicted)²)
```

### Calibration

Good calibration means predicted probabilities match actual frequencies:
- If model predicts 30% probability for 100 users, ~30 should convert
- Check with calibration plots
- Poor calibration → biased treatment effects

### Deviance

```
Deviance: 245.3
Null Deviance: 289.7
```

**Interpretation:**
- Reduction = 289.7 - 245.3 = 44.4
- Model explains substantial variation
- Pseudo R² ≈ 1 - (245.3/289.7) = 0.15

## Common Pitfalls

### 1. Ignoring Clustering
**Problem:** Using regular SEs when users have multiple sessions
**Consequence:** P-values too small, false positives
**Solution:** Always use cluster-robust SEs

### 2. Reporting Link-Scale Coefficients
**Problem:** Showing stakeholders logit coefficients
**Consequence:** Confusion, misinterpretation
**Solution:** Convert to probability scale (ATE, RR)

### 3. Perfect Separation
**Problem:** Predictor perfectly predicts outcome
**Symptoms:**
- Huge coefficients (|β| > 10)
- Warning messages
- Convergence issues

**Solutions:**
- Remove problematic predictors
- Use penalized regression
- Increase sample size

### 4. Post-Treatment Covariates
**Problem:** Including variables affected by treatment
**Consequence:** Biased estimates
**Solution:** Only use pre-treatment characteristics

### 5. Simpson's Paradox
**Problem:** Overall effect opposite of subgroup effects
**Example:** Treatment helps both mobile and desktop, but hurts overall
**Solution:** Check heterogeneous effects, use correct weights

## Practical Examples

### Example 1: Positive Significant Effect

```
Results:
- Treatment coefficient: 0.42 (p=0.001)
- ATE: 0.035 (3.5 pp)
- Risk Ratio: 1.31
- Baseline rate: 11.2%
```

**Business interpretation:**
"The new checkout flow increases conversions by 3.5 percentage points, from 11.2% to 14.7%. This represents a 31% relative improvement and is highly statistically significant (p=0.001)."

### Example 2: Negative Non-Significant Effect

```
Results:
- Treatment coefficient: -0.08 (p=0.31)
- ATE: -0.008 (-0.8 pp)
- Risk Ratio: 0.94
- Baseline rate: 13.5%
```

**Business interpretation:**
"The treatment shows a small negative effect (-0.8 pp), but this is not statistically significant (p=0.31). We cannot conclude the treatment harms conversion, but there's no evidence it helps."

### Example 3: Heterogeneous Effects

```
Results:
- Overall ATE: 0.02 (p=0.15)
- Mobile ATE: 0.05 (p=0.01)
- Desktop ATE: -0.01 (p=0.62)
```

**Business interpretation:**
"While the overall effect is not significant, the treatment significantly improves mobile conversion by 5 pp. Consider implementing for mobile only."

## Quick Reference

### Key Metrics to Report
1. **Sample size** (users and observations)
2. **Baseline conversion rate** (control)
3. **Absolute effect** (ATE with CI)
4. **Relative effect** (RR or % lift)
5. **Statistical significance** (p-value)
6. **Model quality** (Brier score)

### Decision Framework
- **p < 0.05 AND practically significant** → Implement
- **p < 0.05 BUT too small** → Consider cost/benefit
- **p > 0.05 BUT large effect** → Get more data
- **p > 0.05 AND small effect** → Don't implement

### Red Flags
- Huge coefficients (|β| > 5)
- Perfect predictions (some p = 0 or 1)
- Brier score > 0.25
- Dramatically different logit vs probit results
- Treatment effect varies within users

## Further Reading

- [Gelman & Hill (2007)](http://www.stat.columbia.edu/~gelman/arm/) - Data Analysis Using Regression
- [Angrist & Pischke (2009)](https://www.mostlyharmlesseconometrics.com/) - Mostly Harmless Econometrics
- [Imbens & Rubin (2015)](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB) - Causal Inference