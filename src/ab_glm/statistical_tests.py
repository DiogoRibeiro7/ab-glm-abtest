"""
Advanced statistical tests and robustness checks for A/B testing.

This module provides:
- Bootstrap confidence intervals
- Permutation testing
- Multiple testing corrections
- Power analysis
- Sample ratio mismatch detection
- Sequential testing
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, norm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import zt_ind_solve_power
from tqdm import tqdm


@dataclass
class BootstrapResult:
    """Results from bootstrap analysis."""
    estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    p_value: float
    n_iterations: int
    confidence_level: float


@dataclass
class PermutationTestResult:
    """Results from permutation test."""
    observed_statistic: float
    p_value: float
    null_distribution: np.ndarray
    n_permutations: int
    two_sided: bool


@dataclass
class PowerAnalysisResult:
    """Results from power analysis."""
    power: float
    sample_size: Optional[int]
    effect_size: Optional[float]
    alpha: float
    alternative: str


def bootstrap_ci(
    data: pd.DataFrame,
    statistic_func: Callable,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    show_progress: bool = True
) -> BootstrapResult:
    """
    Calculate bootstrap confidence intervals for any statistic.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    statistic_func : Callable
        Function that takes data and returns a statistic
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int, optional
        Random seed for reproducibility
    show_progress : bool
        Show progress bar

    Returns
    -------
    BootstrapResult
        Bootstrap estimates and confidence intervals
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Calculate observed statistic
    observed = statistic_func(data)

    # Bootstrap resampling
    bootstrap_stats = []
    n_samples = len(data)

    iterator = range(n_bootstrap)
    if show_progress:
        iterator = tqdm(iterator, desc="Bootstrap sampling")

    for _ in iterator:
        # Resample with replacement
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_data = data.iloc[boot_indices]

        try:
            boot_stat = statistic_func(boot_data)
            bootstrap_stats.append(boot_stat)
        except Exception:
            # Skip failed iterations
            continue

    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    # Calculate p-value (two-sided test against null of 0)
    if observed > 0:
        p_value = 2 * np.mean(bootstrap_stats <= 0)
    else:
        p_value = 2 * np.mean(bootstrap_stats >= 0)
    p_value = min(p_value, 1.0)

    return BootstrapResult(
        estimate=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=np.std(bootstrap_stats),
        p_value=p_value,
        n_iterations=len(bootstrap_stats),
        confidence_level=confidence_level
    )


def permutation_test(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    statistic_func: Optional[Callable] = None,
    n_permutations: int = 10000,
    two_sided: bool = True,
    random_state: Optional[int] = None,
    show_progress: bool = True
) -> PermutationTestResult:
    """
    Perform permutation test for treatment effect.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    treatment_col : str
        Name of treatment column (0/1)
    outcome_col : str
        Name of outcome column
    statistic_func : Callable, optional
        Custom test statistic (default: difference in means)
    n_permutations : int
        Number of permutations
    two_sided : bool
        Two-sided or one-sided test
    random_state : int, optional
        Random seed
    show_progress : bool
        Show progress bar

    Returns
    -------
    PermutationTestResult
        Test results with p-value and null distribution
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Default statistic: difference in means
    if statistic_func is None:
        def statistic_func(df):
            treated = df[df[treatment_col] == 1][outcome_col]
            control = df[df[treatment_col] == 0][outcome_col]
            return treated.mean() - control.mean()

    # Calculate observed statistic
    observed = statistic_func(data)

    # Permutation test
    permuted_stats = []

    iterator = range(n_permutations)
    if show_progress:
        iterator = tqdm(iterator, desc="Permutation test")

    for _ in iterator:
        # Shuffle treatment labels
        permuted_data = data.copy()
        permuted_data[treatment_col] = np.random.permutation(data[treatment_col].values)

        perm_stat = statistic_func(permuted_data)
        permuted_stats.append(perm_stat)

    permuted_stats = np.array(permuted_stats)

    # Calculate p-value
    if two_sided:
        extreme_stats = np.abs(permuted_stats) >= np.abs(observed)
    else:
        if observed > 0:
            extreme_stats = permuted_stats >= observed
        else:
            extreme_stats = permuted_stats <= observed

    p_value = (np.sum(extreme_stats) + 1) / (n_permutations + 1)

    return PermutationTestResult(
        observed_statistic=observed,
        p_value=p_value,
        null_distribution=permuted_stats,
        n_permutations=n_permutations,
        two_sided=two_sided
    )


def multiple_testing_correction(
    p_values: List[float],
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Dict[str, np.ndarray]:
    """
    Apply multiple testing correction.

    Parameters
    ----------
    p_values : List[float]
        Original p-values
    method : str
        Correction method ('bonferroni', 'fdr_bh', 'fdr_by', 'holm')
    alpha : float
        Significance level

    Returns
    -------
    Dict with corrected p-values and reject decisions
    """
    methods_map = {
        'bonferroni': 'bonferroni',
        'fdr_bh': 'fdr_bh',  # Benjamini-Hochberg
        'fdr_by': 'fdr_by',  # Benjamini-Yekutieli
        'holm': 'holm',
        'sidak': 'sidak',
        'holm-sidak': 'holm-sidak'
    }

    if method not in methods_map:
        raise ValueError(f"Method must be one of {list(methods_map.keys())}")

    reject, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
        p_values, alpha=alpha, method=methods_map[method]
    )

    return {
        'p_values_original': np.array(p_values),
        'p_values_adjusted': p_adjusted,
        'reject': reject,
        'alpha_corrected': alpha_bonf if method == 'bonferroni' else alpha_sidak,
        'method': method
    }


def sample_ratio_mismatch_test(
    n_treatment: int,
    n_control: int,
    expected_ratio: float = 0.5,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Test for sample ratio mismatch (SRM).

    Detects if randomization might be broken.

    Parameters
    ----------
    n_treatment : int
        Number of treatment samples
    n_control : int
        Number of control samples
    expected_ratio : float
        Expected proportion in treatment (default 0.5 for 50/50 split)
    alpha : float
        Significance level

    Returns
    -------
    Dict with test results
    """
    n_total = n_treatment + n_control
    observed_ratio = n_treatment / n_total

    # Binomial test
    p_value = stats.binomtest(
        n_treatment, n_total, expected_ratio, alternative='two-sided'
    ).pvalue

    # Chi-square test
    expected_treatment = n_total * expected_ratio
    expected_control = n_total * (1 - expected_ratio)

    chi2_stat = ((n_treatment - expected_treatment) ** 2 / expected_treatment +
                 (n_control - expected_control) ** 2 / expected_control)
    chi2_p = 1 - stats.chi2.cdf(chi2_stat, df=1)

    return {
        'observed_ratio': observed_ratio,
        'expected_ratio': expected_ratio,
        'srm_detected': p_value < alpha,
        'p_value_binomial': p_value,
        'p_value_chi2': chi2_p,
        'n_treatment': n_treatment,
        'n_control': n_control,
        'warning': "Sample ratio mismatch detected!" if p_value < alpha else None
    }


def power_analysis(
    effect_size: Optional[float] = None,
    sample_size: Optional[int] = None,
    power: Optional[float] = None,
    alpha: float = 0.05,
    alternative: str = 'two-sided',
    ratio: float = 1.0
) -> PowerAnalysisResult:
    """
    Perform power analysis for A/B test.

    Calculates one of: effect_size, sample_size, or power.

    Parameters
    ----------
    effect_size : float, optional
        Standardized effect size (Cohen's d)
    sample_size : int, optional
        Sample size per group
    power : float, optional
        Statistical power (1 - Type II error)
    alpha : float
        Significance level (Type I error)
    alternative : str
        'two-sided', 'larger', or 'smaller'
    ratio : float
        Ratio of sample sizes (n_treatment / n_control)

    Returns
    -------
    PowerAnalysisResult
        Power analysis results
    """
    # Count how many parameters are provided
    n_provided = sum(x is not None for x in [effect_size, sample_size, power])

    if n_provided != 2:
        raise ValueError("Exactly two of effect_size, sample_size, and power must be provided")

    result = zt_ind_solve_power(
        effect_size=effect_size,
        nobs1=sample_size,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative
    )

    # Determine what was calculated
    if effect_size is None:
        calculated_value = result
        calculated_param = 'effect_size'
    elif sample_size is None:
        calculated_value = int(np.ceil(result))
        calculated_param = 'sample_size'
    else:  # power is None
        calculated_value = result
        calculated_param = 'power'

    return PowerAnalysisResult(
        power=power if power is not None else (calculated_value if calculated_param == 'power' else None),
        sample_size=sample_size if sample_size is not None else (calculated_value if calculated_param == 'sample_size' else None),
        effect_size=effect_size if effect_size is not None else (calculated_value if calculated_param == 'effect_size' else None),
        alpha=alpha,
        alternative=alternative
    )


def sequential_testing(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    alpha: float = 0.05,
    power: float = 0.8,
    effect_size: float = 0.2,
    max_samples: Optional[int] = None
) -> Dict[str, any]:
    """
    Perform sequential testing with early stopping.

    Uses O'Brien-Fleming spending function for alpha spending.

    Parameters
    ----------
    data : pd.DataFrame
        Input data (assumed to be ordered by time)
    treatment_col : str
        Treatment column name
    outcome_col : str
        Outcome column name
    alpha : float
        Overall significance level
    power : float
        Desired power
    effect_size : float
        Expected effect size
    max_samples : int, optional
        Maximum sample size

    Returns
    -------
    Dict with sequential testing results
    """
    n = len(data)
    if max_samples is None:
        max_samples = n

    # Define stopping boundaries using O'Brien-Fleming
    n_looks = min(5, n // 100)  # Max 5 interim looks
    if n_looks < 2:
        n_looks = 2

    look_times = np.linspace(n // n_looks, n, n_looks, dtype=int)

    # O'Brien-Fleming boundaries
    def obrien_fleming_boundary(k, K, alpha):
        """Calculate O'Brien-Fleming boundary for look k of K."""
        t_k = k / K  # Information fraction
        return alpha / (2 * np.sqrt(K * t_k))

    results = {
        'stopped_early': False,
        'stopping_n': n,
        'final_p_value': None,
        'boundaries': [],
        'p_values': [],
        'decision': None
    }

    for i, look_n in enumerate(look_times):
        # Get data up to current look
        current_data = data.iloc[:look_n]

        # Calculate test statistic
        treated = current_data[current_data[treatment_col] == 1][outcome_col]
        control = current_data[current_data[treatment_col] == 0][outcome_col]

        if len(treated) == 0 or len(control) == 0:
            continue

        # Two-sample z-test
        mean_diff = treated.mean() - control.mean()
        se_diff = np.sqrt(treated.var() / len(treated) + control.var() / len(control))

        if se_diff > 0:
            z_stat = mean_diff / se_diff
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        else:
            p_value = 1.0

        # Calculate boundary for this look
        boundary = obrien_fleming_boundary(i + 1, n_looks, alpha)

        results['boundaries'].append(boundary)
        results['p_values'].append(p_value)

        # Check stopping criteria
        if p_value < boundary:
            results['stopped_early'] = True
            results['stopping_n'] = look_n
            results['final_p_value'] = p_value
            results['decision'] = 'reject'
            break

    if not results['stopped_early']:
        results['final_p_value'] = results['p_values'][-1] if results['p_values'] else None
        results['decision'] = 'fail to reject' if results['final_p_value'] and results['final_p_value'] > alpha else 'reject'

    return results


def heterogeneous_treatment_effects(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    subgroup_cols: List[str],
    min_samples: int = 30
) -> pd.DataFrame:
    """
    Analyze heterogeneous treatment effects across subgroups.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    treatment_col : str
        Treatment column name
    outcome_col : str
        Outcome column name
    subgroup_cols : List[str]
        Columns defining subgroups
    min_samples : int
        Minimum samples per subgroup

    Returns
    -------
    pd.DataFrame
        Treatment effects by subgroup
    """
    results = []

    for col in subgroup_cols:
        unique_values = data[col].unique()

        for value in unique_values:
            subgroup_data = data[data[col] == value]

            if len(subgroup_data) < min_samples:
                continue

            treated = subgroup_data[subgroup_data[treatment_col] == 1][outcome_col]
            control = subgroup_data[subgroup_data[treatment_col] == 0][outcome_col]

            if len(treated) == 0 or len(control) == 0:
                continue

            # Calculate treatment effect
            effect = treated.mean() - control.mean()

            # Calculate confidence interval
            se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))
            ci_lower = effect - 1.96 * se
            ci_upper = effect + 1.96 * se

            # Two-sample t-test
            _, p_value = stats.ttest_ind(treated, control)

            results.append({
                'subgroup_variable': col,
                'subgroup_value': value,
                'n_treatment': len(treated),
                'n_control': len(control),
                'treatment_mean': treated.mean(),
                'control_mean': control.mean(),
                'effect': effect,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

    return pd.DataFrame(results)


def variance_reduction_cuped(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_col: str
) -> Dict[str, float]:
    """
    Apply CUPED (Controlled-experiment Using Pre-Experiment Data) for variance reduction.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    treatment_col : str
        Treatment column
    outcome_col : str
        Outcome column
    covariate_col : str
        Pre-experiment covariate column

    Returns
    -------
    Dict with original and CUPED-adjusted estimates
    """
    # Calculate theta (optimal coefficient)
    cov_matrix = np.cov(data[outcome_col], data[covariate_col])
    theta = cov_matrix[0, 1] / cov_matrix[1, 1]

    # Create CUPED-adjusted outcome
    data['outcome_cuped'] = data[outcome_col] - theta * (data[covariate_col] - data[covariate_col].mean())

    # Calculate treatment effects
    treated_original = data[data[treatment_col] == 1][outcome_col]
    control_original = data[data[treatment_col] == 0][outcome_col]
    effect_original = treated_original.mean() - control_original.mean()

    treated_cuped = data[data[treatment_col] == 1]['outcome_cuped']
    control_cuped = data[data[treatment_col] == 0]['outcome_cuped']
    effect_cuped = treated_cuped.mean() - control_cuped.mean()

    # Calculate variances
    var_original = treated_original.var() / len(treated_original) + control_original.var() / len(control_original)
    var_cuped = treated_cuped.var() / len(treated_cuped) + control_cuped.var() / len(control_cuped)

    # Calculate p-values
    se_original = np.sqrt(var_original)
    se_cuped = np.sqrt(var_cuped)

    z_original = effect_original / se_original if se_original > 0 else 0
    z_cuped = effect_cuped / se_cuped if se_cuped > 0 else 0

    p_original = 2 * (1 - norm.cdf(abs(z_original)))
    p_cuped = 2 * (1 - norm.cdf(abs(z_cuped)))

    return {
        'effect_original': effect_original,
        'effect_cuped': effect_cuped,
        'se_original': se_original,
        'se_cuped': se_cuped,
        'variance_reduction': 1 - var_cuped / var_original if var_original > 0 else 0,
        'p_value_original': p_original,
        'p_value_cuped': p_cuped,
        'theta': theta
    }


def bayesian_ab_test(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    prior_alpha: float = 1,
    prior_beta: float = 1,
    n_simulations: int = 100000
) -> Dict[str, float]:
    """
    Perform Bayesian A/B test with Beta-Binomial model.

    Parameters
    ----------
    successes_a : int
        Number of successes in group A
    trials_a : int
        Number of trials in group A
    successes_b : int
        Number of successes in group B
    trials_b : int
        Number of trials in group B
    prior_alpha : float
        Beta prior alpha parameter
    prior_beta : float
        Beta prior beta parameter
    n_simulations : int
        Number of Monte Carlo simulations

    Returns
    -------
    Dict with Bayesian test results
    """
    # Posterior parameters
    posterior_alpha_a = prior_alpha + successes_a
    posterior_beta_a = prior_beta + trials_a - successes_a

    posterior_alpha_b = prior_alpha + successes_b
    posterior_beta_b = prior_beta + trials_b - successes_b

    # Sample from posteriors
    samples_a = np.random.beta(posterior_alpha_a, posterior_beta_a, n_simulations)
    samples_b = np.random.beta(posterior_alpha_b, posterior_beta_b, n_simulations)

    # Calculate probabilities
    prob_b_better = np.mean(samples_b > samples_a)
    prob_a_better = 1 - prob_b_better

    # Calculate expected lift
    lift_samples = (samples_b - samples_a) / samples_a
    expected_lift = np.mean(lift_samples)
    lift_ci = np.percentile(lift_samples, [2.5, 97.5])

    # Calculate risk (expected loss if wrong)
    risk_choosing_a = np.mean(np.maximum(samples_b - samples_a, 0))
    risk_choosing_b = np.mean(np.maximum(samples_a - samples_b, 0))

    return {
        'prob_b_better': prob_b_better,
        'prob_a_better': prob_a_better,
        'expected_lift': expected_lift,
        'lift_ci_lower': lift_ci[0],
        'lift_ci_upper': lift_ci[1],
        'risk_choosing_a': risk_choosing_a,
        'risk_choosing_b': risk_choosing_b,
        'recommendation': 'B' if prob_b_better > 0.95 else ('A' if prob_a_better > 0.95 else 'Continue testing')
    }


if __name__ == "__main__":
    # Example usage
    from .pipeline import simulate_ab_data

    print("Testing statistical robustness functions...")

    # Generate sample data
    df = simulate_ab_data(n_users=1000)

    # 1. Bootstrap confidence intervals
    def ate_statistic(data):
        treated = data[data['T'] == 1]['y'].mean()
        control = data[data['T'] == 0]['y'].mean()
        return treated - control

    boot_result = bootstrap_ci(df, ate_statistic, n_bootstrap=1000)
    print(f"\nBootstrap CI: [{boot_result.ci_lower:.4f}, {boot_result.ci_upper:.4f}]")

    # 2. Permutation test
    perm_result = permutation_test(df, 'T', 'y', n_permutations=1000)
    print(f"Permutation test p-value: {perm_result.p_value:.4f}")

    # 3. Sample ratio mismatch
    srm = sample_ratio_mismatch_test(
        n_treatment=df['T'].sum(),
        n_control=len(df) - df['T'].sum()
    )
    print(f"SRM test: {srm['warning'] or 'No issues detected'}")

    # 4. Power analysis
    power = power_analysis(effect_size=0.2, sample_size=500, alpha=0.05)
    print(f"Power for n=500, d=0.2: {power.power:.3f}")

    print("\nAll statistical tests completed successfully!")