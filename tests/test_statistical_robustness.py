"""
Tests for statistical robustness modules.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from ab_glm import simulate_ab_data
from ab_glm.statistical_tests import (
    BootstrapResult,
    PermutationTestResult,
    PowerAnalysisResult,
    bootstrap_ci,
    permutation_test,
    multiple_testing_correction,
    sample_ratio_mismatch_test,
    power_analysis,
    sequential_testing,
    heterogeneous_treatment_effects,
    variance_reduction_cuped,
    bayesian_ab_test,
)
from ab_glm.causal_inference import (
    estimate_cate_with_causal_forest,
    double_ml_ate,
    propensity_score_matching,
    check_covariate_balance,
    regression_discontinuity,
    sensitivity_analysis_unobserved_confounding,
)
from ab_glm.robustness_checks import (
    check_model_assumptions,
    robust_estimation_with_outliers,
    detect_outliers,
    sensitivity_to_specification,
    meta_analysis,
    placebo_test,
    complete_robustness_analysis,
)


class TestStatisticalTests:
    """Test statistical test functions."""

    def test_bootstrap_ci(self):
        """Test bootstrap confidence intervals."""
        np.random.seed(42)
        df = simulate_ab_data(n_users=100)

        def test_statistic(data):
            return data['y'].mean()

        result = bootstrap_ci(
            df, test_statistic, n_bootstrap=100, confidence_level=0.95, show_progress=False
        )

        assert isinstance(result, BootstrapResult)
        assert result.ci_lower < result.estimate < result.ci_upper
        assert result.confidence_level == 0.95
        assert result.n_iterations == 100

    def test_permutation_test(self):
        """Test permutation testing."""
        np.random.seed(42)
        df = simulate_ab_data(n_users=100)

        result = permutation_test(
            df, 'T', 'y', n_permutations=100, show_progress=False
        )

        assert isinstance(result, PermutationTestResult)
        assert 0 <= result.p_value <= 1
        assert len(result.null_distribution) == 100
        assert result.n_permutations == 100

    def test_multiple_testing_correction(self):
        """Test multiple testing corrections."""
        p_values = [0.01, 0.04, 0.03, 0.20, 0.001]

        result = multiple_testing_correction(p_values, method='fdr_bh', alpha=0.05)

        assert 'p_values_adjusted' in result
        assert 'reject' in result
        assert len(result['p_values_adjusted']) == len(p_values)
        assert all(result['p_values_adjusted'] >= result['p_values_original'])

    def test_sample_ratio_mismatch(self):
        """Test SRM detection."""
        result = sample_ratio_mismatch_test(
            n_treatment=500, n_control=500, expected_ratio=0.5
        )

        assert 'observed_ratio' in result
        assert 'p_value_binomial' in result
        assert result['observed_ratio'] == 0.5
        assert not result['srm_detected']  # Should not detect SRM for 50/50 split

        # Test with mismatch
        result_mismatch = sample_ratio_mismatch_test(
            n_treatment=600, n_control=400, expected_ratio=0.5
        )
        assert result_mismatch['observed_ratio'] == 0.6

    def test_power_analysis(self):
        """Test power analysis calculations."""
        # Calculate power
        result = power_analysis(
            effect_size=0.2, sample_size=500, alpha=0.05
        )
        assert isinstance(result, PowerAnalysisResult)
        assert 0 <= result.power <= 1

        # Calculate sample size
        result = power_analysis(
            effect_size=0.2, power=0.8, alpha=0.05
        )
        assert result.sample_size > 0

        # Calculate effect size
        result = power_analysis(
            sample_size=500, power=0.8, alpha=0.05
        )
        assert result.effect_size > 0

    def test_sequential_testing(self):
        """Test sequential testing with early stopping."""
        np.random.seed(42)
        df = simulate_ab_data(n_users=500)

        result = sequential_testing(
            df, 'T', 'y', alpha=0.05, power=0.8, effect_size=0.2
        )

        assert 'stopped_early' in result
        assert 'final_p_value' in result
        assert 'boundaries' in result
        assert result['decision'] in ['reject', 'fail to reject']

    def test_heterogeneous_treatment_effects(self):
        """Test HTE analysis."""
        np.random.seed(42)
        df = simulate_ab_data(n_users=200)

        result = heterogeneous_treatment_effects(
            df, 'T', 'y', subgroup_cols=['country_EU', 'device_mobile'], min_samples=10
        )

        assert isinstance(result, pd.DataFrame)
        assert 'subgroup_variable' in result.columns
        assert 'effect' in result.columns
        assert 'p_value' in result.columns

    def test_variance_reduction_cuped(self):
        """Test CUPED variance reduction."""
        np.random.seed(42)
        df = simulate_ab_data(n_users=200)
        df['prior_outcome'] = df['y'] + np.random.normal(0, 0.1, len(df))

        result = variance_reduction_cuped(
            df, 'T', 'y', 'prior_outcome'
        )

        assert 'effect_original' in result
        assert 'effect_cuped' in result
        assert 'variance_reduction' in result
        assert result['variance_reduction'] >= 0

    def test_bayesian_ab_test(self):
        """Test Bayesian A/B testing."""
        result = bayesian_ab_test(
            successes_a=45, trials_a=100,
            successes_b=55, trials_b=100,
            n_simulations=10000
        )

        assert 'prob_b_better' in result
        assert 'expected_lift' in result
        assert result['prob_b_better'] + result['prob_a_better'] == pytest.approx(1.0, rel=0.01)
        assert result['recommendation'] in ['A', 'B', 'Continue testing']


class TestCausalInference:
    """Test causal inference functions."""

    def test_estimate_cate(self):
        """Test CATE estimation with causal forest."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n)
        })
        treatment = np.random.binomial(1, 0.5, n)
        outcome = 0.5 + 0.2 * treatment + 0.1 * X['x1'] + np.random.normal(0, 0.1, n)

        result = estimate_cate_with_causal_forest(
            X, treatment, outcome, n_estimators=10, cross_fitting_folds=2
        )

        assert len(result.cate_estimates) == n
        assert result.ate is not None
        assert result.ate_std > 0

    def test_double_ml(self):
        """Test Double ML ATE estimation."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n)
        })
        treatment = np.random.binomial(1, 0.5, n)
        outcome = 0.5 + 0.2 * treatment + 0.1 * X['x1'] + np.random.normal(0, 0.1, n)

        result = double_ml_ate(X, treatment, outcome, ml_method='linear', n_folds=2)

        assert 'ate' in result
        assert 'std_error' in result
        assert 'p_value' in result
        assert result['ci_lower'] < result['ate'] < result['ci_upper']

    def test_propensity_score_matching(self):
        """Test propensity score matching."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n)
        })
        treatment = (X['x1'] > 0).astype(int)
        outcome = 0.5 + 0.2 * treatment + 0.1 * X['x1'] + np.random.normal(0, 0.1, n)

        result = propensity_score_matching(
            X, treatment, outcome, caliper=0.1, n_matches=1
        )

        assert result.ate is not None
        assert result.std_error > 0
        assert len(result.matched_pairs) > 0
        assert isinstance(result.balance_statistics, pd.DataFrame)

    def test_covariate_balance(self):
        """Test covariate balance checking."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n)
        })
        treatment = np.random.binomial(1, 0.5, n)

        result = check_covariate_balance(X, treatment)

        assert isinstance(result, pd.DataFrame)
        assert 'std_mean_diff' in result.columns
        assert 'balanced' in result.columns

    def test_regression_discontinuity(self):
        """Test regression discontinuity design."""
        np.random.seed(42)
        n = 200
        running_var = np.random.uniform(-1, 1, n)
        treatment = (running_var >= 0).astype(int)
        outcome = 0.5 + 0.3 * treatment + 0.2 * running_var + np.random.normal(0, 0.1, n)

        result = regression_discontinuity(
            running_var, outcome, cutoff=0, polynomial_order=1
        )

        assert 'estimate' in result
        assert 'std_error' in result
        assert 'bandwidth' in result
        assert result['n_effective'] > 0

    def test_sensitivity_analysis_confounding(self):
        """Test sensitivity to unobserved confounding."""
        result = sensitivity_analysis_unobserved_confounding(
            ate_estimate=0.1, std_error=0.02, gamma_range=np.linspace(1, 2, 5)
        )

        assert isinstance(result, pd.DataFrame)
        assert 'gamma' in result.columns
        assert 'lower_bound' in result.columns
        assert len(result) == 5


class TestRobustnessChecks:
    """Test robustness check functions."""

    @patch('ab_glm.robustness_checks.fit_binomial_glm')
    @patch('ab_glm.robustness_checks.marginal_effects_ate_and_rr')
    def test_check_model_assumptions(self, mock_marginal, mock_fit):
        """Test model assumption checking."""
        np.random.seed(42)
        n = 100

        # Create mock model results
        mock_model = MagicMock()
        mock_model.resid = np.random.normal(0, 1, n)
        mock_model.converged = True

        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n)
        })
        y = np.random.binomial(1, 0.5, n)

        # Mock get_influence to avoid statsmodels complexity
        mock_influence = MagicMock()
        mock_influence.cooks_distance = (np.random.uniform(0, 0.01, n), None)
        mock_model.get_influence.return_value = mock_influence

        result = check_model_assumptions(mock_model, X, y)

        assert 'jarque_bera' in result
        assert 'warnings' in result
        assert isinstance(result['warnings'], list)

    def test_detect_outliers(self):
        """Test outlier detection methods."""
        np.random.seed(42)
        df = simulate_ab_data(n_users=100)

        # IQR method
        outliers_iqr = detect_outliers(df, method='iqr')
        assert len(outliers_iqr) == len(df)
        assert outliers_iqr.dtype == bool

        # Z-score method
        outliers_zscore = detect_outliers(df, method='zscore')
        assert len(outliers_zscore) == len(df)

    def test_sensitivity_to_specification(self):
        """Test specification sensitivity analysis."""
        np.random.seed(42)
        df = simulate_ab_data(n_users=100)

        result = sensitivity_to_specification(df)

        assert isinstance(result, pd.DataFrame)
        assert 'specification' in result.columns
        assert 'ate' in result.columns
        assert len(result) > 0

    def test_meta_analysis(self):
        """Test meta-analysis."""
        experiments = [
            {'effect': 0.05, 'se': 0.02, 'n': 1000},
            {'effect': 0.03, 'se': 0.015, 'n': 1500},
            {'effect': 0.06, 'se': 0.025, 'n': 800}
        ]

        # Fixed effects
        result_fixed = meta_analysis(experiments, method='fixed_effects')
        assert 'pooled_effect' in result_fixed
        assert 'I_squared' in result_fixed
        assert result_fixed['n_experiments'] == 3

        # Random effects
        result_random = meta_analysis(experiments, method='random_effects')
        assert 'pooled_effect' in result_random

    @patch('ab_glm.robustness_checks.fit_binomial_glm')
    @patch('ab_glm.robustness_checks.marginal_effects_ate_and_rr')
    def test_placebo_test(self, mock_marginal, mock_fit):
        """Test placebo test."""
        np.random.seed(42)
        df = simulate_ab_data(n_users=100)
        df['placebo_outcome'] = np.random.binomial(1, 0.5, len(df))

        # Setup mocks
        mock_fit.return_value = (None, None, df, MagicMock())
        mock_marginal.return_value = (0.01, 1.01, None, None)
        mock_fit.return_value[3].bse = pd.Series([0.01, 0.02])
        mock_fit.return_value[3].converged = True

        result = placebo_test(df, 'placebo_outcome')

        assert 'placebo_effect' in result
        assert 'test_passed' in result

    def test_robust_estimation_with_outliers(self):
        """Test robust estimation with outlier handling."""
        np.random.seed(42)
        df = simulate_ab_data(n_users=100)

        result = robust_estimation_with_outliers(
            df, outlier_methods=['iqr', 'zscore']
        )

        assert result.baseline_estimate is not None
        assert 'ate_iqr' in result.robust_estimates
        assert 'ate_zscore' in result.robust_estimates
        assert result.sensitivity_range[0] <= result.sensitivity_range[1]

    def test_complete_robustness_analysis(self):
        """Test complete robustness analysis suite."""
        np.random.seed(42)
        df = simulate_ab_data(n_users=100)

        result = complete_robustness_analysis(df)

        assert 'baseline' in result
        assert 'summary' in result
        assert 'all_checks_passed' in result['summary']
        assert isinstance(result['summary']['warnings'], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])