"""
Comprehensive test suite for ab_glm.pipeline module.

Tests all functions with normal operations, edge cases, error handling,
and statistical correctness validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

from ab_glm.pipeline import (
    ABResults,
    _get_link,
    brier_score,
    fit_binomial_glm,
    marginal_effects_ate_and_rr,
    run_pipeline,
    simulate_ab_data,
)


class TestSimulateABData:
    """Test suite for simulate_ab_data function."""

    def test_basic_simulation(self):
        """Test basic data simulation with default parameters."""
        df = simulate_ab_data(n_users=100, sessions_per_user=(1, 3), seed=42)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"user_id", "T", "country_EU", "device_mobile", "prior_views", "y"}
        assert set(df.columns) == expected_cols

        # Check dimensions
        assert df["user_id"].nunique() == 100
        assert len(df) >= 100  # At least 1 session per user
        assert len(df) <= 300  # At most 3 sessions per user

        # Check value ranges
        assert df["T"].isin([0, 1]).all()
        assert df["country_EU"].isin([0, 1]).all()
        assert df["device_mobile"].isin([0, 1]).all()
        assert df["y"].isin([0, 1]).all()
        assert (df["prior_views"] >= 0).all()

        # Check treatment is assigned at user level (not varying within user)
        user_treatments = df.groupby("user_id")["T"].nunique()
        assert (user_treatments == 1).all()

    def test_deterministic_seed(self):
        """Test that same seed produces identical data."""
        df1 = simulate_ab_data(n_users=50, seed=123)
        df2 = simulate_ab_data(n_users=50, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        """Test that different seeds produce different data."""
        df1 = simulate_ab_data(n_users=50, seed=1)
        df2 = simulate_ab_data(n_users=50, seed=2)
        assert not df1.equals(df2)

    def test_single_session_per_user(self):
        """Test simulation with exactly 1 session per user."""
        df = simulate_ab_data(n_users=100, sessions_per_user=(1, 1), seed=42)
        assert len(df) == 100
        assert df["user_id"].nunique() == 100

    def test_many_sessions_per_user(self):
        """Test simulation with many sessions per user."""
        df = simulate_ab_data(n_users=10, sessions_per_user=(10, 20), seed=42)
        assert len(df) >= 100  # At least 10*10
        assert len(df) <= 200  # At most 10*20

        # Check each user has correct number of sessions
        session_counts = df["user_id"].value_counts()
        assert (session_counts >= 10).all()
        assert (session_counts <= 20).all()

    def test_treatment_balance(self):
        """Test that treatment assignment is approximately balanced."""
        df = simulate_ab_data(n_users=1000, seed=42)
        treatment_rate = df.groupby("user_id")["T"].first().mean()
        # Should be approximately 50/50 with reasonable tolerance
        assert 0.45 <= treatment_rate <= 0.55

    def test_invalid_n_users(self):
        """Test error handling for invalid n_users."""
        with pytest.raises(ValueError, match="n_users must be positive"):
            simulate_ab_data(n_users=0)

        with pytest.raises(ValueError, match="n_users must be positive"):
            simulate_ab_data(n_users=-1)

    def test_invalid_sessions_per_user(self):
        """Test error handling for invalid sessions_per_user."""
        with pytest.raises(ValueError, match="sessions_per_user must be like"):
            simulate_ab_data(sessions_per_user=(0, 5))

        with pytest.raises(ValueError, match="sessions_per_user must be like"):
            simulate_ab_data(sessions_per_user=(5, 3))

        with pytest.raises(ValueError, match="sessions_per_user must be like"):
            simulate_ab_data(sessions_per_user=(-1, 5))


class TestGetLink:
    """Test suite for _get_link function."""

    def test_logit_link(self):
        """Test logit link retrieval."""
        link = _get_link("logit")
        assert isinstance(link, sm.families.links.Logit)

    def test_probit_link(self):
        """Test probit link retrieval."""
        link = _get_link("probit")
        assert isinstance(link, sm.families.links.Probit)

    def test_invalid_link(self):
        """Test error handling for unsupported link."""
        with pytest.raises(ValueError, match="Unsupported link"):
            _get_link("invalid")  # type: ignore

        with pytest.raises(ValueError, match="Unsupported link"):
            _get_link("cloglog")  # type: ignore


class TestFitBinomialGLM:
    """Test suite for fit_binomial_glm function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return simulate_ab_data(n_users=100, seed=42)

    def test_basic_fit_logit(self, sample_data):
        """Test basic GLM fitting with logit link."""
        glm_unfitted, glm_duplicate, df_model, res_robust = fit_binomial_glm(
            sample_data, link="logit"
        )

        # Check return types
        assert isinstance(glm_unfitted, sm.GLM)
        assert isinstance(glm_duplicate, sm.GLM)
        assert isinstance(df_model, pd.DataFrame)
        assert hasattr(res_robust, "params")
        assert hasattr(res_robust, "predict")

        # Check model has expected coefficients
        expected_params = ["Intercept", "T", "country_EU", "device_mobile", "prior_views"]
        assert all(param in res_robust.params.index for param in expected_params)

        # Check predictions are in valid range
        predictions = res_robust.predict(df_model)
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()

    def test_basic_fit_probit(self, sample_data):
        """Test basic GLM fitting with probit link."""
        _, _, df_model, res_robust = fit_binomial_glm(sample_data, link="probit")

        # Check model converged and has coefficients
        assert hasattr(res_robust, "params")
        assert len(res_robust.params) == 5  # Intercept + 4 predictors

        # Check predictions are valid probabilities
        predictions = res_robust.predict(df_model)
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()

    def test_custom_cluster_column(self, sample_data):
        """Test fitting with custom cluster column."""
        sample_data["cluster_id"] = sample_data["user_id"]
        _, _, df_model, res_robust = fit_binomial_glm(
            sample_data, cluster_col="cluster_id"
        )
        assert "cluster_id" in df_model.columns

    def test_missing_columns(self):
        """Test error handling for missing required columns."""
        df = pd.DataFrame({"y": [0, 1], "T": [0, 1]})

        with pytest.raises(ValueError, match="Missing required columns"):
            fit_binomial_glm(df)

    def test_non_binary_treatment(self, sample_data):
        """Test error handling for non-binary treatment."""
        sample_data.loc[0, "T"] = 2

        with pytest.raises(ValueError, match="'T' must be binary"):
            fit_binomial_glm(sample_data)

    def test_non_binary_outcome(self, sample_data):
        """Test error handling for non-binary outcome."""
        sample_data.loc[0, "y"] = 2

        with pytest.raises(ValueError, match="'y' must be binary"):
            fit_binomial_glm(sample_data)

    def test_empty_dataframe(self):
        """Test error handling for empty dataframe."""
        cols = ["y", "T", "country_EU", "device_mobile", "prior_views", "user_id"]
        df = pd.DataFrame(columns=cols)

        with pytest.raises(ValueError, match="No rows left"):
            fit_binomial_glm(df)

    def test_all_na_values(self):
        """Test handling of all NA values."""
        df = pd.DataFrame({
            "y": [np.nan, np.nan],
            "T": [0, 1],
            "country_EU": [0, 1],
            "device_mobile": [0, 1],
            "prior_views": [1, 2],
            "user_id": [0, 1]
        })

        # The function checks for binary y before dropna, so it will fail on NaN
        with pytest.raises(ValueError, match="'y' must be binary"):
            fit_binomial_glm(df)

    def test_partial_na_values(self, sample_data):
        """Test that partial NA values are dropped."""
        original_len = len(sample_data)
        sample_data.loc[0:5, "prior_views"] = np.nan

        _, _, df_model, res_robust = fit_binomial_glm(sample_data)

        # Check that NA rows were dropped
        assert len(df_model) < original_len
        assert df_model["prior_views"].notna().all()


class TestMarginalEffectsATEandRR:
    """Test suite for marginal_effects_ate_and_rr function."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        df = simulate_ab_data(n_users=200, seed=42)
        _, _, df_model, res_robust = fit_binomial_glm(df, link="logit")
        return res_robust, df_model

    def test_basic_computation(self, fitted_model):
        """Test basic ATE and RR computation."""
        res_robust, df_model = fitted_model
        ate_rd, rr, p_treated, p_control = marginal_effects_ate_and_rr(res_robust, df_model)

        # Check return types
        assert isinstance(ate_rd, float)
        assert isinstance(rr, float)
        assert isinstance(p_treated, float)
        assert isinstance(p_control, float)

        # Check value ranges
        assert 0 <= p_treated <= 1
        assert 0 <= p_control <= 1
        assert rr > 0  # Risk ratio must be positive
        assert -1 <= ate_rd <= 1  # ATE is a difference of probabilities

    def test_ate_calculation(self, fitted_model):
        """Test that ATE equals p_treated - p_control."""
        res_robust, df_model = fitted_model
        ate_rd, rr, p_treated, p_control = marginal_effects_ate_and_rr(res_robust, df_model)

        assert_allclose(ate_rd, p_treated - p_control, rtol=1e-10)

    def test_rr_calculation(self, fitted_model):
        """Test that RR equals p_treated / p_control."""
        res_robust, df_model = fitted_model
        ate_rd, rr, p_treated, p_control = marginal_effects_ate_and_rr(res_robust, df_model)

        assert_allclose(rr, p_treated / p_control, rtol=1e-10)

    def test_marginal_predictions_consistency(self, fitted_model):
        """Test that marginal predictions are computed correctly."""
        res_robust, df_model = fitted_model

        # Manual computation
        df1 = df_model.copy()
        df0 = df_model.copy()
        df1["T"] = 1
        df0["T"] = 0

        p1_manual = res_robust.predict(df1)
        p0_manual = res_robust.predict(df0)

        p1_clipped = np.clip(p1_manual, 1e-12, 1 - 1e-12)
        p0_clipped = np.clip(p0_manual, 1e-12, 1 - 1e-12)

        expected_p_treated = float(np.mean(p1_clipped))
        expected_p_control = float(np.mean(p0_clipped))

        # Function computation
        ate_rd, rr, p_treated, p_control = marginal_effects_ate_and_rr(res_robust, df_model)

        assert_allclose(p_treated, expected_p_treated, rtol=1e-10)
        assert_allclose(p_control, expected_p_control, rtol=1e-10)

    def test_with_no_treatment_effect(self):
        """Test with data where treatment has no effect."""
        # Create data with no treatment effect
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "user_id": np.arange(n),
            "T": np.random.binomial(1, 0.5, n),
            "country_EU": np.random.binomial(1, 0.5, n),
            "device_mobile": np.random.binomial(1, 0.5, n),
            "prior_views": np.random.poisson(3, n),
            "y": np.random.binomial(1, 0.3, n)  # Independent of T
        })

        _, _, df_model, res_robust = fit_binomial_glm(df, link="logit")
        ate_rd, rr, p_treated, p_control = marginal_effects_ate_and_rr(res_robust, df_model)

        # ATE should be small and RR close to 1 (but with wider tolerance for random data)
        assert abs(ate_rd) < 0.1  # Small ATE with wider tolerance
        assert 0.7 < rr < 1.3  # RR close to 1 with wider tolerance for random variation


class TestBrierScore:
    """Test suite for brier_score function."""

    def test_perfect_predictions(self):
        """Test Brier score with perfect predictions."""
        y_true = np.array([0, 0, 1, 1], dtype=float)
        p_hat = np.array([0, 0, 1, 1], dtype=float)

        bs = brier_score(y_true, p_hat)
        assert bs == 0.0

    def test_worst_predictions(self):
        """Test Brier score with worst possible predictions."""
        y_true = np.array([0, 0, 1, 1], dtype=float)
        p_hat = np.array([1, 1, 0, 0], dtype=float)

        bs = brier_score(y_true, p_hat)
        assert bs == 1.0

    def test_random_predictions(self):
        """Test Brier score with random predictions."""
        y_true = np.array([0, 1, 1, 0, 1, 0], dtype=float)
        p_hat = np.array([0.2, 0.8, 0.6, 0.3, 0.9, 0.1], dtype=float)

        bs = brier_score(y_true, p_hat)

        # Manual calculation
        expected = np.mean((y_true - p_hat) ** 2)
        assert_allclose(bs, expected, rtol=1e-10)

        # Check reasonable range
        assert 0 <= bs <= 1

    def test_clipping_behavior(self):
        """Test that predictions outside [0,1] are clipped."""
        y_true = np.array([0, 1], dtype=float)
        p_hat = np.array([-0.1, 1.1], dtype=float)

        bs = brier_score(y_true, p_hat)

        # Should clip to [0, 1]
        expected = np.mean((y_true - np.array([0, 1])) ** 2)
        assert bs == expected

    def test_shape_mismatch(self):
        """Test error handling for shape mismatch."""
        y_true = np.array([0, 1, 0])
        p_hat = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="same shape"):
            brier_score(y_true, p_hat)

    def test_non_finite_values(self):
        """Test error handling for non-finite values."""
        y_true = np.array([0, 1, np.nan])
        p_hat = np.array([0.5, 0.5, 0.5])

        with pytest.raises(ValueError, match="finite"):
            brier_score(y_true, p_hat)

        y_true = np.array([0, 1, 0])
        p_hat = np.array([0.5, np.inf, 0.5])

        with pytest.raises(ValueError, match="finite"):
            brier_score(y_true, p_hat)

    def test_constant_predictions(self):
        """Test Brier score with constant predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1], dtype=float)
        p_hat = np.array([0.5] * 6, dtype=float)

        bs = brier_score(y_true, p_hat)
        assert bs == 0.25  # (0.5)^2 for all predictions

    @pytest.mark.parametrize("n", [10, 100, 1000])
    def test_large_arrays(self, n):
        """Test Brier score with arrays of different sizes."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, n).astype(float)
        p_hat = np.random.uniform(0, 1, n)

        bs = brier_score(y_true, p_hat)

        assert 0 <= bs <= 1
        assert isinstance(bs, float)


class TestRunPipeline:
    """Test suite for run_pipeline integration function."""

    def test_run_pipeline_logit(self):
        """Test full pipeline with logit link."""
        results = run_pipeline(link="logit")

        assert isinstance(results, ABResults)
        assert results.link == "logit"

        # Check all fields are populated
        assert results.ate_rd is not None
        assert results.rr is not None
        assert results.p_treated is not None
        assert results.p_control is not None
        assert results.brier is not None
        assert results.n_obs > 0
        assert results.n_users > 0
        assert results.coef_treat is not None
        assert results.robust_se_treat is not None

        # Check value ranges
        assert 0 <= results.p_control <= 1
        assert 0 <= results.p_treated <= 1
        assert results.rr > 0
        assert 0 <= results.brier <= 1
        assert results.n_obs >= results.n_users

    def test_run_pipeline_probit(self):
        """Test full pipeline with probit link."""
        results = run_pipeline(link="probit")

        assert isinstance(results, ABResults)
        assert results.link == "probit"

        # Check value validity
        assert 0 <= results.p_control <= 1
        assert 0 <= results.p_treated <= 1
        assert results.rr > 0
        assert 0 <= results.brier <= 1

    def test_pipeline_consistency(self):
        """Test that pipeline results are internally consistent."""
        results = run_pipeline(link="logit")

        # ATE should equal p_treated - p_control
        expected_ate = results.p_treated - results.p_control
        assert_allclose(results.ate_rd, expected_ate, rtol=1e-10)

        # RR should equal p_treated / p_control
        expected_rr = results.p_treated / results.p_control
        assert_allclose(results.rr, expected_rr, rtol=1e-10)

    def test_pipeline_reproducibility(self):
        """Test that pipeline produces consistent results with same seed."""
        # Since simulate_ab_data uses a fixed seed in run_pipeline,
        # results should be identical
        results1 = run_pipeline(link="logit")
        results2 = run_pipeline(link="logit")

        # Check key metrics are identical
        assert results1.ate_rd == results2.ate_rd
        assert results1.rr == results2.rr
        assert results1.p_treated == results2.p_treated
        assert results1.p_control == results2.p_control
        assert results1.brier == results2.brier

    def test_logit_vs_probit_differences(self):
        """Test that logit and probit produce different but reasonable results."""
        results_logit = run_pipeline(link="logit")
        results_probit = run_pipeline(link="probit")

        # Results should be different
        assert results_logit.coef_treat != results_probit.coef_treat

        # But should be qualitatively similar (same direction)
        assert np.sign(results_logit.ate_rd) == np.sign(results_probit.ate_rd)

        # Both should indicate positive treatment effect if that's what's simulated
        assert results_logit.rr > 1.0
        assert results_probit.rr > 1.0


class TestABResults:
    """Test suite for ABResults dataclass."""

    def test_dataclass_creation(self):
        """Test creating ABResults instance."""
        results = ABResults(
            link="logit",
            ate_rd=0.05,
            rr=1.1,
            p_treated=0.15,
            p_control=0.10,
            brier=0.08,
            n_obs=1000,
            n_users=500,
            robust_se_treat=0.02,
            coef_treat=0.3
        )

        assert results.link == "logit"
        assert results.ate_rd == 0.05
        assert results.rr == 1.1
        assert results.p_treated == 0.15
        assert results.p_control == 0.10
        assert results.brier == 0.08
        assert results.n_obs == 1000
        assert results.n_users == 500
        assert results.robust_se_treat == 0.02
        assert results.coef_treat == 0.3

    def test_optional_fields(self):
        """Test that optional fields can be None."""
        results = ABResults(
            link="probit",
            ate_rd=0.05,
            rr=1.1,
            p_treated=0.15,
            p_control=0.10,
            brier=0.08,
            n_obs=1000,
            n_users=500,
            robust_se_treat=None,
            coef_treat=None
        )

        assert results.robust_se_treat is None
        assert results.coef_treat is None


class TestStatisticalCorrectness:
    """Test suite for verifying statistical correctness of implementations."""

    def test_logit_link_function(self):
        """Test that logit link produces correct probabilities."""
        # Create data with more variation to avoid perfect separation
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "user_id": np.arange(n),
            "T": np.random.binomial(1, 0.5, n),
            "country_EU": np.random.binomial(1, 0.5, n),
            "device_mobile": np.random.binomial(1, 0.5, n),
            "prior_views": np.random.poisson(3, n),
        })
        # Generate outcome with some noise to avoid perfect separation
        logit = -2 + 0.5 * df["T"] + 0.2 * df["country_EU"]
        p = 1 / (1 + np.exp(-logit))
        df["y"] = np.random.binomial(1, p)

        _, _, df_model, res_robust = fit_binomial_glm(df, link="logit")
        predictions = res_robust.predict(df_model)

        # All predictions should be in [0, 1]
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()

    def test_treatment_effect_direction(self):
        """Test that positive treatment increases probability."""
        # Create data with strong positive treatment effect
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "user_id": np.arange(n),
            "T": np.random.binomial(1, 0.5, n),
            "country_EU": np.random.binomial(1, 0.5, n),
            "device_mobile": np.random.binomial(1, 0.5, n),
            "prior_views": np.random.poisson(3, n),
        })
        # Create outcome with positive treatment effect
        logit = -2 + 1.5 * df["T"] + 0.2 * df["country_EU"]
        p = 1 / (1 + np.exp(-logit))
        df["y"] = np.random.binomial(1, p)

        _, _, df_model, res_robust = fit_binomial_glm(df, link="logit")

        # Treatment coefficient should be positive
        assert res_robust.params["T"] > 0

        # ATE should be positive
        ate_rd, rr, p_treated, p_control = marginal_effects_ate_and_rr(res_robust, df_model)
        assert ate_rd > 0
        assert rr > 1

    def test_covariate_adjustment_improves_precision(self):
        """Test that including covariates improves model fit."""
        df = simulate_ab_data(n_users=500, seed=42)

        # Fit with covariates
        formula_full = "y ~ T + country_EU + device_mobile + prior_views"
        glm_full = sm.GLM.from_formula(
            formula_full,
            data=df,
            family=sm.families.Binomial()
        )
        res_full = glm_full.fit()

        # Fit without covariates (T only)
        formula_simple = "y ~ T"
        glm_simple = sm.GLM.from_formula(
            formula_simple,
            data=df,
            family=sm.families.Binomial()
        )
        res_simple = glm_simple.fit()

        # Full model should have better fit (lower deviance)
        assert res_full.deviance < res_simple.deviance


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_single_observation_per_user(self):
        """Test with exactly one observation per user."""
        df = simulate_ab_data(n_users=100, sessions_per_user=(1, 1), seed=42)
        _, _, df_model, res_robust = fit_binomial_glm(df)

        assert len(df_model) == df_model["user_id"].nunique()

    def test_all_control_group(self):
        """Test when all users are in control group."""
        df = simulate_ab_data(n_users=50, seed=42)
        df["T"] = 0  # Set all to control

        # This will cause a singular matrix error since T has no variation
        # We should expect this to fail with cluster-robust SE
        with pytest.raises(np.linalg.LinAlgError):
            fit_binomial_glm(df)

    def test_all_treatment_group(self):
        """Test when all users are in treatment group."""
        df = simulate_ab_data(n_users=50, seed=42)
        df["T"] = 1  # Set all to treatment

        # This will cause a singular matrix error since T has no variation
        # We should expect this to fail with cluster-robust SE
        with pytest.raises(np.linalg.LinAlgError):
            fit_binomial_glm(df)

    def test_no_variation_in_outcome(self):
        """Test when outcome has no variation."""
        df = simulate_ab_data(n_users=50, seed=42)
        df["y"] = 1  # All successes

        # This might cause convergence issues but shouldn't crash
        _, _, df_model, res_robust = fit_binomial_glm(df)

    def test_extreme_covariate_values(self):
        """Test with extreme covariate values."""
        df = simulate_ab_data(n_users=50, seed=42)
        df["prior_views"] = 1000  # Very large values

        _, _, df_model, res_robust = fit_binomial_glm(df)

        # Should handle without numerical issues
        predictions = res_robust.predict(df_model)
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()

    def test_minimum_sample_size(self):
        """Test with minimum viable sample size."""
        df = pd.DataFrame({
            "user_id": [0, 1, 2, 3, 4, 5],
            "T": [0, 0, 0, 1, 1, 1],
            "country_EU": [0, 1, 0, 1, 0, 1],
            "device_mobile": [0, 0, 1, 1, 0, 1],
            "prior_views": [1, 2, 3, 1, 2, 3],
            "y": [0, 0, 1, 1, 0, 1]
        })

        # Should fit with minimum degrees of freedom
        _, _, df_model, res_robust = fit_binomial_glm(df)
        assert len(res_robust.params) == 5


class TestPerformance:
    """Test suite for performance and scalability."""

    @pytest.mark.parametrize("n_users", [100, 500, 1000])
    def test_scalability(self, n_users):
        """Test that functions scale reasonably with data size."""
        df = simulate_ab_data(n_users=n_users, sessions_per_user=(2, 5), seed=42)

        # Should complete without error for different sizes
        _, _, df_model, res_robust = fit_binomial_glm(df)
        ate_rd, rr, p_treated, p_control = marginal_effects_ate_and_rr(res_robust, df_model)

        assert isinstance(ate_rd, float)
        assert isinstance(rr, float)

    def test_large_cluster_count(self):
        """Test with many clusters (users)."""
        df = simulate_ab_data(n_users=2000, sessions_per_user=(1, 2), seed=42)

        # Should handle cluster-robust SE computation efficiently
        _, _, df_model, res_robust = fit_binomial_glm(df)

        # Check that the model fits successfully with many clusters
        assert hasattr(res_robust, "params")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])