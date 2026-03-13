from __future__ import annotations

from .pipeline import (
    ABResults,
    LinkName,
    brier_score,
    fit_binomial_glm,
    marginal_effects_ate_and_rr,
    run_pipeline,
    simulate_ab_data,
)

# Import enhanced versions (optional - users can import explicitly)
try:
    from .enhanced_pipeline import (
        DataValidationError,
        EnhancedABResults,
        ModelConvergenceError,
        calculate_additional_metrics,
        calculate_confidence_intervals,
        fit_binomial_glm_enhanced,
        run_enhanced_pipeline,
        validate_data,
    )
    from .performance import (
        BenchmarkResult,
        benchmark_data_generation,
        benchmark_model_fitting,
        create_performance_report,
        profile_function,
        test_scalability,
    )
    from .security import (
        SecurityError,
        check_data_integrity,
        create_safe_summary,
        detect_pii,
        hash_pii_columns,
        sanitize_column_names,
        validate_ab_test_data,
        validate_data_types,
        validate_file_path,
    )
    from .statistical_tests import (
        BootstrapResult,
        PermutationTestResult,
        PowerAnalysisResult,
        bayesian_ab_test,
        bootstrap_ci,
        heterogeneous_treatment_effects,
        multiple_testing_correction,
        permutation_test,
        power_analysis,
        sample_ratio_mismatch_test,
        sequential_testing,
        variance_reduction_cuped,
    )
    from .causal_inference import (
        CATEResult,
        PropensityScoreResult,
        check_covariate_balance,
        double_ml_ate,
        estimate_cate_with_causal_forest,
        propensity_score_matching,
        regression_discontinuity,
        sensitivity_analysis_unobserved_confounding,
    )
    from .robustness_checks import (
        RobustnessResult,
        check_model_assumptions,
        complete_robustness_analysis,
        detect_outliers,
        meta_analysis,
        placebo_test,
        robust_estimation_with_outliers,
        sensitivity_to_specification,
    )

    __all__ = [
        # Core functions
        "ABResults",
        "LinkName",
        "simulate_ab_data",
        "fit_binomial_glm",
        "marginal_effects_ate_and_rr",
        "brier_score",
        "run_pipeline",
        # Enhanced functions
        "EnhancedABResults",
        "fit_binomial_glm_enhanced",
        "run_enhanced_pipeline",
        "validate_data",
        "calculate_confidence_intervals",
        "calculate_additional_metrics",
        "DataValidationError",
        "ModelConvergenceError",
        # Performance utilities
        "benchmark_data_generation",
        "benchmark_model_fitting",
        "profile_function",
        "test_scalability",
        "create_performance_report",
        "BenchmarkResult",
        # Security functions
        "SecurityError",
        "check_data_integrity",
        "create_safe_summary",
        "detect_pii",
        "hash_pii_columns",
        "sanitize_column_names",
        "validate_ab_test_data",
        "validate_data_types",
        "validate_file_path",
        # Statistical tests
        "BootstrapResult",
        "PermutationTestResult",
        "PowerAnalysisResult",
        "bayesian_ab_test",
        "bootstrap_ci",
        "heterogeneous_treatment_effects",
        "multiple_testing_correction",
        "permutation_test",
        "power_analysis",
        "sample_ratio_mismatch_test",
        "sequential_testing",
        "variance_reduction_cuped",
        # Causal inference
        "CATEResult",
        "PropensityScoreResult",
        "check_covariate_balance",
        "double_ml_ate",
        "estimate_cate_with_causal_forest",
        "propensity_score_matching",
        "regression_discontinuity",
        "sensitivity_analysis_unobserved_confounding",
        # Robustness checks
        "RobustnessResult",
        "check_model_assumptions",
        "complete_robustness_analysis",
        "detect_outliers",
        "meta_analysis",
        "placebo_test",
        "robust_estimation_with_outliers",
        "sensitivity_to_specification",
    ]

except ModuleNotFoundError as exc:
    # Do not hide internal package import errors.
    if exc.name and exc.name.startswith("ab_glm"):
        raise

    # If enhanced modules fail to import (missing dependencies),
    # just export core functions
    __all__ = [
        "ABResults",
        "LinkName",
        "simulate_ab_data",
        "fit_binomial_glm",
        "marginal_effects_ate_and_rr",
        "brier_score",
        "run_pipeline",
    ]

__version__ = "0.4.0"
