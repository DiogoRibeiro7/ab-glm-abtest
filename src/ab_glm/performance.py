"""
Performance benchmarking and profiling utilities for ab-glm-abtest.

This module provides tools for:
- Benchmarking function performance
- Memory profiling
- Scalability testing
- Performance optimization recommendations
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from .pipeline import fit_binomial_glm, marginal_effects_ate_and_rr, simulate_ab_data


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    function_name: str
    n_samples: int
    n_users: int
    execution_time: float
    memory_used_mb: float
    memory_peak_mb: float
    throughput_samples_per_sec: float
    errors: List[str]

    def __str__(self) -> str:
        return (
            f"Benchmark: {self.function_name}\n"
            f"  Samples: {self.n_samples:,}\n"
            f"  Users: {self.n_users:,}\n"
            f"  Time: {self.execution_time:.3f}s\n"
            f"  Memory: {self.memory_used_mb:.1f}MB (peak: {self.memory_peak_mb:.1f}MB)\n"
            f"  Throughput: {self.throughput_samples_per_sec:.0f} samples/sec"
        )


@contextmanager
def timer():
    """Context manager for timing code execution."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    return end - start


@contextmanager
def memory_tracker():
    """Context manager for tracking memory usage."""
    process = psutil.Process()

    # Force garbage collection before measurement
    gc.collect()

    # Get initial memory
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Track peak memory
    peak_memory = mem_before

    def update_peak():
        nonlocal peak_memory
        current = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current)

    yield update_peak

    # Final measurement
    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024

    return {
        'used': mem_after - mem_before,
        'peak': peak_memory - mem_before,
        'final': mem_after
    }


def benchmark_data_generation(sizes: List[int]) -> pd.DataFrame:
    """
    Benchmark data generation performance at different scales.

    Parameters
    ----------
    sizes : List[int]
        List of user counts to test

    Returns
    -------
    pd.DataFrame
        Benchmark results
    """
    results = []

    for n_users in sizes:
        # Time the function
        start = time.perf_counter()
        df = simulate_ab_data(n_users=n_users, sessions_per_user=(2, 5))
        elapsed = time.perf_counter() - start

        results.append({
            'n_users': n_users,
            'n_rows': len(df),
            'time_seconds': elapsed,
            'rows_per_second': len(df) / elapsed,
            'users_per_second': n_users / elapsed
        })

    return pd.DataFrame(results)


def benchmark_model_fitting(sizes: List[int]) -> pd.DataFrame:
    """
    Benchmark model fitting performance at different scales.

    Parameters
    ----------
    sizes : List[int]
        List of user counts to test

    Returns
    -------
    pd.DataFrame
        Benchmark results
    """
    results = []

    for n_users in sizes:
        # Generate data
        df = simulate_ab_data(n_users=n_users, sessions_per_user=(2, 4))

        # Track memory and time
        process = psutil.Process()
        gc.collect()

        mem_before = process.memory_info().rss / 1024 / 1024

        try:
            start = time.perf_counter()

            # Fit model
            _, _, df_model, results_glm = fit_binomial_glm(df)

            # Calculate effects
            ate, rr, _, _ = marginal_effects_ate_and_rr(results_glm, df_model)

            elapsed = time.perf_counter() - start

            gc.collect()
            mem_after = process.memory_info().rss / 1024 / 1024

            results.append({
                'n_users': n_users,
                'n_observations': len(df),
                'fit_time_seconds': elapsed,
                'memory_used_mb': mem_after - mem_before,
                'ate': ate,
                'risk_ratio': rr,
                'observations_per_second': len(df) / elapsed
            })

        except Exception as e:
            results.append({
                'n_users': n_users,
                'n_observations': len(df),
                'fit_time_seconds': None,
                'memory_used_mb': None,
                'ate': None,
                'risk_ratio': None,
                'observations_per_second': None,
                'error': str(e)
            })

    return pd.DataFrame(results)


def profile_function(
    func: Callable,
    *args,
    n_runs: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Profile a function's performance over multiple runs.

    Parameters
    ----------
    func : Callable
        Function to profile
    *args
        Arguments for the function
    n_runs : int
        Number of times to run the function
    **kwargs
        Keyword arguments for the function

    Returns
    -------
    Dict[str, Any]
        Performance statistics
    """
    times = []
    memory_usage = []

    for _ in range(n_runs):
        process = psutil.Process()
        gc.collect()

        mem_before = process.memory_info().rss / 1024 / 1024
        start = time.perf_counter()

        func(*args, **kwargs)

        elapsed = time.perf_counter() - start
        mem_after = process.memory_info().rss / 1024 / 1024

        times.append(elapsed)
        memory_usage.append(mem_after - mem_before)

        gc.collect()

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_memory_mb': np.mean(memory_usage),
        'std_memory_mb': np.std(memory_usage),
        'n_runs': n_runs
    }


def test_scalability(
    max_users: int = 100000,
    steps: int = 5
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Test scalability of the complete pipeline.

    Parameters
    ----------
    max_users : int
        Maximum number of users to test
    steps : int
        Number of steps from small to large

    Returns
    -------
    results : pd.DataFrame
        Scalability test results
    recommendations : Dict[str, str]
        Performance recommendations
    """
    sizes = np.logspace(2, np.log10(max_users), steps).astype(int)
    results = []

    print(f"Testing scalability with sizes: {sizes}")

    for n_users in tqdm(sizes, desc="Scalability test"):
        # Generate data
        start_gen = time.perf_counter()
        df = simulate_ab_data(n_users=n_users, sessions_per_user=(2, 4))
        gen_time = time.perf_counter() - start_gen

        # Fit model
        try:
            process = psutil.Process()
            gc.collect()
            mem_before = process.memory_info().rss / 1024 / 1024

            start_fit = time.perf_counter()
            _, _, df_model, results_glm = fit_binomial_glm(df)
            ate, rr, _, _ = marginal_effects_ate_and_rr(results_glm, df_model)
            fit_time = time.perf_counter() - start_fit

            gc.collect()
            mem_after = process.memory_info().rss / 1024 / 1024

            results.append({
                'n_users': n_users,
                'n_observations': len(df),
                'data_gen_time': gen_time,
                'model_fit_time': fit_time,
                'total_time': gen_time + fit_time,
                'memory_used_mb': mem_after - mem_before,
                'success': True
            })

        except Exception as e:
            results.append({
                'n_users': n_users,
                'n_observations': len(df),
                'data_gen_time': gen_time,
                'model_fit_time': None,
                'total_time': None,
                'memory_used_mb': None,
                'success': False,
                'error': str(e)
            })

    results_df = pd.DataFrame(results)

    # Generate recommendations
    recommendations = {}

    # Check linear scaling
    if len(results_df[results_df['success']]) > 2:
        times = results_df[results_df['success']]['total_time'].values
        users = results_df[results_df['success']]['n_users'].values

        # Fit linear model to log-log plot
        log_times = np.log(times)
        log_users = np.log(users)
        slope = np.polyfit(log_users, log_times, 1)[0]

        if slope > 1.5:
            recommendations['scaling'] = f"Warning: Superlinear scaling detected (O(n^{slope:.1f}))"
        elif slope > 1.1:
            recommendations['scaling'] = f"Slightly superlinear scaling (O(n^{slope:.1f}))"
        else:
            recommendations['scaling'] = f"Good linear scaling (O(n^{slope:.1f}))"

    # Check memory usage
    if len(results_df[results_df['success']]) > 0:
        max_memory = results_df[results_df['success']]['memory_used_mb'].max()

        if max_memory > 1000:
            recommendations['memory'] = f"High memory usage detected ({max_memory:.0f}MB). Consider chunking for large datasets."
        elif max_memory > 500:
            recommendations['memory'] = f"Moderate memory usage ({max_memory:.0f}MB)"
        else:
            recommendations['memory'] = f"Low memory usage ({max_memory:.0f}MB)"

    # Check failure points
    if not results_df['success'].all():
        first_failure = results_df[~results_df['success']].iloc[0]
        recommendations['limits'] = f"Failures start at {first_failure['n_users']:,} users"
    else:
        recommendations['limits'] = f"Successfully tested up to {results_df['n_users'].max():,} users"

    return results_df, recommendations


def get_optimization_recommendations(
    df: pd.DataFrame,
    target_metric: str = "model_fit_time"
) -> List[str]:
    """
    Generate optimization recommendations based on profiling data.

    Parameters
    ----------
    df : pd.DataFrame
        Profiling results
    target_metric : str
        Metric to optimize

    Returns
    -------
    List[str]
        List of recommendations
    """
    recommendations = []

    # Analyze bottlenecks
    if 'data_gen_time' in df.columns and 'model_fit_time' in df.columns:
        gen_pct = df['data_gen_time'].sum() / (df['data_gen_time'].sum() + df['model_fit_time'].sum())

        if gen_pct > 0.3:
            recommendations.append(
                f"Data generation takes {gen_pct:.0%} of time. "
                "Consider pre-generating data or using more efficient simulation."
            )

    # Check scaling
    if len(df) > 2 and 'n_observations' in df.columns:
        # Calculate time per observation
        df['time_per_obs'] = df[target_metric] / df['n_observations']

        # Check if time per observation increases with scale
        correlation = df['n_observations'].corr(df['time_per_obs'])

        if correlation > 0.5:
            recommendations.append(
                "Performance degrades with scale. Consider:\n"
                "  - Using sparse matrices for large categorical variables\n"
                "  - Chunking data processing\n"
                "  - Reducing precision where appropriate"
            )

    # Memory recommendations
    if 'memory_used_mb' in df.columns:
        max_memory = df['memory_used_mb'].max()

        if max_memory > 500:
            recommendations.append(
                f"High memory usage ({max_memory:.0f}MB). Consider:\n"
                "  - Using float32 instead of float64\n"
                "  - Processing in batches\n"
                "  - Removing unnecessary columns"
            )

    # Success rate
    if 'success' in df.columns:
        success_rate = df['success'].mean()

        if success_rate < 1.0:
            recommendations.append(
                f"Failures detected ({(1-success_rate)*100:.0f}% failure rate). "
                "Check for numerical instability or memory limits."
            )

    if not recommendations:
        recommendations.append("Performance looks good! No major issues detected.")

    return recommendations


def create_performance_report(
    max_users: int = 10000
) -> str:
    """
    Create a comprehensive performance report.

    Parameters
    ----------
    max_users : int
        Maximum number of users to test

    Returns
    -------
    str
        Formatted performance report
    """
    report_lines = [
        "=" * 60,
        "PERFORMANCE REPORT",
        "=" * 60,
        ""
    ]

    # Test scalability
    print("Running scalability tests...")
    scalability_df, scalability_recs = test_scalability(max_users=max_users, steps=5)

    report_lines.append("SCALABILITY TEST RESULTS:")
    report_lines.append("-" * 40)

    for _, row in scalability_df.iterrows():
        if row['success']:
            report_lines.append(
                f"  {row['n_users']:6,} users: "
                f"{row['total_time']:6.2f}s "
                f"({row['memory_used_mb']:6.1f}MB)"
            )
        else:
            report_lines.append(
                f"  {row['n_users']:6,} users: FAILED"
            )

    report_lines.append("")
    report_lines.append("SCALABILITY ANALYSIS:")
    for key, value in scalability_recs.items():
        report_lines.append(f"  {key}: {value}")

    # Get optimization recommendations
    report_lines.append("")
    report_lines.append("OPTIMIZATION RECOMMENDATIONS:")
    report_lines.append("-" * 40)

    opt_recs = get_optimization_recommendations(scalability_df)
    for rec in opt_recs:
        report_lines.append(f"  • {rec}")

    # Performance guidelines
    report_lines.append("")
    report_lines.append("PERFORMANCE GUIDELINES:")
    report_lines.append("-" * 40)
    report_lines.append("  Small (< 1K users): < 1 second")
    report_lines.append("  Medium (1K-10K users): 1-5 seconds")
    report_lines.append("  Large (10K-100K users): 5-30 seconds")
    report_lines.append("  XL (100K-1M users): 30 seconds - 5 minutes")

    # System info
    report_lines.extend([
        "",
        "SYSTEM INFORMATION:",
        "-" * 40,
        f"  CPU cores: {psutil.cpu_count()}",
        f"  Total RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB",
        f"  Available RAM: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f}GB"
    ])

    return "\n".join(report_lines)


if __name__ == "__main__":
    # Run performance report when module is executed
    report = create_performance_report(max_users=10000)
    print(report)

    # Save to file
    with open("performance_report.txt", "w") as f:
        f.write(report)