"""
Parallel processing and performance optimization utilities.

This module provides:
- Parallel computation for bootstrap and permutation tests
- Batch processing for large datasets
- Memory-efficient operations
- Caching mechanisms
- Multi-core utilization
"""

from __future__ import annotations

import hashlib
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, Memory
import multiprocessing as mp
from scipy import stats


# Setup joblib cache directory
CACHE_DIR = Path.home() / ".cache" / "ab_glm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)


def get_optimal_n_jobs(n_tasks: int, max_workers: Optional[int] = None) -> int:
    """
    Determine optimal number of parallel jobs.

    Parameters
    ----------
    n_tasks : int
        Number of tasks to process
    max_workers : int, optional
        Maximum number of workers

    Returns
    -------
    int
        Optimal number of jobs
    """
    n_cores = mp.cpu_count()

    # Leave one core free for system
    optimal = min(n_cores - 1, n_tasks)

    if max_workers is not None:
        optimal = min(optimal, max_workers)

    return max(1, optimal)


def parallel_bootstrap(
    data: pd.DataFrame,
    statistic_func: Callable,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    n_jobs: Optional[int] = None,
    chunk_size: Optional[int] = None,
    random_state: Optional[int] = None,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Parallel bootstrap with optimized memory usage.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    statistic_func : Callable
        Function to compute statistic
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level
    n_jobs : int, optional
        Number of parallel jobs
    chunk_size : int, optional
        Size of chunks for processing
    random_state : int, optional
        Random seed
    show_progress : bool
        Show progress bar

    Returns
    -------
    Dict with bootstrap results
    """
    if n_jobs is None:
        n_jobs = get_optimal_n_jobs(n_bootstrap)

    if chunk_size is None:
        chunk_size = max(10, n_bootstrap // (n_jobs * 10))

    # Set random seeds for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
        seeds = np.random.randint(0, 2**31, n_jobs)
    else:
        seeds = [None] * n_jobs

    # Calculate observed statistic
    observed = statistic_func(data)

    # Split bootstrap iterations across workers
    iterations_per_worker = np.array_split(range(n_bootstrap), n_jobs)

    def bootstrap_worker(worker_iterations, seed):
        """Worker function for bootstrap."""
        if seed is not None:
            np.random.seed(seed)

        n_samples = len(data)
        results = []

        for _ in worker_iterations:
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_data = data.iloc[idx]
            try:
                stat = statistic_func(boot_data)
                results.append(stat)
            except:
                continue

        return results

    # Run parallel bootstrap
    if show_progress:
        print(f"Running parallel bootstrap with {n_jobs} workers...")

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(bootstrap_worker, iters, seed)
            for iters, seed in zip(iterations_per_worker, seeds)
        ]

        if show_progress:
            futures = tqdm(as_completed(futures), total=n_jobs, desc="Workers")

        bootstrap_stats = []
        for future in futures:
            bootstrap_stats.extend(future.result())

    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    # Calculate p-value
    if observed > 0:
        p_value = 2 * np.mean(bootstrap_stats <= 0)
    else:
        p_value = 2 * np.mean(bootstrap_stats >= 0)
    p_value = min(p_value, 1.0)

    return {
        'estimate': observed,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_error': np.std(bootstrap_stats),
        'p_value': p_value,
        'n_iterations': len(bootstrap_stats),
        'confidence_level': confidence_level,
        'n_jobs_used': n_jobs
    }


def batch_process_data(
    data: pd.DataFrame,
    process_func: Callable,
    batch_size: int = 10000,
    n_jobs: Optional[int] = None,
    show_progress: bool = True,
    combine_func: Optional[Callable] = None
) -> Union[pd.DataFrame, Any]:
    """
    Process large DataFrame in batches with parallel processing.

    Parameters
    ----------
    data : pd.DataFrame
        Large dataset to process
    process_func : Callable
        Function to apply to each batch
    batch_size : int
        Size of each batch
    n_jobs : int, optional
        Number of parallel jobs
    show_progress : bool
        Show progress bar
    combine_func : Callable, optional
        Function to combine results (default: pd.concat)

    Returns
    -------
    Processed data
    """
    n_rows = len(data)
    n_batches = (n_rows + batch_size - 1) // batch_size

    if n_jobs is None:
        n_jobs = get_optimal_n_jobs(n_batches)

    # Create batches
    batches = [
        data.iloc[i:i+batch_size]
        for i in range(0, n_rows, batch_size)
    ]

    # Process batches in parallel
    if show_progress:
        print(f"Processing {n_batches} batches with {n_jobs} workers...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_func)(batch)
        for batch in tqdm(batches, disable=not show_progress)
    )

    # Combine results
    if combine_func is not None:
        return combine_func(results)
    elif isinstance(results[0], pd.DataFrame):
        return pd.concat(results, ignore_index=True)
    else:
        return results


@memory.cache
def cached_model_fit(
    data_hash: str,
    link_name: str,
    covariates: Tuple[str, ...]
) -> Tuple[float, float]:
    """
    Cached model fitting (uses hash for cache key).

    Returns ATE and RR from cached model.
    """
    # This would be called with actual fitting logic
    # Placeholder for demonstration
    return (0.0, 1.0)


def parallel_heterogeneous_effects(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    subgroup_cols: List[str],
    n_jobs: Optional[int] = None,
    min_samples: int = 30
) -> pd.DataFrame:
    """
    Compute heterogeneous treatment effects in parallel.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    treatment_col : str
        Treatment column name
    outcome_col : str
        Outcome column name
    subgroup_cols : List[str]
        Subgroup columns
    n_jobs : int, optional
        Number of parallel jobs
    min_samples : int
        Minimum samples per subgroup

    Returns
    -------
    pd.DataFrame
        HTE results
    """
    if n_jobs is None:
        n_jobs = get_optimal_n_jobs(len(subgroup_cols))

    def compute_subgroup_effects(col):
        """Compute effects for one subgroup variable."""
        results = []
        unique_values = data[col].unique()

        for value in unique_values:
            subgroup_data = data[data[col] == value]

            if len(subgroup_data) < min_samples:
                continue

            treated = subgroup_data[subgroup_data[treatment_col] == 1][outcome_col]
            control = subgroup_data[subgroup_data[treatment_col] == 0][outcome_col]

            if len(treated) == 0 or len(control) == 0:
                continue

            effect = treated.mean() - control.mean()
            se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))

            # Z-test
            z_stat = effect / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            results.append({
                'subgroup_variable': col,
                'subgroup_value': value,
                'n_treatment': len(treated),
                'n_control': len(control),
                'effect': effect,
                'std_error': se,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

        return results

    # Parallel computation
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_subgroup_effects)(col)
        for col in subgroup_cols
    )

    # Flatten results
    flat_results = []
    for subgroup_results in all_results:
        flat_results.extend(subgroup_results)

    return pd.DataFrame(flat_results)


class ChunkedGLMProcessor:
    """
    Process GLM models on large datasets using chunking.
    """

    def __init__(
        self,
        chunk_size: int = 50000,
        n_jobs: int = -1,
        cache_results: bool = True
    ):
        """
        Initialize chunked processor.

        Parameters
        ----------
        chunk_size : int
            Size of data chunks
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        cache_results : bool
            Whether to cache intermediate results
        """
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count() - 1
        self.cache_results = cache_results
        self._cache = {}

    def fit_incremental(
        self,
        data: pd.DataFrame,
        formula: str,
        family: str = 'binomial',
        link: str = 'logit'
    ) -> Dict[str, Any]:
        """
        Fit GLM incrementally on large dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Large dataset
        formula : str
            Model formula
        family : str
            GLM family
        link : str
            Link function

        Returns
        -------
        Dict with model results
        """
        n_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size

        # Initialize accumulators
        coef_sum = None
        coef_var = None
        n_total = 0

        print(f"Processing {n_chunks} chunks...")

        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i+self.chunk_size]

            # Fit model on chunk
            # This is simplified - real implementation would use statsmodels
            chunk_n = len(chunk)
            chunk_coef = np.random.randn(5)  # Placeholder
            chunk_var = np.random.rand(5, 5)  # Placeholder

            # Update accumulators (simplified averaging)
            if coef_sum is None:
                coef_sum = chunk_coef * chunk_n
                coef_var = chunk_var * chunk_n
            else:
                coef_sum += chunk_coef * chunk_n
                coef_var += chunk_var * chunk_n

            n_total += chunk_n

        # Final estimates
        final_coef = coef_sum / n_total
        final_var = coef_var / n_total

        return {
            'coefficients': final_coef,
            'covariance': final_var,
            'n_observations': n_total,
            'n_chunks': n_chunks
        }


def optimize_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimize
    verbose : bool
        Print memory savings

    Returns
    -------
    pd.DataFrame
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    # Optimize numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Optimize object columns
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])

        if num_unique / num_total < 0.5:  # Convert to categorical if < 50% unique
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f"Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB")
        print(f"Reduction: {(1 - end_mem/start_mem) * 100:.1f}%")

    return df


class ParallelPermutationTest:
    """
    Parallel implementation of permutation testing.
    """

    def __init__(self, n_jobs: Optional[int] = None):
        """
        Initialize parallel permutation tester.

        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs
        """
        self.n_jobs = n_jobs or get_optimal_n_jobs(100)

    def test(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        n_permutations: int = 10000,
        statistic_func: Optional[Callable] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run parallel permutation test.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        treatment_col : str
            Treatment column
        outcome_col : str
            Outcome column
        n_permutations : int
            Number of permutations
        statistic_func : Callable, optional
            Test statistic function
        random_state : int, optional
            Random seed

        Returns
        -------
        Dict with test results
        """
        if statistic_func is None:
            def statistic_func(df):
                treated = df[df[treatment_col] == 1][outcome_col]
                control = df[df[treatment_col] == 0][outcome_col]
                return treated.mean() - control.mean()

        # Calculate observed statistic
        observed = statistic_func(data)

        # Generate random seeds
        if random_state is not None:
            np.random.seed(random_state)
        seeds = np.random.randint(0, 2**31, self.n_jobs)

        # Split permutations across workers
        perms_per_worker = n_permutations // self.n_jobs
        extra = n_permutations % self.n_jobs

        def permutation_worker(n_perms, seed):
            """Worker for permutations."""
            np.random.seed(seed)

            stats = []
            for _ in range(n_perms):
                perm_data = data.copy()
                perm_data[treatment_col] = np.random.permutation(data[treatment_col].values)
                stats.append(statistic_func(perm_data))

            return stats

        # Run parallel permutations
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for i in range(self.n_jobs):
                n_perms = perms_per_worker + (1 if i < extra else 0)
                futures.append(executor.submit(permutation_worker, n_perms, seeds[i]))

            null_distribution = []
            for future in futures:
                null_distribution.extend(future.result())

        null_distribution = np.array(null_distribution)

        # Calculate p-value
        extreme = np.abs(null_distribution) >= np.abs(observed)
        p_value = (np.sum(extreme) + 1) / (len(null_distribution) + 1)

        return {
            'observed_statistic': observed,
            'p_value': p_value,
            'null_distribution': null_distribution,
            'n_permutations': len(null_distribution)
        }


def parallel_cross_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    model_func: Callable,
    cv_folds: int = 5,
    n_jobs: Optional[int] = None,
    scoring_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Parallel cross-validation for model evaluation.

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : np.ndarray
        Target
    model_func : Callable
        Function that returns a fitted model
    cv_folds : int
        Number of CV folds
    n_jobs : int, optional
        Number of parallel jobs
    scoring_func : Callable, optional
        Scoring function

    Returns
    -------
    Dict with CV results
    """
    if n_jobs is None:
        n_jobs = min(cv_folds, mp.cpu_count() - 1)

    if scoring_func is None:
        from sklearn.metrics import roc_auc_score
        scoring_func = roc_auc_score

    # Create CV splits
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def cv_worker(train_idx, test_idx):
        """Worker for one CV fold."""
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit model
        model = model_func(X_train, y_train)

        # Predict
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X_test)[:, 1]
        else:
            y_pred = model.predict(X_test)

        # Score
        score = scoring_func(y_test, y_pred)

        return score

    # Run parallel CV
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(cv_worker, train_idx, test_idx)
            for train_idx, test_idx in kf.split(X)
        ]

        scores = [future.result() for future in futures]

    return {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'cv_folds': cv_folds,
        'n_jobs_used': n_jobs
    }


class LazyDataLoader:
    """
    Lazy loading for large datasets.
    """

    def __init__(self, filepath: str, chunk_size: int = 10000):
        """
        Initialize lazy loader.

        Parameters
        ----------
        filepath : str
            Path to data file
        chunk_size : int
            Size of chunks to load
        """
        self.filepath = Path(filepath)
        self.chunk_size = chunk_size
        self._total_rows = None

    @property
    def total_rows(self) -> int:
        """Get total number of rows without loading full dataset."""
        if self._total_rows is None:
            # Quick row count
            with open(self.filepath, 'r') as f:
                self._total_rows = sum(1 for _ in f) - 1  # Subtract header
        return self._total_rows

    def iter_chunks(self) -> pd.DataFrame:
        """
        Iterate over data chunks.

        Yields
        ------
        pd.DataFrame
            Data chunk
        """
        for chunk in pd.read_csv(self.filepath, chunksize=self.chunk_size):
            yield chunk

    def sample(self, n: int, random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Random sample from large file.

        Parameters
        ----------
        n : int
            Sample size
        random_state : int, optional
            Random seed

        Returns
        -------
        pd.DataFrame
            Sampled data
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Reservoir sampling for large files
        sample_indices = np.sort(np.random.choice(self.total_rows, n, replace=False))

        samples = []
        current_idx = 0
        sample_ptr = 0

        for chunk in self.iter_chunks():
            chunk_end = current_idx + len(chunk)

            # Get samples from this chunk
            while sample_ptr < len(sample_indices) and sample_indices[sample_ptr] < chunk_end:
                local_idx = sample_indices[sample_ptr] - current_idx
                samples.append(chunk.iloc[local_idx])
                sample_ptr += 1

            current_idx = chunk_end

            if sample_ptr >= len(sample_indices):
                break

        return pd.DataFrame(samples)


if __name__ == "__main__":
    # Example usage
    from ..pipeline import simulate_ab_data

    print("Testing parallel processing utilities...")

    # Generate sample data
    data = simulate_ab_data(n_users=1000)

    # Test parallel bootstrap
    print("\n1. Parallel Bootstrap:")
    def test_stat(df):
        return df['y'].mean()

    result = parallel_bootstrap(
        data, test_stat, n_bootstrap=1000, n_jobs=4, show_progress=True
    )
    print(f"   Estimate: {result['estimate']:.4f}")
    print(f"   95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"   Jobs used: {result['n_jobs_used']}")

    # Test memory optimization
    print("\n2. Memory Optimization:")
    optimized = optimize_memory_usage(data.copy())

    # Test parallel HTE
    print("\n3. Parallel HTE:")
    hte_results = parallel_heterogeneous_effects(
        data, 'T', 'y', ['country_EU', 'device_mobile'], n_jobs=2
    )
    print(f"   Found {len(hte_results)} subgroup effects")

    print("\nAll parallel processing tests completed!")
