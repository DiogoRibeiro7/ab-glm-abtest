"""
Scalable processing for very large A/B test datasets.

This module provides:
- Streaming algorithms for large-scale data
- Approximate algorithms for efficiency
- Distributed processing support
- Online/incremental learning
- GPU acceleration utilities
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils import murmurhash3_32


@dataclass
class OnlineStats:
    """Container for online statistics."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared deviations
    min_val: float = float('inf')
    max_val: float = float('-inf')

    @property
    def variance(self) -> float:
        """Calculate variance."""
        return self.m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        """Calculate standard deviation."""
        return np.sqrt(self.variance)

    def update(self, value: float) -> None:
        """Update statistics with new value (Welford's algorithm)."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    def merge(self, other: 'OnlineStats') -> None:
        """Merge with another OnlineStats object."""
        if other.n == 0:
            return

        combined_n = self.n + other.n
        delta = other.mean - self.mean

        # Combine means
        self.mean = (self.n * self.mean + other.n * other.mean) / combined_n

        # Combine variances
        self.m2 = self.m2 + other.m2 + delta**2 * self.n * other.n / combined_n

        self.n = combined_n
        self.min_val = min(self.min_val, other.min_val)
        self.max_val = max(self.max_val, other.max_val)


class StreamingABTest:
    """
    Streaming A/B test analysis for massive datasets.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize streaming analyzer.

        Parameters
        ----------
        confidence_level : float
            Confidence level for intervals
        """
        self.confidence_level = confidence_level
        self.control_stats = OnlineStats()
        self.treatment_stats = OnlineStats()
        self.covariance_sum = 0.0

    def update(self, treatment: int, outcome: float) -> None:
        """
        Update with single observation.

        Parameters
        ----------
        treatment : int
            Treatment indicator (0 or 1)
        outcome : float
            Outcome value
        """
        if treatment == 0:
            self.control_stats.update(outcome)
        else:
            self.treatment_stats.update(outcome)

    def update_batch(
        self,
        treatments: np.ndarray,
        outcomes: np.ndarray
    ) -> None:
        """
        Update with batch of observations.

        Parameters
        ----------
        treatments : np.ndarray
            Treatment indicators
        outcomes : np.ndarray
            Outcome values
        """
        control_mask = treatments == 0
        treatment_mask = treatments == 1

        # Update control stats
        for outcome in outcomes[control_mask]:
            self.control_stats.update(outcome)

        # Update treatment stats
        for outcome in outcomes[treatment_mask]:
            self.treatment_stats.update(outcome)

    def get_results(self) -> Dict[str, float]:
        """
        Get current test results.

        Returns
        -------
        Dict with test statistics
        """
        if self.control_stats.n == 0 or self.treatment_stats.n == 0:
            return {
                'ate': np.nan,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'p_value': np.nan,
                'n_control': self.control_stats.n,
                'n_treatment': self.treatment_stats.n
            }

        # Calculate ATE
        ate = self.treatment_stats.mean - self.control_stats.mean

        # Calculate standard error
        var_control = self.control_stats.variance / self.control_stats.n
        var_treatment = self.treatment_stats.variance / self.treatment_stats.n
        se = np.sqrt(var_control + var_treatment)

        # Calculate confidence interval
        from scipy.stats import norm
        z_score = norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = ate - z_score * se
        ci_upper = ate + z_score * se

        # Calculate p-value
        z_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))

        return {
            'ate': ate,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'n_control': self.control_stats.n,
            'n_treatment': self.treatment_stats.n,
            'control_mean': self.control_stats.mean,
            'treatment_mean': self.treatment_stats.mean,
            'control_std': self.control_stats.std,
            'treatment_std': self.treatment_stats.std
        }

    def reset(self) -> None:
        """Reset statistics."""
        self.control_stats = OnlineStats()
        self.treatment_stats = OnlineStats()
        self.covariance_sum = 0.0


class ApproximateBootstrap:
    """
    Approximate bootstrap using bag of little bootstraps (BLB).
    """

    def __init__(
        self,
        n_subsamples: int = 10,
        subsample_size: Optional[int] = None,
        n_bootstrap: int = 100
    ):
        """
        Initialize BLB.

        Parameters
        ----------
        n_subsamples : int
            Number of subsamples
        subsample_size : int, optional
            Size of each subsample
        n_bootstrap : int
            Bootstrap iterations per subsample
        """
        self.n_subsamples = n_subsamples
        self.subsample_size = subsample_size
        self.n_bootstrap = n_bootstrap

    def compute(
        self,
        data: pd.DataFrame,
        statistic_func: Callable,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute approximate bootstrap CI.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        statistic_func : Callable
            Statistic function
        confidence_level : float
            Confidence level

        Returns
        -------
        Dict with bootstrap results
        """
        n = len(data)

        # Determine subsample size (n^gamma, typically gamma=0.6)
        if self.subsample_size is None:
            self.subsample_size = int(n ** 0.6)

        all_estimates = []

        for _ in range(self.n_subsamples):
            # Draw subsample
            subsample_idx = np.random.choice(n, self.subsample_size, replace=False)
            subsample = data.iloc[subsample_idx]

            # Run bootstrap on subsample
            bootstrap_estimates = []
            for _ in range(self.n_bootstrap):
                # Resample with weights (multinomial)
                weights = np.random.multinomial(n, [1/self.subsample_size] * self.subsample_size)
                weighted_idx = np.repeat(range(self.subsample_size), weights)
                boot_data = subsample.iloc[weighted_idx]

                try:
                    estimate = statistic_func(boot_data)
                    bootstrap_estimates.append(estimate)
                except:
                    continue

            all_estimates.extend(bootstrap_estimates)

        all_estimates = np.array(all_estimates)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        ci_lower = np.percentile(all_estimates, 100 * alpha / 2)
        ci_upper = np.percentile(all_estimates, 100 * (1 - alpha / 2))

        return {
            'estimate': np.mean(all_estimates),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std_error': np.std(all_estimates)
        }


class CountMinSketch:
    """
    Count-Min Sketch for approximate frequency counting in streams.
    """

    def __init__(self, width: int = 1000, depth: int = 5):
        """
        Initialize Count-Min Sketch.

        Parameters
        ----------
        width : int
            Width of sketch
        depth : int
            Number of hash functions
        """
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.hash_seeds = np.random.randint(0, 2**31, depth)

    def update(self, item: str, count: int = 1) -> None:
        """
        Update count for item.

        Parameters
        ----------
        item : str
            Item to count
        count : int
            Count increment
        """
        for i in range(self.depth):
            # Hash to get index
            hash_val = murmurhash3_32(item, seed=self.hash_seeds[i])
            idx = hash_val % self.width
            self.table[i, idx] += count

    def query(self, item: str) -> int:
        """
        Query count for item.

        Parameters
        ----------
        item : str
            Item to query

        Returns
        -------
        int
            Estimated count
        """
        counts = []
        for i in range(self.depth):
            hash_val = murmurhash3_32(item, seed=self.hash_seeds[i])
            idx = hash_val % self.width
            counts.append(self.table[i, idx])

        return int(np.min(counts))


class ReservoirSampling:
    """
    Reservoir sampling for uniform random sampling from streams.
    """

    def __init__(self, k: int, random_state: Optional[int] = None):
        """
        Initialize reservoir sampler.

        Parameters
        ----------
        k : int
            Reservoir size
        random_state : int, optional
            Random seed
        """
        self.k = k
        self.reservoir = []
        self.n = 0

        if random_state is not None:
            np.random.seed(random_state)

    def update(self, item: Any) -> None:
        """
        Update with new item.

        Parameters
        ----------
        item : Any
            Item to potentially add
        """
        self.n += 1

        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
        else:
            # Random replacement
            j = np.random.randint(0, self.n)
            if j < self.k:
                self.reservoir[j] = item

    def get_sample(self) -> List[Any]:
        """
        Get current sample.

        Returns
        -------
        List
            Current reservoir
        """
        return self.reservoir.copy()


class IncrementalPCA:
    """
    Incremental PCA for dimensionality reduction on large datasets.
    """

    def __init__(self, n_components: int = 10):
        """
        Initialize incremental PCA.

        Parameters
        ----------
        n_components : int
            Number of components
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.n_samples_seen_ = 0

    def partial_fit(self, X: np.ndarray) -> None:
        """
        Incrementally fit PCA.

        Parameters
        ----------
        X : np.ndarray
            Batch of data
        """
        n_samples = X.shape[0]

        # Update mean
        if self.mean_ is None:
            self.mean_ = np.mean(X, axis=0)
            self.n_samples_seen_ = n_samples
        else:
            batch_mean = np.mean(X, axis=0)
            n_total = self.n_samples_seen_ + n_samples

            # Incremental mean update
            self.mean_ = (
                self.n_samples_seen_ * self.mean_ + n_samples * batch_mean
            ) / n_total

            self.n_samples_seen_ = n_total

        # Center data
        X_centered = X - self.mean_

        # Update components (simplified)
        if self.components_ is None:
            # Initial SVD
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            self.components_ = Vt[:self.n_components]
        else:
            # Incremental update (simplified)
            # In practice, would use more sophisticated update
            cov = X_centered.T @ X_centered / n_samples
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            self.components_ = eigenvectors[:, idx[:self.n_components]].T

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data.

        Parameters
        ----------
        X : np.ndarray
            Data to transform

        Returns
        -------
        np.ndarray
            Transformed data
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_.T


class MinHashLSH:
    """
    MinHash with Locality Sensitive Hashing for finding similar items.
    """

    def __init__(self, n_perm: int = 128, n_bands: int = 16):
        """
        Initialize MinHash LSH.

        Parameters
        ----------
        n_perm : int
            Number of permutations
        n_bands : int
            Number of bands for LSH
        """
        self.n_perm = n_perm
        self.n_bands = n_bands
        self.rows_per_band = n_perm // n_bands
        self.hash_funcs = self._generate_hash_funcs()
        self.buckets = [{} for _ in range(n_bands)]

    def _generate_hash_funcs(self) -> List[Callable]:
        """Generate hash functions."""
        seeds = np.random.randint(0, 2**31, self.n_perm)
        return [
            lambda x, s=seed: murmurhash3_32(str(x), seed=s)
            for seed in seeds
        ]

    def compute_signature(self, items: set) -> np.ndarray:
        """
        Compute MinHash signature.

        Parameters
        ----------
        items : set
            Set of items

        Returns
        -------
        np.ndarray
            MinHash signature
        """
        signature = np.full(self.n_perm, np.inf)

        for item in items:
            for i, hash_func in enumerate(self.hash_funcs):
                hash_val = hash_func(item)
                signature[i] = min(signature[i], hash_val)

        return signature.astype(np.int64)

    def insert(self, key: str, items: set) -> None:
        """
        Insert item set into LSH.

        Parameters
        ----------
        key : str
            Item identifier
        items : set
            Set of items
        """
        signature = self.compute_signature(items)

        # Hash into bands
        for band_idx in range(self.n_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hash(tuple(signature[start:end]))

            if band_hash not in self.buckets[band_idx]:
                self.buckets[band_idx][band_hash] = []
            self.buckets[band_idx][band_hash].append(key)

    def query(self, items: set) -> List[str]:
        """
        Find similar items.

        Parameters
        ----------
        items : set
            Query items

        Returns
        -------
        List[str]
            Similar item keys
        """
        signature = self.compute_signature(items)
        candidates = set()

        # Check all bands
        for band_idx in range(self.n_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hash(tuple(signature[start:end]))

            if band_hash in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][band_hash])

        return list(candidates)


def sparse_feature_hashing(
    features: pd.DataFrame,
    n_features: int = 2**18,
    dtype: type = np.float32
) -> sparse.csr_matrix:
    """
    Feature hashing for high-dimensional sparse data.

    Parameters
    ----------
    features : pd.DataFrame
        Input features
    n_features : int
        Number of hash buckets
    dtype : type
        Data type for matrix

    Returns
    -------
    sparse.csr_matrix
        Sparse feature matrix
    """
    n_samples = len(features)
    indices = []
    data = []
    indptr = [0]

    for idx, row in features.iterrows():
        row_indices = []
        row_data = []

        for col, value in row.items():
            if pd.notna(value):
                # Hash feature name and value
                feature_str = f"{col}={value}"
                hash_val = murmurhash3_32(feature_str) % n_features

                row_indices.append(hash_val)
                row_data.append(1.0 if dtype == bool else float(value))

        indices.extend(row_indices)
        data.extend(row_data)
        indptr.append(len(indices))

    return sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n_samples, n_features),
        dtype=dtype
    )


class OnlineGradientBoosting:
    """
    Online gradient boosting for incremental learning.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3
    ):
        """
        Initialize online GBM.

        Parameters
        ----------
        n_estimators : int
            Number of trees
        learning_rate : float
            Learning rate
        max_depth : int
            Maximum tree depth
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.feature_importances_ = None

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Incrementally fit the model.

        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Target
        """
        # Simplified online boosting
        # In practice, would use more sophisticated algorithm

        if len(self.trees) == 0:
            # Initialize with first tree
            from sklearn.tree import DecisionTreeRegressor
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, y)
            self.trees.append(tree)
        else:
            # Compute residuals
            predictions = self.predict(X)
            residuals = y - predictions

            # Fit new tree on residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Limit number of trees
            if len(self.trees) > self.n_estimators:
                self.trees.pop(0)  # Remove oldest tree

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X : np.ndarray
            Features

        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self.trees:
            return np.zeros(X.shape[0])

        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions


def distributed_ab_test(
    data_partitions: List[pd.DataFrame],
    treatment_col: str = 'T',
    outcome_col: str = 'y'
) -> Dict[str, float]:
    """
    Distributed A/B test analysis using map-reduce pattern.

    Parameters
    ----------
    data_partitions : List[pd.DataFrame]
        List of data partitions
    treatment_col : str
        Treatment column
    outcome_col : str
        Outcome column

    Returns
    -------
    Dict with combined results
    """
    # Map phase: compute statistics per partition
    partition_stats = []

    for partition in data_partitions:
        treated = partition[partition[treatment_col] == 1][outcome_col]
        control = partition[partition[treatment_col] == 0][outcome_col]

        stats = {
            'n_treatment': len(treated),
            'n_control': len(control),
            'sum_treatment': treated.sum(),
            'sum_control': control.sum(),
            'sum_sq_treatment': (treated ** 2).sum(),
            'sum_sq_control': (control ** 2).sum()
        }
        partition_stats.append(stats)

    # Reduce phase: combine statistics
    total_stats = {
        'n_treatment': sum(s['n_treatment'] for s in partition_stats),
        'n_control': sum(s['n_control'] for s in partition_stats),
        'sum_treatment': sum(s['sum_treatment'] for s in partition_stats),
        'sum_control': sum(s['sum_control'] for s in partition_stats),
        'sum_sq_treatment': sum(s['sum_sq_treatment'] for s in partition_stats),
        'sum_sq_control': sum(s['sum_sq_control'] for s in partition_stats)
    }

    # Calculate final statistics
    mean_treatment = total_stats['sum_treatment'] / total_stats['n_treatment']
    mean_control = total_stats['sum_control'] / total_stats['n_control']

    var_treatment = (
        total_stats['sum_sq_treatment'] / total_stats['n_treatment'] -
        mean_treatment ** 2
    )
    var_control = (
        total_stats['sum_sq_control'] / total_stats['n_control'] -
        mean_control ** 2
    )

    ate = mean_treatment - mean_control
    se = np.sqrt(var_treatment / total_stats['n_treatment'] +
                 var_control / total_stats['n_control'])

    from scipy.stats import norm
    z_stat = ate / se if se > 0 else 0
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    return {
        'ate': ate,
        'std_error': se,
        'p_value': p_value,
        'n_treatment': total_stats['n_treatment'],
        'n_control': total_stats['n_control']
    }


if __name__ == "__main__":
    # Example usage
    print("Testing scalable processing utilities...")

    # Test streaming A/B test
    print("\n1. Streaming A/B Test:")
    streamer = StreamingABTest()

    # Simulate streaming data
    for _ in range(1000):
        treatment = np.random.binomial(1, 0.5)
        outcome = np.random.binomial(1, 0.1 + 0.05 * treatment)
        streamer.update(treatment, outcome)

    results = streamer.get_results()
    print(f"   ATE: {results['ate']:.4f} ± {results['se']:.4f}")
    print(f"   P-value: {results['p_value']:.4f}")

    # Test Count-Min Sketch
    print("\n2. Count-Min Sketch:")
    sketch = CountMinSketch(width=100, depth=5)

    items = ['apple'] * 10 + ['banana'] * 5 + ['orange'] * 3
    for item in items:
        sketch.update(item)

    print(f"   Apple count: {sketch.query('apple')}")
    print(f"   Banana count: {sketch.query('banana')}")
    print(f"   Orange count: {sketch.query('orange')}")

    # Test reservoir sampling
    print("\n3. Reservoir Sampling:")
    sampler = ReservoirSampling(k=10)

    for i in range(100):
        sampler.update(i)

    sample = sampler.get_sample()
    print(f"   Sample size: {len(sample)}")
    print(f"   Sample range: [{min(sample)}, {max(sample)}]")

    print("\nAll scalable processing tests completed!")