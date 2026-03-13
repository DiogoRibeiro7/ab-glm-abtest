"""
Optimization and caching utilities for performance improvement.

This module provides:
- Result caching with TTL
- Memoization decorators
- Query optimization
- Computation graph optimization
- Memory pooling
- JIT compilation helpers
"""

from __future__ import annotations

import functools
import hashlib
import json
import pickle
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class LRUCache:
    """
    Least Recently Used cache with TTL support.
    """

    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        """
        Initialize LRU cache.

        Parameters
        ----------
        maxsize : int
            Maximum cache size
        ttl : float, optional
            Time-to-live in seconds
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Any or None
            Cached value if exists and valid
        """
        if key not in self.cache:
            self.misses += 1
            return None

        # Check TTL
        if self.ttl is not None:
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                self.misses += 1
                return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        # Remove oldest if at capacity
        if key not in self.cache and len(self.cache) >= self.maxsize:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            if oldest in self.timestamps:
                del self.timestamps[oldest]

        self.cache[key] = value
        self.cache.move_to_end(key)

        if self.ttl is not None:
            self.timestamps[key] = time.time()

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.timestamps.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def memoize(maxsize: int = 128, ttl: Optional[float] = None):
    """
    Memoization decorator with LRU cache.

    Parameters
    ----------
    maxsize : int
        Maximum cache size
    ttl : float, optional
        Time-to-live in seconds

    Returns
    -------
    Callable
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(maxsize=maxsize, ttl=ttl)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = _make_cache_key(func.__name__, args, kwargs)

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result)

            return result

        wrapper.cache = cache
        return wrapper

    return decorator


def _make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """
    Create cache key from function arguments.

    Parameters
    ----------
    func_name : str
        Function name
    args : tuple
        Positional arguments
    kwargs : dict
        Keyword arguments

    Returns
    -------
    str
        Cache key
    """
    # Convert args and kwargs to hashable format
    key_parts = [func_name]

    for arg in args:
        if isinstance(arg, (pd.DataFrame, pd.Series)):
            # Hash DataFrame/Series
            key_parts.append(hashlib.md5(
                pd.util.hash_pandas_object(arg).values.tobytes()
            ).hexdigest())
        elif isinstance(arg, np.ndarray):
            # Hash array
            key_parts.append(hashlib.md5(arg.tobytes()).hexdigest())
        else:
            # Hash other types
            key_parts.append(str(hash(arg)))

    # Sort kwargs for consistency
    for k, v in sorted(kwargs.items()):
        key_parts.extend([k, str(hash(v))])

    return '_'.join(key_parts)


class DiskCache:
    """
    Persistent disk-based cache for large results.
    """

    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: float = 1000):
        """
        Initialize disk cache.

        Parameters
        ----------
        cache_dir : str, optional
            Cache directory
        max_size_mb : float
            Maximum cache size in MB
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "ab_glm_disk"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (JSONDecodeError, OSError):
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from disk cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Any or None
            Cached value if exists
        """
        cache_file = self.cache_dir / f"{key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load cache: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        """
        Set value in disk cache.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        # Check cache size
        self._check_size()

        cache_file = self.cache_dir / f"{key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)

            # Update metadata
            self.metadata[key] = {
                'timestamp': time.time(),
                'size': cache_file.stat().st_size
            }
            self._save_metadata()

        except Exception as e:
            warnings.warn(f"Failed to save cache: {e}")

    def _check_size(self) -> None:
        """Check and enforce cache size limit."""
        total_size = sum(
            (self.cache_dir / f"{key}.pkl").stat().st_size
            for key in self.metadata
            if (self.cache_dir / f"{key}.pkl").exists()
        )

        # Remove oldest entries if over limit
        while total_size > self.max_size_bytes and self.metadata:
            oldest_key = min(
                self.metadata.keys(),
                key=lambda k: self.metadata[k]['timestamp']
            )
            cache_file = self.cache_dir / f"{oldest_key}.pkl"

            if cache_file.exists():
                total_size -= cache_file.stat().st_size
                cache_file.unlink()

            del self.metadata[oldest_key]

        self._save_metadata()

    def clear(self) -> None:
        """Clear disk cache."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        self.metadata = {}
        self._save_metadata()


@dataclass
class ComputationNode:
    """Node in computation graph."""
    name: str
    func: Callable
    inputs: List[str]
    cached: bool = False
    computed: bool = False
    result: Any = None


class ComputationGraph:
    """
    Computation graph for optimizing complex calculations.
    """

    def __init__(self):
        """Initialize computation graph."""
        self.nodes = {}
        self.cache = {}

    def add_node(
        self,
        name: str,
        func: Callable,
        inputs: List[str],
        cached: bool = False
    ) -> None:
        """
        Add node to computation graph.

        Parameters
        ----------
        name : str
            Node name
        func : Callable
            Computation function
        inputs : List[str]
            Input node names
        cached : bool
            Whether to cache result
        """
        self.nodes[name] = ComputationNode(name, func, inputs, cached)

    def compute(self, target: str, **kwargs) -> Any:
        """
        Compute target node value.

        Parameters
        ----------
        target : str
            Target node name
        **kwargs
            Input values

        Returns
        -------
        Any
            Computed value
        """
        # Topological sort
        order = self._topological_sort(target)

        # Compute in order
        results = kwargs.copy()

        for node_name in order:
            node = self.nodes[node_name]

            # Check cache
            if node.cached and node_name in self.cache:
                results[node_name] = self.cache[node_name]
                continue

            # Get inputs
            inputs = {inp: results[inp] for inp in node.inputs if inp in results}

            # Compute
            result = node.func(**inputs)
            results[node_name] = result

            # Cache if requested
            if node.cached:
                self.cache[node_name] = result

        return results[target]

    def _topological_sort(self, target: str) -> List[str]:
        """
        Topological sort of dependencies.

        Parameters
        ----------
        target : str
            Target node

        Returns
        -------
        List[str]
            Sorted node names
        """
        visited = set()
        order = []

        def visit(node_name):
            if node_name in visited:
                return
            visited.add(node_name)

            if node_name in self.nodes:
                for inp in self.nodes[node_name].inputs:
                    visit(inp)
                order.append(node_name)

        visit(target)
        return order


class MemoryPool:
    """
    Memory pool for reusing arrays.
    """

    def __init__(self, max_arrays: int = 100):
        """
        Initialize memory pool.

        Parameters
        ----------
        max_arrays : int
            Maximum number of arrays to pool
        """
        self.max_arrays = max_arrays
        self.pool = {}

    def get(self, shape: tuple, dtype: type = np.float64) -> np.ndarray:
        """
        Get array from pool or allocate new.

        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : type
            Data type

        Returns
        -------
        np.ndarray
            Array
        """
        key = (shape, np.dtype(dtype))

        if key in self.pool and self.pool[key]:
            return self.pool[key].pop()

        return np.empty(shape, dtype=dtype)

    def release(self, array: np.ndarray) -> None:
        """
        Release array back to pool.

        Parameters
        ----------
        array : np.ndarray
            Array to release
        """
        key = (array.shape, np.dtype(array.dtype))

        if key not in self.pool:
            self.pool[key] = []

        if len(self.pool[key]) < self.max_arrays:
            self.pool[key].append(array)


def vectorize_operation(func: Callable) -> Callable:
    """
    Decorator to vectorize operations on DataFrames.

    Parameters
    ----------
    func : Callable
        Function to vectorize

    Returns
    -------
    Callable
        Vectorized function
    """
    @functools.wraps(func)
    def wrapper(data: Union[pd.DataFrame, pd.Series], *args, **kwargs):
        if isinstance(data, pd.DataFrame):
            # Apply to each column
            result = data.copy()
            for col in data.columns:
                if data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    result[col] = func(data[col].values, *args, **kwargs)
            return result
        elif isinstance(data, pd.Series):
            return pd.Series(
                func(data.values, *args, **kwargs),
                index=data.index,
                name=data.name
            )
        else:
            return func(data, *args, **kwargs)

    return wrapper


@vectorize_operation
def fast_zscore(x: np.ndarray, ddof: int = 1) -> np.ndarray:
    """
    Fast z-score normalization.

    Parameters
    ----------
    x : np.ndarray
        Input array
    ddof : int
        Degrees of freedom

    Returns
    -------
    np.ndarray
        Z-scores
    """
    mean = np.mean(x)
    std = np.std(x, ddof=ddof)
    return (x - mean) / std if std > 0 else np.zeros_like(x)


class QueryOptimizer:
    """
    Optimize DataFrame queries for performance.
    """

    def __init__(self):
        """Initialize query optimizer."""
        self.query_cache = {}
        self.index_cache = {}

    def optimize_filter(
        self,
        df: pd.DataFrame,
        conditions: List[Tuple[str, str, Any]]
    ) -> pd.DataFrame:
        """
        Optimize DataFrame filtering.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        conditions : List[Tuple[str, str, Any]]
            List of (column, operator, value) tuples

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        # Sort conditions by selectivity (most selective first)
        sorted_conditions = self._sort_by_selectivity(df, conditions)

        # Apply conditions sequentially
        result = df
        for col, op, val in sorted_conditions:
            if op == '==':
                result = result[result[col] == val]
            elif op == '!=':
                result = result[result[col] != val]
            elif op == '<':
                result = result[result[col] < val]
            elif op == '<=':
                result = result[result[col] <= val]
            elif op == '>':
                result = result[result[col] > val]
            elif op == '>=':
                result = result[result[col] >= val]
            elif op == 'in':
                result = result[result[col].isin(val)]
            elif op == 'between':
                result = result[(result[col] >= val[0]) & (result[col] <= val[1])]

            # Early termination if empty
            if len(result) == 0:
                break

        return result

    def _sort_by_selectivity(
        self,
        df: pd.DataFrame,
        conditions: List[Tuple[str, str, Any]]
    ) -> List[Tuple[str, str, Any]]:
        """Sort conditions by estimated selectivity."""
        selectivities = []

        for col, op, val in conditions:
            if op == '==':
                # Equality is usually most selective
                selectivity = 1 / df[col].nunique()
            elif op in ['<', '>', '<=', '>=']:
                # Range queries
                selectivity = 0.5  # Rough estimate
            elif op == 'in':
                selectivity = len(val) / df[col].nunique()
            else:
                selectivity = 0.5

            selectivities.append((selectivity, (col, op, val)))

        # Sort by selectivity (most selective first)
        selectivities.sort(key=lambda x: x[0])

        return [cond for _, cond in selectivities]

    def create_index(self, df: pd.DataFrame, columns: List[str]) -> None:
        """
        Create index for faster lookups.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to index
        columns : List[str]
            Columns to index
        """
        for col in columns:
            if col in df.columns:
                # Create sorted index
                sorted_idx = df[col].argsort()
                self.index_cache[col] = {
                    'values': df[col].iloc[sorted_idx].values,
                    'indices': sorted_idx.values
                }

    def indexed_lookup(
        self,
        df: pd.DataFrame,
        column: str,
        value: Any
    ) -> pd.DataFrame:
        """
        Fast indexed lookup.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame
        column : str
            Column to search
        value : Any
            Value to find

        Returns
        -------
        pd.DataFrame
            Matching rows
        """
        if column in self.index_cache:
            # Use binary search on sorted index
            idx_data = self.index_cache[column]
            pos = np.searchsorted(idx_data['values'], value)

            if pos < len(idx_data['values']) and idx_data['values'][pos] == value:
                # Find all matching values
                start = pos
                while start > 0 and idx_data['values'][start - 1] == value:
                    start -= 1

                end = pos
                while end < len(idx_data['values']) - 1 and idx_data['values'][end + 1] == value:
                    end += 1

                matching_indices = idx_data['indices'][start:end + 1]
                return df.iloc[matching_indices]

        # Fallback to standard lookup
        return df[df[column] == value]


if __name__ == "__main__":
    # Example usage
    print("Testing optimization utilities...")

    # Test LRU cache
    print("\n1. LRU Cache:")
    cache = LRUCache(maxsize=3, ttl=10)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    print(f"   Get 'a': {cache.get('a')}")
    cache.set("d", 4)  # Should evict 'b'
    print(f"   Get 'b' (evicted): {cache.get('b')}")
    print(f"   Hit rate: {cache.hit_rate:.2%}")

    # Test memoization
    print("\n2. Memoization:")
    @memoize(maxsize=10)
    def expensive_func(x):
        time.sleep(0.1)  # Simulate expensive computation
        return x ** 2

    start = time.time()
    result1 = expensive_func(5)
    time1 = time.time() - start

    start = time.time()
    result2 = expensive_func(5)  # Should be cached
    time2 = time.time() - start

    print(f"   First call: {time1:.3f}s")
    print(f"   Cached call: {time2:.3f}s")
    print(f"   Speedup: {time1 / time2:.0f}x")

    # Test computation graph
    print("\n3. Computation Graph:")
    graph = ComputationGraph()
    graph.add_node("a_squared", lambda a: a ** 2, ["a"])
    graph.add_node("b_squared", lambda b: b ** 2, ["b"])
    graph.add_node("sum", lambda a_squared, b_squared: a_squared + b_squared,
                   ["a_squared", "b_squared"], cached=True)

    result = graph.compute("sum", a=3, b=4)
    print(f"   3² + 4² = {result}")

    # Test vectorized operation
    print("\n4. Vectorized Operations:")
    df = pd.DataFrame({
        'x': np.random.randn(1000),
        'y': np.random.randn(1000)
    })

    normalized = fast_zscore(df)
    print(f"   Shape: {normalized.shape}")
    print(f"   Mean: {normalized['x'].mean():.6f}")
    print(f"   Std: {normalized['x'].std():.6f}")

    print("\nAll optimization tests completed!")
