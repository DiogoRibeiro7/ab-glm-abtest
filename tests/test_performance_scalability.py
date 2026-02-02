"""
Tests for performance and scalability modules.
"""

import time
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from ab_glm import simulate_ab_data
from ab_glm.parallel_processing import (
    get_optimal_n_jobs,
    parallel_bootstrap,
    batch_process_data,
    parallel_heterogeneous_effects,
    ChunkedGLMProcessor,
    optimize_memory_usage,
    ParallelPermutationTest,
    LazyDataLoader,
)
from ab_glm.scalable_processing import (
    OnlineStats,
    StreamingABTest,
    ApproximateBootstrap,
    CountMinSketch,
    ReservoirSampling,
    IncrementalPCA,
    MinHashLSH,
    sparse_feature_hashing,
    distributed_ab_test,
)
from ab_glm.optimization import (
    LRUCache,
    memoize,
    DiskCache,
    ComputationGraph,
    MemoryPool,
    fast_zscore,
    QueryOptimizer,
)


class TestParallelProcessing:
    """Test parallel processing functions."""

    def test_get_optimal_n_jobs(self):
        """Test optimal job calculation."""
        # Should not exceed CPU count - 1
        n_jobs = get_optimal_n_jobs(100)
        assert n_jobs >= 1
        assert n_jobs <= 100

        # Respect max_workers
        n_jobs = get_optimal_n_jobs(100, max_workers=2)
        assert n_jobs <= 2

    @patch('ab_glm.parallel_processing.ProcessPoolExecutor')
    def test_parallel_bootstrap(self, mock_executor):
        """Test parallel bootstrap."""
        np.random.seed(42)
        data = simulate_ab_data(n_users=50)

        def test_stat(df):
            return df['y'].mean()

        # Mock executor
        mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = [0.1, 0.11, 0.09]

        result = parallel_bootstrap(
            data, test_stat, n_bootstrap=100, n_jobs=2, show_progress=False
        )

        assert 'estimate' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'n_jobs_used' in result

    def test_batch_process_data(self):
        """Test batch processing."""
        data = pd.DataFrame({
            'x': np.random.randn(1000),
            'y': np.random.randn(1000)
        })

        def process_func(batch):
            return batch.mean()

        result = batch_process_data(
            data, process_func, batch_size=100, n_jobs=1, show_progress=False
        )

        assert len(result) == 10  # 1000 / 100

    def test_optimize_memory_usage(self):
        """Test memory optimization."""
        df = pd.DataFrame({
            'int_col': np.arange(100, dtype=np.int64),
            'float_col': np.random.randn(100).astype(np.float64),
            'cat_col': ['A', 'B'] * 50
        })

        optimized = optimize_memory_usage(df, verbose=False)

        # Check that memory was reduced
        orig_mem = df.memory_usage(deep=True).sum()
        opt_mem = optimized.memory_usage(deep=True).sum()
        assert opt_mem <= orig_mem

    def test_parallel_heterogeneous_effects(self):
        """Test parallel HTE computation."""
        np.random.seed(42)
        data = simulate_ab_data(n_users=100)

        result = parallel_heterogeneous_effects(
            data, 'T', 'y', ['country_EU', 'device_mobile'], n_jobs=1
        )

        assert isinstance(result, pd.DataFrame)
        assert 'effect' in result.columns
        assert 'p_value' in result.columns

    def test_chunked_glm_processor(self):
        """Test chunked GLM processing."""
        processor = ChunkedGLMProcessor(chunk_size=100, n_jobs=1)

        # Create large fake dataset
        data = pd.DataFrame({
            'x': np.random.randn(500),
            'y': np.random.binomial(1, 0.5, 500)
        })

        # Test incremental fitting (simplified)
        result = processor.fit_incremental(data, "y ~ x")

        assert 'coefficients' in result
        assert 'n_observations' in result
        assert result['n_observations'] == 500


class TestScalableProcessing:
    """Test scalable processing functions."""

    def test_online_stats(self):
        """Test online statistics."""
        stats = OnlineStats()

        values = [1, 2, 3, 4, 5]
        for val in values:
            stats.update(val)

        assert stats.n == 5
        assert stats.mean == 3.0
        assert stats.min_val == 1
        assert stats.max_val == 5
        assert stats.std > 0

        # Test merge
        stats2 = OnlineStats()
        for val in [6, 7, 8]:
            stats2.update(val)

        stats.merge(stats2)
        assert stats.n == 8
        assert stats.mean == 4.5

    def test_streaming_ab_test(self):
        """Test streaming A/B test."""
        streamer = StreamingABTest()

        np.random.seed(42)
        for _ in range(100):
            treatment = np.random.binomial(1, 0.5)
            outcome = np.random.binomial(1, 0.1 + 0.05 * treatment)
            streamer.update(treatment, outcome)

        results = streamer.get_results()

        assert 'ate' in results
        assert 'p_value' in results
        assert results['n_control'] + results['n_treatment'] == 100

    def test_approximate_bootstrap(self):
        """Test approximate bootstrap (BLB)."""
        np.random.seed(42)
        data = pd.DataFrame({'x': np.random.randn(100)})

        blb = ApproximateBootstrap(n_subsamples=3, n_bootstrap=10)
        result = blb.compute(data, lambda df: df['x'].mean())

        assert 'estimate' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result

    def test_count_min_sketch(self):
        """Test Count-Min Sketch."""
        sketch = CountMinSketch(width=100, depth=5)

        # Add items
        for _ in range(10):
            sketch.update('apple')
        for _ in range(5):
            sketch.update('banana')

        # Query counts
        apple_count = sketch.query('apple')
        banana_count = sketch.query('banana')
        unknown_count = sketch.query('unknown')

        assert apple_count >= 10  # May overestimate
        assert banana_count >= 5
        assert unknown_count >= 0

    def test_reservoir_sampling(self):
        """Test reservoir sampling."""
        sampler = ReservoirSampling(k=10, random_state=42)

        for i in range(100):
            sampler.update(i)

        sample = sampler.get_sample()

        assert len(sample) == 10
        assert all(0 <= x < 100 for x in sample)

    def test_incremental_pca(self):
        """Test incremental PCA."""
        pca = IncrementalPCA(n_components=2)

        # Fit in batches
        for _ in range(3):
            batch = np.random.randn(20, 5)
            pca.partial_fit(batch)

        # Transform
        X_new = np.random.randn(10, 5)
        X_transformed = pca.transform(X_new)

        assert X_transformed.shape == (10, 2)

    def test_minhash_lsh(self):
        """Test MinHash LSH."""
        lsh = MinHashLSH(n_perm=32, n_bands=4)

        # Insert sets
        lsh.insert('doc1', {'apple', 'banana', 'orange'})
        lsh.insert('doc2', {'apple', 'banana', 'grape'})
        lsh.insert('doc3', {'car', 'bike', 'train'})

        # Query similar
        similar = lsh.query({'apple', 'banana', 'mango'})

        assert 'doc1' in similar or 'doc2' in similar
        # doc3 should not be similar

    def test_sparse_feature_hashing(self):
        """Test feature hashing."""
        features = pd.DataFrame({
            'color': ['red', 'blue', 'red'],
            'size': [10, 20, 15]
        })

        sparse_matrix = sparse_feature_hashing(features, n_features=100)

        assert sparse_matrix.shape == (3, 100)
        assert sparse_matrix.nnz > 0  # Has non-zero elements

    def test_distributed_ab_test(self):
        """Test distributed A/B test."""
        # Create partitions
        np.random.seed(42)
        partitions = []
        for _ in range(3):
            partition = pd.DataFrame({
                'T': np.random.binomial(1, 0.5, 100),
                'y': np.random.binomial(1, 0.15, 100)
            })
            partitions.append(partition)

        result = distributed_ab_test(partitions)

        assert 'ate' in result
        assert 'p_value' in result
        assert result['n_treatment'] + result['n_control'] == 300


class TestOptimization:
    """Test optimization utilities."""

    def test_lru_cache(self):
        """Test LRU cache."""
        cache = LRUCache(maxsize=3)

        cache.set('a', 1)
        cache.set('b', 2)
        cache.set('c', 3)

        assert cache.get('a') == 1
        assert cache.hits == 1

        # Add fourth item, should evict 'b'
        cache.set('d', 4)
        assert cache.get('b') is None
        assert cache.misses == 1

        # Test TTL
        cache_ttl = LRUCache(maxsize=10, ttl=0.1)
        cache_ttl.set('x', 10)
        assert cache_ttl.get('x') == 10
        time.sleep(0.15)
        assert cache_ttl.get('x') is None  # Expired

    def test_memoize_decorator(self):
        """Test memoization decorator."""
        call_count = 0

        @memoize(maxsize=10)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x ** 2

        # First call
        result1 = expensive_func(5)
        assert result1 == 25
        assert call_count == 1

        # Second call (cached)
        result2 = expensive_func(5)
        assert result2 == 25
        assert call_count == 1  # Not called again

        # Different argument
        result3 = expensive_func(3)
        assert result3 == 9
        assert call_count == 2

    @patch('ab_glm.optimization.Path')
    def test_disk_cache(self, mock_path):
        """Test disk cache."""
        mock_path.home.return_value = MagicMock()

        cache = DiskCache(cache_dir='/tmp/test_cache', max_size_mb=1)

        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            with patch('pickle.dump') as mock_dump:
                cache.set('key1', {'data': 'value'})
                mock_dump.assert_called_once()

    def test_computation_graph(self):
        """Test computation graph."""
        graph = ComputationGraph()

        graph.add_node('x2', lambda x: x * 2, ['x'])
        graph.add_node('x3', lambda x: x * 3, ['x'])
        graph.add_node('sum', lambda x2, x3: x2 + x3, ['x2', 'x3'])

        result = graph.compute('sum', x=5)
        assert result == 5 * 2 + 5 * 3

        # Test caching
        graph.add_node('cached', lambda x: x * 10, ['x'], cached=True)
        result1 = graph.compute('cached', x=3)
        assert result1 == 30

        # Should use cache
        result2 = graph.compute('cached', x=3)
        assert result2 == 30

    def test_memory_pool(self):
        """Test memory pool."""
        pool = MemoryPool(max_arrays=5)

        # Get array
        arr1 = pool.get((10, 10), dtype=np.float32)
        assert arr1.shape == (10, 10)

        # Release and reget
        pool.release(arr1)
        arr2 = pool.get((10, 10), dtype=np.float32)
        assert arr1 is arr2  # Same array reused

    def test_fast_zscore(self):
        """Test fast z-score."""
        data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100)
        })

        normalized = fast_zscore(data)

        assert np.abs(normalized['x'].mean()) < 1e-10
        assert np.abs(normalized['x'].std() - 1.0) < 1e-10

    def test_query_optimizer(self):
        """Test query optimizer."""
        optimizer = QueryOptimizer()

        df = pd.DataFrame({
            'a': range(100),
            'b': ['x', 'y'] * 50,
            'c': np.random.randn(100)
        })

        # Test filter optimization
        conditions = [
            ('a', '>', 50),
            ('b', '==', 'x'),
            ('c', '<', 0)
        ]

        result = optimizer.optimize_filter(df, conditions)
        assert len(result) < len(df)

        # Test index creation and lookup
        optimizer.create_index(df, ['a'])
        indexed_result = optimizer.indexed_lookup(df, 'a', 50)
        assert len(indexed_result) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])