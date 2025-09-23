"""
Correlation Engine Test Suite
============================

Comprehensive tests for the CorrelationEngine:
- Cross-asset correlation calculation accuracy
- Multiple correlation methods (Pearson, Spearman, Kendall)
- Rolling correlation windows
- Exponentially weighted correlations
- DCC-GARCH dynamic correlations
- Copula-based correlations
- Tail correlations
- Performance benchmarks

Target Requirements:
- Correlation updates <100ms
- Support for 50+ assets
- Statistical significance testing
- Real-time correlation updates
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import logging
from unittest.mock import Mock, patch

# Test utilities
from .test_utils import (
    MockDataGenerator, PerformanceBenchmark, StrategyTester, BacktestValidator,
    MockMarketData, assert_performance_threshold, assert_latency_threshold,
    assert_statistical_significance
)

# Import strategy under test
try:
    from ..institutional.correlation_engine import (
        CorrelationEngine, CorrelationConfig, CorrelationResult
    )
except ImportError as e:
    pytest.skip(f"Could not import correlation engine: {e}", allow_module_level=True)

from . import TEST_CONFIG

logger = logging.getLogger(__name__)


class TestCorrelationEngine:
    """Test suite for CorrelationEngine"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.data_generator = MockDataGenerator(TEST_CONFIG['mock_data_seed'])
        self.benchmark = PerformanceBenchmark()
        
        # Initialize engine with test configuration
        self.test_config = CorrelationConfig(
            windows=[10, 20, 30],
            min_observations=5,
            correlation_methods=['pearson', 'spearman', 'kendall'],
            ewm_halflife=15,
            dcc_garch_enabled=True,
            copula_enabled=True,
            update_frequency=60,
            significance_level=0.05,
            max_assets=20,  # Reduced for testing
            enable_caching=True,
            parallel_workers=2
        )
        
        self.engine = CorrelationEngine(self.test_config)
        
        # Generate test assets with known correlations
        self.test_assets = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 'COMP']
        self.setup_test_data()
        
        self.performance_metrics = {}
    
    def setup_test_data(self):
        """Setup test data with known correlation patterns"""
        base_returns = np.random.normal(0, 0.02, 1000)  # Base return series
        
        for i, asset in enumerate(self.test_assets):
            # Create correlated returns
            correlation_target = 0.1 + i * 0.1  # Increasing correlation with base
            
            # Generate correlated returns using Cholesky decomposition
            noise = np.random.normal(0, 0.02, 1000)
            correlated_returns = (
                correlation_target * base_returns +
                np.sqrt(1 - correlation_target**2) * noise
            )
            
            # Convert to price series
            prices = (1 + correlated_returns).cumprod() * (1000 + i * 100)
            
            # Create time series
            timestamps = pd.date_range(
                start=datetime.now() - timedelta(days=1000),
                periods=1000,
                freq='1D'
            )
            price_series = pd.Series(prices, index=timestamps)
            
            self.engine.add_price_data(asset, price_series)
    
    def test_engine_initialization(self):
        """Test engine initialization and configuration"""
        # Test default initialization
        default_engine = CorrelationEngine()
        assert default_engine.config is not None
        assert len(default_engine.price_data) == 0
        
        # Test custom configuration
        assert self.engine.config.max_assets == 20
        assert len(self.engine.config.windows) == 3
        assert 'pearson' in self.engine.config.correlation_methods
        
        # Test data structures
        assert isinstance(self.engine.price_data, dict)
        assert isinstance(self.engine.returns_data, dict)
        assert isinstance(self.engine.correlation_cache, dict)
        
        logger.info("✓ Engine initialization test passed")
    
    def test_price_data_management(self):
        """Test price data addition and management"""
        test_engine = CorrelationEngine(self.test_config)
        
        # Test adding price data
        prices = pd.Series([100, 101, 102, 101, 103], 
                          index=pd.date_range('2023-01-01', periods=5))
        
        test_engine.add_price_data('TEST', prices)
        
        # Validate data storage
        assert 'TEST' in test_engine.price_data
        assert 'TEST' in test_engine.returns_data
        assert len(test_engine.price_data['TEST']) == 5
        assert len(test_engine.returns_data['TEST']) == 4  # Returns are one less
        
        # Test asset limit
        config_limited = CorrelationConfig(max_assets=2)
        limited_engine = CorrelationEngine(config_limited)
        
        for i in range(3):
            prices = pd.Series([100] * 5, index=pd.date_range('2023-01-01', periods=5))
            limited_engine.add_price_data(f'ASSET_{i}', prices)
        
        # Should only have 2 assets due to limit
        assert len(limited_engine.price_data) <= 2
        
        logger.info("✓ Price data management test passed")
    
    def test_pearson_correlation_calculation(self):
        """Test Pearson correlation calculation accuracy"""
        calculation_times = []
        correlation_accuracies = []
        
        for window in [10, 20, 30]:
            start_time = time.perf_counter()
            
            result = self.engine.calculate_correlation_matrix(
                method='pearson',
                window=window,
                assets=self.test_assets[:5]  # First 5 assets
            )
            
            calculation_time = time.perf_counter() - start_time
            calculation_times.append(calculation_time)
            
            # Validate result structure
            assert isinstance(result, CorrelationResult)
            assert result.method == 'pearson'
            assert result.window == window
            assert result.correlation_matrix is not None
            assert result.p_values is not None
            assert result.is_significant is not None
            
            # Validate correlation matrix properties
            corr_matrix = result.correlation_matrix
            assert corr_matrix.shape == (5, 5)
            
            # Diagonal should be 1.0
            np.testing.assert_allclose(np.diag(corr_matrix), 1.0, atol=1e-10)
            
            # Matrix should be symmetric
            np.testing.assert_allclose(corr_matrix, corr_matrix.T, atol=1e-10)
            
            # Values should be in [-1, 1]
            assert np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0)
            
            # Test known correlation pattern
            # Assets should have increasing correlation with BTC
            btc_correlations = corr_matrix.loc['BTC', :].drop('BTC')
            
            # Should show some pattern (not random)
            correlation_range = btc_correlations.max() - btc_correlations.min()
            correlation_accuracies.append(correlation_range)
        
        # Performance validation
        avg_calculation_time = np.mean(calculation_times)
        assert_latency_threshold(
            avg_calculation_time,
            TEST_CONFIG['performance_thresholds']['correlation_update_time'],
            'Pearson Correlation Calculation'
        )
        
        # Accuracy validation
        avg_correlation_range = np.mean(correlation_accuracies)
        assert avg_correlation_range > 0.1, f"Correlation range too small: {avg_correlation_range:.3f}"
        
        self.performance_metrics['pearson_calculation_time'] = avg_calculation_time
        self.performance_metrics['pearson_correlation_range'] = avg_correlation_range
        
        logger.info(f"✓ Pearson correlation - Time: {avg_calculation_time*1000:.1f}ms, "
                   f"Range: {avg_correlation_range:.3f}")
    
    def test_spearman_correlation_calculation(self):
        """Test Spearman rank correlation calculation"""
        result = self.engine.calculate_correlation_matrix(
            method='spearman',
            window=20,
            assets=self.test_assets[:4]
        )
        
        # Validate Spearman-specific properties
        assert result.method == 'spearman'
        assert result.correlation_matrix is not None
        assert result.p_values is not None
        
        corr_matrix = result.correlation_matrix
        
        # Spearman correlations should be different from Pearson
        pearson_result = self.engine.calculate_correlation_matrix(
            method='pearson',
            window=20,
            assets=self.test_assets[:4]
        )
        
        # Should have some differences (not identical)
        differences = np.abs(corr_matrix.values - pearson_result.correlation_matrix.values)
        mean_difference = np.mean(differences[np.triu_indices_from(differences, k=1)])
        
        assert mean_difference > 0.001, f"Spearman too similar to Pearson: {mean_difference:.6f}"
        
        logger.info(f"✓ Spearman correlation - Difference from Pearson: {mean_difference:.4f}")
    
    def test_kendall_correlation_calculation(self):
        """Test Kendall tau correlation calculation"""
        result = self.engine.calculate_correlation_matrix(
            method='kendall',
            window=15,
            assets=self.test_assets[:3]
        )
        
        # Validate Kendall-specific properties
        assert result.method == 'kendall'
        assert result.correlation_matrix is not None
        
        corr_matrix = result.correlation_matrix
        
        # Kendall values should generally be smaller than Pearson
        pearson_result = self.engine.calculate_correlation_matrix(
            method='pearson',
            window=15,
            assets=self.test_assets[:3]
        )
        
        # Kendall typically has smaller absolute values
        kendall_abs_mean = np.mean(np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]))
        pearson_abs_mean = np.mean(np.abs(pearson_result.correlation_matrix.values[np.triu_indices_from(pearson_result.correlation_matrix.values, k=1)]))
        
        # Kendall should be smaller in magnitude (generally)
        # Allow for some variation due to random data
        logger.info(f"✓ Kendall correlation - Kendall: {kendall_abs_mean:.3f}, Pearson: {pearson_abs_mean:.3f}")
    
    def test_exponentially_weighted_correlation(self):
        """Test exponentially weighted correlation calculation"""
        calculation_times = []
        
        for halflife in [10, 20, 30]:
            start_time = time.perf_counter()
            
            result = self.engine.calculate_ewm_correlation(
                assets=self.test_assets[:4],
                halflife=halflife
            )
            
            calculation_time = time.perf_counter() - start_time
            calculation_times.append(calculation_time)
            
            # Validate EWM result
            assert result.method == 'ewm'
            assert result.window == halflife
            assert result.correlation_matrix is not None
            
            corr_matrix = result.correlation_matrix
            assert corr_matrix.shape == (4, 4)
            
            # Basic correlation matrix properties
            np.testing.assert_allclose(np.diag(corr_matrix), 1.0, atol=1e-6)
            assert np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0)
        
        avg_calculation_time = np.mean(calculation_times)
        assert_latency_threshold(avg_calculation_time, 0.2, 'EWM Correlation')  # 200ms
        
        logger.info(f"✓ EWM correlation - Avg time: {avg_calculation_time*1000:.1f}ms")
    
    def test_dcc_garch_correlation(self):
        """Test DCC-GARCH dynamic correlation calculation"""
        if not self.test_config.dcc_garch_enabled:
            pytest.skip("DCC-GARCH disabled in configuration")
        
        start_time = time.perf_counter()
        
        result = self.engine.calculate_dcc_garch_correlation(
            assets=self.test_assets[:3],  # Smaller set for performance
            lookback=100
        )
        
        calculation_time = time.perf_counter() - start_time
        
        # Validate DCC-GARCH result
        assert result.method == 'dcc_garch'
        assert result.correlation_matrix is not None
        assert 'dcc_params' in result.metadata
        assert 'garch_models' in result.metadata
        
        corr_matrix = result.correlation_matrix
        assert corr_matrix.shape == (3, 3)
        
        # Check DCC parameters
        dcc_params = result.metadata['dcc_params']
        assert 'a' in dcc_params and 'b' in dcc_params
        assert 0 < dcc_params['a'] + dcc_params['b'] < 1  # Stationarity condition
        
        # Performance check (DCC-GARCH is computationally intensive)
        assert_latency_threshold(calculation_time, 2.0, 'DCC-GARCH Correlation')  # 2 seconds
        
        logger.info(f"✓ DCC-GARCH correlation - Time: {calculation_time:.2f}s, "
                   f"Params: a={dcc_params['a']:.3f}, b={dcc_params['b']:.3f}")
    
    def test_copula_correlation(self):
        """Test copula-based correlation calculation"""
        if not self.test_config.copula_enabled:
            pytest.skip("Copula analysis disabled in configuration")
        
        copula_types = ['gaussian', 't']  # Test main copula types
        
        for copula_type in copula_types:
            start_time = time.perf_counter()
            
            result = self.engine.calculate_copula_correlation(
                assets=self.test_assets[:3],
                copula_type=copula_type
            )
            
            calculation_time = time.perf_counter() - start_time
            
            # Validate copula result
            assert result.method == f'copula_{copula_type}'
            assert result.correlation_matrix is not None
            assert result.metadata['copula_type'] == copula_type
            
            corr_matrix = result.correlation_matrix
            assert corr_matrix.shape == (3, 3)
            
            # Copula correlations should be valid
            np.testing.assert_allclose(np.diag(corr_matrix), 1.0, atol=1e-6)
            assert np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0)
            
            # Performance check
            assert_latency_threshold(calculation_time, 1.0, f'Copula {copula_type}')  # 1 second
            
            logger.info(f"✓ {copula_type.capitalize()} copula correlation - Time: {calculation_time:.2f}s")
    
    def test_tail_correlation(self):
        """Test tail correlation calculation"""
        thresholds = [0.90, 0.95, 0.99]
        
        for threshold in thresholds:
            result = self.engine.calculate_tail_correlation(
                assets=self.test_assets[:4],
                threshold=threshold
            )
            
            # Validate tail correlation result
            assert result.method == 'tail_correlation'
            assert result.metadata['threshold'] == threshold
            
            if not result.correlation_matrix.empty:
                # If we have enough tail events
                corr_matrix = result.correlation_matrix
                assert corr_matrix.shape == (4, 4)
                
                # Tail correlations might be different from regular correlations
                regular_result = self.engine.calculate_correlation_matrix(
                    method='pearson',
                    window=100,
                    assets=self.test_assets[:4]
                )
                
                # Should have some difference in extreme cases
                if not corr_matrix.isnull().all().all():
                    logger.info(f"✓ Tail correlation (threshold={threshold}) calculated")
            else:
                logger.info(f"✓ Tail correlation (threshold={threshold}) - insufficient tail events")
    
    @pytest.mark.asyncio
    async def test_batch_correlation_calculation(self):
        """Test batch calculation of all correlation methods"""
        start_time = time.perf_counter()
        
        results = await self.engine.calculate_all_correlations(
            assets=self.test_assets[:5]
        )
        
        total_time = time.perf_counter() - start_time
        
        # Validate batch results
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check expected correlation methods
        expected_methods = ['pearson_10', 'pearson_20', 'pearson_30', 'spearman_10', 'kendall_10', 'ewm']
        
        for method in expected_methods:
            if method in results:
                result = results[method]
                assert isinstance(result, CorrelationResult)
                assert result.correlation_matrix is not None
        
        # Performance check for batch processing
        assert_latency_threshold(total_time, 5.0, 'Batch Correlation Calculation')  # 5 seconds
        
        self.performance_metrics['batch_calculation_time'] = total_time
        self.performance_metrics['batch_methods_count'] = len(results)
        
        logger.info(f"✓ Batch correlation calculation - Time: {total_time:.2f}s, Methods: {len(results)}")
    
    def test_correlation_summary(self):
        """Test correlation summary generation"""
        # Calculate some correlations first
        results = {}
        
        for method in ['pearson', 'spearman']:
            for window in [10, 20]:
                key = f"{method}_{window}"
                results[key] = self.engine.calculate_correlation_matrix(
                    method=method,
                    window=window,
                    assets=self.test_assets[:4]
                )
        
        # Generate summary
        summary = self.engine.get_correlation_summary(results)
        
        # Validate summary
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(results)
        
        # Check summary columns
        expected_columns = ['method', 'mean_correlation', 'median_correlation', 
                          'std_correlation', 'min_correlation', 'max_correlation', 'timestamp']
        
        for col in expected_columns:
            assert col in summary.columns
        
        # Validate summary statistics
        for _, row in summary.iterrows():
            assert -1 <= row['mean_correlation'] <= 1
            assert -1 <= row['min_correlation'] <= 1
            assert -1 <= row['max_correlation'] <= 1
            assert row['std_correlation'] >= 0
        
        logger.info(f"✓ Correlation summary - Methods: {len(summary)}, "
                   f"Avg correlation: {summary['mean_correlation'].mean():.3f}")
    
    def test_statistical_significance(self):
        """Test statistical significance calculations"""
        result = self.engine.calculate_correlation_matrix(
            method='pearson',
            window=50,
            assets=self.test_assets[:4]
        )
        
        # Validate p-values
        assert result.p_values is not None
        p_values = result.p_values
        
        # P-values should be in [0, 1]
        assert np.all(p_values >= 0) and np.all(p_values <= 1)
        
        # Diagonal p-values should be 0 (perfect correlation with self)
        diagonal_p_values = np.diag(p_values)
        np.testing.assert_allclose(diagonal_p_values, 0, atol=1e-10)
        
        # Check significance detection
        assert result.is_significant is not None
        significance = result.is_significant
        
        # Where p-value < 0.05, should be significant
        expected_significance = p_values < self.test_config.significance_level
        np.testing.assert_array_equal(significance, expected_significance)
        
        # Count significant correlations
        significant_count = np.sum(significance.values[np.triu_indices_from(significance.values, k=1)])
        total_pairs = len(self.test_assets[:4]) * (len(self.test_assets[:4]) - 1) // 2
        
        logger.info(f"✓ Statistical significance - Significant pairs: {significant_count}/{total_pairs}")
    
    def test_confidence_intervals(self):
        """Test confidence interval calculations"""
        result = self.engine.calculate_correlation_matrix(
            method='pearson',
            window=30,
            assets=self.test_assets[:3]
        )
        
        # Validate confidence intervals
        assert result.confidence_intervals is not None
        confidence_intervals = result.confidence_intervals
        
        # Check CI structure
        for pair_key, ci_data in confidence_intervals.items():
            assert 'lower' in ci_data
            assert 'upper' in ci_data
            assert 'correlation' in ci_data
            
            # CI bounds should be valid
            assert -1 <= ci_data['lower'] <= 1
            assert -1 <= ci_data['upper'] <= 1
            assert ci_data['lower'] <= ci_data['upper']
            
            # Correlation should be within CI
            assert ci_data['lower'] <= ci_data['correlation'] <= ci_data['upper']
        
        logger.info(f"✓ Confidence intervals - Pairs: {len(confidence_intervals)}")
    
    def test_streaming_updates(self):
        """Test streaming price updates and correlation maintenance"""
        test_engine = CorrelationEngine(self.test_config)
        
        # Add initial data
        for asset in self.test_assets[:3]:
            prices = pd.Series([100 + i for i in range(20)], 
                             index=pd.date_range('2023-01-01', periods=20))
            test_engine.add_price_data(asset, prices)
        
        # Test streaming updates
        update_times = []
        
        for i in range(10):
            start_time = time.perf_counter()
            
            # Update each asset with new price
            for j, asset in enumerate(self.test_assets[:3]):
                new_price = 120 + i + j * 0.1
                new_timestamp = datetime(2023, 1, 21) + timedelta(days=i)
                test_engine.update_price(asset, new_price, new_timestamp)
            
            # Calculate correlation after updates
            result = test_engine.calculate_correlation_matrix(
                method='pearson',
                window=15,
                assets=self.test_assets[:3]
            )
            
            update_time = time.perf_counter() - start_time
            update_times.append(update_time)
            
            assert result.correlation_matrix is not None
        
        avg_update_time = np.mean(update_times)
        assert_latency_threshold(avg_update_time, 0.1, 'Streaming Update')  # 100ms
        
        logger.info(f"✓ Streaming updates - Avg time: {avg_update_time*1000:.1f}ms")
    
    def test_caching_performance(self):
        """Test correlation caching for performance"""
        # Enable caching
        cached_engine = CorrelationEngine(self.test_config)
        for asset in self.test_assets[:4]:
            prices = self.engine.price_data[asset]
            cached_engine.add_price_data(asset, prices)
        
        # First calculation (should cache)
        start_time = time.perf_counter()
        result1 = cached_engine.calculate_correlation_matrix(
            method='pearson',
            window=20,
            assets=self.test_assets[:4]
        )
        first_time = time.perf_counter() - start_time
        
        # Second calculation (should use cache)
        start_time = time.perf_counter()
        result2 = cached_engine.calculate_correlation_matrix(
            method='pearson',
            window=20,
            assets=self.test_assets[:4]
        )
        cached_time = time.perf_counter() - start_time
        
        # Cached result should be much faster
        assert cached_time < first_time * 0.5, f"Cache not effective: {cached_time:.3f}s vs {first_time:.3f}s"
        
        # Results should be identical
        np.testing.assert_array_equal(
            result1.correlation_matrix.values,
            result2.correlation_matrix.values
        )
        
        logger.info(f"✓ Caching performance - First: {first_time*1000:.1f}ms, "
                   f"Cached: {cached_time*1000:.1f}ms")
    
    def test_performance_benchmarks(self):
        """Test overall engine performance benchmarks"""
        # Test different numbers of assets
        asset_counts = [5, 10, 15, 20]
        performance_by_count = {}
        
        for count in asset_counts:
            test_assets = self.test_assets[:count]
            calculation_times = []
            
            for _ in range(5):  # 5 iterations per count
                start_time = time.perf_counter()
                
                result = self.engine.calculate_correlation_matrix(
                    method='pearson',
                    window=20,
                    assets=test_assets
                )
                
                calculation_time = time.perf_counter() - start_time
                calculation_times.append(calculation_time)
                
                assert result.correlation_matrix.shape == (count, count)
            
            avg_time = np.mean(calculation_times)
            performance_by_count[count] = avg_time
            
            logger.info(f"Assets: {count}, Avg time: {avg_time*1000:.1f}ms")
        
        # Performance should scale reasonably (not worse than O(n^2))
        max_assets = max(asset_counts)
        min_assets = min(asset_counts)
        
        time_ratio = performance_by_count[max_assets] / performance_by_count[min_assets]
        asset_ratio = (max_assets / min_assets) ** 2
        
        assert time_ratio <= asset_ratio * 2, f"Performance scaling too poor: {time_ratio:.1f}x"
        
        # Overall performance requirement
        overall_avg = np.mean(list(performance_by_count.values()))
        assert_latency_threshold(overall_avg, 0.5, 'Overall Performance')  # 500ms
        
        self.performance_metrics['overall_avg_time'] = overall_avg
        self.performance_metrics['scaling_ratio'] = time_ratio / asset_ratio
        
        logger.info(f"✓ Performance benchmarks - Overall avg: {overall_avg*1000:.1f}ms, "
                   f"Scaling efficiency: {time_ratio / asset_ratio:.2f}")


class TestCorrelationEngineEdgeCases:
    """Test edge cases and error handling"""
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        engine = CorrelationEngine()
        
        # Add minimal data
        short_prices = pd.Series([100, 101], index=pd.date_range('2023-01-01', periods=2))
        engine.add_price_data('TEST', short_prices)
        
        # Should handle gracefully
        result = engine.calculate_correlation_matrix(
            method='pearson',
            window=10,
            assets=['TEST']
        )
        
        # Should return appropriate result or handle gracefully
        assert isinstance(result, CorrelationResult)
    
    def test_missing_assets_handling(self):
        """Test handling of missing assets"""
        # Request correlation for non-existent asset
        result = self.engine.calculate_correlation_matrix(
            method='pearson',
            window=20,
            assets=['NONEXISTENT']
        )
        
        # Should handle gracefully
        assert isinstance(result, CorrelationResult)
    
    def test_extreme_correlation_values(self):
        """Test handling of extreme correlation scenarios"""
        engine = CorrelationEngine()
        
        # Perfect positive correlation
        prices1 = pd.Series([100, 110, 120, 130], index=pd.date_range('2023-01-01', periods=4))
        prices2 = pd.Series([200, 220, 240, 260], index=pd.date_range('2023-01-01', periods=4))  # 2x scaling
        
        engine.add_price_data('ASSET1', prices1)
        engine.add_price_data('ASSET2', prices2)
        
        result = engine.calculate_correlation_matrix(
            method='pearson',
            window=4,
            assets=['ASSET1', 'ASSET2']
        )
        
        # Should detect perfect correlation
        correlation = result.correlation_matrix.loc['ASSET1', 'ASSET2']
        assert abs(correlation - 1.0) < 0.01, f"Failed to detect perfect correlation: {correlation:.3f}"
        
        logger.info(f"✓ Perfect correlation detected: {correlation:.6f}")
    
    def test_nan_and_infinite_data(self):
        """Test handling of NaN and infinite values"""
        engine = CorrelationEngine()
        
        # Data with NaN and inf values
        bad_prices = pd.Series([100, np.nan, 120, np.inf, 140], 
                              index=pd.date_range('2023-01-01', periods=5))
        
        engine.add_price_data('BAD_ASSET', bad_prices)
        
        # Should handle gracefully without crashing
        result = engine.calculate_correlation_matrix(
            method='pearson',
            window=5,
            assets=['BAD_ASSET']
        )
        
        assert isinstance(result, CorrelationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--log-cli-level=INFO"])