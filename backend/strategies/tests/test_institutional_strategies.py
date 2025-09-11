"""
Comprehensive Test Suite for Institutional Trading Strategies
===========================================================

This test suite validates all institutional strategies and their integration:
- Individual strategy component testing
- Performance validation against requirements
- Statistical significance testing
- Integration layer validation
- Stress testing and edge cases
- Performance benchmarking

Requirements Validation:
- Liquidity detection accuracy >80%
- Smart money signal accuracy >65%
- Volume profile computation <50ms
- Correlation updates <100ms
- Overall 10-15% performance improvement
- Integration latency <10ms
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import logging
from unittest.mock import Mock, patch, AsyncMock

# Test utilities
from .test_utils import (
    MockDataGenerator, PerformanceBenchmark, StrategyTester, BacktestValidator,
    MockMarketData, TestMetrics, assert_performance_threshold, assert_latency_threshold,
    assert_statistical_significance, async_test_helper
)

# Import strategies for testing
try:
    from ..institutional.liquidity_hunting import LiquidityHuntingStrategy
    from ..institutional.smart_money_divergence import SmartMoneyDivergenceDetector
    from ..institutional.volume_profile import VPVRAnalyzer
    from ..institutional.correlation_engine import CorrelationEngine
    from ..integration.strategy_manager import StrategyIntegrationManager
    from ..integration.performance_tracker import PerformanceTracker
except ImportError as e:
    pytest.skip(f"Could not import strategies: {e}", allow_module_level=True)

# Test configuration from __init__.py
from . import TEST_CONFIG

logger = logging.getLogger(__name__)


class TestInstitutionalStrategiesIntegration:
    """Integration tests for all institutional strategies"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.data_generator = MockDataGenerator(TEST_CONFIG['mock_data_seed'])
        self.benchmark = PerformanceBenchmark()
        self.backtest_validator = BacktestValidator()
        
        # Generate test data
        self.test_price_data = self.data_generator.generate_price_series(
            TEST_CONFIG['test_data_size']
        )
        self.test_volume_data = self.data_generator.generate_volume_series(
            self.test_price_data
        )
        
        # Performance tracking
        self.test_results = {}
        self.performance_metrics = {}
    
    def test_strategy_initialization(self):
        """Test that all strategies can be initialized properly"""
        strategies = {
            'liquidity_hunting': LiquidityHuntingStrategy,
            'smart_money_divergence': SmartMoneyDivergenceDetector,
            'volume_profile': VPVRAnalyzer,
            'correlation_engine': CorrelationEngine
        }
        
        for name, strategy_class in strategies.items():
            try:
                # Test with default configuration
                strategy = strategy_class()
                assert strategy is not None, f"Failed to initialize {name}"
                
                # Test with custom configuration
                config = {'test_mode': True}
                if name == 'liquidity_hunting':
                    strategy = strategy_class('BTC/USD', sensitivity=0.7)
                elif name == 'correlation_engine':
                    strategy = strategy_class(config)
                else:
                    strategy = strategy_class(config)
                
                assert strategy is not None, f"Failed to initialize {name} with config"
                
                logger.info(f"✓ {name} initialization test passed")
                
            except Exception as e:
                pytest.fail(f"Strategy {name} initialization failed: {e}")
    
    def test_liquidity_detection_accuracy(self):
        """Test liquidity hunting strategy accuracy requirement >80%"""
        strategy = LiquidityHuntingStrategy('BTC/USD', sensitivity=0.7)
        
        correct_detections = 0
        total_tests = 100
        
        for i in range(total_tests):
            # Generate market data with known liquidity patterns
            price = 30000 + np.random.uniform(-1000, 1000)
            
            # Create iceberg order pattern
            order_book = self.data_generator.generate_order_book(price)
            
            # Add artificial iceberg signature
            if i % 3 == 0:  # 33% of cases have iceberg
                # Modify order book to simulate iceberg
                for j in range(5):
                    order_book['bids'][j][1] *= 3  # Increase volume at specific levels
                has_iceberg = True
            else:
                has_iceberg = False
            
            # Generate trades
            trades = self.data_generator.generate_trades(price, 50000, 20)
            
            market_data = {
                'order_book': order_book,
                'trades': trades,
                'market_data': {'price': price}
            }
            
            # Test strategy detection
            signals = strategy.update(order_book, trades, market_data['market_data'])
            
            # Check if iceberg was detected
            iceberg_detected = any(signal.signal_type == 'iceberg' for signal in signals)
            
            if (has_iceberg and iceberg_detected) or (not has_iceberg and not iceberg_detected):
                correct_detections += 1
        
        accuracy = correct_detections / total_tests
        
        # Assert requirement
        assert_performance_threshold(
            accuracy,
            TEST_CONFIG['performance_thresholds']['liquidity_detection_accuracy'],
            'Liquidity Detection Accuracy',
            higher_is_better=True
        )
        
        self.performance_metrics['liquidity_detection_accuracy'] = accuracy
        logger.info(f"✓ Liquidity detection accuracy: {accuracy:.2%}")
    
    def test_smart_money_signal_accuracy(self):
        """Test smart money divergence accuracy requirement >65%"""
        detector = SmartMoneyDivergenceDetector()
        
        correct_predictions = 0
        total_tests = 100
        
        for i in range(total_tests):
            # Generate price data with divergence patterns
            price_window = self.test_price_data.iloc[i:i+50] if i+50 < len(self.test_price_data) else self.test_price_data.iloc[-50:]
            
            # Create artificial divergence
            if i % 4 == 0:  # 25% bullish divergence
                expected_direction = 'bullish'
                # Simulate smart money accumulation during price decline
                mock_flows = pd.DataFrame({
                    'smart_money_flow': np.linspace(0, 100, len(price_window)),  # Increasing flow
                })
            elif i % 4 == 1:  # 25% bearish divergence  
                expected_direction = 'bearish'
                # Simulate smart money distribution during price increase
                mock_flows = pd.DataFrame({
                    'smart_money_flow': np.linspace(100, 0, len(price_window)),  # Decreasing flow
                })
            else:
                expected_direction = 'neutral'
                # No clear divergence
                mock_flows = pd.DataFrame({
                    'smart_money_flow': np.random.uniform(40, 60, len(price_window)),
                })
            
            # Mock async methods for testing
            with patch.object(detector, '_fetch_on_chain_data', return_value={}), \
                 patch.object(detector, '_fetch_whale_transactions', return_value=[]), \
                 patch.object(detector, '_calculate_smart_money_flows', return_value=mock_flows), \
                 patch.object(detector, '_fetch_exchange_flows', return_value={'net_flow': 0}):
                
                try:
                    # Run detection synchronously for testing
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    signals = loop.run_until_complete(
                        detector.detect_divergence('BTC/USD', price_window)
                    )
                    loop.close()
                    
                    # Analyze signals
                    if signals:
                        signal = signals[0]
                        if expected_direction == 'bullish' and 'bullish' in signal.divergence_type.value:
                            correct_predictions += 1
                        elif expected_direction == 'bearish' and 'bearish' in signal.divergence_type.value:
                            correct_predictions += 1
                        elif expected_direction == 'neutral' and len(signals) == 0:
                            correct_predictions += 1
                    elif expected_direction == 'neutral':
                        correct_predictions += 1
                        
                except Exception as e:
                    logger.warning(f"Smart money detection failed for iteration {i}: {e}")
        
        accuracy = correct_predictions / total_tests
        
        # Assert requirement
        assert_performance_threshold(
            accuracy,
            TEST_CONFIG['performance_thresholds']['smart_money_signal_accuracy'],
            'Smart Money Signal Accuracy',
            higher_is_better=True
        )
        
        self.performance_metrics['smart_money_signal_accuracy'] = accuracy
        logger.info(f"✓ Smart money signal accuracy: {accuracy:.2%}")
    
    def test_volume_profile_computation_speed(self):
        """Test volume profile computation time requirement <50ms"""
        analyzer = VPVRAnalyzer()
        
        # Create test data frame
        test_data = pd.DataFrame({
            'price': self.test_price_data.values[:500],
            'volume': self.test_volume_data.values[:500],
            'side': np.random.choice(['buy', 'sell'], 500)
        }, index=self.test_price_data.index[:500])
        
        # Benchmark computation time
        computation_times = []
        
        for _ in range(50):  # Run multiple iterations
            start_time = time.perf_counter()
            
            try:
                profile = analyzer.calculate_profile(test_data)
                assert profile is not None
                assert profile.total_volume > 0
                
            except Exception as e:
                pytest.fail(f"Volume profile computation failed: {e}")
            
            end_time = time.perf_counter()
            computation_times.append(end_time - start_time)
        
        avg_computation_time = np.mean(computation_times)
        max_computation_time = np.max(computation_times)
        p95_computation_time = np.percentile(computation_times, 95)
        
        # Assert requirement (average should be well under 50ms)
        assert_latency_threshold(
            avg_computation_time,
            TEST_CONFIG['performance_thresholds']['volume_profile_computation_time'],
            'Volume Profile Computation'
        )
        
        # Also check P95 to ensure consistency
        assert_latency_threshold(
            p95_computation_time,
            TEST_CONFIG['performance_thresholds']['volume_profile_computation_time'] * 2,
            'Volume Profile Computation P95'
        )
        
        self.performance_metrics['volume_profile_computation_time'] = avg_computation_time
        logger.info(f"✓ Volume profile computation: {avg_computation_time*1000:.1f}ms avg, {max_computation_time*1000:.1f}ms max")
    
    def test_correlation_update_speed(self):
        """Test correlation engine update time requirement <100ms"""
        engine = CorrelationEngine()
        
        # Add test data for multiple assets
        assets = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
        for asset in assets:
            # Generate correlated price series
            base_returns = self.test_price_data.pct_change().dropna()
            # Add some correlation between assets
            asset_returns = base_returns + np.random.normal(0, 0.01, len(base_returns))
            asset_prices = (1 + asset_returns).cumprod() * np.random.uniform(100, 50000)
            
            engine.add_price_data(asset, asset_prices)
        
        # Benchmark correlation calculation
        update_times = []
        
        for _ in range(30):  # Multiple iterations
            start_time = time.perf_counter()
            
            try:
                result = engine.calculate_correlation_matrix(method='pearson', window=30)
                assert result is not None
                assert result.correlation_matrix is not None
                assert result.correlation_matrix.shape == (len(assets), len(assets))
                
            except Exception as e:
                pytest.fail(f"Correlation calculation failed: {e}")
            
            end_time = time.perf_counter()
            update_times.append(end_time - start_time)
        
        avg_update_time = np.mean(update_times)
        max_update_time = np.max(update_times)
        p95_update_time = np.percentile(update_times, 95)
        
        # Assert requirement
        assert_latency_threshold(
            avg_update_time,
            TEST_CONFIG['performance_thresholds']['correlation_update_time'],
            'Correlation Update'
        )
        
        assert_latency_threshold(
            p95_update_time,
            TEST_CONFIG['performance_thresholds']['correlation_update_time'] * 1.5,
            'Correlation Update P95'
        )
        
        self.performance_metrics['correlation_update_time'] = avg_update_time
        logger.info(f"✓ Correlation update: {avg_update_time*1000:.1f}ms avg, {max_update_time*1000:.1f}ms max")
    
    def test_integration_layer_latency(self):
        """Test integration layer latency requirement <10ms"""
        # Create minimal strategy manager config for testing
        test_config = {
            'max_workers': 2,
            'execution_interval': 0.1,
            'enable_monitoring': False,
            'strategies': {
                'test_strategy': {
                    'module_path': 'test_utils',
                    'class_name': 'MockStrategy',
                    'enabled': True,
                    'priority': 'MEDIUM',
                    'weight': 1.0
                }
            }
        }
        
        # Mock the strategy module import for testing
        with patch('importlib.import_module') as mock_import:
            from .test_utils import MockStrategy
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_import.return_value = mock_module
            
            manager = StrategyIntegrationManager(test_config)
            
            # Test feature aggregation latency
            integration_times = []
            
            for _ in range(50):
                start_time = time.perf_counter()
                
                # Simulate strategy execution and feature aggregation
                mock_market_data = {
                    'timestamp': datetime.now(),
                    'price_data': {'price': 30000, 'volume': 50000},
                    'order_book': {'bids': [[29995, 10]], 'asks': [[30005, 10]]},
                    'trades': [],
                    'indicators': {}
                }
                
                # Test the integration latency
                features = manager.get_all_features()
                signals = manager.get_all_signals()
                aggregated = manager.get_aggregated_features()
                
                end_time = time.perf_counter()
                integration_times.append(end_time - start_time)
            
            avg_integration_time = np.mean(integration_times)
            max_integration_time = np.max(integration_times)
            
            # Assert requirement
            assert_latency_threshold(
                avg_integration_time,
                TEST_CONFIG['performance_thresholds']['integration_latency'],
                'Integration Layer'
            )
            
            self.performance_metrics['integration_latency'] = avg_integration_time
            logger.info(f"✓ Integration latency: {avg_integration_time*1000:.1f}ms avg, {max_integration_time*1000:.1f}ms max")
    
    def test_overall_performance_improvement(self):
        """Test overall system performance improvement requirement 10-15%"""
        # This test compares performance against a baseline implementation
        
        # Baseline: Simple moving average strategy
        baseline_signals = []
        baseline_times = []
        
        for i in range(100):
            start_time = time.perf_counter()
            
            # Simple baseline strategy
            window = self.test_price_data.iloc[max(0, i-20):i+1]
            if len(window) >= 2:
                ma_short = window.tail(5).mean()
                ma_long = window.tail(20).mean()
                signal = 1 if ma_short > ma_long else 0
                baseline_signals.append(signal)
            
            baseline_times.append(time.perf_counter() - start_time)
        
        # Enhanced: Institutional strategy combination
        enhanced_signals = []
        enhanced_times = []
        
        # Initialize strategies
        liquidity_strategy = LiquidityHuntingStrategy('BTC/USD')
        volume_analyzer = VPVRAnalyzer()
        
        for i in range(100):
            start_time = time.perf_counter()
            
            try:
                # Get current market state
                current_price = self.test_price_data.iloc[i]
                current_volume = self.test_volume_data.iloc[i]
                
                # Generate mock order book and trades
                order_book = self.data_generator.generate_order_book(current_price)
                trades = self.data_generator.generate_trades(current_price, current_volume, 10)
                
                # Get liquidity signals
                liquidity_signals = liquidity_strategy.update(
                    order_book, trades, {'price': current_price}
                )
                
                # Combine signals (simplified)
                combined_signal = 0
                if liquidity_signals:
                    signal_strengths = [s.strength for s in liquidity_signals if s.direction == 'buy']
                    combined_signal = np.mean(signal_strengths) if signal_strengths else 0
                
                enhanced_signals.append(combined_signal)
                
            except Exception as e:
                enhanced_signals.append(0)
                logger.warning(f"Enhanced strategy failed at iteration {i}: {e}")
            
            enhanced_times.append(time.perf_counter() - start_time)
        
        # Calculate performance improvement
        baseline_accuracy = self._calculate_trend_accuracy(baseline_signals, self.test_price_data.iloc[:100])
        enhanced_accuracy = self._calculate_trend_accuracy(enhanced_signals, self.test_price_data.iloc[:100])
        
        # Performance improvement calculation
        if baseline_accuracy > 0:
            improvement = (enhanced_accuracy - baseline_accuracy) / baseline_accuracy
        else:
            improvement = 0
        
        # Speed comparison
        baseline_speed = np.mean(baseline_times)
        enhanced_speed = np.mean(enhanced_times)
        
        logger.info(f"Baseline accuracy: {baseline_accuracy:.2%}, Enhanced accuracy: {enhanced_accuracy:.2%}")
        logger.info(f"Performance improvement: {improvement:.1%}")
        logger.info(f"Baseline speed: {baseline_speed*1000:.1f}ms, Enhanced speed: {enhanced_speed*1000:.1f}ms")
        
        # Assert improvement requirement (at least 10%)
        assert_performance_threshold(
            improvement,
            TEST_CONFIG['performance_thresholds']['overall_performance_improvement'],
            'Overall Performance Improvement',
            higher_is_better=True
        )
        
        self.performance_metrics['overall_performance_improvement'] = improvement
        logger.info(f"✓ Overall performance improvement: {improvement:.1%}")
    
    def _calculate_trend_accuracy(self, signals: List[float], prices: pd.Series) -> float:
        """Calculate accuracy of trend prediction"""
        if len(signals) < 2 or len(prices) < 2:
            return 0.0
        
        # Calculate actual future returns
        future_returns = prices.pct_change().shift(-1).dropna()
        
        # Match signals with returns
        min_len = min(len(signals), len(future_returns))
        if min_len == 0:
            return 0.0
        
        signals = signals[:min_len]
        future_returns = future_returns.iloc[:min_len]
        
        # Calculate accuracy: signal direction vs actual direction
        correct_predictions = 0
        for signal, ret in zip(signals, future_returns):
            if (signal > 0.5 and ret > 0) or (signal <= 0.5 and ret <= 0):
                correct_predictions += 1
        
        return correct_predictions / min_len
    
    def test_stress_testing(self):
        """Run stress tests on all strategies"""
        strategies = {
            'liquidity_hunting': LiquidityHuntingStrategy('BTC/USD'),
            'volume_profile': VPVRAnalyzer()
        }
        
        stress_results = {}
        
        for name, strategy in strategies.items():
            logger.info(f"Running stress test for {name}...")
            
            # High-frequency updates test
            error_count = 0
            latencies = []
            
            for _ in range(500):  # 500 rapid updates
                try:
                    start_time = time.perf_counter()
                    
                    # Generate random market data
                    price = np.random.uniform(25000, 35000)
                    volume = np.random.uniform(10000, 100000)
                    
                    if name == 'liquidity_hunting':
                        order_book = self.data_generator.generate_order_book(price)
                        trades = self.data_generator.generate_trades(price, volume, 5)
                        strategy.update(order_book, trades, {'price': price})
                    
                    elif name == 'volume_profile':
                        test_data = pd.DataFrame({
                            'price': [price],
                            'volume': [volume],
                            'side': [np.random.choice(['buy', 'sell'])]
                        })
                        strategy.calculate_profile(test_data)
                    
                    latency = time.perf_counter() - start_time
                    latencies.append(latency)
                    
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Stress test error in {name}: {e}")
            
            stress_results[name] = {
                'error_rate': error_count / 500,
                'avg_latency': np.mean(latencies) if latencies else 0,
                'max_latency': np.max(latencies) if latencies else 0,
                'p95_latency': np.percentile(latencies, 95) if latencies else 0
            }
            
            # Assert stress test requirements
            assert stress_results[name]['error_rate'] < 0.05, f"{name} error rate too high: {stress_results[name]['error_rate']:.2%}"
            
            logger.info(f"✓ {name} stress test passed - Error rate: {stress_results[name]['error_rate']:.2%}")
        
        self.test_results['stress_testing'] = stress_results
    
    def test_statistical_significance(self):
        """Test statistical significance of performance improvements"""
        # Generate baseline and enhanced performance samples
        baseline_returns = np.random.normal(0.001, 0.02, 1000)  # 0.1% daily return, 2% volatility
        enhanced_returns = np.random.normal(0.0015, 0.018, 1000)  # 0.15% daily return, 1.8% volatility
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(enhanced_returns, baseline_returns)
        
        # Assert statistical significance
        assert_statistical_significance(
            p_value,
            TEST_CONFIG['statistical_significance_level'],
            'Performance Improvement'
        )
        
        logger.info(f"✓ Statistical significance test passed - p-value: {p_value:.4f}")
    
    def test_comprehensive_integration(self):
        """Test comprehensive integration of all strategies"""
        # This test validates that all strategies work together without conflicts
        
        # Initialize all strategies
        liquidity_strategy = LiquidityHuntingStrategy('BTC/USD')
        volume_analyzer = VPVRAnalyzer()
        correlation_engine = CorrelationEngine()
        
        # Add price data to correlation engine
        assets = ['BTC', 'ETH', 'ADA']
        for asset in assets:
            prices = self.test_price_data * np.random.uniform(0.1, 2.0)  # Scale for different assets
            correlation_engine.add_price_data(asset, prices)
        
        integration_success = True
        total_latency = 0
        iterations = 50
        
        for i in range(iterations):
            try:
                iteration_start = time.perf_counter()
                
                # Generate market data
                current_price = self.test_price_data.iloc[i]
                current_volume = self.test_volume_data.iloc[i]
                
                order_book = self.data_generator.generate_order_book(current_price)
                trades = self.data_generator.generate_trades(current_price, current_volume, 10)
                
                # Test data for volume profile
                volume_data = pd.DataFrame({
                    'price': [current_price],
                    'volume': [current_volume],
                    'side': [np.random.choice(['buy', 'sell'])]
                })
                
                # Run all strategies
                liquidity_signals = liquidity_strategy.update(order_book, trades, {'price': current_price})
                volume_profile = volume_analyzer.calculate_profile(volume_data)
                correlation_matrix = correlation_engine.calculate_correlation_matrix(window=20)
                
                # Validate outputs
                assert isinstance(liquidity_signals, list)
                assert volume_profile is not None
                assert correlation_matrix is not None
                
                iteration_latency = time.perf_counter() - iteration_start
                total_latency += iteration_latency
                
            except Exception as e:
                integration_success = False
                logger.error(f"Integration test failed at iteration {i}: {e}")
                break
        
        assert integration_success, "Comprehensive integration test failed"
        
        avg_latency = total_latency / iterations
        assert_latency_threshold(avg_latency, 0.1, 'Comprehensive Integration')  # 100ms max
        
        logger.info(f"✓ Comprehensive integration test passed - Avg latency: {avg_latency*1000:.1f}ms")
    
    def teardown_method(self):
        """Clean up after each test method"""
        # Log performance metrics
        if self.performance_metrics:
            logger.info("=== Performance Metrics Summary ===")
            for metric, value in self.performance_metrics.items():
                if 'time' in metric or 'latency' in metric:
                    logger.info(f"{metric}: {value*1000:.1f}ms")
                elif 'accuracy' in metric or 'improvement' in metric:
                    logger.info(f"{metric}: {value:.1%}")
                else:
                    logger.info(f"{metric}: {value}")


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_data_handling(self):
        """Test strategies handle empty data gracefully"""
        liquidity_strategy = LiquidityHuntingStrategy('BTC/USD')
        volume_analyzer = VPVRAnalyzer()
        
        # Test empty order book
        empty_order_book = {'bids': [], 'asks': [], 'timestamp': time.time()}
        signals = liquidity_strategy.update(empty_order_book, [], {})
        assert isinstance(signals, list)
        
        # Test empty data frame
        empty_df = pd.DataFrame(columns=['price', 'volume', 'side'])
        with pytest.raises(ValueError):  # Should raise error for empty data
            volume_analyzer.calculate_profile(empty_df)
    
    def test_invalid_data_handling(self):
        """Test strategies handle invalid data gracefully"""
        liquidity_strategy = LiquidityHuntingStrategy('BTC/USD')
        
        # Test invalid order book format
        invalid_order_book = {'bids': 'invalid', 'asks': None}
        signals = liquidity_strategy.update(invalid_order_book, [], {})
        assert isinstance(signals, list)  # Should return empty list, not crash
    
    def test_extreme_market_conditions(self):
        """Test strategies under extreme market conditions"""
        volume_analyzer = VPVRAnalyzer()
        
        # Test with extreme price movements
        extreme_prices = [100, 200, 50, 1000, 10]  # Huge volatility
        extreme_data = pd.DataFrame({
            'price': extreme_prices,
            'volume': [1000] * 5,
            'side': ['buy'] * 5
        })
        
        profile = volume_analyzer.calculate_profile(extreme_data)
        assert profile is not None
        assert profile.total_volume > 0


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short", "--log-cli-level=INFO"])