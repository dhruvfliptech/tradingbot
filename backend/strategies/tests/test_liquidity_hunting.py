"""
Liquidity Hunting Strategy Test Suite
====================================

Comprehensive tests for the LiquidityHuntingStrategy implementation:
- Iceberg order detection accuracy
- Stop hunting pattern recognition
- Accumulation/distribution analysis
- Liquidity squeeze detection
- Performance and latency benchmarks
- Edge case handling

Target Requirements:
- Liquidity detection accuracy >80%
- Processing latency <50ms per update
- Signal strength validation
- Microstructure analysis accuracy
"""

import pytest
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
    MockMarketData, assert_performance_threshold, assert_latency_threshold
)

# Import strategy under test
try:
    from ..institutional.liquidity_hunting import (
        LiquidityHuntingStrategy, LiquiditySignal, MarketMicrostructure
    )
except ImportError as e:
    pytest.skip(f"Could not import liquidity hunting strategy: {e}", allow_module_level=True)

from . import TEST_CONFIG

logger = logging.getLogger(__name__)


class TestLiquidityHuntingStrategy:
    """Test suite for LiquidityHuntingStrategy"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.data_generator = MockDataGenerator(TEST_CONFIG['mock_data_seed'])
        self.benchmark = PerformanceBenchmark()
        self.strategy_tester = StrategyTester(LiquidityHuntingStrategy)
        
        # Initialize strategy with test configuration
        self.strategy = LiquidityHuntingStrategy(
            symbol='BTC/USD',
            lookback_periods=50,
            sensitivity=0.7,
            min_signal_strength=0.6
        )
        
        # Test data
        self.test_price_data = self.data_generator.generate_price_series(1000)
        self.performance_metrics = {}
    
    def test_strategy_initialization(self):
        """Test strategy initialization and configuration"""
        # Test default initialization
        strategy = LiquidityHuntingStrategy('BTC/USD')
        assert strategy.symbol == 'BTC/USD'
        assert strategy.lookback_periods == 100
        assert strategy.sensitivity == 0.7
        assert strategy.min_signal_strength == 0.6
        
        # Test custom configuration
        custom_strategy = LiquidityHuntingStrategy(
            symbol='ETH/USD',
            lookback_periods=200,
            sensitivity=0.8,
            min_signal_strength=0.7
        )
        assert custom_strategy.symbol == 'ETH/USD'
        assert custom_strategy.lookback_periods == 200
        assert custom_strategy.sensitivity == 0.8
        assert custom_strategy.min_signal_strength == 0.7
        
        # Test data structures initialization
        assert len(strategy.order_book_history) == 0
        assert len(strategy.trade_history) == 0
        assert len(strategy.signal_history) == 0
        assert isinstance(strategy.liquidity_pools, dict)
        
        logger.info("✓ Strategy initialization test passed")
    
    def test_iceberg_order_detection(self):
        """Test iceberg order detection accuracy"""
        correct_detections = 0
        total_tests = 100
        detection_latencies = []
        
        for i in range(total_tests):
            # Generate base market data
            price = 30000 + np.random.uniform(-2000, 2000)
            order_book = self.data_generator.generate_order_book(price)
            trades = self.data_generator.generate_trades(price, 50000, 20)
            
            # Randomly add iceberg patterns
            has_iceberg = np.random.random() < 0.3  # 30% chance
            
            if has_iceberg:
                # Simulate iceberg order characteristics
                side = np.random.choice(['bids', 'asks'])
                level_idx = np.random.randint(0, min(10, len(order_book[side])))
                
                # Increase volume at specific level (iceberg signature)
                original_volume = order_book[side][level_idx][1]
                order_book[side][level_idx][1] = original_volume * np.random.uniform(3, 8)
                
                # Add historical reloading pattern
                for _ in range(5):  # Simulate order reloading
                    historical_book = self.data_generator.generate_order_book(price)
                    historical_book[side][level_idx][1] = original_volume * np.random.uniform(2, 6)
                    self.strategy.order_book_history.append(historical_book)
            
            # Test detection
            start_time = time.perf_counter()
            signals = self.strategy.update(order_book, trades, {'price': price})
            detection_time = time.perf_counter() - start_time
            detection_latencies.append(detection_time)
            
            # Check detection accuracy
            iceberg_detected = any(signal.signal_type == 'iceberg' for signal in signals)
            
            if (has_iceberg and iceberg_detected) or (not has_iceberg and not iceberg_detected):
                correct_detections += 1
            
            # Validate signal properties if detected
            for signal in signals:
                if signal.signal_type == 'iceberg':
                    assert 0 <= signal.strength <= 1
                    assert signal.direction in ['buy', 'sell']
                    assert signal.price_level > 0
                    assert signal.volume_estimate > 0
                    assert isinstance(signal.metadata, dict)
        
        # Calculate accuracy
        accuracy = correct_detections / total_tests
        avg_latency = np.mean(detection_latencies)
        max_latency = np.max(detection_latencies)
        
        # Assert requirements
        assert_performance_threshold(
            accuracy, 0.80, 'Iceberg Detection Accuracy', higher_is_better=True
        )
        
        assert_latency_threshold(
            avg_latency, 0.05, 'Iceberg Detection Latency'  # 50ms
        )
        
        self.performance_metrics['iceberg_detection_accuracy'] = accuracy
        self.performance_metrics['iceberg_detection_latency'] = avg_latency
        
        logger.info(f"✓ Iceberg detection - Accuracy: {accuracy:.2%}, Avg latency: {avg_latency*1000:.1f}ms")
    
    def test_stop_hunting_detection(self):
        """Test stop hunting pattern detection"""
        correct_detections = 0
        total_tests = 100
        
        for i in range(total_tests):
            price = 30000 + np.random.uniform(-1000, 1000)
            
            # Create market data with stop hunting setup
            market_data = {
                'price': price,
                'high_24h': price * 1.02,
                'low_24h': price * 0.98,
                'sma_50': price * 0.995,
                'sma_200': price * 0.99,
                'support_levels': [price * 0.985, price * 0.975],
                'resistance_levels': [price * 1.015, price * 1.025]
            }
            
            # Generate order book with potential stop hunting setup
            order_book = self.data_generator.generate_order_book(price)
            trades = self.data_generator.generate_trades(price, 50000, 30)
            
            # Simulate stop hunting conditions
            has_stop_hunt = np.random.random() < 0.25  # 25% chance
            
            if has_stop_hunt:
                # Add order flow imbalance toward stop levels
                if np.random.random() > 0.5:  # Hunt stops above
                    # Increase buying pressure
                    for j in range(5):
                        order_book['bids'][j][1] *= np.random.uniform(1.5, 3.0)
                else:  # Hunt stops below
                    # Increase selling pressure
                    for j in range(5):
                        order_book['asks'][j][1] *= np.random.uniform(1.5, 3.0)
            
            # Test detection
            signals = self.strategy.update(order_book, trades, market_data)
            
            # Check for stop hunting signals
            stop_hunt_detected = any(signal.signal_type == 'stop_hunt' for signal in signals)
            
            if (has_stop_hunt and stop_hunt_detected) or (not has_stop_hunt and not stop_hunt_detected):
                correct_detections += 1
            
            # Validate stop hunt signals
            for signal in signals:
                if signal.signal_type == 'stop_hunt':
                    assert signal.strength >= self.strategy.stop_hunt_threshold
                    assert 'level_type' in signal.metadata
                    assert 'distance_from_price' in signal.metadata
        
        accuracy = correct_detections / total_tests
        
        # Assert accuracy requirement (lower threshold for stop hunting)
        assert_performance_threshold(
            accuracy, 0.70, 'Stop Hunting Detection Accuracy', higher_is_better=True
        )
        
        self.performance_metrics['stop_hunting_accuracy'] = accuracy
        logger.info(f"✓ Stop hunting detection accuracy: {accuracy:.2%}")
    
    def test_accumulation_distribution_analysis(self):
        """Test accumulation/distribution pattern detection"""
        correct_identifications = 0
        total_tests = 100
        
        for i in range(total_tests):
            # Generate trades with accumulation/distribution patterns
            price = 30000 + np.random.uniform(-500, 500)
            trades = []
            
            # Create pattern type
            pattern_type = np.random.choice(['accumulation', 'distribution', 'neutral'])
            
            # Generate trades based on pattern
            for j in range(50):
                trade_price = price + np.random.normal(0, price * 0.001)
                
                if pattern_type == 'accumulation':
                    # More buying on weakness (lower prices)
                    buy_probability = 0.7 if trade_price < price else 0.3
                    trade_volume = np.random.exponential(1000) * (1.5 if trade_price < price else 0.8)
                elif pattern_type == 'distribution':
                    # More selling on strength (higher prices)
                    buy_probability = 0.3 if trade_price > price else 0.7
                    trade_volume = np.random.exponential(1000) * (1.5 if trade_price > price else 0.8)
                else:  # neutral
                    buy_probability = 0.5
                    trade_volume = np.random.exponential(1000)
                
                is_buy = np.random.random() < buy_probability
                
                trades.append({
                    'price': trade_price,
                    'volume': trade_volume,
                    'side': 'buy' if is_buy else 'sell',
                    'timestamp': time.time() - (50 - j) * 60
                })
            
            # Generate order book
            order_book = self.data_generator.generate_order_book(price)
            
            # Test pattern detection
            signals = self.strategy.update(order_book, trades, {'price': price})
            
            # Check pattern identification
            detected_pattern = None
            for signal in signals:
                if signal.signal_type in ['accumulation', 'distribution']:
                    detected_pattern = signal.signal_type
                    break
            
            # Scoring (more lenient for this complex pattern)
            if pattern_type == 'neutral' and detected_pattern is None:
                correct_identifications += 1
            elif pattern_type == detected_pattern:
                correct_identifications += 1
            elif pattern_type != 'neutral' and detected_pattern is not None:
                # Partial credit for detecting some pattern
                correct_identifications += 0.5
        
        accuracy = correct_identifications / total_tests
        
        # Assert accuracy (lower threshold due to complexity)
        assert_performance_threshold(
            accuracy, 0.65, 'Accumulation/Distribution Accuracy', higher_is_better=True
        )
        
        self.performance_metrics['accumulation_distribution_accuracy'] = accuracy
        logger.info(f"✓ Accumulation/Distribution analysis accuracy: {accuracy:.2%}")
    
    def test_liquidity_squeeze_detection(self):
        """Test liquidity squeeze detection"""
        correct_detections = 0
        total_tests = 100
        
        for i in range(total_tests):
            price = 30000 + np.random.uniform(-1000, 1000)
            
            # Create squeeze condition
            has_squeeze = np.random.random() < 0.2  # 20% chance
            
            if has_squeeze:
                # Reduce order book depth to simulate squeeze
                order_book = {
                    'bids': [[price - j, np.random.uniform(0.1, 1.0)] for j in range(1, 11)],
                    'asks': [[price + j, np.random.uniform(0.1, 1.0)] for j in range(1, 11)],
                    'timestamp': time.time()
                }
                
                # Add historical data with normal depth for comparison
                for _ in range(25):
                    normal_book = self.data_generator.generate_order_book(price)
                    self.strategy.order_book_history.append(normal_book)
            else:
                # Normal order book depth
                order_book = self.data_generator.generate_order_book(price)
            
            trades = self.data_generator.generate_trades(price, 20000, 10)
            
            # Test detection
            signals = self.strategy.update(order_book, trades, {'price': price})
            
            # Check squeeze detection
            squeeze_detected = any(signal.signal_type == 'squeeze' for signal in signals)
            
            if (has_squeeze and squeeze_detected) or (not has_squeeze and not squeeze_detected):
                correct_detections += 1
            
            # Validate squeeze signals
            for signal in signals:
                if signal.signal_type == 'squeeze':
                    assert signal.direction == 'neutral'
                    assert 'depth_ratio' in signal.metadata
                    assert 'spread_ratio' in signal.metadata
        
        accuracy = correct_detections / total_tests
        
        # Assert accuracy
        assert_performance_threshold(
            accuracy, 0.75, 'Liquidity Squeeze Detection Accuracy', higher_is_better=True
        )
        
        self.performance_metrics['liquidity_squeeze_accuracy'] = accuracy
        logger.info(f"✓ Liquidity squeeze detection accuracy: {accuracy:.2%}")
    
    def test_microstructure_analysis(self):
        """Test market microstructure analysis"""
        microstructure_results = []
        calculation_times = []
        
        for _ in range(50):
            price = 30000 + np.random.uniform(-1000, 1000)
            order_book = self.data_generator.generate_order_book(price)
            trades = self.data_generator.generate_trades(price, 50000, 30)
            
            start_time = time.perf_counter()
            
            # Trigger microstructure analysis through strategy update
            signals = self.strategy.update(order_book, trades, {'price': price})
            microstructure = self.strategy.current_microstructure
            
            calculation_time = time.perf_counter() - start_time
            calculation_times.append(calculation_time)
            
            if microstructure:
                microstructure_results.append(microstructure)
                
                # Validate microstructure metrics
                assert microstructure.bid_ask_spread >= 0
                assert microstructure.effective_spread >= 0
                assert -1 <= microstructure.order_flow_imbalance <= 1
                assert microstructure.kyle_lambda >= 0
                assert 0 <= microstructure.get_liquidity_score() <= 1
        
        # Performance assertions
        avg_calculation_time = np.mean(calculation_times)
        assert_latency_threshold(
            avg_calculation_time, 0.02, 'Microstructure Analysis'  # 20ms
        )
        
        # Quality assertions
        assert len(microstructure_results) > 40  # At least 80% successful calculations
        
        if microstructure_results:
            # Check distribution of liquidity scores
            liquidity_scores = [ms.get_liquidity_score() for ms in microstructure_results]
            score_std = np.std(liquidity_scores)
            assert score_std > 0.01  # Should have some variation
        
        self.performance_metrics['microstructure_calculation_time'] = avg_calculation_time
        logger.info(f"✓ Microstructure analysis - Avg time: {avg_calculation_time*1000:.1f}ms")
    
    def test_signal_strength_validation(self):
        """Test signal strength calculation and validation"""
        signal_strengths = []
        
        for _ in range(100):
            price = 30000 + np.random.uniform(-1000, 1000)
            order_book = self.data_generator.generate_order_book(price)
            trades = self.data_generator.generate_trades(price, 50000, 25)
            
            signals = self.strategy.update(order_book, trades, {'price': price})
            
            for signal in signals:
                signal_strengths.append(signal.strength)
                
                # Validate signal strength properties
                assert 0 <= signal.strength <= 1, f"Invalid signal strength: {signal.strength}"
                assert signal.strength >= self.strategy.min_signal_strength
                
                # Validate signal components
                assert signal.timestamp is not None
                assert signal.symbol == 'BTC/USD'
                assert signal.signal_type in ['iceberg', 'stop_hunt', 'accumulation', 'distribution', 'squeeze']
                assert signal.direction in ['buy', 'sell', 'neutral']
                assert signal.price_level > 0
                assert signal.volume_estimate > 0
        
        if signal_strengths:
            avg_strength = np.mean(signal_strengths)
            min_strength = np.min(signal_strengths)
            max_strength = np.max(signal_strengths)
            
            # Validate signal strength distribution
            assert min_strength >= self.strategy.min_signal_strength
            assert max_strength <= 1.0
            assert 0.6 <= avg_strength <= 0.9  # Reasonable average
            
            logger.info(f"✓ Signal strength validation - Avg: {avg_strength:.2f}, Range: [{min_strength:.2f}, {max_strength:.2f}]")
        else:
            logger.warning("No signals generated during strength validation test")
    
    def test_performance_benchmarks(self):
        """Test overall strategy performance benchmarks"""
        # Test update frequency performance
        update_times = []
        memory_usage = []
        
        for i in range(200):
            price = 30000 + np.random.uniform(-2000, 2000)
            order_book = self.data_generator.generate_order_book(price)
            trades = self.data_generator.generate_trades(price, 50000, 20)
            
            start_time = time.perf_counter()
            signals = self.strategy.update(order_book, trades, {'price': price})
            update_time = time.perf_counter() - start_time
            
            update_times.append(update_time)
            
            # Simulate memory usage tracking
            memory_usage.append(
                len(self.strategy.order_book_history) +
                len(self.strategy.trade_history) +
                len(self.strategy.signal_history)
            )
        
        # Performance metrics
        avg_update_time = np.mean(update_times)
        p95_update_time = np.percentile(update_times, 95)
        p99_update_time = np.percentile(update_times, 99)
        max_memory = np.max(memory_usage)
        
        # Assert performance requirements
        assert_latency_threshold(avg_update_time, 0.05, 'Strategy Update Average')  # 50ms
        assert_latency_threshold(p95_update_time, 0.1, 'Strategy Update P95')  # 100ms
        assert_latency_threshold(p99_update_time, 0.2, 'Strategy Update P99')  # 200ms
        
        # Memory usage should be bounded
        assert max_memory <= self.strategy.lookback_periods * 3  # Reasonable memory bound
        
        self.performance_metrics.update({
            'avg_update_time': avg_update_time,
            'p95_update_time': p95_update_time,
            'p99_update_time': p99_update_time,
            'max_memory_usage': max_memory
        })
        
        logger.info(f"✓ Performance benchmarks - Avg: {avg_update_time*1000:.1f}ms, "
                   f"P95: {p95_update_time*1000:.1f}ms, P99: {p99_update_time*1000:.1f}ms")
    
    def test_rl_feature_extraction(self):
        """Test RL feature vector extraction"""
        feature_vectors = []
        extraction_times = []
        
        for _ in range(50):
            price = 30000 + np.random.uniform(-1000, 1000)
            order_book = self.data_generator.generate_order_book(price)
            trades = self.data_generator.generate_trades(price, 50000, 20)
            
            # Update strategy to generate features
            self.strategy.update(order_book, trades, {'price': price})
            
            start_time = time.perf_counter()
            features = self.strategy.get_rl_features()
            extraction_time = time.perf_counter() - start_time
            
            extraction_times.append(extraction_time)
            feature_vectors.append(features)
            
            # Validate feature vector
            assert isinstance(features, np.ndarray)
            assert len(features) > 0
            assert not np.any(np.isnan(features))  # No NaN values
            assert not np.any(np.isinf(features))  # No infinite values
        
        # Performance validation
        avg_extraction_time = np.mean(extraction_times)
        assert_latency_threshold(avg_extraction_time, 0.001, 'RL Feature Extraction')  # 1ms
        
        # Feature consistency validation
        if len(feature_vectors) > 1:
            feature_lengths = [len(fv) for fv in feature_vectors]
            assert len(set(feature_lengths)) == 1, "Inconsistent feature vector lengths"
            
            # Check feature value ranges
            all_features = np.vstack(feature_vectors)
            feature_means = np.mean(all_features, axis=0)
            feature_stds = np.std(all_features, axis=0)
            
            # Features should have reasonable ranges
            assert np.all(np.abs(feature_means) < 10), "Feature means too large"
            assert np.all(feature_stds > 0.001), "Features have insufficient variation"
        
        self.performance_metrics['rl_feature_extraction_time'] = avg_extraction_time
        logger.info(f"✓ RL feature extraction - Time: {avg_extraction_time*1000:.2f}ms, "
                   f"Features: {len(feature_vectors[0]) if feature_vectors else 0}")
    
    def test_liquidity_metrics(self):
        """Test liquidity metrics calculation and reporting"""
        for _ in range(10):
            price = 30000 + np.random.uniform(-1000, 1000)
            order_book = self.data_generator.generate_order_book(price)
            trades = self.data_generator.generate_trades(price, 50000, 20)
            
            # Update strategy
            self.strategy.update(order_book, trades, {'price': price})
        
        # Get liquidity metrics
        metrics = self.strategy.get_liquidity_metrics()
        
        # Validate metrics structure
        assert isinstance(metrics, dict)
        assert 'liquidity_score' in metrics
        assert 'tracked_pools' in metrics
        assert 'recent_signals' in metrics
        assert 'institutional_levels' in metrics
        
        # Validate metric values
        assert 0 <= metrics['liquidity_score'] <= 1
        assert metrics['tracked_pools'] >= 0
        assert metrics['recent_signals'] >= 0
        assert isinstance(metrics['institutional_levels'], list)
        
        # Validate microstructure data if available
        if metrics.get('microstructure'):
            microstructure = metrics['microstructure']
            assert 'bid_ask_spread' in microstructure
            assert 'effective_spread' in microstructure
            assert 'kyle_lambda' in microstructure
            assert 'order_flow_imbalance' in microstructure
        
        logger.info(f"✓ Liquidity metrics - Score: {metrics['liquidity_score']:.3f}, "
                   f"Pools: {metrics['tracked_pools']}, Signals: {metrics['recent_signals']}")


class TestLiquidityHuntingEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_data_handling(self):
        """Test handling of empty order books and trade lists"""
        strategy = LiquidityHuntingStrategy('BTC/USD')
        
        # Empty order book
        empty_order_book = {'bids': [], 'asks': [], 'timestamp': time.time()}
        signals = strategy.update(empty_order_book, [], {})
        assert isinstance(signals, list)
        assert len(signals) == 0
        
        # None data
        signals = strategy.update(None, None, None)
        assert isinstance(signals, list)
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data"""
        strategy = LiquidityHuntingStrategy('BTC/USD')
        
        # Malformed order book
        bad_order_book = {
            'bids': [['invalid', 'data'], [None, None]],
            'asks': 'not_a_list',
            'timestamp': 'not_a_number'
        }
        
        signals = strategy.update(bad_order_book, [], {})
        assert isinstance(signals, list)  # Should not crash
    
    def test_extreme_values(self):
        """Test handling of extreme market values"""
        strategy = LiquidityHuntingStrategy('BTC/USD')
        
        # Extreme prices and volumes
        extreme_order_book = {
            'bids': [[1e-10, 1e10], [1e10, 1e-10]],
            'asks': [[1e10, 1e10], [1e-10, 1e-10]],
            'timestamp': time.time()
        }
        
        extreme_trades = [
            {'price': 1e-10, 'volume': 1e10, 'side': 'buy'},
            {'price': 1e10, 'volume': 1e-10, 'side': 'sell'}
        ]
        
        signals = strategy.update(extreme_order_book, extreme_trades, {'price': 1e6})
        assert isinstance(signals, list)  # Should handle extreme values gracefully
    
    def test_high_frequency_updates(self):
        """Test strategy under high-frequency update conditions"""
        strategy = LiquidityHuntingStrategy('BTC/USD')
        data_generator = MockDataGenerator()
        
        update_count = 0
        error_count = 0
        
        for _ in range(1000):  # High frequency test
            try:
                price = 30000 + np.random.uniform(-100, 100)
                order_book = data_generator.generate_order_book(price)
                trades = data_generator.generate_trades(price, 10000, 5)
                
                signals = strategy.update(order_book, trades, {'price': price})
                assert isinstance(signals, list)
                update_count += 1
                
            except Exception as e:
                error_count += 1
                logger.warning(f"High frequency update error: {e}")
        
        error_rate = error_count / (update_count + error_count)
        assert error_rate < 0.01, f"High error rate under high frequency: {error_rate:.2%}"
        
        logger.info(f"✓ High frequency test - Updates: {update_count}, Errors: {error_count}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--log-cli-level=INFO"])