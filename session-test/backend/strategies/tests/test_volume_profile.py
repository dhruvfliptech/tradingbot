"""
Volume Profile Analysis Test Suite
==================================

Comprehensive tests for the VPVRAnalyzer (Volume Profile Visible Range):
- Volume profile calculation accuracy
- Support/resistance level detection
- Volume node identification (HVN/LVN)
- Point of Control (POC) calculation
- Value Area calculation
- Performance benchmarks
- RL feature extraction

Target Requirements:
- Volume profile computation <50ms
- POC accuracy validation
- Value area percentage ~70%
- Support/resistance detection accuracy
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import logging

# Test utilities
from .test_utils import (
    MockDataGenerator, PerformanceBenchmark, StrategyTester, BacktestValidator,
    MockMarketData, assert_performance_threshold, assert_latency_threshold
)

# Import strategy under test
try:
    from ..institutional.volume_profile import (
        VPVRAnalyzer, VolumeProfile, VolumeNode
    )
except ImportError as e:
    pytest.skip(f"Could not import volume profile analyzer: {e}", allow_module_level=True)

from . import TEST_CONFIG

logger = logging.getLogger(__name__)


class TestVPVRAnalyzer:
    """Test suite for VPVRAnalyzer"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.data_generator = MockDataGenerator(TEST_CONFIG['mock_data_seed'])
        self.benchmark = PerformanceBenchmark()
        
        # Initialize analyzer with test configuration
        self.analyzer = VPVRAnalyzer(
            tick_size=0.01,
            value_area_percentage=70.0,
            min_node_volume_pct=2.0,
            hvn_threshold_pct=150.0,
            lvn_threshold_pct=50.0
        )
        
        # Generate test data
        self.test_price_data = self.data_generator.generate_price_series(1000)
        self.test_volume_data = self.data_generator.generate_volume_series(self.test_price_data)
        
        self.performance_metrics = {}
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization and configuration"""
        # Test default initialization
        default_analyzer = VPVRAnalyzer()
        assert default_analyzer.tick_size == 0.01
        assert default_analyzer.value_area_percentage == 70.0
        
        # Test custom configuration
        assert self.analyzer.tick_size == 0.01
        assert self.analyzer.value_area_percentage == 70.0
        assert self.analyzer.hvn_threshold_pct == 150.0
        assert self.analyzer.lvn_threshold_pct == 50.0
        
        # Test cache initialization
        assert isinstance(self.analyzer.profile_cache, dict)
        assert isinstance(self.analyzer.node_cache, dict)
        
        logger.info("✓ Analyzer initialization test passed")
    
    def test_volume_profile_calculation(self):
        """Test volume profile calculation accuracy and performance"""
        calculation_times = []
        profiles = []
        
        for i in range(50):
            # Create test data segment
            start_idx = i * 20
            end_idx = start_idx + 100
            
            test_data = pd.DataFrame({
                'price': self.test_price_data.iloc[start_idx:end_idx].values,
                'volume': self.test_volume_data.iloc[start_idx:end_idx].values,
                'side': np.random.choice(['buy', 'sell'], end_idx - start_idx)
            }, index=self.test_price_data.index[start_idx:end_idx])
            
            # Test calculation performance
            start_time = time.perf_counter()
            profile = self.analyzer.calculate_profile(test_data)
            calculation_time = time.perf_counter() - start_time
            
            calculation_times.append(calculation_time)
            profiles.append(profile)
            
            # Validate profile properties
            assert isinstance(profile, VolumeProfile)
            assert profile.total_volume > 0
            assert profile.poc > 0
            assert profile.vah > profile.val  # Value area high > low
            assert profile.vwap > 0
            assert len(profile.price_levels) == len(profile.volumes)
            assert len(profile.buy_volumes) == len(profile.sell_volumes)
            
            # Value area should contain ~70% of volume
            va_percentage = profile.value_area_percentage
            assert 65 <= va_percentage <= 75, f"Value area percentage {va_percentage:.1f}% outside expected range"
            
            # POC should be at maximum volume
            max_vol_idx = np.argmax(profile.volumes)
            poc_price = profile.price_levels[max_vol_idx]
            assert abs(poc_price - profile.poc) < profile.tick_size * 2
        
        # Performance assertions
        avg_calculation_time = np.mean(calculation_times)
        max_calculation_time = np.max(calculation_times)
        p95_calculation_time = np.percentile(calculation_times, 95)
        
        assert_latency_threshold(
            avg_calculation_time,
            TEST_CONFIG['performance_thresholds']['volume_profile_computation_time'],
            'Volume Profile Calculation Average'
        )
        
        assert_latency_threshold(
            p95_calculation_time,
            TEST_CONFIG['performance_thresholds']['volume_profile_computation_time'] * 2,
            'Volume Profile Calculation P95'
        )
        
        self.performance_metrics['volume_profile_calculation_time'] = avg_calculation_time
        self.performance_metrics['volume_profile_p95_time'] = p95_calculation_time
        
        logger.info(f"✓ Volume profile calculation - Avg: {avg_calculation_time*1000:.1f}ms, "
                   f"P95: {p95_calculation_time*1000:.1f}ms, Max: {max_calculation_time*1000:.1f}ms")
    
    def test_poc_calculation_accuracy(self):
        """Test Point of Control calculation accuracy"""
        poc_accuracies = []
        
        for _ in range(30):
            # Create data with known volume concentration
            base_price = 30000
            concentration_price = base_price + np.random.uniform(-500, 500)
            
            # Generate data with volume spike at concentration price
            prices = []
            volumes = []
            sides = []
            
            for _ in range(200):
                price = np.random.normal(base_price, 200)
                
                # Higher volume near concentration price
                distance = abs(price - concentration_price)
                if distance < 50:  # Within $50 of concentration
                    volume = np.random.uniform(1000, 5000)  # High volume
                else:
                    volume = np.random.uniform(100, 800)  # Normal volume
                
                prices.append(price)
                volumes.append(volume)
                sides.append(np.random.choice(['buy', 'sell']))
            
            test_data = pd.DataFrame({
                'price': prices,
                'volume': volumes,
                'side': sides
            })
            
            # Calculate profile
            profile = self.analyzer.calculate_profile(test_data)
            
            # Check POC accuracy
            poc_error = abs(profile.poc - concentration_price)
            accuracy = max(0, 1 - poc_error / 100)  # Accuracy decreases with distance
            poc_accuracies.append(accuracy)
            
            # POC should be reasonably close to concentration point
            assert poc_error < 200, f"POC too far from concentration: {poc_error:.1f}"
        
        avg_poc_accuracy = np.mean(poc_accuracies)
        
        assert_performance_threshold(
            avg_poc_accuracy, 0.80, 'POC Calculation Accuracy', higher_is_better=True
        )
        
        self.performance_metrics['poc_accuracy'] = avg_poc_accuracy
        logger.info(f"✓ POC calculation accuracy: {avg_poc_accuracy:.2%}")
    
    def test_value_area_calculation(self):
        """Test Value Area calculation accuracy"""
        value_area_tests = []
        
        for _ in range(25):
            # Generate test data
            test_data = pd.DataFrame({
                'price': self.test_price_data.iloc[:300].values,
                'volume': self.test_volume_data.iloc[:300].values,
                'side': np.random.choice(['buy', 'sell'], 300)
            }, index=self.test_price_data.index[:300])
            
            profile = self.analyzer.calculate_profile(test_data)
            
            # Validate value area properties
            assert profile.vah > profile.val, "Value area high should be > low"
            assert profile.val <= profile.poc <= profile.vah, "POC should be within value area"
            
            # Value area percentage should be close to 70%
            va_percentage = profile.value_area_percentage
            assert 60 <= va_percentage <= 80, f"Value area percentage {va_percentage:.1f}% out of range"
            
            value_area_tests.append({
                'percentage': va_percentage,
                'width': profile.vah - profile.val,
                'poc_position': (profile.poc - profile.val) / (profile.vah - profile.val)
            })
        
        # Statistical validation
        percentages = [test['percentage'] for test in value_area_tests]
        avg_percentage = np.mean(percentages)
        std_percentage = np.std(percentages)
        
        # Should average close to 70% with reasonable consistency
        assert 68 <= avg_percentage <= 72, f"Average value area percentage {avg_percentage:.1f}% not close to 70%"
        assert std_percentage < 5, f"Value area percentage too variable: {std_percentage:.1f}%"
        
        logger.info(f"✓ Value area calculation - Avg: {avg_percentage:.1f}%, Std: {std_percentage:.1f}%")
    
    def test_volume_node_identification(self):
        """Test High Volume Node (HVN) and Low Volume Node (LVN) identification"""
        hvn_detections = []
        lvn_detections = []
        
        for _ in range(20):
            # Create data with artificial volume nodes
            base_price = 30000
            
            prices = []
            volumes = []
            sides = []
            
            # Create known HVN
            hvn_price = base_price + np.random.uniform(-300, 300)
            
            # Create known LVN
            lvn_price = base_price + np.random.uniform(-300, 300)
            while abs(lvn_price - hvn_price) < 100:  # Ensure separation
                lvn_price = base_price + np.random.uniform(-300, 300)
            
            for _ in range(300):
                price = np.random.normal(base_price, 150)
                
                # High volume at HVN
                if abs(price - hvn_price) < 20:
                    volume = np.random.uniform(2000, 8000)
                # Low volume at LVN
                elif abs(price - lvn_price) < 20:
                    volume = np.random.uniform(50, 200)
                else:
                    volume = np.random.uniform(500, 1500)
                
                prices.append(price)
                volumes.append(volume)
                sides.append(np.random.choice(['buy', 'sell']))
            
            test_data = pd.DataFrame({
                'price': prices,
                'volume': volumes,
                'side': sides
            })
            
            # Calculate profile and identify nodes
            profile = self.analyzer.calculate_profile(test_data)
            nodes = self.analyzer.identify_volume_nodes(profile)
            
            # Check HVN detection
            hvn_nodes = [node for node in nodes if node.node_type == 'HVN']
            hvn_detected = any(abs(node.price - hvn_price) < 50 for node in hvn_nodes)
            hvn_detections.append(hvn_detected)
            
            # Check LVN detection
            lvn_nodes = [node for node in nodes if node.node_type == 'LVN']
            lvn_detected = any(abs(node.price - lvn_price) < 50 for node in lvn_nodes)
            lvn_detections.append(lvn_detected)
            
            # Validate node properties
            for node in nodes:
                assert isinstance(node, VolumeNode)
                assert node.price > 0
                assert node.volume >= 0
                assert node.buy_volume >= 0
                assert node.sell_volume >= 0
                assert 0 <= node.strength <= 100
                assert node.node_type in ['HVN', 'LVN', 'POC']
        
        hvn_accuracy = np.mean(hvn_detections)
        lvn_accuracy = np.mean(lvn_detections)
        
        assert_performance_threshold(
            hvn_accuracy, 0.75, 'HVN Detection Accuracy', higher_is_better=True
        )
        assert_performance_threshold(
            lvn_accuracy, 0.70, 'LVN Detection Accuracy', higher_is_better=True
        )
        
        logger.info(f"✓ Volume node identification - HVN: {hvn_accuracy:.2%}, LVN: {lvn_accuracy:.2%}")
    
    def test_support_resistance_detection(self):
        """Test support and resistance level detection"""
        support_accuracies = []
        resistance_accuracies = []
        
        for _ in range(20):
            current_price = 30000 + np.random.uniform(-1000, 1000)
            
            # Create data with known support/resistance
            known_support = current_price * 0.98  # 2% below
            known_resistance = current_price * 1.025  # 2.5% above
            
            prices = []
            volumes = []
            sides = []
            
            for _ in range(400):
                price = np.random.normal(current_price, current_price * 0.01)
                
                # High volume at support/resistance
                if abs(price - known_support) < current_price * 0.005:  # Within 0.5%
                    volume = np.random.uniform(3000, 10000)
                elif abs(price - known_resistance) < current_price * 0.005:
                    volume = np.random.uniform(3000, 10000)
                else:
                    volume = np.random.uniform(800, 2000)
                
                prices.append(price)
                volumes.append(volume)
                sides.append(np.random.choice(['buy', 'sell']))
            
            test_data = pd.DataFrame({
                'price': prices,
                'volume': volumes,
                'side': sides
            })
            
            # Calculate profile and detect levels
            profile = self.analyzer.calculate_profile(test_data)
            sr_levels = self.analyzer.detect_support_resistance(profile, current_price)
            
            # Check support detection
            support_levels = sr_levels['support']
            support_detected = any(
                abs(level['price'] - known_support) < current_price * 0.01
                for level in support_levels
            )
            support_accuracies.append(support_detected)
            
            # Check resistance detection
            resistance_levels = sr_levels['resistance']
            resistance_detected = any(
                abs(level['price'] - known_resistance) < current_price * 0.01
                for level in resistance_levels
            )
            resistance_accuracies.append(resistance_detected)
            
            # Validate level properties
            for level in support_levels + resistance_levels:
                assert 'price' in level
                assert 'strength' in level
                assert 'volume' in level
                assert level['price'] > 0
                assert 0 <= level['strength'] <= 100
                assert level['volume'] >= 0
        
        support_accuracy = np.mean(support_accuracies)
        resistance_accuracy = np.mean(resistance_accuracies)
        
        assert_performance_threshold(
            support_accuracy, 0.70, 'Support Detection Accuracy', higher_is_better=True
        )
        assert_performance_threshold(
            resistance_accuracy, 0.70, 'Resistance Detection Accuracy', higher_is_better=True
        )
        
        logger.info(f"✓ Support/Resistance detection - Support: {support_accuracy:.2%}, "
                   f"Resistance: {resistance_accuracy:.2%}")
    
    def test_volume_distribution_analysis(self):
        """Test volume distribution analysis"""
        for _ in range(15):
            # Generate test data with different distribution patterns
            distribution_type = np.random.choice(['normal', 'skewed', 'bimodal'])
            
            if distribution_type == 'normal':
                prices = np.random.normal(30000, 300, 500)
                volumes = np.random.exponential(1000, 500)
            elif distribution_type == 'skewed':
                prices = np.random.gamma(2, 150, 500) + 29500  # Right skewed
                volumes = np.random.exponential(800, 500)
            else:  # bimodal
                prices1 = np.random.normal(29800, 100, 250)
                prices2 = np.random.normal(30200, 100, 250)
                prices = np.concatenate([prices1, prices2])
                volumes = np.random.exponential(1200, 500)
            
            test_data = pd.DataFrame({
                'price': prices,
                'volume': volumes,
                'side': np.random.choice(['buy', 'sell'], 500)
            })
            
            # Calculate profile and analyze distribution
            profile = self.analyzer.calculate_profile(test_data)
            analysis = self.analyzer.analyze_volume_distribution(profile)
            
            # Validate analysis results
            assert isinstance(analysis, dict)
            assert 'skewness' in analysis
            assert 'kurtosis' in analysis
            assert 'concentration' in analysis
            assert 'distribution_type' in analysis
            assert 'structure' in analysis
            assert 'bias' in analysis
            
            # Validate metric ranges
            assert -5 <= analysis['skewness'] <= 5  # Reasonable skewness range
            assert -5 <= analysis['kurtosis'] <= 20  # Reasonable kurtosis range
            assert 0 <= analysis['concentration'] <= 1  # Gini coefficient
            assert analysis['structure'] in ['accumulation', 'distribution', 'balanced']
            assert analysis['bias'] in ['bullish', 'bearish']
        
        logger.info("✓ Volume distribution analysis validation passed")
    
    def test_profile_comparison(self):
        """Test profile comparison functionality"""
        # Create two different profiles for comparison
        data1 = pd.DataFrame({
            'price': self.test_price_data.iloc[:200].values,
            'volume': self.test_volume_data.iloc[:200].values,
            'side': np.random.choice(['buy', 'sell'], 200)
        }, index=self.test_price_data.index[:200])
        
        data2 = pd.DataFrame({
            'price': self.test_price_data.iloc[200:400].values * 1.02,  # 2% higher prices
            'volume': self.test_volume_data.iloc[200:400].values * 1.5,  # 50% higher volume
            'side': np.random.choice(['buy', 'sell'], 200)
        }, index=self.test_price_data.index[200:400])
        
        profile1 = self.analyzer.calculate_profile(data1)
        profile2 = self.analyzer.calculate_profile(data2)
        
        # Compare profiles
        comparison = self.analyzer.compare_profiles(profile1, profile2)
        
        # Validate comparison results
        assert isinstance(comparison, dict)
        assert 'poc_shift' in comparison
        assert 'value_area_shift' in comparison
        assert 'volume_change' in comparison
        assert 'volume_change_pct' in comparison
        assert 'migration' in comparison
        assert 'value_area_trend' in comparison
        
        # Validate migration detection
        assert comparison['migration'] in ['upward', 'downward', 'stable']
        assert comparison['value_area_trend'] in ['expanding', 'contracting']
        
        # Check numerical consistency
        assert comparison['poc_shift'] == profile2.poc - profile1.poc
        assert comparison['volume_change'] == profile2.total_volume - profile1.total_volume
        
        if 'correlation' in comparison:
            assert -1 <= comparison['correlation'] <= 1
        
        logger.info(f"✓ Profile comparison - POC shift: {comparison['poc_shift']:.1f}, "
                   f"Migration: {comparison['migration']}")
    
    def test_composite_profile_creation(self):
        """Test composite profile creation from multiple timeframes"""
        profiles = []
        
        # Create profiles for different timeframes
        for i in range(3):
            start_idx = i * 150
            end_idx = start_idx + 200
            
            data = pd.DataFrame({
                'price': self.test_price_data.iloc[start_idx:end_idx].values,
                'volume': self.test_volume_data.iloc[start_idx:end_idx].values,
                'side': np.random.choice(['buy', 'sell'], end_idx - start_idx)
            })
            
            profile = self.analyzer.calculate_profile(data)
            profiles.append(profile)
        
        # Create composite profile
        weights = [0.5, 0.3, 0.2]  # Different weights for each timeframe
        composite = self.analyzer.calculate_composite_profile(profiles, weights)
        
        # Validate composite profile
        assert isinstance(composite, VolumeProfile)
        assert composite.total_volume > 0
        assert composite.poc > 0
        assert composite.vah > composite.val
        assert composite.timeframe == 'composite'
        
        # Composite should have characteristics influenced by all input profiles
        assert len(composite.price_levels) > 0
        assert len(composite.volumes) == len(composite.price_levels)
        
        logger.info(f"✓ Composite profile creation - Total volume: {composite.total_volume:.0f}, "
                   f"POC: {composite.poc:.1f}")
    
    def test_rl_feature_extraction(self):
        """Test RL feature extraction"""
        feature_vectors = []
        extraction_times = []
        
        for _ in range(30):
            current_price = 30000 + np.random.uniform(-1000, 1000)
            
            # Create test data
            test_data = pd.DataFrame({
                'price': np.random.normal(current_price, current_price * 0.01, 200),
                'volume': np.random.exponential(1000, 200),
                'side': np.random.choice(['buy', 'sell'], 200)
            })
            
            profile = self.analyzer.calculate_profile(test_data)
            
            # Test feature extraction
            start_time = time.perf_counter()
            features = self.analyzer.get_rl_features(profile, current_price)
            extraction_time = time.perf_counter() - start_time
            
            extraction_times.append(extraction_time)
            feature_vectors.append(features)
            
            # Validate features
            assert isinstance(features, np.ndarray)
            assert len(features) > 0
            assert not np.any(np.isnan(features))
            assert not np.any(np.isinf(features))
            
            # Features should be normalized/reasonable
            assert np.all(np.abs(features) < 100), "Features have extreme values"
        
        # Performance validation
        avg_extraction_time = np.mean(extraction_times)
        assert_latency_threshold(avg_extraction_time, 0.005, 'RL Feature Extraction')  # 5ms
        
        # Feature consistency
        if len(feature_vectors) > 1:
            feature_lengths = [len(fv) for fv in feature_vectors]
            assert len(set(feature_lengths)) == 1, "Inconsistent feature vector lengths"
            
            # Check feature variation
            all_features = np.vstack(feature_vectors)
            feature_stds = np.std(all_features, axis=0)
            assert np.all(feature_stds > 0.001), "Features lack variation"
        
        self.performance_metrics['rl_feature_extraction_time'] = avg_extraction_time
        logger.info(f"✓ RL feature extraction - Time: {avg_extraction_time*1000:.2f}ms, "
                   f"Features: {len(feature_vectors[0]) if feature_vectors else 0}")
    
    def test_performance_benchmarks(self):
        """Test overall analyzer performance benchmarks"""
        # Test different data sizes
        data_sizes = [100, 500, 1000, 2000]
        size_performance = {}
        
        for size in data_sizes:
            calculation_times = []
            
            for _ in range(10):
                # Generate data of specific size
                test_data = pd.DataFrame({
                    'price': np.random.normal(30000, 300, size),
                    'volume': np.random.exponential(1000, size),
                    'side': np.random.choice(['buy', 'sell'], size)
                })
                
                # Time the calculation
                start_time = time.perf_counter()
                profile = self.analyzer.calculate_profile(test_data)
                nodes = self.analyzer.identify_volume_nodes(profile)
                sr_levels = self.analyzer.detect_support_resistance(profile, 30000)
                calculation_time = time.perf_counter() - start_time
                
                calculation_times.append(calculation_time)
                
                # Validate results exist
                assert profile is not None
                assert isinstance(nodes, list)
                assert isinstance(sr_levels, dict)
            
            avg_time = np.mean(calculation_times)
            size_performance[size] = avg_time
            
            logger.info(f"Data size {size}: {avg_time*1000:.1f}ms avg")
        
        # Performance should scale reasonably
        smallest_time = size_performance[min(data_sizes)]
        largest_time = size_performance[max(data_sizes)]
        
        # Should not scale worse than O(n^2)
        size_ratio = max(data_sizes) / min(data_sizes)
        time_ratio = largest_time / smallest_time
        
        assert time_ratio <= size_ratio ** 1.5, f"Performance scaling too poor: {time_ratio:.1f}x for {size_ratio:.1f}x data"
        
        # Overall performance requirement
        overall_avg = np.mean(list(size_performance.values()))
        assert_latency_threshold(overall_avg, 0.1, 'Overall Performance')  # 100ms max
        
        self.performance_metrics['overall_avg_time'] = overall_avg
        logger.info(f"✓ Performance benchmarks - Overall avg: {overall_avg*1000:.1f}ms")


class TestVolumeProfileEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        analyzer = VPVRAnalyzer()
        
        # Empty DataFrame
        empty_df = pd.DataFrame(columns=['price', 'volume', 'side'])
        
        with pytest.raises(ValueError):
            analyzer.calculate_profile(empty_df)
    
    def test_single_price_data(self):
        """Test handling of single price point"""
        analyzer = VPVRAnalyzer()
        
        # All same price
        single_price_data = pd.DataFrame({
            'price': [30000] * 100,
            'volume': np.random.uniform(100, 1000, 100),
            'side': np.random.choice(['buy', 'sell'], 100)
        })
        
        profile = analyzer.calculate_profile(single_price_data)
        
        # Should handle gracefully
        assert profile is not None
        assert profile.poc == 30000
        assert profile.vah == profile.val  # Same price for all levels
    
    def test_extreme_values(self):
        """Test handling of extreme price and volume values"""
        analyzer = VPVRAnalyzer()
        
        # Extreme values
        extreme_data = pd.DataFrame({
            'price': [1e-6, 1e6, 0.01, 999999],
            'volume': [1e-3, 1e9, 1, 1e6],
            'side': ['buy', 'sell', 'buy', 'sell']
        })
        
        # Should not crash
        profile = analyzer.calculate_profile(extreme_data)
        assert profile is not None
        assert profile.total_volume > 0
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        analyzer = VPVRAnalyzer()
        
        # Invalid data types
        invalid_data = pd.DataFrame({
            'price': ['not_a_number', 30000, None, 30100],
            'volume': [1000, 'invalid', 1500, 1200],
            'side': ['buy', 'sell', 'unknown', None]
        })
        
        # Should handle or raise appropriate error
        try:
            profile = analyzer.calculate_profile(invalid_data)
            # If it succeeds, validate it handled the invalid data
            assert profile is not None
        except (ValueError, TypeError):
            # Expected behavior for invalid data
            pass
    
    def test_large_dataset_memory(self):
        """Test memory efficiency with large datasets"""
        analyzer = VPVRAnalyzer()
        
        # Large dataset (but not too large for CI)
        large_size = 10000
        large_data = pd.DataFrame({
            'price': np.random.normal(30000, 500, large_size),
            'volume': np.random.exponential(1000, large_size),
            'side': np.random.choice(['buy', 'sell'], large_size)
        })
        
        # Should handle without memory issues
        profile = analyzer.calculate_profile(large_data)
        assert profile is not None
        assert profile.total_volume > 0
        
        # Memory usage should be reasonable
        import sys
        profile_size = sys.getsizeof(profile)
        assert profile_size < 10 * 1024 * 1024, f"Profile too large: {profile_size} bytes"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--log-cli-level=INFO"])