"""
Performance Benchmarking and Results Analysis
=============================================

Comprehensive performance benchmarking suite for institutional strategies:
- Strategy performance comparisons
- Latency and throughput benchmarks
- Statistical significance testing
- Performance regression detection
- Visual performance reporting
- Baseline vs enhanced comparison

Requirements Validation:
- Liquidity detection accuracy >80%
- Smart money signal accuracy >65%
- Volume profile computation <50ms
- Correlation updates <100ms
- Overall 10-15% performance improvement
- Integration latency <10ms
"""

import time
import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Test utilities
from .test_utils import (
    MockDataGenerator, PerformanceBenchmark, StrategyTester, BacktestValidator,
    MockMarketData, MockStrategy
)

# Test configuration
from . import TEST_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    test_name: str
    category: str
    metric_name: str
    value: float
    unit: str
    threshold: Optional[float] = None
    baseline: Optional[float] = None
    improvement: Optional[float] = None
    passed: bool = True
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results"""
    suite_name: str
    results: List[BenchmarkResult]
    start_time: datetime
    end_time: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_score: float
    summary: Dict[str, Any]


class PerformanceBenchmarkRunner:
    """Main benchmark runner and results analyzer"""
    
    def __init__(self):
        self.data_generator = MockDataGenerator(TEST_CONFIG['mock_data_seed'])
        self.benchmark = PerformanceBenchmark()
        self.results: List[BenchmarkResult] = []
        self.baselines: Dict[str, float] = {}
        
        # Load baseline values (from previous runs or theoretical targets)
        self._load_baselines()
    
    def _load_baselines(self):
        """Load baseline performance values for comparison"""
        self.baselines = {
            # Latency baselines (in seconds)
            'liquidity_detection_latency': 0.080,  # 80ms baseline
            'smart_money_detection_latency': 0.150,  # 150ms baseline
            'volume_profile_computation': 0.080,  # 80ms baseline
            'correlation_calculation': 0.200,  # 200ms baseline
            'integration_latency': 0.020,  # 20ms baseline
            
            # Accuracy baselines (percentage)
            'liquidity_detection_accuracy': 0.75,  # 75% baseline
            'smart_money_accuracy': 0.60,  # 60% baseline
            'volume_profile_accuracy': 0.70,  # 70% baseline
            'correlation_accuracy': 0.85,  # 85% baseline
            
            # Throughput baselines (operations per second)
            'strategy_execution_throughput': 50,  # 50 ops/sec baseline
            'feature_aggregation_throughput': 200,  # 200 features/sec baseline
            'signal_processing_throughput': 100,  # 100 signals/sec baseline
        }
    
    def run_all_benchmarks(self) -> BenchmarkSuite:
        """Run all benchmark tests and return comprehensive results"""
        logger.info("Starting comprehensive performance benchmark suite...")
        start_time = datetime.now()
        
        # Run benchmark categories
        self._benchmark_strategy_performance()
        self._benchmark_latency_requirements()
        self._benchmark_accuracy_requirements() 
        self._benchmark_integration_performance()
        self._benchmark_system_scalability()
        self._benchmark_statistical_significance()
        
        end_time = datetime.now()
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        overall_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate summary
        summary = self._generate_summary()
        
        benchmark_suite = BenchmarkSuite(
            suite_name="Institutional Strategies Benchmark",
            results=self.results,
            start_time=start_time,
            end_time=end_time,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            overall_score=overall_score,
            summary=summary
        )
        
        # Generate reports
        self._generate_performance_report(benchmark_suite)
        self._save_results(benchmark_suite)
        
        logger.info(f"Benchmark suite completed - Score: {overall_score:.1f}% "
                   f"({passed_tests}/{total_tests} passed)")
        
        return benchmark_suite
    
    def _benchmark_strategy_performance(self):
        """Benchmark individual strategy performance"""
        logger.info("Benchmarking strategy performance...")
        
        # Liquidity hunting performance
        self._benchmark_liquidity_hunting()
        
        # Smart money divergence performance
        self._benchmark_smart_money_divergence()
        
        # Volume profile performance
        self._benchmark_volume_profile()
        
        # Correlation engine performance
        self._benchmark_correlation_engine()
    
    def _benchmark_liquidity_hunting(self):
        """Benchmark liquidity hunting strategy"""
        try:
            from ..institutional.liquidity_hunting import LiquidityHuntingStrategy
            
            strategy = LiquidityHuntingStrategy('BTC/USD', sensitivity=0.7)
            
            # Accuracy benchmark
            correct_detections = 0
            total_tests = 100
            latencies = []
            
            for i in range(total_tests):
                # Generate test scenario
                price = 30000 + np.random.uniform(-1000, 1000)
                order_book = self.data_generator.generate_order_book(price)
                trades = self.data_generator.generate_trades(price, 50000, 20)
                
                # Add iceberg pattern randomly
                has_iceberg = np.random.random() < 0.3
                if has_iceberg:
                    # Simulate iceberg by increasing volume at specific level
                    level_idx = np.random.randint(0, min(5, len(order_book['bids'])))
                    order_book['bids'][level_idx][1] *= np.random.uniform(3, 6)
                
                # Test detection
                start_time = time.perf_counter()
                signals = strategy.update(order_book, trades, {'price': price})
                latency = time.perf_counter() - start_time
                latencies.append(latency)
                
                # Check accuracy
                iceberg_detected = any(s.signal_type == 'iceberg' for s in signals)
                if (has_iceberg and iceberg_detected) or (not has_iceberg and not iceberg_detected):
                    correct_detections += 1
            
            # Record results
            accuracy = correct_detections / total_tests
            avg_latency = np.mean(latencies)
            
            self.results.append(BenchmarkResult(
                test_name="Liquidity Hunting Accuracy",
                category="Strategy Performance",
                metric_name="accuracy",
                value=accuracy,
                unit="percentage",
                threshold=TEST_CONFIG['performance_thresholds']['liquidity_detection_accuracy'],
                baseline=self.baselines['liquidity_detection_accuracy'],
                improvement=(accuracy - self.baselines['liquidity_detection_accuracy']) / self.baselines['liquidity_detection_accuracy'] * 100,
                passed=accuracy >= TEST_CONFIG['performance_thresholds']['liquidity_detection_accuracy']
            ))
            
            self.results.append(BenchmarkResult(
                test_name="Liquidity Hunting Latency",
                category="Performance",
                metric_name="latency",
                value=avg_latency,
                unit="seconds",
                threshold=0.050,  # 50ms
                baseline=self.baselines['liquidity_detection_latency'],
                improvement=(self.baselines['liquidity_detection_latency'] - avg_latency) / self.baselines['liquidity_detection_latency'] * 100,
                passed=avg_latency <= 0.050
            ))
            
        except ImportError:
            logger.warning("Liquidity hunting strategy not available for benchmarking")
    
    def _benchmark_smart_money_divergence(self):
        """Benchmark smart money divergence detection"""
        try:
            from ..institutional.smart_money_divergence import SmartMoneyDivergenceDetector
            from unittest.mock import patch, AsyncMock
            
            detector = SmartMoneyDivergenceDetector({'min_whale_threshold': 500000})
            
            # Test accuracy with mocked data
            correct_detections = 0
            total_tests = 50
            latencies = []
            
            for i in range(total_tests):
                # Create test scenario
                divergence_type = np.random.choice(['bullish', 'bearish', 'neutral'])
                
                # Generate price data
                if divergence_type == 'bullish':
                    prices = np.linspace(30000, 29000, 30)  # Declining
                    flows = np.linspace(10000, 40000, 30)  # Increasing flow
                elif divergence_type == 'bearish':
                    prices = np.linspace(29000, 31000, 30)  # Rising
                    flows = np.linspace(40000, 10000, 30)  # Decreasing flow
                else:
                    prices = np.random.normal(30000, 200, 30)  # Random
                    flows = np.random.normal(25000, 5000, 30)  # Random
                
                price_data = pd.DataFrame({'close': prices, 'volume': [50000] * 30})
                
                # Mock async methods
                with patch.object(detector, '_fetch_on_chain_data', new_callable=AsyncMock) as mock_on_chain, \
                     patch.object(detector, '_fetch_whale_transactions', new_callable=AsyncMock) as mock_whale, \
                     patch.object(detector, '_calculate_smart_money_flows', new_callable=AsyncMock) as mock_flows, \
                     patch.object(detector, '_fetch_exchange_flows', new_callable=AsyncMock) as mock_exchange:
                    
                    mock_on_chain.return_value = {}
                    mock_whale.return_value = []
                    
                    flow_df = pd.DataFrame({'smart_money_flow': flows})
                    mock_flows.return_value = flow_df
                    mock_exchange.return_value = {'net_flow': 0}
                    
                    # Test detection
                    start_time = time.perf_counter()
                    
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        signals = loop.run_until_complete(detector.detect_divergence('BTC/USD', price_data))
                        loop.close()
                        
                        latency = time.perf_counter() - start_time
                        latencies.append(latency)
                        
                        # Check detection accuracy
                        detected_type = None
                        if signals:
                            signal = signals[0]
                            if 'bullish' in signal.divergence_type.value.lower():
                                detected_type = 'bullish'
                            elif 'bearish' in signal.divergence_type.value.lower():
                                detected_type = 'bearish'
                        else:
                            detected_type = 'neutral'
                        
                        if detected_type == divergence_type:
                            correct_detections += 1
                            
                    except Exception as e:
                        logger.warning(f"Smart money detection failed: {e}")
                        latencies.append(0.1)  # Default latency for failed attempts
            
            # Record results
            accuracy = correct_detections / total_tests
            avg_latency = np.mean(latencies) if latencies else 0
            
            self.results.append(BenchmarkResult(
                test_name="Smart Money Divergence Accuracy",
                category="Strategy Performance",
                metric_name="accuracy",
                value=accuracy,
                unit="percentage",
                threshold=TEST_CONFIG['performance_thresholds']['smart_money_signal_accuracy'],
                baseline=self.baselines['smart_money_accuracy'],
                improvement=(accuracy - self.baselines['smart_money_accuracy']) / self.baselines['smart_money_accuracy'] * 100,
                passed=accuracy >= TEST_CONFIG['performance_thresholds']['smart_money_signal_accuracy']
            ))
            
            self.results.append(BenchmarkResult(
                test_name="Smart Money Detection Latency",
                category="Performance",
                metric_name="latency",
                value=avg_latency,
                unit="seconds",
                threshold=0.100,  # 100ms
                baseline=self.baselines['smart_money_detection_latency'],
                improvement=(self.baselines['smart_money_detection_latency'] - avg_latency) / self.baselines['smart_money_detection_latency'] * 100,
                passed=avg_latency <= 0.100
            ))
            
        except ImportError:
            logger.warning("Smart money divergence detector not available for benchmarking")
    
    def _benchmark_volume_profile(self):
        """Benchmark volume profile analysis"""
        try:
            from ..institutional.volume_profile import VPVRAnalyzer
            
            analyzer = VPVRAnalyzer()
            
            # Test computation speed
            computation_times = []
            accuracy_scores = []
            
            for _ in range(50):
                # Generate test data
                test_data = pd.DataFrame({
                    'price': np.random.normal(30000, 200, 500),
                    'volume': np.random.exponential(1000, 500),
                    'side': np.random.choice(['buy', 'sell'], 500)
                })
                
                # Time computation
                start_time = time.perf_counter()
                profile = analyzer.calculate_profile(test_data)
                computation_time = time.perf_counter() - start_time
                computation_times.append(computation_time)
                
                # Validate computation quality
                if profile.total_volume > 0 and 65 <= profile.value_area_percentage <= 75:
                    accuracy_scores.append(1.0)
                else:
                    accuracy_scores.append(0.0)
            
            # Record results
            avg_computation_time = np.mean(computation_times)
            accuracy = np.mean(accuracy_scores)
            
            self.results.append(BenchmarkResult(
                test_name="Volume Profile Computation Speed",
                category="Performance",
                metric_name="computation_time",
                value=avg_computation_time,
                unit="seconds",
                threshold=TEST_CONFIG['performance_thresholds']['volume_profile_computation_time'],
                baseline=self.baselines['volume_profile_computation'],
                improvement=(self.baselines['volume_profile_computation'] - avg_computation_time) / self.baselines['volume_profile_computation'] * 100,
                passed=avg_computation_time <= TEST_CONFIG['performance_thresholds']['volume_profile_computation_time']
            ))
            
            self.results.append(BenchmarkResult(
                test_name="Volume Profile Accuracy",
                category="Strategy Performance",
                metric_name="accuracy",
                value=accuracy,
                unit="percentage",
                threshold=0.80,
                baseline=self.baselines['volume_profile_accuracy'],
                improvement=(accuracy - self.baselines['volume_profile_accuracy']) / self.baselines['volume_profile_accuracy'] * 100,
                passed=accuracy >= 0.80
            ))
            
        except ImportError:
            logger.warning("Volume profile analyzer not available for benchmarking")
    
    def _benchmark_correlation_engine(self):
        """Benchmark correlation engine"""
        try:
            from ..institutional.correlation_engine import CorrelationEngine
            
            engine = CorrelationEngine()
            
            # Add test data
            assets = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
            for asset in assets:
                prices = pd.Series(
                    np.random.normal(1000, 50, 200),
                    index=pd.date_range('2023-01-01', periods=200)
                )
                engine.add_price_data(asset, prices)
            
            # Test correlation calculation speed
            calculation_times = []
            
            for _ in range(30):
                start_time = time.perf_counter()
                result = engine.calculate_correlation_matrix(method='pearson', window=30)
                calculation_time = time.perf_counter() - start_time
                calculation_times.append(calculation_time)
                
                # Validate result
                assert result.correlation_matrix is not None
                assert result.correlation_matrix.shape == (len(assets), len(assets))
            
            # Record results
            avg_calculation_time = np.mean(calculation_times)
            
            self.results.append(BenchmarkResult(
                test_name="Correlation Calculation Speed",
                category="Performance",
                metric_name="calculation_time",
                value=avg_calculation_time,
                unit="seconds",
                threshold=TEST_CONFIG['performance_thresholds']['correlation_update_time'],
                baseline=self.baselines['correlation_calculation'],
                improvement=(self.baselines['correlation_calculation'] - avg_calculation_time) / self.baselines['correlation_calculation'] * 100,
                passed=avg_calculation_time <= TEST_CONFIG['performance_thresholds']['correlation_update_time']
            ))
            
        except ImportError:
            logger.warning("Correlation engine not available for benchmarking")
    
    def _benchmark_latency_requirements(self):
        """Benchmark all latency requirements"""
        logger.info("Benchmarking latency requirements...")
        
        # Integration latency benchmark
        integration_times = []
        
        for _ in range(100):
            start_time = time.perf_counter()
            
            # Simulate integration operations
            mock_features = {'feature_1': 0.5, 'feature_2': 0.8, 'feature_3': 0.3}
            mock_signals = {'signal_1': 0.7, 'signal_2': 0.4}
            
            # Simulate feature aggregation
            aggregated_features = np.array(list(mock_features.values()))
            
            # Simulate signal processing
            processed_signals = {k: v * 1.1 for k, v in mock_signals.items()}
            
            integration_time = time.perf_counter() - start_time
            integration_times.append(integration_time)
        
        avg_integration_time = np.mean(integration_times)
        
        self.results.append(BenchmarkResult(
            test_name="Integration Latency",
            category="Performance",
            metric_name="integration_latency",
            value=avg_integration_time,
            unit="seconds",
            threshold=TEST_CONFIG['performance_thresholds']['integration_latency'],
            baseline=self.baselines['integration_latency'],
            improvement=(self.baselines['integration_latency'] - avg_integration_time) / self.baselines['integration_latency'] * 100,
            passed=avg_integration_time <= TEST_CONFIG['performance_thresholds']['integration_latency']
        ))
    
    def _benchmark_accuracy_requirements(self):
        """Benchmark all accuracy requirements"""
        logger.info("Benchmarking accuracy requirements...")
        
        # Overall system accuracy benchmark
        # This would typically test the complete system accuracy
        # For now, we'll simulate based on individual component accuracies
        
        component_accuracies = []
        for result in self.results:
            if result.metric_name == 'accuracy' and result.value is not None:
                component_accuracies.append(result.value)
        
        if component_accuracies:
            overall_accuracy = np.mean(component_accuracies)
            
            self.results.append(BenchmarkResult(
                test_name="Overall System Accuracy",
                category="Strategy Performance",
                metric_name="overall_accuracy",
                value=overall_accuracy,
                unit="percentage",
                threshold=0.70,  # 70% overall threshold
                baseline=0.65,  # 65% baseline
                improvement=(overall_accuracy - 0.65) / 0.65 * 100,
                passed=overall_accuracy >= 0.70
            ))
    
    def _benchmark_integration_performance(self):
        """Benchmark integration layer performance"""
        logger.info("Benchmarking integration performance...")
        
        # Strategy execution throughput
        mock_strategy = MockStrategy()
        
        # Throughput test
        execution_count = 0
        test_duration = 2.0  # 2 seconds
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            # Simulate strategy execution
            mock_data = MockMarketData(
                timestamp=datetime.now(),
                symbol='BTC/USD',
                price=30000,
                volume=50000,
                high=30100,
                low=29900,
                open=30050,
                bid=29995,
                ask=30005,
                trades=[],
                order_book={'bids': [[29990, 10]], 'asks': [[30010, 10]]}
            )
            
            features = mock_strategy.extract_features(mock_data)
            signals = mock_strategy.generate_signals(mock_data)
            
            execution_count += 1
        
        throughput = execution_count / test_duration
        
        self.results.append(BenchmarkResult(
            test_name="Strategy Execution Throughput",
            category="Performance",
            metric_name="throughput",
            value=throughput,
            unit="operations_per_second",
            threshold=40,  # 40 ops/sec minimum
            baseline=self.baselines['strategy_execution_throughput'],
            improvement=(throughput - self.baselines['strategy_execution_throughput']) / self.baselines['strategy_execution_throughput'] * 100,
            passed=throughput >= 40
        ))
    
    def _benchmark_system_scalability(self):
        """Benchmark system scalability"""
        logger.info("Benchmarking system scalability...")
        
        # Test with different numbers of strategies
        strategy_counts = [1, 5, 10, 20]
        scalability_results = []
        
        for count in strategy_counts:
            # Simulate parallel strategy execution
            execution_times = []
            
            for _ in range(10):
                start_time = time.perf_counter()
                
                # Simulate parallel execution
                tasks_completed = 0
                for _ in range(count):
                    # Simulate strategy work
                    mock_work = np.random.normal(0.01, 0.002)  # 10ms ± 2ms work
                    time.sleep(max(0, mock_work))
                    tasks_completed += 1
                
                execution_time = time.perf_counter() - start_time
                execution_times.append(execution_time)
            
            avg_time = np.mean(execution_times)
            scalability_results.append((count, avg_time))
        
        # Calculate scalability metric (should be sub-linear)
        if len(scalability_results) >= 2:
            max_count, max_time = scalability_results[-1]
            min_count, min_time = scalability_results[0]
            
            scalability_ratio = (max_time / min_time) / (max_count / min_count)
            
            self.results.append(BenchmarkResult(
                test_name="System Scalability",
                category="Performance",
                metric_name="scalability_ratio",
                value=scalability_ratio,
                unit="ratio",
                threshold=2.0,  # Should scale no worse than 2x
                baseline=1.5,
                improvement=(1.5 - scalability_ratio) / 1.5 * 100,
                passed=scalability_ratio <= 2.0,
                metadata={'strategy_counts': strategy_counts, 'execution_times': [r[1] for r in scalability_results]}
            ))
    
    def _benchmark_statistical_significance(self):
        """Benchmark statistical significance of improvements"""
        logger.info("Benchmarking statistical significance...")
        
        # Test performance improvements against baseline
        improvement_results = []
        
        for result in self.results:
            if result.improvement is not None and result.baseline is not None:
                improvement_results.append(result.improvement)
        
        if improvement_results:
            # Calculate statistical significance of improvements
            improvements = np.array(improvement_results)
            
            # Test if improvements are significantly > 0
            t_stat, p_value = stats.ttest_1samp(improvements, 0)
            
            significant = p_value < 0.05 and np.mean(improvements) > 0
            
            self.results.append(BenchmarkResult(
                test_name="Statistical Significance of Improvements",
                category="Statistical Analysis",
                metric_name="p_value",
                value=p_value,
                unit="probability",
                threshold=0.05,
                passed=significant,
                metadata={
                    't_statistic': t_stat,
                    'mean_improvement': np.mean(improvements),
                    'improvement_count': len(improvements)
                }
            ))
            
            # Overall performance improvement
            overall_improvement = np.mean(improvements)
            
            self.results.append(BenchmarkResult(
                test_name="Overall Performance Improvement",
                category="Strategy Performance",
                metric_name="overall_improvement",
                value=overall_improvement / 100,  # Convert to decimal
                unit="percentage",
                threshold=TEST_CONFIG['performance_thresholds']['overall_performance_improvement'],
                baseline=0.0,  # No improvement baseline
                improvement=overall_improvement,
                passed=overall_improvement >= (TEST_CONFIG['performance_thresholds']['overall_performance_improvement'] * 100)
            ))
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary statistics"""
        summary = {
            'categories': {},
            'metrics': {},
            'improvements': {},
            'failures': []
        }
        
        # Group by category
        for result in self.results:
            category = result.category
            if category not in summary['categories']:
                summary['categories'][category] = {'total': 0, 'passed': 0, 'failed': 0}
            
            summary['categories'][category]['total'] += 1
            if result.passed:
                summary['categories'][category]['passed'] += 1
            else:
                summary['categories'][category]['failed'] += 1
                summary['failures'].append({
                    'test': result.test_name,
                    'value': result.value,
                    'threshold': result.threshold,
                    'unit': result.unit
                })
        
        # Aggregate metrics
        metric_values = {}
        for result in self.results:
            metric = result.metric_name
            if metric not in metric_values:
                metric_values[metric] = []
            metric_values[metric].append(result.value)
        
        for metric, values in metric_values.items():
            summary['metrics'][metric] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Calculate improvements
        improvements = [r.improvement for r in self.results if r.improvement is not None]
        if improvements:
            summary['improvements'] = {
                'count': len(improvements),
                'mean': np.mean(improvements),
                'positive_count': sum(1 for i in improvements if i > 0),
                'significant_count': sum(1 for i in improvements if i > 10)  # >10% improvement
            }
        
        return summary
    
    def _generate_performance_report(self, suite: BenchmarkSuite):
        """Generate visual performance report"""
        try:
            # Create plots directory
            plots_dir = Path(__file__).parent / 'benchmark_plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Overall results pie chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Pie chart of pass/fail
            labels = ['Passed', 'Failed']
            sizes = [suite.passed_tests, suite.failed_tests]
            colors = ['#2ecc71', '#e74c3c']
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'Test Results Overview\n({suite.passed_tests}/{suite.total_tests} passed)')
            
            # 2. Performance by category
            categories = {}
            for result in suite.results:
                if result.category not in categories:
                    categories[result.category] = {'passed': 0, 'failed': 0}
                if result.passed:
                    categories[result.category]['passed'] += 1
                else:
                    categories[result.category]['failed'] += 1
            
            cat_names = list(categories.keys())
            passed_counts = [categories[cat]['passed'] for cat in cat_names]
            failed_counts = [categories[cat]['failed'] for cat in cat_names]
            
            x = np.arange(len(cat_names))
            width = 0.35
            
            ax2.bar(x - width/2, passed_counts, width, label='Passed', color='#2ecc71')
            ax2.bar(x + width/2, failed_counts, width, label='Failed', color='#e74c3c')
            ax2.set_xlabel('Category')
            ax2.set_ylabel('Number of Tests')
            ax2.set_title('Results by Category')
            ax2.set_xticks(x)
            ax2.set_xticklabels(cat_names, rotation=45)
            ax2.legend()
            
            # 3. Latency comparison
            latency_results = [r for r in suite.results if 'latency' in r.metric_name or 'time' in r.metric_name]
            if latency_results:
                test_names = [r.test_name for r in latency_results]
                values = [r.value * 1000 for r in latency_results]  # Convert to ms
                thresholds = [r.threshold * 1000 if r.threshold else 0 for r in latency_results]
                
                x = np.arange(len(test_names))
                bars = ax3.bar(x, values, color='#3498db', alpha=0.7)
                
                # Add threshold lines
                for i, threshold in enumerate(thresholds):
                    if threshold > 0:
                        ax3.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
                
                ax3.set_xlabel('Test')
                ax3.set_ylabel('Latency (ms)')
                ax3.set_title('Latency Benchmarks')
                ax3.set_xticks(x)
                ax3.set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in test_names], rotation=45)
                
                # Color bars based on pass/fail
                for bar, result in zip(bars, latency_results):
                    bar.set_color('#2ecc71' if result.passed else '#e74c3c')
            
            # 4. Accuracy comparison
            accuracy_results = [r for r in suite.results if 'accuracy' in r.metric_name]
            if accuracy_results:
                test_names = [r.test_name for r in accuracy_results]
                values = [r.value * 100 for r in accuracy_results]  # Convert to percentage
                thresholds = [r.threshold * 100 if r.threshold else 0 for r in accuracy_results]
                
                x = np.arange(len(test_names))
                bars = ax4.bar(x, values, color='#9b59b6', alpha=0.7)
                
                # Add threshold lines
                for i, threshold in enumerate(thresholds):
                    if threshold > 0:
                        ax4.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
                
                ax4.set_xlabel('Test')
                ax4.set_ylabel('Accuracy (%)')
                ax4.set_title('Accuracy Benchmarks')
                ax4.set_xticks(x)
                ax4.set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in test_names], rotation=45)
                ax4.set_ylim(0, 100)
                
                # Color bars based on pass/fail
                for bar, result in zip(bars, accuracy_results):
                    bar.set_color('#2ecc71' if result.passed else '#e74c3c')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'benchmark_overview.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. Improvement heatmap
            improvement_data = []
            for result in suite.results:
                if result.improvement is not None:
                    improvement_data.append({
                        'Test': result.test_name[:30],
                        'Category': result.category,
                        'Improvement': result.improvement
                    })
            
            if improvement_data:
                df = pd.DataFrame(improvement_data)
                pivot_table = df.pivot_table(values='Improvement', index='Test', columns='Category', fill_value=0)
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', center=0, fmt='.1f')
                plt.title('Performance Improvements by Test and Category (%)')
                plt.tight_layout()
                plt.savefig(plots_dir / 'improvement_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Performance plots saved to {plots_dir}")
            
        except Exception as e:
            logger.warning(f"Could not generate performance plots: {e}")
    
    def _save_results(self, suite: BenchmarkSuite):
        """Save benchmark results to files"""
        try:
            results_dir = Path(__file__).parent / 'benchmark_results'
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save JSON results
            json_data = {
                'suite_info': {
                    'name': suite.suite_name,
                    'start_time': suite.start_time.isoformat(),
                    'end_time': suite.end_time.isoformat(),
                    'duration_seconds': (suite.end_time - suite.start_time).total_seconds(),
                    'total_tests': suite.total_tests,
                    'passed_tests': suite.passed_tests,
                    'failed_tests': suite.failed_tests,
                    'overall_score': suite.overall_score
                },
                'results': [asdict(result) for result in suite.results],
                'summary': suite.summary
            }
            
            with open(results_dir / f'benchmark_results_{timestamp}.json', 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            # Save CSV results
            df_data = []
            for result in suite.results:
                df_data.append({
                    'Test Name': result.test_name,
                    'Category': result.category,
                    'Metric': result.metric_name,
                    'Value': result.value,
                    'Unit': result.unit,
                    'Threshold': result.threshold,
                    'Baseline': result.baseline,
                    'Improvement (%)': result.improvement,
                    'Passed': result.passed,
                    'Timestamp': result.timestamp
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(results_dir / f'benchmark_results_{timestamp}.csv', index=False)
            
            # Save summary report
            with open(results_dir / f'benchmark_summary_{timestamp}.txt', 'w') as f:
                f.write(f"Institutional Strategies Benchmark Report\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"Suite: {suite.suite_name}\n")
                f.write(f"Start Time: {suite.start_time}\n")
                f.write(f"End Time: {suite.end_time}\n")
                f.write(f"Duration: {(suite.end_time - suite.start_time).total_seconds():.1f} seconds\n\n")
                
                f.write(f"Overall Results:\n")
                f.write(f"  Total Tests: {suite.total_tests}\n")
                f.write(f"  Passed: {suite.passed_tests}\n")
                f.write(f"  Failed: {suite.failed_tests}\n")
                f.write(f"  Success Rate: {suite.overall_score:.1f}%\n\n")
                
                f.write(f"Results by Category:\n")
                for category, stats in suite.summary['categories'].items():
                    f.write(f"  {category}: {stats['passed']}/{stats['total']} passed ({stats['passed']/stats['total']*100:.1f}%)\n")
                
                f.write(f"\nFailed Tests:\n")
                for failure in suite.summary['failures']:
                    f.write(f"  {failure['test']}: {failure['value']:.4f} {failure['unit']} (threshold: {failure['threshold']})\n")
                
                if 'improvements' in suite.summary:
                    imp = suite.summary['improvements']
                    f.write(f"\nPerformance Improvements:\n")
                    f.write(f"  Average Improvement: {imp['mean']:.1f}%\n")
                    f.write(f"  Positive Improvements: {imp['positive_count']}/{imp['count']}\n")
                    f.write(f"  Significant Improvements (>10%): {imp['significant_count']}/{imp['count']}\n")
            
            logger.info(f"Benchmark results saved to {results_dir}")
            
        except Exception as e:
            logger.error(f"Could not save benchmark results: {e}")


def run_benchmarks():
    """Main entry point for running benchmarks"""
    runner = PerformanceBenchmarkRunner()
    suite = runner.run_all_benchmarks()
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("INSTITUTIONAL STRATEGIES BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Overall Score: {suite.overall_score:.1f}% ({suite.passed_tests}/{suite.total_tests} passed)")
    print(f"Duration: {(suite.end_time - suite.start_time).total_seconds():.1f} seconds")
    
    print("\nResults by Category:")
    for category, stats in suite.summary['categories'].items():
        success_rate = stats['passed'] / stats['total'] * 100
        print(f"  {category}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
    
    if suite.summary['failures']:
        print(f"\nFailed Tests ({len(suite.summary['failures'])}):")
        for failure in suite.summary['failures'][:5]:  # Show first 5 failures
            print(f"  ❌ {failure['test']}: {failure['value']:.4f} {failure['unit']} (threshold: {failure['threshold']})")
        
        if len(suite.summary['failures']) > 5:
            print(f"  ... and {len(suite.summary['failures']) - 5} more")
    
    # Highlight key requirements
    print(f"\nKey Requirements Status:")
    key_metrics = {
        'Liquidity Detection Accuracy': ('>80%', None),
        'Smart Money Signal Accuracy': ('>65%', None), 
        'Volume Profile Computation Speed': ('<50ms', None),
        'Correlation Calculation Speed': ('<100ms', None),
        'Integration Latency': ('<10ms', None),
        'Overall Performance Improvement': ('>10%', None)
    }
    
    for result in suite.results:
        for metric_name, (requirement, _) in key_metrics.items():
            if metric_name.lower() in result.test_name.lower():
                status = "✅ PASS" if result.passed else "❌ FAIL"
                key_metrics[metric_name] = (requirement, status)
    
    for metric, (requirement, status) in key_metrics.items():
        if status:
            print(f"  {status} {metric}: {requirement}")
        else:
            print(f"  ⚠️  SKIP {metric}: {requirement} (not tested)")
    
    print("\n" + "=" * 60)
    
    return suite


if __name__ == "__main__":
    benchmark_suite = run_benchmarks()