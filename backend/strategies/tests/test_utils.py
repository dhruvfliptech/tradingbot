"""
Test Utilities and Fixtures
===========================

Common utilities, fixtures, and mock data generators for strategy testing.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
import random
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass
import json

# Statistical testing
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass
class MockMarketData:
    """Mock market data for testing"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    high: float
    low: float
    open: float
    bid: float
    ask: float
    trades: List[Dict]
    order_book: Dict[str, List[List[float]]]


@dataclass
class TestMetrics:
    """Test performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency: float
    throughput: float
    error_rate: float
    success_rate: float
    

class MockDataGenerator:
    """Generate realistic mock data for testing"""
    
    def __init__(self, seed: int = 42):
        """Initialize with seed for reproducible tests"""
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
    
    def generate_price_series(self, 
                            length: int = 1000,
                            initial_price: float = 30000.0,
                            volatility: float = 0.02,
                            trend: float = 0.0001) -> pd.Series:
        """Generate realistic price series with GBM"""
        dt = 1/24/60  # 1 minute intervals
        
        # Generate returns using Geometric Brownian Motion
        returns = np.random.normal(trend, volatility, length)
        
        # Add some regime changes and volatility clustering
        for i in range(1, length):
            if np.random.random() < 0.01:  # 1% chance of regime change
                returns[i] *= np.random.uniform(2, 5)  # Volatility spike
        
        # Calculate prices
        log_prices = np.log(initial_price) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        # Create time index
        start_time = datetime.now() - timedelta(minutes=length)
        timestamps = pd.date_range(start=start_time, periods=length, freq='1min')
        
        return pd.Series(prices, index=timestamps)
    
    def generate_volume_series(self, 
                             price_series: pd.Series,
                             base_volume: float = 1000000) -> pd.Series:
        """Generate volume series correlated with price movements"""
        returns = price_series.pct_change().fillna(0)
        
        # Volume increases with price volatility
        volatility_factor = np.abs(returns) * 10
        volume_multiplier = 1 + volatility_factor + np.random.exponential(0.2, len(price_series))
        
        volumes = base_volume * volume_multiplier
        return pd.Series(volumes, index=price_series.index)
    
    def generate_order_book(self, 
                          price: float,
                          spread_bps: float = 5.0,
                          depth_levels: int = 20) -> Dict[str, List[List[float]]]:
        """Generate realistic order book"""
        spread = price * spread_bps / 10000
        
        bids = []
        asks = []
        
        for i in range(depth_levels):
            # Exponentially decreasing volume with distance from mid
            volume_decay = np.exp(-i * 0.1)
            base_volume = np.random.exponential(10) * volume_decay
            
            # Bid side
            bid_price = price - spread/2 - i * spread/depth_levels
            bid_volume = base_volume * (1 + np.random.uniform(-0.3, 0.3))
            bids.append([bid_price, bid_volume])
            
            # Ask side  
            ask_price = price + spread/2 + i * spread/depth_levels
            ask_volume = base_volume * (1 + np.random.uniform(-0.3, 0.3))
            asks.append([ask_price, ask_volume])
        
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': time.time()
        }
    
    def generate_trades(self, 
                       price: float,
                       volume: float,
                       count: int = 100) -> List[Dict]:
        """Generate realistic trade data"""
        trades = []
        spread = price * 0.0005  # 5 bps spread
        
        for i in range(count):
            # Random trade direction
            is_buy = np.random.random() > 0.5
            trade_price = price + (spread/2 if is_buy else -spread/2)
            trade_price += np.random.normal(0, spread/4)
            
            trade_volume = np.random.exponential(volume / count)
            
            trades.append({
                'price': trade_price,
                'volume': trade_volume,
                'side': 'buy' if is_buy else 'sell',
                'timestamp': time.time() - np.random.uniform(0, 60)
            })
        
        return sorted(trades, key=lambda x: x['timestamp'])
    
    def generate_whale_transactions(self, 
                                  count: int = 10,
                                  min_amount: float = 1000000) -> List[Dict]:
        """Generate whale transaction data"""
        transactions = []
        
        for _ in range(count):
            # Transaction types
            tx_types = ['exchange_inflow', 'exchange_outflow', 'wallet_transfer', 'accumulation']
            tx_type = np.random.choice(tx_types)
            
            amount_usd = np.random.uniform(min_amount, min_amount * 10)
            
            transactions.append({
                'hash': f'0x{np.random.randint(0, 16**64):016x}',
                'amount_usd': amount_usd,
                'from_type': 'exchange' if tx_type == 'exchange_outflow' else 'wallet',
                'to_type': 'exchange' if tx_type == 'exchange_inflow' else 'wallet',
                'timestamp': datetime.now() - timedelta(hours=np.random.uniform(0, 24))
            })
        
        return transactions
    
    def generate_market_data_feed(self, 
                                duration_minutes: int = 60,
                                update_interval_seconds: int = 1) -> List[MockMarketData]:
        """Generate continuous market data feed"""
        updates = []
        current_time = datetime.now()
        
        # Generate base price series
        num_updates = duration_minutes * 60 // update_interval_seconds
        base_prices = self.generate_price_series(num_updates, 30000.0, 0.001)
        base_volumes = self.generate_volume_series(base_prices, 50000)
        
        for i, (timestamp, price) in enumerate(base_prices.items()):
            volume = base_volumes.iloc[i]
            
            # Generate OHLC for this interval
            high = price * (1 + np.random.uniform(0, 0.002))
            low = price * (1 - np.random.uniform(0, 0.002))
            open_price = base_prices.iloc[i-1] if i > 0 else price
            
            # Bid/Ask
            spread = price * 0.0005
            bid = price - spread/2
            ask = price + spread/2
            
            # Generate order book and trades
            order_book = self.generate_order_book(price)
            trades = self.generate_trades(price, volume, 20)
            
            market_data = MockMarketData(
                timestamp=timestamp,
                symbol='BTC/USD',
                price=price,
                volume=volume,
                high=high,
                low=low,
                open=open_price,
                bid=bid,
                ask=ask,
                trades=trades,
                order_book=order_book
            )
            
            updates.append(market_data)
        
        return updates


class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    def __init__(self):
        self.results = {}
        self.baselines = {}
    
    def benchmark_function(self, func, *args, iterations: int = 100, **kwargs):
        """Benchmark function execution time"""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            if asyncio.iscoroutinefunction(func):
                asyncio.run(func(*args, **kwargs))
            else:
                func(*args, **kwargs)
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }
    
    def compare_performance(self, 
                          baseline_times: List[float],
                          current_times: List[float],
                          improvement_threshold: float = 0.1) -> Dict[str, Any]:
        """Compare performance improvements"""
        baseline_mean = np.mean(baseline_times)
        current_mean = np.mean(current_times)
        
        improvement = (baseline_mean - current_mean) / baseline_mean
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(baseline_times, current_times)
        
        return {
            'baseline_mean': baseline_mean,
            'current_mean': current_mean,
            'improvement_pct': improvement * 100,
            'meets_threshold': improvement >= improvement_threshold,
            'statistically_significant': p_value < 0.05,
            'p_value': p_value,
            't_statistic': t_stat
        }


class StrategyTester:
    """Base class for strategy testing"""
    
    def __init__(self, strategy_class, config: Optional[Dict] = None):
        self.strategy_class = strategy_class
        self.config = config or {}
        self.data_generator = MockDataGenerator()
        self.benchmark = PerformanceBenchmark()
        
    def create_strategy_instance(self, **kwargs):
        """Create strategy instance with test configuration"""
        test_config = {**self.config, **kwargs}
        return self.strategy_class(**test_config)
    
    def test_initialization(self):
        """Test strategy initialization"""
        strategy = self.create_strategy_instance()
        assert strategy is not None
        return strategy
    
    def test_feature_extraction(self, strategy, market_data: MockMarketData):
        """Test feature extraction"""
        if not hasattr(strategy, 'extract_features'):
            pytest.skip("Strategy doesn't implement extract_features")
        
        start_time = time.perf_counter()
        features = strategy.extract_features(market_data)
        latency = time.perf_counter() - start_time
        
        assert isinstance(features, dict)
        assert len(features) > 0
        assert all(isinstance(v, (int, float, np.number)) for v in features.values())
        
        return features, latency
    
    def test_signal_generation(self, strategy, market_data: MockMarketData):
        """Test signal generation"""
        if not hasattr(strategy, 'generate_signals'):
            pytest.skip("Strategy doesn't implement generate_signals")
        
        start_time = time.perf_counter()
        signals = strategy.generate_signals(market_data)
        latency = time.perf_counter() - start_time
        
        assert signals is not None
        
        return signals, latency
    
    def calculate_signal_accuracy(self, 
                                predictions: List[Any],
                                actual_outcomes: List[Any]) -> float:
        """Calculate prediction accuracy"""
        if len(predictions) != len(actual_outcomes):
            return 0.0
        
        correct = sum(1 for p, a in zip(predictions, actual_outcomes) if p == a)
        return correct / len(predictions)
    
    def run_stress_test(self, 
                       strategy, 
                       duration_seconds: int = 60,
                       update_frequency: float = 10.0):
        """Run stress test with high-frequency updates"""
        errors = []
        latencies = []
        successful_updates = 0
        
        start_time = time.time()
        next_update = start_time
        
        while time.time() - start_time < duration_seconds:
            if time.time() >= next_update:
                try:
                    # Generate market data
                    price_series = self.data_generator.generate_price_series(10, 30000.0)
                    market_data = MockMarketData(
                        timestamp=datetime.now(),
                        symbol='BTC/USD',
                        price=price_series.iloc[-1],
                        volume=50000,
                        high=price_series.max(),
                        low=price_series.min(),
                        open=price_series.iloc[0],
                        bid=price_series.iloc[-1] * 0.9995,
                        ask=price_series.iloc[-1] * 1.0005,
                        trades=[],
                        order_book=self.data_generator.generate_order_book(price_series.iloc[-1])
                    )
                    
                    # Test strategy update
                    update_start = time.perf_counter()
                    if hasattr(strategy, 'update'):
                        strategy.update(market_data)
                    elif hasattr(strategy, 'extract_features'):
                        strategy.extract_features(market_data)
                    
                    latency = time.perf_counter() - update_start
                    latencies.append(latency)
                    successful_updates += 1
                    
                except Exception as e:
                    errors.append(str(e))
                
                next_update += 1.0 / update_frequency
            
            time.sleep(0.001)  # Small sleep to prevent busy waiting
        
        return {
            'duration': duration_seconds,
            'successful_updates': successful_updates,
            'error_count': len(errors),
            'error_rate': len(errors) / (successful_updates + len(errors)) if (successful_updates + len(errors)) > 0 else 0,
            'avg_latency': np.mean(latencies) if latencies else 0,
            'max_latency': np.max(latencies) if latencies else 0,
            'latency_p95': np.percentile(latencies, 95) if latencies else 0,
            'errors': errors[:10]  # First 10 errors
        }


class BacktestValidator:
    """Backtesting validation utilities"""
    
    def __init__(self):
        self.results = {}
    
    def run_backtest(self, 
                    strategy,
                    price_data: pd.Series,
                    initial_capital: float = 100000) -> Dict[str, Any]:
        """Run simple backtest validation"""
        capital = initial_capital
        positions = []
        trades = []
        
        for i in range(1, len(price_data)):
            current_price = price_data.iloc[i]
            previous_price = price_data.iloc[i-1]
            
            # Generate mock market data
            mock_data = MockMarketData(
                timestamp=price_data.index[i],
                symbol='BTC/USD',
                price=current_price,
                volume=50000,
                high=max(current_price, previous_price) * 1.002,
                low=min(current_price, previous_price) * 0.998,
                open=previous_price,
                bid=current_price * 0.9995,
                ask=current_price * 1.0005,
                trades=[],
                order_book={'bids': [[current_price * 0.999, 100]], 'asks': [[current_price * 1.001, 100]]}
            )
            
            # Get strategy signal
            if hasattr(strategy, 'generate_signals'):
                signals = strategy.generate_signals(mock_data)
                
                # Simple signal interpretation
                if signals and isinstance(signals, dict):
                    signal_value = list(signals.values())[0] if signals else 0
                    
                    # Buy signal
                    if signal_value > 0.6 and len(positions) == 0:
                        position_size = capital * 0.1 / current_price  # 10% of capital
                        positions.append({
                            'type': 'long',
                            'entry_price': current_price,
                            'size': position_size,
                            'timestamp': price_data.index[i]
                        })
                        capital -= position_size * current_price
                        trades.append({'type': 'buy', 'price': current_price, 'size': position_size})
                    
                    # Sell signal
                    elif signal_value < 0.4 and len(positions) > 0:
                        position = positions.pop()
                        pnl = (current_price - position['entry_price']) * position['size']
                        capital += position['size'] * current_price
                        trades.append({'type': 'sell', 'price': current_price, 'size': position['size'], 'pnl': pnl})
        
        # Calculate final portfolio value
        final_value = capital
        for position in positions:
            final_value += position['size'] * price_data.iloc[-1]
        
        # Calculate metrics
        total_return = (final_value - initial_capital) / initial_capital
        
        trade_pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        win_rate = sum(1 for pnl in trade_pnls if pnl > 0) / len(trade_pnls) if trade_pnls else 0
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_trade_pnl': np.mean(trade_pnls) if trade_pnls else 0,
            'max_trade_pnl': np.max(trade_pnls) if trade_pnls else 0,
            'min_trade_pnl': np.min(trade_pnls) if trade_pnls else 0
        }


# Pytest fixtures
@pytest.fixture
def mock_data_generator():
    """Fixture for mock data generator"""
    return MockDataGenerator()


@pytest.fixture
def sample_price_data(mock_data_generator):
    """Fixture for sample price data"""
    return mock_data_generator.generate_price_series(1000)


@pytest.fixture
def sample_market_data(mock_data_generator):
    """Fixture for sample market data"""
    price_series = mock_data_generator.generate_price_series(100)
    current_price = price_series.iloc[-1]
    
    return MockMarketData(
        timestamp=datetime.now(),
        symbol='BTC/USD',
        price=current_price,
        volume=50000,
        high=current_price * 1.005,
        low=current_price * 0.995,
        open=current_price * 1.001,
        bid=current_price * 0.9995,
        ask=current_price * 1.0005,
        trades=mock_data_generator.generate_trades(current_price, 50000, 50),
        order_book=mock_data_generator.generate_order_book(current_price)
    )


@pytest.fixture
def performance_benchmark():
    """Fixture for performance benchmark"""
    return PerformanceBenchmark()


@pytest.fixture
def backtest_validator():
    """Fixture for backtest validator"""
    return BacktestValidator()


# Assertion helpers
def assert_performance_threshold(actual: float, 
                               threshold: float, 
                               metric_name: str,
                               higher_is_better: bool = True):
    """Assert performance meets threshold"""
    if higher_is_better:
        assert actual >= threshold, f"{metric_name} {actual:.3f} below threshold {threshold:.3f}"
    else:
        assert actual <= threshold, f"{metric_name} {actual:.3f} above threshold {threshold:.3f}"


def assert_latency_threshold(latency: float, 
                           max_latency: float,
                           operation_name: str):
    """Assert latency meets threshold"""
    assert latency <= max_latency, f"{operation_name} latency {latency:.3f}s exceeds threshold {max_latency:.3f}s"


def assert_statistical_significance(p_value: float, 
                                  alpha: float = 0.05,
                                  test_name: str = "test"):
    """Assert statistical significance"""
    assert p_value < alpha, f"{test_name} not statistically significant (p={p_value:.3f} >= α={alpha})"


# Test utilities for async functions
async def async_test_helper(coro, timeout: float = 5.0):
    """Helper for testing async functions with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        pytest.fail(f"Async operation timed out after {timeout}s")


def measure_async_performance(coro_func):
    """Decorator to measure async function performance"""
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await coro_func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    return wrapper


# Mock strategy for testing framework
class MockStrategy:
    """Mock strategy implementation for testing the test framework itself"""
    
    def __init__(self, **config):
        self.config = config
        self.call_count = 0
    
    def extract_features(self, market_data: MockMarketData) -> Dict[str, float]:
        """Mock feature extraction"""
        self.call_count += 1
        return {
            'price_momentum': np.random.uniform(-1, 1),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'volatility': np.random.uniform(0.01, 0.05)
        }
    
    def generate_signals(self, market_data: MockMarketData) -> Dict[str, float]:
        """Mock signal generation"""
        features = self.extract_features(market_data)
        return {
            'buy_signal': max(0, features['price_momentum'] + np.random.uniform(-0.2, 0.2)),
            'sell_signal': max(0, -features['price_momentum'] + np.random.uniform(-0.2, 0.2))
        }
    
    def update(self, market_data: MockMarketData):
        """Mock update method"""
        self.extract_features(market_data)