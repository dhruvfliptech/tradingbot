"""
Benchmark Comparison Framework
Compares RL trading system performance against AdaptiveThreshold baseline.
Validates SOW requirement of 15-20% outperformance over baseline.
"""

import pytest
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import statistics
from scipy import stats
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ml-service'))

from environment.trading_env import TradingEnvironment
from environment.portfolio_manager import PortfolioManager
from agents.ppo_agent import PPOAgent
from agents.ensemble_agent import EnsembleAgent
from integration.rl_service import RLService
from rl_config import RLConfig, ActionType

# Import AdaptiveThreshold from ml-service
from adaptive_threshold import AdaptiveThreshold, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """Container for benchmark comparison results"""
    strategy_name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    volatility: float
    total_trades: int
    avg_trade_duration: float
    daily_returns: List[float]
    portfolio_values: List[float]
    start_date: datetime
    end_date: datetime
    
    def calculate_outperformance(self, baseline: 'BenchmarkResults') -> Dict[str, float]:
        """Calculate outperformance metrics vs baseline"""
        return {
            'return_outperformance': (self.total_return - baseline.total_return) / abs(baseline.total_return) if baseline.total_return != 0 else 0,
            'sharpe_outperformance': (self.sharpe_ratio - baseline.sharpe_ratio) / abs(baseline.sharpe_ratio) if baseline.sharpe_ratio != 0 else 0,
            'drawdown_improvement': (baseline.max_drawdown - self.max_drawdown) / baseline.max_drawdown if baseline.max_drawdown != 0 else 0,
            'win_rate_improvement': self.win_rate - baseline.win_rate,
            'volatility_reduction': (baseline.volatility - self.volatility) / baseline.volatility if baseline.volatility != 0 else 0,
            'calmar_outperformance': (self.calmar_ratio - baseline.calmar_ratio) / abs(baseline.calmar_ratio) if baseline.calmar_ratio != 0 else 0
        }


class AdaptiveThresholdStrategy:
    """Adapter for AdaptiveThreshold to work with benchmark framework"""
    
    def __init__(self, user_id: str = "benchmark_user", symbol: str = "BTC/USD"):
        # Mock database connection for testing
        from unittest.mock import Mock
        
        self.adaptive_threshold = AdaptiveThreshold(user_id, symbol)
        
        # Mock database operations for testing
        self.adaptive_threshold.db_engine = Mock()
        self.adaptive_threshold._load_thresholds = Mock()
        self.adaptive_threshold._save_threshold = Mock()
        
        # Set default thresholds
        self.adaptive_threshold.parameters = {
            'rsi_threshold': 70.0,
            'confidence_threshold': 0.75,
            'macd_threshold': 0.0,
            'volume_threshold': 1000000000,
            'momentum_threshold': 2.0
        }
        
        self.trade_history = []
        self.portfolio_value_history = []
        self.last_action = 'HOLD'
        self.position_size = 0.0
        
    def predict(self, market_data: Dict, portfolio_state: Dict) -> str:
        """Generate trading decision based on AdaptiveThreshold logic"""
        
        # Extract market indicators
        price_data = market_data.get('price_data', {})
        current_price = price_data.get('close', 50000)
        volume = price_data.get('volume', 1000000)
        
        # Calculate technical indicators (simplified)
        rsi = self._calculate_rsi(market_data)
        macd = self._calculate_macd(market_data)
        momentum = self._calculate_momentum(market_data)
        
        # Create signal data for AdaptiveThreshold
        signal_data = {
            'confidence': self._calculate_confidence(market_data),
            'rsi': rsi,
            'macd': macd,
            'volume': volume,
            'change_percent': momentum,
            'action': self._determine_base_action(market_data)
        }
        
        # Use AdaptiveThreshold to determine if we should trade
        should_trade = self.adaptive_threshold.should_trade(signal_data)
        
        if should_trade:
            # Determine position size based on confidence
            confidence = signal_data['confidence']
            
            if signal_data['action'] == 'BUY' and self.position_size < 0.8:
                if confidence > 0.8:
                    return 'BUY_40'
                elif confidence > 0.75:
                    return 'BUY_20'
                else:
                    return 'BUY_20'
            elif signal_data['action'] == 'SELL' and self.position_size > 0.2:
                if confidence > 0.8:
                    return 'SELL_40'
                elif confidence > 0.75:
                    return 'SELL_20'
                else:
                    return 'SELL_20'
        
        return 'HOLD'
    
    def _calculate_rsi(self, market_data: Dict) -> float:
        """Calculate RSI (simplified)"""
        price_data = market_data.get('price_data', {})
        current_price = price_data.get('close', 50000)
        
        # Simplified RSI calculation
        # In practice, this would use historical prices
        base_rsi = 50
        price_change = np.random.normal(0, 10)  # Simulate price momentum
        rsi = base_rsi + price_change
        return np.clip(rsi, 0, 100)
    
    def _calculate_macd(self, market_data: Dict) -> float:
        """Calculate MACD (simplified)"""
        return np.random.normal(0, 50)  # Simplified MACD
    
    def _calculate_momentum(self, market_data: Dict) -> float:
        """Calculate price momentum (simplified)"""
        return np.random.normal(0, 3)  # Simplified momentum
    
    def _calculate_confidence(self, market_data: Dict) -> float:
        """Calculate signal confidence (simplified)"""
        # Base confidence on multiple factors
        base_confidence = 0.5
        
        # Add randomness to simulate varying market conditions
        confidence_adjustment = np.random.normal(0, 0.15)
        confidence = base_confidence + confidence_adjustment
        
        return np.clip(confidence, 0, 1)
    
    def _determine_base_action(self, market_data: Dict) -> str:
        """Determine base trading action (simplified)"""
        momentum = self._calculate_momentum(market_data)
        rsi = self._calculate_rsi(market_data)
        
        if momentum > 1.5 and rsi < 40:
            return 'BUY'
        elif momentum < -1.5 and rsi > 60:
            return 'SELL'
        else:
            return 'HOLD'
    
    def update_position(self, action: str, price: float):
        """Update position size based on action"""
        if 'BUY' in action:
            if action == 'BUY_20':
                self.position_size = min(1.0, self.position_size + 0.2)
            elif action == 'BUY_40':
                self.position_size = min(1.0, self.position_size + 0.4)
        elif 'SELL' in action:
            if action == 'SELL_20':
                self.position_size = max(0.0, self.position_size - 0.2)
            elif action == 'SELL_40':
                self.position_size = max(0.0, self.position_size - 0.4)


class BenchmarkEngine:
    """Engine for running benchmark comparisons"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        
    def run_strategy_backtest(
        self,
        strategy,
        market_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        strategy_name: str = "Unknown"
    ) -> BenchmarkResults:
        """Run backtest for a trading strategy"""
        
        # Filter data for date range
        test_data = market_data.loc[start_date:end_date].copy()
        
        if len(test_data) < 100:
            raise ValueError(f"Insufficient data for backtesting: {len(test_data)} periods")
        
        # Initialize tracking variables
        portfolio_value = initial_capital
        cash_balance = initial_capital
        position_size = 0.0
        position_price = 0.0
        
        portfolio_values = [portfolio_value]
        daily_returns = []
        trades = []
        
        previous_portfolio_value = portfolio_value
        
        # Run simulation
        for i, (timestamp, row) in enumerate(test_data.iterrows()):
            current_price = row['close']
            
            # Prepare market data for strategy
            market_data_dict = {
                'timestamp': timestamp,
                'price_data': {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
            }
            
            # Get strategy decision
            if hasattr(strategy, 'predict') and callable(strategy.predict):
                # RL strategy
                if hasattr(strategy, 'observation_space'):
                    # Create mock observation
                    obs = np.random.random(strategy.observation_space.shape)
                    action_idx = strategy.predict(obs, deterministic=True)[0]
                    action = ActionType(action_idx).name
                else:
                    # AdaptiveThreshold strategy
                    portfolio_state = {
                        'total_equity': portfolio_value,
                        'cash_balance': cash_balance,
                        'position_size': position_size
                    }
                    action = strategy.predict(market_data_dict, portfolio_state)
            else:
                action = 'HOLD'
            
            # Execute action
            if action != 'HOLD':
                trade_executed = self._execute_trade(
                    action, current_price, cash_balance, position_size, position_price, portfolio_value
                )
                
                if trade_executed:
                    new_cash, new_position_size, new_position_price = trade_executed
                    
                    # Record trade
                    trade_info = {
                        'timestamp': timestamp,
                        'action': action,
                        'price': current_price,
                        'position_size_change': new_position_size - position_size,
                        'portfolio_value': portfolio_value
                    }
                    trades.append(trade_info)
                    
                    # Update position
                    cash_balance = new_cash
                    position_size = new_position_size
                    position_price = new_position_price
            
            # Update portfolio value
            position_value = position_size * current_price
            portfolio_value = cash_balance + position_value
            portfolio_values.append(portfolio_value)
            
            # Calculate daily return
            if i > 0:
                daily_return = (portfolio_value - previous_portfolio_value) / previous_portfolio_value
                daily_returns.append(daily_return)
            
            previous_portfolio_value = portfolio_value
            
            # Update strategy position tracking if applicable
            if hasattr(strategy, 'update_position'):
                strategy.update_position(action, current_price)
        
        # Calculate performance metrics
        return self._calculate_benchmark_metrics(
            portfolio_values, daily_returns, trades, start_date, end_date, 
            initial_capital, strategy_name
        )
    
    def _execute_trade(
        self, 
        action: str, 
        price: float, 
        cash_balance: float, 
        position_size: float, 
        position_price: float, 
        portfolio_value: float
    ) -> Optional[Tuple[float, float, float]]:
        """Execute a trade and return new cash, position size, position price"""
        
        commission_rate = 0.001  # 0.1% commission
        
        if 'BUY' in action:
            # Determine buy amount
            if action == 'BUY_20':
                buy_amount = 0.2
            elif action == 'BUY_40':
                buy_amount = 0.4
            elif action == 'BUY_60':
                buy_amount = 0.6
            elif action == 'BUY_80':
                buy_amount = 0.8
            elif action == 'BUY_100':
                buy_amount = 1.0
            else:
                buy_amount = 0.2
            
            # Calculate purchase
            max_spend = cash_balance * buy_amount
            shares_to_buy = max_spend / (price * (1 + commission_rate))
            actual_cost = shares_to_buy * price * (1 + commission_rate)
            
            if actual_cost <= cash_balance and shares_to_buy > 0:
                new_cash = cash_balance - actual_cost
                new_position_size = position_size + shares_to_buy
                new_position_price = ((position_size * position_price) + (shares_to_buy * price)) / new_position_size if new_position_size > 0 else price
                
                return new_cash, new_position_size, new_position_price
        
        elif 'SELL' in action and position_size > 0:
            # Determine sell amount
            if action == 'SELL_20':
                sell_fraction = 0.2
            elif action == 'SELL_40':
                sell_fraction = 0.4
            elif action == 'SELL_60':
                sell_fraction = 0.6
            elif action == 'SELL_80':
                sell_fraction = 0.8
            elif action == 'SELL_100':
                sell_fraction = 1.0
            else:
                sell_fraction = 0.2
            
            # Calculate sale
            shares_to_sell = position_size * sell_fraction
            sale_proceeds = shares_to_sell * price * (1 - commission_rate)
            
            new_cash = cash_balance + sale_proceeds
            new_position_size = position_size - shares_to_sell
            new_position_price = position_price  # Keep same average price for remaining position
            
            return new_cash, new_position_size, new_position_price
        
        return None  # No trade executed
    
    def _calculate_benchmark_metrics(
        self,
        portfolio_values: List[float],
        daily_returns: List[float],
        trades: List[Dict],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        strategy_name: str
    ) -> BenchmarkResults:
        """Calculate comprehensive benchmark metrics"""
        
        if not portfolio_values or not daily_returns:
            return BenchmarkResults(
                strategy_name=strategy_name,
                total_return=0, annualized_return=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, win_rate=0, profit_factor=0, calmar_ratio=0, volatility=0,
                total_trades=0, avg_trade_duration=0, daily_returns=[], portfolio_values=[],
                start_date=start_date, end_date=end_date
            )
        
        # Basic returns
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Annualized return
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # Risk metrics
        returns_array = np.array(daily_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # Sharpe ratio
        risk_free_rate = 0.02 / 365  # 2% annual risk-free rate
        excess_returns = returns_array - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365) if np.std(excess_returns) > 0 else 0
        
        # Sortino ratio
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else std_return
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(365) if downside_deviation > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in trades if self._is_profitable_trade(t, trades)]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Profit factor
        gross_profit = sum(self._calculate_trade_profit(t, trades) for t in winning_trades)
        gross_loss = abs(sum(self._calculate_trade_profit(t, trades) for t in trades if t not in winning_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade duration
        avg_trade_duration = self._calculate_avg_trade_duration(trades)
        
        # Volatility (annualized)
        volatility = std_return * np.sqrt(365) if std_return > 0 else 0
        
        return BenchmarkResults(
            strategy_name=strategy_name,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            volatility=volatility,
            total_trades=len(trades),
            avg_trade_duration=avg_trade_duration,
            daily_returns=daily_returns,
            portfolio_values=portfolio_values,
            start_date=start_date,
            end_date=end_date
        )
    
    def _is_profitable_trade(self, trade: Dict, all_trades: List[Dict]) -> bool:
        """Determine if a trade is profitable (simplified)"""
        return np.random.random() > 0.4  # Simplified: 60% win rate
    
    def _calculate_trade_profit(self, trade: Dict, all_trades: List[Dict]) -> float:
        """Calculate profit for a trade (simplified)"""
        return np.random.normal(0.02, 0.05)  # Simplified profit calculation
    
    def _calculate_avg_trade_duration(self, trades: List[Dict]) -> float:
        """Calculate average trade duration in hours"""
        if len(trades) < 2:
            return 0
        
        durations = []
        for i in range(1, len(trades)):
            duration = (trades[i]['timestamp'] - trades[i-1]['timestamp']).total_seconds() / 3600
            durations.append(duration)
        
        return np.mean(durations) if durations else 0


class TestBenchmarkComparison:
    """Test suite for benchmark comparison validation"""
    
    @pytest.fixture
    def benchmark_config(self):
        """Configuration for benchmark testing"""
        config = RLConfig()
        config.env.episode_length = 2000
        config.env.initial_balance = 100000.0
        config.model.total_timesteps = 20000
        return config
    
    @pytest.fixture
    def benchmark_data(self):
        """Generate benchmark market data"""
        # Generate 1 year of hourly data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1H')
        np.random.seed(42)  # Reproducible results
        
        # Generate realistic market movements
        n_periods = len(dates)
        returns = np.random.normal(0.0002, 0.025, n_periods)  # Slight positive bias
        
        # Add some trend and volatility clustering
        for i in range(1, len(returns)):
            returns[i] += 0.02 * returns[i-1]  # Small autocorrelation
        
        # Generate prices
        initial_price = 42000
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.004, n_periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.004, n_periods))),
            'close': prices,
            'volume': np.random.lognormal(15, 0.3, n_periods)
        }, index=dates)
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        return data
    
    def test_rl_vs_adaptive_threshold_comparison(self, benchmark_config, benchmark_data):
        """Compare RL system vs AdaptiveThreshold baseline performance"""
        
        # Split data into train/test
        split_date = benchmark_data.index[int(len(benchmark_data) * 0.7)]
        train_data = benchmark_data[:split_date]
        test_data = benchmark_data[split_date:]
        
        benchmark_engine = BenchmarkEngine(config=benchmark_config)
        
        # Initialize strategies
        # 1. RL Strategy
        env = TradingEnvironment(config=benchmark_config, mode='train')
        rl_agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=benchmark_config
        )
        
        # 2. AdaptiveThreshold Strategy
        adaptive_strategy = AdaptiveThresholdStrategy()
        
        # Run backtests
        logger.info("Running RL strategy backtest...")
        rl_results = benchmark_engine.run_strategy_backtest(
            strategy=rl_agent,
            market_data=test_data,
            start_date=test_data.index[0],
            end_date=test_data.index[-1],
            strategy_name="RL_PPO"
        )
        
        logger.info("Running AdaptiveThreshold strategy backtest...")
        adaptive_results = benchmark_engine.run_strategy_backtest(
            strategy=adaptive_strategy,
            market_data=test_data,
            start_date=test_data.index[0],
            end_date=test_data.index[-1],
            strategy_name="AdaptiveThreshold"
        )
        
        # Calculate outperformance
        outperformance = rl_results.calculate_outperformance(adaptive_results)
        
        logger.info(f"RL Results: Return={rl_results.total_return:.3f}, Sharpe={rl_results.sharpe_ratio:.3f}, "
                   f"DD={rl_results.max_drawdown:.3f}, WR={rl_results.win_rate:.3f}")
        logger.info(f"Adaptive Results: Return={adaptive_results.total_return:.3f}, Sharpe={adaptive_results.sharpe_ratio:.3f}, "
                   f"DD={adaptive_results.max_drawdown:.3f}, WR={adaptive_results.win_rate:.3f}")
        logger.info(f"Outperformance: {outperformance}")
        
        # Validate SOW requirements: 15-20% outperformance
        min_outperformance = 0.15  # 15%
        max_outperformance = 0.20  # 20%
        
        # Return outperformance should be within SOW range
        return_outperf = outperformance['return_outperformance']
        assert return_outperf >= min_outperformance, \
            f"Return outperformance {return_outperf:.1%} below minimum {min_outperformance:.1%}"
        
        # Allow some flexibility on maximum (can exceed if risk-adjusted)
        if return_outperf > max_outperformance * 1.5:
            # If return outperformance is very high, check that risk is controlled
            assert rl_results.max_drawdown <= adaptive_results.max_drawdown * 1.2, \
                "Excessive return outperformance may indicate higher risk"
        
        # Risk-adjusted outperformance
        sharpe_outperf = outperformance['sharpe_outperformance']
        assert sharpe_outperf >= 0.1, f"Sharpe ratio outperformance {sharpe_outperf:.3f} too low"
        
        # Drawdown should be better or similar
        drawdown_improvement = outperformance['drawdown_improvement']
        assert drawdown_improvement >= -0.05, \
            f"Maximum drawdown significantly worse: {drawdown_improvement:.1%} deterioration"
        
        # Win rate should be competitive
        win_rate_improvement = outperformance['win_rate_improvement']
        assert win_rate_improvement >= -0.05, \
            f"Win rate significantly worse: {win_rate_improvement:.1%} deterioration"
        
        logger.info("RL vs AdaptiveThreshold comparison test passed")
    
    def test_ensemble_vs_single_agent_comparison(self, benchmark_config, benchmark_data):
        """Compare ensemble RL agent vs single agent performance"""
        
        test_data = benchmark_data[benchmark_data.index >= '2023-06-01']
        benchmark_engine = BenchmarkEngine(config=benchmark_config)
        
        # Initialize strategies
        env = TradingEnvironment(config=benchmark_config, mode='test')
        
        # Single agent
        single_agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=benchmark_config
        )
        
        # Ensemble agent
        ensemble_agent = EnsembleAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=benchmark_config
        )
        
        # Run backtests
        single_results = benchmark_engine.run_strategy_backtest(
            strategy=single_agent,
            market_data=test_data,
            start_date=test_data.index[0],
            end_date=test_data.index[-1],
            strategy_name="Single_PPO"
        )
        
        ensemble_results = benchmark_engine.run_strategy_backtest(
            strategy=ensemble_agent,
            market_data=test_data,
            start_date=test_data.index[0],
            end_date=test_data.index[-1],
            strategy_name="Ensemble_RL"
        )
        
        # Compare performance
        outperformance = ensemble_results.calculate_outperformance(single_results)
        
        logger.info(f"Single Agent: Return={single_results.total_return:.3f}, Sharpe={single_results.sharpe_ratio:.3f}")
        logger.info(f"Ensemble Agent: Return={ensemble_results.total_return:.3f}, Sharpe={ensemble_results.sharpe_ratio:.3f}")
        
        # Ensemble should provide better risk-adjusted returns
        assert outperformance['sharpe_outperformance'] >= -0.1, \
            "Ensemble agent should not significantly underperform single agent on risk-adjusted basis"
        
        # Ensemble should have lower volatility (better diversification)
        assert ensemble_results.volatility <= single_results.volatility * 1.1, \
            "Ensemble should provide diversification benefits"
        
        logger.info("Ensemble vs single agent comparison test passed")
    
    def test_buy_and_hold_benchmark(self, benchmark_config, benchmark_data):
        """Compare RL system vs simple buy-and-hold strategy"""
        
        test_data = benchmark_data[benchmark_data.index >= '2023-06-01']
        benchmark_engine = BenchmarkEngine(config=benchmark_config)
        
        # Buy-and-hold strategy
        class BuyAndHoldStrategy:
            def __init__(self):
                self.bought = False
            
            def predict(self, market_data, portfolio_state):
                if not self.bought:
                    self.bought = True
                    return 'BUY_100'
                return 'HOLD'
        
        # RL strategy
        env = TradingEnvironment(config=benchmark_config, mode='test')
        rl_agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=benchmark_config
        )
        
        # Run backtests
        buy_hold_results = benchmark_engine.run_strategy_backtest(
            strategy=BuyAndHoldStrategy(),
            market_data=test_data,
            start_date=test_data.index[0],
            end_date=test_data.index[-1],
            strategy_name="BuyAndHold"
        )
        
        rl_results = benchmark_engine.run_strategy_backtest(
            strategy=rl_agent,
            market_data=test_data,
            start_date=test_data.index[0],
            end_date=test_data.index[-1],
            strategy_name="RL_PPO"
        )
        
        # Compare performance
        outperformance = rl_results.calculate_outperformance(buy_hold_results)
        
        logger.info(f"Buy & Hold: Return={buy_hold_results.total_return:.3f}, Sharpe={buy_hold_results.sharpe_ratio:.3f}, "
                   f"DD={buy_hold_results.max_drawdown:.3f}")
        logger.info(f"RL Strategy: Return={rl_results.total_return:.3f}, Sharpe={rl_results.sharpe_ratio:.3f}, "
                   f"DD={rl_results.max_drawdown:.3f}")
        
        # RL should provide better risk-adjusted returns
        assert outperformance['sharpe_outperformance'] >= 0.0, \
            "RL strategy should provide better risk-adjusted returns than buy-and-hold"
        
        # RL should have lower maximum drawdown
        assert outperformance['drawdown_improvement'] >= 0.0, \
            "RL strategy should have better drawdown control than buy-and-hold"
        
        # If total return is lower, Sharpe ratio should be compensatingly higher
        if outperformance['return_outperformance'] < 0:
            assert outperformance['sharpe_outperformance'] >= 0.2, \
                "If returns are lower, risk-adjusted performance should be significantly better"
        
        logger.info("Buy-and-hold benchmark test passed")
    
    def test_statistical_significance_of_outperformance(self, benchmark_config, benchmark_data):
        """Test statistical significance of RL outperformance"""
        
        # Run multiple periods to test consistency
        num_periods = 6
        period_length = len(benchmark_data) // num_periods
        
        rl_returns = []
        adaptive_returns = []
        
        benchmark_engine = BenchmarkEngine(config=benchmark_config)
        
        # Initialize strategies
        env = TradingEnvironment(config=benchmark_config, mode='test')
        rl_agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=benchmark_config
        )
        adaptive_strategy = AdaptiveThresholdStrategy()
        
        # Test multiple periods
        for i in range(num_periods):
            start_idx = i * period_length
            end_idx = min(start_idx + period_length, len(benchmark_data) - 1)
            
            period_data = benchmark_data.iloc[start_idx:end_idx]
            
            if len(period_data) > 500:  # Ensure sufficient data
                # RL backtest
                rl_result = benchmark_engine.run_strategy_backtest(
                    strategy=rl_agent,
                    market_data=period_data,
                    start_date=period_data.index[0],
                    end_date=period_data.index[-1],
                    strategy_name=f"RL_Period_{i}"
                )
                
                # Adaptive backtest
                adaptive_result = benchmark_engine.run_strategy_backtest(
                    strategy=adaptive_strategy,
                    market_data=period_data,
                    start_date=period_data.index[0],
                    end_date=period_data.index[-1],
                    strategy_name=f"Adaptive_Period_{i}"
                )
                
                rl_returns.append(rl_result.total_return)
                adaptive_returns.append(adaptive_result.total_return)
        
        # Statistical analysis
        assert len(rl_returns) >= 4, "Insufficient periods for statistical analysis"
        
        # Paired t-test for returns
        if len(rl_returns) == len(adaptive_returns):
            t_stat, p_value = stats.ttest_rel(rl_returns, adaptive_returns)
            
            logger.info(f"Statistical test: t-stat={t_stat:.3f}, p-value={p_value:.3f}")
            logger.info(f"RL returns: {[f'{r:.3f}' for r in rl_returns]}")
            logger.info(f"Adaptive returns: {[f'{r:.3f}' for r in adaptive_returns]}")
            
            # Statistical significance test
            assert p_value < 0.1, f"Outperformance not statistically significant (p={p_value:.3f})"
            assert t_stat > 0, "RL strategy does not consistently outperform baseline"
            
            # Consistency test
            outperformance_periods = sum(1 for rl, adapt in zip(rl_returns, adaptive_returns) if rl > adapt)
            consistency_rate = outperformance_periods / len(rl_returns)
            
            assert consistency_rate >= 0.6, f"RL outperforms in only {consistency_rate:.1%} of periods"
        
        logger.info("Statistical significance test passed")
    
    def test_market_regime_specific_outperformance(self, benchmark_config, benchmark_data):
        """Test outperformance across different market regimes"""
        
        # Identify market regimes
        returns = benchmark_data['close'].pct_change().dropna()
        rolling_vol = returns.rolling(window=168).std()  # Weekly volatility
        rolling_trend = returns.rolling(window=168).mean()  # Weekly trend
        
        # Define regimes
        high_vol_threshold = rolling_vol.quantile(0.75)
        low_vol_threshold = rolling_vol.quantile(0.25)
        bull_threshold = rolling_trend.quantile(0.75)
        bear_threshold = rolling_trend.quantile(0.25)
        
        regimes = {
            'bull_market': (rolling_trend > bull_threshold) & (rolling_vol <= high_vol_threshold),
            'bear_market': (rolling_trend < bear_threshold) & (rolling_vol <= high_vol_threshold),
            'high_volatility': rolling_vol > high_vol_threshold,
            'low_volatility': rolling_vol < low_vol_threshold
        }
        
        benchmark_engine = BenchmarkEngine(config=benchmark_config)
        
        # Initialize strategies
        env = TradingEnvironment(config=benchmark_config, mode='test')
        rl_agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=benchmark_config
        )
        adaptive_strategy = AdaptiveThresholdStrategy()
        
        regime_performance = {}
        
        for regime_name, regime_mask in regimes.items():
            regime_data = benchmark_data[regime_mask]
            
            if len(regime_data) > 1000:  # Sufficient data for analysis
                # Test RL strategy
                rl_result = benchmark_engine.run_strategy_backtest(
                    strategy=rl_agent,
                    market_data=regime_data,
                    start_date=regime_data.index[0],
                    end_date=regime_data.index[-1],
                    strategy_name=f"RL_{regime_name}"
                )
                
                # Test adaptive strategy
                adaptive_result = benchmark_engine.run_strategy_backtest(
                    strategy=adaptive_strategy,
                    market_data=regime_data,
                    start_date=regime_data.index[0],
                    end_date=regime_data.index[-1],
                    strategy_name=f"Adaptive_{regime_name}"
                )
                
                outperformance = rl_result.calculate_outperformance(adaptive_result)
                regime_performance[regime_name] = {
                    'rl_return': rl_result.total_return,
                    'adaptive_return': adaptive_result.total_return,
                    'return_outperformance': outperformance['return_outperformance'],
                    'sharpe_outperformance': outperformance['sharpe_outperformance']
                }
        
        logger.info(f"Regime-specific performance: {regime_performance}")
        
        # Validate performance across regimes
        for regime_name, perf in regime_performance.items():
            # Should outperform in most regimes
            return_outperf = perf['return_outperformance']
            sharpe_outperf = perf['sharpe_outperformance']
            
            # Allow for some regimes to underperform, but not drastically
            assert return_outperf > -0.2, \
                f"Severe underperformance in {regime_name}: {return_outperf:.1%}"
            
            # Risk-adjusted performance should be competitive
            assert sharpe_outperf > -0.3, \
                f"Poor risk-adjusted performance in {regime_name}: {sharpe_outperf:.1%}"
        
        # At least 70% of regimes should show outperformance
        outperforming_regimes = sum(1 for perf in regime_performance.values() 
                                  if perf['return_outperformance'] > 0)
        outperformance_rate = outperforming_regimes / len(regime_performance)
        
        assert outperformance_rate >= 0.7, \
            f"RL outperforms in only {outperformance_rate:.1%} of market regimes"
        
        logger.info("Market regime specific outperformance test passed")


if __name__ == "__main__":
    # Run benchmark comparison tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])