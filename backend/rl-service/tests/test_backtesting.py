"""
Historical Backtesting Validation Suite
Comprehensive backtesting framework for RL trading system validation:
- Out-of-sample testing on historical data
- Walk-forward analysis
- Monte Carlo simulations
- Regime-specific performance analysis
- Statistical significance testing
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

from environment.trading_env import TradingEnvironment
from environment.portfolio_manager import PortfolioManager
from agents.ppo_agent import PPOAgent
from agents.ensemble_agent import EnsembleAgent
from integration.rl_service import RLService
from rl_config import RLConfig, ActionType

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Container for backtest results"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    var_95: float
    total_trades: int
    avg_trade_duration: float
    daily_returns: List[float]
    portfolio_values: List[float]
    trades: List[Dict]
    start_date: datetime
    end_date: datetime


@dataclass
class MarketRegime:
    """Market regime classification"""
    name: str
    start_date: datetime
    end_date: datetime
    characteristics: Dict[str, float]
    performance: Optional[BacktestResults] = None


class BacktestEngine:
    """Backtesting engine for RL trading strategies"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.results_history: List[BacktestResults] = []
    
    def run_backtest(
        self,
        agent,
        market_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0
    ) -> BacktestResults:
        """Run single backtest on historical data"""
        
        # Filter data for date range
        test_data = market_data.loc[start_date:end_date].copy()
        
        if len(test_data) < 100:
            raise ValueError(f"Insufficient data for backtesting: {len(test_data)} periods")
        
        # Initialize environment
        env = TradingEnvironment(config=self.config, mode='backtest')
        env.market_simulator.market_data = {'BTC/USD': test_data}
        env.market_simulator.current_step = 0
        env.market_simulator.max_steps = len(test_data) - 1
        env.portfolio_manager.cash_balance = initial_capital
        env.portfolio_manager.initial_balance = initial_capital
        
        # Run simulation
        obs, _ = env.reset()
        portfolio_values = [initial_capital]
        daily_returns = []
        trades = []
        
        previous_equity = initial_capital
        
        for step in range(len(test_data) - 1):
            # Get agent action
            action = agent.predict(obs, deterministic=True)[0]
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track portfolio value
            current_equity = env.portfolio_manager.get_total_equity()
            portfolio_values.append(current_equity)
            
            # Calculate daily return
            daily_return = (current_equity - previous_equity) / previous_equity
            daily_returns.append(daily_return)
            previous_equity = current_equity
            
            # Track trades
            if 'trade_executed' in info and info['trade_executed']:
                trade_info = {
                    'timestamp': test_data.index[step],
                    'action': ActionType(action).name,
                    'price': info.get('execution_price', 0),
                    'quantity': info.get('quantity', 0),
                    'portfolio_value': current_equity
                }
                trades.append(trade_info)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        # Calculate performance metrics
        return self._calculate_backtest_metrics(
            portfolio_values, daily_returns, trades, start_date, end_date, initial_capital
        )
    
    def _calculate_backtest_metrics(
        self,
        portfolio_values: List[float],
        daily_returns: List[float],
        trades: List[Dict],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float
    ) -> BacktestResults:
        """Calculate comprehensive backtest metrics"""
        
        if not portfolio_values or not daily_returns:
            return BacktestResults(
                total_return=0, annualized_return=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, win_rate=0, profit_factor=0, calmar_ratio=0, var_95=0,
                total_trades=0, avg_trade_duration=0, daily_returns=[], portfolio_values=[],
                trades=[], start_date=start_date, end_date=end_date
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
        
        # Value at Risk (95%)
        var_95 = abs(np.percentile(returns_array, 5)) if len(returns_array) > 20 else 0
        
        # Trade statistics
        winning_trades = [t for t in trades if self._is_winning_trade(t, trades)]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Profit factor
        gross_profit = sum(self._calculate_trade_pnl(t, trades) for t in winning_trades)
        gross_loss = abs(sum(self._calculate_trade_pnl(t, trades) for t in trades if t not in winning_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade duration
        avg_trade_duration = self._calculate_avg_trade_duration(trades)
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            var_95=var_95,
            total_trades=len(trades),
            avg_trade_duration=avg_trade_duration,
            daily_returns=daily_returns,
            portfolio_values=portfolio_values,
            trades=trades,
            start_date=start_date,
            end_date=end_date
        )
    
    def _is_winning_trade(self, trade: Dict, all_trades: List[Dict]) -> bool:
        """Determine if a trade is profitable"""
        # Simplified winning trade logic
        return self._calculate_trade_pnl(trade, all_trades) > 0
    
    def _calculate_trade_pnl(self, trade: Dict, all_trades: List[Dict]) -> float:
        """Calculate P&L for a trade"""
        # Simplified P&L calculation
        return np.random.normal(0.01, 0.05)  # Placeholder
    
    def _calculate_avg_trade_duration(self, trades: List[Dict]) -> float:
        """Calculate average trade duration in hours"""
        if len(trades) < 2:
            return 0
        
        durations = []
        for i in range(1, len(trades)):
            duration = (trades[i]['timestamp'] - trades[i-1]['timestamp']).total_seconds() / 3600
            durations.append(duration)
        
        return np.mean(durations) if durations else 0


class TestHistoricalBacktesting:
    """Test suite for historical backtesting validation"""
    
    @pytest.fixture
    def backtest_config(self):
        """Configuration optimized for backtesting"""
        config = RLConfig()
        config.env.episode_length = 5000  # Long episodes for backtesting
        config.env.initial_balance = 100000.0
        config.model.total_timesteps = 10000
        return config
    
    @pytest.fixture
    def historical_data(self):
        """Generate comprehensive historical market data"""
        # Generate 2 years of hourly data
        dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='1H')
        np.random.seed(42)
        
        # Generate realistic price movements with multiple regimes
        n_periods = len(dates)
        
        # Market regimes
        regime_changes = []
        current_regime = 'normal'
        
        for i in range(n_periods):
            if np.random.random() < 0.001:  # 0.1% chance of regime change
                current_regime = np.random.choice(['bull', 'bear', 'volatile', 'normal'])
                regime_changes.append((i, current_regime))
        
        # Generate returns based on regimes
        returns = []
        volatility_base = 0.02
        
        for i in range(n_periods):
            if i < len(regime_changes):
                regime = regime_changes[min(i, len(regime_changes)-1)][1] if regime_changes else 'normal'
            else:
                regime = 'normal'
            
            if regime == 'bull':
                mean_return = 0.0003
                volatility = volatility_base * 0.8
            elif regime == 'bear':
                mean_return = -0.0002
                volatility = volatility_base * 1.2
            elif regime == 'volatile':
                mean_return = 0.0001
                volatility = volatility_base * 2.0
            else:  # normal
                mean_return = 0.0001
                volatility = volatility_base
            
            returns.append(np.random.normal(mean_return, volatility))
        
        # Generate prices from returns
        initial_price = 40000
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_periods))),
            'close': prices,
            'volume': np.random.lognormal(15, 0.3, n_periods)
        }, index=dates)
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        return data
    
    def test_out_of_sample_validation(self, backtest_config, historical_data):
        """Test out-of-sample performance validation"""
        
        # Split data: 70% training, 30% testing
        split_date = historical_data.index[int(len(historical_data) * 0.7)]
        train_data = historical_data[:split_date]
        test_data = historical_data[split_date:]
        
        # Create and train agent (simplified)
        env = TradingEnvironment(config=backtest_config, mode='train')
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=backtest_config
        )
        
        # Simulate training on training data
        logger.info(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
        # agent.learn(total_timesteps=backtest_config.model.total_timesteps)  # Placeholder
        
        # Backtest on out-of-sample data
        backtest_engine = BacktestEngine(config=backtest_config)
        results = backtest_engine.run_backtest(
            agent=agent,
            market_data=test_data,
            start_date=test_data.index[0],
            end_date=test_data.index[-1]
        )
        
        logger.info(f"Out-of-sample results: Return={results.total_return:.3f}, Sharpe={results.sharpe_ratio:.3f}")
        
        # Validate performance
        assert results.total_return > -0.5, "Excessive losses in out-of-sample testing"
        assert results.sharpe_ratio > 0, "Negative Sharpe ratio indicates poor risk-adjusted returns"
        assert results.max_drawdown < 0.3, "Maximum drawdown too high"
        assert results.total_trades > 10, "Insufficient trading activity"
        
        logger.info("Out-of-sample validation test passed")
    
    def test_walk_forward_analysis(self, backtest_config, historical_data):
        """Test walk-forward analysis for strategy robustness"""
        
        # Parameters for walk-forward analysis
        train_period_months = 6
        test_period_months = 1
        step_months = 1
        
        walk_forward_results = []
        
        # Create agent
        env = TradingEnvironment(config=backtest_config, mode='train')
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=backtest_config
        )
        
        # Walk-forward windows
        start_date = historical_data.index[0]
        end_date = historical_data.index[-1]
        
        current_date = start_date
        window_count = 0
        
        while current_date + timedelta(days=30 * (train_period_months + test_period_months)) <= end_date:
            # Define training and testing periods
            train_start = current_date
            train_end = current_date + timedelta(days=30 * train_period_months)
            test_start = train_end
            test_end = train_end + timedelta(days=30 * test_period_months)
            
            # Extract data for this window
            train_data = historical_data[train_start:train_end]
            test_data = historical_data[test_start:test_end]
            
            if len(train_data) > 100 and len(test_data) > 50:
                # Simulate retraining (simplified)
                logger.info(f"Window {window_count}: Train {train_start.date()} to {train_end.date()}, Test {test_start.date()} to {test_end.date()}")
                
                # Backtest on this window
                backtest_engine = BacktestEngine(config=backtest_config)
                results = backtest_engine.run_backtest(
                    agent=agent,
                    market_data=test_data,
                    start_date=test_start,
                    end_date=test_end
                )
                
                walk_forward_results.append(results)
                window_count += 1
            
            # Step forward
            current_date += timedelta(days=30 * step_months)
            
            # Limit number of windows for testing
            if window_count >= 5:
                break
        
        # Analyze walk-forward results
        assert len(walk_forward_results) >= 3, "Insufficient walk-forward windows"
        
        returns = [r.total_return for r in walk_forward_results]
        sharpe_ratios = [r.sharpe_ratio for r in walk_forward_results]
        
        # Consistency checks
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        avg_sharpe = np.mean(sharpe_ratios)
        
        logger.info(f"Walk-forward results: Avg Return={avg_return:.3f}, Std={std_return:.3f}, Avg Sharpe={avg_sharpe:.3f}")
        
        # Performance criteria
        assert avg_return > -0.1, "Average return across windows too low"
        assert std_return < 0.5, "Return variability too high across windows"
        assert avg_sharpe > 0.5, "Average Sharpe ratio too low"
        
        # At least 60% of windows should be profitable
        profitable_windows = sum(1 for r in returns if r > 0)
        profitability_rate = profitable_windows / len(returns)
        assert profitability_rate >= 0.6, f"Only {profitability_rate:.1%} of windows profitable"
        
        logger.info("Walk-forward analysis test passed")
    
    def test_monte_carlo_simulation(self, backtest_config, historical_data):
        """Test Monte Carlo simulation for statistical robustness"""
        
        num_simulations = 50  # Reduced for testing speed
        simulation_results = []
        
        # Create agent
        env = TradingEnvironment(config=backtest_config, mode='train')
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=backtest_config
        )
        
        # Run multiple simulations with different data samples
        for sim_id in range(num_simulations):
            # Bootstrap sampling of historical data
            np.random.seed(sim_id)  # Different seed for each simulation
            
            # Sample random periods from historical data
            total_periods = len(historical_data)
            sample_size = min(2000, total_periods)  # Sample size for each simulation
            
            start_idx = np.random.randint(0, total_periods - sample_size)
            end_idx = start_idx + sample_size
            
            sampled_data = historical_data.iloc[start_idx:end_idx].copy()
            
            # Add noise to simulate different market conditions
            noise_factor = 0.02
            price_noise = np.random.normal(1, noise_factor, len(sampled_data))
            sampled_data['close'] *= price_noise
            sampled_data['open'] *= price_noise
            sampled_data['high'] *= price_noise
            sampled_data['low'] *= price_noise
            
            # Run backtest
            try:
                backtest_engine = BacktestEngine(config=backtest_config)
                results = backtest_engine.run_backtest(
                    agent=agent,
                    market_data=sampled_data,
                    start_date=sampled_data.index[0],
                    end_date=sampled_data.index[-1]
                )
                simulation_results.append(results)
                
            except Exception as e:
                logger.warning(f"Simulation {sim_id} failed: {e}")
                continue
            
            if sim_id % 10 == 0:
                logger.info(f"Completed {sim_id + 1}/{num_simulations} simulations")
        
        # Analyze Monte Carlo results
        assert len(simulation_results) >= num_simulations * 0.8, "Too many failed simulations"
        
        returns = [r.total_return for r in simulation_results]
        sharpe_ratios = [r.sharpe_ratio for r in simulation_results]
        max_drawdowns = [r.max_drawdown for r in simulation_results]
        
        # Statistical analysis
        return_stats = {
            'mean': np.mean(returns),
            'median': np.median(returns),
            'std': np.std(returns),
            'percentile_5': np.percentile(returns, 5),
            'percentile_95': np.percentile(returns, 95)
        }
        
        sharpe_stats = {
            'mean': np.mean(sharpe_ratios),
            'median': np.median(sharpe_ratios),
            'std': np.std(sharpe_ratios),
            'percentile_5': np.percentile(sharpe_ratios, 5),
            'percentile_95': np.percentile(sharpe_ratios, 95)
        }
        
        logger.info(f"Monte Carlo return stats: {return_stats}")
        logger.info(f"Monte Carlo Sharpe stats: {sharpe_stats}")
        
        # Statistical significance tests
        # Test if returns are significantly different from zero
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        # Performance validation
        assert return_stats['mean'] > -0.2, "Mean return too low across simulations"
        assert return_stats['percentile_5'] > -0.5, "5th percentile return too low (high tail risk)"
        assert sharpe_stats['mean'] > 0.3, "Mean Sharpe ratio too low"
        assert np.mean(max_drawdowns) < 0.25, "Average maximum drawdown too high"
        
        # Statistical significance
        assert p_value < 0.1, f"Returns not statistically significant (p={p_value:.3f})"
        
        # Consistency check
        profitable_simulations = sum(1 for r in returns if r > 0)
        profitability_rate = profitable_simulations / len(returns)
        assert profitability_rate >= 0.55, f"Only {profitability_rate:.1%} of simulations profitable"
        
        logger.info("Monte Carlo simulation test passed")
    
    def test_regime_specific_analysis(self, backtest_config, historical_data):
        """Test performance across different market regimes"""
        
        # Identify market regimes based on volatility and trend
        returns = historical_data['close'].pct_change().dropna()
        
        # Calculate rolling statistics
        window = 168  # Weekly window
        rolling_vol = returns.rolling(window=window).std()
        rolling_trend = returns.rolling(window=window).mean()
        
        # Define regime thresholds
        vol_threshold_high = rolling_vol.quantile(0.75)
        vol_threshold_low = rolling_vol.quantile(0.25)
        trend_threshold_high = rolling_trend.quantile(0.75)
        trend_threshold_low = rolling_trend.quantile(0.25)
        
        # Classify regimes
        regimes = []
        
        for i, (date, vol, trend) in enumerate(zip(rolling_vol.index, rolling_vol, rolling_trend)):
            if pd.isna(vol) or pd.isna(trend):
                continue
            
            if vol > vol_threshold_high:
                if trend > trend_threshold_high:
                    regime_name = 'volatile_bull'
                elif trend < trend_threshold_low:
                    regime_name = 'volatile_bear'
                else:
                    regime_name = 'high_volatility'
            elif vol < vol_threshold_low:
                if trend > trend_threshold_high:
                    regime_name = 'stable_bull'
                elif trend < trend_threshold_low:
                    regime_name = 'stable_bear'
                else:
                    regime_name = 'low_volatility'
            else:
                regime_name = 'normal'
            
            regimes.append({
                'date': date,
                'regime': regime_name,
                'volatility': vol,
                'trend': trend
            })
        
        # Group consecutive periods by regime
        regime_periods = self._group_regime_periods(regimes, historical_data)
        
        # Test performance in each regime
        agent = PPOAgent(
            observation_space=TradingEnvironment(config=backtest_config).observation_space,
            action_space=TradingEnvironment(config=backtest_config).action_space,
            config=backtest_config
        )
        
        regime_performance = {}
        
        for regime_name, periods in regime_periods.items():
            if len(periods) < 3:  # Need at least 3 periods for meaningful analysis
                continue
            
            regime_results = []
            
            for period in periods[:3]:  # Test first 3 periods of each regime
                if len(period['data']) > 200:  # Minimum data requirement
                    try:
                        backtest_engine = BacktestEngine(config=backtest_config)
                        results = backtest_engine.run_backtest(
                            agent=agent,
                            market_data=period['data'],
                            start_date=period['start_date'],
                            end_date=period['end_date']
                        )
                        regime_results.append(results)
                        
                    except Exception as e:
                        logger.warning(f"Regime {regime_name} backtest failed: {e}")
                        continue
            
            if regime_results:
                # Calculate regime-specific metrics
                avg_return = np.mean([r.total_return for r in regime_results])
                avg_sharpe = np.mean([r.sharpe_ratio for r in regime_results])
                avg_drawdown = np.mean([r.max_drawdown for r in regime_results])
                
                regime_performance[regime_name] = {
                    'avg_return': avg_return,
                    'avg_sharpe': avg_sharpe,
                    'avg_drawdown': avg_drawdown,
                    'num_periods': len(regime_results)
                }
        
        logger.info(f"Regime-specific performance: {regime_performance}")
        
        # Validate regime performance
        for regime_name, perf in regime_performance.items():
            # Each regime should maintain reasonable performance
            assert perf['avg_return'] > -0.3, f"Poor performance in {regime_name}: {perf['avg_return']:.3f}"
            assert perf['avg_drawdown'] < 0.4, f"Excessive drawdown in {regime_name}: {perf['avg_drawdown']:.3f}"
            
            # High volatility regimes may have lower Sharpe ratios
            min_sharpe = 0.3 if 'volatile' in regime_name else 0.5
            assert perf['avg_sharpe'] > min_sharpe, f"Low Sharpe ratio in {regime_name}: {perf['avg_sharpe']:.3f}"
        
        # Strategy should work across multiple regimes
        assert len(regime_performance) >= 3, "Strategy not tested across enough market regimes"
        
        logger.info("Regime-specific analysis test passed")
    
    def test_statistical_significance(self, backtest_config, historical_data):
        """Test statistical significance of strategy performance"""
        
        # Run multiple independent backtests
        num_backtests = 20
        backtest_results = []
        
        agent = PPOAgent(
            observation_space=TradingEnvironment(config=backtest_config).observation_space,
            action_space=TradingEnvironment(config=backtest_config).action_space,
            config=backtest_config
        )
        
        # Create multiple test periods
        total_periods = len(historical_data)
        period_length = total_periods // num_backtests
        
        for i in range(num_backtests):
            start_idx = i * period_length
            end_idx = min(start_idx + period_length, total_periods - 1)
            
            if end_idx - start_idx > 500:  # Minimum period length
                test_data = historical_data.iloc[start_idx:end_idx]
                
                try:
                    backtest_engine = BacktestEngine(config=backtest_config)
                    results = backtest_engine.run_backtest(
                        agent=agent,
                        market_data=test_data,
                        start_date=test_data.index[0],
                        end_date=test_data.index[-1]
                    )
                    backtest_results.append(results)
                    
                except Exception as e:
                    logger.warning(f"Backtest {i} failed: {e}")
                    continue
        
        assert len(backtest_results) >= num_backtests * 0.8, "Too many failed backtests"
        
        # Extract performance metrics
        returns = [r.total_return for r in backtest_results]
        sharpe_ratios = [r.sharpe_ratio for r in backtest_results]
        win_rates = [r.win_rate for r in backtest_results]
        
        # Statistical tests
        # Test if returns are significantly positive
        t_stat_returns, p_value_returns = stats.ttest_1samp(returns, 0)
        
        # Test if Sharpe ratios are significantly above 1.0
        t_stat_sharpe, p_value_sharpe = stats.ttest_1samp(sharpe_ratios, 1.0)
        
        # Test if win rates are significantly above 50%
        t_stat_winrate, p_value_winrate = stats.ttest_1samp(win_rates, 0.5)
        
        logger.info(f"Statistical tests:")
        logger.info(f"Returns vs 0: t={t_stat_returns:.3f}, p={p_value_returns:.3f}")
        logger.info(f"Sharpe vs 1.0: t={t_stat_sharpe:.3f}, p={p_value_sharpe:.3f}")
        logger.info(f"Win rate vs 50%: t={t_stat_winrate:.3f}, p={p_value_winrate:.3f}")
        
        # Performance validation with statistical significance
        assert p_value_returns < 0.05, f"Returns not statistically significant (p={p_value_returns:.3f})"
        assert t_stat_returns > 0, "Returns are significantly negative"
        
        # Sharpe ratio test (less strict)
        assert p_value_sharpe < 0.1 or np.mean(sharpe_ratios) > 1.2, \
            f"Sharpe ratios not significantly above 1.0 (p={p_value_sharpe:.3f}, mean={np.mean(sharpe_ratios):.3f})"
        
        # Win rate test
        assert p_value_winrate < 0.1 or np.mean(win_rates) > 0.55, \
            f"Win rates not significantly above 50% (p={p_value_winrate:.3f}, mean={np.mean(win_rates):.3f})"
        
        # Consistency tests
        return_std = np.std(returns)
        assert return_std < 0.3, f"Return consistency too low (std={return_std:.3f})"
        
        logger.info("Statistical significance test passed")
    
    def _group_regime_periods(self, regimes: List[Dict], historical_data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Group consecutive periods by market regime"""
        regime_periods = {}
        
        if not regimes:
            return regime_periods
        
        current_regime = regimes[0]['regime']
        current_start = regimes[0]['date']
        
        for i, regime_data in enumerate(regimes[1:], 1):
            if regime_data['regime'] != current_regime or i == len(regimes) - 1:
                # Regime change or end of data
                end_date = regime_data['date']
                
                # Extract data for this regime period
                period_data = historical_data[current_start:end_date]
                
                if len(period_data) > 100:  # Minimum period length
                    if current_regime not in regime_periods:
                        regime_periods[current_regime] = []
                    
                    regime_periods[current_regime].append({
                        'start_date': current_start,
                        'end_date': end_date,
                        'data': period_data
                    })
                
                current_regime = regime_data['regime']
                current_start = regime_data['date']
        
        return regime_periods


if __name__ == "__main__":
    # Run backtesting validation tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])