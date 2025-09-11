"""
Validation Module for RL Pre-training Pipeline

This module validates the performance of pre-trained RL agents against baselines
and provides comprehensive evaluation metrics to ensure pre-training effectiveness.
"""

import asyncio
import logging
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os

# Import RL components
import sys
sys.path.append('..')
from agents.ppo_agent import PPOAgent
from environment.trading_env import TradingEnvironment
from environment.market_simulator import MarketSimulator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for validation experiments"""
    # Evaluation settings
    validation_episodes: int = 200
    validation_symbols: List[str] = None
    evaluation_periods: List[str] = None  # ['1M', '3M', '6M', '1Y']
    
    # Baseline comparisons
    compare_against_random: bool = True
    compare_against_vanilla_rl: bool = True
    compare_against_buy_hold: bool = True
    
    # Statistical testing
    significance_level: float = 0.05
    confidence_interval: float = 0.95
    
    # Performance metrics
    metrics_to_evaluate: List[str] = None
    
    # Environment settings
    initial_balance: float = 100000.0
    transaction_cost: float = 0.001
    max_episode_steps: int = 1000
    
    def __post_init__(self):
        if self.validation_symbols is None:
            self.validation_symbols = ['BTC-USD', 'ETH-USD', 'SPY', 'TSLA']
        if self.evaluation_periods is None:
            self.evaluation_periods = ['1M', '3M', '6M']
        if self.metrics_to_evaluate is None:
            self.metrics_to_evaluate = [
                'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
                'profit_factor', 'volatility', 'sortino_ratio', 'calmar_ratio'
            ]

@dataclass
class ValidationResult:
    """Results from validation experiment"""
    agent_name: str
    symbol: str
    period: str
    episodes_completed: int
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    
    # Trade statistics
    total_trades: int
    avg_trade_duration: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Episode statistics
    episode_returns: List[float]
    episode_sharpes: List[float]
    episode_drawdowns: List[float]
    
    # Timing
    validation_time: float
    timestamp: datetime

@dataclass
class ComparisonResult:
    """Results from comparing multiple agents"""
    pretrained_results: ValidationResult
    baseline_results: Dict[str, ValidationResult]
    
    # Statistical comparisons
    statistical_significance: Dict[str, Dict[str, float]]  # metric -> baseline -> p_value
    effect_sizes: Dict[str, Dict[str, float]]  # metric -> baseline -> effect_size
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Performance improvements
    relative_improvements: Dict[str, Dict[str, float]]  # metric -> baseline -> improvement_%
    
    # Recommendations
    validation_passed: bool
    improvement_summary: str
    recommendations: List[str]

class TradingEnvironmentValidator:
    """Validates trading environments for consistency"""
    
    def __init__(self):
        self.validation_checks = [
            self._check_data_quality,
            self._check_action_space,
            self._check_observation_space,
            self._check_reward_function,
            self._check_episode_termination
        ]
    
    def validate_environment(self, env: TradingEnvironment) -> Dict[str, bool]:
        """Validate trading environment"""
        results = {}
        
        for check in self.validation_checks:
            try:
                check_name = check.__name__
                results[check_name] = check(env)
                logger.debug(f"Environment check {check_name}: {'PASS' if results[check_name] else 'FAIL'}")
            except Exception as e:
                logger.error(f"Environment check {check.__name__} failed: {e}")
                results[check.__name__] = False
        
        return results
    
    def _check_data_quality(self, env: TradingEnvironment) -> bool:
        """Check data quality in environment"""
        # Reset environment and get initial observation
        obs = env.reset()
        
        # Check for NaN or infinite values
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            return False
        
        # Check observation bounds
        if np.any(obs < -1e6) or np.any(obs > 1e6):
            return False
        
        return True
    
    def _check_action_space(self, env: TradingEnvironment) -> bool:
        """Check action space validity"""
        try:
            # Test all valid actions
            for action in range(env.action_space.n):
                env.reset()
                obs, reward, done, info = env.step(action)
                
                # Check return types
                if not isinstance(reward, (int, float, np.number)):
                    return False
                if not isinstance(done, bool):
                    return False
                if not isinstance(info, dict):
                    return False
            
            return True
        except Exception:
            return False
    
    def _check_observation_space(self, env: TradingEnvironment) -> bool:
        """Check observation space consistency"""
        obs1 = env.reset()
        obs2 = env.reset()
        
        # Check shape consistency
        if obs1.shape != obs2.shape:
            return False
        
        # Check if observations are in expected range
        if not env.observation_space.contains(obs1):
            return False
        
        return True
    
    def _check_reward_function(self, env: TradingEnvironment) -> bool:
        """Check reward function behavior"""
        env.reset()
        
        rewards = []
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            
            if done:
                env.reset()
        
        # Check for reasonable reward range
        if any(abs(r) > 1e3 for r in rewards):
            return False
        
        return True
    
    def _check_episode_termination(self, env: TradingEnvironment) -> bool:
        """Check episode termination logic"""
        env.reset()
        
        step_count = 0
        done = False
        
        while not done and step_count < env.max_episode_steps * 2:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            step_count += 1
        
        # Episode should terminate within reasonable steps
        return step_count <= env.max_episode_steps

class BaselineAgent:
    """Baseline agents for comparison"""
    
    @staticmethod
    def random_agent(observation_space, action_space):
        """Random action agent"""
        class RandomAgent:
            def __init__(self):
                self.action_space = action_space
            
            def predict(self, obs, deterministic=True):
                return self.action_space.sample(), None
        
        return RandomAgent()
    
    @staticmethod
    def buy_and_hold_agent(observation_space, action_space):
        """Buy and hold strategy agent"""
        class BuyHoldAgent:
            def __init__(self):
                self.action_space = action_space
                self.bought = False
            
            def predict(self, obs, deterministic=True):
                if not self.bought:
                    self.bought = True
                    return 1, None  # Buy action
                return 0, None  # Hold action
        
        return BuyHoldAgent()
    
    @staticmethod
    def vanilla_rl_agent(state_dim, action_dim, hidden_dims):
        """Vanilla RL agent without pre-training"""
        return PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        )

class PerformanceMetrics:
    """Calculate trading performance metrics"""
    
    @staticmethod
    def calculate_returns(equity_curve: np.ndarray) -> float:
        """Calculate total return"""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(equity_curve) < 2:
            return 0.0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return abs(np.min(drawdown))
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        annual_return = np.mean(returns) * 252
        max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / max_dd
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> float:
        """Calculate win rate from trades"""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        return winning_trades / len(trades)
    
    @staticmethod
    def calculate_profit_factor(trades: List[Dict]) -> float:
        """Calculate profit factor"""
        if not trades:
            return 0.0
        
        gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss

class PretrainValidator:
    """Main validation class for pre-trained RL agents"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.env_validator = TradingEnvironmentValidator()
        self.metrics_calculator = PerformanceMetrics()
        
        # Results storage
        self.validation_results: List[ValidationResult] = []
        self.comparison_results: List[ComparisonResult] = []
    
    async def validate_pretrained_performance(self,
                                           agent_path: str,
                                           validation_episodes: Optional[int] = None,
                                           validation_envs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate pre-trained agent performance"""
        logger.info("Starting pre-trained agent validation...")
        
        validation_episodes = validation_episodes or self.config.validation_episodes
        validation_envs = validation_envs or self.config.validation_symbols
        
        # Load pre-trained agent
        pretrained_agent = self._load_agent(agent_path)
        if pretrained_agent is None:
            return {'validation_completed': False, 'reason': 'Failed to load agent'}
        
        # Run validation for each environment/symbol
        all_results = {}
        
        for symbol in validation_envs:
            logger.info(f"Validating on {symbol}...")
            
            # Create trading environment
            env = self._create_trading_environment(symbol)
            
            # Validate environment
            env_checks = self.env_validator.validate_environment(env)
            if not all(env_checks.values()):
                logger.warning(f"Environment validation failed for {symbol}: {env_checks}")
                continue
            
            # Run episodes
            symbol_results = await self._run_validation_episodes(
                agent=pretrained_agent,
                env=env,
                symbol=symbol,
                episodes=validation_episodes
            )
            
            all_results[symbol] = symbol_results
        
        # Aggregate results
        aggregated_results = self._aggregate_validation_results(all_results)
        
        # Compare against baselines
        if self.config.compare_against_random or self.config.compare_against_vanilla_rl:
            comparison_results = await self._run_baseline_comparisons(
                pretrained_agent=pretrained_agent,
                validation_envs=validation_envs,
                episodes=validation_episodes
            )
            aggregated_results['baseline_comparisons'] = comparison_results
        
        # Statistical analysis
        statistical_results = self._perform_statistical_analysis(all_results)
        aggregated_results['statistical_analysis'] = statistical_results
        
        # Generate recommendations
        recommendations = self._generate_validation_recommendations(aggregated_results)
        aggregated_results['recommendations'] = recommendations
        
        logger.info("Pre-trained agent validation completed")
        return aggregated_results
    
    def _load_agent(self, agent_path: str) -> Optional[PPOAgent]:
        """Load pre-trained agent from file"""
        try:
            # Load the state dict
            state_dict = torch.load(agent_path, map_location='cpu')
            
            # Create agent with standard configuration
            agent = PPOAgent(
                state_dim=64,  # Should match training config
                action_dim=3,
                hidden_dims=[512, 256, 128]
            )
            
            # Load weights
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                agent.load_state_dict(state_dict['model_state_dict'])
            else:
                agent.load_state_dict(state_dict)
            
            agent.eval()
            logger.info(f"Loaded pre-trained agent from {agent_path}")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
            return None
    
    def _create_trading_environment(self, symbol: str) -> TradingEnvironment:
        """Create trading environment for validation"""
        # Create market simulator
        market_sim = MarketSimulator(
            symbols=[symbol],
            initial_balance=self.config.initial_balance,
            transaction_cost=self.config.transaction_cost
        )
        
        # Create trading environment
        env = TradingEnvironment(
            market_simulator=market_sim,
            max_episode_steps=self.config.max_episode_steps
        )
        
        return env
    
    async def _run_validation_episodes(self,
                                     agent: PPOAgent,
                                     env: TradingEnvironment,
                                     symbol: str,
                                     episodes: int) -> ValidationResult:
        """Run validation episodes for a single agent/environment"""
        start_time = datetime.now()
        
        episode_returns = []
        episode_sharpes = []
        episode_drawdowns = []
        all_trades = []
        
        for episode in range(episodes):
            # Run single episode
            episode_data = await self._run_single_episode(agent, env)
            
            episode_returns.append(episode_data['total_return'])
            episode_sharpes.append(episode_data['sharpe_ratio'])
            episode_drawdowns.append(episode_data['max_drawdown'])
            all_trades.extend(episode_data['trades'])
            
            if episode % 50 == 0:
                logger.debug(f"Completed episode {episode}/{episodes}")
        
        # Calculate aggregate metrics
        total_return = np.mean(episode_returns)
        annualized_return = total_return * 252  # Assuming daily episodes
        sharpe_ratio = np.mean(episode_sharpes)
        sortino_ratio = self.metrics_calculator.calculate_sortino_ratio(
            np.array(episode_returns)
        )
        max_drawdown = np.mean(episode_drawdowns)
        volatility = np.std(episode_returns) * np.sqrt(252)
        
        # Trade statistics
        win_rate = self.metrics_calculator.calculate_win_rate(all_trades)
        profit_factor = self.metrics_calculator.calculate_profit_factor(all_trades)
        
        avg_trade_duration = np.mean([t.get('duration', 0) for t in all_trades]) if all_trades else 0
        avg_win = np.mean([t['pnl'] for t in all_trades if t.get('pnl', 0) > 0]) if all_trades else 0
        avg_loss = np.mean([t['pnl'] for t in all_trades if t.get('pnl', 0) < 0]) if all_trades else 0
        
        # Calculate consecutive win/loss streaks
        pnls = [t.get('pnl', 0) for t in all_trades]
        max_consecutive_wins = self._calculate_max_consecutive(pnls, lambda x: x > 0)
        max_consecutive_losses = self._calculate_max_consecutive(pnls, lambda x: x < 0)
        
        validation_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        return ValidationResult(
            agent_name="pretrained",
            symbol=symbol,
            period="validation",
            episodes_completed=episodes,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(all_trades),
            avg_trade_duration=avg_trade_duration,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            episode_returns=episode_returns,
            episode_sharpes=episode_sharpes,
            episode_drawdowns=episode_drawdowns,
            validation_time=validation_time,
            timestamp=datetime.now()
        )
    
    async def _run_single_episode(self, agent: PPOAgent, env: TradingEnvironment) -> Dict[str, Any]:
        """Run a single validation episode"""
        obs = env.reset()
        done = False
        
        equity_curve = [env.initial_balance]
        returns = []
        trades = []
        
        while not done:
            # Get action from agent
            with torch.no_grad():
                action, _ = agent.predict(obs, deterministic=True)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Record data
            current_value = info.get('portfolio_value', env.initial_balance)
            equity_curve.append(current_value)
            
            if len(equity_curve) > 1:
                daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                returns.append(daily_return)
            
            # Record trades
            if 'trade_completed' in info and info['trade_completed']:
                trades.append(info['trade_info'])
            
            obs = next_obs
        
        # Calculate episode metrics
        episode_data = {
            'total_return': self.metrics_calculator.calculate_returns(np.array(equity_curve)),
            'sharpe_ratio': self.metrics_calculator.calculate_sharpe_ratio(np.array(returns)),
            'max_drawdown': self.metrics_calculator.calculate_max_drawdown(np.array(equity_curve)),
            'trades': trades,
            'equity_curve': equity_curve,
            'returns': returns
        }
        
        return episode_data
    
    def _calculate_max_consecutive(self, values: List[float], condition) -> int:
        """Calculate maximum consecutive occurrences of condition"""
        if not values:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in values:
            if condition(value):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    async def _run_baseline_comparisons(self,
                                      pretrained_agent: PPOAgent,
                                      validation_envs: List[str],
                                      episodes: int) -> Dict[str, Any]:
        """Run comparisons against baseline agents"""
        logger.info("Running baseline comparisons...")
        
        baseline_results = {}
        
        for symbol in validation_envs:
            env = self._create_trading_environment(symbol)
            symbol_baselines = {}
            
            # Random agent
            if self.config.compare_against_random:
                random_agent = BaselineAgent.random_agent(
                    env.observation_space, env.action_space
                )
                random_results = await self._run_validation_episodes(
                    random_agent, env, symbol, episodes // 4  # Fewer episodes for baselines
                )
                symbol_baselines['random'] = random_results
            
            # Buy and hold agent
            if self.config.compare_against_buy_hold:
                buy_hold_agent = BaselineAgent.buy_and_hold_agent(
                    env.observation_space, env.action_space
                )
                buy_hold_results = await self._run_validation_episodes(
                    buy_hold_agent, env, symbol, episodes // 4
                )
                symbol_baselines['buy_hold'] = buy_hold_results
            
            # Vanilla RL agent (untrained)
            if self.config.compare_against_vanilla_rl:
                vanilla_agent = BaselineAgent.vanilla_rl_agent(
                    state_dim=64, action_dim=3, hidden_dims=[512, 256, 128]
                )
                vanilla_results = await self._run_validation_episodes(
                    vanilla_agent, env, symbol, episodes // 4
                )
                symbol_baselines['vanilla_rl'] = vanilla_results
            
            baseline_results[symbol] = symbol_baselines
        
        return baseline_results
    
    def _aggregate_validation_results(self, all_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Aggregate validation results across symbols"""
        if not all_results:
            return {}
        
        # Aggregate metrics
        aggregated = {
            'symbols_tested': list(all_results.keys()),
            'total_episodes': sum(r.episodes_completed for r in all_results.values()),
            'average_metrics': {},
            'per_symbol_results': all_results
        }
        
        # Calculate average metrics
        metrics = self.config.metrics_to_evaluate
        for metric in metrics:
            values = [getattr(result, metric, 0) for result in all_results.values()]
            aggregated['average_metrics'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated
    
    def _perform_statistical_analysis(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Perform statistical analysis of validation results"""
        if not results:
            return {}
        
        analysis = {
            'normality_tests': {},
            'confidence_intervals': {},
            'risk_metrics': {}
        }
        
        # Test normality of returns
        for symbol, result in results.items():
            returns = result.episode_returns
            if len(returns) > 8:  # Minimum for Shapiro-Wilk test
                stat, p_value = stats.shapiro(returns)
                analysis['normality_tests'][symbol] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > self.config.significance_level
                }
        
        # Calculate confidence intervals
        for symbol, result in results.items():
            returns = result.episode_returns
            if len(returns) > 1:
                mean_return = np.mean(returns)
                sem = stats.sem(returns)
                ci = stats.t.interval(
                    confidence=self.config.confidence_interval,
                    df=len(returns)-1,
                    loc=mean_return,
                    scale=sem
                )
                analysis['confidence_intervals'][symbol] = {
                    'lower': ci[0],
                    'upper': ci[1],
                    'mean': mean_return
                }
        
        # Risk metrics
        all_returns = []
        for result in results.values():
            all_returns.extend(result.episode_returns)
        
        if all_returns:
            analysis['risk_metrics'] = {
                'var_95': np.percentile(all_returns, 5),
                'var_99': np.percentile(all_returns, 1),
                'expected_shortfall_95': np.mean([r for r in all_returns if r <= np.percentile(all_returns, 5)]),
                'skewness': stats.skew(all_returns),
                'kurtosis': stats.kurtosis(all_returns)
            }
        
        return analysis
    
    def _generate_validation_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        if 'average_metrics' not in results:
            recommendations.append("Insufficient validation data for recommendations")
            return recommendations
        
        avg_metrics = results['average_metrics']
        
        # Performance recommendations
        sharpe_ratio = avg_metrics.get('sharpe_ratio', {}).get('mean', 0)
        if sharpe_ratio > 1.5:
            recommendations.append("Excellent risk-adjusted performance achieved")
        elif sharpe_ratio > 1.0:
            recommendations.append("Good risk-adjusted performance, consider position sizing optimization")
        elif sharpe_ratio > 0.5:
            recommendations.append("Moderate performance, consider additional pre-training or feature engineering")
        else:
            recommendations.append("Poor performance, pre-training may need significant improvements")
        
        # Drawdown recommendations
        max_drawdown = avg_metrics.get('max_drawdown', {}).get('mean', 0)
        if max_drawdown > 0.3:
            recommendations.append("High drawdown detected - implement stronger risk management")
        elif max_drawdown > 0.2:
            recommendations.append("Moderate drawdown - consider position sizing adjustments")
        
        # Win rate recommendations
        win_rate = avg_metrics.get('win_rate', {}).get('mean', 0)
        if win_rate < 0.4:
            recommendations.append("Low win rate - consider improving entry signal quality")
        elif win_rate > 0.7:
            recommendations.append("High win rate - consider increasing position sizes if risk allows")
        
        # Volatility recommendations
        volatility = avg_metrics.get('volatility', {}).get('mean', 0)
        if volatility > 0.5:
            recommendations.append("High volatility - implement volatility-adjusted position sizing")
        
        # Statistical significance
        if 'statistical_analysis' in results:
            stat_analysis = results['statistical_analysis']
            if 'normality_tests' in stat_analysis:
                non_normal_symbols = [
                    symbol for symbol, test in stat_analysis['normality_tests'].items()
                    if not test['is_normal']
                ]
                if len(non_normal_symbols) > len(stat_analysis['normality_tests']) / 2:
                    recommendations.append("Returns are non-normal - consider robust risk metrics")
        
        return recommendations
    
    def create_validation_report(self, results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Create comprehensive validation report"""
        report = []
        report.append("=" * 80)
        report.append("RL PRE-TRAINING VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        if 'average_metrics' in results:
            avg_metrics = results['average_metrics']
            report.append(f"Symbols Tested: {len(results.get('symbols_tested', []))}")
            report.append(f"Total Episodes: {results.get('total_episodes', 0)}")
            report.append(f"Average Sharpe Ratio: {avg_metrics.get('sharpe_ratio', {}).get('mean', 0):.3f}")
            report.append(f"Average Max Drawdown: {avg_metrics.get('max_drawdown', {}).get('mean', 0):.3f}")
            report.append(f"Average Win Rate: {avg_metrics.get('win_rate', {}).get('mean', 0):.3f}")
        report.append("")
        
        # Detailed Results
        report.append("DETAILED RESULTS BY SYMBOL")
        report.append("-" * 40)
        if 'per_symbol_results' in results:
            for symbol, result in results['per_symbol_results'].items():
                report.append(f"\n{symbol}:")
                report.append(f"  Episodes: {result.episodes_completed}")
                report.append(f"  Total Return: {result.total_return:.3f}")
                report.append(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
                report.append(f"  Max Drawdown: {result.max_drawdown:.3f}")
                report.append(f"  Win Rate: {result.win_rate:.3f}")
                report.append(f"  Profit Factor: {result.profit_factor:.3f}")
        
        # Statistical Analysis
        if 'statistical_analysis' in results:
            report.append("\nSTATISTICAL ANALYSIS")
            report.append("-" * 40)
            stat_analysis = results['statistical_analysis']
            
            if 'risk_metrics' in stat_analysis:
                risk_metrics = stat_analysis['risk_metrics']
                report.append(f"VaR (95%): {risk_metrics.get('var_95', 0):.4f}")
                report.append(f"VaR (99%): {risk_metrics.get('var_99', 0):.4f}")
                report.append(f"Expected Shortfall (95%): {risk_metrics.get('expected_shortfall_95', 0):.4f}")
                report.append(f"Skewness: {risk_metrics.get('skewness', 0):.3f}")
                report.append(f"Kurtosis: {risk_metrics.get('kurtosis', 0):.3f}")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS")
        report.append("-" * 40)
        for i, rec in enumerate(results.get('recommendations', []), 1):
            report.append(f"{i}. {rec}")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {save_path}")
        
        return report_text
    
    def visualize_validation_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Create visualizations of validation results"""
        if 'per_symbol_results' not in results:
            logger.warning("No per-symbol results available for visualization")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RL Pre-training Validation Results', fontsize=16)
        
        # Prepare data
        symbols = list(results['per_symbol_results'].keys())
        sharpe_ratios = [r.sharpe_ratio for r in results['per_symbol_results'].values()]
        max_drawdowns = [r.max_drawdown for r in results['per_symbol_results'].values()]
        win_rates = [r.win_rate for r in results['per_symbol_results'].values()]
        total_returns = [r.total_return for r in results['per_symbol_results'].values()]
        profit_factors = [r.profit_factor for r in results['per_symbol_results'].values()]
        
        # Sharpe ratios by symbol
        axes[0, 0].bar(symbols, sharpe_ratios)
        axes[0, 0].set_title('Sharpe Ratio by Symbol')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Max drawdowns
        axes[0, 1].bar(symbols, max_drawdowns, color='red', alpha=0.7)
        axes[0, 1].set_title('Max Drawdown by Symbol')
        axes[0, 1].set_ylabel('Max Drawdown')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Win rates
        axes[0, 2].bar(symbols, win_rates, color='green', alpha=0.7)
        axes[0, 2].set_title('Win Rate by Symbol')
        axes[0, 2].set_ylabel('Win Rate')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Returns distribution
        all_episode_returns = []
        for result in results['per_symbol_results'].values():
            all_episode_returns.extend(result.episode_returns)
        
        axes[1, 0].hist(all_episode_returns, bins=50, alpha=0.7)
        axes[1, 0].set_title('Distribution of Episode Returns')
        axes[1, 0].set_xlabel('Return')
        axes[1, 0].set_ylabel('Frequency')
        
        # Risk-return scatter
        axes[1, 1].scatter(max_drawdowns, sharpe_ratios, s=100)
        for i, symbol in enumerate(symbols):
            axes[1, 1].annotate(symbol, (max_drawdowns[i], sharpe_ratios[i]))
        axes[1, 1].set_xlabel('Max Drawdown')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].set_title('Risk-Return Profile')
        
        # Profit factors
        axes[1, 2].bar(symbols, profit_factors, color='orange', alpha=0.7)
        axes[1, 2].set_title('Profit Factor by Symbol')
        axes[1, 2].set_ylabel('Profit Factor')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation visualization saved to {save_path}")
        
        plt.show()
    
    def export_validation_results(self, results: Dict[str, Any], export_path: str):
        """Export validation results to JSON file"""
        try:
            # Make results JSON serializable
            serializable_results = self._make_json_serializable(results)
            
            with open(export_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Validation results exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export validation results: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(asdict(obj))
        else:
            return obj

# Example usage
async def main():
    """Example usage of PretrainValidator"""
    config = ValidationConfig(
        validation_episodes=100,
        validation_symbols=['BTC-USD', 'ETH-USD'],
        compare_against_random=True,
        compare_against_vanilla_rl=True
    )
    
    validator = PretrainValidator(config)
    
    # Simulate validation (would use real pre-trained agent path)
    agent_path = "/path/to/pretrained_agent.pt"
    
    if os.path.exists(agent_path):
        results = await validator.validate_pretrained_performance(agent_path)
        
        # Create report
        report = validator.create_validation_report(results, "validation_report.txt")
        print(report)
        
        # Create visualizations
        validator.visualize_validation_results(results, "validation_plots.png")
        
        # Export results
        validator.export_validation_results(results, "validation_results.json")
    else:
        print("Pre-trained agent not found - skipping validation")

if __name__ == "__main__":
    asyncio.run(main())