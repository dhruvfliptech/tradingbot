"""
Ensemble Performance Tracker

This module provides comprehensive performance tracking and analysis for the
multi-agent ensemble system. It tracks individual agent performance, ensemble
performance, regime-specific performance, and provides detailed analytics.

Features:
- Real-time performance monitoring
- Individual agent performance tracking
- Ensemble vs individual comparison
- Regime-specific performance analysis
- Risk-adjusted metrics (Sharpe ratio, Sortino ratio, etc.)
- Drawdown analysis and risk metrics
- Performance attribution and decomposition
- Benchmarking against baseline strategies
- Advanced analytics and reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from collections import defaultdict, deque
import math
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import from other ensemble modules
from ..agents.specialized_agents import AgentType
from ..regime.market_regime_detector import MarketRegime
from ..meta.meta_agent_orchestrator import EnsembleDecision

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Types of performance metrics"""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    AVERAGE_WIN = "average_win"
    AVERAGE_LOSS = "average_loss"
    RECOVERY_FACTOR = "recovery_factor"
    ULCER_INDEX = "ulcer_index"
    VaR_95 = "var_95"
    CVaR_95 = "cvar_95"


@dataclass
class PerformanceConfig:
    """Configuration for performance tracking"""
    
    # Tracking windows
    short_window: int = 50  # Short-term performance window
    medium_window: int = 200  # Medium-term performance window
    long_window: int = 1000  # Long-term performance window
    
    # Risk-free rate for risk-adjusted metrics
    risk_free_rate: float = 0.02  # 2% annual
    
    # Performance thresholds
    significance_threshold: float = 0.05  # Statistical significance
    outperformance_threshold: float = 0.02  # 2% outperformance threshold
    
    # Benchmark settings
    enable_benchmarking: bool = True
    benchmark_returns: List[float] = field(default_factory=list)
    
    # Reporting settings
    enable_detailed_logging: bool = True
    log_frequency: int = 100
    save_detailed_history: bool = True
    
    # Advanced analytics
    enable_regime_analysis: bool = True
    enable_attribution_analysis: bool = True
    enable_monte_carlo: bool = True
    monte_carlo_simulations: int = 1000


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time"""
    
    timestamp: datetime
    total_return: float
    period_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    
    # Additional metrics
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Context information
    regime: Optional[MarketRegime] = None
    regime_confidence: float = 0.0
    active_agents: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_return': self.total_return,
            'period_return': self.period_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'profit_factor': self.profit_factor,
            'recovery_factor': self.recovery_factor,
            'ulcer_index': self.ulcer_index,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'regime': self.regime.value if self.regime else None,
            'regime_confidence': self.regime_confidence,
            'active_agents': self.active_agents
        }


class PerformanceCalculator:
    """Calculate various performance metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns: np.ndarray) -> Tuple[float, int, int]:
        """Calculate maximum drawdown and its duration"""
        if len(cumulative_returns) == 0:
            return 0.0, 0, 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / np.maximum(running_max, 1e-10)
        
        # Find maximum drawdown
        max_dd = np.min(drawdown)
        
        # Find drawdown duration
        max_dd_idx = np.argmin(drawdown)
        
        # Find start of drawdown (last peak before max drawdown)
        start_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdown[i] >= -1e-10:  # Near zero (peak)
                start_idx = i
                break
        
        # Find end of drawdown (recovery)
        end_idx = len(cumulative_returns) - 1
        for i in range(max_dd_idx, len(cumulative_returns)):
            if cumulative_returns[i] >= running_max[max_dd_idx]:
                end_idx = i
                break
        
        duration = end_idx - start_idx
        
        return float(max_dd), start_idx, duration
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if len(returns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        cumulative_returns = np.cumprod(1 + returns)
        max_dd, _, _ = PerformanceCalculator.calculate_max_drawdown(cumulative_returns)
        
        if abs(max_dd) < 1e-10:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def calculate_ulcer_index(cumulative_returns: np.ndarray) -> float:
        """Calculate Ulcer Index (average squared drawdown)"""
        if len(cumulative_returns) == 0:
            return 0.0
        
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / np.maximum(running_max, 1e-10)
        
        return np.sqrt(np.mean(drawdown**2))
    
    @staticmethod
    def calculate_var_cvar(returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        if len(returns) == 0:
            return 0.0, 0.0
        
        # VaR at specified confidence level
        var = np.percentile(returns, (1 - confidence) * 100)
        
        # CVaR (expected shortfall)
        tail_returns = returns[returns <= var]
        cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var
        
        return float(var), float(cvar)
    
    @staticmethod
    def calculate_profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if len(returns) == 0:
            return 0.0
        
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_win_rate(returns: np.ndarray) -> float:
        """Calculate win rate (percentage of positive returns)"""
        if len(returns) == 0:
            return 0.0
        
        winning_trades = np.sum(returns > 0)
        return winning_trades / len(returns)
    
    @staticmethod
    def calculate_recovery_factor(cumulative_returns: np.ndarray) -> float:
        """Calculate recovery factor (total return / max drawdown)"""
        if len(cumulative_returns) == 0:
            return 0.0
        
        total_return = cumulative_returns[-1] - cumulative_returns[0]
        max_dd, _, _ = PerformanceCalculator.calculate_max_drawdown(cumulative_returns)
        
        if abs(max_dd) < 1e-10:
            return float('inf') if total_return > 0 else 0.0
        
        return total_return / abs(max_dd)


class AgentPerformanceTracker:
    """Track performance of individual agents"""
    
    def __init__(self, agent_type: AgentType, config: PerformanceConfig):
        self.agent_type = agent_type
        self.config = config
        
        # Performance data
        self.returns = deque(maxlen=config.long_window)
        self.decisions = deque(maxlen=config.long_window)
        self.timestamps = deque(maxlen=config.long_window)
        
        # Performance snapshots
        self.snapshots = deque(maxlen=1000)
        
        # Regime-specific performance
        self.regime_performance = defaultdict(lambda: {
            'returns': deque(maxlen=200),
            'count': 0,
            'total_return': 0.0
        })
        
        # Current state
        self.current_equity = 1.0
        self.last_update = datetime.now()
    
    def update(self, return_value: float, decision: Optional[Dict[str, Any]] = None,
              regime: Optional[MarketRegime] = None):
        """Update agent performance with new return"""
        
        self.returns.append(return_value)
        self.decisions.append(decision)
        self.timestamps.append(datetime.now())
        
        # Update equity curve
        self.current_equity *= (1 + return_value)
        
        # Update regime-specific performance
        if regime:
            regime_data = self.regime_performance[regime]
            regime_data['returns'].append(return_value)
            regime_data['count'] += 1
            regime_data['total_return'] += return_value
        
        self.last_update = datetime.now()
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        
        if len(self.returns) == 0:
            return self._empty_metrics()
        
        returns_array = np.array(list(self.returns))
        cumulative_returns = np.cumprod(1 + returns_array)
        
        # Calculate all metrics
        metrics = {
            'total_return': float(cumulative_returns[-1] - 1),
            'annualized_return': float(np.mean(returns_array) * 252),
            'volatility': float(np.std(returns_array) * np.sqrt(252)),
            'sharpe_ratio': PerformanceCalculator.calculate_sharpe_ratio(returns_array, self.config.risk_free_rate),
            'sortino_ratio': PerformanceCalculator.calculate_sortino_ratio(returns_array, self.config.risk_free_rate),
            'calmar_ratio': PerformanceCalculator.calculate_calmar_ratio(returns_array),
            'profit_factor': PerformanceCalculator.calculate_profit_factor(returns_array),
            'win_rate': PerformanceCalculator.calculate_win_rate(returns_array),
            'recovery_factor': PerformanceCalculator.calculate_recovery_factor(cumulative_returns),
            'ulcer_index': PerformanceCalculator.calculate_ulcer_index(cumulative_returns),
            'num_trades': len(self.returns)
        }
        
        # Max drawdown
        max_dd, dd_start, dd_duration = PerformanceCalculator.calculate_max_drawdown(cumulative_returns)
        metrics['max_drawdown'] = max_dd
        metrics['drawdown_duration'] = dd_duration
        
        # VaR and CVaR
        var_95, cvar_95 = PerformanceCalculator.calculate_var_cvar(returns_array)
        metrics['var_95'] = var_95
        metrics['cvar_95'] = cvar_95
        
        # Average win/loss
        winning_returns = returns_array[returns_array > 0]
        losing_returns = returns_array[returns_array < 0]
        
        metrics['average_win'] = float(np.mean(winning_returns)) if len(winning_returns) > 0 else 0.0
        metrics['average_loss'] = float(np.mean(losing_returns)) if len(losing_returns) > 0 else 0.0
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary"""
        return {metric.value: 0.0 for metric in PerformanceMetric}
    
    def get_regime_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by regime"""
        
        regime_metrics = {}
        
        for regime, data in self.regime_performance.items():
            if len(data['returns']) == 0:
                continue
            
            returns_array = np.array(list(data['returns']))
            
            regime_metrics[regime.value] = {
                'count': data['count'],
                'total_return': data['total_return'],
                'mean_return': float(np.mean(returns_array)),
                'volatility': float(np.std(returns_array)),
                'sharpe_ratio': PerformanceCalculator.calculate_sharpe_ratio(returns_array, self.config.risk_free_rate),
                'win_rate': PerformanceCalculator.calculate_win_rate(returns_array),
                'profit_factor': PerformanceCalculator.calculate_profit_factor(returns_array)
            }
        
        return regime_metrics
    
    def create_snapshot(self, regime: Optional[MarketRegime] = None,
                       regime_confidence: float = 0.0) -> PerformanceSnapshot:
        """Create performance snapshot"""
        
        metrics = self.get_current_metrics()
        
        # Calculate period return (last period)
        period_return = 0.0
        if len(self.returns) > 0:
            period_return = list(self.returns)[-1]
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            total_return=metrics['total_return'],
            period_return=period_return,
            volatility=metrics['volatility'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            sortino_ratio=metrics['sortino_ratio'],
            calmar_ratio=metrics['calmar_ratio'],
            profit_factor=metrics['profit_factor'],
            recovery_factor=metrics['recovery_factor'],
            ulcer_index=metrics['ulcer_index'],
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            regime=regime,
            regime_confidence=regime_confidence,
            active_agents=[self.agent_type.value]
        )
        
        self.snapshots.append(snapshot)
        return snapshot


class EnsemblePerformanceTracker:
    """
    Main performance tracker for the ensemble system
    
    Tracks overall ensemble performance, individual agent performance,
    and provides comprehensive analytics and reporting.
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        
        # Agent trackers
        self.agent_trackers: Dict[AgentType, AgentPerformanceTracker] = {}
        
        # Ensemble performance tracking
        self.ensemble_returns = deque(maxlen=self.config.long_window)
        self.ensemble_decisions = deque(maxlen=self.config.long_window)
        self.ensemble_snapshots = deque(maxlen=1000)
        
        # Regime tracking
        self.regime_history = deque(maxlen=1000)
        
        # Benchmark tracking
        self.benchmark_returns = deque(maxlen=self.config.long_window)
        
        # Performance attribution
        self.attribution_data = deque(maxlen=1000)
        
        # Current state
        self.current_ensemble_equity = 1.0
        self.step_count = 0
        self.last_snapshot_time = datetime.now()
        
        logger.info("Ensemble performance tracker initialized")
    
    def initialize_agent_tracker(self, agent_type: AgentType):
        """Initialize tracker for a specific agent"""
        
        if agent_type not in self.agent_trackers:
            self.agent_trackers[agent_type] = AgentPerformanceTracker(agent_type, self.config)
            logger.info(f"Initialized performance tracker for {agent_type.value}")
    
    def update_ensemble_performance(self, return_value: float, decision: EnsembleDecision,
                                  individual_returns: Optional[Dict[AgentType, float]] = None):
        """
        Update ensemble performance with new return and decision
        
        Args:
            return_value: Ensemble return for this step
            decision: Ensemble decision that led to this return
            individual_returns: Individual agent returns (for attribution)
        """
        
        self.step_count += 1
        
        # Update ensemble performance
        self.ensemble_returns.append(return_value)
        self.ensemble_decisions.append(decision)
        self.current_ensemble_equity *= (1 + return_value)
        
        # Update individual agent performance
        if individual_returns:
            for agent_type, agent_return in individual_returns.items():
                if agent_type not in self.agent_trackers:
                    self.initialize_agent_tracker(agent_type)
                
                self.agent_trackers[agent_type].update(
                    agent_return, 
                    decision.to_dict(),
                    decision.regime
                )
        
        # Update regime history
        self.regime_history.append({
            'regime': decision.regime,
            'confidence': decision.regime_confidence,
            'return': return_value,
            'timestamp': decision.timestamp
        })
        
        # Performance attribution
        if individual_returns and decision.agent_weights:
            self._update_attribution(return_value, decision, individual_returns)
        
        # Create snapshots periodically
        if self.step_count % self.config.log_frequency == 0:
            self._create_ensemble_snapshot(decision)
        
        # Log performance periodically
        if self.config.enable_detailed_logging and self.step_count % self.config.log_frequency == 0:
            self._log_performance_summary()
    
    def _update_attribution(self, ensemble_return: float, decision: EnsembleDecision,
                          individual_returns: Dict[AgentType, float]):
        """Update performance attribution analysis"""
        
        # Calculate contribution of each agent to ensemble return
        contributions = {}
        total_weighted_return = 0.0
        
        for agent_type, agent_return in individual_returns.items():
            weight = decision.agent_weights.get(agent_type.value, 0.0)
            contribution = agent_return * weight
            contributions[agent_type] = contribution
            total_weighted_return += contribution
        
        # Calculate attribution residual (difference between ensemble and weighted sum)
        residual = ensemble_return - total_weighted_return
        
        attribution_record = {
            'timestamp': decision.timestamp,
            'ensemble_return': ensemble_return,
            'individual_returns': individual_returns,
            'agent_weights': decision.agent_weights,
            'contributions': contributions,
            'residual': residual,
            'regime': decision.regime,
            'strategy': decision.strategy_used
        }
        
        self.attribution_data.append(attribution_record)
    
    def _create_ensemble_snapshot(self, decision: EnsembleDecision):
        """Create ensemble performance snapshot"""
        
        if len(self.ensemble_returns) == 0:
            return
        
        returns_array = np.array(list(self.ensemble_returns))
        cumulative_returns = np.cumprod(1 + returns_array)
        
        # Calculate metrics
        metrics = {
            'total_return': float(cumulative_returns[-1] - 1),
            'volatility': float(np.std(returns_array) * np.sqrt(252)),
            'sharpe_ratio': PerformanceCalculator.calculate_sharpe_ratio(returns_array, self.config.risk_free_rate),
            'win_rate': PerformanceCalculator.calculate_win_rate(returns_array),
            'profit_factor': PerformanceCalculator.calculate_profit_factor(returns_array),
            'sortino_ratio': PerformanceCalculator.calculate_sortino_ratio(returns_array, self.config.risk_free_rate),
            'calmar_ratio': PerformanceCalculator.calculate_calmar_ratio(returns_array),
            'recovery_factor': PerformanceCalculator.calculate_recovery_factor(cumulative_returns),
            'ulcer_index': PerformanceCalculator.calculate_ulcer_index(cumulative_returns)
        }
        
        # Max drawdown
        max_dd, _, _ = PerformanceCalculator.calculate_max_drawdown(cumulative_returns)
        metrics['max_drawdown'] = max_dd
        
        # VaR and CVaR
        var_95, cvar_95 = PerformanceCalculator.calculate_var_cvar(returns_array)
        metrics['var_95'] = var_95
        metrics['cvar_95'] = cvar_95
        
        # Period return
        period_return = list(self.ensemble_returns)[-1] if self.ensemble_returns else 0.0
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            total_return=metrics['total_return'],
            period_return=period_return,
            volatility=metrics['volatility'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            sortino_ratio=metrics['sortino_ratio'],
            calmar_ratio=metrics['calmar_ratio'],
            profit_factor=metrics['profit_factor'],
            recovery_factor=metrics['recovery_factor'],
            ulcer_index=metrics['ulcer_index'],
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            regime=decision.regime,
            regime_confidence=decision.regime_confidence,
            active_agents=decision.explanation.get('active_agents', [])
        )
        
        self.ensemble_snapshots.append(snapshot)
        self.last_snapshot_time = datetime.now()
    
    def _log_performance_summary(self):
        """Log performance summary"""
        
        if not self.config.enable_detailed_logging:
            return
        
        logger.info(f"Ensemble Performance Summary (Step {self.step_count}):")
        
        if len(self.ensemble_returns) > 0:
            returns_array = np.array(list(self.ensemble_returns))
            total_return = (self.current_ensemble_equity - 1) * 100
            
            logger.info(f"  Total Return: {total_return:.2f}%")
            logger.info(f"  Sharpe Ratio: {PerformanceCalculator.calculate_sharpe_ratio(returns_array, self.config.risk_free_rate):.3f}")
            logger.info(f"  Win Rate: {PerformanceCalculator.calculate_win_rate(returns_array):.2%}")
            
            # Max drawdown
            cumulative_returns = np.cumprod(1 + returns_array)
            max_dd, _, _ = PerformanceCalculator.calculate_max_drawdown(cumulative_returns)
            logger.info(f"  Max Drawdown: {max_dd:.2%}")
        
        # Agent performance summary
        for agent_type, tracker in self.agent_trackers.items():
            metrics = tracker.get_current_metrics()
            if metrics['num_trades'] > 0:
                logger.info(f"  {agent_type.value}: Return={metrics['total_return']:.2%}, "
                          f"Sharpe={metrics['sharpe_ratio']:.2f}, Trades={metrics['num_trades']}")
    
    def get_ensemble_metrics(self) -> Dict[str, float]:
        """Get current ensemble performance metrics"""
        
        if len(self.ensemble_returns) == 0:
            return {}
        
        returns_array = np.array(list(self.ensemble_returns))
        cumulative_returns = np.cumprod(1 + returns_array)
        
        # Calculate all metrics
        metrics = {
            'total_return': float(cumulative_returns[-1] - 1),
            'annualized_return': float(np.mean(returns_array) * 252),
            'volatility': float(np.std(returns_array) * np.sqrt(252)),
            'sharpe_ratio': PerformanceCalculator.calculate_sharpe_ratio(returns_array, self.config.risk_free_rate),
            'sortino_ratio': PerformanceCalculator.calculate_sortino_ratio(returns_array, self.config.risk_free_rate),
            'calmar_ratio': PerformanceCalculator.calculate_calmar_ratio(returns_array),
            'profit_factor': PerformanceCalculator.calculate_profit_factor(returns_array),
            'win_rate': PerformanceCalculator.calculate_win_rate(returns_array),
            'recovery_factor': PerformanceCalculator.calculate_recovery_factor(cumulative_returns),
            'ulcer_index': PerformanceCalculator.calculate_ulcer_index(cumulative_returns),
            'num_trades': len(self.ensemble_returns),
            'current_equity': float(self.current_ensemble_equity)
        }
        
        # Max drawdown
        max_dd, dd_start, dd_duration = PerformanceCalculator.calculate_max_drawdown(cumulative_returns)
        metrics['max_drawdown'] = max_dd
        metrics['drawdown_duration'] = dd_duration
        
        # VaR and CVaR
        var_95, cvar_95 = PerformanceCalculator.calculate_var_cvar(returns_array)
        metrics['var_95'] = var_95
        metrics['cvar_95'] = cvar_95
        
        # Average win/loss
        winning_returns = returns_array[returns_array > 0]
        losing_returns = returns_array[returns_array < 0]
        
        metrics['average_win'] = float(np.mean(winning_returns)) if len(winning_returns) > 0 else 0.0
        metrics['average_loss'] = float(np.mean(losing_returns)) if len(losing_returns) > 0 else 0.0
        
        return metrics
    
    def get_agent_comparison(self) -> Dict[str, Dict[str, float]]:
        """Get performance comparison across all agents"""
        
        comparison = {}
        
        for agent_type, tracker in self.agent_trackers.items():
            comparison[agent_type.value] = tracker.get_current_metrics()
        
        # Add ensemble metrics for comparison
        comparison['ensemble'] = self.get_ensemble_metrics()
        
        return comparison
    
    def get_regime_analysis(self) -> Dict[str, Dict[str, float]]:
        """Get performance analysis by market regime"""
        
        if not self.config.enable_regime_analysis or len(self.regime_history) == 0:
            return {}
        
        # Group returns by regime
        regime_data = defaultdict(list)
        
        for i, regime_record in enumerate(self.regime_history):
            regime = regime_record['regime']
            return_value = regime_record['return']
            regime_data[regime.value].append(return_value)
        
        # Calculate metrics for each regime
        regime_analysis = {}
        
        for regime_name, returns_list in regime_data.items():
            if len(returns_list) == 0:
                continue
            
            returns_array = np.array(returns_list)
            
            regime_analysis[regime_name] = {
                'count': len(returns_list),
                'total_return': float(np.sum(returns_array)),
                'mean_return': float(np.mean(returns_array)),
                'volatility': float(np.std(returns_array)),
                'sharpe_ratio': PerformanceCalculator.calculate_sharpe_ratio(returns_array, self.config.risk_free_rate),
                'win_rate': PerformanceCalculator.calculate_win_rate(returns_array),
                'profit_factor': PerformanceCalculator.calculate_profit_factor(returns_array)
            }
        
        return regime_analysis
    
    def get_attribution_analysis(self) -> Dict[str, Any]:
        """Get performance attribution analysis"""
        
        if not self.config.enable_attribution_analysis or len(self.attribution_data) == 0:
            return {}
        
        # Calculate average contribution by agent
        agent_contributions = defaultdict(list)
        total_contributions = []
        residuals = []
        
        for record in self.attribution_data:
            for agent_type, contribution in record['contributions'].items():
                agent_contributions[agent_type.value].append(contribution)
            
            total_contributions.append(sum(record['contributions'].values()))
            residuals.append(record['residual'])
        
        # Calculate attribution metrics
        attribution_summary = {}
        
        for agent_name, contributions in agent_contributions.items():
            if len(contributions) > 0:
                attribution_summary[agent_name] = {
                    'mean_contribution': float(np.mean(contributions)),
                    'total_contribution': float(np.sum(contributions)),
                    'contribution_volatility': float(np.std(contributions)),
                    'contribution_count': len(contributions)
                }
        
        # Overall attribution statistics
        attribution_summary['summary'] = {
            'total_attributed_return': float(np.sum(total_contributions)),
            'total_residual': float(np.sum(residuals)),
            'mean_residual': float(np.mean(residuals)) if residuals else 0.0,
            'residual_volatility': float(np.std(residuals)) if len(residuals) > 1 else 0.0,
            'attribution_quality': 1 - abs(np.mean(residuals)) if residuals else 1.0  # Quality metric
        }
        
        return attribution_summary
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'tracking_period': {
                'start_time': self.ensemble_snapshots[0].timestamp.isoformat() if self.ensemble_snapshots else None,
                'end_time': self.last_snapshot_time.isoformat(),
                'total_steps': self.step_count
            },
            'ensemble_performance': self.get_ensemble_metrics(),
            'agent_comparison': self.get_agent_comparison(),
            'regime_analysis': self.get_regime_analysis(),
            'attribution_analysis': self.get_attribution_analysis()
        }
        
        # Performance ranking
        agent_metrics = self.get_agent_comparison()
        if agent_metrics:
            # Rank by Sharpe ratio
            rankings = sorted(
                [(name, metrics.get('sharpe_ratio', 0)) for name, metrics in agent_metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )
            report['performance_rankings'] = {
                'by_sharpe_ratio': rankings,
                'best_performer': rankings[0][0] if rankings else None,
                'worst_performer': rankings[-1][0] if rankings else None
            }
        
        # Risk assessment
        ensemble_metrics = self.get_ensemble_metrics()
        if ensemble_metrics:
            report['risk_assessment'] = {
                'risk_level': self._assess_risk_level(ensemble_metrics),
                'risk_warnings': self._generate_risk_warnings(ensemble_metrics),
                'risk_metrics': {
                    'max_drawdown': ensemble_metrics.get('max_drawdown', 0),
                    'volatility': ensemble_metrics.get('volatility', 0),
                    'var_95': ensemble_metrics.get('var_95', 0),
                    'ulcer_index': ensemble_metrics.get('ulcer_index', 0)
                }
            }
        
        return report
    
    def _assess_risk_level(self, metrics: Dict[str, float]) -> str:
        """Assess overall risk level based on metrics"""
        
        max_dd = abs(metrics.get('max_drawdown', 0))
        volatility = metrics.get('volatility', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        
        # Risk scoring
        risk_score = 0
        
        if max_dd > 0.20:  # >20% drawdown
            risk_score += 3
        elif max_dd > 0.10:  # >10% drawdown
            risk_score += 2
        elif max_dd > 0.05:  # >5% drawdown
            risk_score += 1
        
        if volatility > 0.30:  # >30% annual volatility
            risk_score += 2
        elif volatility > 0.20:  # >20% annual volatility
            risk_score += 1
        
        if sharpe < 0.5:  # Poor risk-adjusted returns
            risk_score += 1
        
        # Classify risk level
        if risk_score >= 5:
            return "HIGH"
        elif risk_score >= 3:
            return "MEDIUM"
        elif risk_score >= 1:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _generate_risk_warnings(self, metrics: Dict[str, float]) -> List[str]:
        """Generate risk warnings based on metrics"""
        
        warnings = []
        
        max_dd = abs(metrics.get('max_drawdown', 0))
        volatility = metrics.get('volatility', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        win_rate = metrics.get('win_rate', 0)
        
        if max_dd > 0.20:
            warnings.append(f"High maximum drawdown: {max_dd:.1%}")
        
        if volatility > 0.30:
            warnings.append(f"High volatility: {volatility:.1%} annualized")
        
        if sharpe < 0.5:
            warnings.append(f"Poor risk-adjusted returns: Sharpe ratio {sharpe:.2f}")
        
        if win_rate < 0.40:
            warnings.append(f"Low win rate: {win_rate:.1%}")
        
        return warnings
    
    def save_state(self, filepath: str):
        """Save performance tracking state"""
        
        # Prepare agent data
        agent_data = {}
        for agent_type, tracker in self.agent_trackers.items():
            agent_data[agent_type.value] = {
                'current_equity': tracker.current_equity,
                'returns': list(tracker.returns)[-100:],  # Save last 100
                'metrics': tracker.get_current_metrics(),
                'regime_performance': tracker.get_regime_performance()
            }
        
        # Prepare ensemble data
        state = {
            'config': {
                'short_window': self.config.short_window,
                'medium_window': self.config.medium_window,
                'long_window': self.config.long_window,
                'risk_free_rate': self.config.risk_free_rate
            },
            'ensemble_state': {
                'current_equity': self.current_ensemble_equity,
                'step_count': self.step_count,
                'returns': list(self.ensemble_returns)[-100:],  # Save last 100
                'metrics': self.get_ensemble_metrics()
            },
            'agent_data': agent_data,
            'performance_report': self.generate_performance_report(),
            'snapshots': [s.to_dict() for s in list(self.ensemble_snapshots)[-50:]]  # Save last 50 snapshots
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Performance tracker state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load performance tracking state"""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore ensemble state
            ensemble_state = state.get('ensemble_state', {})
            self.current_ensemble_equity = ensemble_state.get('current_equity', 1.0)
            self.step_count = ensemble_state.get('step_count', 0)
            
            # Restore returns
            returns_data = ensemble_state.get('returns', [])
            self.ensemble_returns.clear()
            self.ensemble_returns.extend(returns_data)
            
            # Restore agent data
            agent_data = state.get('agent_data', {})
            for agent_str, data in agent_data.items():
                try:
                    agent_type = AgentType(agent_str)
                    if agent_type not in self.agent_trackers:
                        self.initialize_agent_tracker(agent_type)
                    
                    tracker = self.agent_trackers[agent_type]
                    tracker.current_equity = data.get('current_equity', 1.0)
                    
                    # Restore returns
                    tracker.returns.clear()
                    tracker.returns.extend(data.get('returns', []))
                    
                except ValueError:
                    continue
            
            logger.info(f"Performance tracker state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading performance tracker state: {e}")


if __name__ == "__main__":
    # Example usage and testing
    
    from datetime import datetime
    import time
    
    # Create performance tracker
    config = PerformanceConfig(
        enable_detailed_logging=True,
        log_frequency=10,
        enable_regime_analysis=True,
        enable_attribution_analysis=True
    )
    
    tracker = EnsemblePerformanceTracker(config)
    
    # Initialize agent trackers
    agent_types = [AgentType.CONSERVATIVE, AgentType.AGGRESSIVE, AgentType.BALANCED]
    for agent_type in agent_types:
        tracker.initialize_agent_tracker(agent_type)
    
    print(f"Initialized ensemble tracker with {len(agent_types)} agents")
    
    # Simulate some trading performance
    np.random.seed(42)
    regimes = list(MarketRegime)
    
    for step in range(100):
        # Simulate ensemble return
        ensemble_return = np.random.normal(0.001, 0.02)
        
        # Simulate individual agent returns
        individual_returns = {}
        for agent_type in agent_types:
            # Different performance in different regimes
            if agent_type == AgentType.AGGRESSIVE:
                individual_returns[agent_type] = np.random.normal(0.002, 0.03)
            elif agent_type == AgentType.CONSERVATIVE:
                individual_returns[agent_type] = np.random.normal(0.0005, 0.01)
            else:
                individual_returns[agent_type] = np.random.normal(0.001, 0.02)
        
        # Create mock ensemble decision
        from ..meta.meta_agent_orchestrator import EnsembleDecision, DecisionStrategy
        from ..regime.market_regime_detector import MarketRegime
        
        decision = EnsembleDecision(
            final_action=np.array([ensemble_return]),
            agent_decisions=[],
            agent_weights={agent.value: 1.0/len(agent_types) for agent in agent_types},
            regime=np.random.choice(regimes),
            regime_confidence=np.random.random(),
            strategy_used=DecisionStrategy.WEIGHTED_ENSEMBLE,
            risk_level=None,
            explanation={'active_agents': [a.value for a in agent_types]},
            timestamp=datetime.now()
        )
        
        # Update performance
        tracker.update_ensemble_performance(ensemble_return, decision, individual_returns)
        
        if step % 20 == 0:
            print(f"Step {step}: Ensemble return = {ensemble_return:.3f}")
    
    # Generate final report
    report = tracker.generate_performance_report()
    
    print(f"\nFinal Performance Report:")
    print(f"Total Return: {report['ensemble_performance']['total_return']:.2%}")
    print(f"Sharpe Ratio: {report['ensemble_performance']['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {report['ensemble_performance']['max_drawdown']:.2%}")
    print(f"Win Rate: {report['ensemble_performance']['win_rate']:.2%}")
    
    print(f"\nAgent Rankings:")
    for rank, (agent, sharpe) in enumerate(report['performance_rankings']['by_sharpe_ratio']):
        print(f"{rank+1}. {agent}: Sharpe = {sharpe:.3f}")
    
    print(f"\nRisk Assessment: {report['risk_assessment']['risk_level']}")
    if report['risk_assessment']['risk_warnings']:
        print("Risk Warnings:")
        for warning in report['risk_assessment']['risk_warnings']:
            print(f"  - {warning}")
    
    # Test saving and loading
    tracker.save_state("/tmp/ensemble_performance_test.json")
    print("\nPerformance state saved successfully")