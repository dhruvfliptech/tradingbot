"""
AdaptiveThreshold ML Service
Implements dynamic threshold adjustment based on trading performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine, text
import os
from config import get_config
from monitoring import logger as monitoring_logger
from performance_tracker import performance_tracker

# Use both standard logging and enhanced monitoring
logger = logging.getLogger(__name__)
enhanced_logger = monitoring_logger

@dataclass
class PerformanceMetrics:
    total_return: float
    sharpe_ratio: float
    win_rate: float
    avg_trade_return: float
    max_drawdown: float
    volatility: float

@dataclass
class ThresholdUpdate:
    parameter_name: str
    old_value: float
    new_value: float
    reason: str
    confidence: float

class AdaptiveThreshold:
    """
    Adaptive threshold system that adjusts trading parameters based on performance feedback.
    Pre-RL implementation focusing on simple gradient-based optimization.
    """
    
    def __init__(self, user_id: str, symbol: Optional[str] = None):
        self.user_id = user_id
        self.symbol = symbol
        
        # Get database URL from configuration
        config = get_config()
        self.db_engine = create_engine(config.database.url)
        
        # Default parameters from configuration
        self.parameters = config.adaptation.default_parameters.copy()
        
        # Adaptation settings from configuration
        config = get_config()
        self.learning_rate = config.adaptation.default_learning_rate
        self.performance_window = config.adaptation.performance_window
        self.min_trades_for_adaptation = config.adaptation.min_trades_for_adaptation
        # Load adaptation bounds from configuration
        self.adaptation_bounds = {
            'rsi_threshold': config.adaptation.rsi_bounds,
            'confidence_threshold': config.adaptation.confidence_bounds,
            'macd_threshold': config.adaptation.macd_bounds,
            'volume_threshold': config.adaptation.volume_bounds,
            'momentum_threshold': config.adaptation.momentum_bounds
        }
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.last_adaptation = datetime.now()
        
        # Load existing thresholds from database
        self._load_thresholds()
    
    def _load_thresholds(self):
        """Load existing thresholds from database"""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT parameter_name, current_value 
                    FROM adaptive_thresholds 
                    WHERE user_id = :user_id 
                    AND (symbol = :symbol OR symbol IS NULL)
                    ORDER BY updated_at DESC
                """)
                
                result = conn.execute(query, {
                    'user_id': self.user_id,
                    'symbol': self.symbol
                }).fetchall()
                
                for row in result:
                    if row.parameter_name in self.parameters:
                        self.parameters[row.parameter_name] = float(row.current_value)
                        
        except Exception as e:
            logger.error(f"Error loading thresholds: {e}")
    
    def _save_threshold(self, parameter_name: str, new_value: float, performance: float):
        """Save updated threshold to database"""
        try:
            with self.db_engine.connect() as conn:
                # Upsert threshold
                query = text("""
                    INSERT INTO adaptive_thresholds 
                    (user_id, symbol, parameter_name, current_value, initial_value, 
                     last_performance, adaptation_count, updated_at)
                    VALUES (:user_id, :symbol, :parameter_name, :current_value, 
                            :initial_value, :last_performance, 1, NOW())
                    ON CONFLICT (user_id, symbol, parameter_name) 
                    DO UPDATE SET
                        current_value = :current_value,
                        last_performance = :last_performance,
                        adaptation_count = adaptive_thresholds.adaptation_count + 1,
                        updated_at = NOW()
                """)
                
                conn.execute(query, {
                    'user_id': self.user_id,
                    'symbol': self.symbol,
                    'parameter_name': parameter_name,
                    'current_value': new_value,
                    'initial_value': self.parameters.get(parameter_name, new_value),
                    'last_performance': performance
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving threshold: {e}")
    
    def get_performance_metrics(self, days_back: int = 30) -> PerformanceMetrics:
        """Calculate recent performance metrics from database"""
        try:
            with self.db_engine.connect() as conn:
                # Get recent trades
                query = text("""
                    SELECT pnl, pnl_percent, hold_duration, closed_at
                    FROM trades
                    WHERE user_id = :user_id 
                    AND closed_at >= NOW() - INTERVAL :days DAYS
                    AND (:symbol IS NULL OR symbol = :symbol)
                    ORDER BY closed_at DESC
                    LIMIT :limit
                """)
                
                result = conn.execute(query, {
                    'user_id': self.user_id,
                    'symbol': self.symbol,
                    'days': days_back,
                    'limit': self.performance_window
                }).fetchall()
                
                if len(result) < self.min_trades_for_adaptation:
                    return PerformanceMetrics(0, 0, 0, 0, 0, 0)
                
                # Calculate metrics
                returns = [float(row.pnl_percent) for row in result]
                total_return = sum(returns)
                avg_return = np.mean(returns)
                volatility = np.std(returns) if len(returns) > 1 else 0
                
                # Sharpe ratio (simplified)
                risk_free_rate = 0.02 / 365  # 2% annual risk-free rate
                sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
                
                # Win rate
                winning_trades = sum(1 for r in returns if r > 0)
                win_rate = winning_trades / len(returns) if returns else 0
                
                # Max drawdown calculation
                cumulative_returns = np.cumsum(returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = cumulative_returns - running_max
                max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
                
                return PerformanceMetrics(
                    total_return=total_return,
                    sharpe_ratio=sharpe_ratio,
                    win_rate=win_rate,
                    avg_trade_return=avg_return,
                    max_drawdown=abs(max_drawdown),
                    volatility=volatility
                )
                
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0)
    
    def calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate composite performance score (0-1)"""
        # Weighted performance score
        weights = {
            'total_return': 0.3,
            'sharpe_ratio': 0.25,
            'win_rate': 0.2,
            'max_drawdown': -0.15,  # Negative weight (lower is better)
            'volatility': -0.1      # Negative weight (lower is better)
        }
        
        # Normalize metrics to 0-1 scale
        normalized_return = np.tanh(metrics.total_return / 10)  # Scale total return
        normalized_sharpe = np.tanh(metrics.sharpe_ratio)       # Sharpe already scaled
        normalized_win_rate = metrics.win_rate                  # Already 0-1
        normalized_drawdown = np.tanh(metrics.max_drawdown / 5) # Scale drawdown
        normalized_volatility = np.tanh(metrics.volatility / 3)  # Scale volatility
        
        score = (
            weights['total_return'] * normalized_return +
            weights['sharpe_ratio'] * normalized_sharpe +
            weights['win_rate'] * normalized_win_rate +
            weights['max_drawdown'] * normalized_drawdown +
            weights['volatility'] * normalized_volatility
        )
        
        return max(0, min(1, score))  # Clamp to 0-1
    
    def adapt_thresholds(self) -> List[ThresholdUpdate]:
        """
        Main adaptation logic - adjust thresholds based on recent performance
        """
        updates = []
        
        # Get recent performance
        metrics = self.get_performance_metrics()
        performance_score = self.calculate_performance_score(metrics)
        
        # Store performance history
        self.performance_history.append(metrics)
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
        
        logger.info(f"Performance score: {performance_score:.3f}")
        
        # Only adapt if we have enough data and it's been long enough
        if len(self.performance_history) < 2:
            return updates
        
        # Calculate performance trend
        recent_scores = [self.calculate_performance_score(m) for m in self.performance_history[-3:]]
        if len(recent_scores) < 2:
            return updates
            
        performance_trend = np.mean(np.diff(recent_scores))
        
        # Adapt each parameter based on performance
        for param_name, current_value in self.parameters.items():
            update = self._adapt_parameter(
                param_name, current_value, performance_score, 
                performance_trend, metrics
            )
            if update:
                updates.append(update)
                self.parameters[param_name] = update.new_value
                self._save_threshold(param_name, update.new_value, performance_score)
        
        self.last_adaptation = datetime.now()
        return updates
    
    def _adapt_parameter(self, param_name: str, current_value: float, 
                        performance_score: float, performance_trend: float,
                        metrics: PerformanceMetrics) -> Optional[ThresholdUpdate]:
        """Adapt a specific parameter based on performance"""
        
        min_val, max_val = self.adaptation_bounds[param_name]
        
        # Base adaptation amount
        base_adaptation = self.learning_rate * (performance_score - 0.5) * 2
        
        # Parameter-specific adaptation logic
        if param_name == 'rsi_threshold':
            # Lower RSI threshold if performing well (more aggressive)
            # Raise RSI threshold if performing poorly (more conservative)
            adaptation = -base_adaptation * current_value * 0.1
            reason = f"RSI threshold adjustment based on performance trend: {performance_trend:.3f}"
            
        elif param_name == 'confidence_threshold':
            # Lower confidence threshold if win rate is high
            # Raise confidence threshold if win rate is low
            win_rate_factor = (metrics.win_rate - 0.5) * 2
            adaptation = -win_rate_factor * self.learning_rate * 0.1
            reason = f"Confidence threshold adjustment based on win rate: {metrics.win_rate:.3f}"
            
        elif param_name == 'momentum_threshold':
            # Adjust based on volatility - higher volatility needs higher threshold
            volatility_factor = np.tanh(metrics.volatility / 2)
            adaptation = volatility_factor * self.learning_rate * current_value * 0.1
            reason = f"Momentum threshold adjustment based on volatility: {metrics.volatility:.3f}"
            
        else:
            # Generic adaptation for other parameters
            adaptation = base_adaptation * current_value * 0.05
            reason = f"Generic adaptation based on performance score: {performance_score:.3f}"
        
        # Apply adaptation with bounds checking
        new_value = np.clip(current_value + adaptation, min_val, max_val)
        
        # Only update if change is significant (> 0.5% of current value)
        if abs(new_value - current_value) / current_value < 0.005:
            return None
        
        confidence = min(0.9, performance_score + 0.1)
        
        return ThresholdUpdate(
            parameter_name=param_name,
            old_value=current_value,
            new_value=new_value,
            reason=reason,
            confidence=confidence
        )
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current threshold values"""
        return self.parameters.copy()
    
    def reset_thresholds(self):
        """Reset all thresholds to default values"""
        default_params = {
            'rsi_threshold': 70.0,
            'confidence_threshold': 0.75,
            'macd_threshold': 0.0,
            'volume_threshold': 1000000000,
            'momentum_threshold': 2.0
        }
        
        for param_name, default_value in default_params.items():
            self.parameters[param_name] = default_value
            self._save_threshold(param_name, default_value, 0.5)
    
    def should_trade(self, signal_data: Dict) -> bool:
        """
        Evaluate if a signal should result in a trade based on current thresholds
        """
        try:
            # Check confidence threshold
            if signal_data.get('confidence', 0) < self.parameters['confidence_threshold']:
                return False
            
            # Check RSI threshold
            if signal_data.get('rsi', 50) > self.parameters['rsi_threshold'] and signal_data.get('action') == 'BUY':
                return False
            
            # Check momentum threshold
            momentum = abs(signal_data.get('change_percent', 0))
            if momentum < self.parameters['momentum_threshold']:
                return False
            
            # Check volume threshold
            if signal_data.get('volume', 0) < self.parameters['volume_threshold']:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating trade signal: {e}")
            return False


class AdaptiveThresholdManager:
    """Manager class to handle multiple users and symbols"""
    
    def __init__(self):
        self.instances: Dict[str, AdaptiveThreshold] = {}
    
    def get_instance(self, user_id: str, symbol: Optional[str] = None) -> AdaptiveThreshold:
        """Get or create AdaptiveThreshold instance for user/symbol"""
        key = f"{user_id}_{symbol or 'global'}"
        
        if key not in self.instances:
            self.instances[key] = AdaptiveThreshold(user_id, symbol)
        
        return self.instances[key]
    
    def adapt_all_users(self) -> Dict[str, List[ThresholdUpdate]]:
        """Run adaptation for all active instances"""
        results = {}
        
        for key, instance in self.instances.items():
            try:
                updates = instance.adapt_thresholds()
                results[key] = updates
                
                if updates:
                    logger.info(f"Adapted thresholds for {key}: {len(updates)} updates")
                    
            except Exception as e:
                logger.error(f"Error adapting thresholds for {key}: {e}")
                results[key] = []
        
        return results


# Global manager instance
threshold_manager = AdaptiveThresholdManager()