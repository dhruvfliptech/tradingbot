"""
Central Risk Management System - Production Grade
Enforces SOW constraints and institutional-grade risk controls
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
import asyncio
from collections import deque
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"
    CRITICAL = "CRITICAL"


class RiskAction(Enum):
    """Risk management actions"""
    ALLOW = "ALLOW"
    REDUCE = "REDUCE"
    REJECT = "REJECT"
    CLOSE_ALL = "CLOSE_ALL"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class RiskLimits:
    """SOW-compliant risk limits"""
    max_drawdown: float = 0.15  # 15% max drawdown
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    max_position_size: float = 0.10  # 10% max per position
    max_correlation_exposure: float = 0.60  # 60% max correlated exposure
    max_leverage: float = 5.0  # Maximum 5x leverage
    min_sharpe_ratio: float = 1.5  # Minimum Sharpe ratio
    var_confidence: float = 0.95  # 95% VaR confidence
    max_var_percent: float = 0.03  # 3% max VaR
    min_liquidity_ratio: float = 0.20  # 20% min liquid assets
    max_concentration: float = 0.30  # 30% max sector concentration
    weekly_return_target: float = 0.03  # 3% minimum weekly return
    max_weekly_return: float = 0.05  # 5% maximum weekly return


@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    var_1d: float = 0.0
    cvar_1d: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    leverage: float = 1.0
    correlation_exposure: float = 0.0
    liquidity_ratio: float = 1.0
    concentration_risk: float = 0.0
    stress_test_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Position information"""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    correlation_group: str
    sector: str
    entry_time: datetime
    risk_amount: float  # 1R risk
    current_r_multiple: float = 0.0


class RiskManager:
    """
    Central Risk Management System
    Enforces institutional-grade risk controls and SOW compliance
    """
    
    def __init__(self, 
                 account_balance: float,
                 risk_limits: Optional[RiskLimits] = None,
                 enable_circuit_breakers: bool = True):
        """
        Initialize Risk Manager
        
        Args:
            account_balance: Current account balance
            risk_limits: Risk limit configuration
            enable_circuit_breakers: Enable automatic circuit breakers
        """
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.risk_limits = risk_limits or RiskLimits()
        self.enable_circuit_breakers = enable_circuit_breakers
        
        # Risk tracking
        self.positions: Dict[str, Position] = {}
        self.risk_metrics = RiskMetrics()
        self.pnl_history = deque(maxlen=252)  # 1 year of daily PnL
        self.drawdown_history = deque(maxlen=252)
        self.risk_events = []
        self.circuit_breaker_triggered = False
        self.emergency_mode = False
        
        # Performance tracking
        self.trade_history = []
        self.r_multiples = []
        self.expectancy = 0.0
        self.win_rate = 0.0
        
        # Correlation matrix
        self.correlation_matrix = pd.DataFrame()
        self.correlation_groups = {}
        
        # Risk models
        self.var_model = None
        self.stress_scenarios = {}
        
        logger.info(f"Risk Manager initialized with balance: ${account_balance:,.2f}")
    
    def evaluate_trade(self, 
                       symbol: str,
                       position_size: float,
                       entry_price: float,
                       stop_loss: float,
                       take_profit: float,
                       correlation_group: str = "default",
                       sector: str = "general") -> Tuple[RiskAction, Dict[str, Any]]:
        """
        Evaluate proposed trade against risk limits
        
        Returns:
            Tuple of (RiskAction, risk_assessment_dict)
        """
        risk_assessment = {
            "symbol": symbol,
            "proposed_size": position_size,
            "risk_checks": {},
            "adjustments": {},
            "warnings": [],
            "action": RiskAction.ALLOW
        }
        
        # Check if circuit breaker is active
        if self.circuit_breaker_triggered:
            risk_assessment["action"] = RiskAction.REJECT
            risk_assessment["warnings"].append("Circuit breaker active - trading suspended")
            return RiskAction.REJECT, risk_assessment
        
        # Calculate position risk
        position_value = position_size * entry_price
        risk_amount = abs(position_size * (entry_price - stop_loss))
        risk_percent = risk_amount / self.account_balance
        
        # 1. Position size check
        position_percent = position_value / self.account_balance
        risk_assessment["risk_checks"]["position_size"] = {
            "value": position_percent,
            "limit": self.risk_limits.max_position_size,
            "passed": position_percent <= self.risk_limits.max_position_size
        }
        
        if position_percent > self.risk_limits.max_position_size:
            # Adjust position size
            adjusted_size = (self.risk_limits.max_position_size * self.account_balance) / entry_price
            risk_assessment["adjustments"]["position_size"] = adjusted_size
            risk_assessment["warnings"].append(f"Position size reduced from {position_size} to {adjusted_size}")
            position_size = adjusted_size
            risk_assessment["action"] = RiskAction.REDUCE
        
        # 2. Correlation exposure check
        correlation_exposure = self._calculate_correlation_exposure(correlation_group, position_value)
        risk_assessment["risk_checks"]["correlation_exposure"] = {
            "value": correlation_exposure,
            "limit": self.risk_limits.max_correlation_exposure,
            "passed": correlation_exposure <= self.risk_limits.max_correlation_exposure
        }
        
        if correlation_exposure > self.risk_limits.max_correlation_exposure:
            risk_assessment["warnings"].append(f"High correlation exposure: {correlation_exposure:.2%}")
            if correlation_exposure > self.risk_limits.max_correlation_exposure * 1.2:
                risk_assessment["action"] = RiskAction.REJECT
                return RiskAction.REJECT, risk_assessment
        
        # 3. Drawdown check
        if self.risk_metrics.current_drawdown > self.risk_limits.max_drawdown * 0.8:
            risk_assessment["warnings"].append(f"Approaching max drawdown: {self.risk_metrics.current_drawdown:.2%}")
            risk_assessment["action"] = RiskAction.REDUCE
            position_size *= 0.5  # Reduce position size by 50%
            risk_assessment["adjustments"]["drawdown_reduction"] = position_size
        
        # 4. Daily loss limit check
        potential_daily_loss = self.risk_metrics.daily_pnl - risk_amount
        daily_loss_percent = abs(potential_daily_loss / self.account_balance)
        
        risk_assessment["risk_checks"]["daily_loss_limit"] = {
            "value": daily_loss_percent,
            "limit": self.risk_limits.daily_loss_limit,
            "passed": daily_loss_percent <= self.risk_limits.daily_loss_limit
        }
        
        if daily_loss_percent > self.risk_limits.daily_loss_limit:
            risk_assessment["action"] = RiskAction.REJECT
            risk_assessment["warnings"].append("Daily loss limit would be exceeded")
            return RiskAction.REJECT, risk_assessment
        
        # 5. VaR check
        portfolio_var = self._calculate_portfolio_var(symbol, position_value)
        var_percent = portfolio_var / self.account_balance
        
        risk_assessment["risk_checks"]["var_limit"] = {
            "value": var_percent,
            "limit": self.risk_limits.max_var_percent,
            "passed": var_percent <= self.risk_limits.max_var_percent
        }
        
        if var_percent > self.risk_limits.max_var_percent:
            risk_assessment["warnings"].append(f"High VaR: {var_percent:.2%}")
            risk_assessment["action"] = RiskAction.REDUCE
        
        # 6. Leverage check
        total_exposure = sum(pos.size * pos.current_price for pos in self.positions.values())
        total_exposure += position_value
        leverage = total_exposure / self.account_balance
        
        risk_assessment["risk_checks"]["leverage"] = {
            "value": leverage,
            "limit": self.risk_limits.max_leverage,
            "passed": leverage <= self.risk_limits.max_leverage
        }
        
        if leverage > self.risk_limits.max_leverage:
            risk_assessment["action"] = RiskAction.REJECT
            risk_assessment["warnings"].append(f"Leverage limit exceeded: {leverage:.2f}x")
            return RiskAction.REJECT, risk_assessment
        
        # 7. Concentration risk check
        concentration = self._calculate_concentration_risk(sector, position_value)
        risk_assessment["risk_checks"]["concentration"] = {
            "value": concentration,
            "limit": self.risk_limits.max_concentration,
            "passed": concentration <= self.risk_limits.max_concentration
        }
        
        # Calculate final risk score
        risk_score = self._calculate_risk_score(risk_assessment["risk_checks"])
        risk_assessment["risk_score"] = risk_score
        risk_assessment["risk_level"] = self._get_risk_level(risk_score)
        
        # Final adjustments if needed
        if risk_assessment["action"] == RiskAction.REDUCE:
            risk_assessment["adjustments"]["final_size"] = position_size
            risk_assessment["adjustments"]["final_risk"] = position_size * abs(entry_price - stop_loss)
        
        return risk_assessment["action"], risk_assessment
    
    def add_position(self, position: Position) -> bool:
        """
        Add new position to portfolio
        
        Returns:
            True if position added successfully
        """
        if position.symbol in self.positions:
            logger.warning(f"Position {position.symbol} already exists")
            return False
        
        # Calculate R-multiple
        position.risk_amount = position.size * abs(position.entry_price - position.stop_loss)
        position.current_r_multiple = 0.0
        
        self.positions[position.symbol] = position
        self._update_risk_metrics()
        
        logger.info(f"Position added: {position.symbol} - Size: {position.size} @ ${position.entry_price}")
        return True
    
    def update_position(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """
        Update position with current price and calculate metrics
        
        Returns:
            Position update information
        """
        if symbol not in self.positions:
            return {"error": f"Position {symbol} not found"}
        
        position = self.positions[symbol]
        position.current_price = current_price
        
        # Calculate current R-multiple
        pnl = position.size * (current_price - position.entry_price)
        position.current_r_multiple = pnl / position.risk_amount if position.risk_amount > 0 else 0
        
        # Check stop loss
        if current_price <= position.stop_loss:
            return self.close_position(symbol, current_price, "STOP_LOSS")
        
        # Check take profit
        if current_price >= position.take_profit:
            return self.close_position(symbol, current_price, "TAKE_PROFIT")
        
        # Trail stop loss if profitable
        if position.current_r_multiple > 2.0:
            new_stop = position.entry_price + (current_price - position.entry_price) * 0.5
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                logger.info(f"Trailing stop updated for {symbol}: ${new_stop:.2f}")
        
        self._update_risk_metrics()
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "r_multiple": position.current_r_multiple,
            "pnl": pnl,
            "status": "ACTIVE"
        }
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "MANUAL") -> Dict[str, Any]:
        """
        Close position and record results
        
        Returns:
            Trade result information
        """
        if symbol not in self.positions:
            return {"error": f"Position {symbol} not found"}
        
        position = self.positions[symbol]
        
        # Calculate final metrics
        pnl = position.size * (exit_price - position.entry_price)
        r_multiple = pnl / position.risk_amount if position.risk_amount > 0 else 0
        hold_time = datetime.now() - position.entry_time
        
        # Record trade
        trade_result = {
            "symbol": symbol,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size": position.size,
            "pnl": pnl,
            "r_multiple": r_multiple,
            "hold_time": hold_time.total_seconds() / 3600,  # hours
            "reason": reason,
            "timestamp": datetime.now()
        }
        
        self.trade_history.append(trade_result)
        self.r_multiples.append(r_multiple)
        
        # Update account balance
        self.account_balance += pnl
        
        # Remove position
        del self.positions[symbol]
        
        # Update metrics
        self._update_risk_metrics()
        self._update_expectancy()
        
        logger.info(f"Position closed: {symbol} - PnL: ${pnl:.2f} ({r_multiple:.2f}R) - Reason: {reason}")
        
        return trade_result
    
    def emergency_stop(self, reason: str = "MANUAL") -> List[Dict[str, Any]]:
        """
        Emergency close all positions
        
        Returns:
            List of closed position results
        """
        logger.warning(f"EMERGENCY STOP TRIGGERED: {reason}")
        self.emergency_mode = True
        
        results = []
        symbols = list(self.positions.keys())
        
        for symbol in symbols:
            position = self.positions[symbol]
            result = self.close_position(symbol, position.current_price, f"EMERGENCY_{reason}")
            results.append(result)
        
        self.circuit_breaker_triggered = True
        
        # Log risk event
        self.risk_events.append({
            "type": "EMERGENCY_STOP",
            "reason": reason,
            "timestamp": datetime.now(),
            "positions_closed": len(results),
            "total_pnl": sum(r.get("pnl", 0) for r in results)
        })
        
        return results
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive risk report
        
        Returns:
            Detailed risk report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "account": {
                "balance": self.account_balance,
                "initial_balance": self.initial_balance,
                "total_pnl": self.account_balance - self.initial_balance,
                "return_pct": (self.account_balance / self.initial_balance - 1) * 100
            },
            "risk_metrics": {
                "current_drawdown": self.risk_metrics.current_drawdown,
                "max_drawdown": max(self.drawdown_history) if self.drawdown_history else 0,
                "daily_pnl": self.risk_metrics.daily_pnl,
                "var_1d": self.risk_metrics.var_1d,
                "cvar_1d": self.risk_metrics.cvar_1d,
                "sharpe_ratio": self.risk_metrics.sharpe_ratio,
                "sortino_ratio": self.risk_metrics.sortino_ratio,
                "calmar_ratio": self.risk_metrics.calmar_ratio,
                "leverage": self.risk_metrics.leverage,
                "risk_level": self.risk_metrics.risk_level.value
            },
            "positions": {
                "count": len(self.positions),
                "total_exposure": sum(p.size * p.current_price for p in self.positions.values()),
                "avg_r_multiple": np.mean([p.current_r_multiple for p in self.positions.values()]) if self.positions else 0,
                "positions": [
                    {
                        "symbol": p.symbol,
                        "size": p.size,
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "r_multiple": p.current_r_multiple,
                        "pnl": p.size * (p.current_price - p.entry_price)
                    }
                    for p in self.positions.values()
                ]
            },
            "performance": {
                "total_trades": len(self.trade_history),
                "win_rate": self.win_rate,
                "expectancy": self.expectancy,
                "avg_r_multiple": np.mean(self.r_multiples) if self.r_multiples else 0,
                "best_trade_r": max(self.r_multiples) if self.r_multiples else 0,
                "worst_trade_r": min(self.r_multiples) if self.r_multiples else 0
            },
            "risk_limits": {
                "max_drawdown": self.risk_limits.max_drawdown,
                "daily_loss_limit": self.risk_limits.daily_loss_limit,
                "max_leverage": self.risk_limits.max_leverage,
                "compliance": self._check_compliance()
            },
            "warnings": self._generate_warnings(),
            "circuit_breaker_status": {
                "triggered": self.circuit_breaker_triggered,
                "emergency_mode": self.emergency_mode
            }
        }
        
        return report
    
    def _calculate_correlation_exposure(self, correlation_group: str, position_value: float) -> float:
        """Calculate total exposure to correlated assets"""
        total_correlated = 0
        for pos in self.positions.values():
            if pos.correlation_group == correlation_group:
                total_correlated += pos.size * pos.current_price
        
        total_correlated += position_value
        return total_correlated / self.account_balance
    
    def _calculate_portfolio_var(self, symbol: str, position_value: float) -> float:
        """Calculate portfolio Value at Risk"""
        # Simplified VaR calculation - should be enhanced with historical data
        portfolio_value = sum(pos.size * pos.current_price for pos in self.positions.values())
        portfolio_value += position_value
        
        # Assume 2% daily volatility for simplification
        daily_volatility = 0.02
        var_1d = portfolio_value * daily_volatility * 1.645  # 95% confidence
        
        return var_1d
    
    def _calculate_concentration_risk(self, sector: str, position_value: float) -> float:
        """Calculate sector concentration risk"""
        sector_exposure = 0
        for pos in self.positions.values():
            if pos.sector == sector:
                sector_exposure += pos.size * pos.current_price
        
        sector_exposure += position_value
        total_exposure = sum(pos.size * pos.current_price for pos in self.positions.values()) + position_value
        
        return sector_exposure / total_exposure if total_exposure > 0 else 0
    
    def _calculate_risk_score(self, risk_checks: Dict[str, Dict]) -> float:
        """Calculate overall risk score from individual checks"""
        scores = []
        weights = {
            "position_size": 0.15,
            "correlation_exposure": 0.20,
            "daily_loss_limit": 0.25,
            "var_limit": 0.20,
            "leverage": 0.15,
            "concentration": 0.05
        }
        
        for check_name, check_data in risk_checks.items():
            if check_name in weights:
                ratio = check_data["value"] / check_data["limit"] if check_data["limit"] > 0 else 0
                score = min(ratio, 1.0) * weights.get(check_name, 0.1)
                scores.append(score)
        
        return sum(scores)
    
    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MEDIUM
        elif risk_score < 0.7:
            return RiskLevel.HIGH
        elif risk_score < 0.9:
            return RiskLevel.EXTREME
        else:
            return RiskLevel.CRITICAL
    
    def _update_risk_metrics(self):
        """Update all risk metrics"""
        # Calculate current metrics
        total_pnl = sum(p.size * (p.current_price - p.entry_price) for p in self.positions.values())
        
        # Update drawdown
        peak_balance = max(self.initial_balance, self.account_balance + total_pnl)
        current_balance = self.account_balance + total_pnl
        self.risk_metrics.current_drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0
        
        # Update leverage
        total_exposure = sum(pos.size * pos.current_price for pos in self.positions.values())
        self.risk_metrics.leverage = total_exposure / self.account_balance if self.account_balance > 0 else 0
        
        # Update ratios (simplified - should use historical data)
        if len(self.pnl_history) > 30:
            returns = pd.Series(list(self.pnl_history))
            self.risk_metrics.sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                self.risk_metrics.sortino_ratio = (returns.mean() / downside_returns.std() * np.sqrt(252))
        
        # Update risk level
        risk_score = self._calculate_risk_score({
            "leverage": {"value": self.risk_metrics.leverage, "limit": self.risk_limits.max_leverage},
            "drawdown": {"value": self.risk_metrics.current_drawdown, "limit": self.risk_limits.max_drawdown}
        })
        self.risk_metrics.risk_level = self._get_risk_level(risk_score)
    
    def _update_expectancy(self):
        """Update trading expectancy"""
        if not self.r_multiples:
            return
        
        winning_trades = [r for r in self.r_multiples if r > 0]
        losing_trades = [r for r in self.r_multiples if r <= 0]
        
        if winning_trades:
            self.win_rate = len(winning_trades) / len(self.r_multiples)
            avg_win = np.mean(winning_trades)
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            
            self.expectancy = (self.win_rate * avg_win) - ((1 - self.win_rate) * abs(avg_loss))
    
    def _check_compliance(self) -> Dict[str, bool]:
        """Check compliance with all risk limits"""
        return {
            "drawdown": self.risk_metrics.current_drawdown <= self.risk_limits.max_drawdown,
            "leverage": self.risk_metrics.leverage <= self.risk_limits.max_leverage,
            "sharpe_ratio": self.risk_metrics.sharpe_ratio >= self.risk_limits.min_sharpe_ratio,
            "var": self.risk_metrics.var_1d / self.account_balance <= self.risk_limits.max_var_percent
        }
    
    def _generate_warnings(self) -> List[str]:
        """Generate risk warnings"""
        warnings = []
        
        if self.risk_metrics.current_drawdown > self.risk_limits.max_drawdown * 0.8:
            warnings.append(f"Approaching max drawdown: {self.risk_metrics.current_drawdown:.2%}")
        
        if self.risk_metrics.leverage > self.risk_limits.max_leverage * 0.9:
            warnings.append(f"High leverage: {self.risk_metrics.leverage:.2f}x")
        
        if self.risk_metrics.sharpe_ratio < self.risk_limits.min_sharpe_ratio:
            warnings.append(f"Low Sharpe ratio: {self.risk_metrics.sharpe_ratio:.2f}")
        
        if self.expectancy < 0:
            warnings.append(f"Negative expectancy: {self.expectancy:.2f}R")
        
        return warnings