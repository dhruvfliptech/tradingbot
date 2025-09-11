"""
Circuit Breakers and Emergency Stop Mechanisms
Automatic risk controls to prevent catastrophic losses
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
from collections import deque
import json

logger = logging.getLogger(__name__)


class BreakerType(Enum):
    """Types of circuit breakers"""
    DRAWDOWN = "DRAWDOWN"
    DAILY_LOSS = "DAILY_LOSS"
    VOLATILITY = "VOLATILITY"
    CORRELATION = "CORRELATION"
    LIQUIDITY = "LIQUIDITY"
    TECHNICAL = "TECHNICAL"
    SYSTEM = "SYSTEM"
    MANUAL = "MANUAL"


class BreakerAction(Enum):
    """Circuit breaker actions"""
    WARNING = "WARNING"
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"
    STOP_NEW_TRADES = "STOP_NEW_TRADES"
    CLOSE_LOSING = "CLOSE_LOSING"
    CLOSE_ALL = "CLOSE_ALL"
    EMERGENCY_HEDGE = "EMERGENCY_HEDGE"
    SYSTEM_HALT = "SYSTEM_HALT"


class BreakerStatus(Enum):
    """Circuit breaker status"""
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    COOLDOWN = "COOLDOWN"
    DISABLED = "DISABLED"


@dataclass
class BreakerConfig:
    """Circuit breaker configuration"""
    # Drawdown breakers
    warning_drawdown: float = 0.08  # 8% drawdown warning
    stop_drawdown: float = 0.12  # 12% stop new trades
    emergency_drawdown: float = 0.15  # 15% emergency close all
    
    # Daily loss breakers
    daily_loss_warning: float = 0.03  # 3% daily loss warning
    daily_loss_stop: float = 0.05  # 5% stop trading
    
    # Volatility breakers
    volatility_spike_threshold: float = 3.0  # 3x normal volatility
    volatility_window: int = 20  # Rolling window for volatility
    
    # Correlation breakers
    correlation_threshold: float = 0.90  # 90% correlation trigger
    correlation_assets_min: int = 3  # Minimum assets for correlation check
    
    # Liquidity breakers
    liquidity_ratio_min: float = 0.20  # 20% minimum liquidity
    volume_drop_threshold: float = 0.50  # 50% volume drop
    
    # System breakers
    consecutive_losses: int = 5  # Consecutive losses trigger
    error_rate_threshold: float = 0.10  # 10% error rate
    latency_threshold: int = 5000  # 5 second latency
    
    # Cooldown periods (minutes)
    warning_cooldown: int = 30
    stop_cooldown: int = 60
    emergency_cooldown: int = 1440  # 24 hours
    
    # Recovery conditions
    recovery_profit_threshold: float = 0.02  # 2% profit to reset
    recovery_time_minimum: int = 60  # Minimum 60 minutes


@dataclass
class BreakerEvent:
    """Circuit breaker event record"""
    timestamp: datetime
    breaker_type: BreakerType
    action: BreakerAction
    trigger_value: float
    threshold: float
    message: str
    metadata: Dict = field(default_factory=dict)


class CircuitBreaker:
    """Individual circuit breaker implementation"""
    
    def __init__(self,
                 name: str,
                 breaker_type: BreakerType,
                 threshold: float,
                 action: BreakerAction,
                 cooldown_minutes: int = 30):
        """
        Initialize circuit breaker
        
        Args:
            name: Breaker name
            breaker_type: Type of breaker
            threshold: Trigger threshold
            action: Action when triggered
            cooldown_minutes: Cooldown period
        """
        self.name = name
        self.breaker_type = breaker_type
        self.threshold = threshold
        self.action = action
        self.cooldown_minutes = cooldown_minutes
        
        self.status = BreakerStatus.ACTIVE
        self.last_triggered = None
        self.trigger_count = 0
        self.events = []
    
    def check(self, value: float) -> Optional[BreakerAction]:
        """
        Check if breaker should trigger
        
        Args:
            value: Current value to check
            
        Returns:
            Action if triggered, None otherwise
        """
        # Check if in cooldown
        if self.status == BreakerStatus.COOLDOWN:
            if self._check_cooldown():
                self.status = BreakerStatus.ACTIVE
            else:
                return None
        
        # Check if disabled
        if self.status == BreakerStatus.DISABLED:
            return None
        
        # Check threshold
        if self._should_trigger(value):
            self.trigger()
            return self.action
        
        return None
    
    def trigger(self):
        """Trigger the circuit breaker"""
        self.status = BreakerStatus.TRIGGERED
        self.last_triggered = datetime.now()
        self.trigger_count += 1
        
        event = BreakerEvent(
            timestamp=datetime.now(),
            breaker_type=self.breaker_type,
            action=self.action,
            trigger_value=0,  # Should be passed in
            threshold=self.threshold,
            message=f"{self.name} triggered"
        )
        self.events.append(event)
        
        # Enter cooldown
        self.status = BreakerStatus.COOLDOWN
        
        logger.warning(f"Circuit breaker triggered: {self.name} - Action: {self.action.value}")
    
    def reset(self):
        """Reset the circuit breaker"""
        self.status = BreakerStatus.ACTIVE
        self.last_triggered = None
        logger.info(f"Circuit breaker reset: {self.name}")
    
    def disable(self):
        """Disable the circuit breaker"""
        self.status = BreakerStatus.DISABLED
        logger.info(f"Circuit breaker disabled: {self.name}")
    
    def enable(self):
        """Enable the circuit breaker"""
        self.status = BreakerStatus.ACTIVE
        logger.info(f"Circuit breaker enabled: {self.name}")
    
    def _should_trigger(self, value: float) -> bool:
        """Check if value exceeds threshold"""
        if self.breaker_type in [BreakerType.DRAWDOWN, BreakerType.DAILY_LOSS]:
            return value >= self.threshold
        elif self.breaker_type == BreakerType.VOLATILITY:
            return value >= self.threshold
        elif self.breaker_type == BreakerType.CORRELATION:
            return value >= self.threshold
        elif self.breaker_type == BreakerType.LIQUIDITY:
            return value <= self.threshold
        else:
            return value >= self.threshold
    
    def _check_cooldown(self) -> bool:
        """Check if cooldown period has expired"""
        if self.last_triggered is None:
            return True
        
        elapsed = datetime.now() - self.last_triggered
        return elapsed.total_seconds() / 60 >= self.cooldown_minutes


class CircuitBreakerSystem:
    """
    Comprehensive circuit breaker system for trading
    """
    
    def __init__(self,
                 config: Optional[BreakerConfig] = None,
                 action_callback: Optional[Callable] = None):
        """
        Initialize circuit breaker system
        
        Args:
            config: Circuit breaker configuration
            action_callback: Callback function for breaker actions
        """
        self.config = config or BreakerConfig()
        self.action_callback = action_callback
        
        # Initialize breakers
        self.breakers = self._initialize_breakers()
        
        # State tracking
        self.system_halted = False
        self.trading_allowed = True
        self.exposure_multiplier = 1.0
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=1000)
        self.drawdown_history = deque(maxlen=100)
        self.loss_streak = 0
        self.error_count = 0
        self.last_check = datetime.now()
        
        # Event log
        self.events = []
        
        logger.info("Circuit Breaker System initialized")
    
    def _initialize_breakers(self) -> Dict[str, CircuitBreaker]:
        """Initialize all circuit breakers"""
        breakers = {}
        
        # Drawdown breakers
        breakers['drawdown_warning'] = CircuitBreaker(
            "Drawdown Warning",
            BreakerType.DRAWDOWN,
            self.config.warning_drawdown,
            BreakerAction.WARNING,
            self.config.warning_cooldown
        )
        
        breakers['drawdown_stop'] = CircuitBreaker(
            "Drawdown Stop",
            BreakerType.DRAWDOWN,
            self.config.stop_drawdown,
            BreakerAction.STOP_NEW_TRADES,
            self.config.stop_cooldown
        )
        
        breakers['drawdown_emergency'] = CircuitBreaker(
            "Drawdown Emergency",
            BreakerType.DRAWDOWN,
            self.config.emergency_drawdown,
            BreakerAction.CLOSE_ALL,
            self.config.emergency_cooldown
        )
        
        # Daily loss breakers
        breakers['daily_loss_warning'] = CircuitBreaker(
            "Daily Loss Warning",
            BreakerType.DAILY_LOSS,
            self.config.daily_loss_warning,
            BreakerAction.WARNING,
            self.config.warning_cooldown
        )
        
        breakers['daily_loss_stop'] = CircuitBreaker(
            "Daily Loss Stop",
            BreakerType.DAILY_LOSS,
            self.config.daily_loss_stop,
            BreakerAction.STOP_NEW_TRADES,
            self.config.stop_cooldown
        )
        
        # Volatility breaker
        breakers['volatility_spike'] = CircuitBreaker(
            "Volatility Spike",
            BreakerType.VOLATILITY,
            self.config.volatility_spike_threshold,
            BreakerAction.REDUCE_EXPOSURE,
            self.config.warning_cooldown
        )
        
        # Correlation breaker
        breakers['high_correlation'] = CircuitBreaker(
            "High Correlation",
            BreakerType.CORRELATION,
            self.config.correlation_threshold,
            BreakerAction.REDUCE_EXPOSURE,
            self.config.warning_cooldown
        )
        
        # Liquidity breaker
        breakers['low_liquidity'] = CircuitBreaker(
            "Low Liquidity",
            BreakerType.LIQUIDITY,
            self.config.liquidity_ratio_min,
            BreakerAction.STOP_NEW_TRADES,
            self.config.stop_cooldown
        )
        
        return breakers
    
    async def check_all_breakers(self, metrics: Dict[str, float]) -> List[BreakerAction]:
        """
        Check all circuit breakers
        
        Args:
            metrics: Current system metrics
            
        Returns:
            List of triggered actions
        """
        triggered_actions = []
        
        # Update metrics history
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # Check each breaker
        for breaker_name, breaker in self.breakers.items():
            metric_value = self._get_metric_for_breaker(breaker.breaker_type, metrics)
            
            if metric_value is not None:
                action = breaker.check(metric_value)
                
                if action:
                    triggered_actions.append(action)
                    await self._execute_action(action, breaker_name, metric_value)
        
        # Update system state based on actions
        self._update_system_state(triggered_actions)
        
        return triggered_actions
    
    async def _execute_action(self, action: BreakerAction, breaker_name: str, value: float):
        """Execute circuit breaker action"""
        event = BreakerEvent(
            timestamp=datetime.now(),
            breaker_type=self.breakers[breaker_name].breaker_type,
            action=action,
            trigger_value=value,
            threshold=self.breakers[breaker_name].threshold,
            message=f"Action {action.value} executed by {breaker_name}",
            metadata={'breaker': breaker_name, 'value': value}
        )
        self.events.append(event)
        
        # Execute action
        if action == BreakerAction.WARNING:
            logger.warning(f"RISK WARNING: {breaker_name} triggered with value {value}")
            
        elif action == BreakerAction.REDUCE_EXPOSURE:
            self.exposure_multiplier = 0.5
            logger.warning(f"REDUCING EXPOSURE: Position sizes reduced to 50%")
            
        elif action == BreakerAction.STOP_NEW_TRADES:
            self.trading_allowed = False
            logger.error(f"TRADING STOPPED: No new trades allowed")
            
        elif action == BreakerAction.CLOSE_LOSING:
            logger.error(f"CLOSING LOSING POSITIONS: Breaker {breaker_name} triggered")
            if self.action_callback:
                await self.action_callback('close_losing')
                
        elif action == BreakerAction.CLOSE_ALL:
            logger.critical(f"EMERGENCY: CLOSING ALL POSITIONS")
            if self.action_callback:
                await self.action_callback('close_all')
                
        elif action == BreakerAction.EMERGENCY_HEDGE:
            logger.critical(f"EMERGENCY HEDGE: Initiating hedge positions")
            if self.action_callback:
                await self.action_callback('emergency_hedge')
                
        elif action == BreakerAction.SYSTEM_HALT:
            self.system_halted = True
            logger.critical(f"SYSTEM HALT: All trading operations suspended")
            if self.action_callback:
                await self.action_callback('system_halt')
    
    def _get_metric_for_breaker(self, breaker_type: BreakerType, metrics: Dict[str, float]) -> Optional[float]:
        """Get appropriate metric for breaker type"""
        mapping = {
            BreakerType.DRAWDOWN: 'drawdown',
            BreakerType.DAILY_LOSS: 'daily_loss',
            BreakerType.VOLATILITY: 'volatility_ratio',
            BreakerType.CORRELATION: 'max_correlation',
            BreakerType.LIQUIDITY: 'liquidity_ratio',
            BreakerType.TECHNICAL: 'technical_score',
            BreakerType.SYSTEM: 'system_health'
        }
        
        metric_key = mapping.get(breaker_type)
        return metrics.get(metric_key) if metric_key else None
    
    def _update_system_state(self, actions: List[BreakerAction]):
        """Update system state based on triggered actions"""
        # Determine most severe action
        if BreakerAction.SYSTEM_HALT in actions:
            self.system_halted = True
            self.trading_allowed = False
            self.exposure_multiplier = 0
            
        elif BreakerAction.CLOSE_ALL in actions:
            self.trading_allowed = False
            self.exposure_multiplier = 0
            
        elif BreakerAction.STOP_NEW_TRADES in actions:
            self.trading_allowed = False
            
        elif BreakerAction.REDUCE_EXPOSURE in actions:
            self.exposure_multiplier = min(self.exposure_multiplier, 0.5)
    
    def check_recovery_conditions(self, metrics: Dict[str, float]) -> bool:
        """
        Check if recovery conditions are met
        
        Args:
            metrics: Current system metrics
            
        Returns:
            True if recovery conditions met
        """
        # Check if enough time has passed
        min_time = timedelta(minutes=self.config.recovery_time_minimum)
        time_ok = all(
            not b.last_triggered or 
            (datetime.now() - b.last_triggered) > min_time
            for b in self.breakers.values()
        )
        
        if not time_ok:
            return False
        
        # Check profit threshold
        current_pnl = metrics.get('pnl_percent', 0)
        profit_ok = current_pnl >= self.config.recovery_profit_threshold
        
        # Check drawdown recovered
        current_drawdown = metrics.get('drawdown', 0)
        drawdown_ok = current_drawdown < self.config.warning_drawdown * 0.5
        
        # Check volatility normalized
        volatility_ratio = metrics.get('volatility_ratio', 1.0)
        volatility_ok = volatility_ratio < 1.5
        
        return profit_ok and drawdown_ok and volatility_ok
    
    def reset_breakers(self, force: bool = False):
        """
        Reset circuit breakers
        
        Args:
            force: Force reset regardless of conditions
        """
        if force:
            for breaker in self.breakers.values():
                breaker.reset()
            
            self.system_halted = False
            self.trading_allowed = True
            self.exposure_multiplier = 1.0
            self.loss_streak = 0
            self.error_count = 0
            
            logger.info("Circuit breakers force reset")
        else:
            # Check recovery conditions first
            logger.info("Checking recovery conditions for reset...")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker system status"""
        return {
            "system_halted": self.system_halted,
            "trading_allowed": self.trading_allowed,
            "exposure_multiplier": self.exposure_multiplier,
            "breakers": {
                name: {
                    "status": breaker.status.value,
                    "trigger_count": breaker.trigger_count,
                    "last_triggered": breaker.last_triggered.isoformat() if breaker.last_triggered else None
                }
                for name, breaker in self.breakers.items()
            },
            "loss_streak": self.loss_streak,
            "error_count": self.error_count,
            "last_check": self.last_check.isoformat(),
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.breaker_type.value,
                    "action": event.action.value,
                    "message": event.message
                }
                for event in self.events[-10:]  # Last 10 events
            ]
        }
    
    def emergency_override(self, override_code: str) -> bool:
        """
        Emergency override for manual intervention
        
        Args:
            override_code: Security code for override
            
        Returns:
            True if override successful
        """
        # In production, this should verify against secure code
        if override_code == "EMERGENCY_OVERRIDE_2024":
            logger.critical("EMERGENCY OVERRIDE ACTIVATED")
            self.reset_breakers(force=True)
            
            # Log override event
            event = BreakerEvent(
                timestamp=datetime.now(),
                breaker_type=BreakerType.MANUAL,
                action=BreakerAction.SYSTEM_HALT,
                trigger_value=0,
                threshold=0,
                message="Emergency override activated",
                metadata={"override_code": "***"}
            )
            self.events.append(event)
            
            return True
        
        logger.error("Invalid override code")
        return False
    
    def export_events(self, filepath: str):
        """Export events to file for analysis"""
        events_data = [
            {
                "timestamp": event.timestamp.isoformat(),
                "breaker_type": event.breaker_type.value,
                "action": event.action.value,
                "trigger_value": event.trigger_value,
                "threshold": event.threshold,
                "message": event.message,
                "metadata": event.metadata
            }
            for event in self.events
        ]
        
        with open(filepath, 'w') as f:
            json.dump(events_data, f, indent=2)
        
        logger.info(f"Exported {len(events_data)} events to {filepath}")


class KillSwitch:
    """
    Ultimate kill switch for emergency situations
    """
    
    def __init__(self, callback: Callable):
        """
        Initialize kill switch
        
        Args:
            callback: Emergency callback function
        """
        self.callback = callback
        self.activated = False
        self.activation_time = None
        self.activation_reason = None
    
    async def activate(self, reason: str = "MANUAL"):
        """
        Activate kill switch
        
        Args:
            reason: Reason for activation
        """
        if self.activated:
            logger.warning("Kill switch already activated")
            return
        
        self.activated = True
        self.activation_time = datetime.now()
        self.activation_reason = reason
        
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        
        # Execute emergency callback
        if self.callback:
            await self.callback()
        
        # Log critical event
        with open("kill_switch_log.txt", "a") as f:
            f.write(f"{self.activation_time.isoformat()} - KILL SWITCH: {reason}\n")
    
    def deactivate(self, authorization: str) -> bool:
        """
        Deactivate kill switch
        
        Args:
            authorization: Authorization code
            
        Returns:
            True if deactivated
        """
        if authorization != "DEACTIVATE_KILL_SWITCH_2024":
            logger.error("Invalid kill switch deactivation code")
            return False
        
        self.activated = False
        logger.info("Kill switch deactivated")
        return True
    
    def status(self) -> Dict[str, Any]:
        """Get kill switch status"""
        return {
            "activated": self.activated,
            "activation_time": self.activation_time.isoformat() if self.activation_time else None,
            "activation_reason": self.activation_reason
        }