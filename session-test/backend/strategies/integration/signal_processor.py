"""
Signal Processor
================

Processes and prioritizes trading signals from multiple institutional strategies.
Handles signal aggregation, conflict resolution, confidence weighting, and
real-time signal routing to the RL system.

Key Features:
- Multi-strategy signal aggregation
- Conflict resolution and consensus building
- Confidence-weighted signal combining
- Real-time signal prioritization
- Signal validation and filtering
- Explainable signal reasoning
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from threading import Lock
import json
import heapq

# Statistical libraries
from scipy.stats import combine_pvalues, norm
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    NEUTRAL = "neutral"


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    CRITICAL = 5


class ConflictResolution(Enum):
    """Methods for resolving signal conflicts"""
    WEIGHTED_AVERAGE = "weighted_average"
    CONSENSUS = "consensus"
    HIGHEST_CONFIDENCE = "highest_confidence"
    STRATEGY_PRIORITY = "strategy_priority"
    VOTING = "voting"
    ENSEMBLE = "ensemble"


@dataclass
class TradingSignal:
    """Individual trading signal from a strategy"""
    strategy_name: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: str = "1h"  # Signal timeframe
    reasoning: str = ""  # Explanation for signal
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    expiry: Optional[datetime] = None
    priority: int = 1  # 1 = highest priority
    volume_impact: float = 0.0  # Expected volume impact
    risk_score: float = 0.5  # Risk assessment 0-1


@dataclass
class AggregatedSignal:
    """Aggregated signal combining multiple strategy signals"""
    signal_type: SignalType
    strength: SignalStrength
    confidence: float
    contributing_strategies: List[str]
    strategy_weights: Dict[str, float]
    consensus_score: float  # How much strategies agree
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    signal_id: str = ""
    risk_adjusted_strength: float = 0.0
    execution_priority: int = 1


@dataclass
class SignalConflict:
    """Information about conflicting signals"""
    conflicting_signals: List[TradingSignal]
    conflict_type: str  # "direction", "strength", "timeframe"
    resolution_method: ConflictResolution
    resolved_signal: Optional[AggregatedSignal] = None
    confidence_impact: float = 0.0  # Impact on final confidence


@dataclass
class ProcessingStats:
    """Statistics for signal processing"""
    total_signals_received: int = 0
    signals_processed: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    avg_processing_time: float = 0.0
    avg_consensus_score: float = 0.0
    last_signal_time: Optional[datetime] = None
    error_count: int = 0


class SignalProcessor:
    """
    Advanced signal processing system for institutional trading strategies.
    
    Combines signals from multiple strategies, resolves conflicts, and provides
    high-confidence trading recommendations to the RL system.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize signal processor"""
        self.config = config or self._default_config()
        
        # Signal storage
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history: deque = deque(maxlen=self.config['history_length'])
        self.aggregated_signals: List[AggregatedSignal] = []
        
        # Strategy management
        self.strategy_weights: Dict[str, float] = {}
        self.strategy_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.strategy_reliability: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # Signal processing
        self.processing_queue: List[TradingSignal] = []
        self.conflict_queue: List[SignalConflict] = []
        self.lock = Lock()
        
        # Performance tracking
        self.stats = ProcessingStats()
        
        # Signal validation
        self.validators: List[Callable] = []
        self.filters: List[Callable] = []
        
        # Consensus building
        self.consensus_threshold = self.config.get('consensus_threshold', 0.6)
        self.min_strategies_for_consensus = self.config.get('min_strategies_for_consensus', 2)
        
        logger.info("Signal Processor initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'history_length': 1000,
            'processing_interval': 0.5,  # seconds
            'conflict_resolution_method': 'weighted_average',
            'consensus_threshold': 0.6,
            'min_strategies_for_consensus': 2,
            'signal_expiry_default': 3600,  # seconds
            'confidence_threshold': 0.5,
            'max_concurrent_signals': 10,
            'enable_signal_validation': True,
            'enable_conflict_detection': True,
            'enable_real_time_processing': True,
            'weight_decay_factor': 0.95,  # For strategy weight updates
            'risk_adjustment_factor': 0.8,
            'timeframe_weights': {
                '1m': 0.3,
                '5m': 0.5,
                '15m': 0.7,
                '1h': 1.0,
                '4h': 1.2,
                '1d': 1.5
            },
            'strategy_priority_weights': {
                'whale_tracker': 1.5,
                'volume_profile': 1.3,
                'order_book_analyzer': 1.2,
                'regime_detection': 1.0,
                'smart_money_divergence': 0.9
            }
        }
    
    async def start(self):
        """Start the signal processor"""
        logger.info("Starting Signal Processor...")
        
        # Initialize strategy weights
        self._initialize_strategy_weights()
        
        # Start processing loop
        if self.config.get('enable_real_time_processing', True):
            asyncio.create_task(self._processing_loop())
        
        logger.info("Signal Processor started")
    
    async def stop(self):
        """Stop the signal processor"""
        logger.info("Signal Processor stopped")
    
    def _initialize_strategy_weights(self):
        """Initialize strategy weights from config"""
        priority_weights = self.config.get('strategy_priority_weights', {})
        
        for strategy, weight in priority_weights.items():
            self.strategy_weights[strategy] = weight
            self.strategy_reliability[strategy] = 0.5  # Start neutral
    
    async def add_signals(self, signals: Dict[str, Any]):
        """Add new signals from strategies"""
        with self.lock:
            try:
                for signal_name, signal_data in signals.items():
                    # Parse signal data
                    trading_signal = self._parse_signal(signal_name, signal_data)
                    
                    if trading_signal:
                        # Validate signal
                        if self._validate_signal(trading_signal):
                            # Add to processing queue
                            self.processing_queue.append(trading_signal)
                            self.stats.total_signals_received += 1
                        else:
                            logger.warning(f"Signal validation failed for {signal_name}")
                
                # Process signals if not in real-time mode
                if not self.config.get('enable_real_time_processing', True):
                    await self._process_signals()
                
            except Exception as e:
                self.stats.error_count += 1
                logger.error(f"Error adding signals: {e}")
    
    def _parse_signal(self, signal_name: str, signal_data: Any) -> Optional[TradingSignal]:
        """Parse signal data into TradingSignal object"""
        try:
            # Extract strategy name from signal name
            strategy_name = signal_name.split('_')[0] if '_' in signal_name else 'unknown'
            
            # Handle different signal data formats
            if isinstance(signal_data, dict):
                signal_type = self._parse_signal_type(signal_data.get('type', signal_data.get('action', 'hold')))
                strength = self._parse_signal_strength(signal_data.get('strength', 2))
                confidence = float(signal_data.get('confidence', 0.5))
                price_target = signal_data.get('price_target')
                stop_loss = signal_data.get('stop_loss')
                take_profit = signal_data.get('take_profit')
                reasoning = signal_data.get('reasoning', '')
                metadata = signal_data.get('metadata', {})
            
            elif isinstance(signal_data, (int, float)):
                # Simple numeric signal
                if signal_data > 0.6:
                    signal_type = SignalType.BUY
                    strength = SignalStrength.MODERATE
                elif signal_data < -0.6:
                    signal_type = SignalType.SELL
                    strength = SignalStrength.MODERATE
                else:
                    signal_type = SignalType.HOLD
                    strength = SignalStrength.WEAK
                
                confidence = abs(signal_data)
                price_target = None
                stop_loss = None
                take_profit = None
                reasoning = f"Numeric signal value: {signal_data}"
                metadata = {}
            
            else:
                # String or other format
                signal_str = str(signal_data).lower()
                if 'buy' in signal_str:
                    signal_type = SignalType.BUY
                    strength = SignalStrength.MODERATE
                elif 'sell' in signal_str:
                    signal_type = SignalType.SELL
                    strength = SignalStrength.MODERATE
                else:
                    signal_type = SignalType.HOLD
                    strength = SignalStrength.WEAK
                
                confidence = 0.5
                price_target = None
                stop_loss = None
                take_profit = None
                reasoning = f"String signal: {signal_data}"
                metadata = {}
            
            # Calculate priority based on strategy
            priority = self._calculate_signal_priority(strategy_name, signal_type, confidence)
            
            # Calculate expiry
            expiry = datetime.now() + timedelta(seconds=self.config['signal_expiry_default'])
            
            return TradingSignal(
                strategy_name=strategy_name,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                metadata=metadata,
                timestamp=datetime.now(),
                expiry=expiry,
                priority=priority
            )
            
        except Exception as e:
            logger.error(f"Error parsing signal {signal_name}: {e}")
            return None
    
    def _parse_signal_type(self, signal_type_str: str) -> SignalType:
        """Parse signal type from string"""
        signal_type_str = str(signal_type_str).lower()
        
        if signal_type_str in ['buy', 'long', 'bullish']:
            return SignalType.BUY
        elif signal_type_str in ['sell', 'short', 'bearish']:
            return SignalType.SELL
        elif signal_type_str in ['strong_buy', 'strong_bullish']:
            return SignalType.STRONG_BUY
        elif signal_type_str in ['strong_sell', 'strong_bearish']:
            return SignalType.STRONG_SELL
        elif signal_type_str in ['neutral']:
            return SignalType.NEUTRAL
        else:
            return SignalType.HOLD
    
    def _parse_signal_strength(self, strength_value: Union[int, float, str]) -> SignalStrength:
        """Parse signal strength from various formats"""
        if isinstance(strength_value, str):
            strength_str = strength_value.lower()
            if 'weak' in strength_str:
                return SignalStrength.WEAK
            elif 'strong' in strength_str or 'very' in strength_str:
                return SignalStrength.VERY_STRONG
            elif 'moderate' in strength_str:
                return SignalStrength.MODERATE
            elif 'critical' in strength_str:
                return SignalStrength.CRITICAL
            else:
                return SignalStrength.MODERATE
        
        elif isinstance(strength_value, (int, float)):
            if strength_value <= 1:
                return SignalStrength.WEAK
            elif strength_value <= 2:
                return SignalStrength.MODERATE
            elif strength_value <= 3:
                return SignalStrength.STRONG
            elif strength_value <= 4:
                return SignalStrength.VERY_STRONG
            else:
                return SignalStrength.CRITICAL
        
        return SignalStrength.MODERATE
    
    def _calculate_signal_priority(self, strategy_name: str, signal_type: SignalType, confidence: float) -> int:
        """Calculate signal priority"""
        base_priority = 5
        
        # Strategy weight impact
        strategy_weight = self.strategy_weights.get(strategy_name, 1.0)
        priority_adjustment = int((strategy_weight - 1.0) * 2)
        
        # Confidence impact
        if confidence > 0.8:
            priority_adjustment -= 2
        elif confidence > 0.6:
            priority_adjustment -= 1
        elif confidence < 0.3:
            priority_adjustment += 2
        
        # Signal type impact
        if signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            priority_adjustment -= 1
        
        return max(1, base_priority + priority_adjustment)
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate signal quality and consistency"""
        try:
            # Basic validation
            if not (0.0 <= signal.confidence <= 1.0):
                return False
            
            if signal.timestamp > datetime.now() + timedelta(minutes=1):
                return False  # Future timestamp
            
            # Strategy-specific validation
            if not self._validate_strategy_signal(signal):
                return False
            
            # Custom validators
            for validator in self.validators:
                if not validator(signal):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def _validate_strategy_signal(self, signal: TradingSignal) -> bool:
        """Strategy-specific signal validation"""
        strategy_name = signal.strategy_name
        
        # Check if strategy is known
        if strategy_name not in self.strategy_weights:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return False
        
        # Check signal consistency with strategy reliability
        reliability = self.strategy_reliability.get(strategy_name, 0.5)
        
        # Lower confidence for unreliable strategies
        if reliability < 0.3 and signal.confidence > 0.7:
            signal.confidence *= reliability / 0.3
        
        return True
    
    async def _processing_loop(self):
        """Main signal processing loop"""
        while True:
            try:
                await asyncio.sleep(self.config['processing_interval'])
                
                if self.processing_queue:
                    await self._process_signals()
                
                # Clean up expired signals
                await self._cleanup_expired_signals()
                
                # Update strategy performance
                await self._update_strategy_performance()
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_signals(self):
        """Process queued signals"""
        if not self.processing_queue:
            return
        
        start_time = time.time()
        
        with self.lock:
            # Sort signals by priority
            self.processing_queue.sort(key=lambda s: (s.priority, -s.confidence))
            
            # Process signals in batches
            batch_size = min(len(self.processing_queue), self.config['max_concurrent_signals'])
            current_batch = self.processing_queue[:batch_size]
            self.processing_queue = self.processing_queue[batch_size:]
        
        # Update active signals
        for signal in current_batch:
            signal_key = f"{signal.strategy_name}_{signal.signal_type.value}"
            self.active_signals[signal_key] = signal
            self.signal_history.append(signal)
        
        # Detect conflicts
        conflicts = []
        if self.config.get('enable_conflict_detection', True):
            conflicts = self._detect_conflicts(current_batch)
        
        # Resolve conflicts and aggregate signals
        aggregated = await self._aggregate_signals(current_batch, conflicts)
        
        if aggregated:
            self.aggregated_signals.append(aggregated)
            
            # Keep only recent aggregated signals
            max_aggregated = self.config.get('max_concurrent_signals', 10)
            if len(self.aggregated_signals) > max_aggregated:
                self.aggregated_signals = self.aggregated_signals[-max_aggregated:]
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats.avg_processing_time = (
            self.stats.avg_processing_time * 0.9 + processing_time * 0.1
        )
        self.stats.signals_processed += len(current_batch)
        self.stats.conflicts_detected += len(conflicts)
        self.stats.last_signal_time = datetime.now()
        
        logger.debug(f"Processed {len(current_batch)} signals in {processing_time:.3f}s")
    
    def _detect_conflicts(self, signals: List[TradingSignal]) -> List[SignalConflict]:
        """Detect conflicts between signals"""
        conflicts = []
        
        # Group signals by timeframe
        timeframe_groups = defaultdict(list)
        for signal in signals:
            timeframe_groups[signal.timeframe].append(signal)
        
        # Check for conflicts within each timeframe
        for timeframe, timeframe_signals in timeframe_groups.items():
            if len(timeframe_signals) < 2:
                continue
            
            # Check for directional conflicts
            directional_conflicts = self._find_directional_conflicts(timeframe_signals)
            conflicts.extend(directional_conflicts)
            
            # Check for strength conflicts
            strength_conflicts = self._find_strength_conflicts(timeframe_signals)
            conflicts.extend(strength_conflicts)
        
        return conflicts
    
    def _find_directional_conflicts(self, signals: List[TradingSignal]) -> List[SignalConflict]:
        """Find directional conflicts (buy vs sell)"""
        conflicts = []
        
        buy_signals = [s for s in signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]]
        sell_signals = [s for s in signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]]
        
        if buy_signals and sell_signals:
            conflict = SignalConflict(
                conflicting_signals=buy_signals + sell_signals,
                conflict_type="direction",
                resolution_method=ConflictResolution.WEIGHTED_AVERAGE
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _find_strength_conflicts(self, signals: List[TradingSignal]) -> List[SignalConflict]:
        """Find strength conflicts (different strength levels for same direction)"""
        conflicts = []
        
        # Group by signal type
        type_groups = defaultdict(list)
        for signal in signals:
            type_groups[signal.signal_type].append(signal)
        
        # Check for strength differences within each type
        for signal_type, type_signals in type_groups.items():
            if len(type_signals) > 1:
                strengths = [s.strength.value for s in type_signals]
                if max(strengths) - min(strengths) > 2:  # Significant strength difference
                    conflict = SignalConflict(
                        conflicting_signals=type_signals,
                        conflict_type="strength",
                        resolution_method=ConflictResolution.HIGHEST_CONFIDENCE
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    async def _aggregate_signals(self, signals: List[TradingSignal], conflicts: List[SignalConflict]) -> Optional[AggregatedSignal]:
        """Aggregate signals into final recommendation"""
        if not signals:
            return None
        
        try:
            # Resolve conflicts first
            resolved_signals = signals.copy()
            for conflict in conflicts:
                resolved_signal = await self._resolve_conflict(conflict)
                if resolved_signal:
                    # Replace conflicting signals with resolved signal
                    for conf_signal in conflict.conflicting_signals:
                        if conf_signal in resolved_signals:
                            resolved_signals.remove(conf_signal)
                    
                    # Convert AggregatedSignal back to TradingSignal for processing
                    trading_signal = self._aggregated_to_trading_signal(resolved_signal)
                    resolved_signals.append(trading_signal)
            
            # Calculate weighted consensus
            return await self._calculate_weighted_consensus(resolved_signals)
            
        except Exception as e:
            logger.error(f"Error aggregating signals: {e}")
            return None
    
    async def _resolve_conflict(self, conflict: SignalConflict) -> Optional[AggregatedSignal]:
        """Resolve signal conflict using specified method"""
        signals = conflict.conflicting_signals
        method = conflict.resolution_method
        
        if method == ConflictResolution.WEIGHTED_AVERAGE:
            return await self._resolve_weighted_average(signals)
        elif method == ConflictResolution.HIGHEST_CONFIDENCE:
            return await self._resolve_highest_confidence(signals)
        elif method == ConflictResolution.CONSENSUS:
            return await self._resolve_consensus(signals)
        elif method == ConflictResolution.STRATEGY_PRIORITY:
            return await self._resolve_strategy_priority(signals)
        else:
            return await self._resolve_weighted_average(signals)  # Default
    
    async def _resolve_weighted_average(self, signals: List[TradingSignal]) -> AggregatedSignal:
        """Resolve conflict using weighted average"""
        # Calculate weights based on strategy reliability and confidence
        weights = []
        for signal in signals:
            strategy_weight = self.strategy_weights.get(signal.strategy_name, 1.0)
            reliability = self.strategy_reliability.get(signal.strategy_name, 0.5)
            weight = strategy_weight * reliability * signal.confidence
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(signals)] * len(signals)
        
        # Calculate weighted signal values
        signal_values = []
        for signal in signals:
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                value = signal.strength.value
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                value = -signal.strength.value
            else:
                value = 0
            signal_values.append(value)
        
        # Weighted average
        weighted_value = sum(v * w for v, w in zip(signal_values, weights))
        
        # Determine final signal type and strength
        if weighted_value > 2:
            final_type = SignalType.STRONG_BUY
            final_strength = SignalStrength.STRONG
        elif weighted_value > 0.5:
            final_type = SignalType.BUY
            final_strength = SignalStrength.MODERATE
        elif weighted_value < -2:
            final_type = SignalType.STRONG_SELL
            final_strength = SignalStrength.STRONG
        elif weighted_value < -0.5:
            final_type = SignalType.SELL
            final_strength = SignalStrength.MODERATE
        else:
            final_type = SignalType.HOLD
            final_strength = SignalStrength.WEAK
        
        # Calculate weighted confidence
        weighted_confidence = sum(s.confidence * w for s, w in zip(signals, weights))
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(signals)
        
        return AggregatedSignal(
            signal_type=final_type,
            strength=final_strength,
            confidence=weighted_confidence,
            contributing_strategies=[s.strategy_name for s in signals],
            strategy_weights={s.strategy_name: w for s, w in zip(signals, weights)},
            consensus_score=consensus_score,
            reasoning=self._build_aggregated_reasoning(signals, weights)
        )
    
    async def _resolve_highest_confidence(self, signals: List[TradingSignal]) -> AggregatedSignal:
        """Resolve conflict by selecting highest confidence signal"""
        best_signal = max(signals, key=lambda s: s.confidence)
        
        return AggregatedSignal(
            signal_type=best_signal.signal_type,
            strength=best_signal.strength,
            confidence=best_signal.confidence,
            contributing_strategies=[best_signal.strategy_name],
            strategy_weights={best_signal.strategy_name: 1.0},
            consensus_score=0.0,  # No consensus, single signal
            reasoning=f"Highest confidence signal from {best_signal.strategy_name}: {best_signal.reasoning}"
        )
    
    async def _resolve_consensus(self, signals: List[TradingSignal]) -> Optional[AggregatedSignal]:
        """Resolve conflict by finding consensus"""
        # Group signals by type
        type_groups = defaultdict(list)
        for signal in signals:
            type_groups[signal.signal_type].append(signal)
        
        # Find majority type
        majority_type = max(type_groups.keys(), key=lambda t: len(type_groups[t]))
        majority_signals = type_groups[majority_type]
        
        # Check if consensus threshold is met
        consensus_ratio = len(majority_signals) / len(signals)
        if consensus_ratio < self.consensus_threshold:
            return None  # No consensus
        
        # Create aggregated signal from consensus
        avg_confidence = np.mean([s.confidence for s in majority_signals])
        avg_strength = int(np.mean([s.strength.value for s in majority_signals]))
        
        return AggregatedSignal(
            signal_type=majority_type,
            strength=SignalStrength(avg_strength),
            confidence=avg_confidence,
            contributing_strategies=[s.strategy_name for s in majority_signals],
            strategy_weights={s.strategy_name: 1.0/len(majority_signals) for s in majority_signals},
            consensus_score=consensus_ratio,
            reasoning=f"Consensus signal ({consensus_ratio:.1%} agreement) from {len(majority_signals)} strategies"
        )
    
    async def _resolve_strategy_priority(self, signals: List[TradingSignal]) -> AggregatedSignal:
        """Resolve conflict using strategy priority"""
        # Find highest priority signal
        priority_signal = min(signals, key=lambda s: s.priority)
        
        return AggregatedSignal(
            signal_type=priority_signal.signal_type,
            strength=priority_signal.strength,
            confidence=priority_signal.confidence,
            contributing_strategies=[priority_signal.strategy_name],
            strategy_weights={priority_signal.strategy_name: 1.0},
            consensus_score=0.0,
            reasoning=f"Priority signal from {priority_signal.strategy_name}: {priority_signal.reasoning}"
        )
    
    async def _calculate_weighted_consensus(self, signals: List[TradingSignal]) -> AggregatedSignal:
        """Calculate final weighted consensus from all signals"""
        if len(signals) == 1:
            signal = signals[0]
            return AggregatedSignal(
                signal_type=signal.signal_type,
                strength=signal.strength,
                confidence=signal.confidence,
                contributing_strategies=[signal.strategy_name],
                strategy_weights={signal.strategy_name: 1.0},
                consensus_score=1.0,
                reasoning=signal.reasoning
            )
        
        # Use weighted average method for final aggregation
        return await self._resolve_weighted_average(signals)
    
    def _calculate_consensus_score(self, signals: List[TradingSignal]) -> float:
        """Calculate consensus score for signals"""
        if len(signals) <= 1:
            return 1.0
        
        # Count signals by type
        type_counts = defaultdict(int)
        for signal in signals:
            type_counts[signal.signal_type] += 1
        
        # Calculate agreement ratio
        max_count = max(type_counts.values())
        total_count = len(signals)
        
        return max_count / total_count
    
    def _build_aggregated_reasoning(self, signals: List[TradingSignal], weights: List[float]) -> str:
        """Build explanation for aggregated signal"""
        reasoning_parts = []
        
        for signal, weight in zip(signals, weights):
            if weight > 0.1:  # Only include significant contributors
                reasoning_parts.append(
                    f"{signal.strategy_name} ({weight:.1%}): {signal.reasoning}"
                )
        
        return " | ".join(reasoning_parts)
    
    def _aggregated_to_trading_signal(self, aggregated: AggregatedSignal) -> TradingSignal:
        """Convert AggregatedSignal back to TradingSignal for processing"""
        return TradingSignal(
            strategy_name="aggregated",
            signal_type=aggregated.signal_type,
            strength=aggregated.strength,
            confidence=aggregated.confidence,
            reasoning=aggregated.reasoning,
            timestamp=aggregated.timestamp
        )
    
    async def _cleanup_expired_signals(self):
        """Remove expired signals"""
        current_time = datetime.now()
        
        # Clean active signals
        expired_keys = []
        for key, signal in self.active_signals.items():
            if signal.expiry and current_time > signal.expiry:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.active_signals[key]
        
        # Clean aggregated signals (keep recent ones)
        max_age = timedelta(hours=1)
        self.aggregated_signals = [
            signal for signal in self.aggregated_signals
            if current_time - signal.timestamp < max_age
        ]
    
    async def _update_strategy_performance(self):
        """Update strategy performance and reliability scores"""
        # This would typically use actual trading results
        # For now, we'll use signal accuracy and consistency
        
        for strategy_name in self.strategy_weights.keys():
            # Count recent signals from this strategy
            recent_signals = [
                signal for signal in self.signal_history
                if (signal.strategy_name == strategy_name and 
                    datetime.now() - signal.timestamp < timedelta(hours=24))
            ]
            
            if recent_signals:
                # Calculate consistency (how similar are the signals)
                avg_confidence = np.mean([s.confidence for s in recent_signals])
                confidence_std = np.std([s.confidence for s in recent_signals])
                consistency = 1.0 - min(confidence_std, 1.0)
                
                # Update reliability (would use actual performance in production)
                current_reliability = self.strategy_reliability[strategy_name]
                new_reliability = current_reliability * 0.9 + consistency * 0.1
                self.strategy_reliability[strategy_name] = new_reliability
    
    def get_latest_signal(self) -> Optional[AggregatedSignal]:
        """Get the latest aggregated signal"""
        return self.aggregated_signals[-1] if self.aggregated_signals else None
    
    def get_processed_signals(self) -> List[AggregatedSignal]:
        """Get all processed signals"""
        return self.aggregated_signals.copy()
    
    def get_active_signals(self) -> Dict[str, TradingSignal]:
        """Get all active signals"""
        return self.active_signals.copy()
    
    def update_strategy_weight(self, strategy_name: str, weight: float):
        """Update weight for specific strategy"""
        self.strategy_weights[strategy_name] = weight
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Get strategy performance metrics"""
        performance = {}
        
        for strategy_name in self.strategy_weights.keys():
            recent_signals = [
                signal for signal in self.signal_history
                if (signal.strategy_name == strategy_name and 
                    datetime.now() - signal.timestamp < timedelta(hours=24))
            ]
            
            if recent_signals:
                performance[strategy_name] = {
                    'signal_count': len(recent_signals),
                    'avg_confidence': np.mean([s.confidence for s in recent_signals]),
                    'reliability': self.strategy_reliability.get(strategy_name, 0.5),
                    'weight': self.strategy_weights.get(strategy_name, 1.0)
                }
        
        return performance
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        return {
            'stats': {
                'total_signals_received': self.stats.total_signals_received,
                'signals_processed': self.stats.signals_processed,
                'conflicts_detected': self.stats.conflicts_detected,
                'conflicts_resolved': self.stats.conflicts_resolved,
                'avg_processing_time': self.stats.avg_processing_time,
                'avg_consensus_score': self.stats.avg_consensus_score,
                'last_signal_time': self.stats.last_signal_time.isoformat() if self.stats.last_signal_time else None,
                'error_count': self.stats.error_count
            },
            'current_state': {
                'active_signals': len(self.active_signals),
                'processing_queue': len(self.processing_queue),
                'aggregated_signals': len(self.aggregated_signals),
                'latest_signal': self.get_latest_signal().__dict__ if self.get_latest_signal() else None
            },
            'strategy_performance': self.get_strategy_performance(),
            'config': self.config
        }


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create signal processor
        processor = SignalProcessor()
        
        try:
            await processor.start()
            
            # Add some sample signals
            signals = {
                'whale_tracker_buy_signal': {
                    'type': 'buy',
                    'strength': 3,
                    'confidence': 0.8,
                    'reasoning': 'Large whale accumulation detected'
                },
                'volume_profile_breakout': {
                    'type': 'buy',
                    'strength': 2,
                    'confidence': 0.7,
                    'reasoning': 'Volume breakout above POC'
                },
                'order_book_sell_signal': {
                    'type': 'sell',
                    'strength': 2,
                    'confidence': 0.6,
                    'reasoning': 'Large sell walls detected'
                }
            }
            
            await processor.add_signals(signals)
            
            # Wait for processing
            await asyncio.sleep(1)
            
            # Get latest signal
            latest = processor.get_latest_signal()
            if latest:
                print(f"Latest signal: {latest.signal_type.value} with confidence {latest.confidence:.2f}")
                print(f"Reasoning: {latest.reasoning}")
            
            # Get metrics
            metrics = processor.get_metrics()
            print(f"Processing metrics: {metrics}")
            
        finally:
            await processor.stop()
    
    asyncio.run(main())