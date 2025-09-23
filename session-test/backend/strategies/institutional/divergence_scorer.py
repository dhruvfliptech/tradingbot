"""
Divergence Scorer - Signal Generation and Scoring System
=========================================================

Comprehensive scoring system that combines all smart money indicators
to generate high-confidence trading signals with risk-adjusted scores.

Key Features:
- Multi-factor signal scoring
- Confidence weighting
- Historical validation
- Risk-adjusted signal generation
- RL integration for adaptive scoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class RiskLevel(Enum):
    """Risk levels for signals"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class CompositeSignal:
    """Composite trading signal with all components"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    composite_score: float  # -1 to 1
    confidence: float  # 0 to 1
    risk_level: RiskLevel
    
    # Component scores
    smart_money_score: float
    whale_score: float
    on_chain_score: float
    exchange_flow_score: float
    
    # Signal components
    components: Dict[str, Dict]  # Component name -> details
    
    # Trading recommendations
    position_size: float  # Recommended position size (0-1)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_horizon: str = "medium"  # "immediate", "short", "medium", "long"
    
    # Performance tracking
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    
    # Metadata
    metadata: Dict = field(default_factory=dict)


@dataclass
class SignalPerformance:
    """Track signal performance for validation"""
    signal_id: str
    timestamp: datetime
    predicted_direction: str
    actual_direction: Optional[str] = None
    predicted_magnitude: float = 0
    actual_magnitude: Optional[float] = None
    time_to_resolution: Optional[timedelta] = None
    profit_loss: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class DivergenceScorer:
    """
    Advanced scoring system for smart money divergence signals.
    
    Combines multiple indicators to generate high-confidence trading signals
    with proper risk management and position sizing recommendations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize divergence scorer"""
        self.config = config or self._default_config()
        
        # Component weights for scoring
        self.weights = {
            'smart_money': self.config.get('smart_money_weight', 0.3),
            'whale': self.config.get('whale_weight', 0.25),
            'on_chain': self.config.get('on_chain_weight', 0.25),
            'exchange_flow': self.config.get('exchange_flow_weight', 0.2)
        }
        
        # Signal thresholds
        self.thresholds = {
            'strong_buy': 0.7,
            'buy': 0.3,
            'weak_buy': 0.1,
            'weak_sell': -0.1,
            'sell': -0.3,
            'strong_sell': -0.7
        }
        
        # Risk parameters
        self.risk_params = {
            'max_position_size': 0.1,  # 10% max position
            'base_stop_loss': 0.05,  # 5% stop loss
            'base_take_profit': 0.15,  # 15% take profit
            'risk_adjustment_factor': 0.5
        }
        
        # ML models for scoring
        self.scoring_model = None
        self.confidence_model = None
        self.risk_model = None
        self._initialize_models()
        
        # Signal history and performance
        self.signal_history: List[CompositeSignal] = []
        self.performance_tracker: Dict[str, SignalPerformance] = {}
        
        # Statistical tracking
        self.signal_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'average_return': 0,
            'sharpe_ratio': 0,
            'win_rate': 0
        }
        
        # Adaptive learning parameters
        self.learning_rate = 0.01
        self.adaptation_window = 100  # Signals for adaptation
        
        logger.info("Divergence Scorer initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'smart_money_weight': 0.3,
            'whale_weight': 0.25,
            'on_chain_weight': 0.25,
            'exchange_flow_weight': 0.2,
            'min_confidence': 0.6,
            'min_components': 2,  # Minimum component signals required
            'lookback_window': 24,  # Hours for historical analysis
            'correlation_threshold': 0.7,  # For signal correlation
            'volatility_window': 168,  # Hours for volatility calculation
            'risk_free_rate': 0.02,  # Annual risk-free rate
            'max_correlation': 0.9,  # Maximum allowed correlation between components
            'adaptive_learning': True,
            'validation_period': 30  # Days for performance validation
        }
    
    def _initialize_models(self):
        """Initialize ML models for scoring"""
        try:
            # Try to load existing models
            self.scoring_model = joblib.load('models/scoring_model.pkl')
            self.confidence_model = joblib.load('models/confidence_model.pkl')
            self.risk_model = joblib.load('models/risk_model.pkl')
            logger.info("Loaded existing scoring models")
        except:
            # Create new models if not found
            self.scoring_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.confidence_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            self.risk_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            logger.info("Created new scoring models")
    
    async def generate_signal(
        self,
        symbol: str,
        smart_money_signals: List[Any],
        whale_signals: List[Any],
        on_chain_signals: List[Any],
        exchange_flow_signals: List[Any],
        price_data: pd.DataFrame
    ) -> Optional[CompositeSignal]:
        """
        Generate composite trading signal from all components.
        
        Args:
            symbol: Trading symbol
            smart_money_signals: Smart money divergence signals
            whale_signals: Whale tracking signals
            on_chain_signals: On-chain analysis signals
            exchange_flow_signals: Exchange flow signals
            price_data: Recent price data for context
            
        Returns:
            Composite trading signal or None if insufficient data
        """
        try:
            # Check minimum requirements
            active_components = sum([
                len(smart_money_signals) > 0,
                len(whale_signals) > 0,
                len(on_chain_signals) > 0,
                len(exchange_flow_signals) > 0
            ])
            
            if active_components < self.config['min_components']:
                return None
            
            # Calculate component scores
            smart_money_score = self._score_smart_money(smart_money_signals)
            whale_score = self._score_whale_activity(whale_signals)
            on_chain_score = self._score_on_chain(on_chain_signals)
            flow_score = self._score_exchange_flows(exchange_flow_signals)
            
            # Build component details
            components = {
                'smart_money': {
                    'score': smart_money_score,
                    'signal_count': len(smart_money_signals),
                    'signals': smart_money_signals[:3]  # Top 3 signals
                },
                'whale': {
                    'score': whale_score,
                    'signal_count': len(whale_signals),
                    'signals': whale_signals[:3]
                },
                'on_chain': {
                    'score': on_chain_score,
                    'signal_count': len(on_chain_signals),
                    'signals': on_chain_signals[:3]
                },
                'exchange_flow': {
                    'score': flow_score,
                    'signal_count': len(exchange_flow_signals),
                    'signals': exchange_flow_signals[:3]
                }
            }
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                smart_money_score, whale_score, on_chain_score, flow_score
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(components, price_data)
            
            if confidence < self.config['min_confidence']:
                return None
            
            # Determine signal type
            signal_type = self._determine_signal_type(composite_score)
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(components, price_data)
            
            # Generate trading recommendations
            position_size = self._calculate_position_size(
                composite_score, confidence, risk_level
            )
            
            stop_loss, take_profit = self._calculate_risk_targets(
                price_data, signal_type, risk_level
            )
            
            # Estimate expected performance
            expected_return, expected_volatility, sharpe_ratio = \
                self._estimate_performance(composite_score, confidence, price_data)
            
            # Determine time horizon
            time_horizon = self._determine_time_horizon(components)
            
            # Create composite signal
            signal = CompositeSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                composite_score=composite_score,
                confidence=confidence,
                risk_level=risk_level,
                smart_money_score=smart_money_score,
                whale_score=whale_score,
                on_chain_score=on_chain_score,
                exchange_flow_score=flow_score,
                components=components,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                time_horizon=time_horizon,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio,
                metadata={
                    'active_components': active_components,
                    'price_context': {
                        'current': price_data['close'].iloc[-1] if 'close' in price_data else None,
                        'change_24h': self._calculate_price_change(price_data, 24),
                        'volatility': self._calculate_volatility(price_data)
                    }
                }
            )
            
            # Store signal
            self.signal_history.append(signal)
            self.signal_stats['total_signals'] += 1
            
            # Track for performance
            self._track_signal_performance(signal)
            
            # Adaptive learning
            if self.config['adaptive_learning']:
                self._update_weights(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _score_smart_money(self, signals: List[Any]) -> float:
        """Score smart money divergence signals"""
        if not signals:
            return 0
        
        scores = []
        for signal in signals:
            # Extract signal strength and confidence
            strength = getattr(signal, 'strength', 0.5)
            confidence = getattr(signal, 'confidence', 0.5)
            
            # Determine direction
            divergence_type = getattr(signal, 'divergence_type', None)
            if divergence_type and 'BULLISH' in str(divergence_type).upper():
                direction = 1
            elif divergence_type and 'BEARISH' in str(divergence_type).upper():
                direction = -1
            else:
                direction = 0
            
            # Calculate weighted score
            score = direction * strength * confidence
            scores.append(score)
        
        # Return weighted average
        return np.mean(scores) if scores else 0
    
    def _score_whale_activity(self, signals: List[Any]) -> float:
        """Score whale tracking signals"""
        if not signals:
            return 0
        
        accumulation_score = 0
        distribution_score = 0
        
        for signal in signals:
            alert_type = getattr(signal, 'alert_type', '')
            severity = getattr(signal, 'severity', 'medium')
            
            # Weight by severity
            severity_weights = {
                'critical': 1.0,
                'high': 0.75,
                'medium': 0.5,
                'low': 0.25
            }
            weight = severity_weights.get(severity, 0.5)
            
            if 'accumulation' in alert_type.lower():
                accumulation_score += weight
            elif 'distribution' in alert_type.lower():
                distribution_score += weight
        
        # Normalize to -1 to 1
        total = accumulation_score + distribution_score
        if total > 0:
            return (accumulation_score - distribution_score) / total
        return 0
    
    def _score_on_chain(self, signals: List[Any]) -> float:
        """Score on-chain analysis signals"""
        if not signals:
            return 0
        
        bullish_score = 0
        bearish_score = 0
        
        for signal in signals:
            direction = getattr(signal, 'direction', 'neutral')
            strength = getattr(signal, 'strength', None)
            confidence = getattr(signal, 'confidence', 0.5)
            
            # Map strength enum to numeric value
            strength_map = {
                'VERY_STRONG': 1.0,
                'STRONG': 0.75,
                'MODERATE': 0.5,
                'WEAK': 0.25,
                'NEUTRAL': 0
            }
            
            if hasattr(strength, 'value'):
                strength_value = strength_map.get(strength.value.upper(), 0.5)
            else:
                strength_value = 0.5
            
            weighted_score = strength_value * confidence
            
            if direction == 'bullish':
                bullish_score += weighted_score
            elif direction == 'bearish':
                bearish_score += weighted_score
        
        # Calculate net score
        total = bullish_score + bearish_score
        if total > 0:
            return (bullish_score - bearish_score) / total
        return 0
    
    def _score_exchange_flows(self, signals: List[Any]) -> float:
        """Score exchange flow signals"""
        if not signals:
            return 0
        
        flow_scores = []
        
        for signal in signals:
            direction = getattr(signal, 'direction', 'neutral')
            strength = getattr(signal, 'strength', 0.5)
            confidence = getattr(signal, 'confidence', 0.5)
            
            if direction == 'bullish':
                score = strength * confidence
            elif direction == 'bearish':
                score = -strength * confidence
            else:
                score = 0
            
            # Weight by expected impact
            impact = getattr(signal, 'expected_impact', 'moderate')
            impact_weights = {
                'very_high': 1.2,
                'high': 1.0,
                'moderate': 0.7,
                'low': 0.4
            }
            score *= impact_weights.get(impact, 0.7)
            
            flow_scores.append(score)
        
        return np.mean(flow_scores) if flow_scores else 0
    
    def _calculate_composite_score(
        self,
        smart_money: float,
        whale: float,
        on_chain: float,
        flow: float
    ) -> float:
        """Calculate weighted composite score"""
        # Check for signal correlation
        scores = [smart_money, whale, on_chain, flow]
        
        # If all signals agree strongly, boost the score
        if all(s > 0.5 for s in scores):
            correlation_boost = 1.2
        elif all(s < -0.5 for s in scores):
            correlation_boost = 1.2
        else:
            correlation_boost = 1.0
        
        # Calculate weighted average
        composite = (
            self.weights['smart_money'] * smart_money +
            self.weights['whale'] * whale +
            self.weights['on_chain'] * on_chain +
            self.weights['exchange_flow'] * flow
        ) * correlation_boost
        
        # Clip to -1 to 1 range
        return np.clip(composite, -1, 1)
    
    def _calculate_confidence(
        self,
        components: Dict,
        price_data: pd.DataFrame
    ) -> float:
        """Calculate signal confidence"""
        confidence_factors = []
        
        # Component agreement
        scores = [c['score'] for c in components.values()]
        if all(s > 0 for s in scores) or all(s < 0 for s in scores):
            confidence_factors.append(0.9)  # High agreement
        elif np.std(scores) < 0.3:
            confidence_factors.append(0.7)  # Moderate agreement
        else:
            confidence_factors.append(0.5)  # Low agreement
        
        # Signal count
        total_signals = sum(c['signal_count'] for c in components.values())
        signal_confidence = min(total_signals / 10, 1.0)
        confidence_factors.append(signal_confidence)
        
        # Price trend alignment
        if len(price_data) > 24:
            price_trend = (price_data['close'].iloc[-1] - price_data['close'].iloc[-24]) / \
                         price_data['close'].iloc[-24]
            
            composite_direction = np.mean(scores)
            if (price_trend > 0 and composite_direction > 0) or \
               (price_trend < 0 and composite_direction < 0):
                confidence_factors.append(0.8)  # Trend alignment
            else:
                confidence_factors.append(0.4)  # Trend divergence
        
        # Use ML model if trained
        if self.confidence_model and hasattr(self.confidence_model, 'predict'):
            try:
                features = self._extract_ml_features(components, price_data)
                ml_confidence = self.confidence_model.predict([features])[0]
                confidence_factors.append(ml_confidence)
            except:
                pass
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_risk_level(
        self,
        components: Dict,
        price_data: pd.DataFrame
    ) -> RiskLevel:
        """Calculate risk level of signal"""
        risk_score = 0
        
        # Volatility risk
        volatility = self._calculate_volatility(price_data)
        if volatility > 0.1:  # >10% daily volatility
            risk_score += 2
        elif volatility > 0.05:  # >5% daily volatility
            risk_score += 1
        
        # Component disagreement risk
        scores = [c['score'] for c in components.values()]
        if np.std(scores) > 0.5:
            risk_score += 2
        elif np.std(scores) > 0.3:
            risk_score += 1
        
        # Signal strength risk (extreme signals are riskier)
        avg_score = np.mean([abs(s) for s in scores])
        if avg_score > 0.8:
            risk_score += 1
        
        # Map to risk level
        if risk_score >= 4:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MODERATE
        elif risk_score >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _determine_signal_type(self, composite_score: float) -> SignalType:
        """Determine signal type from composite score"""
        if composite_score >= self.thresholds['strong_buy']:
            return SignalType.STRONG_BUY
        elif composite_score >= self.thresholds['buy']:
            return SignalType.BUY
        elif composite_score >= self.thresholds['weak_buy']:
            return SignalType.WEAK_BUY
        elif composite_score <= self.thresholds['strong_sell']:
            return SignalType.STRONG_SELL
        elif composite_score <= self.thresholds['sell']:
            return SignalType.SELL
        elif composite_score <= self.thresholds['weak_sell']:
            return SignalType.WEAK_SELL
        else:
            return SignalType.NEUTRAL
    
    def _calculate_position_size(
        self,
        composite_score: float,
        confidence: float,
        risk_level: RiskLevel
    ) -> float:
        """Calculate recommended position size using Kelly Criterion variant"""
        # Base position from signal strength
        base_position = abs(composite_score) * self.risk_params['max_position_size']
        
        # Adjust for confidence
        confidence_adjusted = base_position * confidence
        
        # Adjust for risk
        risk_multipliers = {
            RiskLevel.VERY_LOW: 1.2,
            RiskLevel.LOW: 1.0,
            RiskLevel.MODERATE: 0.8,
            RiskLevel.HIGH: 0.5,
            RiskLevel.VERY_HIGH: 0.3
        }
        risk_adjusted = confidence_adjusted * risk_multipliers.get(risk_level, 0.8)
        
        # Apply Kelly Criterion modification
        # f = (p * b - q) / b
        # where p = probability of win, b = odds, q = probability of loss
        win_prob = (confidence + 1) / 2  # Convert confidence to probability
        loss_prob = 1 - win_prob
        odds = 2  # Assume 2:1 reward/risk ratio
        
        kelly_fraction = (win_prob * odds - loss_prob) / odds
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Final position size
        position_size = risk_adjusted * kelly_fraction * 4  # Scale back up
        
        return min(position_size, self.risk_params['max_position_size'])
    
    def _calculate_risk_targets(
        self,
        price_data: pd.DataFrame,
        signal_type: SignalType,
        risk_level: RiskLevel
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        if signal_type == SignalType.NEUTRAL:
            return None, None
        
        current_price = price_data['close'].iloc[-1] if 'close' in price_data else 0
        if current_price == 0:
            return None, None
        
        # Calculate ATR for dynamic stops
        atr = self._calculate_atr(price_data)
        
        # Risk level multipliers
        risk_multipliers = {
            RiskLevel.VERY_LOW: 0.5,
            RiskLevel.LOW: 0.75,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.HIGH: 1.5,
            RiskLevel.VERY_HIGH: 2.0
        }
        risk_mult = risk_multipliers.get(risk_level, 1.0)
        
        # Base stop loss
        if 'BUY' in signal_type.value.upper():
            stop_loss = current_price * (1 - self.risk_params['base_stop_loss'] * risk_mult)
            stop_loss = min(stop_loss, current_price - 2 * atr)
            
            # Take profit with risk/reward ratio
            risk = current_price - stop_loss
            take_profit = current_price + risk * 3  # 3:1 reward/risk
        else:
            stop_loss = current_price * (1 + self.risk_params['base_stop_loss'] * risk_mult)
            stop_loss = max(stop_loss, current_price + 2 * atr)
            
            risk = stop_loss - current_price
            take_profit = current_price - risk * 3
        
        return stop_loss, take_profit
    
    def _estimate_performance(
        self,
        composite_score: float,
        confidence: float,
        price_data: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """Estimate expected performance metrics"""
        # Historical volatility
        volatility = self._calculate_volatility(price_data)
        
        # Expected return based on signal strength and confidence
        base_return = composite_score * 0.15  # 15% max expected return
        expected_return = base_return * confidence
        
        # Expected volatility (higher for stronger signals)
        expected_volatility = volatility * (1 + abs(composite_score) * 0.5)
        
        # Sharpe ratio
        risk_free_rate = self.config['risk_free_rate'] / 252  # Daily risk-free rate
        if expected_volatility > 0:
            sharpe_ratio = (expected_return - risk_free_rate) / expected_volatility
        else:
            sharpe_ratio = 0
        
        return expected_return, expected_volatility, sharpe_ratio
    
    def _determine_time_horizon(self, components: Dict) -> str:
        """Determine signal time horizon"""
        # Check for immediate signals
        for comp in components.values():
            for signal in comp.get('signals', []):
                if hasattr(signal, 'time_horizon'):
                    if signal.time_horizon == 'immediate':
                        return 'immediate'
        
        # Check average time horizons
        horizons = []
        for comp in components.values():
            for signal in comp.get('signals', []):
                if hasattr(signal, 'time_horizon'):
                    horizons.append(signal.time_horizon)
        
        if not horizons:
            return 'medium'
        
        # Most common horizon
        from collections import Counter
        most_common = Counter(horizons).most_common(1)[0][0]
        return most_common
    
    def _calculate_volatility(self, price_data: pd.DataFrame) -> float:
        """Calculate price volatility"""
        if 'close' not in price_data or len(price_data) < 2:
            return 0
        
        returns = price_data['close'].pct_change().dropna()
        if len(returns) > 0:
            return returns.std()
        return 0
    
    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if not all(col in price_data for col in ['high', 'low', 'close']):
            return 0
        
        if len(price_data) < period:
            return 0
        
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else 0
    
    def _calculate_price_change(self, price_data: pd.DataFrame, hours: int) -> float:
        """Calculate price change over specified hours"""
        if 'close' not in price_data or len(price_data) < hours:
            return 0
        
        current = price_data['close'].iloc[-1]
        previous = price_data['close'].iloc[-hours]
        
        if previous > 0:
            return (current - previous) / previous
        return 0
    
    def _extract_ml_features(self, components: Dict, price_data: pd.DataFrame) -> np.ndarray:
        """Extract features for ML models"""
        features = []
        
        # Component scores
        for comp_name in ['smart_money', 'whale', 'on_chain', 'exchange_flow']:
            features.append(components.get(comp_name, {}).get('score', 0))
        
        # Signal counts
        for comp in components.values():
            features.append(comp.get('signal_count', 0))
        
        # Price features
        features.append(self._calculate_volatility(price_data))
        features.append(self._calculate_price_change(price_data, 24))
        features.append(self._calculate_price_change(price_data, 168))
        
        return np.array(features)
    
    def _track_signal_performance(self, signal: CompositeSignal):
        """Track signal for performance validation"""
        signal_id = f"{signal.symbol}_{signal.timestamp.isoformat()}"
        
        performance = SignalPerformance(
            signal_id=signal_id,
            timestamp=signal.timestamp,
            predicted_direction='bullish' if signal.composite_score > 0 else 'bearish',
            predicted_magnitude=abs(signal.composite_score),
            metadata={
                'signal_type': signal.signal_type.value,
                'confidence': signal.confidence,
                'risk_level': signal.risk_level.value
            }
        )
        
        self.performance_tracker[signal_id] = performance
    
    def _update_weights(self, signal: CompositeSignal):
        """Update component weights based on performance"""
        if len(self.signal_history) < self.adaptation_window:
            return
        
        # Get recent signals with known outcomes
        recent_signals = self.signal_history[-self.adaptation_window:]
        
        # Calculate component contributions to successful signals
        component_performance = {
            'smart_money': [],
            'whale': [],
            'on_chain': [],
            'exchange_flow': []
        }
        
        for hist_signal in recent_signals:
            signal_id = f"{hist_signal.symbol}_{hist_signal.timestamp.isoformat()}"
            if signal_id in self.performance_tracker:
                perf = self.performance_tracker[signal_id]
                if perf.actual_direction is not None:
                    # Check if prediction was correct
                    correct = (perf.predicted_direction == perf.actual_direction)
                    
                    # Attribute success/failure to components
                    for comp_name in component_performance.keys():
                        comp_score = getattr(hist_signal, f"{comp_name}_score", 0)
                        if comp_score != 0:
                            contribution = 1 if correct else -1
                            component_performance[comp_name].append(contribution * abs(comp_score))
        
        # Update weights based on performance
        for comp_name, performances in component_performance.items():
            if performances:
                avg_performance = np.mean(performances)
                # Adjust weight
                current_weight = self.weights[comp_name]
                adjustment = self.learning_rate * avg_performance
                new_weight = max(0.1, min(0.4, current_weight + adjustment))
                self.weights[comp_name] = new_weight
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for comp_name in self.weights:
            self.weights[comp_name] /= total_weight
    
    def validate_signal(
        self,
        signal_id: str,
        actual_direction: str,
        actual_magnitude: float,
        profit_loss: float
    ):
        """Validate a signal with actual outcome"""
        if signal_id in self.performance_tracker:
            perf = self.performance_tracker[signal_id]
            perf.actual_direction = actual_direction
            perf.actual_magnitude = actual_magnitude
            perf.profit_loss = profit_loss
            perf.time_to_resolution = datetime.now() - perf.timestamp
            
            # Update statistics
            if perf.predicted_direction == actual_direction:
                self.signal_stats['successful_signals'] += 1
            else:
                self.signal_stats['failed_signals'] += 1
            
            # Update win rate
            total = self.signal_stats['successful_signals'] + self.signal_stats['failed_signals']
            if total > 0:
                self.signal_stats['win_rate'] = self.signal_stats['successful_signals'] / total
            
            # Update average return
            returns = [p.profit_loss for p in self.performance_tracker.values() 
                      if p.profit_loss is not None]
            if returns:
                self.signal_stats['average_return'] = np.mean(returns)
                
                # Calculate Sharpe ratio
                if len(returns) > 1:
                    return_std = np.std(returns)
                    if return_std > 0:
                        self.signal_stats['sharpe_ratio'] = \
                            (self.signal_stats['average_return'] - self.config['risk_free_rate']/252) / return_std
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        report = {
            'statistics': self.signal_stats,
            'component_weights': self.weights,
            'recent_signals': len(self.signal_history),
            'performance_by_type': {},
            'performance_by_risk': {}
        }
        
        # Analyze by signal type
        for signal_type in SignalType:
            type_perfs = [
                p for sid, p in self.performance_tracker.items()
                if p.metadata.get('signal_type') == signal_type.value and p.profit_loss is not None
            ]
            if type_perfs:
                report['performance_by_type'][signal_type.value] = {
                    'count': len(type_perfs),
                    'avg_return': np.mean([p.profit_loss for p in type_perfs]),
                    'win_rate': sum(1 for p in type_perfs if p.profit_loss > 0) / len(type_perfs)
                }
        
        # Analyze by risk level
        for risk_level in RiskLevel:
            risk_perfs = [
                p for sid, p in self.performance_tracker.items()
                if p.metadata.get('risk_level') == risk_level.value and p.profit_loss is not None
            ]
            if risk_perfs:
                report['performance_by_risk'][risk_level.value] = {
                    'count': len(risk_perfs),
                    'avg_return': np.mean([p.profit_loss for p in risk_perfs]),
                    'win_rate': sum(1 for p in risk_perfs if p.profit_loss > 0) / len(risk_perfs)
                }
        
        return report
    
    def save_models(self, path: str = 'models/'):
        """Save ML models to disk"""
        try:
            joblib.dump(self.scoring_model, f'{path}scoring_model.pkl')
            joblib.dump(self.confidence_model, f'{path}confidence_model.pkl')
            joblib.dump(self.risk_model, f'{path}risk_model.pkl')
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")


class DivergenceScorerRL:
    """RL integration for divergence scoring"""
    
    def __init__(self, scorer: DivergenceScorer):
        self.scorer = scorer
    
    def get_features(self, signal: Optional[CompositeSignal]) -> np.ndarray:
        """Extract RL features from composite signal"""
        if not signal:
            return np.zeros(20)
        
        features = [
            # Core scores
            signal.composite_score,
            signal.confidence,
            
            # Component scores
            signal.smart_money_score,
            signal.whale_score,
            signal.on_chain_score,
            signal.exchange_flow_score,
            
            # Risk metrics
            1.0 if signal.risk_level == RiskLevel.VERY_LOW else 0,
            1.0 if signal.risk_level == RiskLevel.LOW else 0,
            1.0 if signal.risk_level == RiskLevel.MODERATE else 0,
            1.0 if signal.risk_level == RiskLevel.HIGH else 0,
            1.0 if signal.risk_level == RiskLevel.VERY_HIGH else 0,
            
            # Trading parameters
            signal.position_size,
            
            # Performance estimates
            signal.expected_return if signal.expected_return else 0,
            signal.expected_volatility if signal.expected_volatility else 0,
            signal.sharpe_ratio if signal.sharpe_ratio else 0,
            
            # Time horizon encoding
            1.0 if signal.time_horizon == 'immediate' else 0,
            1.0 if signal.time_horizon == 'short' else 0,
            1.0 if signal.time_horizon == 'medium' else 0,
            1.0 if signal.time_horizon == 'long' else 0,
            
            # Signal type strength
            abs(signal.composite_score)
        ]
        
        return np.array(features)
    
    def get_action_recommendations(self, signal: Optional[CompositeSignal]) -> Dict:
        """Get RL action recommendations from signal"""
        if not signal:
            return {
                'action': 'hold',
                'confidence': 0,
                'position_size': 0
            }
        
        # Map signal type to action
        action_map = {
            SignalType.STRONG_BUY: 'strong_buy',
            SignalType.BUY: 'buy',
            SignalType.WEAK_BUY: 'weak_buy',
            SignalType.NEUTRAL: 'hold',
            SignalType.WEAK_SELL: 'weak_sell',
            SignalType.SELL: 'sell',
            SignalType.STRONG_SELL: 'strong_sell'
        }
        
        return {
            'action': action_map.get(signal.signal_type, 'hold'),
            'confidence': signal.confidence,
            'position_size': signal.position_size,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'risk_level': signal.risk_level.value
        }