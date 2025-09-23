"""
Market Regime Detection System

This module implements sophisticated market regime detection to identify different
market conditions and select the most appropriate trading agent for each regime.

Regimes detected:
- BULL: Strong upward trending market
- BEAR: Strong downward trending market  
- SIDEWAYS: Range-bound, low volatility market
- HIGH_VOLATILITY: High volatility, uncertain direction
- MEAN_REVERTING: Mean reverting patterns
- MOMENTUM: Strong directional momentum

The system uses multiple indicators and machine learning to classify regimes
in real-time and adapt agent selection accordingly.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import json
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull_market"
    BEAR = "bear_market"
    SIDEWAYS = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    MEAN_REVERTING = "mean_reverting"
    MOMENTUM = "momentum_trending"
    UNKNOWN = "unknown"


@dataclass
class RegimeFeatures:
    """Features used for regime detection"""
    
    # Price-based features
    returns_1d: float = 0.0
    returns_5d: float = 0.0
    returns_20d: float = 0.0
    
    # Volatility features
    volatility_short: float = 0.0
    volatility_long: float = 0.0
    volatility_ratio: float = 1.0
    
    # Trend features
    trend_strength: float = 0.0
    trend_direction: float = 0.0
    trend_consistency: float = 0.0
    
    # Mean reversion features
    distance_from_ma: float = 0.0
    rsi: float = 50.0
    bollinger_position: float = 0.0
    
    # Momentum features
    momentum_short: float = 0.0
    momentum_long: float = 0.0
    momentum_acceleration: float = 0.0
    
    # Volume features (if available)
    volume_ratio: float = 1.0
    volume_trend: float = 0.0
    
    # Market structure features
    higher_highs: bool = False
    higher_lows: bool = False
    lower_highs: bool = False
    lower_lows: bool = False
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML models"""
        return np.array([
            self.returns_1d, self.returns_5d, self.returns_20d,
            self.volatility_short, self.volatility_long, self.volatility_ratio,
            self.trend_strength, self.trend_direction, self.trend_consistency,
            self.distance_from_ma, self.rsi, self.bollinger_position,
            self.momentum_short, self.momentum_long, self.momentum_acceleration,
            self.volume_ratio, self.volume_trend,
            float(self.higher_highs), float(self.higher_lows),
            float(self.lower_highs), float(self.lower_lows)
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get feature names for ML models"""
        return [
            'returns_1d', 'returns_5d', 'returns_20d',
            'volatility_short', 'volatility_long', 'volatility_ratio',
            'trend_strength', 'trend_direction', 'trend_consistency',
            'distance_from_ma', 'rsi', 'bollinger_position',
            'momentum_short', 'momentum_long', 'momentum_acceleration',
            'volume_ratio', 'volume_trend',
            'higher_highs', 'higher_lows', 'lower_highs', 'lower_lows'
        ]


@dataclass
class RegimeDetectionConfig:
    """Configuration for regime detection"""
    
    # Lookback periods
    short_window: int = 5
    medium_window: int = 20
    long_window: int = 60
    
    # Volatility thresholds
    high_volatility_threshold: float = 0.25
    low_volatility_threshold: float = 0.10
    
    # Trend thresholds
    strong_trend_threshold: float = 0.05  # 5% move
    trend_consistency_threshold: float = 0.7
    
    # Mean reversion thresholds
    mean_reversion_threshold: float = 0.15
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    # Momentum thresholds
    momentum_threshold: float = 0.03
    acceleration_threshold: float = 0.01
    
    # Model parameters
    enable_ml_detection: bool = True
    model_retrain_frequency: int = 1000  # Steps
    model_confidence_threshold: float = 0.6
    
    # Regime persistence
    regime_persistence_periods: int = 3
    regime_change_threshold: float = 0.8  # Confidence threshold for regime change
    
    # Feature scaling
    enable_feature_scaling: bool = True
    
    # Logging
    debug_mode: bool = False


class FeatureCalculator:
    """Calculate technical features for regime detection"""
    
    def __init__(self, config: RegimeDetectionConfig):
        self.config = config
    
    def calculate_features(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> RegimeFeatures:
        """Calculate all features from price and volume data"""
        
        if len(prices) < self.config.long_window:
            logger.warning(f"Insufficient data for feature calculation: {len(prices)} < {self.config.long_window}")
            return RegimeFeatures()
        
        features = RegimeFeatures()
        
        # Calculate returns
        features.returns_1d = self._calculate_return(prices, 1)
        features.returns_5d = self._calculate_return(prices, self.config.short_window)
        features.returns_20d = self._calculate_return(prices, self.config.medium_window)
        
        # Calculate volatility
        features.volatility_short = self._calculate_volatility(prices, self.config.short_window)
        features.volatility_long = self._calculate_volatility(prices, self.config.medium_window)
        features.volatility_ratio = features.volatility_short / max(features.volatility_long, 0.001)
        
        # Calculate trend features
        trend_metrics = self._calculate_trend_features(prices)
        features.trend_strength = trend_metrics['strength']
        features.trend_direction = trend_metrics['direction']
        features.trend_consistency = trend_metrics['consistency']
        
        # Calculate mean reversion features
        mean_reversion = self._calculate_mean_reversion_features(prices)
        features.distance_from_ma = mean_reversion['distance_from_ma']
        features.rsi = mean_reversion['rsi']
        features.bollinger_position = mean_reversion['bollinger_position']
        
        # Calculate momentum features
        momentum = self._calculate_momentum_features(prices)
        features.momentum_short = momentum['short']
        features.momentum_long = momentum['long']
        features.momentum_acceleration = momentum['acceleration']
        
        # Calculate volume features
        if volumes is not None:
            volume_metrics = self._calculate_volume_features(volumes)
            features.volume_ratio = volume_metrics['ratio']
            features.volume_trend = volume_metrics['trend']
        
        # Calculate market structure
        structure = self._calculate_market_structure(prices)
        features.higher_highs = structure['higher_highs']
        features.higher_lows = structure['higher_lows']
        features.lower_highs = structure['lower_highs']
        features.lower_lows = structure['lower_lows']
        
        return features
    
    def _calculate_return(self, prices: np.ndarray, periods: int) -> float:
        """Calculate return over specified periods"""
        if len(prices) < periods + 1:
            return 0.0
        
        start_price = prices[-(periods + 1)]
        end_price = prices[-1]
        
        if start_price > 0:
            return (end_price - start_price) / start_price
        return 0.0
    
    def _calculate_volatility(self, prices: np.ndarray, window: int) -> float:
        """Calculate rolling volatility"""
        if len(prices) < window + 1:
            return 0.0
        
        returns = np.diff(prices[-window-1:]) / prices[-window-1:-1]
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_trend_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate trend strength, direction, and consistency"""
        
        # Simple trend slope
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[-self.config.medium_window:], 
                                                                       prices[-self.config.medium_window:])
        
        # Normalize slope by average price
        avg_price = np.mean(prices[-self.config.medium_window:])
        normalized_slope = slope / max(avg_price, 0.001)
        
        # Trend strength (R-squared)
        trend_strength = r_value ** 2
        
        # Trend direction
        trend_direction = np.sign(normalized_slope)
        
        # Trend consistency (percentage of moves in trend direction)
        returns = np.diff(prices[-self.config.medium_window:])
        consistent_moves = np.sum(np.sign(returns) == trend_direction) / len(returns) if len(returns) > 0 else 0
        
        return {
            'strength': trend_strength,
            'direction': trend_direction,
            'consistency': consistent_moves
        }
    
    def _calculate_mean_reversion_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate mean reversion indicators"""
        
        # Distance from moving average
        ma = np.mean(prices[-self.config.medium_window:])
        current_price = prices[-1]
        distance_from_ma = (current_price - ma) / max(ma, 0.001)
        
        # RSI calculation
        rsi = self._calculate_rsi(prices, self.config.medium_window // 2)
        
        # Bollinger Bands position
        std = np.std(prices[-self.config.medium_window:])
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std
        
        if upper_band > lower_band:
            bollinger_position = (current_price - lower_band) / (upper_band - lower_band)
        else:
            bollinger_position = 0.5
        
        return {
            'distance_from_ma': distance_from_ma,
            'rsi': rsi,
            'bollinger_position': bollinger_position
        }
    
    def _calculate_rsi(self, prices: np.ndarray, window: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < window + 1:
            return 50.0
        
        deltas = np.diff(prices[-window-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_momentum_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate momentum indicators"""
        
        # Short-term momentum
        short_momentum = self._calculate_return(prices, self.config.short_window)
        
        # Long-term momentum
        long_momentum = self._calculate_return(prices, self.config.medium_window)
        
        # Momentum acceleration (change in momentum)
        if len(prices) >= self.config.medium_window + self.config.short_window:
            prev_momentum = self._calculate_return(prices[:-self.config.short_window], self.config.short_window)
            momentum_acceleration = short_momentum - prev_momentum
        else:
            momentum_acceleration = 0.0
        
        return {
            'short': short_momentum,
            'long': long_momentum,
            'acceleration': momentum_acceleration
        }
    
    def _calculate_volume_features(self, volumes: np.ndarray) -> Dict[str, float]:
        """Calculate volume-based features"""
        
        # Volume ratio (current vs average)
        avg_volume = np.mean(volumes[-self.config.medium_window:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / max(avg_volume, 1)
        
        # Volume trend
        if len(volumes) >= self.config.medium_window:
            x = np.arange(len(volumes[-self.config.medium_window:]))
            slope, _, _, _, _ = stats.linregress(x, volumes[-self.config.medium_window:])
            volume_trend = slope / max(avg_volume, 1)
        else:
            volume_trend = 0.0
        
        return {
            'ratio': volume_ratio,
            'trend': volume_trend
        }
    
    def _calculate_market_structure(self, prices: np.ndarray) -> Dict[str, bool]:
        """Calculate market structure patterns"""
        
        if len(prices) < self.config.short_window * 2:
            return {
                'higher_highs': False,
                'higher_lows': False,
                'lower_highs': False,
                'lower_lows': False
            }
        
        # Get recent highs and lows
        recent_high = np.max(prices[-self.config.short_window:])
        recent_low = np.min(prices[-self.config.short_window:])
        
        previous_high = np.max(prices[-self.config.short_window*2:-self.config.short_window])
        previous_low = np.min(prices[-self.config.short_window*2:-self.config.short_window])
        
        return {
            'higher_highs': recent_high > previous_high,
            'higher_lows': recent_low > previous_low,
            'lower_highs': recent_high < previous_high,
            'lower_lows': recent_low < previous_low
        }


class RuleBasedRegimeDetector:
    """Rule-based regime detection using technical indicators"""
    
    def __init__(self, config: RegimeDetectionConfig):
        self.config = config
        self.feature_calculator = FeatureCalculator(config)
    
    def detect_regime(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Tuple[MarketRegime, float]:
        """Detect market regime using rule-based approach"""
        
        features = self.feature_calculator.calculate_features(prices, volumes)
        
        # Rule-based classification
        regime_scores = {
            MarketRegime.BULL: self._score_bull_market(features),
            MarketRegime.BEAR: self._score_bear_market(features),
            MarketRegime.SIDEWAYS: self._score_sideways_market(features),
            MarketRegime.HIGH_VOLATILITY: self._score_high_volatility(features),
            MarketRegime.MEAN_REVERTING: self._score_mean_reverting(features),
            MarketRegime.MOMENTUM: self._score_momentum(features)
        }
        
        # Get regime with highest score
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]
        
        if self.config.debug_mode:
            logger.info(f"Regime scores: {regime_scores}")
            logger.info(f"Detected regime: {best_regime.value} (confidence: {confidence:.2f})")
        
        return best_regime, confidence
    
    def _score_bull_market(self, features: RegimeFeatures) -> float:
        """Score bull market conditions"""
        score = 0.0
        
        # Positive returns
        if features.returns_5d > 0:
            score += 0.3
        if features.returns_20d > self.config.strong_trend_threshold:
            score += 0.4
        
        # Strong upward trend
        if features.trend_direction > 0 and features.trend_strength > 0.5:
            score += 0.3
        
        # Market structure
        if features.higher_highs and features.higher_lows:
            score += 0.4
        
        # Low volatility relative to returns
        if features.volatility_short < self.config.high_volatility_threshold:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_bear_market(self, features: RegimeFeatures) -> float:
        """Score bear market conditions"""
        score = 0.0
        
        # Negative returns
        if features.returns_5d < 0:
            score += 0.3
        if features.returns_20d < -self.config.strong_trend_threshold:
            score += 0.4
        
        # Strong downward trend
        if features.trend_direction < 0 and features.trend_strength > 0.5:
            score += 0.3
        
        # Market structure
        if features.lower_highs and features.lower_lows:
            score += 0.4
        
        # Moderate volatility
        if self.config.low_volatility_threshold < features.volatility_short < self.config.high_volatility_threshold:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_sideways_market(self, features: RegimeFeatures) -> float:
        """Score sideways/range-bound market conditions"""
        score = 0.0
        
        # Low returns
        if abs(features.returns_20d) < self.config.strong_trend_threshold / 2:
            score += 0.4
        
        # Weak trend
        if features.trend_strength < 0.3:
            score += 0.3
        
        # Low volatility
        if features.volatility_short < self.config.low_volatility_threshold:
            score += 0.3
        
        # Mean reverting behavior
        if 0.3 < features.bollinger_position < 0.7:  # Middle of Bollinger Bands
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_high_volatility(self, features: RegimeFeatures) -> float:
        """Score high volatility conditions"""
        score = 0.0
        
        # High volatility
        if features.volatility_short > self.config.high_volatility_threshold:
            score += 0.5
        
        # Increasing volatility
        if features.volatility_ratio > 1.5:
            score += 0.3
        
        # Inconsistent trend
        if features.trend_consistency < 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_mean_reverting(self, features: RegimeFeatures) -> float:
        """Score mean reverting conditions"""
        score = 0.0
        
        # Distance from mean
        if abs(features.distance_from_ma) > self.config.mean_reversion_threshold:
            score += 0.4
        
        # Extreme RSI
        if features.rsi > self.config.rsi_overbought or features.rsi < self.config.rsi_oversold:
            score += 0.3
        
        # Bollinger Band extremes
        if features.bollinger_position > 0.8 or features.bollinger_position < 0.2:
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_momentum(self, features: RegimeFeatures) -> float:
        """Score momentum conditions"""
        score = 0.0
        
        # Strong momentum
        if abs(features.momentum_short) > self.config.momentum_threshold:
            score += 0.4
        
        # Accelerating momentum
        if abs(features.momentum_acceleration) > self.config.acceleration_threshold:
            score += 0.3
        
        # Consistent trend
        if features.trend_consistency > self.config.trend_consistency_threshold:
            score += 0.3
        
        return min(score, 1.0)


class MLRegimeDetector:
    """Machine learning-based regime detection"""
    
    def __init__(self, config: RegimeDetectionConfig):
        self.config = config
        self.feature_calculator = FeatureCalculator(config)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler() if config.enable_feature_scaling else None
        self.is_trained = False
        self.feature_history = []
        self.regime_history = []
        self.last_retrain = 0
    
    def add_training_data(self, prices: np.ndarray, true_regime: MarketRegime, 
                         volumes: Optional[np.ndarray] = None):
        """Add labeled training data"""
        features = self.feature_calculator.calculate_features(prices, volumes)
        
        self.feature_history.append(features.to_array())
        self.regime_history.append(true_regime.value)
        
        # Retrain if enough new data
        if len(self.feature_history) - self.last_retrain >= self.config.model_retrain_frequency:
            self.train_model()
    
    def train_model(self):
        """Train the ML model"""
        if len(self.feature_history) < 50:  # Need minimum data
            logger.warning("Insufficient training data for ML model")
            return
        
        try:
            X = np.array(self.feature_history)
            y = np.array(self.regime_history)
            
            # Scale features if enabled
            if self.scaler:
                X = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X, y)
            self.is_trained = True
            self.last_retrain = len(self.feature_history)
            
            # Log performance
            y_pred = self.model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            logger.info(f"ML regime detector trained with accuracy: {accuracy:.3f}")
            
            if self.config.debug_mode:
                logger.info(f"Classification report:\n{classification_report(y, y_pred)}")
        
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
    
    def predict_regime(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Tuple[MarketRegime, float]:
        """Predict regime using ML model"""
        
        if not self.is_trained:
            logger.warning("ML model not trained, returning unknown regime")
            return MarketRegime.UNKNOWN, 0.0
        
        try:
            features = self.feature_calculator.calculate_features(prices, volumes)
            X = features.to_array().reshape(1, -1)
            
            # Scale features if enabled
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Predict
            regime_pred = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = np.max(probabilities)
            
            # Convert string back to enum
            try:
                regime = MarketRegime(regime_pred)
            except ValueError:
                regime = MarketRegime.UNKNOWN
                confidence = 0.0
            
            return regime, confidence
        
        except Exception as e:
            logger.error(f"Error predicting regime: {e}")
            return MarketRegime.UNKNOWN, 0.0


class MarketRegimeDetector:
    """Main market regime detector combining rule-based and ML approaches"""
    
    def __init__(self, config: Optional[RegimeDetectionConfig] = None):
        self.config = config or RegimeDetectionConfig()
        
        # Initialize detectors
        self.rule_detector = RuleBasedRegimeDetector(self.config)
        self.ml_detector = MLRegimeDetector(self.config) if self.config.enable_ml_detection else None
        
        # Regime tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history = []
        self.confidence_history = []
        self.regime_persistence_count = 0
        
        logger.info("Market regime detector initialized")
    
    def detect_regime(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Tuple[MarketRegime, float, Dict[str, Any]]:
        """
        Detect current market regime
        
        Args:
            prices: Price data array
            volumes: Volume data array (optional)
            
        Returns:
            Tuple of (regime, confidence, additional_info)
        """
        
        # Rule-based detection
        rule_regime, rule_confidence = self.rule_detector.detect_regime(prices, volumes)
        
        # ML-based detection
        ml_regime = MarketRegime.UNKNOWN
        ml_confidence = 0.0
        
        if self.ml_detector and self.ml_detector.is_trained:
            ml_regime, ml_confidence = self.ml_detector.predict_regime(prices, volumes)
        
        # Combine predictions
        final_regime, final_confidence = self._combine_predictions(
            rule_regime, rule_confidence,
            ml_regime, ml_confidence
        )
        
        # Apply regime persistence filtering
        stable_regime, stable_confidence = self._apply_regime_persistence(final_regime, final_confidence)
        
        # Update tracking
        self.regime_history.append({
            'timestamp': datetime.now().isoformat(),
            'regime': stable_regime.value,
            'confidence': stable_confidence,
            'rule_regime': rule_regime.value,
            'rule_confidence': rule_confidence,
            'ml_regime': ml_regime.value,
            'ml_confidence': ml_confidence
        })
        
        # Keep history manageable
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-500:]
        
        # Additional info
        info = {
            'rule_based': {'regime': rule_regime.value, 'confidence': rule_confidence},
            'ml_based': {'regime': ml_regime.value, 'confidence': ml_confidence},
            'combined': {'regime': final_regime.value, 'confidence': final_confidence},
            'stable': {'regime': stable_regime.value, 'confidence': stable_confidence},
            'regime_persistence_count': self.regime_persistence_count,
            'history_length': len(self.regime_history)
        }
        
        return stable_regime, stable_confidence, info
    
    def _combine_predictions(self, rule_regime: MarketRegime, rule_confidence: float,
                           ml_regime: MarketRegime, ml_confidence: float) -> Tuple[MarketRegime, float]:
        """Combine rule-based and ML predictions"""
        
        # If ML is not available or has low confidence, use rule-based
        if ml_regime == MarketRegime.UNKNOWN or ml_confidence < self.config.model_confidence_threshold:
            return rule_regime, rule_confidence
        
        # If both agree, use higher confidence
        if rule_regime == ml_regime:
            combined_confidence = max(rule_confidence, ml_confidence)
            return rule_regime, combined_confidence
        
        # If they disagree, use the one with higher confidence
        if rule_confidence > ml_confidence:
            return rule_regime, rule_confidence
        else:
            return ml_regime, ml_confidence
    
    def _apply_regime_persistence(self, regime: MarketRegime, confidence: float) -> Tuple[MarketRegime, float]:
        """Apply regime persistence to reduce regime switching noise"""
        
        # If this is the same regime as current, increase persistence
        if regime == self.current_regime:
            self.regime_persistence_count += 1
            return regime, min(confidence + 0.1 * self.regime_persistence_count, 1.0)
        
        # If different regime but confidence is high enough, change regime
        if confidence >= self.config.regime_change_threshold:
            self.current_regime = regime
            self.regime_persistence_count = 1
            return regime, confidence
        
        # If not confident enough to change, stick with current regime
        if self.current_regime != MarketRegime.UNKNOWN:
            # Gradually decrease confidence in current regime
            current_confidence = max(0.1, confidence - 0.1 * self.regime_persistence_count)
            return self.current_regime, current_confidence
        
        # If no current regime, accept new regime
        self.current_regime = regime
        self.regime_persistence_count = 1
        return regime, confidence
    
    def add_training_data(self, prices: np.ndarray, true_regime: MarketRegime,
                         volumes: Optional[np.ndarray] = None):
        """Add labeled training data for ML model"""
        if self.ml_detector:
            self.ml_detector.add_training_data(prices, true_regime, volumes)
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime detection statistics"""
        if not self.regime_history:
            return {}
        
        # Recent regime distribution
        recent_regimes = [r['regime'] for r in self.regime_history[-100:]]
        regime_counts = {}
        for regime in recent_regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Average confidence
        recent_confidence = [r['confidence'] for r in self.regime_history[-50:]]
        avg_confidence = np.mean(recent_confidence) if recent_confidence else 0
        
        # Regime stability (how often regime changes)
        regime_changes = 0
        for i in range(1, min(len(self.regime_history), 50)):
            if self.regime_history[-i]['regime'] != self.regime_history[-i-1]['regime']:
                regime_changes += 1
        
        stability = 1 - (regime_changes / min(len(self.regime_history), 49)) if len(self.regime_history) > 1 else 1
        
        return {
            'current_regime': self.current_regime.value,
            'regime_persistence_count': self.regime_persistence_count,
            'recent_regime_distribution': regime_counts,
            'average_confidence': avg_confidence,
            'regime_stability': stability,
            'total_detections': len(self.regime_history),
            'ml_model_trained': self.ml_detector.is_trained if self.ml_detector else False
        }
    
    def save_state(self, filepath: str):
        """Save detector state"""
        state = {
            'config': {
                'short_window': self.config.short_window,
                'medium_window': self.config.medium_window,
                'long_window': self.config.long_window,
                'high_volatility_threshold': self.config.high_volatility_threshold,
                'low_volatility_threshold': self.config.low_volatility_threshold
            },
            'current_regime': self.current_regime.value,
            'regime_persistence_count': self.regime_persistence_count,
            'regime_history': self.regime_history[-100:],  # Save last 100 entries
            'statistics': self.get_regime_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Regime detector state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load detector state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_regime = MarketRegime(state['current_regime'])
            self.regime_persistence_count = state['regime_persistence_count']
            self.regime_history = state.get('regime_history', [])
            
            logger.info(f"Regime detector state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")


# Agent-to-regime mapping for ensemble selection
REGIME_AGENT_MAPPING = {
    MarketRegime.BULL: ['aggressive', 'momentum'],
    MarketRegime.BEAR: ['conservative', 'contrarian'],
    MarketRegime.SIDEWAYS: ['contrarian', 'balanced'],
    MarketRegime.HIGH_VOLATILITY: ['conservative', 'balanced'],
    MarketRegime.MEAN_REVERTING: ['contrarian', 'balanced'],
    MarketRegime.MOMENTUM: ['aggressive', 'momentum'],
    MarketRegime.UNKNOWN: ['balanced']
}


def get_optimal_agents_for_regime(regime: MarketRegime) -> List[str]:
    """Get list of optimal agent types for a given regime"""
    return REGIME_AGENT_MAPPING.get(regime, ['balanced'])


if __name__ == "__main__":
    # Example usage and testing
    
    # Generate sample price data
    np.random.seed(42)
    n_points = 200
    
    # Bull market simulation
    bull_trend = np.cumsum(np.random.normal(0.001, 0.02, n_points))
    bull_prices = 100 * np.exp(bull_trend)
    
    # Create detector
    config = RegimeDetectionConfig(debug_mode=True)
    detector = MarketRegimeDetector(config)
    
    # Test regime detection
    regime, confidence, info = detector.detect_regime(bull_prices[-60:])
    
    print(f"Detected regime: {regime.value}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Optimal agents: {get_optimal_agents_for_regime(regime)}")
    print(f"Detection info: {info}")
    
    # Test with different market conditions
    bear_trend = np.cumsum(np.random.normal(-0.001, 0.025, n_points))
    bear_prices = 100 * np.exp(bear_trend)
    
    regime2, confidence2, info2 = detector.detect_regime(bear_prices[-60:])
    print(f"\nBear market test:")
    print(f"Detected regime: {regime2.value}")
    print(f"Optimal agents: {get_optimal_agents_for_regime(regime2)}")
    
    # Get statistics
    stats = detector.get_regime_statistics()
    print(f"\nDetector statistics: {stats}")