"""
Liquidity Scoring System for Trading Opportunities
Evaluates and scores liquidity conditions for optimal trade execution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
from scipy import stats, optimize
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class LiquidityScore:
    """Comprehensive liquidity score for a trading opportunity"""
    timestamp: datetime
    symbol: str
    overall_score: float  # 0-100 score
    components: Dict[str, float]  # Individual component scores
    direction: str  # 'buy', 'sell', 'neutral'
    urgency: float  # 0-1 urgency indicator
    size_recommendation: float  # Recommended trade size
    execution_strategy: str  # Recommended execution approach
    risk_assessment: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_action_signal(self) -> str:
        """Get actionable trading signal"""
        if self.overall_score >= 80 and self.urgency >= 0.7:
            return f"STRONG_{self.direction.upper()}"
        elif self.overall_score >= 60 and self.urgency >= 0.5:
            return f"MODERATE_{self.direction.upper()}"
        elif self.overall_score >= 40:
            return f"WEAK_{self.direction.upper()}"
        return "HOLD"
    
    def to_rl_action(self) -> Dict[str, float]:
        """Convert to RL agent action format"""
        action_map = {
            'buy': 1.0,
            'sell': -1.0,
            'neutral': 0.0
        }
        
        return {
            'action': action_map.get(self.direction, 0.0),
            'confidence': self.overall_score / 100,
            'size': self.size_recommendation,
            'urgency': self.urgency
        }


class LiquidityScorer:
    """
    Advanced liquidity scoring system that evaluates multiple factors
    to identify optimal trading opportunities
    """
    
    def __init__(self,
                 symbol: str,
                 scoring_weights: Optional[Dict[str, float]] = None,
                 risk_tolerance: float = 0.5):
        """
        Initialize liquidity scorer
        
        Args:
            symbol: Trading symbol
            scoring_weights: Custom weights for scoring components
            risk_tolerance: Risk tolerance level (0-1)
        """
        self.symbol = symbol
        self.risk_tolerance = risk_tolerance
        
        # Default scoring weights
        self.weights = scoring_weights or {
            'depth_quality': 0.15,
            'spread_efficiency': 0.15,
            'market_impact': 0.15,
            'flow_toxicity': 0.10,
            'institutional_activity': 0.15,
            'momentum_alignment': 0.10,
            'volatility_regime': 0.10,
            'microstructure_health': 0.10
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Component scorers
        self.scaler = MinMaxScaler(feature_range=(0, 100))
        
        # Historical tracking
        self.score_history = deque(maxlen=100)
        self.opportunity_windows = []
        
        # Calibration parameters
        self.calibration_params = {
            'spread_baseline': None,
            'depth_baseline': None,
            'impact_baseline': None,
            'volatility_baseline': None
        }
        
        # Risk metrics
        self.risk_calculator = RiskCalculator(risk_tolerance)
        
        logger.info(f"LiquidityScorer initialized for {symbol}")
    
    def score_opportunity(self,
                         order_book_analysis: Dict[str, Any],
                         liquidity_signals: List[Any],
                         market_data: Dict[str, Any],
                         iceberg_detection: Optional[Dict[str, Any]] = None) -> LiquidityScore:
        """
        Score a liquidity opportunity based on multiple factors
        
        Args:
            order_book_analysis: Order book analysis results
            liquidity_signals: Liquidity hunting signals
            market_data: Current market data
            iceberg_detection: Iceberg detection results
            
        Returns:
            Comprehensive liquidity score
        """
        try:
            timestamp = datetime.now()
            
            # Calculate component scores
            components = {}
            
            # 1. Depth Quality Score
            components['depth_quality'] = self._score_depth_quality(order_book_analysis)
            
            # 2. Spread Efficiency Score
            components['spread_efficiency'] = self._score_spread_efficiency(order_book_analysis)
            
            # 3. Market Impact Score
            components['market_impact'] = self._score_market_impact(order_book_analysis)
            
            # 4. Flow Toxicity Score
            components['flow_toxicity'] = self._score_flow_toxicity(order_book_analysis)
            
            # 5. Institutional Activity Score
            components['institutional_activity'] = self._score_institutional_activity(
                liquidity_signals, iceberg_detection
            )
            
            # 6. Momentum Alignment Score
            components['momentum_alignment'] = self._score_momentum_alignment(
                market_data, order_book_analysis
            )
            
            # 7. Volatility Regime Score
            components['volatility_regime'] = self._score_volatility_regime(market_data)
            
            # 8. Microstructure Health Score
            components['microstructure_health'] = self._score_microstructure_health(
                order_book_analysis
            )
            
            # Calculate weighted overall score
            overall_score = sum(
                components[key] * self.weights.get(key, 0)
                for key in components
            )
            
            # Determine trading direction
            direction = self._determine_direction(
                order_book_analysis, liquidity_signals, market_data
            )
            
            # Calculate urgency
            urgency = self._calculate_urgency(
                components, liquidity_signals, market_data
            )
            
            # Recommend trade size
            size_recommendation = self._recommend_size(
                overall_score, components, order_book_analysis
            )
            
            # Determine execution strategy
            execution_strategy = self._determine_execution_strategy(
                components, urgency, size_recommendation
            )
            
            # Risk assessment
            risk_assessment = self.risk_calculator.assess_risk(
                components, order_book_analysis, market_data
            )
            
            # Create score object
            score = LiquidityScore(
                timestamp=timestamp,
                symbol=self.symbol,
                overall_score=overall_score,
                components=components,
                direction=direction,
                urgency=urgency,
                size_recommendation=size_recommendation,
                execution_strategy=execution_strategy,
                risk_assessment=risk_assessment,
                metadata={
                    'signal_count': len(liquidity_signals),
                    'has_icebergs': bool(iceberg_detection and iceberg_detection.get('icebergs')),
                    'market_conditions': self._assess_market_conditions(market_data)
                }
            )
            
            # Store in history
            self.score_history.append(score)
            
            # Identify opportunity windows
            self._update_opportunity_windows(score)
            
            return score
            
        except Exception as e:
            logger.error(f"Error scoring opportunity: {e}")
            return self._create_default_score()
    
    def _score_depth_quality(self, order_book_analysis: Dict[str, Any]) -> float:
        """Score order book depth quality"""
        try:
            score = 50.0  # Base score
            
            snapshot = order_book_analysis.get('snapshot')
            if not snapshot:
                return score
            
            # Factor 1: Absolute depth (normalized)
            depth_10 = snapshot.depth_10
            depth_50 = snapshot.depth_50
            depth_100 = snapshot.depth_100
            
            if self.calibration_params['depth_baseline']:
                depth_ratio = depth_10 / self.calibration_params['depth_baseline']
                score += min(25, depth_ratio * 25)  # Up to 25 points for good depth
            else:
                # Use heuristic
                if depth_10 > 0:
                    score += min(25, 25 * (1 - np.exp(-depth_10 / 100)))
            
            # Factor 2: Depth distribution (want gradual increase)
            if depth_100 > 0 and depth_50 > 0 and depth_10 > 0:
                ratio_50_10 = depth_50 / depth_10
                ratio_100_50 = depth_100 / depth_50
                
                # Ideal ratios: 2-4x increase at each level
                ideal_ratio = 3.0
                distribution_score = 25 * (1 - abs(ratio_50_10 - ideal_ratio) / ideal_ratio) * \
                                   (1 - abs(ratio_100_50 - ideal_ratio) / ideal_ratio)
                score += max(0, distribution_score)
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"Error scoring depth quality: {e}")
            return 50.0
    
    def _score_spread_efficiency(self, order_book_analysis: Dict[str, Any]) -> float:
        """Score spread efficiency"""
        try:
            score = 50.0
            
            snapshot = order_book_analysis.get('snapshot')
            microstructure = order_book_analysis.get('microstructure', {})
            
            if not snapshot:
                return score
            
            # Factor 1: Absolute spread (tighter is better)
            spread = snapshot.spread
            if self.calibration_params['spread_baseline']:
                spread_ratio = self.calibration_params['spread_baseline'] / (spread + 0.0001)
                score += min(30, spread_ratio * 30)
            else:
                # Use heuristic (spread < 0.1% is good)
                if spread < 0.001:
                    score += 30
                elif spread < 0.002:
                    score += 20
                elif spread < 0.005:
                    score += 10
            
            # Factor 2: Weighted spread
            weighted_spread = microstructure.get('weighted_spread', spread)
            if weighted_spread < spread:
                # Better liquidity at depth
                score += 20 * (1 - weighted_spread / (spread + 0.0001))
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"Error scoring spread efficiency: {e}")
            return 50.0
    
    def _score_market_impact(self, order_book_analysis: Dict[str, Any]) -> float:
        """Score expected market impact"""
        try:
            score = 50.0
            
            market_impact = order_book_analysis.get('market_impact', {})
            execution_quality = order_book_analysis.get('execution_quality', {})
            
            if not market_impact:
                return score
            
            # Factor 1: Kyle's lambda (lower is better)
            kyle_lambda_buy = market_impact.get('kyle_lambda_buy', 0)
            kyle_lambda_sell = market_impact.get('kyle_lambda_sell', 0)
            avg_lambda = (kyle_lambda_buy + kyle_lambda_sell) / 2
            
            if self.calibration_params['impact_baseline']:
                impact_ratio = self.calibration_params['impact_baseline'] / (avg_lambda + 0.0001)
                score += min(30, impact_ratio * 30)
            else:
                # Use heuristic
                if avg_lambda < 0.0001:
                    score += 30
                elif avg_lambda < 0.0005:
                    score += 20
                elif avg_lambda < 0.001:
                    score += 10
            
            # Factor 2: Expected slippage
            expected_slippage = execution_quality.get('expected_slippage', 0)
            if expected_slippage < 0.001:
                score += 20
            elif expected_slippage < 0.002:
                score += 10
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"Error scoring market impact: {e}")
            return 50.0
    
    def _score_flow_toxicity(self, order_book_analysis: Dict[str, Any]) -> float:
        """Score flow toxicity (lower toxicity is better)"""
        try:
            score = 50.0
            
            toxicity = order_book_analysis.get('flow_toxicity', 0.5)
            
            # Invert toxicity (low toxicity = high score)
            score = 100 * (1 - toxicity)
            
            # Adjust for sudden changes
            if len(self.score_history) > 5:
                recent_toxicities = [
                    s.components.get('flow_toxicity', 50) / 100
                    for s in list(self.score_history)[-5:]
                ]
                toxicity_trend = np.polyfit(range(len(recent_toxicities)), recent_toxicities, 1)[0]
                
                if toxicity_trend < 0:  # Improving toxicity
                    score += 10
                elif toxicity_trend > 0.1:  # Worsening rapidly
                    score -= 20
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"Error scoring flow toxicity: {e}")
            return 50.0
    
    def _score_institutional_activity(self,
                                     liquidity_signals: List[Any],
                                     iceberg_detection: Optional[Dict[str, Any]]) -> float:
        """Score institutional activity level"""
        try:
            score = 30.0  # Base score
            
            # Factor 1: Liquidity hunting signals
            if liquidity_signals:
                signal_strengths = [s.strength for s in liquidity_signals if hasattr(s, 'strength')]
                if signal_strengths:
                    avg_strength = np.mean(signal_strengths)
                    score += avg_strength * 30
                
                # Bonus for specific signal types
                signal_types = [s.signal_type for s in liquidity_signals if hasattr(s, 'signal_type')]
                if 'accumulation' in signal_types:
                    score += 10
                if 'iceberg' in signal_types:
                    score += 10
            
            # Factor 2: Iceberg detection
            if iceberg_detection:
                icebergs = iceberg_detection.get('icebergs', [])
                if icebergs:
                    # More icebergs indicate institutional activity
                    score += min(20, len(icebergs) * 5)
                    
                    # High confidence icebergs
                    high_confidence = [i for i in icebergs if i.confidence_score > 0.8]
                    if high_confidence:
                        score += 10
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"Error scoring institutional activity: {e}")
            return 30.0
    
    def _score_momentum_alignment(self,
                                 market_data: Dict[str, Any],
                                 order_book_analysis: Dict[str, Any]) -> float:
        """Score momentum and order flow alignment"""
        try:
            score = 50.0
            
            # Factor 1: Price momentum
            price = market_data.get('price', 0)
            price_change_24h = market_data.get('price_change_24h', 0)
            
            if price > 0 and price_change_24h != 0:
                momentum = price_change_24h / price
                
                # Check order book imbalance alignment
                imbalance = order_book_analysis.get('imbalance')
                if imbalance:
                    pressure = imbalance.pressure_score
                    
                    # Aligned momentum and pressure
                    if (momentum > 0 and pressure > 0) or (momentum < 0 and pressure < 0):
                        alignment = min(1, abs(momentum) * abs(pressure) * 10)
                        score += alignment * 30
                    else:
                        # Divergence (potential reversal)
                        score -= 10
            
            # Factor 2: Volume momentum
            volume = market_data.get('volume_24h', 0)
            avg_volume = market_data.get('avg_volume_30d', 0)
            
            if avg_volume > 0:
                volume_ratio = volume / avg_volume
                if volume_ratio > 1.5:  # High volume
                    score += 20
                elif volume_ratio > 1.0:
                    score += 10
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"Error scoring momentum alignment: {e}")
            return 50.0
    
    def _score_volatility_regime(self, market_data: Dict[str, Any]) -> float:
        """Score based on volatility regime"""
        try:
            score = 50.0
            
            volatility = market_data.get('volatility', 0)
            
            if self.calibration_params['volatility_baseline']:
                vol_ratio = volatility / self.calibration_params['volatility_baseline']
                
                # Moderate volatility is best (not too low, not too high)
                if 0.8 <= vol_ratio <= 1.2:
                    score = 80
                elif 0.5 <= vol_ratio <= 1.5:
                    score = 60
                elif vol_ratio > 2.0 or vol_ratio < 0.3:
                    score = 30
            else:
                # Use absolute thresholds
                if 0.01 <= volatility <= 0.03:  # 1-3% volatility
                    score = 80
                elif 0.005 <= volatility <= 0.05:  # 0.5-5% volatility
                    score = 60
                else:
                    score = 40
            
            # Adjust for risk tolerance
            if volatility > 0.05 and self.risk_tolerance < 0.3:
                score *= 0.5  # Penalize high volatility for low risk tolerance
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"Error scoring volatility regime: {e}")
            return 50.0
    
    def _score_microstructure_health(self, order_book_analysis: Dict[str, Any]) -> float:
        """Score overall market microstructure health"""
        try:
            score = 50.0
            
            microstructure = order_book_analysis.get('microstructure', {})
            
            if not microstructure:
                return score
            
            # Factor 1: Resilience
            resilience = microstructure.get('resilience', 0.5)
            score += (resilience - 0.5) * 40  # +/- 20 points
            
            # Factor 2: Mean reversion
            mean_reversion = microstructure.get('mean_reversion', 0.5)
            score += (mean_reversion - 0.5) * 20  # +/- 10 points
            
            # Factor 3: Book shape (convexity)
            convexity = microstructure.get('convexity', 0)
            if convexity > 0:  # Convex shape is healthy
                score += 10
            
            # Factor 4: Depth skew
            depth_skew = abs(microstructure.get('depth_skew', 0))
            if depth_skew < 0.2:  # Balanced book
                score += 10
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"Error scoring microstructure health: {e}")
            return 50.0
    
    def _determine_direction(self,
                            order_book_analysis: Dict[str, Any],
                            liquidity_signals: List[Any],
                            market_data: Dict[str, Any]) -> str:
        """Determine optimal trading direction"""
        try:
            buy_score = 0
            sell_score = 0
            
            # Order book imbalance
            imbalance = order_book_analysis.get('imbalance')
            if imbalance:
                if imbalance.pressure_score > 0.2:
                    buy_score += abs(imbalance.pressure_score)
                elif imbalance.pressure_score < -0.2:
                    sell_score += abs(imbalance.pressure_score)
            
            # Liquidity signals
            for signal in liquidity_signals:
                if hasattr(signal, 'direction') and hasattr(signal, 'strength'):
                    if signal.direction == 'buy':
                        buy_score += signal.strength
                    elif signal.direction == 'sell':
                        sell_score += signal.strength
            
            # Market momentum
            price_change = market_data.get('price_change_1h', 0)
            if price_change > 0:
                buy_score += 0.3
            elif price_change < 0:
                sell_score += 0.3
            
            # Determine direction
            if buy_score > sell_score * 1.2:
                return 'buy'
            elif sell_score > buy_score * 1.2:
                return 'sell'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining direction: {e}")
            return 'neutral'
    
    def _calculate_urgency(self,
                          components: Dict[str, float],
                          liquidity_signals: List[Any],
                          market_data: Dict[str, Any]) -> float:
        """Calculate trading urgency (0-1)"""
        try:
            urgency = 0.3  # Base urgency
            
            # Factor 1: Deteriorating conditions
            if len(self.score_history) > 5:
                recent_scores = [s.overall_score for s in list(self.score_history)[-5:]]
                score_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                
                if score_trend < -5:  # Rapidly deteriorating
                    urgency += 0.3
                elif score_trend < -2:
                    urgency += 0.1
            
            # Factor 2: Strong signals
            strong_signals = [s for s in liquidity_signals 
                            if hasattr(s, 'strength') and s.strength > 0.8]
            if strong_signals:
                urgency += 0.2
            
            # Factor 3: Volatility spike
            volatility = market_data.get('volatility', 0)
            if volatility > 0.05:  # High volatility
                urgency += 0.2
            
            # Factor 4: Exceptional scores
            if components.get('institutional_activity', 0) > 80:
                urgency += 0.1
            if components.get('depth_quality', 0) > 80:
                urgency += 0.1
            
            return min(1.0, urgency)
            
        except Exception as e:
            logger.error(f"Error calculating urgency: {e}")
            return 0.3
    
    def _recommend_size(self,
                       overall_score: float,
                       components: Dict[str, float],
                       order_book_analysis: Dict[str, Any]) -> float:
        """Recommend optimal trade size (fraction of available liquidity)"""
        try:
            # Base size on score
            if overall_score >= 80:
                base_size = 0.05  # 5% of available liquidity
            elif overall_score >= 60:
                base_size = 0.03  # 3%
            elif overall_score >= 40:
                base_size = 0.01  # 1%
            else:
                base_size = 0.005  # 0.5%
            
            # Adjust for market impact
            if components.get('market_impact', 50) > 70:
                base_size *= 1.5  # Can trade larger
            elif components.get('market_impact', 50) < 30:
                base_size *= 0.5  # Trade smaller
            
            # Adjust for depth quality
            if components.get('depth_quality', 50) > 70:
                base_size *= 1.2
            elif components.get('depth_quality', 50) < 30:
                base_size *= 0.7
            
            # Risk adjustment
            base_size *= (0.5 + self.risk_tolerance)
            
            # Cap at maximum
            return min(0.1, base_size)  # Max 10% of liquidity
            
        except Exception as e:
            logger.error(f"Error recommending size: {e}")
            return 0.01
    
    def _determine_execution_strategy(self,
                                     components: Dict[str, float],
                                     urgency: float,
                                     size: float) -> str:
        """Determine optimal execution strategy"""
        try:
            # High urgency strategies
            if urgency > 0.7:
                if size < 0.02:
                    return "aggressive_market_order"
                else:
                    return "aggressive_sweep"
            
            # Low market impact environment
            if components.get('market_impact', 50) > 70:
                if size < 0.03:
                    return "passive_limit_order"
                else:
                    return "iceberg_order"
            
            # High institutional activity
            if components.get('institutional_activity', 50) > 70:
                return "follow_institutional"
            
            # Default strategies based on size
            if size < 0.01:
                return "standard_limit_order"
            elif size < 0.05:
                return "time_weighted_execution"
            else:
                return "volume_weighted_execution"
                
        except Exception as e:
            logger.error(f"Error determining execution strategy: {e}")
            return "standard_limit_order"
    
    def _assess_market_conditions(self, market_data: Dict[str, Any]) -> str:
        """Assess overall market conditions"""
        try:
            conditions = []
            
            # Trend
            price_change_24h = market_data.get('price_change_24h', 0)
            if abs(price_change_24h) > 0.05:
                conditions.append("trending")
            else:
                conditions.append("ranging")
            
            # Volume
            volume_ratio = market_data.get('volume_24h', 0) / (market_data.get('avg_volume_30d', 1))
            if volume_ratio > 1.5:
                conditions.append("high_volume")
            elif volume_ratio < 0.5:
                conditions.append("low_volume")
            
            # Volatility
            volatility = market_data.get('volatility', 0)
            if volatility > 0.05:
                conditions.append("high_volatility")
            elif volatility < 0.01:
                conditions.append("low_volatility")
            
            return "_".join(conditions) if conditions else "normal"
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {e}")
            return "unknown"
    
    def _update_opportunity_windows(self, score: LiquidityScore):
        """Update tracked opportunity windows"""
        try:
            # Check if this is a new opportunity window
            if score.overall_score >= 70 and score.urgency >= 0.5:
                # Check if part of existing window
                if self.opportunity_windows:
                    last_window = self.opportunity_windows[-1]
                    time_diff = (score.timestamp - last_window['end_time']).total_seconds()
                    
                    if time_diff < 300:  # Within 5 minutes
                        # Extend window
                        last_window['end_time'] = score.timestamp
                        last_window['scores'].append(score)
                        last_window['peak_score'] = max(last_window['peak_score'], score.overall_score)
                    else:
                        # New window
                        self.opportunity_windows.append({
                            'start_time': score.timestamp,
                            'end_time': score.timestamp,
                            'scores': [score],
                            'peak_score': score.overall_score
                        })
                else:
                    # First window
                    self.opportunity_windows.append({
                        'start_time': score.timestamp,
                        'end_time': score.timestamp,
                        'scores': [score],
                        'peak_score': score.overall_score
                    })
            
            # Clean old windows
            cutoff = datetime.now() - timedelta(hours=1)
            self.opportunity_windows = [
                w for w in self.opportunity_windows
                if w['end_time'] > cutoff
            ]
            
        except Exception as e:
            logger.error(f"Error updating opportunity windows: {e}")
    
    def _create_default_score(self) -> LiquidityScore:
        """Create default score when error occurs"""
        return LiquidityScore(
            timestamp=datetime.now(),
            symbol=self.symbol,
            overall_score=30.0,
            components={k: 30.0 for k in self.weights.keys()},
            direction='neutral',
            urgency=0.1,
            size_recommendation=0.005,
            execution_strategy='standard_limit_order',
            risk_assessment={'risk_level': 'unknown'}
        )
    
    def calibrate(self, historical_data: List[Dict[str, Any]]):
        """Calibrate scorer with historical data"""
        try:
            if not historical_data:
                return
            
            # Extract baselines
            spreads = []
            depths = []
            impacts = []
            volatilities = []
            
            for data in historical_data:
                if 'snapshot' in data:
                    snapshot = data['snapshot']
                    spreads.append(snapshot.spread)
                    depths.append(snapshot.depth_10)
                
                if 'market_impact' in data:
                    kyle_buy = data['market_impact'].get('kyle_lambda_buy', 0)
                    kyle_sell = data['market_impact'].get('kyle_lambda_sell', 0)
                    impacts.append((kyle_buy + kyle_sell) / 2)
                
                if 'market_data' in data:
                    vol = data['market_data'].get('volatility', 0)
                    if vol > 0:
                        volatilities.append(vol)
            
            # Set baselines as medians
            if spreads:
                self.calibration_params['spread_baseline'] = np.median(spreads)
            if depths:
                self.calibration_params['depth_baseline'] = np.median(depths)
            if impacts:
                self.calibration_params['impact_baseline'] = np.median(impacts)
            if volatilities:
                self.calibration_params['volatility_baseline'] = np.median(volatilities)
            
            logger.info(f"Calibrated with baselines: {self.calibration_params}")
            
        except Exception as e:
            logger.error(f"Error calibrating scorer: {e}")
    
    def get_recent_opportunities(self, lookback_minutes: int = 30) -> List[Dict[str, Any]]:
        """Get recent high-scoring opportunities"""
        cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
        
        opportunities = []
        for window in self.opportunity_windows:
            if window['end_time'] > cutoff:
                opportunities.append({
                    'start': window['start_time'],
                    'end': window['end_time'],
                    'duration_seconds': (window['end_time'] - window['start_time']).total_seconds(),
                    'peak_score': window['peak_score'],
                    'avg_score': np.mean([s.overall_score for s in window['scores']]),
                    'direction': window['scores'][0].direction
                })
        
        return opportunities
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scoring statistics"""
        if not self.score_history:
            return {}
        
        recent_scores = list(self.score_history)[-20:]
        
        return {
            'avg_score': np.mean([s.overall_score for s in recent_scores]),
            'max_score': max([s.overall_score for s in recent_scores]),
            'min_score': min([s.overall_score for s in recent_scores]),
            'current_opportunities': len(self.opportunity_windows),
            'avg_urgency': np.mean([s.urgency for s in recent_scores]),
            'calibration': self.calibration_params
        }


class RiskCalculator:
    """Risk assessment for liquidity opportunities"""
    
    def __init__(self, risk_tolerance: float):
        self.risk_tolerance = risk_tolerance
    
    def assess_risk(self,
                   components: Dict[str, float],
                   order_book_analysis: Dict[str, Any],
                   market_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess risk metrics"""
        risk = {
            'risk_level': 'medium',
            'max_drawdown': 0.02,
            'var_95': 0.01,
            'liquidity_risk': 0.3,
            'execution_risk': 0.2
        }
        
        try:
            # Liquidity risk
            depth_score = components.get('depth_quality', 50)
            if depth_score < 30:
                risk['liquidity_risk'] = 0.7
                risk['risk_level'] = 'high'
            elif depth_score > 70:
                risk['liquidity_risk'] = 0.1
                risk['risk_level'] = 'low'
            
            # Execution risk
            impact_score = components.get('market_impact', 50)
            if impact_score < 30:
                risk['execution_risk'] = 0.6
            elif impact_score > 70:
                risk['execution_risk'] = 0.1
            
            # Market risk (VaR)
            volatility = market_data.get('volatility', 0.02)
            risk['var_95'] = volatility * 1.65  # 95% VaR
            risk['max_drawdown'] = volatility * 3  # Approximate max drawdown
            
            # Overall risk level
            total_risk = (risk['liquidity_risk'] + risk['execution_risk']) / 2
            if total_risk > 0.5:
                risk['risk_level'] = 'high'
            elif total_risk < 0.2:
                risk['risk_level'] = 'low'
            else:
                risk['risk_level'] = 'medium'
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
        
        return risk


if __name__ == "__main__":
    # Example usage
    scorer = LiquidityScorer(
        symbol="BTC/USDT",
        risk_tolerance=0.5
    )
    
    # Mock data
    from liquidity_hunting import LiquiditySignal
    
    mock_order_book_analysis = {
        'snapshot': type('obj', (object,), {
            'spread': 0.001,
            'depth_10': 100,
            'depth_50': 500,
            'depth_100': 1000
        }),
        'imbalance': type('obj', (object,), {
            'pressure_score': 0.3
        }),
        'microstructure': {
            'resilience': 0.7,
            'mean_reversion': 0.6,
            'convexity': 0.1,
            'depth_skew': 0.1
        },
        'flow_toxicity': 0.3,
        'market_impact': {
            'kyle_lambda_buy': 0.0002,
            'kyle_lambda_sell': 0.0003
        },
        'execution_quality': {
            'expected_slippage': 0.0015
        }
    }
    
    mock_signals = [
        LiquiditySignal(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            signal_type='accumulation',
            direction='buy',
            strength=0.8,
            price_level=50000,
            volume_estimate=10
        )
    ]
    
    mock_market_data = {
        'price': 50000,
        'price_change_24h': 500,
        'price_change_1h': 50,
        'volume_24h': 1000000,
        'avg_volume_30d': 800000,
        'volatility': 0.02
    }
    
    # Score opportunity
    score = scorer.score_opportunity(
        mock_order_book_analysis,
        mock_signals,
        mock_market_data
    )
    
    print(f"Overall Score: {score.overall_score:.1f}/100")
    print(f"Direction: {score.direction}")
    print(f"Urgency: {score.urgency:.2f}")
    print(f"Action Signal: {score.get_action_signal()}")
    print(f"Execution Strategy: {score.execution_strategy}")
    print(f"Risk Level: {score.risk_assessment['risk_level']}")
    
    print("\nComponent Scores:")
    for component, value in score.components.items():
        print(f"  {component}: {value:.1f}")
    
    # Get statistics
    stats = scorer.get_statistics()
    print(f"\nStatistics: {stats}")