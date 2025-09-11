"""
Volume Profile Analysis Integration
Comprehensive integration example showing how to use all volume analysis components together
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

# Import all volume analysis modules
from .volume_profile import VPVRAnalyzer, VolumeProfile
from .market_profile import MarketProfileAnalyzer, TPOProfile, MarketStructure
from .delta_volume import DeltaVolumeAnalyzer, DeltaBar
from .profile_patterns import ProfilePatternRecognizer, ProfilePattern
from .volume_levels import VolumeLevelAnalyzer, VolumeLevel

logger = logging.getLogger(__name__)


class VolumeAnalysisEngine:
    """
    Comprehensive volume analysis engine integrating all components
    """
    
    def __init__(self,
                 tick_size: float = 0.01,
                 tpo_size: int = 30,
                 min_confidence: float = 60.0):
        """
        Initialize the volume analysis engine
        
        Args:
            tick_size: Minimum price increment
            tpo_size: Minutes per TPO period
            min_confidence: Minimum confidence for pattern recognition
        """
        # Initialize all analyzers
        self.vpvr_analyzer = VPVRAnalyzer(tick_size=tick_size)
        self.market_profile_analyzer = MarketProfileAnalyzer(
            tick_size=tick_size, 
            tpo_size=tpo_size
        )
        self.delta_analyzer = DeltaVolumeAnalyzer()
        self.pattern_recognizer = ProfilePatternRecognizer(min_confidence=min_confidence)
        self.level_analyzer = VolumeLevelAnalyzer()
        
        # Storage for analysis results
        self.current_volume_profile: Optional[VolumeProfile] = None
        self.current_market_profile: Optional[TPOProfile] = None
        self.current_market_structure: Optional[MarketStructure] = None
        self.current_patterns: List[ProfilePattern] = []
        self.current_levels: List[VolumeLevel] = []
        
        # Historical data for context
        self.profile_history: List[VolumeProfile] = []
        self.market_profile_history: List[TPOProfile] = []
    
    def analyze_market_data(self,
                           price_data: pd.DataFrame,
                           volume_data: Optional[pd.DataFrame] = None,
                           order_book_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of market data using all volume analysis components
        
        Args:
            price_data: DataFrame with OHLCV data
            volume_data: Optional detailed volume data with buy/sell breakdown
            order_book_data: Optional order book data for flow analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting comprehensive volume analysis")
        
        results = {
            'timestamp': datetime.now(),
            'volume_profile': None,
            'market_profile': None,
            'market_structure': None,
            'delta_analysis': None,
            'patterns': [],
            'levels': [],
            'trading_signals': {},
            'rl_features': {},
            'confluence_analysis': {},
            'alerts': []
        }
        
        try:
            # 1. Volume Profile Analysis
            logger.info("Calculating volume profile")
            self.current_volume_profile = self.vpvr_analyzer.calculate_profile(
                price_data,
                price_col='close',
                volume_col='volume'
            )
            results['volume_profile'] = self._serialize_volume_profile(self.current_volume_profile)
            self.profile_history.append(self.current_volume_profile)
            
            # 2. Market Profile Analysis
            logger.info("Building market profile")
            self.current_market_profile = self.market_profile_analyzer.build_tpo_profile(
                price_data
            )
            results['market_profile'] = self._serialize_market_profile(self.current_market_profile)
            self.market_profile_history.append(self.current_market_profile)
            
            # Market Structure Analysis
            if len(self.market_profile_history) >= 3:
                logger.info("Analyzing market structure")
                self.current_market_structure = self.market_profile_analyzer.analyze_market_structure(
                    self.market_profile_history[-20:]  # Last 20 profiles
                )
                results['market_structure'] = self._serialize_market_structure(self.current_market_structure)
            
            # 3. Delta Volume Analysis
            if volume_data is not None:
                logger.info("Analyzing delta volume")
                delta_bars = self.delta_analyzer.calculate_delta_volume(volume_data)
                delta_stats = self.delta_analyzer.calculate_delta_statistics()
                delta_patterns = self.delta_analyzer.analyze_delta_patterns()
                
                results['delta_analysis'] = {
                    'statistics': delta_stats,
                    'patterns': delta_patterns,
                    'recent_bars': [self._serialize_delta_bar(bar) for bar in delta_bars[-10:]]
                }
                
                # Order flow analysis
                if order_book_data is not None:
                    imbalances = self.delta_analyzer.analyze_order_flow_imbalance(order_book_data)
                    results['delta_analysis']['order_flow'] = [
                        self._serialize_imbalance(imb) for imb in imbalances[-20:]
                    ]
            
            # 4. Pattern Recognition
            logger.info("Identifying patterns")
            self.current_patterns = self.pattern_recognizer.identify_patterns(
                self.current_volume_profile,
                self.current_market_profile,
                self.profile_history[-10:] if len(self.profile_history) >= 3 else None
            )
            results['patterns'] = [self._serialize_pattern(p) for p in self.current_patterns]
            
            # Composite pattern analysis
            if len(self.current_patterns) >= 2:
                composite = self.pattern_recognizer.combine_patterns(self.current_patterns)
                if composite:
                    results['composite_pattern'] = self._serialize_composite_pattern(composite)
            
            # 5. Volume Level Analysis
            logger.info("Extracting volume levels")
            self.current_levels = self.level_analyzer.extract_levels_from_profile(
                self.current_volume_profile,
                self.current_market_profile
            )
            
            # Level interaction analysis
            if not price_data.empty:
                interactions = self.level_analyzer.detect_level_interactions(price_data)
                results['level_interactions'] = [
                    self._serialize_level_test(test) for test in interactions[-20:]
                ]
            
            # Zone analysis
            zones = self.level_analyzer.identify_level_zones()
            results['level_zones'] = [self._serialize_zone(zone) for zone in zones]
            
            results['levels'] = [self._serialize_level(level) for level in self.current_levels]
            
            # 6. Integrated Trading Signals
            logger.info("Generating trading signals")
            current_price = price_data['close'].iloc[-1] if not price_data.empty else 0
            results['trading_signals'] = self._generate_integrated_signals(current_price)
            
            # 7. RL Feature Extraction
            logger.info("Extracting RL features")
            results['rl_features'] = self._extract_all_rl_features(current_price)
            
            # 8. Confluence Analysis
            logger.info("Analyzing confluence")
            results['confluence_analysis'] = self._analyze_total_confluence(current_price)
            
            # 9. Alert Generation
            logger.info("Generating alerts")
            results['alerts'] = self._generate_comprehensive_alerts(current_price)
            
            logger.info("Volume analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def get_trading_decision(self,
                           current_price: float,
                           market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive trading decision based on all analysis components
        """
        decision = {
            'action': 'hold',
            'confidence': 0,
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': [],
            'position_size': 0,
            'reasons': [],
            'risk_reward': 0,
            'time_horizon': 'short'
        }
        
        # Collect signals from all components
        signals = []
        
        # Volume profile signals
        if self.current_volume_profile:
            vp_levels = self.vpvr_analyzer.detect_support_resistance(
                self.current_volume_profile, current_price
            )
            
            # Check position relative to value area
            if current_price < self.current_volume_profile.val:
                signals.append({
                    'source': 'volume_profile',
                    'action': 'buy',
                    'strength': 70,
                    'reason': 'below_value_area'
                })
            elif current_price > self.current_volume_profile.vah:
                signals.append({
                    'source': 'volume_profile',
                    'action': 'sell',
                    'strength': 70,
                    'reason': 'above_value_area'
                })
        
        # Market profile signals
        if self.current_market_profile and self.current_market_structure:
            mp_signals = self.market_profile_analyzer.generate_trading_signals(
                self.current_market_profile,
                self.current_market_structure,
                current_price
            )
            
            if mp_signals['action'] != 'hold':
                signals.append({
                    'source': 'market_profile',
                    'action': mp_signals['action'],
                    'strength': mp_signals['strength'],
                    'reason': mp_signals['reasons'][0] if mp_signals['reasons'] else 'market_profile_signal'
                })
        
        # Delta analysis signals
        delta_signals = self.delta_analyzer.generate_trading_signals(current_price)
        if delta_signals['signal'] != 'neutral':
            signals.append({
                'source': 'delta_volume',
                'action': delta_signals['signal'],
                'strength': delta_signals['strength'],
                'reason': delta_signals['reasons'][0] if delta_signals['reasons'] else 'delta_signal'
            })
        
        # Pattern signals
        for pattern in self.current_patterns:
            if pattern.trading_bias in ['bullish', 'bearish']:
                action = 'buy' if pattern.trading_bias == 'bullish' else 'sell'
                signals.append({
                    'source': 'patterns',
                    'action': action,
                    'strength': pattern.confidence,
                    'reason': f'{pattern.pattern_type}_pattern'
                })
        
        # Level signals
        key_levels = self.level_analyzer.find_key_levels_for_price(current_price)
        
        # Check for level bounces
        for support in key_levels['support'][:2]:
            if abs(current_price - support.price) / current_price < 0.01:  # Within 1%
                signals.append({
                    'source': 'levels',
                    'action': 'buy',
                    'strength': support.strength,
                    'reason': 'support_level_bounce'
                })
        
        for resistance in key_levels['resistance'][:2]:
            if abs(current_price - resistance.price) / current_price < 0.01:  # Within 1%
                signals.append({
                    'source': 'levels',
                    'action': 'sell',
                    'strength': resistance.strength,
                    'reason': 'resistance_level_rejection'
                })
        
        # Aggregate signals
        if signals:
            buy_signals = [s for s in signals if s['action'] == 'buy']
            sell_signals = [s for s in signals if s['action'] == 'sell']
            
            buy_strength = sum(s['strength'] for s in buy_signals)
            sell_strength = sum(s['strength'] for s in sell_signals)
            
            if buy_strength > sell_strength and buy_strength > 100:
                decision['action'] = 'buy'
                decision['confidence'] = min(100, buy_strength / len(buy_signals))
                decision['reasons'] = [s['reason'] for s in buy_signals]
                
                # Set stops and targets
                if key_levels['support']:
                    decision['stop_loss'] = key_levels['support'][0].price * 0.995
                
                if key_levels['resistance']:
                    decision['take_profit'] = [r.price for r in key_levels['resistance'][:3]]
                
            elif sell_strength > buy_strength and sell_strength > 100:
                decision['action'] = 'sell'
                decision['confidence'] = min(100, sell_strength / len(sell_signals))
                decision['reasons'] = [s['reason'] for s in sell_signals]
                
                # Set stops and targets
                if key_levels['resistance']:
                    decision['stop_loss'] = key_levels['resistance'][0].price * 1.005
                
                if key_levels['support']:
                    decision['take_profit'] = [s.price for s in key_levels['support'][:3]]
        
        # Calculate position sizing based on confidence and risk
        if decision['action'] != 'hold' and decision['stop_loss']:
            risk_per_trade = 0.02  # 2% risk per trade
            stop_distance = abs(current_price - decision['stop_loss']) / current_price
            
            if stop_distance > 0:
                decision['position_size'] = (risk_per_trade / stop_distance) * (decision['confidence'] / 100)
        
        return decision
    
    def _generate_integrated_signals(self, current_price: float) -> Dict[str, Any]:
        """Generate integrated trading signals from all components"""
        signals = {
            'overall_bias': 'neutral',
            'strength': 0,
            'components': {},
            'confluence_score': 0,
            'key_levels': {},
            'recommended_action': 'wait'
        }
        
        component_signals = []
        
        # Volume profile component
        if self.current_volume_profile:
            vp_bias = self._get_volume_profile_bias(current_price)
            component_signals.append(vp_bias)
            signals['components']['volume_profile'] = vp_bias
        
        # Market profile component
        if self.current_market_profile:
            mp_bias = self._get_market_profile_bias(current_price)
            component_signals.append(mp_bias)
            signals['components']['market_profile'] = mp_bias
        
        # Delta component
        delta_bias = self._get_delta_bias()
        component_signals.append(delta_bias)
        signals['components']['delta'] = delta_bias
        
        # Pattern component
        pattern_bias = self._get_pattern_bias()
        component_signals.append(pattern_bias)
        signals['components']['patterns'] = pattern_bias
        
        # Level component
        level_bias = self._get_level_bias(current_price)
        component_signals.append(level_bias)
        signals['components']['levels'] = level_bias
        
        # Aggregate signals
        if component_signals:
            bullish_weight = sum(s['weight'] for s in component_signals if s['bias'] == 'bullish')
            bearish_weight = sum(s['weight'] for s in component_signals if s['bias'] == 'bearish')
            total_weight = bullish_weight + bearish_weight
            
            if total_weight > 0:
                if bullish_weight > bearish_weight * 1.5:
                    signals['overall_bias'] = 'bullish'
                    signals['strength'] = (bullish_weight / total_weight) * 100
                elif bearish_weight > bullish_weight * 1.5:
                    signals['overall_bias'] = 'bearish'
                    signals['strength'] = (bearish_weight / total_weight) * 100
                else:
                    signals['overall_bias'] = 'neutral'
                    signals['strength'] = 50
        
        # Confluence analysis
        signals['confluence_score'] = len([s for s in component_signals if s['bias'] != 'neutral'])
        
        # Key levels
        signals['key_levels'] = self.level_analyzer.find_key_levels_for_price(current_price)
        
        # Recommendation
        if signals['strength'] > 70 and signals['confluence_score'] >= 3:
            signals['recommended_action'] = signals['overall_bias']
        elif signals['strength'] > 60 and signals['confluence_score'] >= 2:
            signals['recommended_action'] = f"weak_{signals['overall_bias']}"
        else:
            signals['recommended_action'] = 'wait'
        
        return signals
    
    def _extract_all_rl_features(self, current_price: float) -> Dict[str, np.ndarray]:
        """Extract RL features from all components"""
        features = {}
        
        # Volume profile features
        if self.current_volume_profile:
            features['volume_profile'] = self.vpvr_analyzer.get_rl_features(
                self.current_volume_profile, current_price
            )
        
        # Market profile features
        if self.current_market_profile and self.current_market_structure:
            features['market_profile'] = self.market_profile_analyzer.get_rl_features(
                self.current_market_profile,
                self.current_market_structure,
                current_price
            )
        
        # Delta features
        features['delta'] = self.delta_analyzer.get_rl_features()
        
        # Pattern features
        features['patterns'] = self.pattern_recognizer.get_rl_features(self.current_patterns)
        
        # Level features
        features['levels'] = self.level_analyzer.get_rl_features(current_price)
        
        # Combined features
        all_features = []
        for component, feat_array in features.items():
            all_features.extend(feat_array.tolist())
        
        features['combined'] = np.array(all_features)
        
        return features
    
    def _analyze_total_confluence(self, current_price: float) -> Dict[str, Any]:
        """Analyze confluence across all analysis methods"""
        confluence = {
            'total_score': 0,
            'components': {},
            'key_prices': [],
            'strength_rating': 'weak'
        }
        
        # Volume level confluence
        level_confluence = self.level_analyzer.analyze_level_confluence(current_price)
        confluence['components']['levels'] = level_confluence
        confluence['total_score'] += level_confluence['confluence_score'] * 0.3
        
        # Value area confluence
        if self.current_volume_profile:
            va_distance = min(
                abs(current_price - self.current_volume_profile.vah) / current_price,
                abs(current_price - self.current_volume_profile.val) / current_price,
                abs(current_price - self.current_volume_profile.poc) / current_price
            )
            
            if va_distance < 0.01:  # Within 1%
                confluence['total_score'] += 25
                confluence['components']['value_area'] = 25
        
        # Pattern confluence
        pattern_score = len([p for p in self.current_patterns if p.confidence > 70]) * 15
        confluence['total_score'] += pattern_score
        confluence['components']['patterns'] = pattern_score
        
        # Market structure confluence
        if self.current_market_structure:
            structure_score = self.current_market_structure.confidence * 0.2
            confluence['total_score'] += structure_score
            confluence['components']['structure'] = structure_score
        
        # Rate overall confluence
        if confluence['total_score'] >= 80:
            confluence['strength_rating'] = 'very_strong'
        elif confluence['total_score'] >= 60:
            confluence['strength_rating'] = 'strong'
        elif confluence['total_score'] >= 40:
            confluence['strength_rating'] = 'moderate'
        else:
            confluence['strength_rating'] = 'weak'
        
        return confluence
    
    def _generate_comprehensive_alerts(self, current_price: float) -> List[Dict[str, Any]]:
        """Generate alerts from all analysis components"""
        alerts = []
        
        # Level alerts
        level_alerts = self.level_analyzer.generate_level_alerts(current_price)
        alerts.extend([{
            'type': 'level_alert',
            'priority': self._calculate_alert_priority(alert),
            'message': f"Approaching {alert['alert_type']}: {alert['level'].context} at {alert['level'].price:.2f}",
            'data': alert
        } for alert in level_alerts])
        
        # Pattern alerts
        for pattern in self.current_patterns:
            if pattern.confidence > 80:
                alerts.append({
                    'type': 'pattern_alert',
                    'priority': 'high' if pattern.confidence > 90 else 'medium',
                    'message': f"High confidence {pattern.pattern_type} pattern detected",
                    'data': pattern
                })
        
        # Delta alerts
        delta_patterns = self.delta_analyzer.analyze_delta_patterns()
        if delta_patterns.get('exhaustion_signals'):
            latest_exhaustion = delta_patterns['exhaustion_signals'][-1]
            alerts.append({
                'type': 'delta_alert',
                'priority': 'high',
                'message': f"Volume exhaustion detected: {latest_exhaustion['type']}",
                'data': latest_exhaustion
            })
        
        # Market structure alerts
        if self.current_market_structure and self.current_market_structure.confidence > 80:
            alerts.append({
                'type': 'structure_alert',
                'priority': 'medium',
                'message': f"Market structure: {self.current_market_structure.trend} trend confirmed",
                'data': self.current_market_structure
            })
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        alerts.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
        
        return alerts[:10]  # Top 10 alerts
    
    # Helper methods for bias calculation
    def _get_volume_profile_bias(self, current_price: float) -> Dict[str, Any]:
        """Get bias from volume profile"""
        if not self.current_volume_profile:
            return {'bias': 'neutral', 'weight': 0, 'confidence': 0}
        
        vp = self.current_volume_profile
        
        if current_price < vp.val:
            return {'bias': 'bullish', 'weight': 0.25, 'confidence': 70}
        elif current_price > vp.vah:
            return {'bias': 'bearish', 'weight': 0.25, 'confidence': 70}
        elif abs(current_price - vp.poc) / vp.poc < 0.005:
            return {'bias': 'neutral', 'weight': 0.1, 'confidence': 80}
        else:
            return {'bias': 'neutral', 'weight': 0.15, 'confidence': 60}
    
    def _get_market_profile_bias(self, current_price: float) -> Dict[str, Any]:
        """Get bias from market profile"""
        if not self.current_market_profile:
            return {'bias': 'neutral', 'weight': 0, 'confidence': 0}
        
        mp = self.current_market_profile
        
        if mp.profile_type == 'p-shaped':
            return {'bias': 'bearish', 'weight': 0.2, 'confidence': 75}
        elif mp.profile_type == 'd-shaped':
            return {'bias': 'bullish', 'weight': 0.2, 'confidence': 75}
        elif len(mp.single_prints) > 3:
            return {'bias': 'bullish', 'weight': 0.15, 'confidence': 65}
        else:
            return {'bias': 'neutral', 'weight': 0.1, 'confidence': 50}
    
    def _get_delta_bias(self) -> Dict[str, Any]:
        """Get bias from delta analysis"""
        if len(self.delta_analyzer.delta_bars) < 10:
            return {'bias': 'neutral', 'weight': 0, 'confidence': 0}
        
        recent_bars = list(self.delta_analyzer.delta_bars)[-10:]
        bullish_bars = sum(1 for bar in recent_bars if bar.delta > 0)
        
        if bullish_bars >= 7:
            return {'bias': 'bullish', 'weight': 0.2, 'confidence': 75}
        elif bullish_bars <= 3:
            return {'bias': 'bearish', 'weight': 0.2, 'confidence': 75}
        else:
            return {'bias': 'neutral', 'weight': 0.1, 'confidence': 50}
    
    def _get_pattern_bias(self) -> Dict[str, Any]:
        """Get bias from patterns"""
        if not self.current_patterns:
            return {'bias': 'neutral', 'weight': 0, 'confidence': 0}
        
        bullish_patterns = [p for p in self.current_patterns if p.trading_bias == 'bullish']
        bearish_patterns = [p for p in self.current_patterns if p.trading_bias == 'bearish']
        
        if len(bullish_patterns) > len(bearish_patterns):
            avg_conf = np.mean([p.confidence for p in bullish_patterns])
            return {'bias': 'bullish', 'weight': 0.15, 'confidence': avg_conf}
        elif len(bearish_patterns) > len(bullish_patterns):
            avg_conf = np.mean([p.confidence for p in bearish_patterns])
            return {'bias': 'bearish', 'weight': 0.15, 'confidence': avg_conf}
        else:
            return {'bias': 'neutral', 'weight': 0.1, 'confidence': 50}
    
    def _get_level_bias(self, current_price: float) -> Dict[str, Any]:
        """Get bias from levels"""
        key_levels = self.level_analyzer.find_key_levels_for_price(current_price, max_distance_pct=2.0)
        
        support_strength = sum(s.strength for s in key_levels['support'])
        resistance_strength = sum(r.strength for r in key_levels['resistance'])
        
        if support_strength > resistance_strength * 1.5:
            return {'bias': 'bullish', 'weight': 0.18, 'confidence': 70}
        elif resistance_strength > support_strength * 1.5:
            return {'bias': 'bearish', 'weight': 0.18, 'confidence': 70}
        else:
            return {'bias': 'neutral', 'weight': 0.1, 'confidence': 50}
    
    def _calculate_alert_priority(self, alert: Dict) -> str:
        """Calculate priority for alert"""
        if alert['level'].strength > 80 and alert['distance'] < 0.5:
            return 'high'
        elif alert['level'].strength > 60 and alert['distance'] < 1.0:
            return 'medium'
        else:
            return 'low'
    
    # Serialization methods for JSON output
    def _serialize_volume_profile(self, profile: VolumeProfile) -> Dict:
        """Serialize volume profile for JSON output"""
        return {
            'poc': float(profile.poc),
            'vah': float(profile.vah),
            'val': float(profile.val),
            'vwap': float(profile.vwap),
            'value_area_percentage': float(profile.value_area_percentage),
            'total_volume': float(profile.total_volume),
            'timestamp': profile.timestamp.isoformat(),
            'timeframe': profile.timeframe
        }
    
    def _serialize_market_profile(self, profile: TPOProfile) -> Dict:
        """Serialize market profile for JSON output"""
        return {
            'date': profile.date.isoformat(),
            'profile_type': profile.profile_type,
            'initial_balance': [float(profile.initial_balance[0]), float(profile.initial_balance[1])],
            'value_area': [float(profile.value_area[0]), float(profile.value_area[1]), float(profile.value_area[2])],
            'range_extension': profile.range_extension,
            'single_prints_count': len(profile.single_prints),
            'poor_highs_count': len(profile.poor_highs),
            'poor_lows_count': len(profile.poor_lows)
        }
    
    def _serialize_market_structure(self, structure: MarketStructure) -> Dict:
        """Serialize market structure for JSON output"""
        return {
            'trend': structure.trend,
            'migration': structure.migration,
            'confidence': float(structure.confidence),
            'balance_areas_count': len(structure.balance_areas),
            'acceptance_levels_count': len(structure.acceptance_levels),
            'rejection_levels_count': len(structure.rejection_levels)
        }
    
    def _serialize_delta_bar(self, bar: DeltaBar) -> Dict:
        """Serialize delta bar for JSON output"""
        return {
            'timestamp': bar.timestamp.isoformat(),
            'delta': float(bar.delta),
            'cumulative_delta': float(bar.cumulative_delta),
            'delta_percentage': float(bar.delta_percentage),
            'total_volume': float(bar.total_volume),
            'is_bullish': bar.is_bullish
        }
    
    def _serialize_imbalance(self, imbalance) -> Dict:
        """Serialize order flow imbalance for JSON output"""
        return {
            'timestamp': imbalance.timestamp.isoformat(),
            'imbalance_type': imbalance.imbalance_type,
            'total_imbalance': float(imbalance.total_imbalance),
            'stacked_imbalances': imbalance.stacked_imbalances
        }
    
    def _serialize_pattern(self, pattern: ProfilePattern) -> Dict:
        """Serialize pattern for JSON output"""
        return {
            'pattern_type': pattern.pattern_type,
            'confidence': float(pattern.confidence),
            'trading_bias': pattern.trading_bias,
            'start_time': pattern.start_time.isoformat(),
            'end_time': pattern.end_time.isoformat(),
            'price_levels': [float(pattern.price_levels[0]), float(pattern.price_levels[1])],
            'expected_move': float(pattern.expected_move) if pattern.expected_move else None,
            'risk_reward': float(pattern.risk_reward)
        }
    
    def _serialize_composite_pattern(self, composite) -> Dict:
        """Serialize composite pattern for JSON output"""
        return {
            'pattern_name': composite.pattern_name,
            'overall_confidence': float(composite.overall_confidence),
            'formation_stage': composite.formation_stage,
            'component_count': len(composite.component_patterns),
            'target_levels': [float(t) for t in composite.target_levels],
            'invalidation_level': float(composite.invalidation_level)
        }
    
    def _serialize_level(self, level: VolumeLevel) -> Dict:
        """Serialize volume level for JSON output"""
        return {
            'price': float(level.price),
            'level_type': level.level_type,
            'strength': float(level.strength),
            'confidence': float(level.confidence),
            'touch_count': level.touch_count,
            'context': level.context,
            'age_days': float(level.age_days),
            'reliability_score': float(level.reliability_score)
        }
    
    def _serialize_level_test(self, test) -> Dict:
        """Serialize level test for JSON output"""
        return {
            'timestamp': test.timestamp.isoformat(),
            'test_type': test.test_type,
            'level_price': float(test.level_price),
            'price': float(test.price),
            'reaction_strength': float(test.reaction_strength)
        }
    
    def _serialize_zone(self, zone) -> Dict:
        """Serialize level zone for JSON output"""
        return {
            'zone_center': float(zone.zone_center),
            'zone_range': [float(zone.zone_range[0]), float(zone.zone_range[1])],
            'zone_strength': float(zone.zone_strength),
            'dominant_type': zone.dominant_type,
            'level_count': len(zone.levels),
            'zone_width': float(zone.zone_width)
        }


# Example usage function
def example_usage():
    """
    Example of how to use the integrated volume analysis engine
    """
    # Initialize the engine
    engine = VolumeAnalysisEngine(tick_size=0.01, tpo_size=30)
    
    # Sample data (replace with real market data)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    price_data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(105, 115, 100),
        'low': np.random.uniform(95, 105, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    
    # Run comprehensive analysis
    results = engine.analyze_market_data(price_data)
    
    # Get trading decision
    current_price = price_data['close'].iloc[-1]
    decision = engine.get_trading_decision(current_price, {})
    
    # Print results
    print("Volume Analysis Results:")
    print(f"Current Price: {current_price:.2f}")
    print(f"Overall Bias: {results['trading_signals']['overall_bias']}")
    print(f"Confidence: {results['trading_signals']['strength']:.1f}%")
    print(f"Trading Decision: {decision['action']}")
    print(f"Decision Confidence: {decision['confidence']:.1f}%")
    
    if results['patterns']:
        print(f"Patterns Detected: {len(results['patterns'])}")
        for pattern in results['patterns']:
            print(f"  - {pattern['pattern_type']}: {pattern['confidence']:.1f}% confidence")
    
    if results['alerts']:
        print(f"Active Alerts: {len(results['alerts'])}")
        for alert in results['alerts'][:3]:  # Top 3 alerts
            print(f"  - {alert['type']}: {alert['message']}")


if __name__ == "__main__":
    example_usage()