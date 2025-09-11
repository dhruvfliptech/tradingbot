"""
Profile Pattern Recognition
Advanced pattern recognition in volume and market profiles for trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from scipy import stats, signal, spatial
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ProfilePattern:
    """Identified pattern in profile"""
    pattern_type: str
    confidence: float  # 0-100
    start_time: datetime
    end_time: datetime
    price_levels: Tuple[float, float]  # (high, low)
    key_features: Dict[str, Any]
    trading_bias: str  # 'bullish', 'bearish', 'neutral'
    expected_move: Optional[float]
    risk_reward: float
    
    @property
    def duration(self) -> timedelta:
        """Pattern duration"""
        return self.end_time - self.start_time
    
    @property
    def price_range(self) -> float:
        """Pattern price range"""
        return self.price_levels[0] - self.price_levels[1]


@dataclass
class CompositePattern:
    """Complex pattern from multiple profiles"""
    pattern_name: str
    component_patterns: List[ProfilePattern]
    overall_confidence: float
    formation_stage: str  # 'forming', 'complete', 'failed'
    target_levels: List[float]
    invalidation_level: float
    time_to_target: Optional[timedelta]


class ProfilePatternRecognizer:
    """Advanced pattern recognition in volume and market profiles"""
    
    def __init__(self,
                 min_confidence: float = 60.0,
                 pattern_library: Optional[Dict] = None,
                 similarity_threshold: float = 0.75):
        """
        Initialize pattern recognizer
        
        Args:
            min_confidence: Minimum confidence to report pattern
            pattern_library: Pre-defined pattern templates
            similarity_threshold: Threshold for pattern matching
        """
        self.min_confidence = min_confidence
        self.similarity_threshold = similarity_threshold
        
        # Initialize pattern library
        self.pattern_library = pattern_library or self._initialize_pattern_library()
        
        # Pattern cache
        self.identified_patterns: List[ProfilePattern] = []
        self.composite_patterns: List[CompositePattern] = []
        
        # Analysis tools
        self.scaler = StandardScaler()
    
    def _initialize_pattern_library(self) -> Dict[str, Any]:
        """Initialize library of known patterns"""
        return {
            # Volume patterns
            'accumulation': {
                'description': 'Volume accumulation pattern',
                'features': ['increasing_volume', 'narrowing_range', 'poc_stability'],
                'bias': 'bullish',
                'confidence_weights': {'volume': 0.4, 'range': 0.3, 'poc': 0.3}
            },
            'distribution': {
                'description': 'Volume distribution pattern',
                'features': ['high_volume_top', 'widening_range', 'poc_decline'],
                'bias': 'bearish',
                'confidence_weights': {'volume': 0.4, 'range': 0.3, 'poc': 0.3}
            },
            'wyckoff_spring': {
                'description': 'Wyckoff spring pattern',
                'features': ['false_breakdown', 'volume_spike', 'quick_recovery'],
                'bias': 'bullish',
                'confidence_weights': {'breakdown': 0.5, 'volume': 0.3, 'recovery': 0.2}
            },
            'volume_breakout': {
                'description': 'Volume-confirmed breakout',
                'features': ['range_break', 'volume_expansion', 'follow_through'],
                'bias': 'directional',
                'confidence_weights': {'break': 0.4, 'volume': 0.4, 'follow': 0.2}
            },
            
            # Market profile patterns
            'double_distribution': {
                'description': 'Two distinct value areas',
                'features': ['two_peaks', 'value_gap', 'volume_migration'],
                'bias': 'neutral',
                'confidence_weights': {'peaks': 0.4, 'gap': 0.3, 'migration': 0.3}
            },
            'ledge_formation': {
                'description': 'Ledge pattern in profile',
                'features': ['flat_top_bottom', 'volume_shelf', 'price_acceptance'],
                'bias': 'continuation',
                'confidence_weights': {'shape': 0.4, 'volume': 0.3, 'acceptance': 0.3}
            },
            'p_shape': {
                'description': 'P-shaped profile',
                'features': ['top_heavy', 'thin_bottom', 'selling_tail'],
                'bias': 'bearish',
                'confidence_weights': {'shape': 0.5, 'volume': 0.3, 'tail': 0.2}
            },
            'b_shape': {
                'description': 'B-shaped profile',
                'features': ['bottom_heavy', 'thin_top', 'buying_tail'],
                'bias': 'bullish',
                'confidence_weights': {'shape': 0.5, 'volume': 0.3, 'tail': 0.2}
            },
            
            # Composite patterns
            'value_area_rule': {
                'description': 'Value area rule pattern',
                'features': ['va_break', 'va_test', 'va_hold'],
                'bias': 'directional',
                'confidence_weights': {'break': 0.4, 'test': 0.3, 'hold': 0.3}
            },
            'initiative_tail': {
                'description': 'Initiative activity tail',
                'features': ['single_prints', 'directional_move', 'no_return'],
                'bias': 'directional',
                'confidence_weights': {'prints': 0.4, 'move': 0.4, 'return': 0.2}
            },
            'balance_area_breakout': {
                'description': 'Breakout from balance area',
                'features': ['multiple_overlapping_values', 'tightening_range', 'volume_surge'],
                'bias': 'directional',
                'confidence_weights': {'overlap': 0.3, 'range': 0.3, 'volume': 0.4}
            }
        }
    
    def identify_patterns(self,
                         volume_profile: Any,
                         market_profile: Optional[Any] = None,
                         historical_profiles: Optional[List] = None) -> List[ProfilePattern]:
        """
        Identify patterns in profiles
        """
        patterns = []
        
        # Check volume profile patterns
        if volume_profile:
            volume_patterns = self._identify_volume_patterns(volume_profile)
            patterns.extend(volume_patterns)
        
        # Check market profile patterns
        if market_profile:
            market_patterns = self._identify_market_profile_patterns(market_profile)
            patterns.extend(market_patterns)
        
        # Check composite patterns using historical data
        if historical_profiles and len(historical_profiles) >= 3:
            composite_patterns = self._identify_composite_patterns(
                volume_profile, market_profile, historical_profiles
            )
            patterns.extend(composite_patterns)
        
        # Filter by minimum confidence
        patterns = [p for p in patterns if p.confidence >= self.min_confidence]
        
        # Store identified patterns
        self.identified_patterns.extend(patterns)
        
        return patterns
    
    def _identify_volume_patterns(self, profile: Any) -> List[ProfilePattern]:
        """Identify patterns in volume profile"""
        patterns = []
        
        # Check for accumulation
        if self._check_accumulation_pattern(profile):
            pattern = ProfilePattern(
                pattern_type='accumulation',
                confidence=self._calculate_pattern_confidence('accumulation', profile),
                start_time=profile.timestamp,
                end_time=profile.timestamp,
                price_levels=(profile.vah, profile.val),
                key_features={
                    'volume_concentration': profile.value_area_percentage,
                    'poc_position': (profile.poc - profile.val) / (profile.vah - profile.val) if profile.vah != profile.val else 0.5,
                    'volume_skew': self._calculate_volume_skew(profile)
                },
                trading_bias='bullish',
                expected_move=profile.vah - profile.poc,
                risk_reward=2.0
            )
            patterns.append(pattern)
        
        # Check for distribution
        if self._check_distribution_pattern(profile):
            pattern = ProfilePattern(
                pattern_type='distribution',
                confidence=self._calculate_pattern_confidence('distribution', profile),
                start_time=profile.timestamp,
                end_time=profile.timestamp,
                price_levels=(profile.vah, profile.val),
                key_features={
                    'volume_dispersion': 100 - profile.value_area_percentage,
                    'poc_weakness': self._calculate_poc_weakness(profile),
                    'top_heavy': self._check_top_heavy(profile)
                },
                trading_bias='bearish',
                expected_move=profile.poc - profile.val,
                risk_reward=2.0
            )
            patterns.append(pattern)
        
        # Check for volume breakout setup
        if self._check_volume_breakout_setup(profile):
            direction = self._determine_breakout_direction(profile)
            pattern = ProfilePattern(
                pattern_type='volume_breakout',
                confidence=self._calculate_pattern_confidence('volume_breakout', profile),
                start_time=profile.timestamp,
                end_time=profile.timestamp,
                price_levels=(profile.vah, profile.val),
                key_features={
                    'compression': self._calculate_compression(profile),
                    'volume_buildup': self._calculate_volume_buildup(profile),
                    'breakout_direction': direction
                },
                trading_bias='bullish' if direction > 0 else 'bearish',
                expected_move=profile.vah - profile.val if direction > 0 else profile.val - profile.vah,
                risk_reward=3.0
            )
            patterns.append(pattern)
        
        return patterns
    
    def _identify_market_profile_patterns(self, profile: Any) -> List[ProfilePattern]:
        """Identify patterns in market profile"""
        patterns = []
        
        # Check profile shape patterns
        if profile.profile_type == 'p-shaped':
            pattern = ProfilePattern(
                pattern_type='p_shape',
                confidence=self._calculate_shape_confidence(profile, 'p-shaped'),
                start_time=profile.date,
                end_time=profile.date,
                price_levels=(profile.price_levels.max(), profile.price_levels.min()),
                key_features={
                    'shape': profile.profile_type,
                    'poor_high': len(profile.poor_highs) > 0,
                    'range_extension': profile.range_extension
                },
                trading_bias='bearish',
                expected_move=(profile.value_area[0] - profile.value_area[2]) * 0.5,
                risk_reward=2.5
            )
            patterns.append(pattern)
        
        elif profile.profile_type == 'b-shaped':
            pattern = ProfilePattern(
                pattern_type='double_distribution',
                confidence=self._calculate_shape_confidence(profile, 'b-shaped'),
                start_time=profile.date,
                end_time=profile.date,
                price_levels=(profile.price_levels.max(), profile.price_levels.min()),
                key_features={
                    'shape': profile.profile_type,
                    'distributions': 2,
                    'value_split': self._identify_value_split(profile)
                },
                trading_bias='neutral',
                expected_move=None,
                risk_reward=1.5
            )
            patterns.append(pattern)
        
        # Check for initiative tails
        if len(profile.single_prints) >= 3:
            tail_direction = self._analyze_single_prints(profile)
            if tail_direction != 0:
                pattern = ProfilePattern(
                    pattern_type='initiative_tail',
                    confidence=min(100, 60 + len(profile.single_prints) * 5),
                    start_time=profile.date,
                    end_time=profile.date,
                    price_levels=(profile.price_levels.max(), profile.price_levels.min()),
                    key_features={
                        'single_prints': len(profile.single_prints),
                        'tail_direction': tail_direction,
                        'conviction': len(profile.single_prints) / 10
                    },
                    trading_bias='bullish' if tail_direction > 0 else 'bearish',
                    expected_move=profile.ib_range * 1.5,
                    risk_reward=3.0
                )
                patterns.append(pattern)
        
        # Check for ledge patterns
        if self._check_ledge_pattern(profile):
            pattern = ProfilePattern(
                pattern_type='ledge_formation',
                confidence=self._calculate_ledge_confidence(profile),
                start_time=profile.date,
                end_time=profile.date,
                price_levels=(profile.value_area[0], profile.value_area[2]),
                key_features={
                    'ledge_width': self._calculate_ledge_width(profile),
                    'time_at_ledge': self._calculate_time_at_ledge(profile),
                    'ledge_position': 'top' if self._is_top_ledge(profile) else 'bottom'
                },
                trading_bias='continuation',
                expected_move=profile.value_area_width,
                risk_reward=2.0
            )
            patterns.append(pattern)
        
        return patterns
    
    def _identify_composite_patterns(self,
                                    current_volume: Any,
                                    current_market: Any,
                                    historical: List) -> List[ProfilePattern]:
        """Identify patterns using multiple profiles"""
        patterns = []
        
        # Value Area Rule
        if self._check_value_area_rule(current_volume, historical):
            pattern = ProfilePattern(
                pattern_type='value_area_rule',
                confidence=self._calculate_var_confidence(current_volume, historical),
                start_time=historical[0].timestamp if hasattr(historical[0], 'timestamp') else datetime.now(),
                end_time=current_volume.timestamp,
                price_levels=(current_volume.vah, current_volume.val),
                key_features={
                    'va_relationship': self._analyze_va_relationship(current_volume, historical),
                    'acceptance': self._check_price_acceptance(current_volume, historical),
                    'rotation': self._check_rotation(historical)
                },
                trading_bias=self._determine_var_bias(current_volume, historical),
                expected_move=current_volume.vah - current_volume.val,
                risk_reward=2.5
            )
            patterns.append(pattern)
        
        # Balance Area Breakout
        if self._check_balance_breakout(historical):
            breakout_direction = self._determine_balance_breakout_direction(historical)
            pattern = ProfilePattern(
                pattern_type='balance_area_breakout',
                confidence=self._calculate_balance_breakout_confidence(historical),
                start_time=historical[0].timestamp if hasattr(historical[0], 'timestamp') else datetime.now(),
                end_time=historical[-1].timestamp if hasattr(historical[-1], 'timestamp') else datetime.now(),
                price_levels=self._get_balance_range(historical),
                key_features={
                    'balance_duration': len(historical),
                    'compression_rate': self._calculate_compression_rate(historical),
                    'volume_divergence': self._check_volume_divergence(historical)
                },
                trading_bias='bullish' if breakout_direction > 0 else 'bearish',
                expected_move=self._calculate_breakout_target(historical),
                risk_reward=3.5
            )
            patterns.append(pattern)
        
        # Wyckoff patterns
        wyckoff_pattern = self._identify_wyckoff_pattern(historical)
        if wyckoff_pattern:
            patterns.append(wyckoff_pattern)
        
        return patterns
    
    def match_pattern_template(self,
                              profile_features: np.ndarray,
                              template_name: str) -> float:
        """
        Match profile features against pattern template
        """
        if template_name not in self.pattern_library:
            return 0.0
        
        template = self.pattern_library[template_name]
        
        # Calculate similarity score
        similarity_scores = []
        
        for feature in template['features']:
            if feature in profile_features:
                score = self._calculate_feature_similarity(profile_features[feature], template[feature])
                weight = template['confidence_weights'].get(feature, 1.0)
                similarity_scores.append(score * weight)
        
        if similarity_scores:
            return np.mean(similarity_scores)
        
        return 0.0
    
    def predict_pattern_outcome(self,
                               pattern: ProfilePattern,
                               market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict likely outcome of identified pattern
        """
        prediction = {
            'success_probability': 0,
            'expected_duration': None,
            'target_levels': [],
            'stop_levels': [],
            'optimal_entry': None
        }
        
        # Base probability on pattern confidence
        base_probability = pattern.confidence / 100
        
        # Adjust for market context
        context_adjustment = self._calculate_context_adjustment(pattern, market_context)
        prediction['success_probability'] = min(1.0, base_probability * context_adjustment)
        
        # Calculate targets based on pattern type
        if pattern.pattern_type == 'accumulation':
            prediction['target_levels'] = [
                pattern.price_levels[1] * 1.05,  # 5% above support
                pattern.price_levels[1] * 1.10,  # 10% above
                pattern.price_levels[1] * 1.15   # 15% above
            ]
            prediction['stop_levels'] = [pattern.price_levels[1] * 0.98]
            prediction['expected_duration'] = timedelta(days=5)
            
        elif pattern.pattern_type == 'distribution':
            prediction['target_levels'] = [
                pattern.price_levels[0] * 0.95,  # 5% below resistance
                pattern.price_levels[0] * 0.90,  # 10% below
                pattern.price_levels[0] * 0.85   # 15% below
            ]
            prediction['stop_levels'] = [pattern.price_levels[0] * 1.02]
            prediction['expected_duration'] = timedelta(days=5)
            
        elif pattern.pattern_type == 'volume_breakout':
            if pattern.trading_bias == 'bullish':
                prediction['target_levels'] = [
                    pattern.price_levels[0],
                    pattern.price_levels[0] * 1.05,
                    pattern.price_levels[0] * 1.10
                ]
                prediction['stop_levels'] = [pattern.price_levels[1]]
            else:
                prediction['target_levels'] = [
                    pattern.price_levels[1],
                    pattern.price_levels[1] * 0.95,
                    pattern.price_levels[1] * 0.90
                ]
                prediction['stop_levels'] = [pattern.price_levels[0]]
            prediction['expected_duration'] = timedelta(days=2)
        
        # Determine optimal entry
        prediction['optimal_entry'] = self._calculate_optimal_entry(pattern, market_context)
        
        return prediction
    
    def combine_patterns(self,
                        patterns: List[ProfilePattern]) -> Optional[CompositePattern]:
        """
        Combine multiple patterns into composite pattern
        """
        if len(patterns) < 2:
            return None
        
        # Group patterns by compatibility
        compatible_groups = self._group_compatible_patterns(patterns)
        
        if not compatible_groups:
            return None
        
        # Create composite from strongest group
        strongest_group = max(compatible_groups, key=lambda g: sum(p.confidence for p in g))
        
        # Determine composite pattern type
        pattern_types = [p.pattern_type for p in strongest_group]
        composite_name = self._determine_composite_name(pattern_types)
        
        # Calculate overall confidence
        overall_confidence = np.mean([p.confidence for p in strongest_group])
        
        # Determine formation stage
        formation_stage = self._assess_formation_stage(strongest_group)
        
        # Calculate targets and invalidation
        targets = self._calculate_composite_targets(strongest_group)
        invalidation = self._calculate_invalidation_level(strongest_group)
        
        composite = CompositePattern(
            pattern_name=composite_name,
            component_patterns=strongest_group,
            overall_confidence=overall_confidence,
            formation_stage=formation_stage,
            target_levels=targets,
            invalidation_level=invalidation,
            time_to_target=self._estimate_time_to_target(strongest_group)
        )
        
        self.composite_patterns.append(composite)
        
        return composite
    
    def get_rl_features(self, patterns: List[ProfilePattern]) -> np.ndarray:
        """
        Extract features for RL model from patterns
        """
        features = []
        
        # Pattern counts by type
        pattern_types = ['accumulation', 'distribution', 'breakout', 'p_shape', 'b_shape']
        for ptype in pattern_types:
            count = sum(1 for p in patterns if ptype in p.pattern_type.lower())
            features.append(count / 10)  # Normalized
        
        # Average confidence
        if patterns:
            avg_confidence = np.mean([p.confidence for p in patterns])
            max_confidence = max(p.confidence for p in patterns)
            features.extend([avg_confidence / 100, max_confidence / 100])
        else:
            features.extend([0, 0])
        
        # Trading bias distribution
        bullish = sum(1 for p in patterns if p.trading_bias == 'bullish')
        bearish = sum(1 for p in patterns if p.trading_bias == 'bearish')
        neutral = sum(1 for p in patterns if p.trading_bias == 'neutral')
        total = max(1, bullish + bearish + neutral)
        features.extend([bullish / total, bearish / total, neutral / total])
        
        # Risk-reward metrics
        if patterns:
            avg_rr = np.mean([p.risk_reward for p in patterns])
            max_rr = max(p.risk_reward for p in patterns)
            features.extend([avg_rr / 10, max_rr / 10])
        else:
            features.extend([0, 0])
        
        # Pattern freshness (how recent)
        if patterns:
            now = datetime.now()
            avg_age = np.mean([(now - p.end_time).total_seconds() / 3600 for p in patterns])
            features.append(np.exp(-avg_age / 24))  # Decay over 24 hours
        else:
            features.append(0)
        
        # Composite pattern features
        if self.composite_patterns:
            recent_composite = self.composite_patterns[-1]
            features.extend([
                recent_composite.overall_confidence / 100,
                1 if recent_composite.formation_stage == 'complete' else 0,
                len(recent_composite.target_levels) / 5
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    # Helper methods for pattern detection
    def _check_accumulation_pattern(self, profile: Any) -> bool:
        """Check if profile shows accumulation"""
        # High value area percentage
        if profile.value_area_percentage < 65:
            return False
        
        # POC near middle or lower
        poc_position = (profile.poc - profile.val) / (profile.vah - profile.val) if profile.vah != profile.val else 0.5
        if poc_position > 0.7:
            return False
        
        # Check for volume concentration
        volume_concentration = self._calculate_volume_concentration(profile)
        if volume_concentration < 0.6:
            return False
        
        return True
    
    def _check_distribution_pattern(self, profile: Any) -> bool:
        """Check if profile shows distribution"""
        # Wide value area
        if profile.value_area_percentage > 80:
            return False
        
        # POC weakness
        poc_weakness = self._calculate_poc_weakness(profile)
        if poc_weakness < 0.3:
            return False
        
        return True
    
    def _check_volume_breakout_setup(self, profile: Any) -> bool:
        """Check for volume breakout setup"""
        # Tight value area
        va_width_ratio = (profile.vah - profile.val) / profile.poc if profile.poc > 0 else 0
        if va_width_ratio > 0.05:
            return False
        
        # Volume buildup
        volume_buildup = self._calculate_volume_buildup(profile)
        if volume_buildup < 1.5:
            return False
        
        return True
    
    def _check_ledge_pattern(self, profile: Any) -> bool:
        """Check for ledge formation in market profile"""
        if not hasattr(profile, 'tpo_counts'):
            return False
        
        # Look for flat areas with consistent TPO counts
        tpo_values = list(profile.tpo_counts.values())
        if len(tpo_values) < 5:
            return False
        
        # Check for plateaus
        for i in range(len(tpo_values) - 4):
            window = tpo_values[i:i+5]
            if np.std(window) / np.mean(window) < 0.1:  # Low variation
                return True
        
        return False
    
    def _check_value_area_rule(self, current: Any, historical: List) -> bool:
        """Check for value area rule setup"""
        if len(historical) < 2:
            return False
        
        prev = historical[-1]
        
        # Check if price opened outside previous value area
        if hasattr(current, 'open_price') and hasattr(prev, 'vah') and hasattr(prev, 'val'):
            if current.open_price > prev.vah or current.open_price < prev.val:
                return True
        
        return False
    
    def _check_balance_breakout(self, profiles: List) -> bool:
        """Check for balance area breakout setup"""
        if len(profiles) < 5:
            return False
        
        # Check for overlapping value areas
        overlaps = 0
        for i in range(len(profiles) - 1):
            if hasattr(profiles[i], 'vah') and hasattr(profiles[i+1], 'val'):
                if profiles[i].val <= profiles[i+1].vah and profiles[i+1].val <= profiles[i].vah:
                    overlaps += 1
        
        return overlaps >= 3
    
    def _identify_wyckoff_pattern(self, profiles: List) -> Optional[ProfilePattern]:
        """Identify Wyckoff patterns"""
        if len(profiles) < 10:
            return None
        
        # Look for spring pattern (false breakdown)
        for i in range(5, len(profiles)):
            if self._check_spring_pattern(profiles[i-5:i+1]):
                return ProfilePattern(
                    pattern_type='wyckoff_spring',
                    confidence=75,
                    start_time=profiles[i-5].timestamp if hasattr(profiles[i-5], 'timestamp') else datetime.now(),
                    end_time=profiles[i].timestamp if hasattr(profiles[i], 'timestamp') else datetime.now(),
                    price_levels=(profiles[i].vah, profiles[i].val),
                    key_features={'pattern': 'spring', 'phase': 'accumulation'},
                    trading_bias='bullish',
                    expected_move=profiles[i].vah - profiles[i].val,
                    risk_reward=4.0
                )
        
        return None
    
    def _check_spring_pattern(self, window: List) -> bool:
        """Check for Wyckoff spring pattern"""
        if len(window) < 6:
            return False
        
        # Look for breakdown below support followed by quick recovery
        lows = [p.val if hasattr(p, 'val') else 0 for p in window]
        
        # Find potential spring
        min_idx = np.argmin(lows)
        if min_idx > 2 and min_idx < len(lows) - 1:
            # Check if it was a false breakdown
            if lows[-1] > lows[min_idx] * 1.01:  # Recovery above breakdown
                return True
        
        return False
    
    # Calculation helper methods
    def _calculate_pattern_confidence(self, pattern_type: str, profile: Any) -> float:
        """Calculate confidence for specific pattern"""
        if pattern_type not in self.pattern_library:
            return 0.0
        
        template = self.pattern_library[pattern_type]
        confidence = 50.0  # Base confidence
        
        # Add confidence based on feature matches
        for feature in template['features']:
            if self._check_feature(feature, profile):
                weight = template['confidence_weights'].get(feature, 0.3)
                confidence += weight * 50
        
        return min(100, confidence)
    
    def _calculate_volume_skew(self, profile: Any) -> float:
        """Calculate skewness of volume distribution"""
        if hasattr(profile, 'volumes'):
            return stats.skew(profile.volumes)
        return 0.0
    
    def _calculate_poc_weakness(self, profile: Any) -> float:
        """Calculate POC weakness indicator"""
        if not hasattr(profile, 'volumes'):
            return 0.0
        
        poc_volume = np.max(profile.volumes)
        avg_volume = np.mean(profile.volumes)
        
        if avg_volume > 0:
            return 1 - (poc_volume / (avg_volume * 3))
        return 0.0
    
    def _calculate_volume_concentration(self, profile: Any) -> float:
        """Calculate volume concentration metric"""
        if not hasattr(profile, 'volumes'):
            return 0.0
        
        # Use Gini coefficient
        sorted_volumes = np.sort(profile.volumes)
        n = len(sorted_volumes)
        index = np.arange(1, n + 1)
        
        return (2 * np.sum(index * sorted_volumes)) / (n * np.sum(sorted_volumes)) - (n + 1) / n
    
    def _calculate_volume_buildup(self, profile: Any) -> float:
        """Calculate volume buildup indicator"""
        if not hasattr(profile, 'total_volume'):
            return 1.0
        
        # Compare to average (would need historical data)
        return 1.5  # Placeholder
    
    def _calculate_compression(self, profile: Any) -> float:
        """Calculate price compression metric"""
        if hasattr(profile, 'vah') and hasattr(profile, 'val') and hasattr(profile, 'poc'):
            range_width = profile.vah - profile.val
            if profile.poc > 0:
                return 1 - (range_width / profile.poc)
        return 0.0
    
    def _determine_breakout_direction(self, profile: Any) -> int:
        """Determine likely breakout direction"""
        if not hasattr(profile, 'volumes'):
            return 0
        
        # Check volume distribution
        upper_half = profile.volumes[len(profile.volumes)//2:]
        lower_half = profile.volumes[:len(profile.volumes)//2]
        
        if np.sum(upper_half) > np.sum(lower_half) * 1.2:
            return 1  # Bullish
        elif np.sum(lower_half) > np.sum(upper_half) * 1.2:
            return -1  # Bearish
        
        return 0  # Neutral
    
    def _check_feature(self, feature: str, profile: Any) -> bool:
        """Check if profile has specific feature"""
        # Simplified feature checking
        feature_checks = {
            'increasing_volume': hasattr(profile, 'total_volume'),
            'narrowing_range': hasattr(profile, 'vah') and hasattr(profile, 'val'),
            'poc_stability': hasattr(profile, 'poc'),
            'high_volume_top': hasattr(profile, 'volumes'),
            'volume_spike': hasattr(profile, 'total_volume'),
        }
        
        return feature_checks.get(feature, False)
    
    def _check_top_heavy(self, profile: Any) -> bool:
        """Check if profile is top-heavy"""
        if hasattr(profile, 'volumes') and hasattr(profile, 'poc'):
            poc_idx = np.argmax(profile.volumes)
            if poc_idx > len(profile.volumes) * 0.7:
                return True
        return False
    
    def _calculate_shape_confidence(self, profile: Any, expected_shape: str) -> float:
        """Calculate confidence for shape-based pattern"""
        if hasattr(profile, 'profile_type') and profile.profile_type == expected_shape:
            return 80.0
        return 40.0
    
    def _identify_value_split(self, profile: Any) -> float:
        """Identify value area split point in double distribution"""
        if hasattr(profile, 'volumes'):
            # Find valley between peaks
            peaks, _ = signal.find_peaks(profile.volumes)
            if len(peaks) >= 2:
                valley_idx = (peaks[0] + peaks[1]) // 2
                return profile.price_levels[valley_idx] if hasattr(profile, 'price_levels') else 0
        return 0
    
    def _analyze_single_prints(self, profile: Any) -> int:
        """Analyze single prints for direction"""
        if not hasattr(profile, 'single_prints'):
            return 0
        
        if not profile.single_prints:
            return 0
        
        # Check position of single prints
        avg_price = np.mean([sp[0] for sp in profile.single_prints])
        
        if hasattr(profile, 'value_area'):
            poc = profile.value_area[1]
            if avg_price > poc:
                return 1  # Bullish
            else:
                return -1  # Bearish
        
        return 0
    
    def _calculate_ledge_confidence(self, profile: Any) -> float:
        """Calculate confidence for ledge pattern"""
        return 70.0  # Simplified
    
    def _calculate_ledge_width(self, profile: Any) -> float:
        """Calculate ledge width"""
        if hasattr(profile, 'value_area_width'):
            return profile.value_area_width
        return 0.0
    
    def _calculate_time_at_ledge(self, profile: Any) -> float:
        """Calculate time spent at ledge level"""
        return 0.5  # Simplified
    
    def _is_top_ledge(self, profile: Any) -> bool:
        """Check if ledge is at top of profile"""
        return True  # Simplified
    
    def _calculate_var_confidence(self, current: Any, historical: List) -> float:
        """Calculate Value Area Rule confidence"""
        return 75.0  # Simplified
    
    def _analyze_va_relationship(self, current: Any, historical: List) -> str:
        """Analyze value area relationships"""
        return 'overlapping'  # Simplified
    
    def _check_price_acceptance(self, current: Any, historical: List) -> bool:
        """Check if price is accepted in value area"""
        return True  # Simplified
    
    def _check_rotation(self, profiles: List) -> bool:
        """Check for market rotation"""
        return False  # Simplified
    
    def _determine_var_bias(self, current: Any, historical: List) -> str:
        """Determine Value Area Rule bias"""
        return 'bullish'  # Simplified
    
    def _calculate_balance_breakout_confidence(self, profiles: List) -> float:
        """Calculate confidence for balance breakout"""
        return 70.0  # Simplified
    
    def _determine_balance_breakout_direction(self, profiles: List) -> int:
        """Determine balance breakout direction"""
        return 1  # Simplified to bullish
    
    def _get_balance_range(self, profiles: List) -> Tuple[float, float]:
        """Get balance area range"""
        if profiles and hasattr(profiles[0], 'vah') and hasattr(profiles[0], 'val'):
            high = max(p.vah for p in profiles if hasattr(p, 'vah'))
            low = min(p.val for p in profiles if hasattr(p, 'val'))
            return (high, low)
        return (0, 0)
    
    def _calculate_compression_rate(self, profiles: List) -> float:
        """Calculate compression rate over time"""
        return 0.5  # Simplified
    
    def _check_volume_divergence(self, profiles: List) -> bool:
        """Check for volume divergence"""
        return False  # Simplified
    
    def _calculate_breakout_target(self, profiles: List) -> float:
        """Calculate breakout target"""
        if profiles and hasattr(profiles[0], 'vah') and hasattr(profiles[0], 'val'):
            return (profiles[-1].vah - profiles[-1].val) * 1.5
        return 0.0
    
    def _calculate_context_adjustment(self, pattern: ProfilePattern, context: Dict) -> float:
        """Adjust pattern probability based on market context"""
        return 1.0  # Simplified
    
    def _calculate_optimal_entry(self, pattern: ProfilePattern, context: Dict) -> float:
        """Calculate optimal entry price for pattern"""
        if pattern.trading_bias == 'bullish':
            return pattern.price_levels[1] * 1.01  # Just above support
        else:
            return pattern.price_levels[0] * 0.99  # Just below resistance
    
    def _group_compatible_patterns(self, patterns: List[ProfilePattern]) -> List[List[ProfilePattern]]:
        """Group compatible patterns"""
        # Simplified: group by bias
        groups = defaultdict(list)
        for pattern in patterns:
            groups[pattern.trading_bias].append(pattern)
        return list(groups.values())
    
    def _determine_composite_name(self, pattern_types: List[str]) -> str:
        """Determine name for composite pattern"""
        if 'accumulation' in pattern_types and 'volume_breakout' in pattern_types:
            return 'accumulation_breakout'
        return 'composite_pattern'
    
    def _assess_formation_stage(self, patterns: List[ProfilePattern]) -> str:
        """Assess formation stage of composite pattern"""
        avg_confidence = np.mean([p.confidence for p in patterns])
        if avg_confidence > 80:
            return 'complete'
        elif avg_confidence > 60:
            return 'forming'
        return 'failed'
    
    def _calculate_composite_targets(self, patterns: List[ProfilePattern]) -> List[float]:
        """Calculate targets for composite pattern"""
        targets = []
        for pattern in patterns:
            if pattern.expected_move:
                targets.append(pattern.price_levels[0] + pattern.expected_move)
        return targets[:3]  # Top 3 targets
    
    def _calculate_invalidation_level(self, patterns: List[ProfilePattern]) -> float:
        """Calculate invalidation level for composite pattern"""
        if patterns:
            return min(p.price_levels[1] for p in patterns) * 0.98
        return 0.0
    
    def _estimate_time_to_target(self, patterns: List[ProfilePattern]) -> timedelta:
        """Estimate time to reach target"""
        return timedelta(days=3)  # Simplified
    
    def _calculate_feature_similarity(self, feature_value: Any, template_value: Any) -> float:
        """Calculate similarity between feature and template"""
        # Simplified similarity calculation
        return 0.8