"""
Volume-Based Support and Resistance Levels
Advanced detection and analysis of support/resistance levels using volume profile analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
from scipy import stats, signal, interpolate
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class VolumeLevel:
    """Volume-based support/resistance level"""
    price: float
    level_type: str  # 'support', 'resistance', 'pivot'
    strength: float  # 0-100 strength score
    volume: float
    touch_count: int
    last_test: datetime
    first_established: datetime
    break_count: int  # Number of times level was broken
    recovery_rate: float  # How quickly price recovered after breaks
    context: str  # 'POC', 'VAH', 'VAL', 'HVN', 'LVN'
    confidence: float  # 0-100 confidence in level
    
    @property
    def age_days(self) -> float:
        """Age of level in days"""
        return (datetime.now() - self.first_established).total_seconds() / 86400
    
    @property
    def test_frequency(self) -> float:
        """Average tests per day"""
        if self.age_days > 0:
            return self.touch_count / self.age_days
        return 0
    
    @property
    def reliability_score(self) -> float:
        """Reliability score based on tests and breaks"""
        if self.touch_count == 0:
            return 0
        return (1 - self.break_count / self.touch_count) * 100


@dataclass
class LevelZone:
    """Zone of support/resistance with multiple levels"""
    zone_center: float
    zone_range: Tuple[float, float]  # (upper, lower)
    levels: List[VolumeLevel]
    zone_strength: float
    dominant_type: str  # 'support', 'resistance'
    formation_period: Tuple[datetime, datetime]
    
    @property
    def zone_width(self) -> float:
        """Width of zone"""
        return self.zone_range[0] - self.zone_range[1]
    
    @property
    def level_density(self) -> float:
        """Density of levels in zone"""
        return len(self.levels) / max(self.zone_width, 0.001)


@dataclass
class LevelTest:
    """Record of a level test"""
    timestamp: datetime
    price: float
    level_price: float
    test_type: str  # 'touch', 'break', 'hold'
    volume: float
    duration: float  # Seconds at level
    reaction_strength: float  # Strength of price reaction
    
    @property
    def distance_from_level(self) -> float:
        """Distance from actual level"""
        return abs(self.price - self.level_price)


class VolumeLevelAnalyzer:
    """Advanced volume-based support/resistance level analyzer"""
    
    def __init__(self,
                 level_tolerance: float = 0.002,  # 0.2% tolerance
                 min_volume_threshold: float = 1000,
                 min_touch_count: int = 2,
                 level_expiry_days: int = 30,
                 zone_proximity: float = 0.005):  # 0.5% for zone grouping
        """
        Initialize volume level analyzer
        
        Args:
            level_tolerance: Price tolerance for level identification
            min_volume_threshold: Minimum volume to establish level
            min_touch_count: Minimum touches to confirm level
            level_expiry_days: Days after which unused levels expire
            zone_proximity: Proximity threshold for grouping levels into zones
        """
        self.level_tolerance = level_tolerance
        self.min_volume_threshold = min_volume_threshold
        self.min_touch_count = min_touch_count
        self.level_expiry_days = level_expiry_days
        self.zone_proximity = zone_proximity
        
        # Storage
        self.volume_levels: List[VolumeLevel] = []
        self.level_zones: List[LevelZone] = []
        self.level_tests: List[LevelTest] = []
        self.historical_breaks: List[Dict] = []
        
        # Analysis tools
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.level_performance: Dict[float, Dict] = defaultdict(dict)
    
    def extract_levels_from_profile(self,
                                   volume_profile: Any,
                                   market_profile: Optional[Any] = None) -> List[VolumeLevel]:
        """
        Extract support/resistance levels from volume profile
        """
        levels = []
        
        # Extract key profile levels
        if hasattr(volume_profile, 'poc'):
            poc_level = VolumeLevel(
                price=volume_profile.poc,
                level_type='pivot',
                strength=100,  # POC is strongest level
                volume=np.max(volume_profile.volumes) if hasattr(volume_profile, 'volumes') else 0,
                touch_count=1,
                last_test=volume_profile.timestamp,
                first_established=volume_profile.timestamp,
                break_count=0,
                recovery_rate=0,
                context='POC',
                confidence=95
            )
            levels.append(poc_level)
        
        # Value Area High/Low
        if hasattr(volume_profile, 'vah') and hasattr(volume_profile, 'val'):
            vah_level = VolumeLevel(
                price=volume_profile.vah,
                level_type='resistance',
                strength=85,
                volume=self._get_volume_at_price(volume_profile, volume_profile.vah),
                touch_count=1,
                last_test=volume_profile.timestamp,
                first_established=volume_profile.timestamp,
                break_count=0,
                recovery_rate=0,
                context='VAH',
                confidence=80
            )
            levels.append(vah_level)
            
            val_level = VolumeLevel(
                price=volume_profile.val,
                level_type='support',
                strength=85,
                volume=self._get_volume_at_price(volume_profile, volume_profile.val),
                touch_count=1,
                last_test=volume_profile.timestamp,
                first_established=volume_profile.timestamp,
                break_count=0,
                recovery_rate=0,
                context='VAL',
                confidence=80
            )
            levels.append(val_level)
        
        # High Volume Nodes
        if hasattr(volume_profile, 'volumes') and hasattr(volume_profile, 'price_levels'):
            hvn_levels = self._identify_hvn_levels(volume_profile)
            levels.extend(hvn_levels)
        
        # Low Volume Nodes (potential breakout levels)
        if hasattr(volume_profile, 'volumes') and hasattr(volume_profile, 'price_levels'):
            lvn_levels = self._identify_lvn_levels(volume_profile)
            levels.extend(lvn_levels)
        
        # Market profile levels
        if market_profile:
            mp_levels = self._extract_market_profile_levels(market_profile)
            levels.extend(mp_levels)
        
        # Update existing levels or add new ones
        updated_levels = self._update_or_add_levels(levels)
        
        return updated_levels
    
    def detect_level_interactions(self,
                                 price_data: pd.DataFrame,
                                 volume_data: Optional[pd.DataFrame] = None) -> List[LevelTest]:
        """
        Detect price interactions with established levels
        """
        interactions = []
        
        for idx, row in price_data.iterrows():
            timestamp = pd.to_datetime(idx)
            high = row.get('high', row.get('price', 0))
            low = row.get('low', row.get('price', 0))
            close = row.get('close', row.get('price', 0))
            volume = row.get('volume', 0)
            
            # Check interactions with all levels
            for level in self.volume_levels:
                interaction = self._check_level_interaction(
                    level, timestamp, high, low, close, volume
                )
                if interaction:
                    interactions.append(interaction)
                    self.level_tests.append(interaction)
                    
                    # Update level statistics
                    self._update_level_from_test(level, interaction)
        
        return interactions
    
    def calculate_level_strength(self,
                                level: VolumeLevel,
                                current_price: float,
                                lookback_days: int = 30) -> float:
        """
        Calculate dynamic strength of a level
        """
        base_strength = level.strength
        
        # Age factor (newer levels are less reliable)
        age_factor = min(1.0, level.age_days / 10)  # Full strength after 10 days
        
        # Touch frequency factor
        frequency_factor = min(1.0, level.test_frequency * 5)  # Optimal around 0.2 tests/day
        
        # Reliability factor
        reliability_factor = level.reliability_score / 100
        
        # Volume factor
        volume_factor = min(1.0, level.volume / self.min_volume_threshold / 5)
        
        # Distance factor (levels closer to current price are more relevant)
        distance = abs(current_price - level.price) / current_price
        distance_factor = np.exp(-distance * 100)  # Exponential decay
        
        # Context factor
        context_weights = {
            'POC': 1.0,
            'VAH': 0.9,
            'VAL': 0.9,
            'HVN': 0.8,
            'LVN': 0.6
        }
        context_factor = context_weights.get(level.context, 0.7)
        
        # Recent activity factor
        recent_tests = [t for t in self.level_tests 
                       if abs(t.level_price - level.price) < level.price * self.level_tolerance
                       and (datetime.now() - t.timestamp).days <= lookback_days]
        
        if recent_tests:
            recent_activity = len(recent_tests) / lookback_days
            activity_factor = min(1.0, recent_activity * 10)
        else:
            activity_factor = 0.5
        
        # Combine all factors
        dynamic_strength = base_strength * age_factor * frequency_factor * \
                          reliability_factor * volume_factor * distance_factor * \
                          context_factor * activity_factor
        
        return min(100, dynamic_strength)
    
    def identify_level_zones(self,
                           levels: Optional[List[VolumeLevel]] = None) -> List[LevelZone]:
        """
        Group nearby levels into zones
        """
        if levels is None:
            levels = self.volume_levels
        
        if not levels:
            return []
        
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda l: l.price)
        
        zones = []
        current_zone_levels = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            current_level = sorted_levels[i]
            last_in_zone = current_zone_levels[-1]
            
            # Check if level belongs to current zone
            distance_pct = (current_level.price - last_in_zone.price) / last_in_zone.price
            
            if distance_pct <= self.zone_proximity:
                current_zone_levels.append(current_level)
            else:
                # Create zone from current levels
                if len(current_zone_levels) >= 2:
                    zone = self._create_level_zone(current_zone_levels)
                    zones.append(zone)
                
                # Start new zone
                current_zone_levels = [current_level]
        
        # Create final zone
        if len(current_zone_levels) >= 2:
            zone = self._create_level_zone(current_zone_levels)
            zones.append(zone)
        
        self.level_zones = zones
        return zones
    
    def predict_level_hold(self,
                          level: VolumeLevel,
                          approach_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict probability that level will hold
        """
        prediction = {
            'hold_probability': 0.5,
            'break_probability': 0.5,
            'confidence': 0,
            'expected_reaction': 0,
            'factors': {}
        }
        
        # Level strength factor
        strength_factor = level.reliability_score / 100
        prediction['factors']['strength'] = strength_factor
        
        # Volume approach factor
        approach_volume = approach_data.get('volume', 0)
        avg_volume = approach_data.get('avg_volume', approach_volume)
        volume_ratio = approach_volume / max(avg_volume, 1)
        
        if volume_ratio > 2:  # High volume approach
            volume_factor = 0.3  # More likely to break
        elif volume_ratio < 0.5:  # Low volume approach
            volume_factor = 0.8  # More likely to hold
        else:
            volume_factor = 0.6
        
        prediction['factors']['volume'] = volume_factor
        
        # Approach angle factor
        approach_angle = approach_data.get('approach_angle', 0)  # Degrees
        if abs(approach_angle) > 45:  # Steep approach
            angle_factor = 0.4  # More likely to break
        else:
            angle_factor = 0.7
        
        prediction['factors']['angle'] = angle_factor
        
        # Multiple test factor
        if level.touch_count > 3:
            multiple_test_factor = max(0.2, 1 - (level.touch_count - 3) * 0.1)
        else:
            multiple_test_factor = 0.8
        
        prediction['factors']['multiple_tests'] = multiple_test_factor
        
        # Market context factor
        market_trend = approach_data.get('trend', 'neutral')
        if ((level.level_type == 'resistance' and market_trend == 'bullish') or
            (level.level_type == 'support' and market_trend == 'bearish')):
            context_factor = 0.4  # Counter-trend, more likely to break
        else:
            context_factor = 0.7
        
        prediction['factors']['market_context'] = context_factor
        
        # Time since last test
        time_factor = min(1.0, (datetime.now() - level.last_test).days / 7)
        prediction['factors']['time_decay'] = time_factor
        
        # Combine factors
        hold_prob = (strength_factor * 0.3 + 
                    volume_factor * 0.25 + 
                    angle_factor * 0.2 + 
                    multiple_test_factor * 0.15 + 
                    context_factor * 0.1)
        
        prediction['hold_probability'] = hold_prob
        prediction['break_probability'] = 1 - hold_prob
        prediction['confidence'] = abs(hold_prob - 0.5) * 200  # 0-100 scale
        
        # Expected reaction strength
        if hold_prob > 0.6:
            prediction['expected_reaction'] = min(100, level.strength * hold_prob)
        else:
            prediction['expected_reaction'] = max(-100, -level.strength * (1 - hold_prob))
        
        return prediction
    
    def find_key_levels_for_price(self,
                                 current_price: float,
                                 direction: str = 'both',
                                 max_distance_pct: float = 5.0,
                                 max_levels: int = 5) -> Dict[str, List[VolumeLevel]]:
        """
        Find key levels near current price
        """
        result = {
            'support': [],
            'resistance': [],
            'pivots': []
        }
        
        # Filter levels by distance
        relevant_levels = []
        for level in self.volume_levels:
            distance_pct = abs(level.price - current_price) / current_price * 100
            if distance_pct <= max_distance_pct:
                # Update dynamic strength
                level.strength = self.calculate_level_strength(level, current_price)
                relevant_levels.append(level)
        
        # Sort by strength
        relevant_levels.sort(key=lambda l: l.strength, reverse=True)
        
        # Categorize levels
        for level in relevant_levels:
            if level.price < current_price and (direction in ['both', 'down']):
                if level.level_type in ['support', 'pivot'] and len(result['support']) < max_levels:
                    result['support'].append(level)
            elif level.price > current_price and (direction in ['both', 'up']):
                if level.level_type in ['resistance', 'pivot'] and len(result['resistance']) < max_levels:
                    result['resistance'].append(level)
            elif level.level_type == 'pivot':
                if len(result['pivots']) < max_levels:
                    result['pivots'].append(level)
        
        # Sort each category by distance from current price
        for key in result:
            result[key].sort(key=lambda l: abs(l.price - current_price))
        
        return result
    
    def calculate_breakout_targets(self,
                                  broken_level: VolumeLevel,
                                  direction: str) -> List[float]:
        """
        Calculate targets after level breakout
        """
        targets = []
        
        # Measured move target
        if hasattr(broken_level, 'formation_range'):
            formation_height = broken_level.formation_range
        else:
            formation_height = broken_level.price * 0.02  # 2% default
        
        if direction == 'up':
            targets.append(broken_level.price + formation_height)
            targets.append(broken_level.price + formation_height * 1.618)  # Fibonacci extension
        else:
            targets.append(broken_level.price - formation_height)
            targets.append(broken_level.price - formation_height * 1.618)
        
        # Next significant level target
        if direction == 'up':
            next_resistance = self._find_next_level_above(broken_level.price)
            if next_resistance:
                targets.append(next_resistance.price)
        else:
            next_support = self._find_next_level_below(broken_level.price)
            if next_support:
                targets.append(next_support.price)
        
        # Round number targets
        round_targets = self._find_round_number_targets(broken_level.price, direction)
        targets.extend(round_targets)
        
        # Remove duplicates and sort
        targets = sorted(list(set(targets)))
        
        return targets[:3]  # Top 3 targets
    
    def analyze_level_confluence(self,
                                price: float,
                                tolerance_pct: float = 0.5) -> Dict[str, Any]:
        """
        Analyze confluence of multiple analysis methods at price level
        """
        confluence_score = 0
        contributing_factors = []
        
        # Check volume profile confluence
        for level in self.volume_levels:
            distance_pct = abs(level.price - price) / price * 100
            if distance_pct <= tolerance_pct:
                confluence_score += level.strength * 0.01
                contributing_factors.append({
                    'type': 'volume_level',
                    'context': level.context,
                    'strength': level.strength
                })
        
        # Check zone confluence
        for zone in self.level_zones:
            if zone.zone_range[1] <= price <= zone.zone_range[0]:
                confluence_score += zone.zone_strength * 0.01
                contributing_factors.append({
                    'type': 'volume_zone',
                    'strength': zone.zone_strength,
                    'level_count': len(zone.levels)
                })
        
        # Check technical analysis confluence (would integrate with other modules)
        # Fibonacci levels, trend lines, moving averages, etc.
        technical_confluence = self._check_technical_confluence(price)
        confluence_score += technical_confluence
        
        # Check round number confluence
        round_number_strength = self._check_round_number(price)
        confluence_score += round_number_strength
        if round_number_strength > 0:
            contributing_factors.append({
                'type': 'round_number',
                'strength': round_number_strength
            })
        
        return {
            'confluence_score': min(100, confluence_score),
            'rating': self._rate_confluence(confluence_score),
            'contributing_factors': contributing_factors,
            'recommendation': self._get_confluence_recommendation(confluence_score)
        }
    
    def generate_level_alerts(self,
                             current_price: float,
                             price_direction: str = 'unknown') -> List[Dict[str, Any]]:
        """
        Generate alerts for approaching key levels
        """
        alerts = []
        
        # Find levels within alert range
        alert_range_pct = 1.0  # 1% alert range
        
        for level in self.volume_levels:
            distance_pct = abs(level.price - current_price) / current_price * 100
            
            if distance_pct <= alert_range_pct:
                # Calculate approach characteristics
                approach_info = self._analyze_level_approach(level, current_price, price_direction)
                
                # Generate alert
                alert = {
                    'level': level,
                    'alert_type': self._determine_alert_type(level, current_price, approach_info),
                    'distance': distance_pct,
                    'expected_reaction': approach_info['expected_reaction'],
                    'confidence': approach_info['confidence'],
                    'recommended_action': approach_info['recommended_action'],
                    'risk_level': approach_info['risk_level']
                }
                
                alerts.append(alert)
        
        # Sort by relevance (distance and strength)
        alerts.sort(key=lambda a: (a['distance'], -a['level'].strength))
        
        return alerts[:5]  # Top 5 alerts
    
    def get_rl_features(self,
                       current_price: float,
                       lookback_levels: int = 10) -> np.ndarray:
        """
        Extract features for RL model from volume levels
        """
        features = []
        
        # Find nearest levels
        key_levels = self.find_key_levels_for_price(current_price, max_levels=5)
        
        # Support level features
        if key_levels['support']:
            nearest_support = key_levels['support'][0]
            features.extend([
                (current_price - nearest_support.price) / current_price,  # Distance to support
                nearest_support.strength / 100,
                nearest_support.reliability_score / 100,
                min(1.0, nearest_support.touch_count / 10)
            ])
        else:
            features.extend([0.1, 0, 0, 0])  # No nearby support
        
        # Resistance level features
        if key_levels['resistance']:
            nearest_resistance = key_levels['resistance'][0]
            features.extend([
                (nearest_resistance.price - current_price) / current_price,  # Distance to resistance
                nearest_resistance.strength / 100,
                nearest_resistance.reliability_score / 100,
                min(1.0, nearest_resistance.touch_count / 10)
            ])
        else:
            features.extend([0.1, 0, 0, 0])  # No nearby resistance
        
        # Level density features
        levels_above = len([l for l in self.volume_levels if l.price > current_price and l.price < current_price * 1.05])
        levels_below = len([l for l in self.volume_levels if l.price < current_price and l.price > current_price * 0.95])
        features.extend([levels_above / 10, levels_below / 10])
        
        # Zone features
        current_zone = self._find_zone_for_price(current_price)
        if current_zone:
            features.extend([
                current_zone.zone_strength / 100,
                len(current_zone.levels) / 10,
                (current_price - current_zone.zone_center) / current_zone.zone_center
            ])
        else:
            features.extend([0, 0, 0])
        
        # Recent interaction features
        recent_tests = [t for t in self.level_tests if (datetime.now() - t.timestamp).days <= 7]
        if recent_tests:
            avg_reaction = np.mean([t.reaction_strength for t in recent_tests])
            features.append(avg_reaction / 100)
        else:
            features.append(0)
        
        # Confluence features
        confluence = self.analyze_level_confluence(current_price)
        features.extend([
            confluence['confluence_score'] / 100,
            len(confluence['contributing_factors']) / 10
        ])
        
        return np.array(features)
    
    # Private helper methods
    def _get_volume_at_price(self, profile: Any, price: float) -> float:
        """Get volume at specific price level"""
        if not hasattr(profile, 'price_levels') or not hasattr(profile, 'volumes'):
            return 0
        
        # Find closest price level
        closest_idx = np.argmin(np.abs(profile.price_levels - price))
        return profile.volumes[closest_idx]
    
    def _identify_hvn_levels(self, profile: Any) -> List[VolumeLevel]:
        """Identify High Volume Node levels"""
        levels = []
        
        if not hasattr(profile, 'volumes') or not hasattr(profile, 'price_levels'):
            return levels
        
        # Find peaks in volume
        avg_volume = np.mean(profile.volumes)
        threshold = avg_volume * 1.5
        
        peaks, properties = signal.find_peaks(
            profile.volumes,
            height=threshold,
            prominence=avg_volume * 0.3
        )
        
        for peak_idx in peaks:
            level = VolumeLevel(
                price=profile.price_levels[peak_idx],
                level_type='pivot',
                strength=min(100, (profile.volumes[peak_idx] / avg_volume) * 30),
                volume=profile.volumes[peak_idx],
                touch_count=1,
                last_test=profile.timestamp,
                first_established=profile.timestamp,
                break_count=0,
                recovery_rate=0,
                context='HVN',
                confidence=70
            )
            levels.append(level)
        
        return levels
    
    def _identify_lvn_levels(self, profile: Any) -> List[VolumeLevel]:
        """Identify Low Volume Node levels"""
        levels = []
        
        if not hasattr(profile, 'volumes') or not hasattr(profile, 'price_levels'):
            return levels
        
        # Find valleys in volume
        avg_volume = np.mean(profile.volumes)
        threshold = avg_volume * 0.3
        
        valleys, properties = signal.find_peaks(
            -profile.volumes,
            height=-threshold
        )
        
        for valley_idx in valleys:
            if profile.volumes[valley_idx] < threshold:
                level = VolumeLevel(
                    price=profile.price_levels[valley_idx],
                    level_type='pivot',
                    strength=max(20, 100 - (profile.volumes[valley_idx] / avg_volume) * 50),
                    volume=profile.volumes[valley_idx],
                    touch_count=1,
                    last_test=profile.timestamp,
                    first_established=profile.timestamp,
                    break_count=0,
                    recovery_rate=0,
                    context='LVN',
                    confidence=60
                )
                levels.append(level)
        
        return levels
    
    def _extract_market_profile_levels(self, market_profile: Any) -> List[VolumeLevel]:
        """Extract levels from market profile"""
        levels = []
        
        # Initial Balance levels
        if hasattr(market_profile, 'initial_balance'):
            ib_high, ib_low = market_profile.initial_balance
            
            levels.append(VolumeLevel(
                price=ib_high,
                level_type='resistance',
                strength=75,
                volume=0,
                touch_count=1,
                last_test=market_profile.date,
                first_established=market_profile.date,
                break_count=0,
                recovery_rate=0,
                context='IB_HIGH',
                confidence=75
            ))
            
            levels.append(VolumeLevel(
                price=ib_low,
                level_type='support',
                strength=75,
                volume=0,
                touch_count=1,
                last_test=market_profile.date,
                first_established=market_profile.date,
                break_count=0,
                recovery_rate=0,
                context='IB_LOW',
                confidence=75
            ))
        
        # Single print levels
        if hasattr(market_profile, 'single_prints'):
            for sp_price, sp_letter in market_profile.single_prints:
                levels.append(VolumeLevel(
                    price=sp_price,
                    level_type='pivot',
                    strength=60,
                    volume=0,
                    touch_count=1,
                    last_test=market_profile.date,
                    first_established=market_profile.date,
                    break_count=0,
                    recovery_rate=0,
                    context='SINGLE_PRINT',
                    confidence=70
                ))
        
        return levels
    
    def _update_or_add_levels(self, new_levels: List[VolumeLevel]) -> List[VolumeLevel]:
        """Update existing levels or add new ones"""
        updated_levels = []
        
        for new_level in new_levels:
            # Check if level already exists
            existing_level = self._find_existing_level(new_level.price)
            
            if existing_level:
                # Update existing level
                existing_level.touch_count += 1
                existing_level.last_test = new_level.last_test
                existing_level.volume = max(existing_level.volume, new_level.volume)
                
                # Update strength based on confirmations
                existing_level.strength = min(100, existing_level.strength + 5)
                existing_level.confidence = min(100, existing_level.confidence + 3)
                
                updated_levels.append(existing_level)
            else:
                # Add new level
                self.volume_levels.append(new_level)
                updated_levels.append(new_level)
        
        # Clean up expired levels
        self._cleanup_expired_levels()
        
        return updated_levels
    
    def _find_existing_level(self, price: float) -> Optional[VolumeLevel]:
        """Find existing level near price"""
        for level in self.volume_levels:
            if abs(level.price - price) / price <= self.level_tolerance:
                return level
        return None
    
    def _cleanup_expired_levels(self):
        """Remove expired or weak levels"""
        current_time = datetime.now()
        
        # Remove expired levels
        self.volume_levels = [
            level for level in self.volume_levels
            if (current_time - level.last_test).days <= self.level_expiry_days
        ]
        
        # Remove weak levels with low confidence and few touches
        self.volume_levels = [
            level for level in self.volume_levels
            if level.confidence >= 50 or level.touch_count >= self.min_touch_count
        ]
    
    def _check_level_interaction(self,
                                level: VolumeLevel,
                                timestamp: datetime,
                                high: float,
                                low: float,
                                close: float,
                                volume: float) -> Optional[LevelTest]:
        """Check if price interacted with level"""
        # Check if price came near level
        tolerance = level.price * self.level_tolerance
        
        if low <= level.price + tolerance and high >= level.price - tolerance:
            # Determine interaction type
            if low <= level.price <= high:
                # Direct interaction
                if level.level_type == 'support' and close > level.price:
                    test_type = 'hold'
                elif level.level_type == 'resistance' and close < level.price:
                    test_type = 'hold'
                elif abs(close - level.price) <= tolerance:
                    test_type = 'touch'
                else:
                    test_type = 'break'
            else:
                test_type = 'touch'
            
            # Calculate reaction strength
            if test_type == 'hold':
                if level.level_type == 'support':
                    reaction_strength = (close - low) / level.price * 100
                else:
                    reaction_strength = (high - close) / level.price * 100
            else:
                reaction_strength = abs(close - level.price) / level.price * 100
            
            return LevelTest(
                timestamp=timestamp,
                price=close,
                level_price=level.price,
                test_type=test_type,
                volume=volume,
                duration=0,  # Would need tick data for accurate duration
                reaction_strength=reaction_strength
            )
        
        return None
    
    def _update_level_from_test(self, level: VolumeLevel, test: LevelTest):
        """Update level statistics from test"""
        level.touch_count += 1
        level.last_test = test.timestamp
        
        if test.test_type == 'break':
            level.break_count += 1
            
            # Track recovery if price comes back
            # This would need follow-up logic
        elif test.test_type == 'hold':
            # Strengthen level on successful hold
            level.strength = min(100, level.strength + 2)
            level.confidence = min(100, level.confidence + 1)
    
    def _create_level_zone(self, levels: List[VolumeLevel]) -> LevelZone:
        """Create zone from grouped levels"""
        prices = [level.price for level in levels]
        zone_center = np.mean(prices)
        zone_range = (max(prices), min(prices))
        
        # Calculate zone strength
        avg_strength = np.mean([level.strength for level in levels])
        zone_strength = min(100, avg_strength * (1 + len(levels) * 0.1))
        
        # Determine dominant type
        support_count = sum(1 for level in levels if level.level_type == 'support')
        resistance_count = sum(1 for level in levels if level.level_type == 'resistance')
        
        if support_count > resistance_count:
            dominant_type = 'support'
        elif resistance_count > support_count:
            dominant_type = 'resistance'
        else:
            dominant_type = 'pivot'
        
        # Formation period
        earliest = min(level.first_established for level in levels)
        latest = max(level.last_test for level in levels)
        
        return LevelZone(
            zone_center=zone_center,
            zone_range=zone_range,
            levels=levels,
            zone_strength=zone_strength,
            dominant_type=dominant_type,
            formation_period=(earliest, latest)
        )
    
    def _find_next_level_above(self, price: float) -> Optional[VolumeLevel]:
        """Find next significant level above price"""
        candidates = [level for level in self.volume_levels 
                     if level.price > price and level.strength > 60]
        
        if candidates:
            return min(candidates, key=lambda l: l.price)
        return None
    
    def _find_next_level_below(self, price: float) -> Optional[VolumeLevel]:
        """Find next significant level below price"""
        candidates = [level for level in self.volume_levels 
                     if level.price < price and level.strength > 60]
        
        if candidates:
            return max(candidates, key=lambda l: l.price)
        return None
    
    def _find_round_number_targets(self, price: float, direction: str) -> List[float]:
        """Find round number targets"""
        targets = []
        
        # Determine round number levels
        if price > 1000:
            increments = [50, 100, 250, 500]
        elif price > 100:
            increments = [5, 10, 25, 50]
        elif price > 10:
            increments = [0.5, 1, 2.5, 5]
        else:
            increments = [0.05, 0.1, 0.25, 0.5]
        
        for increment in increments:
            if direction == 'up':
                target = np.ceil(price / increment) * increment
                if target > price:
                    targets.append(target)
            else:
                target = np.floor(price / increment) * increment
                if target < price:
                    targets.append(target)
        
        return targets[:2]  # Top 2 round number targets
    
    def _check_technical_confluence(self, price: float) -> float:
        """Check technical analysis confluence at price"""
        # Placeholder for technical analysis integration
        # Would integrate with moving averages, Fibonacci, trend lines, etc.
        return 0
    
    def _check_round_number(self, price: float) -> float:
        """Check if price is near round number"""
        # Check various round number levels
        round_levels = []
        
        if price > 1000:
            round_levels = [50, 100, 250, 500, 1000]
        elif price > 100:
            round_levels = [5, 10, 25, 50, 100]
        elif price > 10:
            round_levels = [0.5, 1, 2.5, 5, 10]
        else:
            round_levels = [0.05, 0.1, 0.25, 0.5, 1]
        
        for level in round_levels:
            remainder = price % level
            distance_to_round = min(remainder, level - remainder)
            distance_pct = distance_to_round / price * 100
            
            if distance_pct < 0.1:  # Very close to round number
                return 30
            elif distance_pct < 0.5:  # Close to round number
                return 15
        
        return 0
    
    def _rate_confluence(self, score: float) -> str:
        """Rate confluence strength"""
        if score >= 80:
            return 'very_strong'
        elif score >= 60:
            return 'strong'
        elif score >= 40:
            return 'moderate'
        elif score >= 20:
            return 'weak'
        else:
            return 'minimal'
    
    def _get_confluence_recommendation(self, score: float) -> str:
        """Get trading recommendation based on confluence"""
        if score >= 70:
            return 'high_probability_level'
        elif score >= 50:
            return 'significant_level'
        elif score >= 30:
            return 'moderate_level'
        else:
            return 'weak_level'
    
    def _analyze_level_approach(self,
                               level: VolumeLevel,
                               current_price: float,
                               direction: str) -> Dict[str, Any]:
        """Analyze characteristics of level approach"""
        return {
            'expected_reaction': level.strength,
            'confidence': level.confidence,
            'recommended_action': 'watch',
            'risk_level': 'medium'
        }
    
    def _determine_alert_type(self,
                             level: VolumeLevel,
                             current_price: float,
                             approach_info: Dict) -> str:
        """Determine type of alert for level"""
        if current_price < level.price:
            return 'approaching_resistance'
        else:
            return 'approaching_support'
    
    def _find_zone_for_price(self, price: float) -> Optional[LevelZone]:
        """Find zone containing price"""
        for zone in self.level_zones:
            if zone.zone_range[1] <= price <= zone.zone_range[0]:
                return zone
        return None