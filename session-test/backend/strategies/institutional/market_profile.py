"""
Market Profile and Time Price Opportunity (TPO) Analysis
Advanced market profile analysis based on auction market theory
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
import logging
from collections import defaultdict, Counter
import string
from scipy import stats, signal
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class TPOProfile:
    """Time Price Opportunity profile structure"""
    date: datetime
    price_levels: np.ndarray
    tpo_matrix: Dict[float, List[str]]  # Price -> List of TPO letters
    tpo_counts: Dict[float, int]  # Price -> TPO count
    initial_balance: Tuple[float, float]  # IB range (high, low)
    value_area: Tuple[float, float, float]  # VAH, POC, VAL
    profile_type: str  # 'normal', 'b-shaped', 'p-shaped', 'd-shaped'
    range_extension: str  # 'none', 'above', 'below', 'both'
    single_prints: List[Tuple[float, str]]  # Isolated TPOs
    poor_highs: List[float]
    poor_lows: List[float]
    
    @property
    def ib_range(self) -> float:
        """Initial Balance range"""
        return self.initial_balance[0] - self.initial_balance[1]
    
    @property
    def day_range(self) -> float:
        """Total day range"""
        if self.price_levels.size > 0:
            return self.price_levels.max() - self.price_levels.min()
        return 0
    
    @property
    def range_extension_ratio(self) -> float:
        """Range extension as ratio of IB"""
        if self.ib_range > 0:
            return self.day_range / self.ib_range
        return 1.0
    
    @property
    def value_area_width(self) -> float:
        """Width of value area"""
        return self.value_area[0] - self.value_area[2]
    
    def get_tpo_string(self, price: float) -> str:
        """Get TPO string for a price level"""
        return ''.join(self.tpo_matrix.get(price, []))


@dataclass
class MarketStructure:
    """Market structure analysis from profile"""
    trend: str  # 'up', 'down', 'balanced'
    balance_areas: List[Tuple[float, float]]  # List of balance/consolidation areas
    breakout_levels: Dict[str, float]  # 'upper', 'lower' breakout levels
    acceptance_levels: List[float]  # Prices with high time acceptance
    rejection_levels: List[float]  # Prices with rejection (single prints)
    composite_value: Tuple[float, float]  # Composite VAH, VAL from multiple days
    migration: str  # 'higher', 'lower', 'overlapping'
    confidence: float  # 0-100 confidence score


class MarketProfileAnalyzer:
    """Advanced Market Profile and TPO analyzer"""
    
    def __init__(self,
                 tick_size: float = 0.25,
                 tpo_size: int = 30,  # Minutes per TPO
                 value_area_pct: float = 70.0,
                 ib_periods: int = 2,  # Number of TPO periods for Initial Balance
                 session_start: time = time(9, 30),
                 session_end: time = time(16, 0)):
        """
        Initialize Market Profile analyzer
        
        Args:
            tick_size: Minimum price increment for TPO rows
            tpo_size: Minutes per TPO period (typically 30)
            value_area_pct: Percentage for value area calculation
            ib_periods: Number of initial periods for IB
            session_start: Market session start time
            session_end: Market session end time
        """
        self.tick_size = tick_size
        self.tpo_size = tpo_size
        self.value_area_pct = value_area_pct
        self.ib_periods = ib_periods
        self.session_start = session_start
        self.session_end = session_end
        
        # TPO letters (A-Z, then a-z for extended sessions)
        self.tpo_letters = list(string.ascii_uppercase) + list(string.ascii_lowercase)
        
        # Profile cache
        self.profile_cache: Dict[str, TPOProfile] = {}
        self.composite_profiles: Dict[str, Any] = {}
        
    def build_tpo_profile(self,
                         data: pd.DataFrame,
                         date: Optional[datetime] = None) -> TPOProfile:
        """
        Build TPO profile from price data
        
        Args:
            data: DataFrame with timestamp, high, low, close columns
            date: Profile date (uses data date if not provided)
        """
        if data.empty:
            raise ValueError("Empty data provided")
        
        if date is None:
            date = pd.to_datetime(data.index[0]).date()
        
        # Initialize TPO matrix
        tpo_matrix = defaultdict(list)
        
        # Calculate price levels
        price_min = data['low'].min()
        price_max = data['high'].max()
        price_levels = self._create_price_levels(price_min, price_max)
        
        # Build TPO matrix
        period_idx = 0
        ib_high = -np.inf
        ib_low = np.inf
        
        for timestamp, row in data.iterrows():
            # Determine TPO letter
            time_offset = (pd.to_datetime(timestamp) - pd.to_datetime(date)).total_seconds() / 60
            period_idx = min(int(time_offset / self.tpo_size), len(self.tpo_letters) - 1)
            tpo_letter = self.tpo_letters[period_idx]
            
            # Mark TPOs for price range
            high = row['high']
            low = row['low']
            
            for price_level in price_levels:
                if low <= price_level <= high:
                    if tpo_letter not in tpo_matrix[price_level]:
                        tpo_matrix[price_level].append(tpo_letter)
            
            # Track Initial Balance
            if period_idx < self.ib_periods:
                ib_high = max(ib_high, high)
                ib_low = min(ib_low, low)
        
        # Calculate TPO counts
        tpo_counts = {price: len(tpos) for price, tpos in tpo_matrix.items()}
        
        # Calculate value area
        vah, poc, val = self._calculate_value_area(tpo_counts)
        
        # Identify profile type
        profile_type = self._identify_profile_type(tpo_matrix, price_levels)
        
        # Determine range extension
        range_extension = self._analyze_range_extension(
            data, ib_high, ib_low, price_max, price_min
        )
        
        # Find single prints
        single_prints = self._find_single_prints(tpo_matrix)
        
        # Identify poor highs/lows
        poor_highs, poor_lows = self._find_poor_structure(tpo_matrix, price_levels)
        
        profile = TPOProfile(
            date=date,
            price_levels=price_levels,
            tpo_matrix=dict(tpo_matrix),
            tpo_counts=tpo_counts,
            initial_balance=(ib_high, ib_low),
            value_area=(vah, poc, val),
            profile_type=profile_type,
            range_extension=range_extension,
            single_prints=single_prints,
            poor_highs=poor_highs,
            poor_lows=poor_lows
        )
        
        # Cache profile
        self.profile_cache[str(date)] = profile
        
        return profile
    
    def analyze_market_structure(self,
                                profiles: List[TPOProfile],
                                lookback_days: int = 20) -> MarketStructure:
        """
        Analyze market structure from multiple profiles
        """
        if not profiles:
            raise ValueError("No profiles provided")
        
        # Sort by date
        profiles = sorted(profiles, key=lambda p: p.date)
        
        # Analyze trend from value area migration
        trend = self._analyze_value_migration(profiles)
        
        # Find balance areas
        balance_areas = self._find_balance_areas(profiles)
        
        # Identify breakout levels
        breakout_levels = self._identify_breakout_levels(profiles[-1], profiles[:-1])
        
        # Find acceptance and rejection levels
        acceptance_levels = self._find_acceptance_levels(profiles)
        rejection_levels = self._find_rejection_levels(profiles)
        
        # Calculate composite value area
        composite_value = self._calculate_composite_value(profiles[-lookback_days:])
        
        # Determine migration direction
        migration = self._analyze_migration(profiles[-5:]) if len(profiles) >= 5 else 'balanced'
        
        # Calculate confidence score
        confidence = self._calculate_structure_confidence(profiles)
        
        return MarketStructure(
            trend=trend,
            balance_areas=balance_areas,
            breakout_levels=breakout_levels,
            acceptance_levels=acceptance_levels,
            rejection_levels=rejection_levels,
            composite_value=composite_value,
            migration=migration,
            confidence=confidence
        )
    
    def identify_day_type(self, profile: TPOProfile) -> Dict[str, Any]:
        """
        Identify day type and trading characteristics
        """
        day_type = {
            'type': 'unknown',
            'characteristics': [],
            'bias': 'neutral',
            'confidence': 0
        }
        
        # Calculate metrics
        range_extension = profile.range_extension_ratio
        value_width_ratio = profile.value_area_width / profile.day_range if profile.day_range > 0 else 0
        
        # Identify day type based on profile shape and range
        if profile.profile_type == 'normal':
            if range_extension < 1.5:
                day_type['type'] = 'normal_day'
                day_type['characteristics'].append('balanced')
            else:
                day_type['type'] = 'normal_variation_day'
                day_type['characteristics'].append('directional')
        
        elif profile.profile_type == 'b-shaped':
            day_type['type'] = 'double_distribution_day'
            day_type['characteristics'].append('two_sided_trade')
            
        elif profile.profile_type == 'p-shaped':
            day_type['type'] = 'p_shaped_day'
            day_type['bias'] = 'bearish'
            day_type['characteristics'].append('selling_dominant')
            
        elif profile.profile_type == 'd-shaped':
            day_type['type'] = 'd_shaped_day'
            day_type['bias'] = 'bullish'
            day_type['characteristics'].append('buying_dominant')
        
        # Additional characteristics
        if range_extension > 2.0:
            day_type['characteristics'].append('trend_day')
            day_type['confidence'] = 80
        elif range_extension < 1.2:
            day_type['characteristics'].append('balance_day')
            day_type['confidence'] = 70
        
        if len(profile.single_prints) > 5:
            day_type['characteristics'].append('initiative_activity')
        
        if profile.poor_highs:
            day_type['characteristics'].append('weak_high')
        if profile.poor_lows:
            day_type['characteristics'].append('weak_low')
        
        return day_type
    
    def calculate_market_internals(self,
                                  profile: TPOProfile,
                                  volume_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate market internal statistics
        """
        internals = {}
        
        # TPO statistics
        tpo_counts = list(profile.tpo_counts.values())
        internals['tpo_total'] = sum(tpo_counts)
        internals['tpo_mean'] = np.mean(tpo_counts) if tpo_counts else 0
        internals['tpo_std'] = np.std(tpo_counts) if tpo_counts else 0
        internals['tpo_skew'] = stats.skew(tpo_counts) if len(tpo_counts) > 2 else 0
        
        # Time at price levels
        max_tpo = max(tpo_counts) if tpo_counts else 0
        internals['time_at_poc'] = max_tpo
        internals['poc_percentage'] = (max_tpo / internals['tpo_total'] * 100) if internals['tpo_total'] > 0 else 0
        
        # Value area statistics
        va_prices = [p for p, c in profile.tpo_counts.items() 
                    if profile.value_area[2] <= p <= profile.value_area[0]]
        internals['value_area_prices'] = len(va_prices)
        internals['value_area_tpos'] = sum(profile.tpo_counts.get(p, 0) for p in va_prices)
        
        # Range metrics
        internals['ib_percentage'] = (profile.ib_range / profile.day_range * 100) if profile.day_range > 0 else 0
        internals['range_extension_type'] = profile.range_extension
        
        # Balance/imbalance
        upper_tpos = sum(c for p, c in profile.tpo_counts.items() if p > profile.value_area[1])
        lower_tpos = sum(c for p, c in profile.tpo_counts.items() if p < profile.value_area[1])
        total_tpos = upper_tpos + lower_tpos
        
        if total_tpos > 0:
            internals['balance_ratio'] = upper_tpos / total_tpos
            internals['imbalance'] = abs(internals['balance_ratio'] - 0.5) * 200
        else:
            internals['balance_ratio'] = 0.5
            internals['imbalance'] = 0
        
        # Volume analysis if available
        if volume_data is not None and not volume_data.empty:
            internals['volume_poc_correlation'] = self._calculate_volume_poc_correlation(
                profile, volume_data
            )
        
        return internals
    
    def find_market_phases(self,
                          profiles: List[TPOProfile],
                          min_phase_days: int = 3) -> List[Dict[str, Any]]:
        """
        Identify market phases (accumulation, markup, distribution, markdown)
        """
        if len(profiles) < min_phase_days:
            return []
        
        phases = []
        current_phase = None
        phase_start = 0
        
        for i in range(len(profiles)):
            profile = profiles[i]
            
            # Determine phase type based on profile characteristics
            phase_type = self._determine_phase_type(
                profile,
                profiles[max(0, i-5):i] if i > 0 else []
            )
            
            if phase_type != current_phase:
                # Save previous phase
                if current_phase and i - phase_start >= min_phase_days:
                    phases.append({
                        'type': current_phase,
                        'start': profiles[phase_start].date,
                        'end': profiles[i-1].date,
                        'duration': i - phase_start,
                        'characteristics': self._get_phase_characteristics(
                            profiles[phase_start:i]
                        )
                    })
                
                # Start new phase
                current_phase = phase_type
                phase_start = i
        
        # Add final phase
        if current_phase and len(profiles) - phase_start >= min_phase_days:
            phases.append({
                'type': current_phase,
                'start': profiles[phase_start].date,
                'end': profiles[-1].date,
                'duration': len(profiles) - phase_start,
                'characteristics': self._get_phase_characteristics(
                    profiles[phase_start:]
                )
            })
        
        return phases
    
    def generate_trading_signals(self,
                               profile: TPOProfile,
                               structure: MarketStructure,
                               current_price: float) -> Dict[str, Any]:
        """
        Generate trading signals based on market profile analysis
        """
        signals = {
            'action': 'hold',
            'strength': 0,
            'reasons': [],
            'targets': [],
            'stops': []
        }
        
        # Check position relative to value area
        if current_price < profile.value_area[2]:  # Below VAL
            signals['action'] = 'buy'
            signals['reasons'].append('price_below_value')
            signals['targets'].append(profile.value_area[1])  # Target POC
            signals['stops'].append(profile.value_area[2] - (profile.value_area_width * 0.2))
            
        elif current_price > profile.value_area[0]:  # Above VAH
            signals['action'] = 'sell'
            signals['reasons'].append('price_above_value')
            signals['targets'].append(profile.value_area[1])  # Target POC
            signals['stops'].append(profile.value_area[0] + (profile.value_area_width * 0.2))
        
        # Check for poor structure
        if profile.poor_highs and current_price > profile.poor_highs[0]:
            signals['action'] = 'sell'
            signals['reasons'].append('poor_high_structure')
            signals['strength'] = 75
            
        elif profile.poor_lows and current_price < profile.poor_lows[0]:
            signals['action'] = 'buy'
            signals['reasons'].append('poor_low_structure')
            signals['strength'] = 75
        
        # Check single prints (gaps)
        for sp_price, sp_letter in profile.single_prints:
            if abs(current_price - sp_price) / current_price < 0.002:  # Within 0.2%
                signals['reasons'].append('single_print_level')
                signals['strength'] = max(signals['strength'], 60)
        
        # Check market structure
        if structure.trend == 'up' and signals['action'] == 'buy':
            signals['strength'] = min(100, signals['strength'] + 20)
            signals['reasons'].append('trend_alignment')
            
        elif structure.trend == 'down' and signals['action'] == 'sell':
            signals['strength'] = min(100, signals['strength'] + 20)
            signals['reasons'].append('trend_alignment')
        
        # Add composite value levels
        if structure.composite_value:
            if current_price < structure.composite_value[1]:
                signals['reasons'].append('below_composite_value')
            elif current_price > structure.composite_value[0]:
                signals['reasons'].append('above_composite_value')
        
        return signals
    
    def get_rl_features(self,
                       profile: TPOProfile,
                       structure: MarketStructure,
                       current_price: float) -> np.ndarray:
        """
        Extract features for RL model integration
        """
        features = []
        
        # Price relative to profile levels
        features.extend([
            (current_price - profile.value_area[1]) / profile.value_area[1],  # POC
            (current_price - profile.value_area[0]) / profile.value_area[0],  # VAH
            (current_price - profile.value_area[2]) / profile.value_area[2],  # VAL
            (current_price - profile.initial_balance[0]) / profile.initial_balance[0],  # IB high
            (current_price - profile.initial_balance[1]) / profile.initial_balance[1]   # IB low
        ])
        
        # Profile shape encoding (one-hot)
        profile_types = ['normal', 'b-shaped', 'p-shaped', 'd-shaped']
        profile_encoding = [1 if profile.profile_type == pt else 0 for pt in profile_types]
        features.extend(profile_encoding)
        
        # Range metrics
        features.extend([
            profile.range_extension_ratio,
            profile.ib_range / current_price,
            profile.value_area_width / current_price
        ])
        
        # Structure features
        features.append(1 if structure.trend == 'up' else -1 if structure.trend == 'down' else 0)
        features.append(structure.confidence / 100)
        
        # Market internals
        internals = self.calculate_market_internals(profile)
        features.extend([
            internals['poc_percentage'] / 100,
            internals['ib_percentage'] / 100,
            internals['balance_ratio'],
            internals['imbalance'] / 100
        ])
        
        # Single prints and poor structure
        features.extend([
            len(profile.single_prints) / 10,  # Normalized
            len(profile.poor_highs) / 5,
            len(profile.poor_lows) / 5
        ])
        
        return np.array(features)
    
    # Private helper methods
    def _create_price_levels(self, price_min: float, price_max: float) -> np.ndarray:
        """Create price levels based on tick size"""
        # Round to tick size
        price_min = np.floor(price_min / self.tick_size) * self.tick_size
        price_max = np.ceil(price_max / self.tick_size) * self.tick_size
        
        levels = np.arange(price_min, price_max + self.tick_size, self.tick_size)
        return levels
    
    def _calculate_value_area(self, tpo_counts: Dict[float, int]) -> Tuple[float, float, float]:
        """Calculate Value Area High, Point of Control, and Value Area Low"""
        if not tpo_counts:
            return 0, 0, 0
        
        # Find POC (price with most TPOs)
        poc = max(tpo_counts, key=tpo_counts.get)
        
        # Calculate total TPOs
        total_tpos = sum(tpo_counts.values())
        target_tpos = total_tpos * (self.value_area_pct / 100)
        
        # Build value area from POC
        accumulated_tpos = tpo_counts[poc]
        value_prices = [poc]
        
        remaining_prices = sorted([p for p in tpo_counts if p != poc])
        
        while accumulated_tpos < target_tpos and remaining_prices:
            # Find next price with highest TPO count
            next_price = max(remaining_prices, key=lambda p: tpo_counts[p])
            value_prices.append(next_price)
            accumulated_tpos += tpo_counts[next_price]
            remaining_prices.remove(next_price)
        
        vah = max(value_prices)
        val = min(value_prices)
        
        return vah, poc, val
    
    def _identify_profile_type(self,
                              tpo_matrix: Dict[float, List[str]],
                              price_levels: np.ndarray) -> str:
        """Identify profile shape type"""
        if not tpo_matrix:
            return 'unknown'
        
        # Get TPO counts by price
        tpo_counts = [len(tpo_matrix.get(p, [])) for p in price_levels]
        
        if not tpo_counts:
            return 'unknown'
        
        # Find peaks
        peaks, _ = signal.find_peaks(tpo_counts, prominence=np.std(tpo_counts))
        
        if len(peaks) == 0:
            return 'flat'
        elif len(peaks) == 1:
            # Check skewness for p/d shape
            peak_idx = peaks[0]
            peak_position = peak_idx / len(tpo_counts)
            
            if peak_position < 0.3:
                return 'p-shaped'
            elif peak_position > 0.7:
                return 'd-shaped'
            else:
                return 'normal'
        elif len(peaks) == 2:
            return 'b-shaped'
        else:
            return 'multi-distribution'
    
    def _analyze_range_extension(self,
                                data: pd.DataFrame,
                                ib_high: float,
                                ib_low: float,
                                day_high: float,
                                day_low: float) -> str:
        """Analyze range extension from Initial Balance"""
        extension_above = day_high > ib_high * 1.001  # 0.1% tolerance
        extension_below = day_low < ib_low * 0.999
        
        if extension_above and extension_below:
            return 'both'
        elif extension_above:
            return 'above'
        elif extension_below:
            return 'below'
        else:
            return 'none'
    
    def _find_single_prints(self, tpo_matrix: Dict[float, List[str]]) -> List[Tuple[float, str]]:
        """Find single print TPOs (gaps in profile)"""
        single_prints = []
        
        for price, tpos in tpo_matrix.items():
            if len(tpos) == 1:
                # Check if isolated (no TPOs immediately above/below)
                price_above = price + self.tick_size
                price_below = price - self.tick_size
                
                if (price_above not in tpo_matrix or not tpo_matrix[price_above]) and \
                   (price_below not in tpo_matrix or not tpo_matrix[price_below]):
                    single_prints.append((price, tpos[0]))
        
        return single_prints
    
    def _find_poor_structure(self,
                            tpo_matrix: Dict[float, List[str]],
                            price_levels: np.ndarray) -> Tuple[List[float], List[float]]:
        """Find poor highs and lows (weak structure)"""
        if len(price_levels) == 0:
            return [], []
        
        poor_highs = []
        poor_lows = []
        
        # Poor high: limited TPOs at day's high
        high_prices = sorted(price_levels)[-3:]  # Top 3 prices
        for price in high_prices:
            if price in tpo_matrix and len(tpo_matrix[price]) <= 2:
                poor_highs.append(price)
        
        # Poor low: limited TPOs at day's low
        low_prices = sorted(price_levels)[:3]  # Bottom 3 prices
        for price in low_prices:
            if price in tpo_matrix and len(tpo_matrix[price]) <= 2:
                poor_lows.append(price)
        
        return poor_highs, poor_lows
    
    def _analyze_value_migration(self, profiles: List[TPOProfile]) -> str:
        """Analyze trend from value area migration"""
        if len(profiles) < 3:
            return 'balanced'
        
        # Get recent POCs
        recent_pocs = [p.value_area[1] for p in profiles[-5:]]
        
        # Calculate trend
        if len(recent_pocs) >= 2:
            trend_slope = np.polyfit(range(len(recent_pocs)), recent_pocs, 1)[0]
            
            if trend_slope > 0:
                return 'up'
            elif trend_slope < 0:
                return 'down'
        
        return 'balanced'
    
    def _find_balance_areas(self, profiles: List[TPOProfile]) -> List[Tuple[float, float]]:
        """Find price areas with repeated balance"""
        if len(profiles) < 3:
            return []
        
        # Collect all value areas
        all_vals = [p.value_area[2] for p in profiles]
        all_vahs = [p.value_area[0] for p in profiles]
        
        # Find overlapping areas
        balance_areas = []
        
        for i in range(len(profiles) - 2):
            # Check if value areas overlap
            val1, vah1 = all_vals[i], all_vahs[i]
            val2, vah2 = all_vals[i+1], all_vahs[i+1]
            
            if val1 <= vah2 and val2 <= vah1:  # Overlap
                balance_low = max(val1, val2)
                balance_high = min(vah1, vah2)
                balance_areas.append((balance_high, balance_low))
        
        # Merge overlapping balance areas
        if balance_areas:
            merged = []
            current = balance_areas[0]
            
            for area in balance_areas[1:]:
                if area[1] <= current[0] and area[0] >= current[1]:
                    # Merge
                    current = (max(current[0], area[0]), min(current[1], area[1]))
                else:
                    merged.append(current)
                    current = area
            
            merged.append(current)
            return merged
        
        return []
    
    def _identify_breakout_levels(self,
                                 current: TPOProfile,
                                 historical: List[TPOProfile]) -> Dict[str, float]:
        """Identify potential breakout levels"""
        levels = {}
        
        if not historical:
            levels['upper'] = current.value_area[0]
            levels['lower'] = current.value_area[2]
            return levels
        
        # Find recent high volume nodes
        recent_pocs = [p.value_area[1] for p in historical[-10:]]
        recent_vahs = [p.value_area[0] for p in historical[-10:]]
        recent_vals = [p.value_area[2] for p in historical[-10:]]
        
        # Upper breakout: highest recent VAH or POC cluster
        levels['upper'] = max(recent_vahs + [current.value_area[0]])
        
        # Lower breakout: lowest recent VAL or POC cluster
        levels['lower'] = min(recent_vals + [current.value_area[2]])
        
        return levels
    
    def _find_acceptance_levels(self, profiles: List[TPOProfile]) -> List[float]:
        """Find prices with high time acceptance"""
        if not profiles:
            return []
        
        # Aggregate TPO counts across profiles
        price_acceptance = defaultdict(int)
        
        for profile in profiles[-10:]:  # Last 10 profiles
            for price, count in profile.tpo_counts.items():
                price_acceptance[price] += count
        
        # Find high acceptance prices
        if price_acceptance:
            threshold = np.percentile(list(price_acceptance.values()), 70)
            return [p for p, c in price_acceptance.items() if c >= threshold]
        
        return []
    
    def _find_rejection_levels(self, profiles: List[TPOProfile]) -> List[float]:
        """Find prices with rejection (single prints)"""
        rejection_levels = []
        
        for profile in profiles[-5:]:  # Last 5 profiles
            for sp_price, _ in profile.single_prints:
                rejection_levels.append(sp_price)
        
        return list(set(rejection_levels))
    
    def _calculate_composite_value(self, profiles: List[TPOProfile]) -> Tuple[float, float]:
        """Calculate composite value area from multiple profiles"""
        if not profiles:
            return 0, 0
        
        # Weight recent profiles more heavily
        weights = np.linspace(0.5, 1.0, len(profiles))
        weights = weights / weights.sum()
        
        composite_vah = sum(p.value_area[0] * w for p, w in zip(profiles, weights))
        composite_val = sum(p.value_area[2] * w for p, w in zip(profiles, weights))
        
        return composite_vah, composite_val
    
    def _analyze_migration(self, profiles: List[TPOProfile]) -> str:
        """Analyze value area migration direction"""
        if len(profiles) < 2:
            return 'balanced'
        
        # Compare first and last profile
        first_poc = profiles[0].value_area[1]
        last_poc = profiles[-1].value_area[1]
        
        change_pct = (last_poc - first_poc) / first_poc * 100
        
        if change_pct > 1:
            return 'higher'
        elif change_pct < -1:
            return 'lower'
        else:
            return 'overlapping'
    
    def _calculate_structure_confidence(self, profiles: List[TPOProfile]) -> float:
        """Calculate confidence in market structure analysis"""
        if len(profiles) < 5:
            return 30.0
        
        confidence = 50.0  # Base confidence
        
        # Add confidence for consistent patterns
        recent_types = [p.profile_type for p in profiles[-5:]]
        if len(set(recent_types)) == 1:  # All same type
            confidence += 20
        
        # Add confidence for clear trend
        pocs = [p.value_area[1] for p in profiles[-10:]]
        if len(pocs) >= 3:
            correlation = abs(np.corrcoef(range(len(pocs)), pocs)[0, 1])
            confidence += correlation * 20
        
        # Add confidence for volume (if available)
        # This would need volume data integration
        
        return min(100, confidence)
    
    def _determine_phase_type(self,
                            profile: TPOProfile,
                            recent_profiles: List[TPOProfile]) -> str:
        """Determine market phase type"""
        if not recent_profiles:
            return 'unknown'
        
        # Check for accumulation (balance with narrowing range)
        if profile.profile_type == 'normal' and profile.range_extension_ratio < 1.2:
            ranges = [p.day_range for p in recent_profiles[-3:]]
            if len(ranges) >= 2 and ranges[-1] < ranges[0]:
                return 'accumulation'
        
        # Check for markup (trending with expansion)
        if profile.range_extension_ratio > 1.5:
            pocs = [p.value_area[1] for p in recent_profiles[-3:]]
            if len(pocs) >= 2 and pocs[-1] > pocs[0]:
                return 'markup'
        
        # Check for distribution (balance at highs)
        if profile.profile_type in ['normal', 'b-shaped']:
            recent_highs = [p.price_levels.max() for p in recent_profiles[-5:]]
            if len(recent_highs) >= 3:
                if np.std(recent_highs) / np.mean(recent_highs) < 0.01:
                    return 'distribution'
        
        # Check for markdown (trending down)
        if profile.range_extension == 'below':
            pocs = [p.value_area[1] for p in recent_profiles[-3:]]
            if len(pocs) >= 2 and pocs[-1] < pocs[0]:
                return 'markdown'
        
        return 'transition'
    
    def _get_phase_characteristics(self, profiles: List[TPOProfile]) -> List[str]:
        """Get characteristics of market phase"""
        characteristics = []
        
        if not profiles:
            return characteristics
        
        # Average range extension
        avg_extension = np.mean([p.range_extension_ratio for p in profiles])
        if avg_extension > 1.5:
            characteristics.append('trending')
        elif avg_extension < 1.2:
            characteristics.append('balancing')
        
        # Profile types
        types = [p.profile_type for p in profiles]
        most_common = Counter(types).most_common(1)[0][0]
        characteristics.append(f'{most_common}_dominant')
        
        # Value area width trend
        va_widths = [p.value_area_width for p in profiles]
        if len(va_widths) >= 3:
            if va_widths[-1] > va_widths[0]:
                characteristics.append('expanding_value')
            else:
                characteristics.append('contracting_value')
        
        return characteristics
    
    def _calculate_volume_poc_correlation(self,
                                        profile: TPOProfile,
                                        volume_data: pd.DataFrame) -> float:
        """Calculate correlation between volume and TPO POC"""
        # This would need actual implementation with volume data
        # Placeholder for now
        return 0.0