"""
Volume Profile Visible Range (VPVR) Analysis
Advanced volume profile analysis for identifying key price levels and market structure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from scipy import stats, signal, interpolate
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfile:
    """Volume profile data structure"""
    price_levels: np.ndarray
    volumes: np.ndarray
    buy_volumes: np.ndarray
    sell_volumes: np.ndarray
    poc: float  # Point of Control
    vah: float  # Value Area High
    val: float  # Value Area Low
    vwap: float  # Volume Weighted Average Price
    profile_range: Tuple[float, float]
    timestamp: datetime
    timeframe: str
    
    @property
    def value_area_volume(self) -> float:
        """Calculate volume within value area"""
        mask = (self.price_levels >= self.val) & (self.price_levels <= self.vah)
        return self.volumes[mask].sum()
    
    @property
    def total_volume(self) -> float:
        """Total profile volume"""
        return self.volumes.sum()
    
    @property
    def value_area_percentage(self) -> float:
        """Percentage of volume in value area"""
        if self.total_volume > 0:
            return (self.value_area_volume / self.total_volume) * 100
        return 0
    
    @property
    def delta_volume(self) -> np.ndarray:
        """Volume delta (buy - sell)"""
        return self.buy_volumes - self.sell_volumes
    
    @property
    def cumulative_delta(self) -> np.ndarray:
        """Cumulative volume delta"""
        return np.cumsum(self.delta_volume)


@dataclass
class VolumeNode:
    """Individual volume node in profile"""
    price: float
    volume: float
    buy_volume: float
    sell_volume: float
    node_type: str  # 'HVN', 'LVN', 'POC'
    strength: float  # 0-100 strength score
    
    @property
    def delta(self) -> float:
        """Buy-sell delta"""
        return self.buy_volume - self.sell_volume
    
    @property
    def delta_percentage(self) -> float:
        """Delta as percentage of total"""
        if self.volume > 0:
            return (self.delta / self.volume) * 100
        return 0


class VPVRAnalyzer:
    """Advanced Volume Profile Visible Range analyzer"""
    
    def __init__(self,
                 tick_size: float = 0.01,
                 value_area_percentage: float = 70.0,
                 min_node_volume_pct: float = 2.0,
                 hvn_threshold_pct: float = 150.0,
                 lvn_threshold_pct: float = 50.0):
        """
        Initialize VPVR analyzer
        
        Args:
            tick_size: Minimum price increment
            value_area_percentage: Percentage for value area (typically 70%)
            min_node_volume_pct: Minimum volume % to consider a node
            hvn_threshold_pct: Threshold for High Volume Node (% of average)
            lvn_threshold_pct: Threshold for Low Volume Node (% of average)
        """
        self.tick_size = tick_size
        self.value_area_percentage = value_area_percentage
        self.min_node_volume_pct = min_node_volume_pct
        self.hvn_threshold_pct = hvn_threshold_pct
        self.lvn_threshold_pct = lvn_threshold_pct
        
        # Cache for profiles
        self.profile_cache: Dict[str, VolumeProfile] = {}
        self.node_cache: Dict[str, List[VolumeNode]] = {}
        
        # Statistical models
        self.kde_estimator = None
        self.scaler = StandardScaler()
    
    def calculate_profile(self,
                         data: pd.DataFrame,
                         price_col: str = 'price',
                         volume_col: str = 'volume',
                         side_col: Optional[str] = 'side',
                         bins: Optional[int] = None,
                         range_type: str = 'visible') -> VolumeProfile:
        """
        Calculate volume profile for given data
        
        Args:
            data: DataFrame with price and volume data
            price_col: Name of price column
            volume_col: Name of volume column
            side_col: Name of side column (buy/sell)
            bins: Number of price bins (auto-calculated if None)
            range_type: 'visible', 'fixed', or 'session'
        """
        if data.empty:
            raise ValueError("Empty data provided")
        
        # Determine price range
        price_min, price_max = self._get_price_range(data, price_col, range_type)
        
        # Calculate optimal bins if not provided
        if bins is None:
            bins = self._calculate_optimal_bins(price_min, price_max, len(data))
        
        # Create price levels
        price_levels = np.linspace(price_min, price_max, bins)
        bin_width = price_levels[1] - price_levels[0]
        
        # Initialize volume arrays
        volumes = np.zeros(bins)
        buy_volumes = np.zeros(bins)
        sell_volumes = np.zeros(bins)
        
        # Aggregate volume by price level
        for _, row in data.iterrows():
            price = row[price_col]
            volume = row[volume_col]
            
            # Find corresponding bin
            bin_idx = int((price - price_min) / bin_width)
            bin_idx = min(max(0, bin_idx), bins - 1)
            
            volumes[bin_idx] += volume
            
            # Track buy/sell volumes if available
            if side_col and side_col in row:
                if row[side_col] in ['buy', 'BUY', 1]:
                    buy_volumes[bin_idx] += volume
                else:
                    sell_volumes[bin_idx] += volume
            else:
                # Estimate buy/sell based on price movement
                buy_volumes[bin_idx] += volume * 0.5
                sell_volumes[bin_idx] += volume * 0.5
        
        # Calculate key metrics
        poc = self._calculate_poc(price_levels, volumes)
        vah, val = self._calculate_value_area(price_levels, volumes)
        vwap = self._calculate_vwap(data, price_col, volume_col)
        
        profile = VolumeProfile(
            price_levels=price_levels,
            volumes=volumes,
            buy_volumes=buy_volumes,
            sell_volumes=sell_volumes,
            poc=poc,
            vah=vah,
            val=val,
            vwap=vwap,
            profile_range=(price_min, price_max),
            timestamp=datetime.now(),
            timeframe=self._detect_timeframe(data)
        )
        
        # Cache profile
        cache_key = f"{data.index[0]}_{data.index[-1]}_{range_type}"
        self.profile_cache[cache_key] = profile
        
        return profile
    
    def identify_volume_nodes(self, profile: VolumeProfile) -> List[VolumeNode]:
        """
        Identify High Volume Nodes (HVN) and Low Volume Nodes (LVN)
        """
        nodes = []
        avg_volume = profile.volumes.mean()
        
        # Apply smoothing to reduce noise
        smoothed_volumes = self._smooth_profile(profile.volumes)
        
        # Find peaks (HVN) and valleys (LVN)
        peaks, peak_props = signal.find_peaks(
            smoothed_volumes,
            height=avg_volume * (self.hvn_threshold_pct / 100),
            prominence=avg_volume * 0.3
        )
        
        valleys, valley_props = signal.find_peaks(
            -smoothed_volumes,
            height=-avg_volume * (self.lvn_threshold_pct / 100)
        )
        
        # Create HVN nodes
        for idx in peaks:
            if profile.volumes[idx] > avg_volume * (self.min_node_volume_pct / 100):
                node = VolumeNode(
                    price=profile.price_levels[idx],
                    volume=profile.volumes[idx],
                    buy_volume=profile.buy_volumes[idx],
                    sell_volume=profile.sell_volumes[idx],
                    node_type='HVN',
                    strength=min(100, (profile.volumes[idx] / avg_volume) * 50)
                )
                nodes.append(node)
        
        # Create LVN nodes
        for idx in valleys:
            if profile.volumes[idx] < avg_volume * (self.lvn_threshold_pct / 100):
                node = VolumeNode(
                    price=profile.price_levels[idx],
                    volume=profile.volumes[idx],
                    buy_volume=profile.buy_volumes[idx],
                    sell_volume=profile.sell_volumes[idx],
                    node_type='LVN',
                    strength=min(100, (1 - profile.volumes[idx] / avg_volume) * 50)
                )
                nodes.append(node)
        
        # Add POC node
        poc_idx = np.argmax(profile.volumes)
        poc_node = VolumeNode(
            price=profile.poc,
            volume=profile.volumes[poc_idx],
            buy_volume=profile.buy_volumes[poc_idx],
            sell_volume=profile.sell_volumes[poc_idx],
            node_type='POC',
            strength=100
        )
        nodes.append(poc_node)
        
        # Sort by price
        nodes.sort(key=lambda x: x.price)
        
        return nodes
    
    def detect_support_resistance(self,
                                 profile: VolumeProfile,
                                 current_price: float,
                                 sensitivity: float = 1.5) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels from volume profile
        """
        nodes = self.identify_volume_nodes(profile)
        
        support_levels = []
        resistance_levels = []
        
        for node in nodes:
            # HVN nodes act as support/resistance
            if node.node_type == 'HVN':
                if node.price < current_price:
                    support_levels.append({
                        'price': node.price,
                        'strength': node.strength,
                        'volume': node.volume,
                        'delta': node.delta
                    })
                else:
                    resistance_levels.append({
                        'price': node.price,
                        'strength': node.strength,
                        'volume': node.volume,
                        'delta': node.delta
                    })
            
            # LVN nodes can act as acceleration zones
            elif node.node_type == 'LVN':
                # Less significant but still relevant
                strength_adjusted = node.strength * 0.5
                if node.price < current_price:
                    support_levels.append({
                        'price': node.price,
                        'strength': strength_adjusted,
                        'volume': node.volume,
                        'delta': node.delta,
                        'type': 'acceleration_zone'
                    })
        
        # Add value area boundaries
        support_levels.append({
            'price': profile.val,
            'strength': 75,
            'type': 'value_area_low'
        })
        
        resistance_levels.append({
            'price': profile.vah,
            'strength': 75,
            'type': 'value_area_high'
        })
        
        # Sort by strength
        support_levels.sort(key=lambda x: x['strength'], reverse=True)
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'support': support_levels[:5],  # Top 5 levels
            'resistance': resistance_levels[:5]
        }
    
    def analyze_volume_distribution(self, profile: VolumeProfile) -> Dict[str, Any]:
        """
        Analyze the distribution characteristics of volume profile
        """
        volumes = profile.volumes[profile.volumes > 0]
        
        if len(volumes) == 0:
            return {}
        
        # Statistical analysis
        analysis = {
            'skewness': stats.skew(volumes),
            'kurtosis': stats.kurtosis(volumes),
            'concentration': self._calculate_concentration(volumes),
            'distribution_type': self._identify_distribution_type(volumes),
            'volume_clusters': self._identify_clusters(profile),
            'balance': self._calculate_balance(profile)
        }
        
        # Market structure interpretation
        if analysis['skewness'] > 0.5:
            analysis['structure'] = 'accumulation'
        elif analysis['skewness'] < -0.5:
            analysis['structure'] = 'distribution'
        else:
            analysis['structure'] = 'balanced'
        
        # Trend bias from volume
        delta_sum = profile.delta_volume.sum()
        if delta_sum > 0:
            analysis['bias'] = 'bullish'
            analysis['bias_strength'] = min(100, abs(delta_sum) / profile.total_volume * 100)
        else:
            analysis['bias'] = 'bearish'
            analysis['bias_strength'] = min(100, abs(delta_sum) / profile.total_volume * 100)
        
        return analysis
    
    def compare_profiles(self,
                        profile1: VolumeProfile,
                        profile2: VolumeProfile) -> Dict[str, Any]:
        """
        Compare two volume profiles for migration and shifts
        """
        comparison = {
            'poc_shift': profile2.poc - profile1.poc,
            'value_area_shift': {
                'high': profile2.vah - profile1.vah,
                'low': profile2.val - profile1.val,
                'width': (profile2.vah - profile2.val) - (profile1.vah - profile1.val)
            },
            'volume_change': profile2.total_volume - profile1.total_volume,
            'volume_change_pct': ((profile2.total_volume / profile1.total_volume) - 1) * 100
        }
        
        # Analyze migration direction
        if comparison['poc_shift'] > 0:
            comparison['migration'] = 'upward'
        elif comparison['poc_shift'] < 0:
            comparison['migration'] = 'downward'
        else:
            comparison['migration'] = 'stable'
        
        # Value area expansion/contraction
        if comparison['value_area_shift']['width'] > 0:
            comparison['value_area_trend'] = 'expanding'
        else:
            comparison['value_area_trend'] = 'contracting'
        
        # Calculate correlation between profiles
        min_len = min(len(profile1.volumes), len(profile2.volumes))
        if min_len > 0:
            correlation = np.corrcoef(
                profile1.volumes[:min_len],
                profile2.volumes[:min_len]
            )[0, 1]
            comparison['correlation'] = correlation
        
        return comparison
    
    def calculate_composite_profile(self,
                                   profiles: List[VolumeProfile],
                                   weights: Optional[List[float]] = None) -> VolumeProfile:
        """
        Create composite profile from multiple timeframes
        """
        if not profiles:
            raise ValueError("No profiles provided")
        
        if weights is None:
            weights = [1.0] * len(profiles)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Find common price range
        price_min = min(p.profile_range[0] for p in profiles)
        price_max = max(p.profile_range[1] for p in profiles)
        
        # Create unified price levels
        bins = max(len(p.price_levels) for p in profiles)
        price_levels = np.linspace(price_min, price_max, bins)
        
        # Aggregate volumes
        composite_volumes = np.zeros(bins)
        composite_buy = np.zeros(bins)
        composite_sell = np.zeros(bins)
        
        for profile, weight in zip(profiles, weights):
            # Interpolate to common price levels
            interp_volumes = np.interp(price_levels, profile.price_levels, profile.volumes)
            interp_buy = np.interp(price_levels, profile.price_levels, profile.buy_volumes)
            interp_sell = np.interp(price_levels, profile.price_levels, profile.sell_volumes)
            
            composite_volumes += interp_volumes * weight
            composite_buy += interp_buy * weight
            composite_sell += interp_sell * weight
        
        # Calculate composite metrics
        poc = self._calculate_poc(price_levels, composite_volumes)
        vah, val = self._calculate_value_area(price_levels, composite_volumes)
        
        # Weighted VWAP
        vwap = sum(p.vwap * w for p, w in zip(profiles, weights))
        
        return VolumeProfile(
            price_levels=price_levels,
            volumes=composite_volumes,
            buy_volumes=composite_buy,
            sell_volumes=composite_sell,
            poc=poc,
            vah=vah,
            val=val,
            vwap=vwap,
            profile_range=(price_min, price_max),
            timestamp=datetime.now(),
            timeframe='composite'
        )
    
    def get_rl_features(self, profile: VolumeProfile, current_price: float) -> np.ndarray:
        """
        Extract features for RL model integration
        """
        features = []
        
        # Price relative to key levels
        features.extend([
            (current_price - profile.poc) / profile.poc,
            (current_price - profile.vah) / profile.vah,
            (current_price - profile.val) / profile.val,
            (current_price - profile.vwap) / profile.vwap
        ])
        
        # Volume distribution metrics
        analysis = self.analyze_volume_distribution(profile)
        features.extend([
            analysis.get('skewness', 0),
            analysis.get('kurtosis', 0),
            analysis.get('concentration', 0),
            analysis.get('bias_strength', 0) / 100
        ])
        
        # Value area metrics
        features.extend([
            profile.value_area_percentage / 100,
            (profile.vah - profile.val) / current_price,  # Value area width
        ])
        
        # Volume nodes
        nodes = self.identify_volume_nodes(profile)
        hvn_count = sum(1 for n in nodes if n.node_type == 'HVN')
        lvn_count = sum(1 for n in nodes if n.node_type == 'LVN')
        features.extend([hvn_count / 10, lvn_count / 10])  # Normalized counts
        
        # Delta analysis
        total_delta = profile.delta_volume.sum()
        features.append(total_delta / profile.total_volume if profile.total_volume > 0 else 0)
        
        return np.array(features)
    
    # Private helper methods
    def _get_price_range(self, data: pd.DataFrame, price_col: str, range_type: str) -> Tuple[float, float]:
        """Determine price range based on type"""
        if range_type == 'visible':
            return data[price_col].min(), data[price_col].max()
        elif range_type == 'fixed':
            # Fixed percentage around current price
            current = data[price_col].iloc[-1]
            range_pct = 0.05  # 5% range
            return current * (1 - range_pct), current * (1 + range_pct)
        else:  # session
            # Use session bounds if available
            return data[price_col].min(), data[price_col].max()
    
    def _calculate_optimal_bins(self, price_min: float, price_max: float, data_points: int) -> int:
        """Calculate optimal number of bins"""
        # Use Sturges' rule with adjustment for price range
        base_bins = int(np.log2(data_points) + 1)
        price_range = price_max - price_min
        
        # Adjust based on tick size
        tick_bins = int(price_range / self.tick_size)
        
        # Find balance
        optimal = min(max(base_bins * 3, 30), min(tick_bins, 500))
        
        return optimal
    
    def _calculate_poc(self, price_levels: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate Point of Control"""
        if len(volumes) == 0:
            return np.mean(price_levels)
        
        max_idx = np.argmax(volumes)
        return price_levels[max_idx]
    
    def _calculate_value_area(self, price_levels: np.ndarray, volumes: np.ndarray) -> Tuple[float, float]:
        """Calculate Value Area High and Low"""
        if len(volumes) == 0:
            mid = np.mean(price_levels)
            return mid, mid
        
        total_volume = volumes.sum()
        target_volume = total_volume * (self.value_area_percentage / 100)
        
        # Start from POC
        poc_idx = np.argmax(volumes)
        accumulated_volume = volumes[poc_idx]
        
        upper_idx = poc_idx
        lower_idx = poc_idx
        
        # Expand from POC until target volume reached
        while accumulated_volume < target_volume:
            # Check which side has more volume
            upper_vol = volumes[upper_idx + 1] if upper_idx < len(volumes) - 1 else 0
            lower_vol = volumes[lower_idx - 1] if lower_idx > 0 else 0
            
            if upper_vol >= lower_vol and upper_idx < len(volumes) - 1:
                upper_idx += 1
                accumulated_volume += upper_vol
            elif lower_idx > 0:
                lower_idx -= 1
                accumulated_volume += lower_vol
            else:
                break
        
        return price_levels[upper_idx], price_levels[lower_idx]
    
    def _calculate_vwap(self, data: pd.DataFrame, price_col: str, volume_col: str) -> float:
        """Calculate Volume Weighted Average Price"""
        if data.empty:
            return 0
        
        total_volume = data[volume_col].sum()
        if total_volume == 0:
            return data[price_col].mean()
        
        return (data[price_col] * data[volume_col]).sum() / total_volume
    
    def _detect_timeframe(self, data: pd.DataFrame) -> str:
        """Detect timeframe from data"""
        if len(data) < 2:
            return 'unknown'
        
        # Estimate based on data points and time range
        time_range = (data.index[-1] - data.index[0]).total_seconds()
        avg_interval = time_range / len(data)
        
        if avg_interval < 60:
            return '1m'
        elif avg_interval < 300:
            return '5m'
        elif avg_interval < 900:
            return '15m'
        elif avg_interval < 3600:
            return '1h'
        elif avg_interval < 14400:
            return '4h'
        elif avg_interval < 86400:
            return '1d'
        else:
            return '1w'
    
    def _smooth_profile(self, volumes: np.ndarray, window: int = 3) -> np.ndarray:
        """Smooth volume profile to reduce noise"""
        if len(volumes) < window:
            return volumes
        
        # Use Savitzky-Golay filter for smoothing
        from scipy.signal import savgol_filter
        return savgol_filter(volumes, window, 1)
    
    def _calculate_concentration(self, volumes: np.ndarray) -> float:
        """Calculate volume concentration (Gini coefficient)"""
        sorted_volumes = np.sort(volumes)
        n = len(volumes)
        index = np.arange(1, n + 1)
        
        return (2 * np.sum(index * sorted_volumes)) / (n * np.sum(sorted_volumes)) - (n + 1) / n
    
    def _identify_distribution_type(self, volumes: np.ndarray) -> str:
        """Identify distribution type of volume"""
        skew = stats.skew(volumes)
        kurt = stats.kurtosis(volumes)
        
        if abs(skew) < 0.5 and abs(kurt) < 1:
            return 'normal'
        elif skew > 1:
            return 'right_skewed'
        elif skew < -1:
            return 'left_skewed'
        elif kurt > 3:
            return 'leptokurtic'
        elif kurt < -1:
            return 'platykurtic'
        else:
            return 'mixed'
    
    def _identify_clusters(self, profile: VolumeProfile) -> List[Dict[str, Any]]:
        """Identify volume clusters using DBSCAN"""
        # Prepare data for clustering
        X = np.column_stack((profile.price_levels, profile.volumes))
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X)
        
        # Cluster
        clustering = DBSCAN(eps=0.3, min_samples=3).fit(X_scaled)
        
        clusters = []
        for label in set(clustering.labels_):
            if label == -1:  # Skip noise
                continue
            
            mask = clustering.labels_ == label
            cluster_prices = profile.price_levels[mask]
            cluster_volumes = profile.volumes[mask]
            
            clusters.append({
                'center': np.mean(cluster_prices),
                'range': (cluster_prices.min(), cluster_prices.max()),
                'total_volume': cluster_volumes.sum(),
                'avg_volume': cluster_volumes.mean()
            })
        
        return clusters
    
    def _calculate_balance(self, profile: VolumeProfile) -> Dict[str, float]:
        """Calculate market balance metrics"""
        # Split profile into upper and lower halves
        mid_idx = len(profile.volumes) // 2
        
        upper_volume = profile.volumes[mid_idx:].sum()
        lower_volume = profile.volumes[:mid_idx].sum()
        
        total = upper_volume + lower_volume
        if total > 0:
            balance_ratio = upper_volume / total
        else:
            balance_ratio = 0.5
        
        return {
            'upper_volume': upper_volume,
            'lower_volume': lower_volume,
            'balance_ratio': balance_ratio,
            'imbalance': abs(balance_ratio - 0.5) * 200  # 0-100 scale
        }