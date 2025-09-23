"""
Delta Volume Analysis
Advanced buy/sell volume delta analysis for order flow and market sentiment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class DeltaBar:
    """Individual delta volume bar"""
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    total_volume: float
    buy_volume: float
    sell_volume: float
    delta: float  # buy - sell
    cumulative_delta: float
    delta_percentage: float  # delta as % of total volume
    
    @property
    def is_bullish(self) -> bool:
        """Check if bar is bullish"""
        return self.delta > 0
    
    @property
    def is_bearish(self) -> bool:
        """Check if bar is bearish"""
        return self.delta < 0
    
    @property
    def absorption_ratio(self) -> float:
        """Calculate absorption ratio (price movement vs volume)"""
        price_change = abs(self.close_price - self.open_price)
        if self.total_volume > 0 and self.open_price > 0:
            return (price_change / self.open_price) / self.total_volume * 1000000
        return 0


@dataclass
class DeltaDivergence:
    """Delta divergence signal"""
    divergence_type: str  # 'bullish' or 'bearish'
    start_time: datetime
    end_time: datetime
    price_trend: str  # 'up' or 'down'
    delta_trend: str  # 'up' or 'down'
    strength: float  # 0-100
    confirmed: bool
    
    @property
    def duration(self) -> timedelta:
        """Get divergence duration"""
        return self.end_time - self.start_time


@dataclass
class OrderFlowImbalance:
    """Order flow imbalance metrics"""
    timestamp: datetime
    bid_imbalance: float  # Bid volume imbalance
    ask_imbalance: float  # Ask volume imbalance
    total_imbalance: float
    imbalance_ratio: float
    stacked_imbalances: int  # Number of consecutive imbalances
    imbalance_type: str  # 'bid_dominant', 'ask_dominant', 'balanced'


class DeltaVolumeAnalyzer:
    """Advanced delta volume and order flow analyzer"""
    
    def __init__(self,
                 min_volume_threshold: float = 100,
                 imbalance_threshold: float = 3.0,
                 divergence_lookback: int = 20,
                 cumulative_reset_period: Optional[str] = 'daily'):
        """
        Initialize delta volume analyzer
        
        Args:
            min_volume_threshold: Minimum volume to consider
            imbalance_threshold: Ratio threshold for imbalance detection
            divergence_lookback: Periods to look back for divergence
            cumulative_reset_period: Period to reset cumulative delta ('daily', 'weekly', None)
        """
        self.min_volume_threshold = min_volume_threshold
        self.imbalance_threshold = imbalance_threshold
        self.divergence_lookback = divergence_lookback
        self.cumulative_reset_period = cumulative_reset_period
        
        # Storage
        self.delta_bars: deque = deque(maxlen=1000)
        self.cumulative_delta_history: List[float] = []
        self.divergences: List[DeltaDivergence] = []
        self.imbalances: List[OrderFlowImbalance] = []
        
        # Analysis tools
        self.scaler = StandardScaler()
        self.regression_model = LinearRegression()
    
    def calculate_delta_volume(self,
                              data: pd.DataFrame,
                              method: str = 'tick') -> List[DeltaBar]:
        """
        Calculate delta volume from price and volume data
        
        Args:
            data: DataFrame with price, volume, and optionally bid/ask data
            method: 'tick', 'bid_ask', or 'trade_classification'
        """
        delta_bars = []
        cumulative_delta = 0
        last_reset = None
        
        for idx, row in data.iterrows():
            # Extract base data
            timestamp = pd.to_datetime(idx)
            open_price = row.get('open', row.get('price', 0))
            high_price = row.get('high', open_price)
            low_price = row.get('low', open_price)
            close_price = row.get('close', row.get('price', open_price))
            total_volume = row.get('volume', 0)
            
            # Skip if volume too low
            if total_volume < self.min_volume_threshold:
                continue
            
            # Calculate buy/sell volumes based on method
            if method == 'bid_ask' and 'bid_volume' in row and 'ask_volume' in row:
                buy_volume = row['bid_volume']
                sell_volume = row['ask_volume']
            elif method == 'trade_classification':
                buy_volume, sell_volume = self._classify_trades(row)
            else:  # tick method
                buy_volume, sell_volume = self._estimate_delta_tick(
                    open_price, high_price, low_price, close_price, total_volume
                )
            
            # Calculate delta
            delta = buy_volume - sell_volume
            
            # Reset cumulative delta if needed
            if self._should_reset_cumulative(timestamp, last_reset):
                cumulative_delta = 0
                last_reset = timestamp
            
            cumulative_delta += delta
            
            # Calculate delta percentage
            delta_percentage = (delta / total_volume * 100) if total_volume > 0 else 0
            
            # Create delta bar
            delta_bar = DeltaBar(
                timestamp=timestamp,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                total_volume=total_volume,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                delta=delta,
                cumulative_delta=cumulative_delta,
                delta_percentage=delta_percentage
            )
            
            delta_bars.append(delta_bar)
            self.delta_bars.append(delta_bar)
            self.cumulative_delta_history.append(cumulative_delta)
        
        return delta_bars
    
    def detect_divergences(self,
                         delta_bars: Optional[List[DeltaBar]] = None,
                         min_strength: float = 50.0) -> List[DeltaDivergence]:
        """
        Detect price/delta divergences
        """
        if delta_bars is None:
            delta_bars = list(self.delta_bars)
        
        if len(delta_bars) < self.divergence_lookback:
            return []
        
        divergences = []
        
        for i in range(self.divergence_lookback, len(delta_bars)):
            window = delta_bars[i-self.divergence_lookback:i]
            
            # Extract price and delta series
            prices = [bar.close_price for bar in window]
            deltas = [bar.cumulative_delta for bar in window]
            
            # Calculate trends
            price_trend = self._calculate_trend(prices)
            delta_trend = self._calculate_trend(deltas)
            
            # Detect divergence
            divergence = None
            
            if price_trend > 0.1 and delta_trend < -0.1:
                # Bearish divergence: price up, delta down
                divergence = DeltaDivergence(
                    divergence_type='bearish',
                    start_time=window[0].timestamp,
                    end_time=window[-1].timestamp,
                    price_trend='up',
                    delta_trend='down',
                    strength=self._calculate_divergence_strength(price_trend, delta_trend),
                    confirmed=False
                )
            elif price_trend < -0.1 and delta_trend > 0.1:
                # Bullish divergence: price down, delta up
                divergence = DeltaDivergence(
                    divergence_type='bullish',
                    start_time=window[0].timestamp,
                    end_time=window[-1].timestamp,
                    price_trend='down',
                    delta_trend='up',
                    strength=self._calculate_divergence_strength(price_trend, delta_trend),
                    confirmed=False
                )
            
            # Add if strong enough
            if divergence and divergence.strength >= min_strength:
                # Check for confirmation
                if i < len(delta_bars) - 1:
                    next_bar = delta_bars[i]
                    if divergence.divergence_type == 'bullish' and next_bar.close_price > next_bar.open_price:
                        divergence.confirmed = True
                    elif divergence.divergence_type == 'bearish' and next_bar.close_price < next_bar.open_price:
                        divergence.confirmed = True
                
                divergences.append(divergence)
                self.divergences.append(divergence)
        
        return divergences
    
    def analyze_order_flow_imbalance(self,
                                    order_book_data: pd.DataFrame) -> List[OrderFlowImbalance]:
        """
        Analyze order flow imbalances from order book data
        """
        imbalances = []
        stacked_count = 0
        last_imbalance_type = None
        
        for idx, row in order_book_data.iterrows():
            timestamp = pd.to_datetime(idx)
            
            # Calculate bid/ask imbalances
            bid_volume = row.get('bid_volume', 0)
            ask_volume = row.get('ask_volume', 0)
            
            if bid_volume > 0 and ask_volume > 0:
                bid_imbalance = bid_volume / ask_volume
                ask_imbalance = ask_volume / bid_volume
            else:
                bid_imbalance = 0
                ask_imbalance = 0
            
            # Determine imbalance type
            if bid_imbalance > self.imbalance_threshold:
                imbalance_type = 'bid_dominant'
                total_imbalance = bid_imbalance
            elif ask_imbalance > self.imbalance_threshold:
                imbalance_type = 'ask_dominant'
                total_imbalance = ask_imbalance
            else:
                imbalance_type = 'balanced'
                total_imbalance = max(bid_imbalance, ask_imbalance)
            
            # Track stacked imbalances
            if imbalance_type == last_imbalance_type and imbalance_type != 'balanced':
                stacked_count += 1
            else:
                stacked_count = 1 if imbalance_type != 'balanced' else 0
            
            last_imbalance_type = imbalance_type
            
            # Create imbalance object
            imbalance = OrderFlowImbalance(
                timestamp=timestamp,
                bid_imbalance=bid_imbalance,
                ask_imbalance=ask_imbalance,
                total_imbalance=total_imbalance,
                imbalance_ratio=bid_imbalance / ask_imbalance if ask_imbalance > 0 else 0,
                stacked_imbalances=stacked_count,
                imbalance_type=imbalance_type
            )
            
            imbalances.append(imbalance)
            self.imbalances.append(imbalance)
        
        return imbalances
    
    def calculate_absorption_zones(self,
                                  delta_bars: Optional[List[DeltaBar]] = None,
                                  threshold_percentile: float = 80) -> List[Dict[str, Any]]:
        """
        Identify absorption zones (high volume with little price movement)
        """
        if delta_bars is None:
            delta_bars = list(self.delta_bars)
        
        if not delta_bars:
            return []
        
        # Calculate absorption ratios
        absorption_ratios = [bar.absorption_ratio for bar in delta_bars]
        
        # Find threshold
        threshold = np.percentile(absorption_ratios, threshold_percentile)
        
        # Identify absorption zones
        zones = []
        current_zone = None
        
        for i, bar in enumerate(delta_bars):
            if bar.absorption_ratio < threshold:  # Low ratio = high absorption
                if current_zone is None:
                    current_zone = {
                        'start_time': bar.timestamp,
                        'start_price': bar.open_price,
                        'bars': [bar]
                    }
                else:
                    current_zone['bars'].append(bar)
            else:
                if current_zone and len(current_zone['bars']) >= 3:
                    # Complete the zone
                    current_zone['end_time'] = current_zone['bars'][-1].timestamp
                    current_zone['end_price'] = current_zone['bars'][-1].close_price
                    current_zone['total_volume'] = sum(b.total_volume for b in current_zone['bars'])
                    current_zone['net_delta'] = sum(b.delta for b in current_zone['bars'])
                    current_zone['price_range'] = max(b.high_price for b in current_zone['bars']) - \
                                                 min(b.low_price for b in current_zone['bars'])
                    current_zone['absorption_strength'] = 100 - (bar.absorption_ratio / threshold * 100)
                    
                    # Determine zone type
                    if current_zone['net_delta'] > 0:
                        current_zone['type'] = 'bullish_absorption'
                    else:
                        current_zone['type'] = 'bearish_absorption'
                    
                    zones.append(current_zone)
                
                current_zone = None
        
        # Add final zone if exists
        if current_zone and len(current_zone['bars']) >= 3:
            current_zone['end_time'] = current_zone['bars'][-1].timestamp
            current_zone['end_price'] = current_zone['bars'][-1].close_price
            current_zone['total_volume'] = sum(b.total_volume for b in current_zone['bars'])
            current_zone['net_delta'] = sum(b.delta for b in current_zone['bars'])
            current_zone['price_range'] = max(b.high_price for b in current_zone['bars']) - \
                                         min(b.low_price for b in current_zone['bars'])
            zones.append(current_zone)
        
        return zones
    
    def analyze_delta_patterns(self,
                              delta_bars: Optional[List[DeltaBar]] = None) -> Dict[str, Any]:
        """
        Analyze patterns in delta volume
        """
        if delta_bars is None:
            delta_bars = list(self.delta_bars)
        
        if len(delta_bars) < 10:
            return {}
        
        patterns = {
            'exhaustion_signals': [],
            'continuation_signals': [],
            'reversal_signals': [],
            'delta_shifts': []
        }
        
        # Detect exhaustion (high volume, small price move, delta reversal)
        for i in range(2, len(delta_bars)):
            current = delta_bars[i]
            prev = delta_bars[i-1]
            prev2 = delta_bars[i-2]
            
            # Volume spike
            avg_volume = np.mean([b.total_volume for b in delta_bars[max(0, i-10):i]])
            if current.total_volume > avg_volume * 2:
                # Check for exhaustion
                price_move = abs(current.close_price - current.open_price)
                expected_move = abs(prev.close_price - prev.open_price)
                
                if price_move < expected_move * 0.5:  # Small move despite volume
                    # Check delta reversal
                    if (prev.delta > 0 and current.delta < 0) or \
                       (prev.delta < 0 and current.delta > 0):
                        patterns['exhaustion_signals'].append({
                            'timestamp': current.timestamp,
                            'price': current.close_price,
                            'volume_spike': current.total_volume / avg_volume,
                            'delta_reversal': True,
                            'type': 'selling_exhaustion' if prev.delta > 0 else 'buying_exhaustion'
                        })
        
        # Detect continuation (delta alignment with price)
        for i in range(5, len(delta_bars)):
            window = delta_bars[i-5:i]
            price_direction = 1 if window[-1].close_price > window[0].close_price else -1
            delta_direction = 1 if window[-1].cumulative_delta > window[0].cumulative_delta else -1
            
            if price_direction == delta_direction:
                avg_delta = np.mean([abs(b.delta) for b in window])
                if abs(window[-1].delta) > avg_delta * 1.5:
                    patterns['continuation_signals'].append({
                        'timestamp': window[-1].timestamp,
                        'price': window[-1].close_price,
                        'direction': 'bullish' if price_direction > 0 else 'bearish',
                        'strength': min(100, abs(window[-1].delta) / avg_delta * 50)
                    })
        
        # Detect reversals (divergence + volume)
        divergences = self.detect_divergences(delta_bars)
        for div in divergences:
            if div.confirmed and div.strength > 70:
                patterns['reversal_signals'].append({
                    'timestamp': div.end_time,
                    'type': div.divergence_type,
                    'strength': div.strength,
                    'duration': div.duration.total_seconds() / 3600  # Hours
                })
        
        # Detect delta shifts (sudden change in order flow)
        for i in range(10, len(delta_bars)):
            recent_delta = np.mean([b.delta for b in delta_bars[i-5:i]])
            historical_delta = np.mean([b.delta for b in delta_bars[i-10:i-5]])
            
            if abs(recent_delta - historical_delta) > abs(historical_delta) * 2:
                patterns['delta_shifts'].append({
                    'timestamp': delta_bars[i-1].timestamp,
                    'price': delta_bars[i-1].close_price,
                    'shift_magnitude': (recent_delta - historical_delta) / abs(historical_delta) if historical_delta != 0 else 0,
                    'new_bias': 'bullish' if recent_delta > 0 else 'bearish'
                })
        
        return patterns
    
    def calculate_delta_statistics(self,
                                  delta_bars: Optional[List[DeltaBar]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive delta statistics
        """
        if delta_bars is None:
            delta_bars = list(self.delta_bars)
        
        if not delta_bars:
            return {}
        
        deltas = [bar.delta for bar in delta_bars]
        cumulative_deltas = [bar.cumulative_delta for bar in delta_bars]
        volumes = [bar.total_volume for bar in delta_bars]
        
        stats = {
            # Basic statistics
            'mean_delta': np.mean(deltas),
            'std_delta': np.std(deltas),
            'skew_delta': stats.skew(deltas),
            'kurtosis_delta': stats.kurtosis(deltas),
            
            # Cumulative statistics
            'final_cumulative_delta': cumulative_deltas[-1] if cumulative_deltas else 0,
            'max_cumulative_delta': max(cumulative_deltas) if cumulative_deltas else 0,
            'min_cumulative_delta': min(cumulative_deltas) if cumulative_deltas else 0,
            'cumulative_range': max(cumulative_deltas) - min(cumulative_deltas) if cumulative_deltas else 0,
            
            # Volume-weighted statistics
            'volume_weighted_delta': np.average(deltas, weights=volumes) if volumes else 0,
            
            # Directional statistics
            'positive_delta_bars': sum(1 for d in deltas if d > 0),
            'negative_delta_bars': sum(1 for d in deltas if d < 0),
            'neutral_delta_bars': sum(1 for d in deltas if d == 0),
            'bullish_percentage': sum(1 for d in deltas if d > 0) / len(deltas) * 100 if deltas else 0,
            
            # Momentum statistics
            'delta_momentum': self._calculate_momentum(deltas),
            'delta_acceleration': self._calculate_acceleration(deltas),
            
            # Efficiency statistics
            'delta_efficiency': abs(cumulative_deltas[-1]) / sum(abs(d) for d in deltas) if deltas and sum(abs(d) for d in deltas) > 0 else 0
        }
        
        return stats
    
    def generate_trading_signals(self,
                                current_price: float,
                                lookback: int = 20) -> Dict[str, Any]:
        """
        Generate trading signals from delta analysis
        """
        if len(self.delta_bars) < lookback:
            return {'signal': 'neutral', 'strength': 0, 'reasons': []}
        
        recent_bars = list(self.delta_bars)[-lookback:]
        
        signals = {
            'signal': 'neutral',
            'strength': 0,
            'reasons': [],
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None
        }
        
        # Get statistics
        stats = self.calculate_delta_statistics(recent_bars)
        patterns = self.analyze_delta_patterns(recent_bars)
        
        # Check for strong bullish delta
        if stats['final_cumulative_delta'] > 0 and stats['bullish_percentage'] > 60:
            signals['signal'] = 'buy'
            signals['strength'] = min(100, stats['bullish_percentage'])
            signals['reasons'].append('positive_delta_bias')
            
            # Set stops
            recent_low = min(bar.low_price for bar in recent_bars[-5:])
            signals['stop_loss'] = recent_low * 0.995
            signals['take_profit'] = current_price * 1.02
        
        # Check for strong bearish delta
        elif stats['final_cumulative_delta'] < 0 and stats['bullish_percentage'] < 40:
            signals['signal'] = 'sell'
            signals['strength'] = min(100, 100 - stats['bullish_percentage'])
            signals['reasons'].append('negative_delta_bias')
            
            # Set stops
            recent_high = max(bar.high_price for bar in recent_bars[-5:])
            signals['stop_loss'] = recent_high * 1.005
            signals['take_profit'] = current_price * 0.98
        
        # Check for exhaustion
        if patterns['exhaustion_signals']:
            latest_exhaustion = patterns['exhaustion_signals'][-1]
            if latest_exhaustion['type'] == 'selling_exhaustion':
                signals['signal'] = 'buy'
                signals['reasons'].append('selling_exhaustion')
                signals['strength'] = max(signals['strength'], 70)
            else:
                signals['signal'] = 'sell'
                signals['reasons'].append('buying_exhaustion')
                signals['strength'] = max(signals['strength'], 70)
        
        # Check for reversal signals
        if patterns['reversal_signals']:
            latest_reversal = patterns['reversal_signals'][-1]
            if latest_reversal['type'] == 'bullish':
                signals['signal'] = 'buy'
                signals['reasons'].append('bullish_reversal')
                signals['strength'] = max(signals['strength'], latest_reversal['strength'])
            else:
                signals['signal'] = 'sell'
                signals['reasons'].append('bearish_reversal')
                signals['strength'] = max(signals['strength'], latest_reversal['strength'])
        
        # Check order flow imbalances
        if self.imbalances:
            recent_imbalances = self.imbalances[-10:]
            bid_dominant = sum(1 for i in recent_imbalances if i.imbalance_type == 'bid_dominant')
            ask_dominant = sum(1 for i in recent_imbalances if i.imbalance_type == 'ask_dominant')
            
            if bid_dominant > ask_dominant * 2:
                signals['signal'] = 'buy'
                signals['reasons'].append('bid_imbalance')
                signals['strength'] = min(100, signals['strength'] + 20)
            elif ask_dominant > bid_dominant * 2:
                signals['signal'] = 'sell'
                signals['reasons'].append('ask_imbalance')
                signals['strength'] = min(100, signals['strength'] + 20)
        
        return signals
    
    def get_rl_features(self, lookback: int = 20) -> np.ndarray:
        """
        Extract features for RL model integration
        """
        if len(self.delta_bars) < lookback:
            return np.zeros(20)  # Return zero features if not enough data
        
        recent_bars = list(self.delta_bars)[-lookback:]
        
        # Calculate statistics
        stats = self.calculate_delta_statistics(recent_bars)
        patterns = self.analyze_delta_patterns(recent_bars)
        
        features = []
        
        # Delta statistics (normalized)
        features.extend([
            stats['mean_delta'] / 10000,  # Normalized
            stats['std_delta'] / 10000,
            stats['skew_delta'] / 10,
            stats['final_cumulative_delta'] / 100000,
            stats['bullish_percentage'] / 100,
            stats['delta_efficiency']
        ])
        
        # Pattern counts
        features.extend([
            len(patterns.get('exhaustion_signals', [])) / 10,
            len(patterns.get('continuation_signals', [])) / 10,
            len(patterns.get('reversal_signals', [])) / 10,
            len(patterns.get('delta_shifts', [])) / 10
        ])
        
        # Recent delta trend
        if len(recent_bars) >= 5:
            recent_deltas = [b.delta for b in recent_bars[-5:]]
            delta_trend = self._calculate_trend(recent_deltas)
            features.append(delta_trend)
        else:
            features.append(0)
        
        # Absorption ratio
        recent_absorption = np.mean([b.absorption_ratio for b in recent_bars])
        features.append(recent_absorption / 100)
        
        # Imbalance features
        if self.imbalances:
            recent_imbalances = self.imbalances[-10:]
            avg_imbalance = np.mean([i.total_imbalance for i in recent_imbalances])
            max_stacked = max(i.stacked_imbalances for i in recent_imbalances)
            features.extend([avg_imbalance / 10, max_stacked / 10])
        else:
            features.extend([0, 0])
        
        # Momentum features
        features.extend([
            stats.get('delta_momentum', 0) / 100,
            stats.get('delta_acceleration', 0) / 100
        ])
        
        # Divergence features
        recent_divergences = [d for d in self.divergences if d.end_time >= recent_bars[0].timestamp]
        bullish_div = sum(1 for d in recent_divergences if d.divergence_type == 'bullish')
        bearish_div = sum(1 for d in recent_divergences if d.divergence_type == 'bearish')
        features.extend([bullish_div / 5, bearish_div / 5])
        
        return np.array(features)
    
    # Private helper methods
    def _estimate_delta_tick(self,
                           open_price: float,
                           high_price: float,
                           low_price: float,
                           close_price: float,
                           volume: float) -> Tuple[float, float]:
        """Estimate buy/sell volume using tick method"""
        # Simplified tick rule
        if close_price > (high_price + low_price) / 2:
            # Bullish bar
            buy_ratio = 0.6 + (close_price - open_price) / (high_price - low_price + 0.0001) * 0.2
        else:
            # Bearish bar
            buy_ratio = 0.4 - (open_price - close_price) / (high_price - low_price + 0.0001) * 0.2
        
        buy_ratio = max(0, min(1, buy_ratio))
        
        buy_volume = volume * buy_ratio
        sell_volume = volume * (1 - buy_ratio)
        
        return buy_volume, sell_volume
    
    def _classify_trades(self, row: pd.Series) -> Tuple[float, float]:
        """Classify trades as buy or sell based on trade data"""
        # This would need actual trade classification logic
        # Placeholder implementation
        total_volume = row.get('volume', 0)
        
        # Use price movement as proxy
        if 'close' in row and 'open' in row:
            if row['close'] > row['open']:
                return total_volume * 0.6, total_volume * 0.4
            else:
                return total_volume * 0.4, total_volume * 0.6
        
        return total_volume * 0.5, total_volume * 0.5
    
    def _should_reset_cumulative(self, current_time: datetime, last_reset: Optional[datetime]) -> bool:
        """Check if cumulative delta should be reset"""
        if self.cumulative_reset_period is None or last_reset is None:
            return False
        
        if self.cumulative_reset_period == 'daily':
            return current_time.date() != last_reset.date()
        elif self.cumulative_reset_period == 'weekly':
            return current_time.isocalendar()[1] != last_reset.isocalendar()[1]
        
        return False
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression"""
        if len(values) < 2:
            return 0
        
        X = np.arange(len(values)).reshape(-1, 1)
        y = np.array(values)
        
        try:
            self.regression_model.fit(X, y)
            slope = self.regression_model.coef_[0]
            
            # Normalize by average value
            avg_value = np.mean(np.abs(values))
            if avg_value > 0:
                return slope / avg_value
            return slope
        except:
            return 0
    
    def _calculate_divergence_strength(self, price_trend: float, delta_trend: float) -> float:
        """Calculate divergence strength"""
        # Stronger divergence when trends are more opposite
        divergence_magnitude = abs(price_trend - delta_trend)
        
        # Scale to 0-100
        strength = min(100, divergence_magnitude * 100)
        
        # Adjust for trend strength
        trend_strength = (abs(price_trend) + abs(delta_trend)) / 2
        strength = strength * (0.5 + trend_strength * 0.5)
        
        return strength
    
    def _calculate_momentum(self, values: List[float]) -> float:
        """Calculate momentum of values"""
        if len(values) < 10:
            return 0
        
        recent = np.mean(values[-5:])
        historical = np.mean(values[-10:-5])
        
        if historical != 0:
            return (recent - historical) / abs(historical) * 100
        return 0
    
    def _calculate_acceleration(self, values: List[float]) -> float:
        """Calculate acceleration (change in momentum)"""
        if len(values) < 15:
            return 0
        
        momentum1 = self._calculate_momentum(values[-15:-5])
        momentum2 = self._calculate_momentum(values[-10:])
        
        return momentum2 - momentum1