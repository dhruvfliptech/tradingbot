"""
Sophisticated Liquidity Hunting Strategy for Crypto Trading
Identifies hidden liquidity pools, iceberg orders, and institutional activity patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
from scipy import stats, signal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class LiquiditySignal:
    """Container for liquidity hunting signals"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'iceberg', 'stop_hunt', 'accumulation', 'distribution', 'squeeze'
    direction: str  # 'buy', 'sell', 'neutral'
    strength: float  # 0-1 confidence score
    price_level: float
    volume_estimate: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for RL integration"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'direction': self.direction,
            'strength': self.strength,
            'price_level': self.price_level,
            'volume_estimate': self.volume_estimate,
            'metadata': self.metadata
        }


@dataclass
class MarketMicrostructure:
    """Market microstructure analysis results"""
    bid_ask_spread: float
    effective_spread: float
    realized_spread: float
    price_impact: float
    order_flow_imbalance: float
    kyle_lambda: float  # Price impact coefficient
    adverse_selection: float
    inventory_risk: float
    
    def get_liquidity_score(self) -> float:
        """Calculate overall liquidity score"""
        # Lower spreads and impacts indicate better liquidity
        spread_score = 1 / (1 + self.effective_spread * 100)
        impact_score = 1 / (1 + abs(self.price_impact) * 100)
        imbalance_score = 1 / (1 + abs(self.order_flow_imbalance))
        
        # Weighted average
        weights = [0.3, 0.3, 0.4]
        scores = [spread_score, impact_score, imbalance_score]
        
        return np.average(scores, weights=weights)


class LiquidityHuntingStrategy:
    """
    Advanced liquidity hunting strategy using order book dynamics,
    volume analysis, and microstructure patterns
    """
    
    def __init__(self, 
                 symbol: str,
                 lookback_periods: int = 100,
                 sensitivity: float = 0.7,
                 min_signal_strength: float = 0.6):
        """
        Initialize liquidity hunting strategy
        
        Args:
            symbol: Trading symbol
            lookback_periods: Historical periods for pattern analysis
            sensitivity: Detection sensitivity (0-1)
            min_signal_strength: Minimum signal strength to trigger
        """
        self.symbol = symbol
        self.lookback_periods = lookback_periods
        self.sensitivity = sensitivity
        self.min_signal_strength = min_signal_strength
        
        # Historical data storage
        self.order_book_history = deque(maxlen=lookback_periods)
        self.trade_history = deque(maxlen=lookback_periods * 10)
        self.signal_history = deque(maxlen=50)
        
        # Pattern detection parameters
        self.iceberg_threshold = 0.7
        self.stop_hunt_threshold = 0.75
        self.accumulation_threshold = 0.65
        
        # Microstructure models
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # State tracking
        self.current_microstructure = None
        self.liquidity_pools = {}
        self.institutional_levels = []
        
        logger.info(f"LiquidityHuntingStrategy initialized for {symbol}")
    
    def update(self, 
               order_book: Dict[str, Any],
               trades: List[Dict[str, Any]],
               market_data: Optional[Dict[str, Any]] = None) -> List[LiquiditySignal]:
        """
        Update strategy with new market data and generate signals
        
        Args:
            order_book: Current order book snapshot
            trades: Recent trades
            market_data: Additional market data
            
        Returns:
            List of liquidity signals
        """
        try:
            # Store historical data
            self.order_book_history.append(order_book)
            self.trade_history.extend(trades)
            
            # Analyze microstructure
            self.current_microstructure = self._analyze_microstructure(
                order_book, trades
            )
            
            # Generate signals
            signals = []
            
            # Detect iceberg orders
            iceberg_signals = self._detect_iceberg_orders(order_book, trades)
            signals.extend(iceberg_signals)
            
            # Detect stop hunting
            stop_hunt_signals = self._detect_stop_hunting(order_book, trades, market_data)
            signals.extend(stop_hunt_signals)
            
            # Detect accumulation/distribution
            acc_dist_signals = self._detect_accumulation_distribution(
                order_book, trades, market_data
            )
            signals.extend(acc_dist_signals)
            
            # Detect liquidity squeezes
            squeeze_signals = self._detect_liquidity_squeeze(order_book, trades)
            signals.extend(squeeze_signals)
            
            # Identify institutional levels
            self._update_institutional_levels(order_book, trades)
            
            # Filter signals by strength
            signals = [s for s in signals if s.strength >= self.min_signal_strength]
            
            # Store signals
            self.signal_history.extend(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error updating liquidity hunting strategy: {e}")
            return []
    
    def _analyze_microstructure(self, 
                                order_book: Dict[str, Any],
                                trades: List[Dict[str, Any]]) -> MarketMicrostructure:
        """Analyze market microstructure metrics"""
        try:
            bids = np.array(order_book.get('bids', []))
            asks = np.array(order_book.get('asks', []))
            
            if len(bids) == 0 or len(asks) == 0:
                return None
            
            # Best bid/ask
            best_bid = bids[0][0] if len(bids) > 0 else 0
            best_ask = asks[0][0] if len(asks) > 0 else float('inf')
            mid_price = (best_bid + best_ask) / 2
            
            # Spreads
            bid_ask_spread = (best_ask - best_bid) / mid_price
            
            # Effective spread from trades
            effective_spread = self._calculate_effective_spread(trades, mid_price)
            
            # Realized spread (temporary vs permanent impact)
            realized_spread = self._calculate_realized_spread(trades, mid_price)
            
            # Price impact (Kyle's lambda)
            kyle_lambda = self._estimate_kyle_lambda(trades, order_book)
            
            # Order flow imbalance
            bid_volume = sum([b[1] for b in bids[:10]])
            ask_volume = sum([a[1] for a in asks[:10]])
            order_flow_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # Price impact from order flow
            price_impact = self._calculate_price_impact(trades, order_book)
            
            # Adverse selection component
            adverse_selection = effective_spread - realized_spread
            
            # Inventory risk
            inventory_risk = self._calculate_inventory_risk(trades, order_book)
            
            return MarketMicrostructure(
                bid_ask_spread=bid_ask_spread,
                effective_spread=effective_spread,
                realized_spread=realized_spread,
                price_impact=price_impact,
                order_flow_imbalance=order_flow_imbalance,
                kyle_lambda=kyle_lambda,
                adverse_selection=adverse_selection,
                inventory_risk=inventory_risk
            )
            
        except Exception as e:
            logger.error(f"Error analyzing microstructure: {e}")
            return None
    
    def _detect_iceberg_orders(self, 
                               order_book: Dict[str, Any],
                               trades: List[Dict[str, Any]]) -> List[LiquiditySignal]:
        """Detect hidden iceberg orders in the order book"""
        signals = []
        
        try:
            bids = np.array(order_book.get('bids', []))
            asks = np.array(order_book.get('asks', []))
            
            # Analyze bid side
            bid_signals = self._analyze_iceberg_side(
                bids, trades, 'buy', order_book['timestamp']
            )
            signals.extend(bid_signals)
            
            # Analyze ask side
            ask_signals = self._analyze_iceberg_side(
                asks, trades, 'sell', order_book['timestamp']
            )
            signals.extend(ask_signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting iceberg orders: {e}")
            return []
    
    def _analyze_iceberg_side(self, 
                              orders: np.ndarray,
                              trades: List[Dict[str, Any]],
                              side: str,
                              timestamp: datetime) -> List[LiquiditySignal]:
        """Analyze one side of order book for iceberg orders"""
        signals = []
        
        if len(orders) < 10:
            return signals
        
        try:
            prices = orders[:, 0]
            volumes = orders[:, 1]
            
            # Pattern 1: Persistent reloading at specific price levels
            if len(self.order_book_history) >= 10:
                reload_signals = self._detect_order_reloading(
                    prices, volumes, side, timestamp
                )
                signals.extend(reload_signals)
            
            # Pattern 2: Volume clustering with execution patterns
            cluster_signals = self._detect_volume_clustering(
                prices, volumes, trades, side, timestamp
            )
            signals.extend(cluster_signals)
            
            # Pattern 3: Anomalous volume distribution
            anomaly_signals = self._detect_volume_anomalies(
                prices, volumes, side, timestamp
            )
            signals.extend(anomaly_signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing iceberg side: {e}")
            return []
    
    def _detect_order_reloading(self, 
                                prices: np.ndarray,
                                volumes: np.ndarray,
                                side: str,
                                timestamp: datetime) -> List[LiquiditySignal]:
        """Detect order reloading patterns (iceberg characteristic)"""
        signals = []
        
        try:
            # Track volume changes at each price level
            reload_scores = {}
            
            for i, price in enumerate(prices[:20]):  # Check top 20 levels
                price_history = []
                
                # Look through historical order books
                for hist_book in self.order_book_history:
                    hist_side = hist_book.get('bids' if side == 'buy' else 'asks', [])
                    
                    # Find same price level
                    for hist_price, hist_vol in hist_side:
                        if abs(hist_price - price) < price * 0.0001:  # Within 0.01%
                            price_history.append(hist_vol)
                            break
                
                if len(price_history) >= 5:
                    # Check for reloading pattern
                    volume_std = np.std(price_history)
                    volume_mean = np.mean(price_history)
                    
                    # High variance with consistent presence indicates reloading
                    if volume_std > volume_mean * 0.3 and len(price_history) > len(self.order_book_history) * 0.7:
                        reload_score = min(1.0, volume_std / volume_mean)
                        reload_scores[price] = reload_score
            
            # Generate signals for significant reload patterns
            for price, score in reload_scores.items():
                if score >= self.iceberg_threshold:
                    signal = LiquiditySignal(
                        timestamp=timestamp,
                        symbol=self.symbol,
                        signal_type='iceberg',
                        direction=side,
                        strength=score,
                        price_level=price,
                        volume_estimate=volumes[prices == price][0] * 3,  # Estimate hidden size
                        metadata={
                            'pattern': 'order_reloading',
                            'reload_frequency': len(price_history)
                        }
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting order reloading: {e}")
            return []
    
    def _detect_volume_clustering(self, 
                                  prices: np.ndarray,
                                  volumes: np.ndarray,
                                  trades: List[Dict[str, Any]],
                                  side: str,
                                  timestamp: datetime) -> List[LiquiditySignal]:
        """Detect volume clustering patterns"""
        signals = []
        
        try:
            # Use DBSCAN-like approach for volume clustering
            volume_threshold = np.percentile(volumes, 75)
            
            # Find price levels with significant volume
            significant_levels = prices[volumes > volume_threshold]
            
            if len(significant_levels) > 0:
                # Check execution patterns at these levels
                for price_level in significant_levels[:5]:  # Top 5 levels
                    execution_rate = self._calculate_execution_rate(
                        price_level, trades, side
                    )
                    
                    # High volume with low execution suggests hidden liquidity
                    if execution_rate < 0.3:
                        volume_idx = np.where(prices == price_level)[0][0]
                        strength = (1 - execution_rate) * (volumes[volume_idx] / volume_threshold)
                        strength = min(1.0, strength)
                        
                        if strength >= self.iceberg_threshold:
                            signal = LiquiditySignal(
                                timestamp=timestamp,
                                symbol=self.symbol,
                                signal_type='iceberg',
                                direction=side,
                                strength=strength,
                                price_level=price_level,
                                volume_estimate=volumes[volume_idx] * 2.5,
                                metadata={
                                    'pattern': 'volume_clustering',
                                    'execution_rate': execution_rate
                                }
                            )
                            signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting volume clustering: {e}")
            return []
    
    def _detect_volume_anomalies(self, 
                                 prices: np.ndarray,
                                 volumes: np.ndarray,
                                 side: str,
                                 timestamp: datetime) -> List[LiquiditySignal]:
        """Detect anomalous volume patterns using statistical methods"""
        signals = []
        
        try:
            if len(volumes) < 20:
                return signals
            
            # Fit exponential decay model (normal pattern)
            log_volumes = np.log(volumes[:20] + 1)
            x = np.arange(20)
            
            # Linear regression on log volumes
            slope, intercept = np.polyfit(x, log_volumes, 1)
            expected_volumes = np.exp(slope * x + intercept)
            
            # Calculate residuals
            residuals = volumes[:20] - expected_volumes
            residual_std = np.std(residuals)
            
            # Find anomalies
            for i, (price, volume, residual) in enumerate(zip(prices[:20], volumes[:20], residuals)):
                if residual > 2 * residual_std:  # Significant positive anomaly
                    z_score = residual / residual_std
                    strength = min(1.0, z_score / 4)  # Normalize to 0-1
                    
                    if strength >= self.iceberg_threshold:
                        signal = LiquiditySignal(
                            timestamp=timestamp,
                            symbol=self.symbol,
                            signal_type='iceberg',
                            direction=side,
                            strength=strength,
                            price_level=price,
                            volume_estimate=volume * (1 + z_score),
                            metadata={
                                'pattern': 'volume_anomaly',
                                'z_score': z_score,
                                'level_depth': i
                            }
                        )
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting volume anomalies: {e}")
            return []
    
    def _detect_stop_hunting(self, 
                             order_book: Dict[str, Any],
                             trades: List[Dict[str, Any]],
                             market_data: Optional[Dict[str, Any]]) -> List[LiquiditySignal]:
        """Detect stop hunting patterns"""
        signals = []
        
        try:
            if not market_data:
                return signals
            
            current_price = market_data.get('price', 0)
            high_24h = market_data.get('high_24h', 0)
            low_24h = market_data.get('low_24h', 0)
            
            if current_price == 0:
                return signals
            
            # Identify key levels where stops likely cluster
            stop_levels = self._identify_stop_levels(
                current_price, high_24h, low_24h, market_data
            )
            
            # Check for stop hunting patterns at each level
            for level, level_type in stop_levels:
                hunting_score = self._calculate_stop_hunting_score(
                    level, level_type, order_book, trades, current_price
                )
                
                if hunting_score >= self.stop_hunt_threshold:
                    direction = 'sell' if level > current_price else 'buy'
                    
                    signal = LiquiditySignal(
                        timestamp=datetime.fromtimestamp(order_book['timestamp']),
                        symbol=self.symbol,
                        signal_type='stop_hunt',
                        direction=direction,
                        strength=hunting_score,
                        price_level=level,
                        volume_estimate=self._estimate_stop_volume(level, order_book),
                        metadata={
                            'level_type': level_type,
                            'distance_from_price': abs(level - current_price) / current_price
                        }
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting stop hunting: {e}")
            return []
    
    def _identify_stop_levels(self, 
                              current_price: float,
                              high_24h: float,
                              low_24h: float,
                              market_data: Dict[str, Any]) -> List[Tuple[float, str]]:
        """Identify price levels where stops are likely to cluster"""
        stop_levels = []
        
        try:
            # Recent highs and lows
            stop_levels.append((high_24h * 1.002, 'recent_high'))
            stop_levels.append((low_24h * 0.998, 'recent_low'))
            
            # Round number levels
            round_levels = [
                round(current_price, -int(np.log10(current_price)) + 1),
                round(current_price, -int(np.log10(current_price)) + 2)
            ]
            for level in round_levels:
                if abs(level - current_price) / current_price < 0.02:  # Within 2%
                    stop_levels.append((level, 'round_number'))
            
            # Technical levels (if available)
            if 'support_levels' in market_data:
                for level in market_data['support_levels']:
                    stop_levels.append((level * 0.995, 'support'))
            
            if 'resistance_levels' in market_data:
                for level in market_data['resistance_levels']:
                    stop_levels.append((level * 1.005, 'resistance'))
            
            # Moving average levels (if available)
            for ma_type in ['sma_50', 'sma_200', 'ema_20']:
                if ma_type in market_data:
                    ma_value = market_data[ma_type]
                    stop_levels.append((ma_value * 0.997, f'{ma_type}_below'))
                    stop_levels.append((ma_value * 1.003, f'{ma_type}_above'))
            
            # Filter levels within reasonable distance
            filtered_levels = []
            for level, level_type in stop_levels:
                distance = abs(level - current_price) / current_price
                if 0.001 < distance < 0.05:  # Between 0.1% and 5%
                    filtered_levels.append((level, level_type))
            
            return filtered_levels
            
        except Exception as e:
            logger.error(f"Error identifying stop levels: {e}")
            return []
    
    def _calculate_stop_hunting_score(self, 
                                      level: float,
                                      level_type: str,
                                      order_book: Dict[str, Any],
                                      trades: List[Dict[str, Any]],
                                      current_price: float) -> float:
        """Calculate probability of stop hunting at a level"""
        try:
            score_components = []
            
            # 1. Order book imbalance towards the level
            imbalance_score = self._calculate_directional_imbalance(
                level, order_book, current_price
            )
            score_components.append(imbalance_score * 0.3)
            
            # 2. Recent price action approaching level
            approach_score = self._calculate_approach_pattern(
                level, trades, current_price
            )
            score_components.append(approach_score * 0.25)
            
            # 3. Volume spike near level
            volume_score = self._calculate_volume_spike(level, trades)
            score_components.append(volume_score * 0.25)
            
            # 4. Level significance
            significance_weights = {
                'recent_high': 0.9,
                'recent_low': 0.9,
                'round_number': 0.7,
                'support': 0.8,
                'resistance': 0.8
            }
            significance_score = significance_weights.get(level_type, 0.5)
            score_components.append(significance_score * 0.2)
            
            return sum(score_components)
            
        except Exception as e:
            logger.error(f"Error calculating stop hunting score: {e}")
            return 0.0
    
    def _detect_accumulation_distribution(self, 
                                         order_book: Dict[str, Any],
                                         trades: List[Dict[str, Any]],
                                         market_data: Optional[Dict[str, Any]]) -> List[LiquiditySignal]:
        """Detect accumulation and distribution patterns"""
        signals = []
        
        try:
            if len(trades) < 20:
                return signals
            
            # Calculate Accumulation/Distribution line
            ad_line = self._calculate_ad_line(trades)
            
            # Calculate On-Balance Volume
            obv = self._calculate_obv(trades)
            
            # Analyze patterns
            if len(ad_line) >= 20 and len(obv) >= 20:
                # Trend analysis
                ad_trend = np.polyfit(range(len(ad_line[-20:])), ad_line[-20:], 1)[0]
                obv_trend = np.polyfit(range(len(obv[-20:])), obv[-20:], 1)[0]
                
                # Price trend
                prices = [t['price'] for t in trades[-20:]]
                price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
                
                # Divergence detection
                if ad_trend > 0 and price_trend <= 0:
                    # Accumulation (bullish divergence)
                    strength = min(1.0, abs(ad_trend) / (abs(price_trend) + 0.001))
                    if strength >= self.accumulation_threshold:
                        signal = LiquiditySignal(
                            timestamp=datetime.fromtimestamp(order_book['timestamp']),
                            symbol=self.symbol,
                            signal_type='accumulation',
                            direction='buy',
                            strength=strength,
                            price_level=prices[-1],
                            volume_estimate=np.mean([t['volume'] for t in trades[-20:]]) * 10,
                            metadata={
                                'ad_trend': ad_trend,
                                'obv_trend': obv_trend,
                                'price_trend': price_trend
                            }
                        )
                        signals.append(signal)
                
                elif ad_trend < 0 and price_trend >= 0:
                    # Distribution (bearish divergence)
                    strength = min(1.0, abs(ad_trend) / (abs(price_trend) + 0.001))
                    if strength >= self.accumulation_threshold:
                        signal = LiquiditySignal(
                            timestamp=datetime.fromtimestamp(order_book['timestamp']),
                            symbol=self.symbol,
                            signal_type='distribution',
                            direction='sell',
                            strength=strength,
                            price_level=prices[-1],
                            volume_estimate=np.mean([t['volume'] for t in trades[-20:]]) * 10,
                            metadata={
                                'ad_trend': ad_trend,
                                'obv_trend': obv_trend,
                                'price_trend': price_trend
                            }
                        )
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting accumulation/distribution: {e}")
            return []
    
    def _detect_liquidity_squeeze(self, 
                                 order_book: Dict[str, Any],
                                 trades: List[Dict[str, Any]]) -> List[LiquiditySignal]:
        """Detect liquidity squeeze conditions"""
        signals = []
        
        try:
            bids = np.array(order_book.get('bids', []))
            asks = np.array(order_book.get('asks', []))
            
            if len(bids) < 10 or len(asks) < 10:
                return signals
            
            # Calculate liquidity metrics
            bid_depth = sum([b[1] for b in bids[:10]])
            ask_depth = sum([a[1] for a in asks[:10]])
            total_depth = bid_depth + ask_depth
            
            # Historical comparison
            if len(self.order_book_history) >= 20:
                historical_depths = []
                for hist_book in self.order_book_history[-20:]:
                    hist_bids = hist_book.get('bids', [])
                    hist_asks = hist_book.get('asks', [])
                    if len(hist_bids) >= 10 and len(hist_asks) >= 10:
                        hist_depth = sum([b[1] for b in hist_bids[:10]]) + \
                                   sum([a[1] for a in hist_asks[:10]])
                        historical_depths.append(hist_depth)
                
                if historical_depths:
                    avg_depth = np.mean(historical_depths)
                    depth_ratio = total_depth / avg_depth
                    
                    # Squeeze detected when current depth is significantly lower
                    if depth_ratio < 0.5:
                        # Calculate spread widening
                        best_bid = bids[0][0]
                        best_ask = asks[0][0]
                        spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)
                        
                        historical_spreads = []
                        for hist_book in self.order_book_history[-20:]:
                            hist_bids = hist_book.get('bids', [])
                            hist_asks = hist_book.get('asks', [])
                            if hist_bids and hist_asks:
                                hist_spread = (hist_asks[0][0] - hist_bids[0][0]) / \
                                            ((hist_asks[0][0] + hist_bids[0][0]) / 2)
                                historical_spreads.append(hist_spread)
                        
                        avg_spread = np.mean(historical_spreads) if historical_spreads else spread
                        spread_ratio = spread / (avg_spread + 0.0001)
                        
                        # Combine metrics for squeeze score
                        squeeze_score = (1 - depth_ratio) * 0.6 + min(1.0, (spread_ratio - 1)) * 0.4
                        
                        if squeeze_score >= 0.6:
                            signal = LiquiditySignal(
                                timestamp=datetime.fromtimestamp(order_book['timestamp']),
                                symbol=self.symbol,
                                signal_type='squeeze',
                                direction='neutral',
                                strength=min(1.0, squeeze_score),
                                price_level=(best_bid + best_ask) / 2,
                                volume_estimate=total_depth,
                                metadata={
                                    'depth_ratio': depth_ratio,
                                    'spread_ratio': spread_ratio,
                                    'bid_depth': bid_depth,
                                    'ask_depth': ask_depth
                                }
                            )
                            signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting liquidity squeeze: {e}")
            return []
    
    def _update_institutional_levels(self, 
                                     order_book: Dict[str, Any],
                                     trades: List[Dict[str, Any]]):
        """Update tracked institutional trading levels"""
        try:
            # Identify large order levels
            bids = np.array(order_book.get('bids', []))
            asks = np.array(order_book.get('asks', []))
            
            if len(bids) > 0:
                bid_volumes = bids[:, 1]
                bid_threshold = np.percentile(bid_volumes, 95)
                large_bid_levels = bids[bid_volumes > bid_threshold, 0].tolist()
                
                for level in large_bid_levels:
                    if level not in self.liquidity_pools:
                        self.liquidity_pools[level] = {
                            'side': 'buy',
                            'first_seen': datetime.now(),
                            'volume_history': [],
                            'execution_count': 0
                        }
                    self.liquidity_pools[level]['volume_history'].append(
                        bids[bids[:, 0] == level, 1][0]
                    )
            
            if len(asks) > 0:
                ask_volumes = asks[:, 1]
                ask_threshold = np.percentile(ask_volumes, 95)
                large_ask_levels = asks[ask_volumes > ask_threshold, 0].tolist()
                
                for level in large_ask_levels:
                    if level not in self.liquidity_pools:
                        self.liquidity_pools[level] = {
                            'side': 'sell',
                            'first_seen': datetime.now(),
                            'volume_history': [],
                            'execution_count': 0
                        }
                    self.liquidity_pools[level]['volume_history'].append(
                        asks[asks[:, 0] == level, 1][0]
                    )
            
            # Clean up old levels
            current_time = datetime.now()
            levels_to_remove = []
            for level, info in self.liquidity_pools.items():
                age = (current_time - info['first_seen']).total_seconds()
                if age > 3600:  # Remove levels older than 1 hour
                    levels_to_remove.append(level)
            
            for level in levels_to_remove:
                del self.liquidity_pools[level]
                
        except Exception as e:
            logger.error(f"Error updating institutional levels: {e}")
    
    # Helper methods
    
    def _calculate_effective_spread(self, trades: List[Dict[str, Any]], mid_price: float) -> float:
        """Calculate effective spread from trades"""
        if not trades:
            return 0.0
        
        spreads = []
        for trade in trades[-20:]:  # Last 20 trades
            spread = 2 * abs(trade['price'] - mid_price) / mid_price
            spreads.append(spread)
        
        return np.mean(spreads) if spreads else 0.0
    
    def _calculate_realized_spread(self, trades: List[Dict[str, Any]], mid_price: float) -> float:
        """Calculate realized spread (temporary component)"""
        if len(trades) < 10:
            return 0.0
        
        realized_spreads = []
        for i in range(len(trades) - 5):
            trade_price = trades[i]['price']
            future_mid = np.mean([t['price'] for t in trades[i+1:i+6]])
            realized = 2 * abs(trade_price - future_mid) / future_mid
            realized_spreads.append(realized)
        
        return np.mean(realized_spreads) if realized_spreads else 0.0
    
    def _estimate_kyle_lambda(self, trades: List[Dict[str, Any]], order_book: Dict[str, Any]) -> float:
        """Estimate Kyle's lambda (price impact coefficient)"""
        if len(trades) < 10:
            return 0.0
        
        try:
            # Calculate signed volume and price changes
            signed_volumes = []
            price_changes = []
            
            for i in range(1, min(len(trades), 50)):
                # Determine trade direction
                mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
                is_buy = trades[i]['price'] > mid_price
                
                signed_volume = trades[i]['volume'] if is_buy else -trades[i]['volume']
                price_change = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                
                signed_volumes.append(signed_volume)
                price_changes.append(price_change)
            
            if len(signed_volumes) > 5:
                # Linear regression: price_change = lambda * signed_volume
                lambda_estimate = np.polyfit(signed_volumes, price_changes, 1)[0]
                return abs(lambda_estimate)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error estimating Kyle's lambda: {e}")
            return 0.0
    
    def _calculate_price_impact(self, trades: List[Dict[str, Any]], order_book: Dict[str, Any]) -> float:
        """Calculate price impact of trades"""
        if len(trades) < 5:
            return 0.0
        
        impacts = []
        for i in range(len(trades) - 1):
            volume = trades[i]['volume']
            price_change = abs(trades[i+1]['price'] - trades[i]['price']) / trades[i]['price']
            impact = price_change / (volume + 0.001)
            impacts.append(impact)
        
        return np.mean(impacts) if impacts else 0.0
    
    def _calculate_inventory_risk(self, trades: List[Dict[str, Any]], order_book: Dict[str, Any]) -> float:
        """Calculate inventory risk from order flow"""
        if not trades:
            return 0.0
        
        # Calculate net order flow
        buy_volume = sum([t['volume'] for t in trades if t.get('side') == 'buy'])
        sell_volume = sum([t['volume'] for t in trades if t.get('side') == 'sell'])
        
        net_flow = buy_volume - sell_volume
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            inventory_risk = abs(net_flow) / total_volume
            return inventory_risk
        
        return 0.0
    
    def _calculate_execution_rate(self, price_level: float, trades: List[Dict[str, Any]], side: str) -> float:
        """Calculate execution rate at a price level"""
        if not trades:
            return 0.0
        
        executions = 0
        opportunities = 0
        
        for trade in trades:
            if side == 'buy' and trade['price'] <= price_level * 1.001:
                executions += 1
            elif side == 'sell' and trade['price'] >= price_level * 0.999:
                executions += 1
            opportunities += 1
        
        return executions / opportunities if opportunities > 0 else 0.0
    
    def _calculate_directional_imbalance(self, level: float, order_book: Dict[str, Any], current_price: float) -> float:
        """Calculate order book imbalance towards a level"""
        bids = np.array(order_book.get('bids', []))
        asks = np.array(order_book.get('asks', []))
        
        if len(bids) == 0 or len(asks) == 0:
            return 0.0
        
        if level > current_price:  # Level above (resistance)
            # More buying pressure suggests stop hunting above
            bid_volume = sum([b[1] for b in bids[:10]])
            ask_volume = sum([a[1] for a in asks[:10]])
        else:  # Level below (support)
            # More selling pressure suggests stop hunting below
            ask_volume = sum([a[1] for a in asks[:10]])
            bid_volume = sum([b[1] for b in bids[:10]])
        
        total = bid_volume + ask_volume
        if total > 0:
            imbalance = (bid_volume - ask_volume) / total
            return abs(imbalance)
        
        return 0.0
    
    def _calculate_approach_pattern(self, level: float, trades: List[Dict[str, Any]], current_price: float) -> float:
        """Calculate pattern of price approaching a level"""
        if len(trades) < 10:
            return 0.0
        
        recent_prices = [t['price'] for t in trades[-10:]]
        distances = [abs(p - level) / level for p in recent_prices]
        
        # Check if getting closer
        if distances[-1] < distances[0]:
            approach_rate = (distances[0] - distances[-1]) / distances[0]
            return min(1.0, approach_rate)
        
        return 0.0
    
    def _calculate_volume_spike(self, level: float, trades: List[Dict[str, Any]]) -> float:
        """Calculate volume spike near a level"""
        if len(trades) < 20:
            return 0.0
        
        near_level_volume = 0
        total_volume = 0
        
        for trade in trades[-20:]:
            total_volume += trade['volume']
            if abs(trade['price'] - level) / level < 0.005:  # Within 0.5%
                near_level_volume += trade['volume']
        
        if total_volume > 0:
            spike_ratio = near_level_volume / (total_volume / 20)
            return min(1.0, spike_ratio / 3)  # Normalize
        
        return 0.0
    
    def _estimate_stop_volume(self, level: float, order_book: Dict[str, Any]) -> float:
        """Estimate volume of stops at a level"""
        # This is a heuristic based on order book depth
        bids = np.array(order_book.get('bids', []))
        asks = np.array(order_book.get('asks', []))
        
        total_depth = 0
        if len(bids) > 0:
            total_depth += sum([b[1] for b in bids[:20]])
        if len(asks) > 0:
            total_depth += sum([a[1] for a in asks[:20]])
        
        # Estimate stops as percentage of visible liquidity
        return total_depth * 0.15  # Assume 15% hidden as stops
    
    def _calculate_ad_line(self, trades: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate Accumulation/Distribution line"""
        ad_values = []
        cumulative_ad = 0
        
        for i, trade in enumerate(trades):
            if i == 0:
                continue
            
            # Money Flow Multiplier
            high = max(trade['price'], trades[i-1]['price'])
            low = min(trade['price'], trades[i-1]['price'])
            close = trade['price']
            
            if high != low:
                mfm = ((close - low) - (high - close)) / (high - low)
            else:
                mfm = 0
            
            # Money Flow Volume
            mfv = mfm * trade['volume']
            
            # Accumulation/Distribution
            cumulative_ad += mfv
            ad_values.append(cumulative_ad)
        
        return np.array(ad_values)
    
    def _calculate_obv(self, trades: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate On-Balance Volume"""
        obv_values = []
        cumulative_obv = 0
        
        for i, trade in enumerate(trades):
            if i == 0:
                cumulative_obv = trade['volume']
            else:
                if trade['price'] > trades[i-1]['price']:
                    cumulative_obv += trade['volume']
                elif trade['price'] < trades[i-1]['price']:
                    cumulative_obv -= trade['volume']
                # If price unchanged, OBV stays the same
            
            obv_values.append(cumulative_obv)
        
        return np.array(obv_values)
    
    def get_liquidity_metrics(self) -> Dict[str, Any]:
        """Get current liquidity metrics"""
        metrics = {
            'microstructure': None,
            'liquidity_score': 0.0,
            'tracked_pools': len(self.liquidity_pools),
            'recent_signals': len(self.signal_history),
            'institutional_levels': []
        }
        
        if self.current_microstructure:
            metrics['microstructure'] = {
                'bid_ask_spread': self.current_microstructure.bid_ask_spread,
                'effective_spread': self.current_microstructure.effective_spread,
                'kyle_lambda': self.current_microstructure.kyle_lambda,
                'order_flow_imbalance': self.current_microstructure.order_flow_imbalance
            }
            metrics['liquidity_score'] = self.current_microstructure.get_liquidity_score()
        
        # Add institutional levels
        for level, info in self.liquidity_pools.items():
            metrics['institutional_levels'].append({
                'price': level,
                'side': info['side'],
                'age_minutes': (datetime.now() - info['first_seen']).total_seconds() / 60,
                'avg_volume': np.mean(info['volume_history']) if info['volume_history'] else 0
            })
        
        return metrics
    
    def get_rl_features(self) -> np.ndarray:
        """Get feature vector for RL agent integration"""
        features = []
        
        # Microstructure features
        if self.current_microstructure:
            features.extend([
                self.current_microstructure.bid_ask_spread,
                self.current_microstructure.effective_spread,
                self.current_microstructure.realized_spread,
                self.current_microstructure.kyle_lambda,
                self.current_microstructure.order_flow_imbalance,
                self.current_microstructure.price_impact,
                self.current_microstructure.adverse_selection,
                self.current_microstructure.inventory_risk,
                self.current_microstructure.get_liquidity_score()
            ])
        else:
            features.extend([0.0] * 9)
        
        # Signal features (last 5 signals)
        signal_features = []
        for signal in list(self.signal_history)[-5:]:
            signal_features.extend([
                1.0 if signal.signal_type == 'iceberg' else 0.0,
                1.0 if signal.signal_type == 'stop_hunt' else 0.0,
                1.0 if signal.signal_type == 'accumulation' else 0.0,
                1.0 if signal.signal_type == 'distribution' else 0.0,
                1.0 if signal.signal_type == 'squeeze' else 0.0,
                signal.strength,
                1.0 if signal.direction == 'buy' else -1.0 if signal.direction == 'sell' else 0.0
            ])
        
        # Pad if fewer than 5 signals
        while len(signal_features) < 35:  # 7 features per signal * 5 signals
            signal_features.append(0.0)
        
        features.extend(signal_features)
        
        # Liquidity pool features
        features.append(len(self.liquidity_pools))
        features.append(len([p for p in self.liquidity_pools.values() if p['side'] == 'buy']))
        features.append(len([p for p in self.liquidity_pools.values() if p['side'] == 'sell']))
        
        return np.array(features)


if __name__ == "__main__":
    # Example usage
    strategy = LiquidityHuntingStrategy(
        symbol="BTC/USDT",
        lookback_periods=100,
        sensitivity=0.7
    )
    
    # Mock data for testing
    mock_order_book = {
        'timestamp': datetime.now().timestamp(),
        'bids': [[50000, 1.5], [49999, 2.0], [49998, 1.8], [49997, 3.0], [49996, 2.5]],
        'asks': [[50001, 1.2], [50002, 1.8], [50003, 2.2], [50004, 1.5], [50005, 2.0]]
    }
    
    mock_trades = [
        {'price': 50000, 'volume': 0.5, 'side': 'buy', 'timestamp': datetime.now().timestamp()},
        {'price': 50001, 'volume': 0.3, 'side': 'sell', 'timestamp': datetime.now().timestamp()}
    ]
    
    mock_market_data = {
        'price': 50000,
        'high_24h': 51000,
        'low_24h': 49000,
        'sma_50': 49500,
        'sma_200': 48000
    }
    
    # Generate signals
    signals = strategy.update(mock_order_book, mock_trades, mock_market_data)
    
    print(f"Generated {len(signals)} signals")
    for signal in signals:
        print(f"Signal: {signal.signal_type} - {signal.direction} - Strength: {signal.strength:.2f}")
    
    # Get metrics
    metrics = strategy.get_liquidity_metrics()
    print(f"\nLiquidity Score: {metrics['liquidity_score']:.3f}")
    print(f"Tracked Pools: {metrics['tracked_pools']}")
    
    # Get RL features
    rl_features = strategy.get_rl_features()
    print(f"\nRL Feature Vector Shape: {rl_features.shape}")