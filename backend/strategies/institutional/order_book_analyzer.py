"""
Advanced Order Book Depth Analysis
Real-time analysis of order book dynamics, imbalances, and microstructure patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
from scipy import stats, optimize, interpolate
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class OrderBookSnapshot:
    """Point-in-time order book state"""
    timestamp: datetime
    symbol: str
    bids: np.ndarray  # [[price, volume], ...]
    asks: np.ndarray  # [[price, volume], ...]
    mid_price: float
    spread: float
    depth_10: float  # Total volume within 10 bps
    depth_50: float  # Total volume within 50 bps
    depth_100: float  # Total volume within 100 bps
    
    def get_weighted_mid_price(self) -> float:
        """Calculate volume-weighted mid price"""
        if len(self.bids) == 0 or len(self.asks) == 0:
            return self.mid_price
        
        bid_weight = self.bids[0, 1]
        ask_weight = self.asks[0, 1]
        total_weight = bid_weight + ask_weight
        
        if total_weight > 0:
            return (self.bids[0, 0] * ask_weight + self.asks[0, 0] * bid_weight) / total_weight
        return self.mid_price


@dataclass
class OrderBookImbalance:
    """Order book imbalance metrics"""
    volume_imbalance: float  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    value_imbalance: float  # Dollar-weighted imbalance
    depth_imbalance: Dict[int, float]  # Imbalance at different depths (bps)
    pressure_score: float  # Overall buying/selling pressure (-1 to 1)
    imbalance_derivative: float  # Rate of change of imbalance
    
    def get_signal_strength(self) -> float:
        """Get trading signal strength from imbalance"""
        return abs(self.pressure_score)
    
    def get_direction(self) -> str:
        """Get signal direction"""
        if self.pressure_score > 0.3:
            return 'buy'
        elif self.pressure_score < -0.3:
            return 'sell'
        return 'neutral'


@dataclass
class LiquidityProfile:
    """Order book liquidity profile"""
    bid_liquidity_curve: np.ndarray  # Cumulative bid volume at price levels
    ask_liquidity_curve: np.ndarray  # Cumulative ask volume at price levels
    resistance_levels: List[float]  # Price levels with high ask liquidity
    support_levels: List[float]  # Price levels with high bid liquidity
    liquidity_holes: List[Tuple[float, float]]  # Price ranges with low liquidity
    average_fill_cost: Dict[str, float]  # Expected cost for different order sizes


class OrderBookAnalyzer:
    """
    Advanced order book analysis for detecting liquidity patterns,
    market microstructure, and institutional activity
    """
    
    def __init__(self,
                 symbol: str,
                 depth_levels: int = 50,
                 history_size: int = 100,
                 update_frequency: int = 1):
        """
        Initialize order book analyzer
        
        Args:
            symbol: Trading symbol
            depth_levels: Number of order book levels to analyze
            history_size: Size of historical snapshot buffer
            update_frequency: Minimum seconds between updates
        """
        self.symbol = symbol
        self.depth_levels = depth_levels
        self.history_size = history_size
        self.update_frequency = update_frequency
        
        # Historical data storage
        self.snapshots = deque(maxlen=history_size)
        self.imbalance_history = deque(maxlen=history_size)
        self.liquidity_profiles = deque(maxlen=20)
        
        # Analysis components
        self.pca = PCA(n_components=5)
        self.cluster_model = DBSCAN(eps=0.3, min_samples=5)
        
        # State tracking
        self.last_update = datetime.now()
        self.current_snapshot = None
        self.current_imbalance = None
        self.current_profile = None
        
        # Calibration parameters
        self.spread_ema = None
        self.volume_ema = None
        self.imbalance_ema = None
        
        logger.info(f"OrderBookAnalyzer initialized for {symbol}")
    
    def update(self, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update analyzer with new order book data
        
        Args:
            order_book: Order book data with bids, asks, timestamp
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Rate limiting
            now = datetime.now()
            if (now - self.last_update).total_seconds() < self.update_frequency:
                return self._get_current_analysis()
            
            self.last_update = now
            
            # Create snapshot
            snapshot = self._create_snapshot(order_book)
            self.current_snapshot = snapshot
            self.snapshots.append(snapshot)
            
            # Analyze imbalance
            imbalance = self._analyze_imbalance(snapshot)
            self.current_imbalance = imbalance
            self.imbalance_history.append(imbalance)
            
            # Build liquidity profile
            profile = self._build_liquidity_profile(snapshot)
            self.current_profile = profile
            self.liquidity_profiles.append(profile)
            
            # Update calibration
            self._update_calibration(snapshot, imbalance)
            
            # Generate comprehensive analysis
            analysis = {
                'timestamp': snapshot.timestamp,
                'snapshot': snapshot,
                'imbalance': imbalance,
                'profile': profile,
                'microstructure': self._analyze_microstructure(snapshot),
                'flow_toxicity': self._calculate_flow_toxicity(),
                'price_levels': self._identify_key_levels(snapshot, profile),
                'execution_quality': self._estimate_execution_quality(snapshot),
                'hidden_liquidity': self._detect_hidden_liquidity(snapshot),
                'market_impact': self._estimate_market_impact(snapshot, profile)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error updating order book analyzer: {e}")
            return self._get_current_analysis()
    
    def _create_snapshot(self, order_book: Dict[str, Any]) -> OrderBookSnapshot:
        """Create order book snapshot"""
        try:
            bids = np.array(order_book.get('bids', []))[:self.depth_levels]
            asks = np.array(order_book.get('asks', []))[:self.depth_levels]
            
            if len(bids) == 0 or len(asks) == 0:
                raise ValueError("Empty order book")
            
            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            mid_price = (best_bid + best_ask) / 2
            spread = (best_ask - best_bid) / mid_price
            
            # Calculate depth at different levels
            depth_10 = self._calculate_depth_at_bps(bids, asks, mid_price, 10)
            depth_50 = self._calculate_depth_at_bps(bids, asks, mid_price, 50)
            depth_100 = self._calculate_depth_at_bps(bids, asks, mid_price, 100)
            
            return OrderBookSnapshot(
                timestamp=datetime.fromtimestamp(order_book.get('timestamp', datetime.now().timestamp())),
                symbol=self.symbol,
                bids=bids,
                asks=asks,
                mid_price=mid_price,
                spread=spread,
                depth_10=depth_10,
                depth_50=depth_50,
                depth_100=depth_100
            )
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return None
    
    def _analyze_imbalance(self, snapshot: OrderBookSnapshot) -> OrderBookImbalance:
        """Analyze order book imbalance"""
        try:
            # Volume imbalance at different depths
            depth_imbalances = {}
            for bps in [10, 25, 50, 100, 200]:
                bid_vol, ask_vol = self._get_volume_at_bps(
                    snapshot.bids, snapshot.asks, snapshot.mid_price, bps
                )
                
                total_vol = bid_vol + ask_vol
                if total_vol > 0:
                    depth_imbalances[bps] = (bid_vol - ask_vol) / total_vol
                else:
                    depth_imbalances[bps] = 0.0
            
            # Overall volume imbalance
            total_bid_vol = np.sum(snapshot.bids[:, 1])
            total_ask_vol = np.sum(snapshot.asks[:, 1])
            total_vol = total_bid_vol + total_ask_vol
            
            if total_vol > 0:
                volume_imbalance = (total_bid_vol - total_ask_vol) / total_vol
            else:
                volume_imbalance = 0.0
            
            # Value (dollar-weighted) imbalance
            bid_value = np.sum(snapshot.bids[:, 0] * snapshot.bids[:, 1])
            ask_value = np.sum(snapshot.asks[:, 0] * snapshot.asks[:, 1])
            total_value = bid_value + ask_value
            
            if total_value > 0:
                value_imbalance = (bid_value - ask_value) / total_value
            else:
                value_imbalance = 0.0
            
            # Pressure score (weighted average of imbalances)
            weights = [0.3, 0.2, 0.2, 0.15, 0.1, 0.05]  # Weights for different depths
            imbalance_values = [
                depth_imbalances.get(10, 0),
                depth_imbalances.get(25, 0),
                depth_imbalances.get(50, 0),
                depth_imbalances.get(100, 0),
                depth_imbalances.get(200, 0),
                volume_imbalance
            ]
            pressure_score = np.average(imbalance_values, weights=weights)
            
            # Imbalance derivative (rate of change)
            imbalance_derivative = 0.0
            if len(self.imbalance_history) > 1:
                prev_pressure = self.imbalance_history[-1].pressure_score
                time_diff = (snapshot.timestamp - self.snapshots[-2].timestamp).total_seconds()
                if time_diff > 0:
                    imbalance_derivative = (pressure_score - prev_pressure) / time_diff
            
            return OrderBookImbalance(
                volume_imbalance=volume_imbalance,
                value_imbalance=value_imbalance,
                depth_imbalance=depth_imbalances,
                pressure_score=pressure_score,
                imbalance_derivative=imbalance_derivative
            )
            
        except Exception as e:
            logger.error(f"Error analyzing imbalance: {e}")
            return OrderBookImbalance(0, 0, {}, 0, 0)
    
    def _build_liquidity_profile(self, snapshot: OrderBookSnapshot) -> LiquidityProfile:
        """Build liquidity profile from order book"""
        try:
            # Build cumulative liquidity curves
            bid_prices = snapshot.bids[:, 0]
            bid_volumes = snapshot.bids[:, 1]
            bid_cumulative = np.cumsum(bid_volumes)
            
            ask_prices = snapshot.asks[:, 0]
            ask_volumes = snapshot.asks[:, 1]
            ask_cumulative = np.cumsum(ask_volumes)
            
            # Identify support and resistance levels
            support_levels = self._find_liquidity_clusters(bid_prices, bid_volumes, 'support')
            resistance_levels = self._find_liquidity_clusters(ask_prices, ask_volumes, 'resistance')
            
            # Find liquidity holes
            liquidity_holes = self._find_liquidity_holes(snapshot)
            
            # Calculate average fill costs for different order sizes
            fill_costs = {}
            for size_mult in [0.1, 0.25, 0.5, 1.0, 2.0]:
                size = snapshot.depth_10 * size_mult
                fill_costs[f'{size_mult}x'] = {
                    'buy': self._calculate_fill_cost(snapshot.asks, size),
                    'sell': self._calculate_fill_cost(snapshot.bids, size)
                }
            
            return LiquidityProfile(
                bid_liquidity_curve=bid_cumulative,
                ask_liquidity_curve=ask_cumulative,
                resistance_levels=resistance_levels,
                support_levels=support_levels,
                liquidity_holes=liquidity_holes,
                average_fill_cost=fill_costs
            )
            
        except Exception as e:
            logger.error(f"Error building liquidity profile: {e}")
            return LiquidityProfile(
                np.array([]), np.array([]), [], [], [], {}
            )
    
    def _analyze_microstructure(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Analyze market microstructure metrics"""
        try:
            metrics = {}
            
            # Spread metrics
            metrics['quoted_spread'] = snapshot.spread
            metrics['weighted_spread'] = self._calculate_weighted_spread(snapshot)
            
            # Depth metrics
            metrics['depth_ratio'] = snapshot.depth_10 / (snapshot.depth_50 + 0.001)
            metrics['depth_skew'] = self._calculate_depth_skew(snapshot)
            
            # Shape metrics
            metrics['book_shape'] = self._calculate_book_shape(snapshot)
            metrics['convexity'] = self._calculate_convexity(snapshot)
            
            # Resilience metrics
            if len(self.snapshots) > 10:
                metrics['resilience'] = self._calculate_resilience()
                metrics['mean_reversion'] = self._calculate_mean_reversion()
            else:
                metrics['resilience'] = 0.5
                metrics['mean_reversion'] = 0.5
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing microstructure: {e}")
            return {}
    
    def _calculate_flow_toxicity(self) -> float:
        """
        Calculate VPIN (Volume-Synchronized Probability of Informed Trading)
        Measures the toxicity of order flow
        """
        try:
            if len(self.snapshots) < 20:
                return 0.5  # Neutral
            
            # Classify volume buckets
            buy_volumes = []
            sell_volumes = []
            
            for i in range(1, len(self.snapshots)):
                prev_mid = self.snapshots[i-1].mid_price
                curr_mid = self.snapshots[i].mid_price
                
                # Estimate executed volume from depth changes
                prev_bid_vol = np.sum(self.snapshots[i-1].bids[:10, 1])
                curr_bid_vol = np.sum(self.snapshots[i].bids[:10, 1])
                prev_ask_vol = np.sum(self.snapshots[i-1].asks[:10, 1])
                curr_ask_vol = np.sum(self.snapshots[i].asks[:10, 1])
                
                if curr_mid > prev_mid:
                    # Price up - likely buy volume
                    buy_vol = max(0, prev_ask_vol - curr_ask_vol)
                    sell_vol = 0
                else:
                    # Price down - likely sell volume
                    buy_vol = 0
                    sell_vol = max(0, prev_bid_vol - curr_bid_vol)
                
                buy_volumes.append(buy_vol)
                sell_volumes.append(sell_vol)
            
            # Calculate VPIN
            total_buy = sum(buy_volumes)
            total_sell = sum(sell_volumes)
            total_volume = total_buy + total_sell
            
            if total_volume > 0:
                vpin = abs(total_buy - total_sell) / total_volume
                return vpin
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating flow toxicity: {e}")
            return 0.5
    
    def _identify_key_levels(self, 
                            snapshot: OrderBookSnapshot,
                            profile: LiquidityProfile) -> Dict[str, List[float]]:
        """Identify key price levels from order book"""
        try:
            levels = {
                'support': profile.support_levels[:3],  # Top 3 support levels
                'resistance': profile.resistance_levels[:3],  # Top 3 resistance levels
                'liquidity_voids': [],
                'iceberg_candidates': []
            }
            
            # Identify liquidity voids
            for hole_start, hole_end in profile.liquidity_holes[:3]:
                levels['liquidity_voids'].append((hole_start + hole_end) / 2)
            
            # Identify potential iceberg order levels
            iceberg_levels = self._detect_iceberg_levels(snapshot)
            levels['iceberg_candidates'] = iceberg_levels[:3]
            
            return levels
            
        except Exception as e:
            logger.error(f"Error identifying key levels: {e}")
            return {'support': [], 'resistance': [], 'liquidity_voids': [], 'iceberg_candidates': []}
    
    def _estimate_execution_quality(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Estimate execution quality metrics"""
        try:
            quality = {}
            
            # Spread cost
            quality['spread_cost'] = snapshot.spread / 2  # Half-spread cost
            
            # Market impact for different sizes
            for size_pct in [10, 25, 50, 100]:
                size = snapshot.depth_10 * (size_pct / 10)
                
                buy_impact = self._calculate_price_impact(snapshot.asks, size, snapshot.mid_price)
                sell_impact = self._calculate_price_impact(snapshot.bids, size, snapshot.mid_price)
                
                quality[f'impact_{size_pct}pct'] = (buy_impact + sell_impact) / 2
            
            # Slippage estimate
            if len(self.snapshots) > 5:
                quality['expected_slippage'] = self._estimate_slippage()
            else:
                quality['expected_slippage'] = snapshot.spread
            
            return quality
            
        except Exception as e:
            logger.error(f"Error estimating execution quality: {e}")
            return {}
    
    def _detect_hidden_liquidity(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Detect hidden liquidity patterns"""
        try:
            hidden = {
                'iceberg_probability': 0.0,
                'dark_pool_indicator': 0.0,
                'reserve_orders': [],
                'pegged_orders': []
            }
            
            # Iceberg detection
            iceberg_score = self._calculate_iceberg_probability(snapshot)
            hidden['iceberg_probability'] = iceberg_score
            
            # Dark pool activity indicator
            if len(self.snapshots) > 10:
                dark_pool_score = self._calculate_dark_pool_activity()
                hidden['dark_pool_indicator'] = dark_pool_score
            
            # Reserve order detection
            reserve_levels = self._detect_reserve_orders(snapshot)
            hidden['reserve_orders'] = reserve_levels
            
            # Pegged order detection
            pegged_levels = self._detect_pegged_orders(snapshot)
            hidden['pegged_orders'] = pegged_levels
            
            return hidden
            
        except Exception as e:
            logger.error(f"Error detecting hidden liquidity: {e}")
            return {'iceberg_probability': 0, 'dark_pool_indicator': 0, 'reserve_orders': [], 'pegged_orders': []}
    
    def _estimate_market_impact(self, 
                                snapshot: OrderBookSnapshot,
                                profile: LiquidityProfile) -> Dict[str, float]:
        """Estimate market impact for different order sizes"""
        try:
            impact = {}
            
            # Linear impact model coefficients
            lambda_buy = self._calculate_kyle_lambda(snapshot.asks, 'buy')
            lambda_sell = self._calculate_kyle_lambda(snapshot.bids, 'sell')
            
            impact['kyle_lambda_buy'] = lambda_buy
            impact['kyle_lambda_sell'] = lambda_sell
            
            # Square-root impact model (Alameda/Barra model)
            for size_mult in [0.5, 1.0, 2.0, 5.0]:
                size = snapshot.depth_10 * size_mult
                
                # Impact = spread_cost + lambda * sqrt(size/ADV)
                adv = snapshot.depth_100  # Use depth as proxy for average daily volume
                
                buy_impact = snapshot.spread/2 + lambda_buy * np.sqrt(size / (adv + 1))
                sell_impact = snapshot.spread/2 + lambda_sell * np.sqrt(size / (adv + 1))
                
                impact[f'buy_{size_mult}x'] = buy_impact
                impact[f'sell_{size_mult}x'] = sell_impact
            
            # Permanent vs temporary impact
            if len(self.snapshots) > 10:
                impact['permanent_ratio'] = self._calculate_permanent_impact_ratio()
            else:
                impact['permanent_ratio'] = 0.6  # Default assumption
            
            return impact
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return {}
    
    # Helper methods
    
    def _calculate_depth_at_bps(self, bids: np.ndarray, asks: np.ndarray, 
                                mid_price: float, bps: int) -> float:
        """Calculate total depth within specified basis points"""
        try:
            threshold = mid_price * (bps / 10000)
            
            bid_depth = 0
            for price, volume in bids:
                if mid_price - price <= threshold:
                    bid_depth += volume
                else:
                    break
            
            ask_depth = 0
            for price, volume in asks:
                if price - mid_price <= threshold:
                    ask_depth += volume
                else:
                    break
            
            return bid_depth + ask_depth
            
        except Exception as e:
            logger.error(f"Error calculating depth at {bps} bps: {e}")
            return 0.0
    
    def _get_volume_at_bps(self, bids: np.ndarray, asks: np.ndarray,
                           mid_price: float, bps: int) -> Tuple[float, float]:
        """Get bid and ask volume within specified basis points"""
        try:
            threshold = mid_price * (bps / 10000)
            
            bid_volume = 0
            for price, volume in bids:
                if mid_price - price <= threshold:
                    bid_volume += volume
                else:
                    break
            
            ask_volume = 0
            for price, volume in asks:
                if price - mid_price <= threshold:
                    ask_volume += volume
                else:
                    break
            
            return bid_volume, ask_volume
            
        except Exception as e:
            logger.error(f"Error getting volume at {bps} bps: {e}")
            return 0.0, 0.0
    
    def _find_liquidity_clusters(self, prices: np.ndarray, volumes: np.ndarray,
                                 cluster_type: str) -> List[float]:
        """Find price levels with liquidity clusters"""
        try:
            if len(prices) < 5:
                return []
            
            # Use DBSCAN to find volume clusters
            X = np.column_stack([prices, volumes])
            
            # Normalize for clustering
            X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(X_norm)
            
            clusters = {}
            for i, label in enumerate(clustering.labels_):
                if label != -1:  # Not noise
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append((prices[i], volumes[i]))
            
            # Find cluster centers weighted by volume
            cluster_levels = []
            for cluster_points in clusters.values():
                prices_c = [p for p, v in cluster_points]
                volumes_c = [v for p, v in cluster_points]
                
                # Volume-weighted average price
                weighted_price = np.average(prices_c, weights=volumes_c)
                total_volume = sum(volumes_c)
                
                cluster_levels.append((weighted_price, total_volume))
            
            # Sort by volume and return price levels
            cluster_levels.sort(key=lambda x: x[1], reverse=True)
            
            return [price for price, volume in cluster_levels]
            
        except Exception as e:
            logger.error(f"Error finding liquidity clusters: {e}")
            return []
    
    def _find_liquidity_holes(self, snapshot: OrderBookSnapshot) -> List[Tuple[float, float]]:
        """Find price ranges with low liquidity"""
        try:
            holes = []
            
            # Check bid side
            for i in range(len(snapshot.bids) - 1):
                price_gap = snapshot.bids[i, 0] - snapshot.bids[i+1, 0]
                relative_gap = price_gap / snapshot.mid_price
                
                if relative_gap > 0.001:  # More than 10 bps gap
                    holes.append((snapshot.bids[i+1, 0], snapshot.bids[i, 0]))
            
            # Check ask side
            for i in range(len(snapshot.asks) - 1):
                price_gap = snapshot.asks[i+1, 0] - snapshot.asks[i, 0]
                relative_gap = price_gap / snapshot.mid_price
                
                if relative_gap > 0.001:  # More than 10 bps gap
                    holes.append((snapshot.asks[i, 0], snapshot.asks[i+1, 0]))
            
            return holes[:5]  # Return top 5 holes
            
        except Exception as e:
            logger.error(f"Error finding liquidity holes: {e}")
            return []
    
    def _calculate_fill_cost(self, orders: np.ndarray, size: float) -> float:
        """Calculate average fill cost for a given size"""
        try:
            if len(orders) == 0 or size <= 0:
                return 0.0
            
            remaining_size = size
            total_cost = 0.0
            
            for price, volume in orders:
                if remaining_size <= 0:
                    break
                
                fill_size = min(remaining_size, volume)
                total_cost += fill_size * price
                remaining_size -= fill_size
            
            if size - remaining_size > 0:
                avg_price = total_cost / (size - remaining_size)
                # Return as percentage from best price
                return abs(avg_price - orders[0, 0]) / orders[0, 0]
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating fill cost: {e}")
            return 0.0
    
    def _calculate_weighted_spread(self, snapshot: OrderBookSnapshot) -> float:
        """Calculate volume-weighted spread"""
        try:
            # Weight by depth at each level
            weights = []
            spreads = []
            
            for i in range(min(10, len(snapshot.bids), len(snapshot.asks))):
                bid_price = snapshot.bids[i, 0]
                bid_vol = snapshot.bids[i, 1]
                ask_price = snapshot.asks[i, 0]
                ask_vol = snapshot.asks[i, 1]
                
                level_spread = (ask_price - bid_price) / snapshot.mid_price
                level_weight = bid_vol + ask_vol
                
                spreads.append(level_spread)
                weights.append(level_weight)
            
            if sum(weights) > 0:
                return np.average(spreads, weights=weights)
            
            return snapshot.spread
            
        except Exception as e:
            logger.error(f"Error calculating weighted spread: {e}")
            return snapshot.spread
    
    def _calculate_depth_skew(self, snapshot: OrderBookSnapshot) -> float:
        """Calculate skewness of order book depth"""
        try:
            bid_depth = np.sum(snapshot.bids[:10, 1])
            ask_depth = np.sum(snapshot.asks[:10, 1])
            
            total = bid_depth + ask_depth
            if total > 0:
                return (bid_depth - ask_depth) / total
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating depth skew: {e}")
            return 0.0
    
    def _calculate_book_shape(self, snapshot: OrderBookSnapshot) -> float:
        """Calculate order book shape metric (convexity)"""
        try:
            # Fit power law to depth decay
            bid_depths = snapshot.bids[:20, 1]
            ask_depths = snapshot.asks[:20, 1]
            
            x = np.arange(1, 21)
            
            # Log-log regression for power law
            bid_coef = np.polyfit(np.log(x), np.log(bid_depths + 1), 1)[0]
            ask_coef = np.polyfit(np.log(x), np.log(ask_depths + 1), 1)[0]
            
            # Average decay rate (more negative = steeper decay)
            shape = (bid_coef + ask_coef) / 2
            
            return shape
            
        except Exception as e:
            logger.error(f"Error calculating book shape: {e}")
            return 0.0
    
    def _calculate_convexity(self, snapshot: OrderBookSnapshot) -> float:
        """Calculate order book convexity"""
        try:
            # Second derivative of cumulative depth curve
            bid_cumsum = np.cumsum(snapshot.bids[:20, 1])
            ask_cumsum = np.cumsum(snapshot.asks[:20, 1])
            
            bid_convexity = np.diff(bid_cumsum, 2).mean()
            ask_convexity = np.diff(ask_cumsum, 2).mean()
            
            return (bid_convexity + ask_convexity) / 2
            
        except Exception as e:
            logger.error(f"Error calculating convexity: {e}")
            return 0.0
    
    def _calculate_resilience(self) -> float:
        """Calculate order book resilience (speed of recovery after trades)"""
        try:
            if len(self.snapshots) < 10:
                return 0.5
            
            # Measure how quickly depth recovers after depletion
            depth_changes = []
            recoveries = []
            
            for i in range(1, len(self.snapshots)):
                prev_depth = self.snapshots[i-1].depth_10
                curr_depth = self.snapshots[i].depth_10
                
                depth_change = (curr_depth - prev_depth) / (prev_depth + 1)
                depth_changes.append(depth_change)
                
                if depth_change < -0.1:  # Significant depletion
                    # Check recovery in next snapshots
                    recovery_rate = 0
                    for j in range(i+1, min(i+5, len(self.snapshots))):
                        future_depth = self.snapshots[j].depth_10
                        if future_depth >= prev_depth * 0.9:
                            recovery_rate = 1 / (j - i)
                            break
                    recoveries.append(recovery_rate)
            
            if recoveries:
                return min(1.0, np.mean(recoveries))
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating resilience: {e}")
            return 0.5
    
    def _calculate_mean_reversion(self) -> float:
        """Calculate mean reversion tendency of spreads"""
        try:
            if len(self.snapshots) < 20:
                return 0.5
            
            spreads = [s.spread for s in self.snapshots[-20:]]
            
            # Augmented Dickey-Fuller test approximation
            y = np.array(spreads)
            x = y[:-1]
            y_diff = np.diff(y)
            
            # Simple regression
            beta = np.cov(x, y_diff)[0, 1] / np.var(x)
            
            # More negative beta indicates stronger mean reversion
            mean_reversion = max(0, min(1, -beta * 10))
            
            return mean_reversion
            
        except Exception as e:
            logger.error(f"Error calculating mean reversion: {e}")
            return 0.5
    
    def _detect_iceberg_levels(self, snapshot: OrderBookSnapshot) -> List[float]:
        """Detect potential iceberg order levels"""
        try:
            iceberg_levels = []
            
            # Look for persistent volume at specific levels
            if len(self.snapshots) < 5:
                return []
            
            # Check bid side
            for i in range(min(10, len(snapshot.bids))):
                price = snapshot.bids[i, 0]
                volume = snapshot.bids[i, 1]
                
                # Check persistence
                persistence_count = 0
                volume_variance = []
                
                for hist_snap in self.snapshots[-5:]:
                    for hist_price, hist_vol in hist_snap.bids[:10]:
                        if abs(hist_price - price) / price < 0.0001:
                            persistence_count += 1
                            volume_variance.append(hist_vol)
                            break
                
                if persistence_count >= 4 and len(volume_variance) > 2:
                    # High persistence with volume variation suggests reloading
                    cv = np.std(volume_variance) / (np.mean(volume_variance) + 1e-8)
                    if cv > 0.3:  # Coefficient of variation > 30%
                        iceberg_levels.append(price)
            
            # Check ask side
            for i in range(min(10, len(snapshot.asks))):
                price = snapshot.asks[i, 0]
                volume = snapshot.asks[i, 1]
                
                persistence_count = 0
                volume_variance = []
                
                for hist_snap in self.snapshots[-5:]:
                    for hist_price, hist_vol in hist_snap.asks[:10]:
                        if abs(hist_price - price) / price < 0.0001:
                            persistence_count += 1
                            volume_variance.append(hist_vol)
                            break
                
                if persistence_count >= 4 and len(volume_variance) > 2:
                    cv = np.std(volume_variance) / (np.mean(volume_variance) + 1e-8)
                    if cv > 0.3:
                        iceberg_levels.append(price)
            
            return iceberg_levels
            
        except Exception as e:
            logger.error(f"Error detecting iceberg levels: {e}")
            return []
    
    def _calculate_iceberg_probability(self, snapshot: OrderBookSnapshot) -> float:
        """Calculate probability of iceberg orders in the book"""
        try:
            if len(self.snapshots) < 10:
                return 0.0
            
            # Features for iceberg detection
            features = []
            
            # 1. Volume reload patterns
            reload_score = 0
            for i in range(min(5, len(snapshot.bids))):
                price = snapshot.bids[i, 0]
                reload_count = 0
                
                for j in range(1, min(10, len(self.snapshots))):
                    prev_snap = self.snapshots[-j]
                    for prev_price, prev_vol in prev_snap.bids[:10]:
                        if abs(prev_price - price) / price < 0.0001:
                            if prev_vol > 0:
                                reload_count += 1
                            break
                
                if reload_count >= 7:
                    reload_score += 1
            
            features.append(reload_score / 5)
            
            # 2. Abnormal volume concentration
            bid_volumes = snapshot.bids[:20, 1]
            volume_concentration = np.max(bid_volumes) / (np.mean(bid_volumes) + 1e-8)
            features.append(min(1, volume_concentration / 5))
            
            # 3. Price level persistence
            persistence_score = len(self._detect_iceberg_levels(snapshot)) / 10
            features.append(min(1, persistence_score))
            
            # Combine features
            iceberg_prob = np.mean(features)
            
            return iceberg_prob
            
        except Exception as e:
            logger.error(f"Error calculating iceberg probability: {e}")
            return 0.0
    
    def _calculate_dark_pool_activity(self) -> float:
        """Estimate dark pool activity from lit market behavior"""
        try:
            # Indicators of dark pool activity:
            # 1. Price moves without visible volume
            # 2. Sudden liquidity appearance/disappearance
            # 3. Execution quality deterioration
            
            indicators = []
            
            # Check for price moves without volume
            for i in range(1, len(self.snapshots)):
                price_change = abs(self.snapshots[i].mid_price - self.snapshots[i-1].mid_price)
                price_change_pct = price_change / self.snapshots[i-1].mid_price
                
                volume_change = abs(self.snapshots[i].depth_10 - self.snapshots[i-1].depth_10)
                
                if price_change_pct > 0.001 and volume_change < self.snapshots[i-1].depth_10 * 0.1:
                    indicators.append(1)
                else:
                    indicators.append(0)
            
            if indicators:
                return np.mean(indicators)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating dark pool activity: {e}")
            return 0.0
    
    def _detect_reserve_orders(self, snapshot: OrderBookSnapshot) -> List[float]:
        """Detect reserve/hidden orders"""
        try:
            reserve_levels = []
            
            # Look for levels where execution exceeds visible liquidity
            # This requires trade data which we'll approximate
            
            # For now, identify levels with suspicious volume patterns
            for i in range(min(10, len(snapshot.bids))):
                price = snapshot.bids[i, 0]
                volume = snapshot.bids[i, 1]
                
                # Check if volume is suspiciously round
                if volume == round(volume, -1):  # Round to nearest 10
                    reserve_levels.append(price)
            
            return reserve_levels[:3]
            
        except Exception as e:
            logger.error(f"Error detecting reserve orders: {e}")
            return []
    
    def _detect_pegged_orders(self, snapshot: OrderBookSnapshot) -> List[float]:
        """Detect pegged orders that move with the market"""
        try:
            if len(self.snapshots) < 5:
                return []
            
            pegged_levels = []
            
            # Look for orders that maintain constant distance from mid
            for offset in [0.001, 0.002, 0.005, 0.01]:  # Common pegging offsets
                bid_peg_price = snapshot.mid_price * (1 - offset)
                ask_peg_price = snapshot.mid_price * (1 + offset)
                
                # Check if these levels persist across snapshots
                bid_persistence = 0
                ask_persistence = 0
                
                for hist_snap in self.snapshots[-5:]:
                    hist_bid_peg = hist_snap.mid_price * (1 - offset)
                    hist_ask_peg = hist_snap.mid_price * (1 + offset)
                    
                    # Check for orders near pegged levels
                    for price, vol in hist_snap.bids[:10]:
                        if abs(price - hist_bid_peg) / hist_bid_peg < 0.0001:
                            bid_persistence += 1
                            break
                    
                    for price, vol in hist_snap.asks[:10]:
                        if abs(price - hist_ask_peg) / hist_ask_peg < 0.0001:
                            ask_persistence += 1
                            break
                
                if bid_persistence >= 4:
                    pegged_levels.append(bid_peg_price)
                if ask_persistence >= 4:
                    pegged_levels.append(ask_peg_price)
            
            return pegged_levels
            
        except Exception as e:
            logger.error(f"Error detecting pegged orders: {e}")
            return []
    
    def _calculate_price_impact(self, orders: np.ndarray, size: float, mid_price: float) -> float:
        """Calculate price impact for a given order size"""
        try:
            if len(orders) == 0 or size <= 0:
                return 0.0
            
            remaining_size = size
            volume_weighted_price = 0.0
            filled_size = 0.0
            
            for price, volume in orders:
                if remaining_size <= 0:
                    break
                
                fill = min(remaining_size, volume)
                volume_weighted_price += price * fill
                filled_size += fill
                remaining_size -= fill
            
            if filled_size > 0:
                avg_price = volume_weighted_price / filled_size
                impact = abs(avg_price - mid_price) / mid_price
                return impact
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating price impact: {e}")
            return 0.0
    
    def _estimate_slippage(self) -> float:
        """Estimate expected slippage from historical data"""
        try:
            if len(self.snapshots) < 10:
                return 0.001  # Default 10 bps
            
            # Calculate historical spread volatility
            spreads = [s.spread for s in self.snapshots[-20:]]
            spread_vol = np.std(spreads)
            
            # Estimate slippage as function of spread and volatility
            avg_spread = np.mean(spreads)
            slippage = avg_spread + 2 * spread_vol  # Conservative estimate
            
            return slippage
            
        except Exception as e:
            logger.error(f"Error estimating slippage: {e}")
            return 0.001
    
    def _calculate_kyle_lambda(self, orders: np.ndarray, side: str) -> float:
        """Calculate Kyle's lambda (price impact coefficient)"""
        try:
            if len(orders) < 5:
                return 0.0
            
            # Estimate lambda from order book shape
            prices = orders[:10, 0]
            volumes = orders[:10, 1]
            cumulative_volume = np.cumsum(volumes)
            
            # Price impact = lambda * volume^gamma
            # For linear impact, gamma = 1
            # For square-root impact, gamma = 0.5
            
            # Calculate price distances from best
            best_price = prices[0]
            price_impacts = np.abs(prices - best_price) / best_price
            
            # Fit linear model: impact = lambda * volume
            valid_indices = cumulative_volume > 0
            if np.sum(valid_indices) > 2:
                lambda_est = np.polyfit(
                    cumulative_volume[valid_indices],
                    price_impacts[valid_indices],
                    1
                )[0]
                
                return abs(lambda_est)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Kyle's lambda: {e}")
            return 0.0
    
    def _calculate_permanent_impact_ratio(self) -> float:
        """Calculate ratio of permanent to total impact"""
        try:
            # Permanent impact persists, temporary impact reverts
            # Measure by analyzing price recovery after large trades
            
            if len(self.snapshots) < 20:
                return 0.6  # Default assumption
            
            # Look for significant price moves
            price_moves = []
            recoveries = []
            
            for i in range(10, len(self.snapshots) - 10):
                price_change = (self.snapshots[i].mid_price - self.snapshots[i-1].mid_price) / self.snapshots[i-1].mid_price
                
                if abs(price_change) > 0.002:  # Significant move (>20 bps)
                    # Measure recovery over next 10 snapshots
                    initial_price = self.snapshots[i-1].mid_price
                    impact_price = self.snapshots[i].mid_price
                    final_price = self.snapshots[i+10].mid_price
                    
                    total_impact = abs(impact_price - initial_price)
                    permanent_impact = abs(final_price - initial_price)
                    
                    if total_impact > 0:
                        ratio = permanent_impact / total_impact
                        recoveries.append(min(1.0, ratio))
            
            if recoveries:
                return np.mean(recoveries)
            
            return 0.6
            
        except Exception as e:
            logger.error(f"Error calculating permanent impact ratio: {e}")
            return 0.6
    
    def _update_calibration(self, snapshot: OrderBookSnapshot, imbalance: OrderBookImbalance):
        """Update calibration parameters with exponential moving averages"""
        try:
            alpha = 0.1  # EMA decay factor
            
            if self.spread_ema is None:
                self.spread_ema = snapshot.spread
            else:
                self.spread_ema = alpha * snapshot.spread + (1 - alpha) * self.spread_ema
            
            if self.volume_ema is None:
                self.volume_ema = snapshot.depth_10
            else:
                self.volume_ema = alpha * snapshot.depth_10 + (1 - alpha) * self.volume_ema
            
            if self.imbalance_ema is None:
                self.imbalance_ema = imbalance.pressure_score
            else:
                self.imbalance_ema = alpha * imbalance.pressure_score + (1 - alpha) * self.imbalance_ema
                
        except Exception as e:
            logger.error(f"Error updating calibration: {e}")
    
    def _get_current_analysis(self) -> Dict[str, Any]:
        """Get current analysis results"""
        if self.current_snapshot is None:
            return {}
        
        return {
            'timestamp': self.current_snapshot.timestamp,
            'snapshot': self.current_snapshot,
            'imbalance': self.current_imbalance,
            'profile': self.current_profile,
            'microstructure': self._analyze_microstructure(self.current_snapshot) if self.current_snapshot else {},
            'flow_toxicity': self._calculate_flow_toxicity(),
            'price_levels': {},
            'execution_quality': {},
            'hidden_liquidity': {},
            'market_impact': {}
        }
    
    def get_rl_features(self) -> np.ndarray:
        """Get feature vector for RL agent integration"""
        features = []
        
        if self.current_snapshot:
            # Basic metrics
            features.extend([
                self.current_snapshot.spread,
                self.current_snapshot.depth_10,
                self.current_snapshot.depth_50,
                self.current_snapshot.depth_100
            ])
        else:
            features.extend([0.0] * 4)
        
        if self.current_imbalance:
            # Imbalance metrics
            features.extend([
                self.current_imbalance.volume_imbalance,
                self.current_imbalance.value_imbalance,
                self.current_imbalance.pressure_score,
                self.current_imbalance.imbalance_derivative
            ])
        else:
            features.extend([0.0] * 4)
        
        # Flow toxicity
        features.append(self._calculate_flow_toxicity())
        
        # Microstructure
        if self.current_snapshot:
            micro = self._analyze_microstructure(self.current_snapshot)
            features.extend([
                micro.get('weighted_spread', 0),
                micro.get('depth_ratio', 0),
                micro.get('book_shape', 0),
                micro.get('resilience', 0.5)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.5])
        
        return np.array(features)


if __name__ == "__main__":
    # Example usage
    analyzer = OrderBookAnalyzer(
        symbol="BTC/USDT",
        depth_levels=50,
        history_size=100
    )
    
    # Mock order book data
    mock_order_book = {
        'timestamp': datetime.now().timestamp(),
        'bids': [[50000 - i*10, np.random.uniform(0.5, 2.0)] for i in range(50)],
        'asks': [[50001 + i*10, np.random.uniform(0.5, 2.0)] for i in range(50)]
    }
    
    # Analyze order book
    analysis = analyzer.update(mock_order_book)
    
    if analysis and 'imbalance' in analysis:
        print(f"Pressure Score: {analysis['imbalance'].pressure_score:.3f}")
        print(f"Signal Direction: {analysis['imbalance'].get_direction()}")
        print(f"Flow Toxicity: {analysis.get('flow_toxicity', 0):.3f}")
    
    # Get RL features
    features = analyzer.get_rl_features()
    print(f"\nRL Feature Vector Shape: {features.shape}")
    print(f"Feature Values: {features[:5]}")