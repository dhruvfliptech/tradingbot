"""
Advanced Iceberg Order Detection System
Identifies hidden orders, reserve orders, and algorithmic trading patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque, defaultdict
from scipy import stats, signal
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class IcebergOrder:
    """Detected iceberg order information"""
    price_level: float
    side: str  # 'bid' or 'ask'
    visible_size: float
    estimated_total_size: float
    confidence_score: float
    detection_time: datetime
    reload_count: int
    execution_pattern: str  # 'aggressive', 'passive', 'mixed'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def hidden_ratio(self) -> float:
        """Calculate ratio of hidden to total size"""
        if self.estimated_total_size > 0:
            return (self.estimated_total_size - self.visible_size) / self.estimated_total_size
        return 0.0
    
    @property
    def remaining_size(self) -> float:
        """Estimate remaining hidden size"""
        executed = self.metadata.get('executed_volume', 0)
        return max(0, self.estimated_total_size - executed)


@dataclass
class AlgorithmicPattern:
    """Detected algorithmic trading pattern"""
    pattern_type: str  # 'twap', 'vwap', 'pov', 'implementation_shortfall'
    start_time: datetime
    end_time: Optional[datetime]
    price_range: Tuple[float, float]
    volume_profile: List[float]
    participation_rate: float
    aggressiveness: float  # 0-1 scale
    confidence: float
    

class IcebergDetector:
    """
    Sophisticated iceberg order and hidden liquidity detection system
    using machine learning and pattern recognition
    """
    
    def __init__(self,
                 symbol: str,
                 detection_sensitivity: float = 0.7,
                 min_confidence: float = 0.6,
                 lookback_window: int = 100):
        """
        Initialize iceberg detector
        
        Args:
            symbol: Trading symbol
            detection_sensitivity: Sensitivity for detection (0-1)
            min_confidence: Minimum confidence for reporting
            lookback_window: Historical window for pattern analysis
        """
        self.symbol = symbol
        self.sensitivity = detection_sensitivity
        self.min_confidence = min_confidence
        self.lookback_window = lookback_window
        
        # Detection models
        self.ml_detector = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Historical tracking
        self.order_book_history = deque(maxlen=lookback_window)
        self.trade_history = deque(maxlen=lookback_window * 10)
        self.detected_icebergs: Dict[float, IcebergOrder] = {}
        self.algorithmic_patterns: List[AlgorithmicPattern] = []
        
        # Pattern tracking
        self.price_level_tracker: Dict[float, Dict] = defaultdict(lambda: {
            'appearances': 0,
            'volumes': [],
            'timestamps': [],
            'executions': 0,
            'reload_events': []
        })
        
        # Calibration
        self.is_calibrated = False
        self.feature_importance = {}
        
        logger.info(f"IcebergDetector initialized for {symbol}")
    
    def detect(self,
               order_book: Dict[str, Any],
               trades: List[Dict[str, Any]],
               market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect iceberg orders and hidden liquidity
        
        Args:
            order_book: Current order book
            trades: Recent trades
            market_data: Additional market data
            
        Returns:
            Detection results with identified icebergs and patterns
        """
        try:
            # Update history
            self.order_book_history.append(order_book)
            self.trade_history.extend(trades)
            
            # Track price levels
            self._update_price_level_tracking(order_book, trades)
            
            # Detect icebergs using multiple methods
            results = {
                'timestamp': datetime.fromtimestamp(order_book.get('timestamp', datetime.now().timestamp())),
                'icebergs': [],
                'algorithmic_patterns': [],
                'hidden_liquidity_estimate': 0.0,
                'detection_confidence': {}
            }
            
            # Method 1: Statistical detection
            statistical_icebergs = self._statistical_detection(order_book)
            results['icebergs'].extend(statistical_icebergs)
            
            # Method 2: Machine learning detection (if calibrated)
            if self.is_calibrated:
                ml_icebergs = self._ml_detection(order_book)
                results['icebergs'].extend(ml_icebergs)
            
            # Method 3: Pattern-based detection
            pattern_icebergs = self._pattern_detection(order_book, trades)
            results['icebergs'].extend(pattern_icebergs)
            
            # Deduplicate and merge detections
            results['icebergs'] = self._merge_detections(results['icebergs'])
            
            # Detect algorithmic patterns
            algo_patterns = self._detect_algorithmic_patterns(trades)
            results['algorithmic_patterns'] = algo_patterns
            
            # Estimate total hidden liquidity
            results['hidden_liquidity_estimate'] = self._estimate_hidden_liquidity(
                results['icebergs'], order_book
            )
            
            # Calculate detection confidence
            results['detection_confidence'] = self._calculate_confidence_scores(
                results['icebergs']
            )
            
            # Update tracked icebergs
            self._update_tracked_icebergs(results['icebergs'], trades)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in iceberg detection: {e}")
            return {
                'timestamp': datetime.now(),
                'icebergs': [],
                'algorithmic_patterns': [],
                'hidden_liquidity_estimate': 0.0,
                'detection_confidence': {}
            }
    
    def _statistical_detection(self, order_book: Dict[str, Any]) -> List[IcebergOrder]:
        """Statistical method for iceberg detection"""
        detected = []
        
        try:
            bids = np.array(order_book.get('bids', []))
            asks = np.array(order_book.get('asks', []))
            
            # Detect on bid side
            bid_icebergs = self._detect_statistical_anomalies(
                bids, 'bid', order_book['timestamp']
            )
            detected.extend(bid_icebergs)
            
            # Detect on ask side
            ask_icebergs = self._detect_statistical_anomalies(
                asks, 'ask', order_book['timestamp']
            )
            detected.extend(ask_icebergs)
            
            return detected
            
        except Exception as e:
            logger.error(f"Error in statistical detection: {e}")
            return []
    
    def _detect_statistical_anomalies(self,
                                     orders: np.ndarray,
                                     side: str,
                                     timestamp: float) -> List[IcebergOrder]:
        """Detect statistical anomalies in order book"""
        detected = []
        
        if len(orders) < 20:
            return detected
        
        try:
            prices = orders[:, 0]
            volumes = orders[:, 1]
            
            # 1. Volume distribution analysis
            log_volumes = np.log(volumes + 1)
            mean_log_vol = np.mean(log_volumes)
            std_log_vol = np.std(log_volumes)
            
            # Find outliers (potential icebergs)
            z_scores = (log_volumes - mean_log_vol) / (std_log_vol + 1e-8)
            
            for i, z_score in enumerate(z_scores[:10]):  # Check top 10 levels
                if z_score > 2.5 * self.sensitivity:  # Significant outlier
                    price = prices[i]
                    volume = volumes[i]
                    
                    # Check persistence
                    persistence_score = self._check_persistence(price, side)
                    
                    if persistence_score > 0.5:
                        # Check reload pattern
                        reload_count = self._count_reloads(price)
                        
                        confidence = min(1.0, (z_score / 4) * persistence_score)
                        
                        if confidence >= self.min_confidence:
                            iceberg = IcebergOrder(
                                price_level=price,
                                side=side,
                                visible_size=volume,
                                estimated_total_size=volume * (1 + z_score),
                                confidence_score=confidence,
                                detection_time=datetime.fromtimestamp(timestamp),
                                reload_count=reload_count,
                                execution_pattern='passive',
                                metadata={
                                    'detection_method': 'statistical',
                                    'z_score': z_score,
                                    'persistence_score': persistence_score
                                }
                            )
                            detected.append(iceberg)
            
            # 2. Shape anomaly detection
            shape_anomalies = self._detect_shape_anomalies(prices, volumes, side)
            for anomaly in shape_anomalies:
                if anomaly['confidence'] >= self.min_confidence:
                    iceberg = IcebergOrder(
                        price_level=anomaly['price'],
                        side=side,
                        visible_size=anomaly['volume'],
                        estimated_total_size=anomaly['estimated_total'],
                        confidence_score=anomaly['confidence'],
                        detection_time=datetime.fromtimestamp(timestamp),
                        reload_count=0,
                        execution_pattern='mixed',
                        metadata={
                            'detection_method': 'shape_anomaly',
                            'anomaly_type': anomaly['type']
                        }
                    )
                    detected.append(iceberg)
            
            return detected
            
        except Exception as e:
            logger.error(f"Error detecting statistical anomalies: {e}")
            return []
    
    def _ml_detection(self, order_book: Dict[str, Any]) -> List[IcebergOrder]:
        """Machine learning based iceberg detection"""
        detected = []
        
        try:
            # Extract features
            features = self._extract_ml_features(order_book)
            
            if features is None:
                return detected
            
            # Normalize features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict iceberg probability
            iceberg_prob = self.ml_detector.predict_proba(features_scaled)[0, 1]
            
            # Anomaly detection
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            
            # Combine scores
            combined_score = (iceberg_prob * 0.7 + (1 - 1/(1 + np.exp(-anomaly_score))) * 0.3)
            
            if combined_score >= self.min_confidence:
                # Identify specific levels
                bids = np.array(order_book.get('bids', []))
                asks = np.array(order_book.get('asks', []))
                
                # Find most suspicious levels
                suspicious_levels = self._identify_suspicious_levels(
                    bids, asks, features, combined_score
                )
                
                for level_info in suspicious_levels:
                    iceberg = IcebergOrder(
                        price_level=level_info['price'],
                        side=level_info['side'],
                        visible_size=level_info['volume'],
                        estimated_total_size=level_info['estimated_total'],
                        confidence_score=level_info['confidence'],
                        detection_time=datetime.now(),
                        reload_count=0,
                        execution_pattern='aggressive',
                        metadata={
                            'detection_method': 'machine_learning',
                            'ml_probability': iceberg_prob,
                            'anomaly_score': anomaly_score
                        }
                    )
                    detected.append(iceberg)
            
            return detected
            
        except Exception as e:
            logger.error(f"Error in ML detection: {e}")
            return []
    
    def _pattern_detection(self,
                          order_book: Dict[str, Any],
                          trades: List[Dict[str, Any]]) -> List[IcebergOrder]:
        """Pattern-based iceberg detection"""
        detected = []
        
        try:
            # Pattern 1: Reload detection
            reload_patterns = self._detect_reload_patterns()
            
            for price, pattern_info in reload_patterns.items():
                if pattern_info['confidence'] >= self.min_confidence:
                    iceberg = IcebergOrder(
                        price_level=price,
                        side=pattern_info['side'],
                        visible_size=pattern_info['current_volume'],
                        estimated_total_size=pattern_info['estimated_total'],
                        confidence_score=pattern_info['confidence'],
                        detection_time=datetime.now(),
                        reload_count=pattern_info['reload_count'],
                        execution_pattern='passive',
                        metadata={
                            'detection_method': 'reload_pattern',
                            'reload_frequency': pattern_info['reload_frequency']
                        }
                    )
                    detected.append(iceberg)
            
            # Pattern 2: Execution pattern analysis
            execution_patterns = self._analyze_execution_patterns(trades)
            
            for pattern in execution_patterns:
                if pattern['confidence'] >= self.min_confidence:
                    iceberg = IcebergOrder(
                        price_level=pattern['price'],
                        side=pattern['side'],
                        visible_size=pattern['visible_size'],
                        estimated_total_size=pattern['estimated_total'],
                        confidence_score=pattern['confidence'],
                        detection_time=datetime.now(),
                        reload_count=0,
                        execution_pattern=pattern['execution_type'],
                        metadata={
                            'detection_method': 'execution_pattern',
                            'pattern_details': pattern['details']
                        }
                    )
                    detected.append(iceberg)
            
            # Pattern 3: Time-based patterns
            time_patterns = self._detect_time_patterns(order_book)
            detected.extend(time_patterns)
            
            return detected
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            return []
    
    def _detect_reload_patterns(self) -> Dict[float, Dict]:
        """Detect order reload patterns indicating icebergs"""
        reload_patterns = {}
        
        try:
            for price, tracker in self.price_level_tracker.items():
                if len(tracker['volumes']) < 5:
                    continue
                
                # Analyze volume changes
                volumes = tracker['volumes']
                timestamps = tracker['timestamps']
                
                # Detect reloads (volume increases after decreases)
                reloads = []
                for i in range(1, len(volumes)):
                    if i > 0 and volumes[i] > volumes[i-1] * 1.2:  # 20% increase
                        if i > 1 and volumes[i-1] < volumes[i-2] * 0.8:  # After decrease
                            reloads.append(i)
                
                if len(reloads) >= 2:
                    # Calculate reload frequency
                    if len(timestamps) > 1:
                        time_span = (timestamps[-1] - timestamps[0]).total_seconds()
                        reload_frequency = len(reloads) / (time_span / 60) if time_span > 0 else 0
                    else:
                        reload_frequency = 0
                    
                    # Estimate total size
                    avg_visible = np.mean(volumes)
                    max_visible = np.max(volumes)
                    estimated_total = max_visible * (1 + len(reloads))
                    
                    # Calculate confidence
                    confidence = min(1.0, len(reloads) / 5 * 0.5 + reload_frequency / 10 * 0.5)
                    
                    reload_patterns[price] = {
                        'side': 'bid' if price < np.mean(list(self.price_level_tracker.keys())) else 'ask',
                        'current_volume': volumes[-1] if volumes else 0,
                        'estimated_total': estimated_total,
                        'reload_count': len(reloads),
                        'reload_frequency': reload_frequency,
                        'confidence': confidence * self.sensitivity
                    }
            
            return reload_patterns
            
        except Exception as e:
            logger.error(f"Error detecting reload patterns: {e}")
            return {}
    
    def _analyze_execution_patterns(self, trades: List[Dict[str, Any]]) -> List[Dict]:
        """Analyze trade execution patterns for iceberg detection"""
        patterns = []
        
        try:
            if len(trades) < 10:
                return patterns
            
            # Group trades by price level
            price_groups = defaultdict(list)
            for trade in trades:
                # Round to nearest tick
                price_key = round(trade['price'], 2)
                price_groups[price_key].append(trade)
            
            for price, price_trades in price_groups.items():
                if len(price_trades) < 3:
                    continue
                
                # Analyze execution characteristics
                volumes = [t['volume'] for t in price_trades]
                timestamps = [t['timestamp'] for t in price_trades]
                
                # Check for consistent execution size (iceberg characteristic)
                volume_cv = np.std(volumes) / (np.mean(volumes) + 1e-8)
                
                if volume_cv < 0.3:  # Low variation suggests algorithmic execution
                    # Check execution frequency
                    if len(timestamps) > 1:
                        time_diffs = np.diff(timestamps)
                        avg_interval = np.mean(time_diffs)
                        interval_cv = np.std(time_diffs) / (avg_interval + 1e-8)
                        
                        if interval_cv < 0.5:  # Regular execution intervals
                            pattern = {
                                'price': price,
                                'side': 'bid' if price_trades[0].get('side') == 'buy' else 'ask',
                                'visible_size': np.mean(volumes),
                                'estimated_total': sum(volumes) * 3,  # Estimate remaining
                                'execution_type': 'algorithmic',
                                'confidence': (1 - volume_cv) * (1 - interval_cv) * self.sensitivity,
                                'details': {
                                    'avg_size': np.mean(volumes),
                                    'execution_count': len(price_trades),
                                    'avg_interval': avg_interval
                                }
                            }
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing execution patterns: {e}")
            return []
    
    def _detect_time_patterns(self, order_book: Dict[str, Any]) -> List[IcebergOrder]:
        """Detect time-based patterns in order book"""
        detected = []
        
        try:
            if len(self.order_book_history) < 20:
                return detected
            
            # Analyze order appearance patterns
            bids = np.array(order_book.get('bids', []))
            asks = np.array(order_book.get('asks', []))
            
            # Check for orders that appear at specific times
            current_time = datetime.fromtimestamp(order_book['timestamp'])
            
            for price, tracker in self.price_level_tracker.items():
                if len(tracker['timestamps']) < 5:
                    continue
                
                # Check for regular time patterns
                timestamps = tracker['timestamps']
                if len(timestamps) > 2:
                    # Calculate time intervals
                    intervals = []
                    for i in range(1, len(timestamps)):
                        intervals.append((timestamps[i] - timestamps[i-1]).total_seconds())
                    
                    if intervals:
                        # Check for regularity
                        avg_interval = np.mean(intervals)
                        std_interval = np.std(intervals)
                        
                        if std_interval < avg_interval * 0.3:  # Regular pattern
                            # Find current volume
                            current_volume = 0
                            side = ''
                            
                            for bid_price, bid_vol in bids[:20]:
                                if abs(bid_price - price) < price * 0.0001:
                                    current_volume = bid_vol
                                    side = 'bid'
                                    break
                            
                            if current_volume == 0:
                                for ask_price, ask_vol in asks[:20]:
                                    if abs(ask_price - price) < price * 0.0001:
                                        current_volume = ask_vol
                                        side = 'ask'
                                        break
                            
                            if current_volume > 0 and side:
                                confidence = min(1.0, (1 - std_interval / (avg_interval + 1)) * self.sensitivity)
                                
                                if confidence >= self.min_confidence:
                                    iceberg = IcebergOrder(
                                        price_level=price,
                                        side=side,
                                        visible_size=current_volume,
                                        estimated_total_size=current_volume * len(timestamps),
                                        confidence_score=confidence,
                                        detection_time=current_time,
                                        reload_count=len(timestamps) - 1,
                                        execution_pattern='time_based',
                                        metadata={
                                            'detection_method': 'time_pattern',
                                            'avg_interval': avg_interval,
                                            'pattern_regularity': 1 - std_interval / (avg_interval + 1)
                                        }
                                    )
                                    detected.append(iceberg)
            
            return detected
            
        except Exception as e:
            logger.error(f"Error detecting time patterns: {e}")
            return []
    
    def _detect_algorithmic_patterns(self, trades: List[Dict[str, Any]]) -> List[AlgorithmicPattern]:
        """Detect algorithmic trading patterns"""
        patterns = []
        
        try:
            if len(trades) < 20:
                return patterns
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(trades)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.set_index('timestamp')
                
                # Resample to regular intervals
                resampled = df.resample('1min').agg({
                    'price': ['mean', 'min', 'max'],
                    'volume': 'sum'
                })
                
                # TWAP Detection
                twap_pattern = self._detect_twap(resampled)
                if twap_pattern:
                    patterns.append(twap_pattern)
                
                # VWAP Detection
                vwap_pattern = self._detect_vwap(df)
                if vwap_pattern:
                    patterns.append(vwap_pattern)
                
                # POV (Percentage of Volume) Detection
                pov_pattern = self._detect_pov(df)
                if pov_pattern:
                    patterns.append(pov_pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting algorithmic patterns: {e}")
            return []
    
    def _detect_twap(self, resampled_data: pd.DataFrame) -> Optional[AlgorithmicPattern]:
        """Detect TWAP (Time-Weighted Average Price) execution"""
        try:
            if len(resampled_data) < 5:
                return None
            
            volumes = resampled_data[('volume', 'sum')].values
            
            # Check for consistent volume distribution
            volume_cv = np.std(volumes) / (np.mean(volumes) + 1e-8)
            
            if volume_cv < 0.3:  # Low variation suggests TWAP
                prices = resampled_data[('price', 'mean')].values
                
                pattern = AlgorithmicPattern(
                    pattern_type='twap',
                    start_time=resampled_data.index[0],
                    end_time=resampled_data.index[-1],
                    price_range=(np.min(prices), np.max(prices)),
                    volume_profile=volumes.tolist(),
                    participation_rate=np.mean(volumes) / np.sum(volumes),
                    aggressiveness=0.3,  # TWAP is typically passive
                    confidence=(1 - volume_cv) * self.sensitivity
                )
                
                if pattern.confidence >= self.min_confidence:
                    return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting TWAP: {e}")
            return None
    
    def _detect_vwap(self, trades_df: pd.DataFrame) -> Optional[AlgorithmicPattern]:
        """Detect VWAP (Volume-Weighted Average Price) execution"""
        try:
            if len(trades_df) < 10:
                return None
            
            # Calculate cumulative volume profile
            trades_df['cum_volume'] = trades_df['volume'].cumsum()
            trades_df['vwap'] = (trades_df['price'] * trades_df['volume']).cumsum() / trades_df['cum_volume']
            
            # Check if trades cluster around VWAP
            price_deviations = abs(trades_df['price'] - trades_df['vwap']) / trades_df['vwap']
            
            if price_deviations.mean() < 0.002:  # Within 20 bps of VWAP
                pattern = AlgorithmicPattern(
                    pattern_type='vwap',
                    start_time=trades_df.index[0],
                    end_time=trades_df.index[-1],
                    price_range=(trades_df['price'].min(), trades_df['price'].max()),
                    volume_profile=trades_df['volume'].values.tolist(),
                    participation_rate=0.1,  # Placeholder
                    aggressiveness=0.5,
                    confidence=(1 - price_deviations.mean() * 100) * self.sensitivity
                )
                
                if pattern.confidence >= self.min_confidence:
                    return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting VWAP: {e}")
            return None
    
    def _detect_pov(self, trades_df: pd.DataFrame) -> Optional[AlgorithmicPattern]:
        """Detect POV (Percentage of Volume) execution"""
        try:
            if len(trades_df) < 20:
                return None
            
            # Calculate rolling participation rate
            window = 10
            rolling_volume = trades_df['volume'].rolling(window).sum()
            
            # Look for consistent participation
            participation_rates = trades_df['volume'] / rolling_volume
            participation_rates = participation_rates.dropna()
            
            if len(participation_rates) > 0:
                avg_participation = participation_rates.mean()
                participation_cv = participation_rates.std() / (avg_participation + 1e-8)
                
                if 0.05 <= avg_participation <= 0.3 and participation_cv < 0.5:
                    pattern = AlgorithmicPattern(
                        pattern_type='pov',
                        start_time=trades_df.index[0],
                        end_time=trades_df.index[-1],
                        price_range=(trades_df['price'].min(), trades_df['price'].max()),
                        volume_profile=trades_df['volume'].values.tolist(),
                        participation_rate=avg_participation,
                        aggressiveness=avg_participation * 2,  # Higher participation = more aggressive
                        confidence=(1 - participation_cv) * self.sensitivity
                    )
                    
                    if pattern.confidence >= self.min_confidence:
                        return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting POV: {e}")
            return None
    
    def _update_price_level_tracking(self,
                                    order_book: Dict[str, Any],
                                    trades: List[Dict[str, Any]]):
        """Update tracking for price levels"""
        try:
            current_time = datetime.fromtimestamp(order_book.get('timestamp', datetime.now().timestamp()))
            
            # Track bid levels
            bids = order_book.get('bids', [])
            for price, volume in bids[:20]:
                tracker = self.price_level_tracker[price]
                tracker['appearances'] += 1
                tracker['volumes'].append(volume)
                tracker['timestamps'].append(current_time)
                
                # Track executions at this level
                for trade in trades:
                    if abs(trade['price'] - price) < price * 0.0001:
                        tracker['executions'] += 1
                
                # Detect reload events
                if len(tracker['volumes']) > 1:
                    if volume > tracker['volumes'][-2] * 1.2:  # 20% increase
                        tracker['reload_events'].append(current_time)
            
            # Track ask levels
            asks = order_book.get('asks', [])
            for price, volume in asks[:20]:
                tracker = self.price_level_tracker[price]
                tracker['appearances'] += 1
                tracker['volumes'].append(volume)
                tracker['timestamps'].append(current_time)
                
                for trade in trades:
                    if abs(trade['price'] - price) < price * 0.0001:
                        tracker['executions'] += 1
                
                if len(tracker['volumes']) > 1:
                    if volume > tracker['volumes'][-2] * 1.2:
                        tracker['reload_events'].append(current_time)
            
            # Clean old tracking data
            self._clean_old_tracking()
            
        except Exception as e:
            logger.error(f"Error updating price level tracking: {e}")
    
    def _clean_old_tracking(self):
        """Clean old price level tracking data"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=1)
            
            levels_to_remove = []
            for price, tracker in self.price_level_tracker.items():
                if tracker['timestamps']:
                    if tracker['timestamps'][-1] < cutoff_time:
                        levels_to_remove.append(price)
                    else:
                        # Clean old entries within tracker
                        valid_indices = [i for i, t in enumerate(tracker['timestamps']) if t > cutoff_time]
                        if valid_indices:
                            tracker['volumes'] = [tracker['volumes'][i] for i in valid_indices]
                            tracker['timestamps'] = [tracker['timestamps'][i] for i in valid_indices]
                            tracker['reload_events'] = [t for t in tracker['reload_events'] if t > cutoff_time]
            
            for price in levels_to_remove:
                del self.price_level_tracker[price]
                
        except Exception as e:
            logger.error(f"Error cleaning tracking data: {e}")
    
    def _check_persistence(self, price: float, side: str) -> float:
        """Check persistence of a price level"""
        try:
            if len(self.order_book_history) < 5:
                return 0.0
            
            appearances = 0
            for hist_book in self.order_book_history[-10:]:
                book_side = hist_book.get('bids' if side == 'bid' else 'asks', [])
                
                for book_price, _ in book_side[:20]:
                    if abs(book_price - price) < price * 0.0001:
                        appearances += 1
                        break
            
            return appearances / min(10, len(self.order_book_history))
            
        except Exception as e:
            logger.error(f"Error checking persistence: {e}")
            return 0.0
    
    def _count_reloads(self, price: float) -> int:
        """Count reload events for a price level"""
        try:
            if price in self.price_level_tracker:
                return len(self.price_level_tracker[price]['reload_events'])
            return 0
            
        except Exception as e:
            logger.error(f"Error counting reloads: {e}")
            return 0
    
    def _detect_shape_anomalies(self,
                                prices: np.ndarray,
                                volumes: np.ndarray,
                                side: str) -> List[Dict]:
        """Detect shape anomalies in order book"""
        anomalies = []
        
        try:
            if len(prices) < 10:
                return anomalies
            
            # Expected decay pattern
            expected_decay = np.exp(-np.arange(len(volumes)) * 0.1)
            expected_volumes = expected_decay * volumes[0]
            
            # Find deviations
            deviations = volumes - expected_volumes
            
            for i, deviation in enumerate(deviations[:10]):
                if deviation > expected_volumes[i] * 0.5:  # 50% above expected
                    anomaly = {
                        'price': prices[i],
                        'volume': volumes[i],
                        'estimated_total': volumes[i] * 3,
                        'confidence': min(1.0, deviation / expected_volumes[i]) * self.sensitivity,
                        'type': 'volume_spike'
                    }
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting shape anomalies: {e}")
            return []
    
    def _extract_ml_features(self, order_book: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features for ML detection"""
        try:
            features = []
            
            bids = np.array(order_book.get('bids', []))
            asks = np.array(order_book.get('asks', []))
            
            if len(bids) < 20 or len(asks) < 20:
                return None
            
            # Volume distribution features
            bid_volumes = bids[:20, 1]
            ask_volumes = asks[:20, 1]
            
            features.extend([
                np.mean(bid_volumes),
                np.std(bid_volumes),
                np.max(bid_volumes),
                stats.skew(bid_volumes),
                stats.kurtosis(bid_volumes),
                np.mean(ask_volumes),
                np.std(ask_volumes),
                np.max(ask_volumes),
                stats.skew(ask_volumes),
                stats.kurtosis(ask_volumes)
            ])
            
            # Price level features
            bid_prices = bids[:20, 0]
            ask_prices = asks[:20, 0]
            
            mid_price = (bid_prices[0] + ask_prices[0]) / 2
            
            features.extend([
                (ask_prices[0] - bid_prices[0]) / mid_price,  # Spread
                np.std(np.diff(bid_prices)),  # Price level regularity
                np.std(np.diff(ask_prices))
            ])
            
            # Imbalance features
            total_bid_vol = np.sum(bid_volumes)
            total_ask_vol = np.sum(ask_volumes)
            
            features.append((total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol))
            
            # Historical features (if available)
            if len(self.order_book_history) > 5:
                volume_changes = []
                for i in range(1, min(6, len(self.order_book_history))):
                    prev_bids = np.array(self.order_book_history[-i].get('bids', []))
                    if len(prev_bids) > 0:
                        prev_total = np.sum(prev_bids[:20, 1])
                        curr_total = total_bid_vol
                        volume_changes.append((curr_total - prev_total) / (prev_total + 1))
                
                if volume_changes:
                    features.extend([
                        np.mean(volume_changes),
                        np.std(volume_changes)
                    ])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting ML features: {e}")
            return None
    
    def _identify_suspicious_levels(self,
                                   bids: np.ndarray,
                                   asks: np.ndarray,
                                   features: np.ndarray,
                                   ml_score: float) -> List[Dict]:
        """Identify specific suspicious price levels"""
        suspicious = []
        
        try:
            # Use feature importance to identify levels
            bid_volumes = bids[:10, 1] if len(bids) > 10 else bids[:, 1]
            ask_volumes = asks[:10, 1] if len(asks) > 10 else asks[:, 1]
            
            # Find volume outliers
            bid_mean = np.mean(bid_volumes)
            bid_std = np.std(bid_volumes)
            
            for i, (price, volume) in enumerate(bids[:10]):
                if volume > bid_mean + 2 * bid_std:
                    suspicious.append({
                        'price': price,
                        'side': 'bid',
                        'volume': volume,
                        'estimated_total': volume * (1 + ml_score * 3),
                        'confidence': ml_score * (volume / (bid_mean + bid_std))
                    })
            
            ask_mean = np.mean(ask_volumes)
            ask_std = np.std(ask_volumes)
            
            for i, (price, volume) in enumerate(asks[:10]):
                if volume > ask_mean + 2 * ask_std:
                    suspicious.append({
                        'price': price,
                        'side': 'ask',
                        'volume': volume,
                        'estimated_total': volume * (1 + ml_score * 3),
                        'confidence': ml_score * (volume / (ask_mean + ask_std))
                    })
            
            return suspicious
            
        except Exception as e:
            logger.error(f"Error identifying suspicious levels: {e}")
            return []
    
    def _merge_detections(self, detections: List[IcebergOrder]) -> List[IcebergOrder]:
        """Merge and deduplicate iceberg detections"""
        if not detections:
            return []
        
        merged = {}
        
        for detection in detections:
            key = (round(detection.price_level, 2), detection.side)
            
            if key not in merged:
                merged[key] = detection
            else:
                # Merge with existing detection
                existing = merged[key]
                
                # Use weighted average for estimates
                total_confidence = existing.confidence_score + detection.confidence_score
                
                existing.estimated_total_size = (
                    existing.estimated_total_size * existing.confidence_score +
                    detection.estimated_total_size * detection.confidence_score
                ) / total_confidence
                
                existing.confidence_score = min(1.0, total_confidence / 2)
                existing.reload_count = max(existing.reload_count, detection.reload_count)
                
                # Merge metadata
                existing.metadata.update(detection.metadata)
        
        return list(merged.values())
    
    def _estimate_hidden_liquidity(self,
                                  icebergs: List[IcebergOrder],
                                  order_book: Dict[str, Any]) -> float:
        """Estimate total hidden liquidity"""
        try:
            total_hidden = 0.0
            
            for iceberg in icebergs:
                hidden_size = iceberg.estimated_total_size - iceberg.visible_size
                total_hidden += hidden_size * iceberg.confidence_score
            
            # Add estimated dark pool liquidity
            visible_liquidity = 0.0
            bids = np.array(order_book.get('bids', []))
            asks = np.array(order_book.get('asks', []))
            
            if len(bids) > 0:
                visible_liquidity += np.sum(bids[:, 1])
            if len(asks) > 0:
                visible_liquidity += np.sum(asks[:, 1])
            
            # Estimate dark pool as percentage of visible
            dark_pool_estimate = visible_liquidity * 0.3  # Assume 30% in dark pools
            
            return total_hidden + dark_pool_estimate
            
        except Exception as e:
            logger.error(f"Error estimating hidden liquidity: {e}")
            return 0.0
    
    def _calculate_confidence_scores(self, icebergs: List[IcebergOrder]) -> Dict[str, float]:
        """Calculate confidence scores for detection"""
        scores = {
            'overall': 0.0,
            'statistical': 0.0,
            'ml': 0.0,
            'pattern': 0.0
        }
        
        try:
            if not icebergs:
                return scores
            
            statistical_scores = []
            ml_scores = []
            pattern_scores = []
            
            for iceberg in icebergs:
                method = iceberg.metadata.get('detection_method', '')
                
                if 'statistical' in method:
                    statistical_scores.append(iceberg.confidence_score)
                elif 'machine_learning' in method:
                    ml_scores.append(iceberg.confidence_score)
                elif 'pattern' in method or 'reload' in method:
                    pattern_scores.append(iceberg.confidence_score)
            
            if statistical_scores:
                scores['statistical'] = np.mean(statistical_scores)
            if ml_scores:
                scores['ml'] = np.mean(ml_scores)
            if pattern_scores:
                scores['pattern'] = np.mean(pattern_scores)
            
            # Overall confidence
            all_scores = [s for s in [scores['statistical'], scores['ml'], scores['pattern']] if s > 0]
            if all_scores:
                scores['overall'] = np.mean(all_scores)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {e}")
            return scores
    
    def _update_tracked_icebergs(self,
                                 new_detections: List[IcebergOrder],
                                 trades: List[Dict[str, Any]]):
        """Update tracked iceberg orders with execution data"""
        try:
            # Update existing icebergs
            for price, iceberg in list(self.detected_icebergs.items()):
                # Check for executions
                executed_volume = 0
                for trade in trades:
                    if abs(trade['price'] - price) < price * 0.0001:
                        executed_volume += trade['volume']
                
                if executed_volume > 0:
                    iceberg.metadata['executed_volume'] = iceberg.metadata.get('executed_volume', 0) + executed_volume
                
                # Remove if fully executed
                if iceberg.remaining_size <= 0:
                    del self.detected_icebergs[price]
            
            # Add new detections
            for detection in new_detections:
                self.detected_icebergs[detection.price_level] = detection
            
            # Clean old detections
            current_time = datetime.now()
            for price, iceberg in list(self.detected_icebergs.items()):
                age = (current_time - iceberg.detection_time).total_seconds()
                if age > 3600:  # Remove after 1 hour
                    del self.detected_icebergs[price]
                    
        except Exception as e:
            logger.error(f"Error updating tracked icebergs: {e}")
    
    def calibrate(self, training_data: List[Dict[str, Any]], labels: List[int]):
        """Calibrate ML models with training data"""
        try:
            features = []
            for data in training_data:
                feature_vector = self._extract_ml_features(data)
                if feature_vector is not None:
                    features.append(feature_vector)
            
            if len(features) >= 10:
                X = np.array(features)
                y = np.array(labels[:len(features)])
                
                # Fit scaler
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                
                # Train models
                self.ml_detector.fit(X_scaled, y)
                self.anomaly_detector.fit(X_scaled[y == 0])  # Train on normal data
                
                # Calculate feature importance
                self.feature_importance = dict(zip(
                    range(len(features[0])),
                    self.ml_detector.feature_importances_
                ))
                
                self.is_calibrated = True
                logger.info("Iceberg detector calibrated successfully")
            else:
                logger.warning("Insufficient training data for calibration")
                
        except Exception as e:
            logger.error(f"Error calibrating iceberg detector: {e}")
    
    def get_active_icebergs(self) -> List[IcebergOrder]:
        """Get currently active iceberg orders"""
        return list(self.detected_icebergs.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        stats = {
            'active_icebergs': len(self.detected_icebergs),
            'total_hidden_volume': sum([i.remaining_size for i in self.detected_icebergs.values()]),
            'algorithmic_patterns': len(self.algorithmic_patterns),
            'tracked_price_levels': len(self.price_level_tracker),
            'is_calibrated': self.is_calibrated
        }
        
        if self.detected_icebergs:
            stats['avg_confidence'] = np.mean([i.confidence_score for i in self.detected_icebergs.values()])
            stats['avg_hidden_ratio'] = np.mean([i.hidden_ratio for i in self.detected_icebergs.values()])
        else:
            stats['avg_confidence'] = 0.0
            stats['avg_hidden_ratio'] = 0.0
        
        return stats


if __name__ == "__main__":
    # Example usage
    detector = IcebergDetector(
        symbol="BTC/USDT",
        detection_sensitivity=0.7,
        min_confidence=0.6
    )
    
    # Mock order book
    mock_order_book = {
        'timestamp': datetime.now().timestamp(),
        'bids': [[50000 - i*10, np.random.uniform(0.5, 2.0)] for i in range(50)],
        'asks': [[50001 + i*10, np.random.uniform(0.5, 2.0)] for i in range(50)]
    }
    
    # Add an artificial iceberg
    mock_order_book['bids'][3] = [49970, 5.0]  # Large order
    
    # Mock trades
    mock_trades = [
        {'price': 50000, 'volume': 0.1, 'timestamp': datetime.now().timestamp()},
        {'price': 49970, 'volume': 0.5, 'timestamp': datetime.now().timestamp()},
        {'price': 49970, 'volume': 0.5, 'timestamp': datetime.now().timestamp() - 60},
        {'price': 49970, 'volume': 0.5, 'timestamp': datetime.now().timestamp() - 120}
    ]
    
    # Detect icebergs
    results = detector.detect(mock_order_book, mock_trades)
    
    print(f"Detected {len(results['icebergs'])} iceberg orders")
    for iceberg in results['icebergs']:
        print(f"  Price: {iceberg.price_level}, Side: {iceberg.side}, "
              f"Hidden Ratio: {iceberg.hidden_ratio:.2%}, Confidence: {iceberg.confidence_score:.2f}")
    
    print(f"\nEstimated Hidden Liquidity: {results['hidden_liquidity_estimate']:.2f}")
    print(f"Detection Confidence: {results['detection_confidence']}")
    
    # Get statistics
    stats = detector.get_statistics()
    print(f"\nStatistics: {stats}")