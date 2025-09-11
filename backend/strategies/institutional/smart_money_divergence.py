"""
Smart Money Divergence Detection
=================================

Advanced system for identifying institutional accumulation/distribution patterns
and detecting divergences between smart money flows and retail sentiment.

Key Features:
- Real-time smart money flow analysis
- Accumulation/distribution pattern recognition
- Price-volume divergence detection
- Institutional vs retail sentiment analysis
- Multi-timeframe confluence scoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from collections import deque
import aiohttp
import json

logger = logging.getLogger(__name__)


class DivergenceType(Enum):
    """Types of smart money divergences"""
    BULLISH_ACCUMULATION = "bullish_accumulation"  # Price down, smart money buying
    BEARISH_DISTRIBUTION = "bearish_distribution"  # Price up, smart money selling
    HIDDEN_BULLISH = "hidden_bullish"  # Higher lows in price, lower lows in flow
    HIDDEN_BEARISH = "hidden_bearish"  # Lower highs in price, higher highs in flow
    VOLUME_DIVERGENCE = "volume_divergence"  # Price/volume relationship broken
    FLOW_DIVERGENCE = "flow_divergence"  # Exchange flows vs price action


class MoneyFlowType(Enum):
    """Classification of money flows"""
    SMART_MONEY = "smart_money"  # Institutional, whale wallets
    RETAIL_FLOW = "retail_flow"  # Small transactions
    EXCHANGE_INFLOW = "exchange_inflow"  # To exchanges (bearish)
    EXCHANGE_OUTFLOW = "exchange_outflow"  # From exchanges (bullish)
    ACCUMULATION = "accumulation"  # Net buying pressure
    DISTRIBUTION = "distribution"  # Net selling pressure


@dataclass
class SmartMoneySignal:
    """Smart money divergence signal"""
    timestamp: datetime
    symbol: str
    divergence_type: DivergenceType
    strength: float  # 0-1 signal strength
    confidence: float  # 0-1 confidence score
    smart_money_flow: float  # Net flow in USD
    retail_flow: float  # Retail flow in USD
    price_change: float  # Price change %
    volume_ratio: float  # Smart/retail volume ratio
    accumulation_score: float  # -1 to 1 (distribution to accumulation)
    timeframe: str  # Signal timeframe
    whale_transactions: List[Dict]  # Recent whale txs
    exchange_flows: Dict[str, float]  # Exchange in/out flows
    on_chain_metrics: Dict[str, Any]  # Additional metrics
    signal_components: Dict[str, float]  # Individual signal contributions
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DivergencePattern:
    """Detected divergence pattern"""
    pattern_type: str
    start_time: datetime
    end_time: datetime
    pivot_points: List[Tuple[datetime, float]]  # Time, price pivots
    flow_pivots: List[Tuple[datetime, float]]  # Time, flow pivots
    divergence_angle: float  # Angle between price and flow trends
    statistical_significance: float  # P-value
    expected_resolution: str  # Expected price direction
    confidence_interval: Tuple[float, float]  # CI for prediction


class SmartMoneyDivergenceDetector:
    """
    Advanced smart money divergence detection system.
    
    Tracks institutional flows and identifies divergences between
    smart money accumulation/distribution and price action.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize divergence detector"""
        self.config = config or self._default_config()
        
        # API configurations
        self.api_keys = {
            'etherscan': self.config.get('etherscan_api_key', ''),
            'bitquery': self.config.get('bitquery_api_key', ''),
            'whalealert': self.config.get('whalealert_api_key', ''),
            'glassnode': self.config.get('glassnode_api_key', '')
        }
        
        # Detection parameters
        self.min_whale_threshold = self.config.get('min_whale_threshold', 1_000_000)  # $1M USD
        self.lookback_periods = self.config.get('lookback_periods', {
            '1h': 24,  # 24 hours
            '4h': 42,  # 7 days
            '1d': 30   # 30 days
        })
        
        # State tracking
        self.price_history = {}  # Symbol -> deque of prices
        self.flow_history = {}  # Symbol -> deque of flows
        self.whale_cache = {}  # Recent whale transactions
        self.divergence_patterns = {}  # Active patterns
        self.signal_buffer = deque(maxlen=1000)
        
        # Statistical models
        self.scaler = StandardScaler()
        self.flow_momentum = {}  # Flow momentum indicators
        self.accumulation_index = {}  # Wyckoff accumulation index
        
        # Performance tracking
        self.signal_accuracy = {}
        self.false_positive_rate = 0.0
        self.detection_latency = []
        
        logger.info("Smart Money Divergence Detector initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'min_whale_threshold': 1_000_000,  # $1M USD
            'divergence_threshold': 0.3,  # 30% divergence threshold
            'min_confidence': 0.6,  # Minimum confidence for signals
            'volume_ma_period': 20,  # Volume MA period
            'flow_ema_period': 9,  # Flow EMA period
            'accumulation_window': 14,  # Accumulation/distribution window
            'divergence_lookback': 50,  # Bars to look back for divergence
            'statistical_significance': 0.05,  # P-value threshold
            'max_api_retries': 3,
            'api_timeout': 10,
            'cache_ttl': 300  # 5 minutes
        }
    
    async def detect_divergence(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        timeframe: str = '1h'
    ) -> List[SmartMoneySignal]:
        """
        Main divergence detection method.
        
        Args:
            symbol: Trading symbol
            price_data: OHLCV price data
            timeframe: Analysis timeframe
            
        Returns:
            List of detected divergence signals
        """
        try:
            # Fetch on-chain data
            on_chain_data = await self._fetch_on_chain_data(symbol)
            
            # Get whale transactions
            whale_txs = await self._fetch_whale_transactions(symbol)
            
            # Calculate smart money flows
            smart_flows = await self._calculate_smart_money_flows(
                symbol, on_chain_data, whale_txs
            )
            
            # Get exchange flows
            exchange_flows = await self._fetch_exchange_flows(symbol)
            
            # Detect divergence patterns
            patterns = self._identify_divergence_patterns(
                price_data, smart_flows, exchange_flows
            )
            
            # Generate signals
            signals = []
            for pattern in patterns:
                signal = await self._create_divergence_signal(
                    symbol, pattern, price_data, smart_flows,
                    whale_txs, exchange_flows, timeframe
                )
                if signal and signal.confidence >= self.config['min_confidence']:
                    signals.append(signal)
                    self.signal_buffer.append(signal)
            
            # Update performance metrics
            self._update_performance_metrics(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting divergence for {symbol}: {e}")
            return []
    
    async def _fetch_on_chain_data(self, symbol: str) -> Dict:
        """Fetch on-chain metrics"""
        data = {
            'network_value': 0,
            'transaction_count': 0,
            'active_addresses': 0,
            'exchange_balance': 0,
            'miner_balance': 0,
            'long_term_holder_balance': 0
        }
        
        # Etherscan API for Ethereum-based tokens
        if self.api_keys['etherscan']:
            try:
                async with aiohttp.ClientSession() as session:
                    # Get token metrics
                    url = f"https://api.etherscan.io/api"
                    params = {
                        'module': 'stats',
                        'action': 'tokensupply',
                        'contractaddress': self._get_contract_address(symbol),
                        'apikey': self.api_keys['etherscan']
                    }
                    
                    async with session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            if result['status'] == '1':
                                data['network_value'] = float(result['result'])
                    
                    # Get recent transactions
                    params['action'] = 'tokentx'
                    params['sort'] = 'desc'
                    params['page'] = 1
                    params['offset'] = 100
                    
                    async with session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            if result['status'] == '1':
                                txs = result['result']
                                data['transaction_count'] = len(txs)
                                data['active_addresses'] = len(set(
                                    [tx['from'] for tx in txs] + 
                                    [tx['to'] for tx in txs]
                                ))
                                
            except Exception as e:
                logger.warning(f"Etherscan API error: {e}")
        
        # Bitquery for additional metrics
        if self.api_keys['bitquery']:
            try:
                data.update(await self._fetch_bitquery_metrics(symbol))
            except Exception as e:
                logger.warning(f"Bitquery API error: {e}")
        
        return data
    
    async def _fetch_whale_transactions(self, symbol: str) -> List[Dict]:
        """Fetch recent whale transactions"""
        whale_txs = []
        
        if self.api_keys['whalealert']:
            try:
                async with aiohttp.ClientSession() as session:
                    url = "https://api.whale-alert.io/v1/transactions"
                    params = {
                        'api_key': self.api_keys['whalealert'],
                        'min_value': self.min_whale_threshold,
                        'start': int((datetime.now() - timedelta(hours=24)).timestamp()),
                        'cursor': None,
                        'limit': 100
                    }
                    
                    # Filter for specific symbol if possible
                    if symbol.upper() in ['BTC', 'ETH', 'USDT', 'USDC']:
                        params['currency'] = symbol.lower()
                    
                    async with session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for tx in data.get('transactions', []):
                                whale_txs.append({
                                    'hash': tx['hash'],
                                    'from': tx.get('from', {}).get('address'),
                                    'to': tx.get('to', {}).get('address'),
                                    'amount': tx['amount'],
                                    'amount_usd': tx['amount_usd'],
                                    'timestamp': datetime.fromtimestamp(tx['timestamp']),
                                    'from_type': tx.get('from', {}).get('type'),
                                    'to_type': tx.get('to', {}).get('type')
                                })
                                
            except Exception as e:
                logger.warning(f"WhaleAlert API error: {e}")
        
        # Sort by amount
        whale_txs.sort(key=lambda x: x.get('amount_usd', 0), reverse=True)
        
        return whale_txs
    
    async def _calculate_smart_money_flows(
        self,
        symbol: str,
        on_chain_data: Dict,
        whale_txs: List[Dict]
    ) -> pd.DataFrame:
        """Calculate smart money flow metrics"""
        
        # Initialize flow dataframe
        now = datetime.now()
        timestamps = pd.date_range(
            end=now,
            periods=self.lookback_periods.get('1h', 24),
            freq='1h'
        )
        
        flows = pd.DataFrame(index=timestamps, columns=[
            'smart_money_flow',
            'retail_flow',
            'net_flow',
            'accumulation_index',
            'distribution_index',
            'flow_momentum'
        ])
        
        # Aggregate whale transactions by hour
        for tx in whale_txs:
            tx_hour = tx['timestamp'].replace(minute=0, second=0, microsecond=0)
            if tx_hour in flows.index:
                # Classify transaction
                if self._is_accumulation(tx):
                    flows.loc[tx_hour, 'smart_money_flow'] = \
                        flows.loc[tx_hour, 'smart_money_flow'] or 0
                    flows.loc[tx_hour, 'smart_money_flow'] += tx['amount_usd']
                elif self._is_distribution(tx):
                    flows.loc[tx_hour, 'smart_money_flow'] = \
                        flows.loc[tx_hour, 'smart_money_flow'] or 0
                    flows.loc[tx_hour, 'smart_money_flow'] -= tx['amount_usd']
        
        # Fill NaN values
        flows.fillna(0, inplace=True)
        
        # Calculate flow momentum (rate of change)
        flows['flow_momentum'] = flows['smart_money_flow'].pct_change(periods=3)
        
        # Calculate accumulation/distribution index
        flows['accumulation_index'] = self._calculate_accumulation_index(flows)
        flows['distribution_index'] = -flows['accumulation_index']
        
        # Estimate retail flow (inverse of smart money concentration)
        total_volume = on_chain_data.get('transaction_count', 1) * 1000  # Rough estimate
        flows['retail_flow'] = total_volume - flows['smart_money_flow'].abs()
        
        # Net flow
        flows['net_flow'] = flows['smart_money_flow'] - flows['retail_flow'] * 0.1
        
        return flows
    
    async def _fetch_exchange_flows(self, symbol: str) -> Dict[str, float]:
        """Fetch exchange inflow/outflow data"""
        flows = {
            'exchange_inflow': 0,
            'exchange_outflow': 0,
            'net_flow': 0,
            'flow_ratio': 0,
            'major_exchanges': {}
        }
        
        # This would typically use exchange-specific APIs or services like:
        # - CryptoQuant
        # - Glassnode
        # - IntoTheBlock
        
        # Simulated data based on whale transactions
        whale_txs = self.whale_cache.get(symbol, [])
        
        for tx in whale_txs:
            if tx.get('to_type') == 'exchange':
                flows['exchange_inflow'] += tx.get('amount_usd', 0)
                exchange_name = tx.get('to', {}).get('name', 'unknown')
                flows['major_exchanges'][exchange_name] = \
                    flows['major_exchanges'].get(exchange_name, 0) + tx.get('amount_usd', 0)
                    
            elif tx.get('from_type') == 'exchange':
                flows['exchange_outflow'] += tx.get('amount_usd', 0)
                exchange_name = tx.get('from', {}).get('name', 'unknown')
                flows['major_exchanges'][exchange_name] = \
                    flows['major_exchanges'].get(exchange_name, 0) - tx.get('amount_usd', 0)
        
        flows['net_flow'] = flows['exchange_outflow'] - flows['exchange_inflow']
        
        # Calculate flow ratio (bullish if > 1)
        if flows['exchange_inflow'] > 0:
            flows['flow_ratio'] = flows['exchange_outflow'] / flows['exchange_inflow']
        else:
            flows['flow_ratio'] = float('inf') if flows['exchange_outflow'] > 0 else 1.0
        
        return flows
    
    def _identify_divergence_patterns(
        self,
        price_data: pd.DataFrame,
        flow_data: pd.DataFrame,
        exchange_flows: Dict
    ) -> List[DivergencePattern]:
        """Identify divergence patterns between price and flows"""
        patterns = []
        
        if len(price_data) < self.config['divergence_lookback']:
            return patterns
        
        # Ensure we have price data
        if 'close' not in price_data.columns:
            return patterns
        
        # Get recent data
        lookback = self.config['divergence_lookback']
        recent_prices = price_data['close'].tail(lookback).values
        recent_flows = flow_data['smart_money_flow'].tail(lookback).values
        
        # Find price pivots
        price_pivots = self._find_pivots(recent_prices)
        
        # Find flow pivots
        flow_pivots = self._find_pivots(recent_flows)
        
        # Check for classic divergences
        if len(price_pivots['lows']) >= 2 and len(flow_pivots['lows']) >= 2:
            # Bullish divergence: lower price lows, higher flow lows
            if (price_pivots['lows'][-1][1] < price_pivots['lows'][-2][1] and
                flow_pivots['lows'][-1][1] > flow_pivots['lows'][-2][1]):
                
                pattern = DivergencePattern(
                    pattern_type=DivergenceType.BULLISH_ACCUMULATION.value,
                    start_time=price_data.index[-lookback + price_pivots['lows'][-2][0]],
                    end_time=price_data.index[-lookback + price_pivots['lows'][-1][0]],
                    pivot_points=[(price_data.index[-lookback + p[0]], p[1]) 
                                  for p in price_pivots['lows'][-2:]],
                    flow_pivots=[(price_data.index[-lookback + p[0]], p[1])
                                 for p in flow_pivots['lows'][-2:]],
                    divergence_angle=self._calculate_divergence_angle(
                        price_pivots['lows'][-2:], flow_pivots['lows'][-2:]
                    ),
                    statistical_significance=self._calculate_significance(
                        recent_prices, recent_flows
                    ),
                    expected_resolution='bullish',
                    confidence_interval=(0.6, 0.85)
                )
                patterns.append(pattern)
        
        if len(price_pivots['highs']) >= 2 and len(flow_pivots['highs']) >= 2:
            # Bearish divergence: higher price highs, lower flow highs
            if (price_pivots['highs'][-1][1] > price_pivots['highs'][-2][1] and
                flow_pivots['highs'][-1][1] < flow_pivots['highs'][-2][1]):
                
                pattern = DivergencePattern(
                    pattern_type=DivergenceType.BEARISH_DISTRIBUTION.value,
                    start_time=price_data.index[-lookback + price_pivots['highs'][-2][0]],
                    end_time=price_data.index[-lookback + price_pivots['highs'][-1][0]],
                    pivot_points=[(price_data.index[-lookback + p[0]], p[1])
                                  for p in price_pivots['highs'][-2:]],
                    flow_pivots=[(price_data.index[-lookback + p[0]], p[1])
                                 for p in flow_pivots['highs'][-2:]],
                    divergence_angle=self._calculate_divergence_angle(
                        price_pivots['highs'][-2:], flow_pivots['highs'][-2:]
                    ),
                    statistical_significance=self._calculate_significance(
                        recent_prices, recent_flows
                    ),
                    expected_resolution='bearish',
                    confidence_interval=(0.6, 0.85)
                )
                patterns.append(pattern)
        
        # Check for volume divergence
        if 'volume' in price_data.columns:
            volume_correlation = np.corrcoef(
                recent_prices, 
                price_data['volume'].tail(lookback).values
            )[0, 1]
            
            if abs(volume_correlation) < 0.3:  # Weak correlation indicates divergence
                pattern = DivergencePattern(
                    pattern_type=DivergenceType.VOLUME_DIVERGENCE.value,
                    start_time=price_data.index[-lookback],
                    end_time=price_data.index[-1],
                    pivot_points=[],
                    flow_pivots=[],
                    divergence_angle=0,
                    statistical_significance=1 - abs(volume_correlation),
                    expected_resolution='uncertain',
                    confidence_interval=(0.4, 0.6)
                )
                patterns.append(pattern)
        
        # Check exchange flow divergence
        if exchange_flows['net_flow'] != 0:
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Bullish: price down but outflows (accumulation)
            if price_trend < -0.05 and exchange_flows['net_flow'] > 0:
                pattern = DivergencePattern(
                    pattern_type=DivergenceType.FLOW_DIVERGENCE.value,
                    start_time=price_data.index[-lookback],
                    end_time=price_data.index[-1],
                    pivot_points=[],
                    flow_pivots=[],
                    divergence_angle=0,
                    statistical_significance=0.95,
                    expected_resolution='bullish',
                    confidence_interval=(0.65, 0.8)
                )
                patterns.append(pattern)
            
            # Bearish: price up but inflows (distribution)
            elif price_trend > 0.05 and exchange_flows['net_flow'] < 0:
                pattern = DivergencePattern(
                    pattern_type=DivergenceType.FLOW_DIVERGENCE.value,
                    start_time=price_data.index[-lookback],
                    end_time=price_data.index[-1],
                    pivot_points=[],
                    flow_pivots=[],
                    divergence_angle=0,
                    statistical_significance=0.95,
                    expected_resolution='bearish',
                    confidence_interval=(0.65, 0.8)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_pivots(self, data: np.ndarray, order: int = 5) -> Dict[str, List]:
        """Find pivot highs and lows in data"""
        pivots = {'highs': [], 'lows': []}
        
        if len(data) < order * 2 + 1:
            return pivots
        
        # Find peaks (highs)
        peaks, _ = signal.find_peaks(data, distance=order)
        for peak in peaks:
            pivots['highs'].append((peak, data[peak]))
        
        # Find troughs (lows)
        troughs, _ = signal.find_peaks(-data, distance=order)
        for trough in troughs:
            pivots['lows'].append((trough, data[trough]))
        
        return pivots
    
    def _calculate_divergence_angle(
        self,
        price_pivots: List[Tuple],
        flow_pivots: List[Tuple]
    ) -> float:
        """Calculate angle between price and flow trends"""
        if len(price_pivots) < 2 or len(flow_pivots) < 2:
            return 0.0
        
        # Calculate price trend angle
        price_slope = (price_pivots[-1][1] - price_pivots[-2][1]) / \
                      (price_pivots[-1][0] - price_pivots[-2][0] + 1)
        price_angle = np.arctan(price_slope)
        
        # Calculate flow trend angle
        flow_slope = (flow_pivots[-1][1] - flow_pivots[-2][1]) / \
                     (flow_pivots[-1][0] - flow_pivots[-2][0] + 1)
        flow_angle = np.arctan(flow_slope)
        
        # Return angle difference in degrees
        return np.degrees(abs(price_angle - flow_angle))
    
    def _calculate_significance(
        self,
        prices: np.ndarray,
        flows: np.ndarray
    ) -> float:
        """Calculate statistical significance of divergence"""
        if len(prices) != len(flows) or len(prices) < 10:
            return 0.5
        
        # Perform correlation test
        correlation, p_value = stats.pearsonr(prices, flows)
        
        # Lower correlation with significant p-value indicates divergence
        if p_value < 0.05 and abs(correlation) < 0.5:
            return 1 - p_value
        
        return p_value
    
    def _is_accumulation(self, tx: Dict) -> bool:
        """Check if transaction represents accumulation"""
        # From exchange to private wallet (accumulation)
        if tx.get('from_type') == 'exchange' and tx.get('to_type') in ['unknown', 'wallet']:
            return True
        
        # Large buys from DEX
        if 'dex' in str(tx.get('from_type', '')).lower():
            return True
        
        # To known accumulation addresses
        accumulation_addresses = self.config.get('accumulation_addresses', [])
        if tx.get('to') in accumulation_addresses:
            return True
        
        return False
    
    def _is_distribution(self, tx: Dict) -> bool:
        """Check if transaction represents distribution"""
        # From private wallet to exchange (distribution)
        if tx.get('from_type') in ['unknown', 'wallet'] and tx.get('to_type') == 'exchange':
            return True
        
        # Large sells to DEX
        if 'dex' in str(tx.get('to_type', '')).lower():
            return True
        
        # From known distribution addresses
        distribution_addresses = self.config.get('distribution_addresses', [])
        if tx.get('from') in distribution_addresses:
            return True
        
        return False
    
    def _calculate_accumulation_index(self, flows: pd.DataFrame) -> np.ndarray:
        """Calculate Wyckoff-style accumulation/distribution index"""
        if 'smart_money_flow' not in flows.columns:
            return np.zeros(len(flows))
        
        # Normalize flows
        flows_norm = flows['smart_money_flow'].values
        if flows_norm.std() > 0:
            flows_norm = (flows_norm - flows_norm.mean()) / flows_norm.std()
        
        # Calculate cumulative flow with decay
        decay_factor = 0.95
        accumulation = np.zeros(len(flows_norm))
        
        for i in range(len(flows_norm)):
            if i == 0:
                accumulation[i] = flows_norm[i]
            else:
                accumulation[i] = accumulation[i-1] * decay_factor + flows_norm[i]
        
        # Normalize to -1 to 1
        if accumulation.std() > 0:
            accumulation = np.tanh(accumulation / accumulation.std())
        
        return accumulation
    
    async def _create_divergence_signal(
        self,
        symbol: str,
        pattern: DivergencePattern,
        price_data: pd.DataFrame,
        flow_data: pd.DataFrame,
        whale_txs: List[Dict],
        exchange_flows: Dict,
        timeframe: str
    ) -> Optional[SmartMoneySignal]:
        """Create divergence signal from pattern"""
        try:
            # Calculate signal strength based on pattern
            strength = self._calculate_signal_strength(pattern, price_data, flow_data)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(
                pattern, flow_data, exchange_flows, whale_txs
            )
            
            # Get recent price change
            price_change = 0
            if len(price_data) > 1:
                price_change = ((price_data['close'].iloc[-1] - price_data['close'].iloc[-2]) / 
                               price_data['close'].iloc[-2] * 100)
            
            # Calculate volume ratio
            smart_volume = sum([tx['amount_usd'] for tx in whale_txs[:10]])
            total_volume = price_data['volume'].tail(24).sum() if 'volume' in price_data else 1
            volume_ratio = smart_volume / max(total_volume, 1)
            
            # Get accumulation score
            accumulation_score = flow_data['accumulation_index'].iloc[-1] if \
                'accumulation_index' in flow_data else 0
            
            # Build signal components
            components = {
                'pattern_strength': pattern.statistical_significance,
                'divergence_angle': pattern.divergence_angle / 90.0,  # Normalize
                'exchange_flow_signal': min(abs(exchange_flows['net_flow']) / 10_000_000, 1),
                'whale_activity': min(len(whale_txs) / 50, 1),
                'volume_divergence': min(volume_ratio, 1)
            }
            
            # Create signal
            signal = SmartMoneySignal(
                timestamp=datetime.now(),
                symbol=symbol,
                divergence_type=DivergenceType(pattern.pattern_type),
                strength=strength,
                confidence=confidence,
                smart_money_flow=flow_data['smart_money_flow'].iloc[-1],
                retail_flow=flow_data['retail_flow'].iloc[-1] if 'retail_flow' in flow_data else 0,
                price_change=price_change,
                volume_ratio=volume_ratio,
                accumulation_score=accumulation_score,
                timeframe=timeframe,
                whale_transactions=whale_txs[:5],  # Top 5 whale txs
                exchange_flows=exchange_flows,
                on_chain_metrics={
                    'pattern_type': pattern.pattern_type,
                    'expected_resolution': pattern.expected_resolution,
                    'divergence_duration': (pattern.end_time - pattern.start_time).total_seconds() / 3600
                },
                signal_components=components,
                metadata={
                    'pattern_confidence_interval': pattern.confidence_interval,
                    'statistical_significance': pattern.statistical_significance
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating divergence signal: {e}")
            return None
    
    def _calculate_signal_strength(
        self,
        pattern: DivergencePattern,
        price_data: pd.DataFrame,
        flow_data: pd.DataFrame
    ) -> float:
        """Calculate overall signal strength"""
        strength_factors = []
        
        # Pattern significance
        strength_factors.append(pattern.statistical_significance)
        
        # Divergence angle (stronger divergence = higher strength)
        angle_strength = min(pattern.divergence_angle / 45.0, 1.0)
        strength_factors.append(angle_strength)
        
        # Flow momentum
        if 'flow_momentum' in flow_data:
            momentum = abs(flow_data['flow_momentum'].iloc[-1])
            strength_factors.append(min(momentum * 10, 1.0))
        
        # Volume confirmation
        if 'volume' in price_data:
            recent_vol = price_data['volume'].tail(5).mean()
            avg_vol = price_data['volume'].tail(20).mean()
            vol_ratio = recent_vol / max(avg_vol, 1)
            strength_factors.append(min(vol_ratio, 1.0))
        
        # Calculate weighted average
        return np.mean(strength_factors)
    
    def _calculate_confidence(
        self,
        pattern: DivergencePattern,
        flow_data: pd.DataFrame,
        exchange_flows: Dict,
        whale_txs: List[Dict]
    ) -> float:
        """Calculate signal confidence"""
        confidence_factors = []
        
        # Pattern confidence interval midpoint
        ci_mid = np.mean(pattern.confidence_interval)
        confidence_factors.append(ci_mid)
        
        # Whale transaction consistency
        if whale_txs:
            # Check if whale transactions align with pattern
            if pattern.expected_resolution == 'bullish':
                accumulation_txs = sum(1 for tx in whale_txs if self._is_accumulation(tx))
                whale_confidence = accumulation_txs / max(len(whale_txs), 1)
            else:
                distribution_txs = sum(1 for tx in whale_txs if self._is_distribution(tx))
                whale_confidence = distribution_txs / max(len(whale_txs), 1)
            confidence_factors.append(whale_confidence)
        
        # Exchange flow alignment
        if exchange_flows['net_flow'] != 0:
            if (pattern.expected_resolution == 'bullish' and exchange_flows['net_flow'] > 0) or \
               (pattern.expected_resolution == 'bearish' and exchange_flows['net_flow'] < 0):
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.3)
        
        # Flow consistency
        if 'smart_money_flow' in flow_data:
            flow_std = flow_data['smart_money_flow'].tail(10).std()
            flow_mean = abs(flow_data['smart_money_flow'].tail(10).mean())
            if flow_mean > 0:
                consistency = 1 - min(flow_std / flow_mean, 1)
                confidence_factors.append(consistency)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _get_contract_address(self, symbol: str) -> str:
        """Get contract address for symbol"""
        # This would be loaded from configuration
        contracts = {
            'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7',
            'USDC': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
            'LINK': '0x514910771af9ca656af840dff83e8264ecf986ca',
            'UNI': '0x1f9840a85d5af5bf1d1762f925bdaddc4201f984',
            # Add more contract addresses
        }
        return contracts.get(symbol.upper(), '')
    
    async def _fetch_bitquery_metrics(self, symbol: str) -> Dict:
        """Fetch additional metrics from Bitquery"""
        metrics = {}
        
        if not self.api_keys['bitquery']:
            return metrics
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://graphql.bitquery.io"
                headers = {
                    'X-API-KEY': self.api_keys['bitquery'],
                    'Content-Type': 'application/json'
                }
                
                # GraphQL query for DEX trades
                query = """
                query ($network: EthereumNetwork!, $token: String!, $from: ISO8601DateTime!) {
                    ethereum(network: $network) {
                        dexTrades(
                            baseCurrency: {is: $token}
                            after: $from
                        ) {
                            count
                            tradeAmount(in: USD)
                            buyers: count(uniq: buyers)
                            sellers: count(uniq: sellers)
                        }
                    }
                }
                """
                
                variables = {
                    'network': 'ethereum',
                    'token': self._get_contract_address(symbol),
                    'from': (datetime.now() - timedelta(days=1)).isoformat()
                }
                
                payload = {
                    'query': query,
                    'variables': variables
                }
                
                async with session.post(url, headers=headers, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'data' in data and 'ethereum' in data['data']:
                            dex_data = data['data']['ethereum']['dexTrades']
                            if dex_data:
                                metrics['dex_volume'] = dex_data[0].get('tradeAmount', 0)
                                metrics['unique_buyers'] = dex_data[0].get('buyers', 0)
                                metrics['unique_sellers'] = dex_data[0].get('sellers', 0)
                                metrics['buy_sell_ratio'] = metrics['unique_buyers'] / \
                                    max(metrics['unique_sellers'], 1)
                                    
        except Exception as e:
            logger.warning(f"Bitquery metrics fetch error: {e}")
        
        return metrics
    
    def _update_performance_metrics(self, signals: List[SmartMoneySignal]):
        """Update detector performance metrics"""
        for signal in signals:
            # Track signal accuracy (would need outcome data)
            symbol = signal.symbol
            if symbol not in self.signal_accuracy:
                self.signal_accuracy[symbol] = {
                    'total': 0,
                    'correct': 0,
                    'accuracy': 0.0
                }
            
            self.signal_accuracy[symbol]['total'] += 1
            
            # Update latency
            self.detection_latency.append(
                (datetime.now() - signal.timestamp).total_seconds()
            )
            
            # Keep only recent latency measurements
            if len(self.detection_latency) > 100:
                self.detection_latency = self.detection_latency[-100:]
    
    async def get_historical_divergences(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[SmartMoneySignal]:
        """Get historical divergence signals for backtesting"""
        # This would fetch historical data and run detection
        # For now, return empty list
        return []
    
    def get_performance_report(self) -> Dict:
        """Get performance metrics report"""
        return {
            'signal_accuracy': self.signal_accuracy,
            'false_positive_rate': self.false_positive_rate,
            'avg_detection_latency': np.mean(self.detection_latency) if self.detection_latency else 0,
            'total_signals_generated': len(self.signal_buffer),
            'active_patterns': len(self.divergence_patterns)
        }


class SmartMoneyDivergenceRL:
    """RL integration for smart money divergence signals"""
    
    def __init__(self, detector: SmartMoneyDivergenceDetector):
        self.detector = detector
        
    def get_features(self, signals: List[SmartMoneySignal]) -> np.ndarray:
        """Extract RL features from divergence signals"""
        if not signals:
            return np.zeros(15)
        
        # Take most recent signal
        signal = signals[0]
        
        features = [
            # Signal strength and confidence
            signal.strength,
            signal.confidence,
            
            # Flow metrics
            np.sign(signal.smart_money_flow) * np.log1p(abs(signal.smart_money_flow)),
            np.sign(signal.retail_flow) * np.log1p(abs(signal.retail_flow)),
            signal.volume_ratio,
            signal.accumulation_score,
            
            # Price action
            signal.price_change / 100.0,
            
            # Exchange flows
            signal.exchange_flows.get('flow_ratio', 1.0),
            np.sign(signal.exchange_flows.get('net_flow', 0)),
            
            # Divergence type encoding
            1.0 if signal.divergence_type == DivergenceType.BULLISH_ACCUMULATION else 0.0,
            1.0 if signal.divergence_type == DivergenceType.BEARISH_DISTRIBUTION else 0.0,
            
            # Component signals
            signal.signal_components.get('pattern_strength', 0),
            signal.signal_components.get('divergence_angle', 0),
            signal.signal_components.get('whale_activity', 0),
            signal.signal_components.get('volume_divergence', 0)
        ]
        
        return np.array(features)
    
    def get_action_mask(self, signals: List[SmartMoneySignal]) -> np.ndarray:
        """Get valid actions based on divergence signals"""
        # Allow all actions by default
        mask = np.ones(3)  # [hold, buy, sell]
        
        if signals:
            signal = signals[0]
            
            # Strong bullish divergence: discourage selling
            if (signal.divergence_type == DivergenceType.BULLISH_ACCUMULATION and
                signal.confidence > 0.7):
                mask[2] = 0.1  # Reduce sell probability
            
            # Strong bearish divergence: discourage buying
            elif (signal.divergence_type == DivergenceType.BEARISH_DISTRIBUTION and
                  signal.confidence > 0.7):
                mask[1] = 0.1  # Reduce buy probability
        
        return mask