"""
Data Connector - Integration with DataAggregatorService
=====================================================

This module provides seamless integration with the existing DataAggregatorService,
enabling the RL system to access real-time market data, on-chain analytics,
whale movements, and other data sources required for intelligent trading decisions.

Features:
- Real-time market data fetching
- On-chain analytics integration
- Whale movement tracking
- Smart money flow analysis
- Data caching and preprocessing
- Feature engineering for RL models
- Asynchronous data pipeline
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class DataSource(Enum):
    TRADING_SERVICE = "trading_service"
    DATA_AGGREGATOR = "data_aggregator"
    MARKET_DATA = "market_data"
    ON_CHAIN = "on_chain"
    WHALE_ALERTS = "whale_alerts"
    FUNDING_RATES = "funding_rates"

@dataclass
class DataRequest:
    """Data request specification"""
    symbol: str
    data_sources: List[DataSource]
    time_range: Tuple[datetime, datetime]
    resolution: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d
    include_indicators: bool = True
    include_features: bool = True

@dataclass
class MarketDataPoint:
    """Single market data point"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)

@dataclass
class AggregatedMarketData:
    """Comprehensive market data with all sources"""
    symbol: str
    timeframe: str
    data_points: List[MarketDataPoint]
    on_chain_data: Optional[Dict[str, Any]] = None
    whale_alerts: Optional[List[Dict[str, Any]]] = None
    funding_rates: Optional[List[Dict[str, Any]]] = None
    smart_money_flows: Optional[List[Dict[str, Any]]] = None
    liquidations: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataConnector:
    """
    Connector for integrating with DataAggregatorService and other data sources
    
    Provides:
    - Real-time market data access
    - Multi-source data aggregation
    - Feature engineering for RL models
    - Data caching and preprocessing
    - Asynchronous data pipeline
    """
    
    def __init__(self, data_aggregator_url: str, trading_service_url: str = None):
        self.data_aggregator_url = data_aggregator_url
        self.trading_service_url = trading_service_url or "http://backend:3000"
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Data cache
        self.cache = {}
        self.cache_ttl = {}
        self.default_cache_duration = 300  # 5 minutes
        
        # Feature engineering configuration
        self.feature_config = {
            'technical_indicators': ['rsi', 'macd', 'bollinger_bands', 'sma', 'ema', 'atr'],
            'price_features': ['returns', 'volatility', 'price_momentum', 'volume_momentum'],
            'on_chain_features': ['whale_activity', 'smart_money_sentiment', 'funding_bias'],
            'market_structure': ['bid_ask_spread', 'order_flow_imbalance', 'market_impact']
        }
        
        # Performance tracking
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'cached_requests': 0,
            'avg_response_time': 0.0
        }
        
        logger.info(f"DataConnector initialized - DataAggregator: {self.data_aggregator_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure HTTP session is available"""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
    
    def _cache_key(self, symbol: str, data_type: str, params: Dict[str, Any]) -> str:
        """Generate cache key"""
        params_str = json.dumps(params, sort_keys=True)
        return f"{symbol}_{data_type}_{hash(params_str)}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_ttl:
            return False
        return datetime.now() < self.cache_ttl[cache_key]
    
    def _set_cache(self, cache_key: str, data: Any, ttl_seconds: int = None):
        """Set cached data with TTL"""
        ttl_seconds = ttl_seconds or self.default_cache_duration
        self.cache[cache_key] = data
        self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=ttl_seconds)
    
    def _get_cache(self, cache_key: str) -> Optional[Any]:
        """Get cached data if valid"""
        if self._is_cache_valid(cache_key):
            self.request_stats['cached_requests'] += 1
            return self.cache[cache_key]
        return None
    
    async def get_recent_market_data(self, symbol: str, window_hours: int = 24,
                                   resolution: str = "1h") -> pd.DataFrame:
        """Get recent market data for a symbol"""
        cache_key = self._cache_key(symbol, "market_data", {
            "window_hours": window_hours,
            "resolution": resolution
        })
        
        # Check cache first
        cached_data = self._get_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            start_time = time.time()
            await self._ensure_session()
            
            # Call DataAggregatorService for comprehensive market data
            url = f"{self.data_aggregator_url}/api/data/market/{symbol}"
            params = {
                "window_hours": window_hours,
                "resolution": resolution,
                "include_indicators": True
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to DataFrame
                    df = self._convert_market_data_to_dataframe(data)
                    
                    # Add technical indicators if not present
                    df = self._add_technical_indicators(df)
                    
                    # Cache the result
                    self._set_cache(cache_key, df, ttl_seconds=300)
                    
                    # Update stats
                    response_time = (time.time() - start_time) * 1000
                    self._update_request_stats(True, response_time)
                    
                    return df
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            self._update_request_stats(False, 0)
            
            # Return empty DataFrame as fallback
            return self._create_empty_market_dataframe()
    
    async def get_aggregated_data(self, symbol: str, data_sources: List[DataSource] = None,
                                window_hours: int = 24) -> AggregatedMarketData:
        """Get comprehensive aggregated data for RL model"""
        if data_sources is None:
            data_sources = [DataSource.MARKET_DATA, DataSource.ON_CHAIN, 
                          DataSource.WHALE_ALERTS, DataSource.FUNDING_RATES]
        
        cache_key = self._cache_key(symbol, "aggregated", {
            "data_sources": [ds.value for ds in data_sources],
            "window_hours": window_hours
        })
        
        # Check cache first
        cached_data = self._get_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            start_time = time.time()
            await self._ensure_session()
            
            # Call DataAggregatorService aggregation endpoint
            url = f"{self.data_aggregator_url}/api/data/aggregate"
            payload = {
                "symbols": [symbol],
                "options": {
                    "includeOnchain": DataSource.ON_CHAIN in data_sources,
                    "includeFunding": DataSource.FUNDING_RATES in data_sources,
                    "includeWhales": DataSource.WHALE_ALERTS in data_sources,
                    "includeSmartMoney": True,
                    "includeLiquidations": True
                }
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Get market data
                    market_df = await self.get_recent_market_data(symbol, window_hours)
                    market_points = self._dataframe_to_market_points(market_df, symbol)
                    
                    # Create aggregated data object
                    aggregated_data = AggregatedMarketData(
                        symbol=symbol,
                        timeframe=f"{window_hours}h",
                        data_points=market_points,
                        on_chain_data=data.get('onchain'),
                        whale_alerts=data.get('whales'),
                        funding_rates=data.get('funding'),
                        smart_money_flows=data.get('smartMoney'),
                        liquidations=data.get('liquidations'),
                        metadata={
                            'timestamp': datetime.now().isoformat(),
                            'sources': data.get('metadata', {}).get('sources', []),
                            'reliability': data.get('metadata', {}).get('reliability', 0)
                        }
                    )
                    
                    # Cache the result
                    self._set_cache(cache_key, aggregated_data, ttl_seconds=600)
                    
                    # Update stats
                    response_time = (time.time() - start_time) * 1000
                    self._update_request_stats(True, response_time)
                    
                    return aggregated_data
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            logger.error(f"Failed to get aggregated data for {symbol}: {e}")
            self._update_request_stats(False, 0)
            
            # Return minimal data as fallback
            market_df = await self.get_recent_market_data(symbol, window_hours)
            market_points = self._dataframe_to_market_points(market_df, symbol)
            
            return AggregatedMarketData(
                symbol=symbol,
                timeframe=f"{window_hours}h",
                data_points=market_points,
                metadata={'error': str(e)}
            )
    
    def _convert_market_data_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert API response to pandas DataFrame"""
        if 'data' not in data:
            return self._create_empty_market_dataframe()
        
        records = data['data']
        if not records:
            return self._create_empty_market_dataframe()
        
        df = pd.DataFrame(records)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'timestamp':
                    df[col] = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                          periods=len(df), freq='1H')
                else:
                    df[col] = 0.0
        
        # Convert timestamp to datetime
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def _create_empty_market_dataframe(self) -> pd.DataFrame:
        """Create empty market data DataFrame"""
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to market data"""
        if len(df) < 20:  # Not enough data for indicators
            return df
        
        try:
            # RSI
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # Moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_mean = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = bb_mean + (bb_std * 2)
            df['bb_lower'] = bb_mean - (bb_std * 2)
            df['bb_position'] = (df['close'] - bb_lower) / (df['bb_upper'] - df['bb_lower'])
            
            # ATR
            df['atr'] = self._calculate_atr(df)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price momentum
            df['price_momentum_5'] = df['close'].pct_change(5)
            df['price_momentum_10'] = df['close'].pct_change(10)
            
            # Volatility
            df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window).mean()
        return atr
    
    def _dataframe_to_market_points(self, df: pd.DataFrame, symbol: str) -> List[MarketDataPoint]:
        """Convert DataFrame to list of MarketDataPoint objects"""
        points = []
        
        for _, row in df.iterrows():
            # Extract indicators
            indicators = {}
            indicator_columns = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 
                               'ema_12', 'ema_26', 'bb_position', 'atr']
            for col in indicator_columns:
                if col in row and not pd.isna(row[col]):
                    indicators[col] = float(row[col])
            
            # Extract features
            features = {}
            feature_columns = ['price_momentum_5', 'price_momentum_10', 'volatility_20', 'volume_ratio']
            for col in feature_columns:
                if col in row and not pd.isna(row[col]):
                    features[col] = float(row[col])
            
            point = MarketDataPoint(
                timestamp=row['timestamp'],
                symbol=symbol,
                open=float(row.get('open', 0)),
                high=float(row.get('high', 0)),
                low=float(row.get('low', 0)),
                close=float(row.get('close', 0)),
                volume=float(row.get('volume', 0)),
                indicators=indicators,
                features=features
            )
            points.append(point)
        
        return points
    
    async def get_features_for_rl(self, symbol: str, lookback_hours: int = 24) -> np.ndarray:
        """Get feature vector optimized for RL model input"""
        try:
            # Get aggregated data
            aggregated_data = await self.get_aggregated_data(symbol, window_hours=lookback_hours)
            
            if not aggregated_data.data_points:
                return np.zeros(50)  # Return default feature vector
            
            # Extract latest market point
            latest_point = aggregated_data.data_points[-1]
            
            # Build feature vector
            features = []
            
            # Price features (normalized)
            price_features = [
                latest_point.close / 50000.0,  # Normalize price (assuming ~$50k base)
                latest_point.volume / 1e9,     # Normalize volume
                (latest_point.high - latest_point.low) / latest_point.close,  # Price range
                latest_point.close / latest_point.open - 1.0,  # Intraday return
            ]
            features.extend(price_features)
            
            # Technical indicators (already mostly normalized)
            indicator_features = [
                latest_point.indicators.get('rsi', 50.0) / 100.0,
                np.tanh(latest_point.indicators.get('macd', 0.0) / 1000.0),
                latest_point.indicators.get('bb_position', 0.5),
                latest_point.indicators.get('atr', 0.0) / latest_point.close,
            ]
            features.extend(indicator_features)
            
            # Price momentum features
            momentum_features = [
                np.tanh(latest_point.features.get('price_momentum_5', 0.0) * 100),
                np.tanh(latest_point.features.get('price_momentum_10', 0.0) * 100),
                np.tanh(latest_point.features.get('volatility_20', 0.0) * 100),
                np.tanh(latest_point.features.get('volume_ratio', 1.0) - 1.0),
            ]
            features.extend(momentum_features)
            
            # On-chain features
            on_chain_features = self._extract_on_chain_features(aggregated_data.on_chain_data)
            features.extend(on_chain_features)
            
            # Whale activity features
            whale_features = self._extract_whale_features(aggregated_data.whale_alerts)
            features.extend(whale_features)
            
            # Funding rate features
            funding_features = self._extract_funding_features(aggregated_data.funding_rates)
            features.extend(funding_features)
            
            # Smart money features
            smart_money_features = self._extract_smart_money_features(aggregated_data.smart_money_flows)
            features.extend(smart_money_features)
            
            # Liquidation features
            liquidation_features = self._extract_liquidation_features(aggregated_data.liquidations)
            features.extend(liquidation_features)
            
            # Pad or truncate to expected size (50 features)
            target_size = 50
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            elif len(features) > target_size:
                features = features[:target_size]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting RL features for {symbol}: {e}")
            return np.zeros(50, dtype=np.float32)
    
    def _extract_on_chain_features(self, on_chain_data: Optional[List[Dict[str, Any]]]) -> List[float]:
        """Extract features from on-chain data"""
        if not on_chain_data:
            return [0.0, 0.0, 0.0, 0.0]
        
        try:
            # Aggregate on-chain metrics
            total_activity = len(on_chain_data)
            avg_risk_score = np.mean([item.get('riskScore', 0) for item in on_chain_data])
            total_balance = sum([item.get('balance', 0) for item in on_chain_data])
            
            # Recent activity (last 24 hours)
            recent_activity = 0
            cutoff_time = datetime.now() - timedelta(hours=24)
            for item in on_chain_data:
                try:
                    last_activity = datetime.fromisoformat(item.get('lastActivity', '').replace('Z', '+00:00'))
                    if last_activity > cutoff_time:
                        recent_activity += 1
                except:
                    pass
            
            features = [
                np.tanh(total_activity / 100.0),  # Normalize activity count
                avg_risk_score / 100.0,          # Risk score (0-1)
                np.tanh(total_balance / 1e6),     # Normalize balance
                np.tanh(recent_activity / 10.0)   # Recent activity
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting on-chain features: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _extract_whale_features(self, whale_alerts: Optional[List[Dict[str, Any]]]) -> List[float]:
        """Extract features from whale alerts"""
        if not whale_alerts:
            return [0.0, 0.0, 0.0, 0.0]
        
        try:
            # Recent whale activity (last 4 hours)
            cutoff_time = datetime.now() - timedelta(hours=4)
            recent_whales = []
            
            for whale in whale_alerts:
                try:
                    timestamp = datetime.fromisoformat(whale.get('timestamp', '').replace('Z', '+00:00'))
                    if timestamp > cutoff_time:
                        recent_whales.append(whale)
                except:
                    pass
            
            if not recent_whales:
                return [0.0, 0.0, 0.0, 0.0]
            
            # Calculate whale sentiment
            inflow_value = sum([w.get('amountUsd', 0) for w in recent_whales if w.get('type') == 'exchange_inflow'])
            outflow_value = sum([w.get('amountUsd', 0) for w in recent_whales if w.get('type') == 'exchange_outflow'])
            total_value = inflow_value + outflow_value
            
            whale_sentiment = (outflow_value - inflow_value) / total_value if total_value > 0 else 0.0
            whale_activity = len(recent_whales)
            avg_confidence = np.mean([w.get('confidence', 0.5) for w in recent_whales])
            
            features = [
                np.tanh(whale_sentiment),                    # Whale sentiment (-1 to 1)
                np.tanh(whale_activity / 10.0),            # Activity level
                avg_confidence,                             # Average confidence
                np.tanh(total_value / 1e7)                 # Total whale volume
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting whale features: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _extract_funding_features(self, funding_rates: Optional[List[Dict[str, Any]]]) -> List[float]:
        """Extract features from funding rates"""
        if not funding_rates:
            return [0.0, 0.0, 0.0]
        
        try:
            # Get most recent funding rate
            latest_funding = funding_rates[-1] if funding_rates else {}
            
            current_rate = latest_funding.get('rate', 0.0)
            historical_avg = latest_funding.get('historicalAverage', 0.0)
            predicted_rate = latest_funding.get('predictedRate', current_rate)
            
            features = [
                np.tanh(current_rate * 10000),              # Current funding rate (basis points)
                np.tanh((current_rate - historical_avg) * 10000),  # Deviation from average
                np.tanh(predicted_rate * 10000)             # Predicted next rate
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting funding features: {e}")
            return [0.0, 0.0, 0.0]
    
    def _extract_smart_money_features(self, smart_money_flows: Optional[List[Dict[str, Any]]]) -> List[float]:
        """Extract features from smart money flows"""
        if not smart_money_flows:
            return [0.0, 0.0, 0.0]
        
        try:
            # Recent smart money activity (last 6 hours)
            cutoff_time = datetime.now() - timedelta(hours=6)
            recent_flows = []
            
            for flow in smart_money_flows:
                try:
                    timestamp = datetime.fromisoformat(flow.get('timestamp', '').replace('Z', '+00:00'))
                    if timestamp > cutoff_time:
                        recent_flows.append(flow)
                except:
                    pass
            
            if not recent_flows:
                return [0.0, 0.0, 0.0]
            
            # Calculate smart money sentiment
            buy_volume = sum([f.get('amountUsd', 0) for f in recent_flows if f.get('action') == 'buy'])
            sell_volume = sum([f.get('amountUsd', 0) for f in recent_flows if f.get('action') == 'sell'])
            total_volume = buy_volume + sell_volume
            
            smart_money_sentiment = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0
            avg_confidence = np.mean([f.get('confidence', 0.5) for f in recent_flows])
            
            features = [
                np.tanh(smart_money_sentiment),              # Smart money sentiment
                avg_confidence,                             # Average confidence
                np.tanh(total_volume / 1e6)                 # Total smart money volume
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting smart money features: {e}")
            return [0.0, 0.0, 0.0]
    
    def _extract_liquidation_features(self, liquidations: Optional[List[Dict[str, Any]]]) -> List[float]:
        """Extract features from liquidation data"""
        if not liquidations:
            return [0.0, 0.0, 0.0]
        
        try:
            # Recent liquidations (last 1 hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_liquidations = []
            
            for liq in liquidations:
                try:
                    timestamp = datetime.fromisoformat(liq.get('timestamp', '').replace('Z', '+00:00'))
                    if timestamp > cutoff_time:
                        recent_liquidations.append(liq)
                except:
                    pass
            
            if not recent_liquidations:
                return [0.0, 0.0, 0.0]
            
            # Calculate liquidation metrics
            long_liquidations = sum([l.get('amountUsd', 0) for l in recent_liquidations if l.get('side') == 'long'])
            short_liquidations = sum([l.get('amountUsd', 0) for l in recent_liquidations if l.get('side') == 'short'])
            total_liquidations = long_liquidations + short_liquidations
            
            liquidation_bias = (short_liquidations - long_liquidations) / total_liquidations if total_liquidations > 0 else 0.0
            
            features = [
                np.tanh(liquidation_bias),                   # Liquidation bias (shorts vs longs)
                np.tanh(total_liquidations / 1e6),          # Total liquidation volume
                np.tanh(len(recent_liquidations) / 100.0)   # Liquidation frequency
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting liquidation features: {e}")
            return [0.0, 0.0, 0.0]
    
    def _update_request_stats(self, success: bool, response_time: float):
        """Update request statistics"""
        self.request_stats['total_requests'] += 1
        if success:
            self.request_stats['successful_requests'] += 1
        
        # Update average response time
        total_successful = self.request_stats['successful_requests']
        if total_successful > 0:
            current_avg = self.request_stats['avg_response_time']
            self.request_stats['avg_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
    
    async def health_check(self) -> str:
        """Check health of DataConnector"""
        try:
            await self._ensure_session()
            
            # Test DataAggregatorService health
            url = f"{self.data_aggregator_url}/health"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return "healthy"
                else:
                    return "degraded"
                    
        except Exception as e:
            logger.error(f"DataConnector health check failed: {e}")
            return "unhealthy"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_requests = self.request_stats['total_requests']
        success_rate = (self.request_stats['successful_requests'] / total_requests) if total_requests > 0 else 0.0
        cache_hit_rate = (self.request_stats['cached_requests'] / total_requests) if total_requests > 0 else 0.0
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.request_stats['successful_requests'],
            'cached_requests': self.request_stats['cached_requests'],
            'success_rate': success_rate,
            'cache_hit_rate': cache_hit_rate,
            'avg_response_time_ms': self.request_stats['avg_response_time'],
            'cache_size': len(self.cache)
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
        
        # Clear cache
        self.cache.clear()
        self.cache_ttl.clear()
        
        logger.info("DataConnector cleanup completed")


# Factory function
def create_data_connector(data_aggregator_url: str, trading_service_url: str = None) -> DataConnector:
    """Create DataConnector instance"""
    return DataConnector(data_aggregator_url, trading_service_url)


if __name__ == "__main__":
    # Test the data connector
    async def test_connector():
        async with create_data_connector("http://localhost:3000", "http://localhost:3000") as connector:
            # Test market data fetching
            market_data = await connector.get_recent_market_data("BTC", window_hours=24)
            print(f"Market data shape: {market_data.shape}")
            
            # Test aggregated data
            aggregated = await connector.get_aggregated_data("BTC")
            print(f"Aggregated data: {len(aggregated.data_points)} points")
            
            # Test RL features
            features = await connector.get_features_for_rl("BTC")
            print(f"RL features: {features.shape}")
            
            # Get performance stats
            stats = connector.get_performance_stats()
            print(f"Performance stats: {json.dumps(stats, indent=2)}")
    
    # Run test
    asyncio.run(test_connector())