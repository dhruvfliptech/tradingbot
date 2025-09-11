"""
On-Chain Analyzer - Blockchain Data Analysis
=============================================

Advanced on-chain metrics analysis for identifying smart money movements
and predicting market trends based on blockchain fundamentals.

Key Features:
- Network health metrics
- Token velocity and circulation
- Holder distribution analysis
- Smart contract interaction patterns
- Cross-chain flow analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import aiohttp
import json

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of on-chain metrics"""
    NETWORK_VALUE = "network_value"  # NVT, NVTS ratios
    CIRCULATION = "circulation"  # Token velocity, supply dynamics
    HOLDER_DISTRIBUTION = "holder_distribution"  # Concentration metrics
    MINING_METRICS = "mining_metrics"  # Hash rate, difficulty
    DEFI_METRICS = "defi_metrics"  # TVL, lending rates
    DERIVATIVES = "derivatives"  # Futures, options data


class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"


@dataclass
class OnChainMetrics:
    """Complete on-chain metrics snapshot"""
    timestamp: datetime
    symbol: str
    
    # Network metrics
    nvt_ratio: float  # Network Value to Transactions
    nvts_ratio: float  # NVT Signal (smoothed)
    active_addresses: int
    transaction_count: int
    transaction_volume: float
    average_fee: float
    
    # Supply metrics
    circulating_supply: float
    total_supply: float
    inflation_rate: float
    velocity: float  # Transaction volume / Market cap
    
    # Holder metrics
    unique_addresses: int
    top_10_concentration: float  # % held by top 10
    top_100_concentration: float  # % held by top 100
    retail_holders: int  # Addresses < $1000
    whale_holders: int  # Addresses > $1M
    
    # Exchange metrics
    exchange_balance: float
    exchange_inflow: float
    exchange_outflow: float
    exchange_netflow: float
    
    # Mining/Staking metrics
    hash_rate: Optional[float] = None
    difficulty: Optional[float] = None
    miner_revenue: Optional[float] = None
    staking_ratio: Optional[float] = None
    
    # DeFi metrics
    tvl: Optional[float] = None  # Total Value Locked
    lending_rate: Optional[float] = None
    borrowing_rate: Optional[float] = None
    defi_dominance: Optional[float] = None  # % in DeFi
    
    # Derivative metrics
    futures_volume: Optional[float] = None
    futures_oi: Optional[float] = None  # Open Interest
    funding_rate: Optional[float] = None
    options_volume: Optional[float] = None


@dataclass
class OnChainSignal:
    """On-chain analysis signal"""
    timestamp: datetime
    symbol: str
    signal_type: str
    strength: SignalStrength
    direction: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0-1
    metrics: Dict[str, float]
    interpretation: str
    historical_accuracy: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class OnChainAnalyzer:
    """
    Advanced on-chain data analyzer for blockchain metrics.
    
    Provides comprehensive analysis of blockchain fundamentals to identify
    smart money movements and predict market trends.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize on-chain analyzer"""
        self.config = config or self._default_config()
        
        # API configurations
        self.api_keys = {
            'glassnode': self.config.get('glassnode_api_key', ''),
            'santiment': self.config.get('santiment_api_key', ''),
            'intotheblock': self.config.get('intotheblock_api_key', ''),
            'cryptoquant': self.config.get('cryptoquant_api_key', ''),
            'messari': self.config.get('messari_api_key', '')
        }
        
        # Analysis parameters
        self.lookback_periods = {
            '1h': 24,
            '1d': 30,
            '1w': 12
        }
        
        # Historical data cache
        self.metrics_cache: Dict[str, pd.DataFrame] = {}
        self.signal_history: List[OnChainSignal] = []
        
        # Thresholds for signals
        self.signal_thresholds = self._load_signal_thresholds()
        
        # Performance tracking
        self.signal_accuracy = {}
        self.best_indicators = {}
        
        logger.info("On-Chain Analyzer initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'nvt_oversold': 40,  # NVT ratio oversold level
            'nvt_overbought': 100,  # NVT ratio overbought level
            'velocity_high': 2.0,  # High velocity threshold
            'velocity_low': 0.5,  # Low velocity threshold
            'concentration_high': 0.5,  # High concentration threshold
            'exchange_flow_threshold': 1_000_000,  # Significant flow USD
            'confidence_threshold': 0.6,  # Minimum confidence for signals
            'cache_ttl': 3600,  # 1 hour
            'max_api_retries': 3
        }
    
    def _load_signal_thresholds(self) -> Dict:
        """Load signal generation thresholds"""
        return {
            'nvt_ratio': {
                'very_oversold': 30,
                'oversold': 40,
                'neutral_low': 50,
                'neutral_high': 80,
                'overbought': 100,
                'very_overbought': 120
            },
            'exchange_netflow': {
                'strong_outflow': -10_000_000,
                'moderate_outflow': -5_000_000,
                'neutral': (-1_000_000, 1_000_000),
                'moderate_inflow': 5_000_000,
                'strong_inflow': 10_000_000
            },
            'holder_concentration': {
                'very_distributed': 0.2,
                'distributed': 0.3,
                'moderate': 0.5,
                'concentrated': 0.7,
                'very_concentrated': 0.85
            }
        }
    
    async def analyze(
        self,
        symbol: str,
        timeframe: str = '1d'
    ) -> Tuple[OnChainMetrics, List[OnChainSignal]]:
        """
        Perform comprehensive on-chain analysis.
        
        Args:
            symbol: Token symbol to analyze
            timeframe: Analysis timeframe
            
        Returns:
            Current metrics and list of signals
        """
        try:
            # Fetch current metrics
            metrics = await self._fetch_on_chain_metrics(symbol)
            
            # Fetch historical data for trend analysis
            historical = await self._fetch_historical_metrics(symbol, timeframe)
            
            # Generate signals
            signals = []
            
            # Network value signals
            nvt_signal = self._analyze_nvt_ratio(metrics, historical)
            if nvt_signal:
                signals.append(nvt_signal)
            
            # Circulation signals
            velocity_signal = self._analyze_velocity(metrics, historical)
            if velocity_signal:
                signals.append(velocity_signal)
            
            # Holder distribution signals
            distribution_signal = self._analyze_holder_distribution(metrics, historical)
            if distribution_signal:
                signals.append(distribution_signal)
            
            # Exchange flow signals
            flow_signal = self._analyze_exchange_flows(metrics, historical)
            if flow_signal:
                signals.append(flow_signal)
            
            # DeFi signals if available
            if metrics.tvl is not None:
                defi_signal = self._analyze_defi_metrics(metrics, historical)
                if defi_signal:
                    signals.append(defi_signal)
            
            # Mining/Staking signals if available
            if metrics.hash_rate is not None or metrics.staking_ratio is not None:
                consensus_signal = self._analyze_consensus_metrics(metrics, historical)
                if consensus_signal:
                    signals.append(consensus_signal)
            
            # Composite signal
            composite = self._generate_composite_signal(metrics, signals)
            if composite:
                signals.append(composite)
            
            # Store in history
            self.signal_history.extend(signals)
            
            # Update performance tracking
            self._update_performance_metrics(signals)
            
            return metrics, signals
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return self._empty_metrics(symbol), []
    
    async def _fetch_on_chain_metrics(self, symbol: str) -> OnChainMetrics:
        """Fetch current on-chain metrics"""
        metrics_data = {}
        
        # Glassnode API
        if self.api_keys['glassnode']:
            glassnode_data = await self._fetch_glassnode_metrics(symbol)
            metrics_data.update(glassnode_data)
        
        # Santiment API
        if self.api_keys['santiment']:
            santiment_data = await self._fetch_santiment_metrics(symbol)
            metrics_data.update(santiment_data)
        
        # IntoTheBlock API
        if self.api_keys['intotheblock']:
            itb_data = await self._fetch_intotheblock_metrics(symbol)
            metrics_data.update(itb_data)
        
        # CryptoQuant API
        if self.api_keys['cryptoquant']:
            cq_data = await self._fetch_cryptoquant_metrics(symbol)
            metrics_data.update(cq_data)
        
        # Create metrics object with available data
        return OnChainMetrics(
            timestamp=datetime.now(),
            symbol=symbol,
            nvt_ratio=metrics_data.get('nvt_ratio', 0),
            nvts_ratio=metrics_data.get('nvts_ratio', 0),
            active_addresses=metrics_data.get('active_addresses', 0),
            transaction_count=metrics_data.get('transaction_count', 0),
            transaction_volume=metrics_data.get('transaction_volume', 0),
            average_fee=metrics_data.get('average_fee', 0),
            circulating_supply=metrics_data.get('circulating_supply', 0),
            total_supply=metrics_data.get('total_supply', 0),
            inflation_rate=metrics_data.get('inflation_rate', 0),
            velocity=metrics_data.get('velocity', 0),
            unique_addresses=metrics_data.get('unique_addresses', 0),
            top_10_concentration=metrics_data.get('top_10_concentration', 0),
            top_100_concentration=metrics_data.get('top_100_concentration', 0),
            retail_holders=metrics_data.get('retail_holders', 0),
            whale_holders=metrics_data.get('whale_holders', 0),
            exchange_balance=metrics_data.get('exchange_balance', 0),
            exchange_inflow=metrics_data.get('exchange_inflow', 0),
            exchange_outflow=metrics_data.get('exchange_outflow', 0),
            exchange_netflow=metrics_data.get('exchange_netflow', 0),
            hash_rate=metrics_data.get('hash_rate'),
            difficulty=metrics_data.get('difficulty'),
            miner_revenue=metrics_data.get('miner_revenue'),
            staking_ratio=metrics_data.get('staking_ratio'),
            tvl=metrics_data.get('tvl'),
            lending_rate=metrics_data.get('lending_rate'),
            borrowing_rate=metrics_data.get('borrowing_rate'),
            defi_dominance=metrics_data.get('defi_dominance'),
            futures_volume=metrics_data.get('futures_volume'),
            futures_oi=metrics_data.get('futures_oi'),
            funding_rate=metrics_data.get('funding_rate'),
            options_volume=metrics_data.get('options_volume')
        )
    
    async def _fetch_glassnode_metrics(self, symbol: str) -> Dict:
        """Fetch metrics from Glassnode"""
        metrics = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                base_url = "https://api.glassnode.com/v1/metrics"
                headers = {'X-Api-Key': self.api_keys['glassnode']}
                
                # Map symbols to Glassnode format
                asset = symbol.lower()
                if asset not in ['btc', 'eth']:
                    return metrics
                
                # Fetch NVT ratio
                url = f"{base_url}/indicators/nvt"
                params = {'a': asset, 'i': '24h'}
                
                async with session.get(url, params=params, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            metrics['nvt_ratio'] = data[-1]['v'] if isinstance(data, list) else data['v']
                
                # Fetch active addresses
                url = f"{base_url}/addresses/active_count"
                async with session.get(url, params=params, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            metrics['active_addresses'] = int(data[-1]['v'] if isinstance(data, list) else data['v'])
                
                # Fetch exchange netflow
                url = f"{base_url}/transactions/transfers_volume_exchanges_net"
                async with session.get(url, params=params, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            metrics['exchange_netflow'] = data[-1]['v'] if isinstance(data, list) else data['v']
                            
        except Exception as e:
            logger.warning(f"Glassnode API error: {e}")
        
        return metrics
    
    async def _fetch_santiment_metrics(self, symbol: str) -> Dict:
        """Fetch metrics from Santiment"""
        metrics = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                base_url = "https://api.santiment.net/graphql"
                headers = {
                    'Authorization': f'Bearer {self.api_keys["santiment"]}',
                    'Content-Type': 'application/json'
                }
                
                # GraphQL query for metrics
                query = """
                query getMetrics($slug: String!, $from: DateTime!, $to: DateTime!) {
                    getMetric(metric: "daily_active_addresses") {
                        timeseriesData(slug: $slug, from: $from, to: $to, interval: "1d") {
                            datetime
                            value
                        }
                    }
                }
                """
                
                variables = {
                    'slug': self._get_santiment_slug(symbol),
                    'from': (datetime.now() - timedelta(days=2)).isoformat(),
                    'to': datetime.now().isoformat()
                }
                
                payload = {'query': query, 'variables': variables}
                
                async with session.post(base_url, json=payload, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'data' in data:
                            active_data = data['data'].get('getMetric', {}).get('timeseriesData', [])
                            if active_data:
                                metrics['active_addresses'] = active_data[-1]['value']
                                
        except Exception as e:
            logger.warning(f"Santiment API error: {e}")
        
        return metrics
    
    async def _fetch_intotheblock_metrics(self, symbol: str) -> Dict:
        """Fetch metrics from IntoTheBlock"""
        metrics = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                base_url = "https://api.intotheblock.com/v1"
                headers = {'x-api-key': self.api_keys['intotheblock']}
                
                # Get holder composition
                url = f"{base_url}/coins/{symbol.lower()}/holder-composition"
                
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'data' in data:
                            metrics['whale_holders'] = data['data'].get('whales', 0)
                            metrics['retail_holders'] = data['data'].get('retail', 0)
                
                # Get concentration
                url = f"{base_url}/coins/{symbol.lower()}/concentration"
                
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'data' in data:
                            metrics['top_10_concentration'] = data['data'].get('top10', 0)
                            metrics['top_100_concentration'] = data['data'].get('top100', 0)
                            
        except Exception as e:
            logger.warning(f"IntoTheBlock API error: {e}")
        
        return metrics
    
    async def _fetch_cryptoquant_metrics(self, symbol: str) -> Dict:
        """Fetch metrics from CryptoQuant"""
        metrics = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                base_url = "https://api.cryptoquant.com/v1"
                headers = {'Authorization': f'Bearer {self.api_keys["cryptoquant"]}'}
                
                # Get exchange flows
                url = f"{base_url}/bitcoin/exchange-flows/all"
                
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'result' in data:
                            flows = data['result']['data']
                            if flows:
                                latest = flows[-1]
                                metrics['exchange_inflow'] = latest.get('inflow', 0)
                                metrics['exchange_outflow'] = latest.get('outflow', 0)
                                metrics['exchange_balance'] = latest.get('balance', 0)
                                
        except Exception as e:
            logger.warning(f"CryptoQuant API error: {e}")
        
        return metrics
    
    async def _fetch_historical_metrics(
        self,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch historical on-chain metrics"""
        # Check cache
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.metrics_cache:
            cached = self.metrics_cache[cache_key]
            if len(cached) > 0 and (datetime.now() - cached.index[-1]).seconds < 3600:
                return cached
        
        # Fetch new data
        periods = self.lookback_periods.get(timeframe, 30)
        historical_data = []
        
        # This would fetch actual historical data from APIs
        # For now, generate sample data
        for i in range(periods):
            timestamp = datetime.now() - timedelta(days=periods-i)
            historical_data.append({
                'timestamp': timestamp,
                'nvt_ratio': np.random.uniform(40, 100),
                'active_addresses': np.random.randint(100000, 500000),
                'exchange_netflow': np.random.uniform(-10_000_000, 10_000_000),
                'velocity': np.random.uniform(0.5, 2.0),
                'top_10_concentration': np.random.uniform(0.2, 0.6)
            })
        
        df = pd.DataFrame(historical_data)
        df.set_index('timestamp', inplace=True)
        
        # Cache data
        self.metrics_cache[cache_key] = df
        
        return df
    
    def _analyze_nvt_ratio(
        self,
        metrics: OnChainMetrics,
        historical: pd.DataFrame
    ) -> Optional[OnChainSignal]:
        """Analyze NVT ratio for valuation signals"""
        if metrics.nvt_ratio == 0:
            return None
        
        thresholds = self.signal_thresholds['nvt_ratio']
        
        # Determine signal strength
        if metrics.nvt_ratio < thresholds['very_oversold']:
            strength = SignalStrength.VERY_STRONG
            direction = 'bullish'
            interpretation = "Network is extremely undervalued relative to transaction volume"
        elif metrics.nvt_ratio < thresholds['oversold']:
            strength = SignalStrength.STRONG
            direction = 'bullish'
            interpretation = "Network shows strong value relative to economic activity"
        elif metrics.nvt_ratio > thresholds['very_overbought']:
            strength = SignalStrength.VERY_STRONG
            direction = 'bearish'
            interpretation = "Network is extremely overvalued relative to transaction volume"
        elif metrics.nvt_ratio > thresholds['overbought']:
            strength = SignalStrength.STRONG
            direction = 'bearish'
            interpretation = "Network shows poor value relative to economic activity"
        else:
            strength = SignalStrength.NEUTRAL
            direction = 'neutral'
            interpretation = "Network valuation is within normal range"
        
        # Calculate confidence based on historical context
        if 'nvt_ratio' in historical.columns and len(historical) > 10:
            percentile = stats.percentileofscore(
                historical['nvt_ratio'].values,
                metrics.nvt_ratio
            )
            confidence = abs(percentile - 50) / 50  # 0 at 50th percentile, 1 at extremes
        else:
            confidence = 0.5
        
        if strength == SignalStrength.NEUTRAL and confidence < 0.3:
            return None  # Don't generate weak neutral signals
        
        return OnChainSignal(
            timestamp=datetime.now(),
            symbol=metrics.symbol,
            signal_type='nvt_ratio',
            strength=strength,
            direction=direction,
            confidence=confidence,
            metrics={'nvt_ratio': metrics.nvt_ratio},
            interpretation=interpretation,
            metadata={'percentile': percentile if 'percentile' in locals() else None}
        )
    
    def _analyze_velocity(
        self,
        metrics: OnChainMetrics,
        historical: pd.DataFrame
    ) -> Optional[OnChainSignal]:
        """Analyze token velocity for circulation signals"""
        if metrics.velocity == 0:
            return None
        
        # High velocity = more speculation, low velocity = holding
        if metrics.velocity > self.config['velocity_high']:
            strength = SignalStrength.MODERATE
            direction = 'bearish'
            interpretation = "High velocity indicates excessive speculation or selling pressure"
        elif metrics.velocity < self.config['velocity_low']:
            strength = SignalStrength.MODERATE
            direction = 'bullish'
            interpretation = "Low velocity indicates strong holding behavior and accumulation"
        else:
            return None
        
        # Calculate confidence
        if 'velocity' in historical.columns and len(historical) > 10:
            z_score = (metrics.velocity - historical['velocity'].mean()) / historical['velocity'].std()
            confidence = min(abs(z_score) / 3, 1.0)  # Normalize z-score to 0-1
        else:
            confidence = 0.5
        
        return OnChainSignal(
            timestamp=datetime.now(),
            symbol=metrics.symbol,
            signal_type='velocity',
            strength=strength,
            direction=direction,
            confidence=confidence,
            metrics={'velocity': metrics.velocity},
            interpretation=interpretation,
            metadata={'z_score': z_score if 'z_score' in locals() else None}
        )
    
    def _analyze_holder_distribution(
        self,
        metrics: OnChainMetrics,
        historical: pd.DataFrame
    ) -> Optional[OnChainSignal]:
        """Analyze holder distribution for concentration signals"""
        if metrics.top_10_concentration == 0:
            return None
        
        thresholds = self.signal_thresholds['holder_concentration']
        
        # Determine signal based on concentration
        if metrics.top_10_concentration < thresholds['very_distributed']:
            strength = SignalStrength.STRONG
            direction = 'bullish'
            interpretation = "Highly distributed ownership indicates broad adoption"
        elif metrics.top_10_concentration > thresholds['very_concentrated']:
            strength = SignalStrength.STRONG
            direction = 'bearish'
            interpretation = "Extreme concentration creates manipulation risk"
        elif metrics.top_10_concentration > thresholds['concentrated']:
            strength = SignalStrength.MODERATE
            direction = 'bearish'
            interpretation = "High concentration suggests potential for price manipulation"
        else:
            return None
        
        # Check trend
        if 'top_10_concentration' in historical.columns and len(historical) > 5:
            recent_trend = historical['top_10_concentration'].tail(5).diff().mean()
            if recent_trend < -0.01:  # Decreasing concentration
                direction = 'bullish' if direction != 'bearish' else 'neutral'
                interpretation += " (improving distribution)"
            elif recent_trend > 0.01:  # Increasing concentration
                direction = 'bearish' if direction != 'bullish' else 'neutral'
                interpretation += " (worsening concentration)"
        
        confidence = 0.7  # Holder distribution is generally reliable
        
        return OnChainSignal(
            timestamp=datetime.now(),
            symbol=metrics.symbol,
            signal_type='holder_distribution',
            strength=strength,
            direction=direction,
            confidence=confidence,
            metrics={
                'top_10_concentration': metrics.top_10_concentration,
                'top_100_concentration': metrics.top_100_concentration,
                'whale_holders': metrics.whale_holders,
                'retail_holders': metrics.retail_holders
            },
            interpretation=interpretation
        )
    
    def _analyze_exchange_flows(
        self,
        metrics: OnChainMetrics,
        historical: pd.DataFrame
    ) -> Optional[OnChainSignal]:
        """Analyze exchange flows for accumulation/distribution signals"""
        if metrics.exchange_netflow == 0:
            return None
        
        thresholds = self.signal_thresholds['exchange_netflow']
        
        # Determine signal based on netflow
        if metrics.exchange_netflow < thresholds['strong_outflow']:
            strength = SignalStrength.VERY_STRONG
            direction = 'bullish'
            interpretation = "Massive exchange outflows indicate strong accumulation"
        elif metrics.exchange_netflow < thresholds['moderate_outflow']:
            strength = SignalStrength.STRONG
            direction = 'bullish'
            interpretation = "Significant withdrawal from exchanges suggests accumulation"
        elif metrics.exchange_netflow > thresholds['strong_inflow']:
            strength = SignalStrength.VERY_STRONG
            direction = 'bearish'
            interpretation = "Massive exchange inflows indicate distribution and selling pressure"
        elif metrics.exchange_netflow > thresholds['moderate_inflow']:
            strength = SignalStrength.STRONG
            direction = 'bearish'
            interpretation = "Significant deposits to exchanges suggest upcoming selling"
        else:
            return None
        
        # Calculate confidence based on flow consistency
        if 'exchange_netflow' in historical.columns and len(historical) > 3:
            recent_flows = historical['exchange_netflow'].tail(3)
            if all(recent_flows < 0) and metrics.exchange_netflow < 0:
                confidence = 0.8  # Consistent outflows
            elif all(recent_flows > 0) and metrics.exchange_netflow > 0:
                confidence = 0.8  # Consistent inflows
            else:
                confidence = 0.6  # Mixed signals
        else:
            confidence = 0.65
        
        return OnChainSignal(
            timestamp=datetime.now(),
            symbol=metrics.symbol,
            signal_type='exchange_flows',
            strength=strength,
            direction=direction,
            confidence=confidence,
            metrics={
                'exchange_netflow': metrics.exchange_netflow,
                'exchange_inflow': metrics.exchange_inflow,
                'exchange_outflow': metrics.exchange_outflow,
                'exchange_balance': metrics.exchange_balance
            },
            interpretation=interpretation
        )
    
    def _analyze_defi_metrics(
        self,
        metrics: OnChainMetrics,
        historical: pd.DataFrame
    ) -> Optional[OnChainSignal]:
        """Analyze DeFi metrics for ecosystem health signals"""
        if metrics.tvl is None or metrics.tvl == 0:
            return None
        
        # TVL growth indicates confidence
        if 'tvl' in historical.columns and len(historical) > 7:
            tvl_change = (metrics.tvl - historical['tvl'].iloc[-7]) / historical['tvl'].iloc[-7]
            
            if tvl_change > 0.2:  # 20% growth in a week
                strength = SignalStrength.STRONG
                direction = 'bullish'
                interpretation = f"TVL increased {tvl_change:.1%} indicating growing DeFi adoption"
            elif tvl_change < -0.2:  # 20% decline
                strength = SignalStrength.STRONG
                direction = 'bearish'
                interpretation = f"TVL decreased {abs(tvl_change):.1%} indicating DeFi exodus"
            else:
                return None
            
            confidence = min(abs(tvl_change) * 2, 0.9)
        else:
            return None
        
        return OnChainSignal(
            timestamp=datetime.now(),
            symbol=metrics.symbol,
            signal_type='defi_metrics',
            strength=strength,
            direction=direction,
            confidence=confidence,
            metrics={
                'tvl': metrics.tvl,
                'lending_rate': metrics.lending_rate,
                'borrowing_rate': metrics.borrowing_rate,
                'defi_dominance': metrics.defi_dominance
            },
            interpretation=interpretation,
            metadata={'tvl_change': tvl_change}
        )
    
    def _analyze_consensus_metrics(
        self,
        metrics: OnChainMetrics,
        historical: pd.DataFrame
    ) -> Optional[OnChainSignal]:
        """Analyze mining/staking metrics for network security signals"""
        if metrics.hash_rate is not None:
            # Analyze hash rate for PoW chains
            if 'hash_rate' in historical.columns and len(historical) > 7:
                hash_rate_change = (metrics.hash_rate - historical['hash_rate'].mean()) / historical['hash_rate'].mean()
                
                if hash_rate_change > 0.1:
                    return OnChainSignal(
                        timestamp=datetime.now(),
                        symbol=metrics.symbol,
                        signal_type='mining_metrics',
                        strength=SignalStrength.MODERATE,
                        direction='bullish',
                        confidence=0.7,
                        metrics={'hash_rate': metrics.hash_rate, 'difficulty': metrics.difficulty},
                        interpretation="Rising hash rate indicates miner confidence and network security"
                    )
                elif hash_rate_change < -0.1:
                    return OnChainSignal(
                        timestamp=datetime.now(),
                        symbol=metrics.symbol,
                        signal_type='mining_metrics',
                        strength=SignalStrength.MODERATE,
                        direction='bearish',
                        confidence=0.7,
                        metrics={'hash_rate': metrics.hash_rate, 'difficulty': metrics.difficulty},
                        interpretation="Declining hash rate suggests miner capitulation"
                    )
        
        elif metrics.staking_ratio is not None:
            # Analyze staking for PoS chains
            if metrics.staking_ratio > 0.7:
                return OnChainSignal(
                    timestamp=datetime.now(),
                    symbol=metrics.symbol,
                    signal_type='staking_metrics',
                    strength=SignalStrength.MODERATE,
                    direction='bullish',
                    confidence=0.65,
                    metrics={'staking_ratio': metrics.staking_ratio},
                    interpretation="High staking ratio reduces circulating supply"
                )
        
        return None
    
    def _generate_composite_signal(
        self,
        metrics: OnChainMetrics,
        signals: List[OnChainSignal]
    ) -> Optional[OnChainSignal]:
        """Generate composite signal from multiple indicators"""
        if len(signals) < 2:
            return None
        
        # Calculate weighted composite score
        bullish_score = 0
        bearish_score = 0
        total_confidence = 0
        
        for signal in signals:
            weight = signal.confidence
            if signal.direction == 'bullish':
                if signal.strength == SignalStrength.VERY_STRONG:
                    bullish_score += weight * 2
                elif signal.strength == SignalStrength.STRONG:
                    bullish_score += weight * 1.5
                else:
                    bullish_score += weight
            elif signal.direction == 'bearish':
                if signal.strength == SignalStrength.VERY_STRONG:
                    bearish_score += weight * 2
                elif signal.strength == SignalStrength.STRONG:
                    bearish_score += weight * 1.5
                else:
                    bearish_score += weight
            
            total_confidence += weight
        
        # Determine composite direction
        net_score = bullish_score - bearish_score
        avg_confidence = total_confidence / len(signals)
        
        if abs(net_score) < 0.5:
            return None  # No clear signal
        
        if net_score > 1.5:
            strength = SignalStrength.VERY_STRONG
            direction = 'bullish'
        elif net_score > 0.5:
            strength = SignalStrength.STRONG
            direction = 'bullish'
        elif net_score < -1.5:
            strength = SignalStrength.VERY_STRONG
            direction = 'bearish'
        else:
            strength = SignalStrength.STRONG
            direction = 'bearish'
        
        # Build interpretation
        signal_types = [s.signal_type for s in signals]
        interpretation = f"Composite signal from {len(signals)} indicators: {', '.join(signal_types)}"
        
        return OnChainSignal(
            timestamp=datetime.now(),
            symbol=metrics.symbol,
            signal_type='composite',
            strength=strength,
            direction=direction,
            confidence=avg_confidence,
            metrics={
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'net_score': net_score,
                'signal_count': len(signals)
            },
            interpretation=interpretation,
            metadata={'component_signals': [s.signal_type for s in signals]}
        )
    
    def _empty_metrics(self, symbol: str) -> OnChainMetrics:
        """Return empty metrics object"""
        return OnChainMetrics(
            timestamp=datetime.now(),
            symbol=symbol,
            nvt_ratio=0,
            nvts_ratio=0,
            active_addresses=0,
            transaction_count=0,
            transaction_volume=0,
            average_fee=0,
            circulating_supply=0,
            total_supply=0,
            inflation_rate=0,
            velocity=0,
            unique_addresses=0,
            top_10_concentration=0,
            top_100_concentration=0,
            retail_holders=0,
            whale_holders=0,
            exchange_balance=0,
            exchange_inflow=0,
            exchange_outflow=0,
            exchange_netflow=0
        )
    
    def _get_santiment_slug(self, symbol: str) -> str:
        """Get Santiment slug for symbol"""
        slugs = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'LINK': 'chainlink',
            'UNI': 'uniswap',
            'AAVE': 'aave'
        }
        return slugs.get(symbol.upper(), symbol.lower())
    
    def _update_performance_metrics(self, signals: List[OnChainSignal]):
        """Update performance tracking"""
        for signal in signals:
            key = f"{signal.symbol}_{signal.signal_type}"
            if key not in self.signal_accuracy:
                self.signal_accuracy[key] = {
                    'total': 0,
                    'correct': 0,
                    'accuracy': 0.0
                }
            self.signal_accuracy[key]['total'] += 1
    
    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        limit: int = 100
    ) -> List[OnChainSignal]:
        """Get historical signals"""
        filtered = self.signal_history
        
        if symbol:
            filtered = [s for s in filtered if s.symbol == symbol]
        
        if signal_type:
            filtered = [s for s in filtered if s.signal_type == signal_type]
        
        return filtered[-limit:]
    
    def get_performance_report(self) -> Dict:
        """Get performance metrics"""
        return {
            'signal_accuracy': self.signal_accuracy,
            'total_signals': len(self.signal_history),
            'best_indicators': sorted(
                self.signal_accuracy.items(),
                key=lambda x: x[1].get('accuracy', 0),
                reverse=True
            )[:5]
        }


class OnChainRL:
    """RL integration for on-chain signals"""
    
    def __init__(self, analyzer: OnChainAnalyzer):
        self.analyzer = analyzer
    
    def get_features(
        self,
        metrics: OnChainMetrics,
        signals: List[OnChainSignal]
    ) -> np.ndarray:
        """Extract RL features from on-chain data"""
        features = [
            # Network metrics
            np.log1p(metrics.nvt_ratio) / 100,
            metrics.active_addresses / 1_000_000,
            metrics.transaction_count / 100_000,
            
            # Supply metrics
            metrics.velocity,
            metrics.inflation_rate,
            
            # Holder metrics
            metrics.top_10_concentration,
            metrics.top_100_concentration,
            metrics.whale_holders / 1000,
            
            # Exchange flows
            np.sign(metrics.exchange_netflow) * np.log1p(abs(metrics.exchange_netflow)) / 20,
            metrics.exchange_balance / metrics.circulating_supply if metrics.circulating_supply > 0 else 0,
            
            # Signal aggregation
            len([s for s in signals if s.direction == 'bullish']) / max(len(signals), 1),
            len([s for s in signals if s.direction == 'bearish']) / max(len(signals), 1),
            max([s.confidence for s in signals]) if signals else 0,
            
            # DeFi metrics if available
            np.log1p(metrics.tvl) / 20 if metrics.tvl else 0,
            metrics.defi_dominance if metrics.defi_dominance else 0
        ]
        
        return np.array(features)