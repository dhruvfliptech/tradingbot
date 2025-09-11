"""
Exchange Flows Tracker - Exchange Inflow/Outflow Monitoring
=============================================================

Real-time tracking and analysis of cryptocurrency flows between exchanges
and private wallets to identify accumulation and distribution patterns.

Key Features:
- Exchange deposit/withdrawal tracking
- Cross-exchange flow analysis
- Institutional vs retail flow segregation
- Flow velocity and momentum indicators
- Predictive flow modeling
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from scipy import stats, signal as sig
import aiohttp
import json

logger = logging.getLogger(__name__)


class FlowType(Enum):
    """Types of exchange flows"""
    DEPOSIT = "deposit"  # To exchange
    WITHDRAWAL = "withdrawal"  # From exchange
    INTER_EXCHANGE = "inter_exchange"  # Between exchanges
    INTERNAL = "internal"  # Within same exchange
    OTC = "otc"  # Over-the-counter
    DEFI_BRIDGE = "defi_bridge"  # To/from DeFi protocols


class ExchangeType(Enum):
    """Types of exchanges"""
    SPOT = "spot"
    DERIVATIVES = "derivatives"
    DEX = "dex"  # Decentralized exchange
    HYBRID = "hybrid"
    OTC_DESK = "otc_desk"


@dataclass
class ExchangeFlow:
    """Individual exchange flow transaction"""
    timestamp: datetime
    flow_type: FlowType
    exchange: str
    token: str
    amount: float
    amount_usd: float
    from_address: str
    to_address: str
    transaction_hash: str
    block_number: int
    is_whale: bool  # Large transaction flag
    metadata: Dict = field(default_factory=dict)


@dataclass
class FlowMetrics:
    """Aggregated flow metrics for analysis"""
    timestamp: datetime
    exchange: str
    token: str
    
    # Flow volumes
    inflow_volume: float  # Total deposits
    outflow_volume: float  # Total withdrawals
    net_flow: float  # Outflow - Inflow (positive = accumulation)
    
    # Transaction counts
    deposit_count: int
    withdrawal_count: int
    unique_depositors: int
    unique_withdrawers: int
    
    # Flow characteristics
    avg_deposit_size: float
    avg_withdrawal_size: float
    whale_ratio: float  # % of volume from whales
    retail_ratio: float  # % of volume from retail
    
    # Momentum indicators
    flow_velocity: float  # Rate of change
    flow_acceleration: float  # Rate of velocity change
    momentum_score: float  # -1 to 1
    
    # Statistical measures
    flow_volatility: float
    flow_skewness: float
    concentration_index: float  # Gini coefficient
    
    metadata: Dict = field(default_factory=dict)


@dataclass
class FlowSignal:
    """Exchange flow trading signal"""
    timestamp: datetime
    signal_type: str
    exchange: str
    token: str
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-1
    confidence: float  # 0-1
    metrics: FlowMetrics
    interpretation: str
    expected_impact: str
    time_horizon: str  # "immediate", "short", "medium", "long"
    metadata: Dict = field(default_factory=dict)


class ExchangeFlowTracker:
    """
    Advanced exchange flow tracking and analysis system.
    
    Monitors cryptocurrency flows between exchanges and wallets to identify
    smart money movements and predict market trends.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize exchange flow tracker"""
        self.config = config or self._default_config()
        
        # API configurations
        self.api_keys = {
            'cryptoquant': self.config.get('cryptoquant_api_key', ''),
            'glassnode': self.config.get('glassnode_api_key', ''),
            'chainalysis': self.config.get('chainalysis_api_key', ''),
            'tokenanalyst': self.config.get('tokenanalyst_api_key', '')
        }
        
        # Exchange addresses database
        self.exchange_addresses = self._load_exchange_addresses()
        self.exchange_labels = self._load_exchange_labels()
        
        # Flow tracking
        self.flow_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.flow_metrics: Dict[str, List[FlowMetrics]] = defaultdict(list)
        self.active_signals: List[FlowSignal] = []
        
        # Analysis parameters
        self.whale_threshold = self.config.get('whale_threshold', 1_000_000)  # $1M USD
        self.significant_flow = self.config.get('significant_flow', 10_000_000)  # $10M USD
        
        # Statistical models
        self.flow_models = {}  # Predictive models for each exchange
        self.anomaly_detectors = {}  # Anomaly detection models
        
        # Performance tracking
        self.signal_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.prediction_accuracy = {}
        
        logger.info("Exchange Flow Tracker initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'whale_threshold': 1_000_000,  # $1M USD
            'significant_flow': 10_000_000,  # $10M USD
            'flow_window': 24,  # Hours for flow analysis
            'momentum_period': 6,  # Hours for momentum calculation
            'volatility_window': 168,  # Hours (1 week) for volatility
            'anomaly_threshold': 3.0,  # Standard deviations for anomaly
            'min_confidence': 0.6,  # Minimum signal confidence
            'cache_ttl': 300,  # 5 minutes
            'api_rate_limit': 5  # Requests per second
        }
    
    def _load_exchange_addresses(self) -> Dict[str, Set[str]]:
        """Load known exchange addresses"""
        return {
            'binance': {
                '0x28c6c06298d514db089934071355e5743bf21d60',
                '0x21a31ee1afc51d94c2efccaa2092ad1028285549',
                '0xdfd5293d8e347dfe59e90efd55b2956a1343963d',
                '0x56eddb7aa87536c09ccc2793473599fd21a8b17f',
                '0x9696f59e4d72e237be84ffd425dcad154bf96976'
            },
            'coinbase': {
                '0x71660c4005ba85c37ccec55d0c4493e66fe775d3',
                '0x503828976d22510aad0201ac7ec88293211d23da',
                '0x3cd751e6b0078be393132286c442345e5dc49699',
                '0xb5d85cbf7cb3ee0d56b3bb207d5fc4b82f43f511',
                '0xeb2629a2734e272bcc07bda959863f316f4bd4cf'
            },
            'kraken': {
                '0x2910543af39aba0cd09dbb2d50200b3e800a63d2',
                '0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13',
                '0xe853c56864a2ebe4576a807d26fdc4a0ada51919',
                '0x267be94b2e4dae6cc8e2a3be7e710d5f0b86b4e4'
            },
            'huobi': {
                '0xab5c66752a9e8167967685f1450532fb96d5d24f',
                '0xe93381fb4c4f14bda253907b18fad305d799241a',
                '0xfa4b5be3f2f84f56703c42eb22142744e95a2c58',
                '0x1062a747393198f70f71ec65a582423dba7e5ab3'
            },
            # Add more exchanges
        }
    
    def _load_exchange_labels(self) -> Dict[str, ExchangeType]:
        """Load exchange type labels"""
        return {
            'binance': ExchangeType.SPOT,
            'coinbase': ExchangeType.SPOT,
            'kraken': ExchangeType.SPOT,
            'huobi': ExchangeType.SPOT,
            'deribit': ExchangeType.DERIVATIVES,
            'bitmex': ExchangeType.DERIVATIVES,
            'uniswap': ExchangeType.DEX,
            'sushiswap': ExchangeType.DEX,
            'curve': ExchangeType.DEX
        }
    
    async def track_flows(
        self,
        token: str,
        exchanges: Optional[List[str]] = None
    ) -> Tuple[Dict[str, FlowMetrics], List[FlowSignal]]:
        """
        Track and analyze exchange flows for a token.
        
        Args:
            token: Token symbol to track
            exchanges: List of exchanges to monitor (None = all)
            
        Returns:
            Current flow metrics and generated signals
        """
        try:
            if exchanges is None:
                exchanges = list(self.exchange_addresses.keys())
            
            metrics = {}
            signals = []
            
            for exchange in exchanges:
                # Fetch flow data
                flows = await self._fetch_exchange_flows(exchange, token)
                
                # Calculate metrics
                exchange_metrics = self._calculate_flow_metrics(exchange, token, flows)
                metrics[exchange] = exchange_metrics
                
                # Store metrics
                self.flow_metrics[f"{exchange}_{token}"].append(exchange_metrics)
                
                # Detect anomalies
                anomaly = self._detect_flow_anomaly(exchange, token, exchange_metrics)
                if anomaly:
                    signals.append(anomaly)
                
                # Generate flow signals
                flow_signal = self._generate_flow_signal(exchange, token, exchange_metrics)
                if flow_signal:
                    signals.append(flow_signal)
            
            # Cross-exchange analysis
            cross_signals = self._analyze_cross_exchange_flows(token, metrics)
            signals.extend(cross_signals)
            
            # Update active signals
            self.active_signals = signals
            
            # Track performance
            self._update_performance_tracking(signals)
            
            return metrics, signals
            
        except Exception as e:
            logger.error(f"Error tracking flows for {token}: {e}")
            return {}, []
    
    async def _fetch_exchange_flows(
        self,
        exchange: str,
        token: str
    ) -> List[ExchangeFlow]:
        """Fetch exchange flow data"""
        flows = []
        
        # CryptoQuant API
        if self.api_keys['cryptoquant']:
            flows.extend(await self._fetch_cryptoquant_flows(exchange, token))
        
        # Glassnode API
        if self.api_keys['glassnode']:
            flows.extend(await self._fetch_glassnode_flows(exchange, token))
        
        # TokenAnalyst API
        if self.api_keys['tokenanalyst']:
            flows.extend(await self._fetch_tokenanalyst_flows(exchange, token))
        
        # Deduplicate flows by transaction hash
        unique_flows = {}
        for flow in flows:
            if flow.transaction_hash not in unique_flows:
                unique_flows[flow.transaction_hash] = flow
        
        return list(unique_flows.values())
    
    async def _fetch_cryptoquant_flows(
        self,
        exchange: str,
        token: str
    ) -> List[ExchangeFlow]:
        """Fetch flows from CryptoQuant"""
        flows = []
        
        try:
            async with aiohttp.ClientSession() as session:
                base_url = "https://api.cryptoquant.com/v1"
                headers = {'Authorization': f'Bearer {self.api_keys["cryptoquant"]}'}
                
                # Get exchange flows
                endpoint = f"/{token.lower()}/exchange-flows/{exchange}"
                url = f"{base_url}{endpoint}"
                
                params = {
                    'window': 'day',
                    'from': (datetime.now() - timedelta(days=1)).isoformat(),
                    'to': datetime.now().isoformat()
                }
                
                async with session.get(url, headers=headers, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        for item in data.get('result', {}).get('data', []):
                            # Process inflows
                            if item.get('inflow_sum', 0) > 0:
                                flow = ExchangeFlow(
                                    timestamp=datetime.fromisoformat(item['date']),
                                    flow_type=FlowType.DEPOSIT,
                                    exchange=exchange,
                                    token=token,
                                    amount=item['inflow_sum'],
                                    amount_usd=item.get('inflow_sum_usd', 0),
                                    from_address='',  # Would need transaction details
                                    to_address=exchange,
                                    transaction_hash='',  # Would need transaction details
                                    block_number=0,
                                    is_whale=item.get('inflow_sum_usd', 0) > self.whale_threshold,
                                    metadata={'source': 'cryptoquant'}
                                )
                                flows.append(flow)
                            
                            # Process outflows
                            if item.get('outflow_sum', 0) > 0:
                                flow = ExchangeFlow(
                                    timestamp=datetime.fromisoformat(item['date']),
                                    flow_type=FlowType.WITHDRAWAL,
                                    exchange=exchange,
                                    token=token,
                                    amount=item['outflow_sum'],
                                    amount_usd=item.get('outflow_sum_usd', 0),
                                    from_address=exchange,
                                    to_address='',  # Would need transaction details
                                    transaction_hash='',  # Would need transaction details
                                    block_number=0,
                                    is_whale=item.get('outflow_sum_usd', 0) > self.whale_threshold,
                                    metadata={'source': 'cryptoquant'}
                                )
                                flows.append(flow)
                                
        except Exception as e:
            logger.warning(f"CryptoQuant API error for {exchange} {token}: {e}")
        
        return flows
    
    async def _fetch_glassnode_flows(
        self,
        exchange: str,
        token: str
    ) -> List[ExchangeFlow]:
        """Fetch flows from Glassnode"""
        flows = []
        
        try:
            async with aiohttp.ClientSession() as session:
                base_url = "https://api.glassnode.com/v1/metrics"
                headers = {'X-Api-Key': self.api_keys['glassnode']}
                
                # Map token to Glassnode asset
                asset = token.lower()
                if asset not in ['btc', 'eth']:
                    return flows
                
                # Fetch exchange flows
                url = f"{base_url}/transactions/transfers_volume_exchanges_net"
                params = {
                    'a': asset,
                    'e': exchange.lower(),
                    'i': '1h',
                    's': int((datetime.now() - timedelta(hours=24)).timestamp()),
                    'u': int(datetime.now().timestamp())
                }
                
                async with session.get(url, headers=headers, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        for item in data:
                            net_flow = item['v']
                            timestamp = datetime.fromtimestamp(item['t'])
                            
                            if net_flow != 0:
                                flow = ExchangeFlow(
                                    timestamp=timestamp,
                                    flow_type=FlowType.WITHDRAWAL if net_flow > 0 else FlowType.DEPOSIT,
                                    exchange=exchange,
                                    token=token,
                                    amount=abs(net_flow),
                                    amount_usd=abs(net_flow) * self._get_token_price(token),
                                    from_address=exchange if net_flow > 0 else '',
                                    to_address='' if net_flow > 0 else exchange,
                                    transaction_hash='',
                                    block_number=0,
                                    is_whale=abs(net_flow) * self._get_token_price(token) > self.whale_threshold,
                                    metadata={'source': 'glassnode', 'net_flow': net_flow}
                                )
                                flows.append(flow)
                                
        except Exception as e:
            logger.warning(f"Glassnode API error for {exchange} {token}: {e}")
        
        return flows
    
    async def _fetch_tokenanalyst_flows(
        self,
        exchange: str,
        token: str
    ) -> List[ExchangeFlow]:
        """Fetch flows from TokenAnalyst"""
        # TokenAnalyst API implementation would go here
        # Similar structure to other API fetchers
        return []
    
    def _calculate_flow_metrics(
        self,
        exchange: str,
        token: str,
        flows: List[ExchangeFlow]
    ) -> FlowMetrics:
        """Calculate flow metrics from raw flow data"""
        now = datetime.now()
        window_start = now - timedelta(hours=self.config['flow_window'])
        
        # Filter flows within window
        recent_flows = [f for f in flows if f.timestamp >= window_start]
        
        # Separate deposits and withdrawals
        deposits = [f for f in recent_flows if f.flow_type == FlowType.DEPOSIT]
        withdrawals = [f for f in recent_flows if f.flow_type == FlowType.WITHDRAWAL]
        
        # Calculate volumes
        inflow_volume = sum(f.amount_usd for f in deposits)
        outflow_volume = sum(f.amount_usd for f in withdrawals)
        net_flow = outflow_volume - inflow_volume  # Positive = accumulation
        
        # Transaction counts
        deposit_count = len(deposits)
        withdrawal_count = len(withdrawals)
        
        # Unique addresses
        unique_depositors = len(set(f.from_address for f in deposits if f.from_address))
        unique_withdrawers = len(set(f.to_address for f in withdrawals if f.to_address))
        
        # Average sizes
        avg_deposit_size = inflow_volume / deposit_count if deposit_count > 0 else 0
        avg_withdrawal_size = outflow_volume / withdrawal_count if withdrawal_count > 0 else 0
        
        # Whale vs retail ratio
        whale_deposits = [f for f in deposits if f.is_whale]
        whale_withdrawals = [f for f in withdrawals if f.is_whale]
        
        whale_volume = sum(f.amount_usd for f in whale_deposits + whale_withdrawals)
        total_volume = inflow_volume + outflow_volume
        whale_ratio = whale_volume / total_volume if total_volume > 0 else 0
        retail_ratio = 1 - whale_ratio
        
        # Calculate momentum
        momentum_metrics = self._calculate_flow_momentum(exchange, token, net_flow)
        
        # Statistical measures
        all_amounts = [f.amount_usd for f in recent_flows]
        flow_volatility = np.std(all_amounts) if len(all_amounts) > 1 else 0
        flow_skewness = stats.skew(all_amounts) if len(all_amounts) > 2 else 0
        concentration_index = self._calculate_gini_coefficient(all_amounts)
        
        return FlowMetrics(
            timestamp=now,
            exchange=exchange,
            token=token,
            inflow_volume=inflow_volume,
            outflow_volume=outflow_volume,
            net_flow=net_flow,
            deposit_count=deposit_count,
            withdrawal_count=withdrawal_count,
            unique_depositors=unique_depositors,
            unique_withdrawers=unique_withdrawers,
            avg_deposit_size=avg_deposit_size,
            avg_withdrawal_size=avg_withdrawal_size,
            whale_ratio=whale_ratio,
            retail_ratio=retail_ratio,
            flow_velocity=momentum_metrics['velocity'],
            flow_acceleration=momentum_metrics['acceleration'],
            momentum_score=momentum_metrics['score'],
            flow_volatility=flow_volatility,
            flow_skewness=flow_skewness,
            concentration_index=concentration_index,
            metadata={'flow_count': len(recent_flows)}
        )
    
    def _calculate_flow_momentum(
        self,
        exchange: str,
        token: str,
        current_net_flow: float
    ) -> Dict[str, float]:
        """Calculate flow momentum indicators"""
        key = f"{exchange}_{token}"
        historical = self.flow_metrics.get(key, [])
        
        if len(historical) < 2:
            return {'velocity': 0, 'acceleration': 0, 'score': 0}
        
        # Get recent net flows
        lookback = min(self.config['momentum_period'], len(historical))
        recent_flows = [m.net_flow for m in historical[-lookback:]]
        recent_flows.append(current_net_flow)
        
        # Calculate velocity (rate of change)
        if len(recent_flows) >= 2:
            velocity = recent_flows[-1] - recent_flows[-2]
        else:
            velocity = 0
        
        # Calculate acceleration (rate of velocity change)
        if len(recent_flows) >= 3:
            prev_velocity = recent_flows[-2] - recent_flows[-3]
            acceleration = velocity - prev_velocity
        else:
            acceleration = 0
        
        # Calculate momentum score
        # Positive score = accelerating accumulation
        # Negative score = accelerating distribution
        if current_net_flow > 0 and velocity > 0:
            score = min(1.0, (velocity / abs(current_net_flow)) * (1 + acceleration / abs(velocity + 1)))
        elif current_net_flow < 0 and velocity < 0:
            score = max(-1.0, (velocity / abs(current_net_flow)) * (1 + acceleration / abs(velocity + 1)))
        else:
            score = 0
        
        return {
            'velocity': velocity,
            'acceleration': acceleration,
            'score': score
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for concentration"""
        if not values or len(values) < 2:
            return 0
        
        # Remove negative values and sort
        values = sorted([v for v in values if v >= 0])
        n = len(values)
        
        if n == 0 or sum(values) == 0:
            return 0
        
        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
    
    def _detect_flow_anomaly(
        self,
        exchange: str,
        token: str,
        metrics: FlowMetrics
    ) -> Optional[FlowSignal]:
        """Detect anomalous flow patterns"""
        key = f"{exchange}_{token}"
        historical = self.flow_metrics.get(key, [])
        
        if len(historical) < 20:  # Need sufficient history
            return None
        
        # Calculate historical statistics
        hist_net_flows = [m.net_flow for m in historical[-100:]]
        mean_flow = np.mean(hist_net_flows)
        std_flow = np.std(hist_net_flows)
        
        if std_flow == 0:
            return None
        
        # Calculate z-score
        z_score = (metrics.net_flow - mean_flow) / std_flow
        
        # Check for anomaly
        if abs(z_score) > self.config['anomaly_threshold']:
            direction = 'bullish' if metrics.net_flow > 0 else 'bearish'
            
            if abs(z_score) > 4:
                strength = 1.0
                interpretation = f"Extreme anomaly detected: {abs(z_score):.1f} standard deviations from mean"
            else:
                strength = min(abs(z_score) / 5, 1.0)
                interpretation = f"Significant flow anomaly: {abs(z_score):.1f} standard deviations from mean"
            
            return FlowSignal(
                timestamp=datetime.now(),
                signal_type='flow_anomaly',
                exchange=exchange,
                token=token,
                direction=direction,
                strength=strength,
                confidence=min(abs(z_score) / 4, 0.95),
                metrics=metrics,
                interpretation=interpretation,
                expected_impact='high' if abs(z_score) > 4 else 'moderate',
                time_horizon='immediate' if abs(z_score) > 4 else 'short',
                metadata={'z_score': z_score, 'mean_flow': mean_flow, 'std_flow': std_flow}
            )
        
        return None
    
    def _generate_flow_signal(
        self,
        exchange: str,
        token: str,
        metrics: FlowMetrics
    ) -> Optional[FlowSignal]:
        """Generate trading signal from flow metrics"""
        # Check for significant flows
        if abs(metrics.net_flow) < self.config['significant_flow'] / 10:
            return None
        
        # Determine signal direction and strength
        if metrics.net_flow > self.config['significant_flow']:
            # Large outflows (accumulation)
            direction = 'bullish'
            strength = min(metrics.net_flow / (self.config['significant_flow'] * 2), 1.0)
            interpretation = f"Massive withdrawal of ${metrics.net_flow:,.0f} indicates strong accumulation"
            expected_impact = 'high'
            
        elif metrics.net_flow < -self.config['significant_flow']:
            # Large inflows (distribution)
            direction = 'bearish'
            strength = min(abs(metrics.net_flow) / (self.config['significant_flow'] * 2), 1.0)
            interpretation = f"Massive deposit of ${abs(metrics.net_flow):,.0f} indicates distribution pressure"
            expected_impact = 'high'
            
        elif metrics.net_flow > 0 and metrics.momentum_score > 0.5:
            # Accelerating accumulation
            direction = 'bullish'
            strength = metrics.momentum_score
            interpretation = "Accelerating exchange outflows suggest growing accumulation"
            expected_impact = 'moderate'
            
        elif metrics.net_flow < 0 and metrics.momentum_score < -0.5:
            # Accelerating distribution
            direction = 'bearish'
            strength = abs(metrics.momentum_score)
            interpretation = "Accelerating exchange inflows suggest growing selling pressure"
            expected_impact = 'moderate'
            
        else:
            return None
        
        # Calculate confidence based on multiple factors
        confidence_factors = []
        
        # Volume consistency
        if metrics.deposit_count > 0 and metrics.withdrawal_count > 0:
            balance_ratio = min(metrics.deposit_count, metrics.withdrawal_count) / \
                          max(metrics.deposit_count, metrics.withdrawal_count)
            confidence_factors.append(1 - balance_ratio)  # Imbalance = higher confidence
        
        # Whale participation
        confidence_factors.append(metrics.whale_ratio)
        
        # Flow concentration
        confidence_factors.append(1 - metrics.concentration_index)  # Distributed = higher confidence
        
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        if confidence < self.config['min_confidence']:
            return None
        
        # Determine time horizon
        if abs(metrics.net_flow) > self.config['significant_flow']:
            time_horizon = 'immediate'
        elif metrics.momentum_score != 0:
            time_horizon = 'short'
        else:
            time_horizon = 'medium'
        
        return FlowSignal(
            timestamp=datetime.now(),
            signal_type='exchange_flow',
            exchange=exchange,
            token=token,
            direction=direction,
            strength=strength,
            confidence=confidence,
            metrics=metrics,
            interpretation=interpretation,
            expected_impact=expected_impact,
            time_horizon=time_horizon,
            metadata={
                'whale_participation': metrics.whale_ratio,
                'unique_actors': metrics.unique_depositors + metrics.unique_withdrawers
            }
        )
    
    def _analyze_cross_exchange_flows(
        self,
        token: str,
        metrics: Dict[str, FlowMetrics]
    ) -> List[FlowSignal]:
        """Analyze flows across multiple exchanges"""
        signals = []
        
        if len(metrics) < 2:
            return signals
        
        # Calculate aggregate metrics
        total_inflow = sum(m.inflow_volume for m in metrics.values())
        total_outflow = sum(m.outflow_volume for m in metrics.values())
        total_net_flow = total_outflow - total_inflow
        
        # Check for coordinated flows
        net_flows = [m.net_flow for m in metrics.values()]
        
        # All exchanges showing same direction
        if all(f > 0 for f in net_flows) and total_net_flow > self.config['significant_flow']:
            signal = FlowSignal(
                timestamp=datetime.now(),
                signal_type='coordinated_accumulation',
                exchange='all',
                token=token,
                direction='bullish',
                strength=min(total_net_flow / (self.config['significant_flow'] * 3), 1.0),
                confidence=0.85,
                metrics=list(metrics.values())[0],  # Use first exchange metrics as representative
                interpretation=f"Coordinated withdrawal across {len(metrics)} exchanges totaling ${total_net_flow:,.0f}",
                expected_impact='very_high',
                time_horizon='immediate',
                metadata={'exchanges': list(metrics.keys()), 'total_net_flow': total_net_flow}
            )
            signals.append(signal)
            
        elif all(f < 0 for f in net_flows) and abs(total_net_flow) > self.config['significant_flow']:
            signal = FlowSignal(
                timestamp=datetime.now(),
                signal_type='coordinated_distribution',
                exchange='all',
                token=token,
                direction='bearish',
                strength=min(abs(total_net_flow) / (self.config['significant_flow'] * 3), 1.0),
                confidence=0.85,
                metrics=list(metrics.values())[0],
                interpretation=f"Coordinated deposits across {len(metrics)} exchanges totaling ${abs(total_net_flow):,.0f}",
                expected_impact='very_high',
                time_horizon='immediate',
                metadata={'exchanges': list(metrics.keys()), 'total_net_flow': total_net_flow}
            )
            signals.append(signal)
        
        # Check for inter-exchange arbitrage patterns
        if len(metrics) >= 3:
            # High variance in flows might indicate arbitrage
            flow_variance = np.var(net_flows)
            if flow_variance > (self.config['significant_flow'] / 2) ** 2:
                signal = FlowSignal(
                    timestamp=datetime.now(),
                    signal_type='arbitrage_activity',
                    exchange='multi',
                    token=token,
                    direction='neutral',
                    strength=0.5,
                    confidence=0.6,
                    metrics=list(metrics.values())[0],
                    interpretation="High variance in exchange flows suggests arbitrage activity",
                    expected_impact='low',
                    time_horizon='immediate',
                    metadata={'flow_variance': flow_variance}
                )
                signals.append(signal)
        
        return signals
    
    def _get_token_price(self, token: str) -> float:
        """Get current token price in USD"""
        # This would fetch real prices from price feeds
        prices = {
            'BTC': 30000,
            'ETH': 2000,
            'BNB': 300,
            'USDT': 1,
            'USDC': 1
        }
        return prices.get(token.upper(), 100)
    
    def _update_performance_tracking(self, signals: List[FlowSignal]):
        """Update signal performance tracking"""
        for signal in signals:
            key = f"{signal.exchange}_{signal.token}_{signal.signal_type}"
            self.signal_performance[key]['total'] += 1
            # Actual performance would be tracked after market moves
    
    def get_flow_summary(self, token: str) -> Dict:
        """Get summary of flows for a token"""
        summary = {
            'token': token,
            'exchanges': {},
            'aggregate': {
                'total_inflow': 0,
                'total_outflow': 0,
                'net_flow': 0,
                'dominant_direction': 'neutral'
            }
        }
        
        for exchange in self.exchange_addresses.keys():
            key = f"{exchange}_{token}"
            if key in self.flow_metrics and self.flow_metrics[key]:
                latest = self.flow_metrics[key][-1]
                summary['exchanges'][exchange] = {
                    'net_flow': latest.net_flow,
                    'momentum': latest.momentum_score,
                    'whale_ratio': latest.whale_ratio
                }
                summary['aggregate']['total_inflow'] += latest.inflow_volume
                summary['aggregate']['total_outflow'] += latest.outflow_volume
        
        summary['aggregate']['net_flow'] = \
            summary['aggregate']['total_outflow'] - summary['aggregate']['total_inflow']
        
        if summary['aggregate']['net_flow'] > self.config['significant_flow'] / 10:
            summary['aggregate']['dominant_direction'] = 'accumulation'
        elif summary['aggregate']['net_flow'] < -self.config['significant_flow'] / 10:
            summary['aggregate']['dominant_direction'] = 'distribution'
        
        return summary
    
    def get_historical_flows(
        self,
        exchange: str,
        token: str,
        hours: int = 24
    ) -> pd.DataFrame:
        """Get historical flow data"""
        key = f"{exchange}_{token}"
        if key not in self.flow_metrics:
            return pd.DataFrame()
        
        metrics = self.flow_metrics[key]
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [m for m in metrics if m.timestamp >= cutoff]
        
        if not recent:
            return pd.DataFrame()
        
        data = []
        for m in recent:
            data.append({
                'timestamp': m.timestamp,
                'inflow': m.inflow_volume,
                'outflow': m.outflow_volume,
                'net_flow': m.net_flow,
                'momentum': m.momentum_score,
                'whale_ratio': m.whale_ratio
            })
        
        return pd.DataFrame(data).set_index('timestamp')
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {
            'total_signals': sum(s['total'] for s in self.signal_performance.values()),
            'by_type': {},
            'accuracy': {}
        }
        
        for key, perf in self.signal_performance.items():
            signal_type = key.split('_')[-1]
            if signal_type not in stats['by_type']:
                stats['by_type'][signal_type] = {'total': 0, 'correct': 0}
            
            stats['by_type'][signal_type]['total'] += perf['total']
            stats['by_type'][signal_type]['correct'] += perf['correct']
            
            if perf['total'] > 0:
                stats['accuracy'][key] = perf['correct'] / perf['total']
        
        return stats


class ExchangeFlowRL:
    """RL integration for exchange flow signals"""
    
    def __init__(self, tracker: ExchangeFlowTracker):
        self.tracker = tracker
    
    def get_features(self, metrics: Dict[str, FlowMetrics]) -> np.ndarray:
        """Extract RL features from flow metrics"""
        if not metrics:
            return np.zeros(12)
        
        # Aggregate metrics across exchanges
        total_inflow = sum(m.inflow_volume for m in metrics.values())
        total_outflow = sum(m.outflow_volume for m in metrics.values())
        net_flow = total_outflow - total_inflow
        
        # Average metrics
        avg_momentum = np.mean([m.momentum_score for m in metrics.values()])
        avg_whale_ratio = np.mean([m.whale_ratio for m in metrics.values()])
        avg_concentration = np.mean([m.concentration_index for m in metrics.values()])
        
        # Flow characteristics
        max_inflow = max(m.inflow_volume for m in metrics.values())
        max_outflow = max(m.outflow_volume for m in metrics.values())
        
        features = [
            # Normalized flows
            np.sign(net_flow) * np.log1p(abs(net_flow)) / 20,
            np.log1p(total_inflow) / 20,
            np.log1p(total_outflow) / 20,
            
            # Flow ratios
            total_outflow / (total_inflow + 1),  # Accumulation ratio
            
            # Momentum and velocity
            avg_momentum,
            
            # Participation metrics
            avg_whale_ratio,
            1 - avg_concentration,  # Distribution score
            
            # Exchange diversity
            len(metrics) / 10,  # Number of exchanges normalized
            
            # Flow imbalance
            (max_outflow - max_inflow) / (max_outflow + max_inflow + 1),
            
            # Volatility indicators
            np.std([m.net_flow for m in metrics.values()]) / (abs(net_flow) + 1),
            
            # Skewness
            np.mean([m.flow_skewness for m in metrics.values()]),
            
            # Active participants
            sum(m.unique_depositors + m.unique_withdrawers for m in metrics.values()) / 1000
        ]
        
        return np.array(features)