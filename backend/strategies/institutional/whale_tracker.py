"""
Whale Tracker - Large Wallet Monitoring System
===============================================

Real-time tracking and analysis of whale wallet activities to identify
smart money movements and institutional trading patterns.

Key Features:
- Real-time whale transaction monitoring
- Wallet clustering and identification
- Behavioral pattern analysis
- Accumulation/distribution tracking
- Alert system for significant movements
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import aiohttp
import json
import hashlib

logger = logging.getLogger(__name__)


class WalletType(Enum):
    """Classification of wallet types"""
    EXCHANGE_HOT = "exchange_hot"  # Exchange hot wallet
    EXCHANGE_COLD = "exchange_cold"  # Exchange cold storage
    INSTITUTIONAL = "institutional"  # Hedge funds, investment firms
    WHALE_INDIVIDUAL = "whale_individual"  # Individual large holders
    MINER = "miner"  # Mining pools
    DEFI_PROTOCOL = "defi_protocol"  # DeFi smart contracts
    MARKET_MAKER = "market_maker"  # Market making bots
    UNKNOWN_LARGE = "unknown_large"  # Unidentified large wallet


class TransactionType(Enum):
    """Types of whale transactions"""
    ACCUMULATION = "accumulation"  # Building position
    DISTRIBUTION = "distribution"  # Reducing position
    TRANSFER = "transfer"  # Internal transfer
    EXCHANGE_DEPOSIT = "exchange_deposit"  # To exchange
    EXCHANGE_WITHDRAWAL = "exchange_withdrawal"  # From exchange
    DEFI_INTERACTION = "defi_interaction"  # DeFi protocol interaction
    WASH_TRADE = "wash_trade"  # Potential wash trading


@dataclass
class WhaleWallet:
    """Whale wallet profile"""
    address: str
    wallet_type: WalletType
    first_seen: datetime
    last_active: datetime
    total_balance: float  # In USD
    tokens_held: Dict[str, float]  # Token -> amount
    transaction_count: int
    avg_transaction_size: float
    activity_pattern: str  # e.g., "accumulator", "trader", "holder"
    risk_score: float  # 0-1, higher = riskier
    cluster_id: Optional[int] = None  # Wallet cluster
    labels: Set[str] = field(default_factory=set)  # Tags/labels
    historical_behavior: Dict = field(default_factory=dict)
    related_wallets: Set[str] = field(default_factory=set)


@dataclass
class WhaleTransaction:
    """Whale transaction data"""
    hash: str
    timestamp: datetime
    from_address: str
    to_address: str
    token: str
    amount: float
    amount_usd: float
    transaction_type: TransactionType
    gas_price: Optional[float] = None
    block_number: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class WhaleAlert:
    """Alert for significant whale activity"""
    alert_id: str
    timestamp: datetime
    alert_type: str  # "large_transfer", "accumulation", "distribution", etc.
    severity: str  # "low", "medium", "high", "critical"
    wallet_address: str
    transaction: Optional[WhaleTransaction] = None
    message: str = ""
    impact_assessment: Dict = field(default_factory=dict)
    recommended_action: Optional[str] = None


class WhaleTracker:
    """
    Advanced whale wallet tracking and analysis system.
    
    Monitors large wallet movements, identifies patterns, and provides
    actionable intelligence on institutional trading behavior.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize whale tracker"""
        self.config = config or self._default_config()
        
        # API configurations
        self.api_keys = {
            'etherscan': self.config.get('etherscan_api_key', ''),
            'bscscan': self.config.get('bscscan_api_key', ''),
            'whale_alert': self.config.get('whale_alert_api_key', ''),
            'nansen': self.config.get('nansen_api_key', ''),
            'arkham': self.config.get('arkham_api_key', '')
        }
        
        # Tracking thresholds
        self.min_whale_balance = self.config.get('min_whale_balance', 1_000_000)  # $1M USD
        self.min_transaction_size = self.config.get('min_transaction_size', 100_000)  # $100k USD
        
        # Whale database
        self.whales: Dict[str, WhaleWallet] = {}
        self.transactions: deque = deque(maxlen=10000)
        self.wallet_clusters: Dict[int, Set[str]] = {}
        self.known_exchanges: Set[str] = self._load_known_exchanges()
        self.known_institutions: Set[str] = self._load_known_institutions()
        
        # Pattern detection
        self.accumulation_patterns: Dict[str, List] = defaultdict(list)
        self.distribution_patterns: Dict[str, List] = defaultdict(list)
        self.wash_trade_suspects: Set[str] = set()
        
        # Alert system
        self.alert_queue: deque = deque(maxlen=1000)
        self.alert_handlers: List = []
        
        # Performance metrics
        self.tracking_stats = {
            'total_whales_tracked': 0,
            'total_transactions': 0,
            'total_volume_tracked': 0,
            'alerts_generated': 0
        }
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        
        logger.info("Whale Tracker initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'min_whale_balance': 1_000_000,  # $1M USD
            'min_transaction_size': 100_000,  # $100k USD
            'cluster_threshold': 0.3,  # Similarity threshold for clustering
            'accumulation_window': 7,  # Days to detect accumulation
            'distribution_window': 7,  # Days to detect distribution
            'wash_trade_threshold': 0.9,  # Similarity threshold for wash trades
            'alert_cooldown': 300,  # Seconds between similar alerts
            'max_api_retries': 3,
            'api_rate_limit': 5,  # Requests per second
            'cache_ttl': 600  # 10 minutes
        }
    
    def _load_known_exchanges(self) -> Set[str]:
        """Load known exchange addresses"""
        # This would typically be loaded from a database or file
        return {
            # Binance
            '0x28c6c06298d514db089934071355e5743bf21d60',
            '0x21a31ee1afc51d94c2efccaa2092ad1028285549',
            '0xdfd5293d8e347dfe59e90efd55b2956a1343963d',
            
            # Coinbase
            '0x71660c4005ba85c37ccec55d0c4493e66fe775d3',
            '0x503828976d22510aad0201ac7ec88293211d23da',
            
            # Add more exchange addresses
        }
    
    def _load_known_institutions(self) -> Set[str]:
        """Load known institutional addresses"""
        # This would typically be loaded from a database or file
        return {
            # Example institutional addresses
            # These would be real addresses of known funds, companies, etc.
        }
    
    async def track_wallet(self, address: str) -> Optional[WhaleWallet]:
        """
        Track and analyze a specific whale wallet.
        
        Args:
            address: Wallet address to track
            
        Returns:
            WhaleWallet profile if qualified as whale
        """
        try:
            # Fetch wallet data
            wallet_data = await self._fetch_wallet_data(address)
            
            if not wallet_data or wallet_data.get('total_usd', 0) < self.min_whale_balance:
                return None
            
            # Classify wallet type
            wallet_type = self._classify_wallet(address, wallet_data)
            
            # Analyze transaction history
            transactions = await self._fetch_wallet_transactions(address)
            activity_pattern = self._analyze_activity_pattern(transactions)
            
            # Calculate risk score
            risk_score = self._calculate_wallet_risk(wallet_data, transactions)
            
            # Find related wallets
            related = await self._find_related_wallets(address, transactions)
            
            # Create or update wallet profile
            wallet = WhaleWallet(
                address=address,
                wallet_type=wallet_type,
                first_seen=wallet_data.get('first_seen', datetime.now()),
                last_active=wallet_data.get('last_active', datetime.now()),
                total_balance=wallet_data.get('total_usd', 0),
                tokens_held=wallet_data.get('tokens', {}),
                transaction_count=len(transactions),
                avg_transaction_size=np.mean([tx.get('amount_usd', 0) for tx in transactions]) if transactions else 0,
                activity_pattern=activity_pattern,
                risk_score=risk_score,
                related_wallets=related,
                historical_behavior=self._build_historical_profile(transactions)
            )
            
            # Store wallet
            self.whales[address] = wallet
            self.tracking_stats['total_whales_tracked'] += 1
            
            # Check for alerts
            await self._check_wallet_alerts(wallet, transactions)
            
            return wallet
            
        except Exception as e:
            logger.error(f"Error tracking wallet {address}: {e}")
            return None
    
    async def _fetch_wallet_data(self, address: str) -> Dict:
        """Fetch wallet balance and token holdings"""
        wallet_data = {
            'address': address,
            'total_usd': 0,
            'tokens': {},
            'first_seen': datetime.now(),
            'last_active': datetime.now()
        }
        
        # Etherscan API for Ethereum
        if self.api_keys['etherscan']:
            try:
                async with aiohttp.ClientSession() as session:
                    # Get ETH balance
                    url = "https://api.etherscan.io/api"
                    params = {
                        'module': 'account',
                        'action': 'balance',
                        'address': address,
                        'apikey': self.api_keys['etherscan']
                    }
                    
                    async with session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data['status'] == '1':
                                eth_balance = int(data['result']) / 1e18
                                # Assume ETH price (would fetch real price)
                                eth_price = 2000
                                wallet_data['tokens']['ETH'] = eth_balance
                                wallet_data['total_usd'] += eth_balance * eth_price
                    
                    # Get token balances
                    params['action'] = 'tokentx'
                    params['sort'] = 'desc'
                    
                    async with session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data['status'] == '1':
                                tokens = {}
                                for tx in data['result']:
                                    token = tx['tokenSymbol']
                                    if token not in tokens:
                                        tokens[token] = 0
                                    
                                    if tx['to'].lower() == address.lower():
                                        tokens[token] += float(tx['value']) / (10 ** int(tx['tokenDecimal']))
                                    elif tx['from'].lower() == address.lower():
                                        tokens[token] -= float(tx['value']) / (10 ** int(tx['tokenDecimal']))
                                
                                # Update wallet data
                                for token, balance in tokens.items():
                                    if balance > 0:
                                        wallet_data['tokens'][token] = balance
                                        # Estimate USD value (would use real prices)
                                        token_price = self._estimate_token_price(token)
                                        wallet_data['total_usd'] += balance * token_price
                                        
            except Exception as e:
                logger.warning(f"Error fetching wallet data from Etherscan: {e}")
        
        return wallet_data
    
    async def _fetch_wallet_transactions(self, address: str, limit: int = 100) -> List[Dict]:
        """Fetch recent wallet transactions"""
        transactions = []
        
        if self.api_keys['etherscan']:
            try:
                async with aiohttp.ClientSession() as session:
                    url = "https://api.etherscan.io/api"
                    params = {
                        'module': 'account',
                        'action': 'txlist',
                        'address': address,
                        'sort': 'desc',
                        'apikey': self.api_keys['etherscan'],
                        'page': 1,
                        'offset': limit
                    }
                    
                    async with session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data['status'] == '1':
                                for tx in data['result']:
                                    # Calculate USD value
                                    value_eth = int(tx['value']) / 1e18
                                    eth_price = 2000  # Would fetch real price
                                    amount_usd = value_eth * eth_price
                                    
                                    if amount_usd >= self.min_transaction_size:
                                        transactions.append({
                                            'hash': tx['hash'],
                                            'timestamp': datetime.fromtimestamp(int(tx['timeStamp'])),
                                            'from': tx['from'],
                                            'to': tx['to'],
                                            'amount': value_eth,
                                            'amount_usd': amount_usd,
                                            'gas_price': int(tx['gasPrice']) / 1e9,  # In Gwei
                                            'block': int(tx['blockNumber'])
                                        })
                                        
            except Exception as e:
                logger.warning(f"Error fetching transactions: {e}")
        
        return transactions
    
    def _classify_wallet(self, address: str, wallet_data: Dict) -> WalletType:
        """Classify wallet type based on behavior and characteristics"""
        address_lower = address.lower()
        
        # Check known addresses
        if address_lower in self.known_exchanges:
            return WalletType.EXCHANGE_HOT
        
        if address_lower in self.known_institutions:
            return WalletType.INSTITUTIONAL
        
        # Check for DEX/DeFi patterns
        tokens = wallet_data.get('tokens', {})
        if any(token in ['UNI', 'SUSHI', 'AAVE', 'COMP'] for token in tokens):
            if wallet_data.get('total_usd', 0) > 10_000_000:
                return WalletType.DEFI_PROTOCOL
        
        # Check transaction patterns
        # This would analyze transaction history for patterns
        
        # Default classification based on balance
        balance = wallet_data.get('total_usd', 0)
        if balance > 50_000_000:
            return WalletType.INSTITUTIONAL
        elif balance > 10_000_000:
            return WalletType.WHALE_INDIVIDUAL
        else:
            return WalletType.UNKNOWN_LARGE
    
    def _analyze_activity_pattern(self, transactions: List[Dict]) -> str:
        """Analyze wallet activity pattern"""
        if not transactions:
            return "inactive"
        
        # Calculate metrics
        df = pd.DataFrame(transactions)
        
        if len(df) < 10:
            return "new"
        
        # Time between transactions
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        time_diffs = df['timestamp'].diff().dt.total_seconds() / 3600  # Hours
        
        avg_time_between = time_diffs.mean()
        
        # Transaction size patterns
        amounts = df['amount_usd'].values
        amount_std = np.std(amounts)
        amount_mean = np.mean(amounts)
        cv = amount_std / amount_mean if amount_mean > 0 else 0
        
        # Classify pattern
        if avg_time_between < 24:  # Less than daily
            if cv > 0.5:
                return "active_trader"
            else:
                return "market_maker"
        elif avg_time_between < 168:  # Less than weekly
            return "regular_trader"
        elif avg_time_between < 720:  # Less than monthly
            return "position_trader"
        else:
            return "long_term_holder"
    
    def _calculate_wallet_risk(self, wallet_data: Dict, transactions: List[Dict]) -> float:
        """Calculate wallet risk score"""
        risk_factors = []
        
        # Concentration risk
        tokens = wallet_data.get('tokens', {})
        if tokens:
            values = list(tokens.values())
            total = sum(values)
            if total > 0:
                concentrations = [v/total for v in values]
                herfindahl = sum([c**2 for c in concentrations])
                risk_factors.append(herfindahl)  # Higher = more concentrated
        
        # Transaction frequency risk
        if transactions:
            # High frequency might indicate bot/wash trading
            tx_per_day = len(transactions) / max((transactions[0]['timestamp'] - transactions[-1]['timestamp']).days, 1)
            if tx_per_day > 10:
                risk_factors.append(0.8)
            elif tx_per_day > 5:
                risk_factors.append(0.5)
            else:
                risk_factors.append(0.2)
        
        # Size volatility risk
        if len(transactions) > 5:
            amounts = [tx['amount_usd'] for tx in transactions]
            cv = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 0
            risk_factors.append(min(cv, 1.0))
        
        # Calculate weighted risk score
        return np.mean(risk_factors) if risk_factors else 0.5
    
    async def _find_related_wallets(self, address: str, transactions: List[Dict]) -> Set[str]:
        """Find wallets related to the given address"""
        related = set()
        
        # Direct interactions
        for tx in transactions[:50]:  # Check recent 50 transactions
            if tx['from'].lower() == address.lower():
                related.add(tx['to'].lower())
            elif tx['to'].lower() == address.lower():
                related.add(tx['from'].lower())
        
        # Filter out contracts and exchanges
        related = {addr for addr in related 
                  if addr not in self.known_exchanges and 
                  not self._is_contract_address(addr)}
        
        # Check for common patterns (same gas price, timing, amounts)
        # This would implement more sophisticated relationship detection
        
        return related
    
    def _is_contract_address(self, address: str) -> bool:
        """Check if address is a smart contract"""
        # Simple heuristic - contracts often have specific patterns
        # Would use actual blockchain data in production
        return len(address) == 42 and address[:2] == '0x'
    
    def _build_historical_profile(self, transactions: List[Dict]) -> Dict:
        """Build historical behavior profile"""
        profile = {
            'total_volume': 0,
            'avg_transaction_size': 0,
            'peak_activity_hour': 0,
            'preferred_gas_price': 0,
            'accumulation_periods': [],
            'distribution_periods': []
        }
        
        if not transactions:
            return profile
        
        df = pd.DataFrame(transactions)
        
        # Total volume
        profile['total_volume'] = df['amount_usd'].sum()
        
        # Average transaction size
        profile['avg_transaction_size'] = df['amount_usd'].mean()
        
        # Peak activity hour
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            profile['peak_activity_hour'] = df['hour'].mode()[0] if not df['hour'].mode().empty else 0
        
        # Preferred gas price
        if 'gas_price' in df.columns:
            profile['preferred_gas_price'] = df['gas_price'].median()
        
        # Detect accumulation/distribution periods
        # This would implement sophisticated pattern detection
        
        return profile
    
    async def _check_wallet_alerts(self, wallet: WhaleWallet, transactions: List[Dict]):
        """Check for alert conditions"""
        alerts = []
        
        # Large transaction alert
        for tx in transactions[:5]:  # Check recent transactions
            if tx['amount_usd'] > 5_000_000:
                alert = WhaleAlert(
                    alert_id=self._generate_alert_id(wallet.address, tx['hash']),
                    timestamp=datetime.now(),
                    alert_type='large_transfer',
                    severity='high' if tx['amount_usd'] > 10_000_000 else 'medium',
                    wallet_address=wallet.address,
                    message=f"Large transfer of ${tx['amount_usd']:,.0f} detected",
                    impact_assessment={
                        'potential_market_impact': 'high' if tx['amount_usd'] > 10_000_000 else 'medium',
                        'direction': 'bearish' if tx['to'] in self.known_exchanges else 'bullish'
                    }
                )
                alerts.append(alert)
        
        # Accumulation alert
        if wallet.activity_pattern == 'accumulator':
            recent_buys = sum(1 for tx in transactions[:20] 
                            if tx['to'].lower() == wallet.address.lower())
            if recent_buys > 10:
                alert = WhaleAlert(
                    alert_id=self._generate_alert_id(wallet.address, 'accumulation'),
                    timestamp=datetime.now(),
                    alert_type='accumulation',
                    severity='medium',
                    wallet_address=wallet.address,
                    message=f"Whale accumulation detected: {recent_buys} buy transactions",
                    recommended_action='Consider following smart money accumulation'
                )
                alerts.append(alert)
        
        # Add alerts to queue
        for alert in alerts:
            self.alert_queue.append(alert)
            self.tracking_stats['alerts_generated'] += 1
            
            # Trigger alert handlers
            for handler in self.alert_handlers:
                await handler(alert)
    
    def _generate_alert_id(self, address: str, identifier: str) -> str:
        """Generate unique alert ID"""
        content = f"{address}_{identifier}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _estimate_token_price(self, symbol: str) -> float:
        """Estimate token price in USD"""
        # This would use real price feeds
        prices = {
            'ETH': 2000,
            'BTC': 30000,
            'USDT': 1,
            'USDC': 1,
            'BNB': 300,
            'LINK': 15,
            'UNI': 10,
            'AAVE': 100
        }
        return prices.get(symbol, 1)
    
    async def monitor_whale_activity(self, symbols: List[str]):
        """Start real-time whale monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                for symbol in symbols:
                    # Fetch recent whale transactions
                    whale_txs = await self._fetch_recent_whale_transactions(symbol)
                    
                    for tx in whale_txs:
                        # Track wallets involved
                        await self.track_wallet(tx['from'])
                        await self.track_wallet(tx['to'])
                        
                        # Store transaction
                        whale_tx = WhaleTransaction(
                            hash=tx['hash'],
                            timestamp=tx['timestamp'],
                            from_address=tx['from'],
                            to_address=tx['to'],
                            token=symbol,
                            amount=tx['amount'],
                            amount_usd=tx['amount_usd'],
                            transaction_type=self._classify_transaction(tx)
                        )
                        self.transactions.append(whale_tx)
                        self.tracking_stats['total_transactions'] += 1
                        self.tracking_stats['total_volume_tracked'] += tx['amount_usd']
                
                # Cluster wallets periodically
                if len(self.whales) > 100 and len(self.whales) % 50 == 0:
                    await self.cluster_wallets()
                
                # Sleep before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in whale monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _fetch_recent_whale_transactions(self, symbol: str) -> List[Dict]:
        """Fetch recent whale transactions for a symbol"""
        transactions = []
        
        # Use WhaleAlert API if available
        if self.api_keys['whale_alert']:
            try:
                async with aiohttp.ClientSession() as session:
                    url = "https://api.whale-alert.io/v1/transactions"
                    params = {
                        'api_key': self.api_keys['whale_alert'],
                        'min_value': self.min_transaction_size,
                        'start': int((datetime.now() - timedelta(minutes=5)).timestamp()),
                        'currency': symbol.lower() if symbol in ['BTC', 'ETH'] else None
                    }
                    
                    async with session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for tx in data.get('transactions', []):
                                transactions.append({
                                    'hash': tx['hash'],
                                    'timestamp': datetime.fromtimestamp(tx['timestamp']),
                                    'from': tx.get('from', {}).get('address', ''),
                                    'to': tx.get('to', {}).get('address', ''),
                                    'amount': tx['amount'],
                                    'amount_usd': tx['amount_usd']
                                })
                                
            except Exception as e:
                logger.warning(f"WhaleAlert API error: {e}")
        
        return transactions
    
    def _classify_transaction(self, tx: Dict) -> TransactionType:
        """Classify transaction type"""
        from_addr = tx.get('from', '').lower()
        to_addr = tx.get('to', '').lower()
        
        # Exchange deposits/withdrawals
        if to_addr in self.known_exchanges:
            return TransactionType.EXCHANGE_DEPOSIT
        elif from_addr in self.known_exchanges:
            return TransactionType.EXCHANGE_WITHDRAWAL
        
        # Check for wash trading patterns
        if from_addr in self.whales and to_addr in self.whales:
            from_wallet = self.whales[from_addr]
            to_wallet = self.whales[to_addr]
            if from_wallet.cluster_id == to_wallet.cluster_id:
                return TransactionType.WASH_TRADE
        
        # Check transaction history for patterns
        # More sophisticated analysis would be implemented here
        
        return TransactionType.TRANSFER
    
    async def cluster_wallets(self):
        """Cluster related wallets using hierarchical clustering"""
        if len(self.whales) < 10:
            return
        
        try:
            # Build feature matrix
            addresses = list(self.whales.keys())
            features = []
            
            for addr in addresses:
                wallet = self.whales[addr]
                feature_vec = [
                    np.log1p(wallet.total_balance),
                    wallet.transaction_count,
                    wallet.avg_transaction_size,
                    wallet.risk_score,
                    len(wallet.related_wallets),
                    hash(wallet.activity_pattern) % 100 / 100.0  # Simple encoding
                ]
                features.append(feature_vec)
            
            features = np.array(features)
            
            # Normalize features
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            
            # Compute distance matrix
            distances = pdist(features, metric='euclidean')
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distances, method='ward')
            
            # Get clusters
            clusters = fcluster(linkage_matrix, 
                              t=self.config['cluster_threshold'], 
                              criterion='distance')
            
            # Update wallet clusters
            self.wallet_clusters.clear()
            for i, addr in enumerate(addresses):
                cluster_id = int(clusters[i])
                self.whales[addr].cluster_id = cluster_id
                
                if cluster_id not in self.wallet_clusters:
                    self.wallet_clusters[cluster_id] = set()
                self.wallet_clusters[cluster_id].add(addr)
            
            logger.info(f"Clustered {len(self.whales)} wallets into {len(self.wallet_clusters)} clusters")
            
        except Exception as e:
            logger.error(f"Error clustering wallets: {e}")
    
    def get_whale_summary(self, address: str) -> Optional[Dict]:
        """Get summary of whale wallet"""
        if address not in self.whales:
            return None
        
        wallet = self.whales[address]
        
        return {
            'address': wallet.address,
            'type': wallet.wallet_type.value,
            'balance_usd': wallet.total_balance,
            'tokens': wallet.tokens_held,
            'activity': wallet.activity_pattern,
            'risk_score': wallet.risk_score,
            'cluster_size': len(self.wallet_clusters.get(wallet.cluster_id, set())) if wallet.cluster_id else 1,
            'related_wallets': len(wallet.related_wallets),
            'last_active': wallet.last_active.isoformat()
        }
    
    def get_market_sentiment(self) -> Dict:
        """Analyze overall whale sentiment"""
        if not self.transactions:
            return {'sentiment': 'neutral', 'confidence': 0}
        
        recent_txs = list(self.transactions)[-100:]  # Last 100 transactions
        
        accumulation_count = sum(1 for tx in recent_txs 
                                if tx.transaction_type == TransactionType.ACCUMULATION)
        distribution_count = sum(1 for tx in recent_txs 
                               if tx.transaction_type == TransactionType.DISTRIBUTION)
        
        exchange_deposits = sum(1 for tx in recent_txs 
                              if tx.transaction_type == TransactionType.EXCHANGE_DEPOSIT)
        exchange_withdrawals = sum(1 for tx in recent_txs 
                                 if tx.transaction_type == TransactionType.EXCHANGE_WITHDRAWAL)
        
        # Calculate sentiment
        bullish_score = accumulation_count + exchange_withdrawals
        bearish_score = distribution_count + exchange_deposits
        
        total = bullish_score + bearish_score
        if total == 0:
            return {'sentiment': 'neutral', 'confidence': 0}
        
        bull_ratio = bullish_score / total
        
        if bull_ratio > 0.6:
            sentiment = 'bullish'
        elif bull_ratio < 0.4:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        confidence = abs(bull_ratio - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'total_transactions': len(recent_txs)
        }
    
    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            **self.tracking_stats,
            'active_whales': len(self.whales),
            'wallet_clusters': len(self.wallet_clusters),
            'recent_alerts': len(self.alert_queue),
            'market_sentiment': self.get_market_sentiment()
        }
    
    def register_alert_handler(self, handler):
        """Register alert handler callback"""
        self.alert_handlers.append(handler)
    
    async def stop_monitoring(self):
        """Stop whale monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            await self.monitoring_task