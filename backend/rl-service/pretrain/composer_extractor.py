"""
Composer Strategy Extractor for RL Pre-training Pipeline

This module extracts successful trading patterns from Composer MCP's 1000+ strategies
to create supervised learning datasets for pre-training RL agents.
"""

import asyncio
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StrategyPattern:
    """Represents an extracted trading pattern from a strategy"""
    strategy_id: str
    pattern_type: str  # 'momentum', 'mean_reversion', 'breakout', 'trend_following'
    market_regime: str  # 'bull', 'bear', 'sideways', 'volatile'
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    state_features: np.ndarray
    action_taken: int  # 0: hold, 1: buy, 2: sell
    reward: float
    confidence: float
    performance_metrics: Dict[str, float]
    timestamp: datetime

@dataclass
class MarketState:
    """Market state representation for pattern extraction"""
    price_features: np.ndarray  # OHLCV + technical indicators
    volume_features: np.ndarray
    volatility_features: np.ndarray
    sentiment_features: np.ndarray
    regime_features: np.ndarray
    timestamp: datetime

class ComposerExtractor:
    """Extracts trading patterns from Composer MCP strategies"""
    
    def __init__(self, 
                 composer_mcp_url: str = "https://ai.composer.trade/mcp",
                 storage_path: str = "/tmp/composer_patterns.db"):
        self.mcp_url = composer_mcp_url
        self.storage_path = storage_path
        self.session: Optional[aiohttp.ClientSession] = None
        self.pattern_cache = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize local storage
        self._init_storage()
        
        # Pattern classification configs
        self.pattern_classifiers = {
            'momentum': self._classify_momentum_pattern,
            'mean_reversion': self._classify_mean_reversion_pattern,
            'breakout': self._classify_breakout_pattern,
            'trend_following': self._classify_trend_following_pattern
        }
        
        # Market regime detection configs
        self.regime_thresholds = {
            'volatility_low': 0.15,
            'volatility_high': 0.35,
            'trend_strong': 0.7,
            'trend_weak': 0.3
        }
    
    def _init_storage(self):
        """Initialize SQLite database for pattern storage"""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                market_regime TEXT NOT NULL,
                entry_conditions TEXT NOT NULL,
                exit_conditions TEXT NOT NULL,
                state_features BLOB NOT NULL,
                action_taken INTEGER NOT NULL,
                reward REAL NOT NULL,
                confidence REAL NOT NULL,
                performance_metrics TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_pattern_type ON strategy_patterns(pattern_type);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_market_regime ON strategy_patterns(market_regime);
        ''')
        
        conn.commit()
        conn.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def extract_all_strategies(self) -> List[Dict]:
        """Extract all available strategies from Composer MCP"""
        try:
            logger.info("Fetching all available strategies from Composer MCP...")
            
            async with self.session.get(f"{self.mcp_url}/api/strategies") as response:
                if response.status == 200:
                    data = await response.json()
                    strategies = data.get('data', [])
                    logger.info(f"Found {len(strategies)} strategies")
                    return strategies
                else:
                    logger.error(f"Failed to fetch strategies: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching strategies: {e}")
            return []
    
    async def get_strategy_performance(self, strategy_id: str) -> Dict:
        """Get historical performance data for a strategy"""
        try:
            payload = {
                "strategyId": strategy_id,
                "symbols": ["BTC-USD", "ETH-USD", "SPY"],
                "startDate": (datetime.now() - timedelta(days=365)).isoformat(),
                "endDate": datetime.now().isoformat(),
                "initialCapital": 100000,
                "parameters": {},
                "riskSettings": {
                    "maxPositionSize": 0.1,
                    "stopLoss": 0.05,
                    "takeProfit": 0.15
                }
            }
            
            async with self.session.post(
                f"{self.mcp_url}/api/backtest",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get performance for strategy {strategy_id}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {}
    
    async def extract_strategy_patterns(self, strategy_id: str) -> List[StrategyPattern]:
        """Extract trading patterns from a specific strategy"""
        patterns = []
        
        try:
            # Get strategy definition and performance
            strategy_data = await self.get_strategy_performance(strategy_id)
            if not strategy_data:
                return patterns
            
            trades = strategy_data.get('trades', [])
            performance = strategy_data.get('performance', {})
            
            if not trades:
                return patterns
            
            # Get market data for the same period
            market_data = await self._get_market_data_for_trades(trades)
            
            # Extract patterns from each trade
            for trade in trades:
                try:
                    pattern = await self._extract_pattern_from_trade(
                        strategy_id, trade, market_data, performance
                    )
                    if pattern:
                        patterns.append(pattern)
                except Exception as e:
                    logger.warning(f"Failed to extract pattern from trade: {e}")
                    continue
            
            logger.info(f"Extracted {len(patterns)} patterns from strategy {strategy_id}")
            
        except Exception as e:
            logger.error(f"Error extracting patterns from strategy {strategy_id}: {e}")
        
        return patterns
    
    async def _get_market_data_for_trades(self, trades: List[Dict]) -> Dict:
        """Get market data corresponding to trade periods"""
        if not trades:
            return {}
        
        start_time = min(trade['entryTime'] for trade in trades)
        end_time = max(trade.get('exitTime', trade['entryTime']) for trade in trades)
        
        symbols = list(set(trade['symbol'] for trade in trades))
        
        try:
            payload = {
                "symbols": symbols,
                "startDate": start_time,
                "endDate": end_time,
                "timeframe": "1h"
            }
            
            async with self.session.post(
                f"{self.mcp_url}/api/historical-data",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._organize_market_data(data.get('data', []))
                else:
                    logger.warning("Failed to fetch market data for trades")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    def _organize_market_data(self, raw_data: List[Dict]) -> Dict:
        """Organize raw market data by symbol and timestamp"""
        organized = {}
        
        for candle in raw_data:
            symbol = candle['symbol']
            timestamp = candle['timestamp']
            
            if symbol not in organized:
                organized[symbol] = {}
            
            organized[symbol][timestamp] = candle
        
        return organized
    
    async def _extract_pattern_from_trade(self, 
                                        strategy_id: str,
                                        trade: Dict,
                                        market_data: Dict,
                                        strategy_performance: Dict) -> Optional[StrategyPattern]:
        """Extract a trading pattern from a single trade"""
        try:
            symbol = trade['symbol']
            entry_time = trade['entryTime']
            
            # Get market state at entry time
            market_state = self._get_market_state(symbol, entry_time, market_data)
            if not market_state:
                return None
            
            # Classify pattern type
            pattern_type = self._classify_pattern_type(trade, market_state)
            
            # Determine market regime
            market_regime = self._determine_market_regime(market_state)
            
            # Extract conditions
            entry_conditions = self._extract_entry_conditions(trade, market_state)
            exit_conditions = self._extract_exit_conditions(trade, market_state)
            
            # Calculate reward
            reward = self._calculate_trade_reward(trade, strategy_performance)
            
            # Determine action
            action = 1 if trade['side'] == 'buy' else 2  # 0=hold, 1=buy, 2=sell
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(trade, strategy_performance)
            
            return StrategyPattern(
                strategy_id=strategy_id,
                pattern_type=pattern_type,
                market_regime=market_regime,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                state_features=market_state.price_features,
                action_taken=action,
                reward=reward,
                confidence=confidence,
                performance_metrics=strategy_performance,
                timestamp=datetime.fromisoformat(entry_time)
            )
            
        except Exception as e:
            logger.error(f"Error extracting pattern from trade: {e}")
            return None
    
    def _get_market_state(self, symbol: str, timestamp: str, market_data: Dict) -> Optional[MarketState]:
        """Get market state at specific timestamp"""
        try:
            if symbol not in market_data or timestamp not in market_data[symbol]:
                return None
            
            candle = market_data[symbol][timestamp]
            
            # Extract basic price features (OHLCV)
            price_features = np.array([
                candle['open'],
                candle['high'],
                candle['low'],
                candle['close'],
                candle['volume']
            ])
            
            # Add technical indicators if available
            indicators = candle.get('indicators', {})
            if indicators:
                indicator_values = list(indicators.values())
                price_features = np.concatenate([price_features, indicator_values])
            
            # Calculate additional features
            volatility = (candle['high'] - candle['low']) / candle['close']
            
            return MarketState(
                price_features=price_features,
                volume_features=np.array([candle['volume']]),
                volatility_features=np.array([volatility]),
                sentiment_features=np.array([0.5]),  # Placeholder
                regime_features=np.array([0.5]),     # Placeholder
                timestamp=datetime.fromisoformat(timestamp)
            )
            
        except Exception as e:
            logger.error(f"Error getting market state: {e}")
            return None
    
    def _classify_pattern_type(self, trade: Dict, market_state: MarketState) -> str:
        """Classify the trading pattern type"""
        # Simple heuristic-based classification
        # In production, this would use more sophisticated ML models
        
        reason = trade.get('reason', '').lower()
        
        if any(word in reason for word in ['momentum', 'trend', 'moving']):
            return 'momentum'
        elif any(word in reason for word in ['revert', 'mean', 'oversold', 'overbought']):
            return 'mean_reversion'
        elif any(word in reason for word in ['breakout', 'break', 'resistance', 'support']):
            return 'breakout'
        elif any(word in reason for word in ['trend', 'follow']):
            return 'trend_following'
        else:
            # Default classification based on price action
            if len(market_state.price_features) >= 5:
                volatility = market_state.volatility_features[0]
                if volatility > 0.03:
                    return 'breakout'
                else:
                    return 'momentum'
            return 'momentum'
    
    def _determine_market_regime(self, market_state: MarketState) -> str:
        """Determine market regime at the time of trade"""
        volatility = market_state.volatility_features[0]
        
        if volatility > self.regime_thresholds['volatility_high']:
            return 'volatile'
        elif volatility < self.regime_thresholds['volatility_low']:
            return 'sideways'
        else:
            # Use price momentum to determine bull/bear
            if len(market_state.price_features) >= 4:
                close = market_state.price_features[3]
                open_price = market_state.price_features[0]
                if close > open_price:
                    return 'bull'
                else:
                    return 'bear'
        
        return 'sideways'
    
    def _extract_entry_conditions(self, trade: Dict, market_state: MarketState) -> Dict[str, Any]:
        """Extract entry conditions from trade and market state"""
        return {
            'price': market_state.price_features[3],  # close price
            'volume': market_state.volume_features[0],
            'volatility': market_state.volatility_features[0],
            'side': trade['side'],
            'confidence': trade.get('confidence', 0.5),
            'reason': trade.get('reason', '')
        }
    
    def _extract_exit_conditions(self, trade: Dict, market_state: MarketState) -> Dict[str, Any]:
        """Extract exit conditions from trade"""
        return {
            'exit_price': trade.get('exitPrice'),
            'exit_time': trade.get('exitTime'),
            'pnl': trade.get('pnl', 0),
            'pnl_percent': trade.get('pnlPercent', 0),
            'duration': self._calculate_trade_duration(trade)
        }
    
    def _calculate_trade_duration(self, trade: Dict) -> float:
        """Calculate trade duration in hours"""
        if 'exitTime' not in trade or not trade['exitTime']:
            return 0.0
        
        try:
            entry_time = datetime.fromisoformat(trade['entryTime'])
            exit_time = datetime.fromisoformat(trade['exitTime'])
            duration = (exit_time - entry_time).total_seconds() / 3600  # hours
            return duration
        except:
            return 0.0
    
    def _calculate_trade_reward(self, trade: Dict, strategy_performance: Dict) -> float:
        """Calculate reward for a trade based on PnL and strategy performance"""
        pnl_percent = trade.get('pnlPercent', 0)
        
        # Normalize by strategy's overall performance
        sharpe_ratio = strategy_performance.get('sharpeRatio', 1.0)
        max_drawdown = strategy_performance.get('maxDrawdown', 0.1)
        
        # Risk-adjusted reward
        risk_factor = max(0.1, max_drawdown)
        reward = (pnl_percent / 100) * sharpe_ratio / risk_factor
        
        return np.clip(reward, -1.0, 1.0)
    
    def _calculate_pattern_confidence(self, trade: Dict, strategy_performance: Dict) -> float:
        """Calculate confidence score for the pattern"""
        base_confidence = trade.get('confidence', 0.5)
        
        # Adjust based on strategy performance
        win_rate = strategy_performance.get('winRate', 0.5)
        profit_factor = strategy_performance.get('profitFactor', 1.0)
        
        performance_factor = (win_rate + min(profit_factor / 2.0, 1.0)) / 2.0
        
        return np.clip(base_confidence * performance_factor, 0.0, 1.0)
    
    def _classify_momentum_pattern(self, pattern: StrategyPattern) -> bool:
        """Classify if pattern is momentum-based"""
        return pattern.pattern_type == 'momentum'
    
    def _classify_mean_reversion_pattern(self, pattern: StrategyPattern) -> bool:
        """Classify if pattern is mean reversion-based"""
        return pattern.pattern_type == 'mean_reversion'
    
    def _classify_breakout_pattern(self, pattern: StrategyPattern) -> bool:
        """Classify if pattern is breakout-based"""
        return pattern.pattern_type == 'breakout'
    
    def _classify_trend_following_pattern(self, pattern: StrategyPattern) -> bool:
        """Classify if pattern is trend following-based"""
        return pattern.pattern_type == 'trend_following'
    
    async def save_patterns(self, patterns: List[StrategyPattern]):
        """Save extracted patterns to local storage"""
        if not patterns:
            return
        
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        for pattern in patterns:
            try:
                cursor.execute('''
                    INSERT INTO strategy_patterns (
                        strategy_id, pattern_type, market_regime, entry_conditions,
                        exit_conditions, state_features, action_taken, reward,
                        confidence, performance_metrics, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.strategy_id,
                    pattern.pattern_type,
                    pattern.market_regime,
                    json.dumps(pattern.entry_conditions),
                    json.dumps(pattern.exit_conditions),
                    pickle.dumps(pattern.state_features),
                    pattern.action_taken,
                    pattern.reward,
                    pattern.confidence,
                    json.dumps(pattern.performance_metrics),
                    pattern.timestamp.isoformat()
                ))
            except Exception as e:
                logger.error(f"Error saving pattern: {e}")
                continue
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(patterns)} patterns to storage")
    
    def load_patterns(self, 
                     pattern_types: Optional[List[str]] = None,
                     market_regimes: Optional[List[str]] = None,
                     min_confidence: float = 0.0) -> List[StrategyPattern]:
        """Load patterns from storage with filtering"""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM strategy_patterns WHERE confidence >= ?"
        params = [min_confidence]
        
        if pattern_types:
            placeholders = ','.join(['?' for _ in pattern_types])
            query += f" AND pattern_type IN ({placeholders})"
            params.extend(pattern_types)
        
        if market_regimes:
            placeholders = ','.join(['?' for _ in market_regimes])
            query += f" AND market_regime IN ({placeholders})"
            params.extend(market_regimes)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        patterns = []
        for row in rows:
            try:
                pattern = StrategyPattern(
                    strategy_id=row[1],
                    pattern_type=row[2],
                    market_regime=row[3],
                    entry_conditions=json.loads(row[4]),
                    exit_conditions=json.loads(row[5]),
                    state_features=pickle.loads(row[6]),
                    action_taken=row[7],
                    reward=row[8],
                    confidence=row[9],
                    performance_metrics=json.loads(row[10]),
                    timestamp=datetime.fromisoformat(row[11])
                )
                patterns.append(pattern)
            except Exception as e:
                logger.error(f"Error loading pattern: {e}")
                continue
        
        logger.info(f"Loaded {len(patterns)} patterns from storage")
        return patterns
    
    async def run_extraction_pipeline(self, 
                                    max_strategies: int = 100,
                                    min_performance_threshold: float = 0.1) -> int:
        """Run the complete pattern extraction pipeline"""
        logger.info("Starting Composer strategy extraction pipeline...")
        
        # Get all available strategies
        strategies = await self.extract_all_strategies()
        
        if not strategies:
            logger.error("No strategies found")
            return 0
        
        # Filter strategies by performance if possible
        # For now, we'll process the first max_strategies
        strategies_to_process = strategies[:max_strategies]
        
        total_patterns = 0
        
        # Process strategies in batches
        batch_size = 10
        for i in range(0, len(strategies_to_process), batch_size):
            batch = strategies_to_process[i:i + batch_size]
            batch_patterns = []
            
            # Process batch concurrently
            tasks = [
                self.extract_strategy_patterns(strategy['id']) 
                for strategy in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Strategy processing failed: {result}")
                else:
                    batch_patterns.extend(result)
            
            # Save batch patterns
            if batch_patterns:
                await self.save_patterns(batch_patterns)
                total_patterns += len(batch_patterns)
            
            logger.info(f"Processed batch {i//batch_size + 1}, total patterns: {total_patterns}")
        
        logger.info(f"Extraction pipeline completed. Total patterns extracted: {total_patterns}")
        return total_patterns

# Example usage
async def main():
    """Example usage of ComposerExtractor"""
    async with ComposerExtractor() as extractor:
        # Run extraction pipeline
        pattern_count = await extractor.run_extraction_pipeline(max_strategies=50)
        print(f"Extracted {pattern_count} patterns")
        
        # Load and analyze patterns
        patterns = extractor.load_patterns(min_confidence=0.7)
        print(f"High-confidence patterns: {len(patterns)}")
        
        # Analyze pattern distribution
        pattern_types = {}
        market_regimes = {}
        
        for pattern in patterns:
            pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
            market_regimes[pattern.market_regime] = market_regimes.get(pattern.market_regime, 0) + 1
        
        print("Pattern type distribution:", pattern_types)
        print("Market regime distribution:", market_regimes)

if __name__ == "__main__":
    asyncio.run(main())