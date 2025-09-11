"""
Market Simulator for RL Trading Environment
Handles market data simulation, backtesting, and realistic market dynamics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"


class DataMode(Enum):
    """Data source modes"""
    HISTORICAL = "historical"
    SIMULATED = "simulated"
    LIVE = "live"


@dataclass
class MarketConditions:
    """Current market conditions"""
    regime: MarketRegime
    volatility: float
    trend_strength: float
    volume_profile: float
    bid_ask_spread: float
    market_impact_factor: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketTick:
    """Represents a market tick/candle"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_trade_price: float
    last_trade_size: float
    
    @property
    def mid_price(self) -> float:
        """Mid price between bid and ask"""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Bid-ask spread"""
        return self.ask - self.bid
    
    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points"""
        return (self.spread / self.mid_price) * 10000 if self.mid_price > 0 else 0


class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    @abstractmethod
    def get_historical_data(self, 
                          symbol: str, 
                          start_date: datetime, 
                          end_date: datetime,
                          timeframe: str = '1h') -> pd.DataFrame:
        """Get historical market data"""
        pass
    
    @abstractmethod
    def get_current_tick(self, symbol: str) -> Optional[MarketTick]:
        """Get current market tick"""
        pass


class HistoricalDataProvider(MarketDataProvider):
    """Provides historical market data for backtesting"""
    
    def __init__(self, data_source: str = 'alpaca'):
        self.data_source = data_source
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
    def get_historical_data(self, 
                          symbol: str, 
                          start_date: datetime, 
                          end_date: datetime,
                          timeframe: str = '1h') -> pd.DataFrame:
        """Get historical data from cache or data source"""
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # In a real implementation, this would connect to actual data sources
        # For now, we'll generate synthetic data
        data = self._generate_synthetic_data(symbol, start_date, end_date, timeframe)
        self.data_cache[cache_key] = data
        return data
    
    def get_current_tick(self, symbol: str) -> Optional[MarketTick]:
        """Get current tick (latest from historical data)"""
        # This would be implemented to get the latest available data
        return None
    
    def _generate_synthetic_data(self, 
                               symbol: str, 
                               start_date: datetime, 
                               end_date: datetime, 
                               timeframe: str) -> pd.DataFrame:
        """Generate synthetic market data for testing"""
        try:
            # Create time index
            if timeframe == '1h':
                freq = 'H'
            elif timeframe == '1d':
                freq = 'D'
            elif timeframe == '15m':
                freq = '15T'
            else:
                freq = 'H'
            
            timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
            
            # Generate synthetic price data using geometric Brownian motion
            n_periods = len(timestamps)
            
            # Base parameters (can be made symbol-specific)
            initial_price = 50000.0 if 'BTC' in symbol else 3000.0 if 'ETH' in symbol else 100.0
            drift = 0.0001  # Slight upward drift
            volatility = 0.02  # 2% volatility per period
            
            # Generate returns
            returns = np.random.normal(drift, volatility, n_periods)
            
            # Add some market regime changes
            for i in range(1, n_periods):
                # Add autocorrelation
                returns[i] += 0.1 * returns[i-1]
                
                # Random regime changes
                if random.random() < 0.01:  # 1% chance of regime change
                    returns[i] += random.choice([-0.05, 0.05])  # Big move
            
            # Calculate prices
            log_prices = np.cumsum(returns)
            prices = initial_price * np.exp(log_prices)
            
            # Generate OHLCV data
            data = []
            for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
                # Generate intrabar price action
                open_price = prices[i-1] if i > 0 else close_price
                
                # Random high/low within reasonable bounds
                price_range = abs(close_price - open_price) * 2 + close_price * 0.005
                high = max(open_price, close_price) + random.uniform(0, price_range * 0.5)
                low = min(open_price, close_price) - random.uniform(0, price_range * 0.5)
                
                # Volume (log-normal distribution)
                base_volume = 1000000 if 'BTC' in symbol else 500000
                volume = np.random.lognormal(np.log(base_volume), 0.5)
                
                data.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return pd.DataFrame()


class LiveDataProvider(MarketDataProvider):
    """Provides live market data for live trading"""
    
    def __init__(self, api_client=None):
        self.api_client = api_client
        
    def get_historical_data(self, 
                          symbol: str, 
                          start_date: datetime, 
                          end_date: datetime,
                          timeframe: str = '1h') -> pd.DataFrame:
        """Get historical data from live API"""
        # Implementation would connect to actual API
        return pd.DataFrame()
    
    def get_current_tick(self, symbol: str) -> Optional[MarketTick]:
        """Get current live market tick"""
        # Implementation would get real-time data
        return None


class MarketImpactModel:
    """Models market impact of trades"""
    
    def __init__(self, 
                 impact_factor: float = 0.001,
                 decay_factor: float = 0.9):
        self.impact_factor = impact_factor
        self.decay_factor = decay_factor
        self.recent_trades: List[Tuple[float, float, datetime]] = []  # (size, price, timestamp)
    
    def calculate_impact(self, 
                        trade_size: float, 
                        current_price: float, 
                        avg_volume: float,
                        is_buy: bool) -> float:
        """
        Calculate market impact of a trade
        
        Args:
            trade_size: Size of the trade
            current_price: Current market price
            avg_volume: Average trading volume
            is_buy: True for buy orders, False for sell orders
            
        Returns:
            Price impact (positive for buys, negative for sells)
        """
        try:
            # Simple square-root impact model
            volume_ratio = trade_size / max(avg_volume, 1e-8)
            impact = self.impact_factor * np.sqrt(volume_ratio) * current_price
            
            # Apply direction
            if not is_buy:
                impact = -impact
            
            # Add temporary impact from recent trades
            self._cleanup_old_trades()
            recent_impact = sum(size * self.decay_factor ** ((datetime.now() - ts).seconds / 3600) 
                              for size, _, ts in self.recent_trades)
            
            total_impact = impact + recent_impact * self.impact_factor * 0.1
            
            # Record this trade
            self.recent_trades.append((trade_size, current_price, datetime.now()))
            
            return total_impact
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {e}")
            return 0.0
    
    def _cleanup_old_trades(self):
        """Remove trades older than 24 hours"""
        cutoff = datetime.now() - timedelta(hours=24)
        self.recent_trades = [(size, price, ts) for size, price, ts in self.recent_trades 
                            if ts > cutoff]


class MarketSimulator:
    """Main market simulator for RL trading environment"""
    
    def __init__(self, 
                 mode: DataMode = DataMode.HISTORICAL,
                 symbols: List[str] = None,
                 timeframe: str = '1h',
                 enable_market_impact: bool = True,
                 enable_realistic_spreads: bool = True,
                 volatility_factor: float = 1.0):
        """
        Initialize market simulator
        
        Args:
            mode: Data mode (historical, simulated, live)
            symbols: List of trading symbols
            timeframe: Data timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            enable_market_impact: Enable market impact modeling
            enable_realistic_spreads: Enable realistic bid-ask spreads
            volatility_factor: Volatility scaling factor
        """
        self.mode = mode
        self.symbols = symbols or ['BTC/USD', 'ETH/USD']
        self.timeframe = timeframe
        self.enable_market_impact = enable_market_impact
        self.enable_realistic_spreads = enable_realistic_spreads
        self.volatility_factor = volatility_factor
        
        # Initialize data provider
        if mode == DataMode.HISTORICAL:
            self.data_provider = HistoricalDataProvider()
        elif mode == DataMode.LIVE:
            self.data_provider = LiveDataProvider()
        else:
            self.data_provider = HistoricalDataProvider()  # Default
        
        # Market state
        self.current_step = 0
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}
        self.market_conditions = MarketConditions(
            regime=MarketRegime.SIDEWAYS,
            volatility=0.02,
            trend_strength=0.0,
            volume_profile=1.0,
            bid_ask_spread=0.001,
            market_impact_factor=0.001
        )
        
        # Market impact model
        self.impact_model = MarketImpactModel() if enable_market_impact else None
        
        # Historical data for backtesting
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.current_timestamp: Optional[datetime] = None
        
        logger.info(f"MarketSimulator initialized in {mode.value} mode")
    
    def load_data(self, 
                  start_date: datetime, 
                  end_date: datetime,
                  symbols: Optional[List[str]] = None) -> bool:
        """
        Load market data for the specified period
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            symbols: List of symbols (optional, uses default if None)
            
        Returns:
            True if data loaded successfully
        """
        try:
            self.start_date = start_date
            self.end_date = end_date
            symbols_to_load = symbols or self.symbols
            
            logger.info(f"Loading data for {len(symbols_to_load)} symbols from {start_date} to {end_date}")
            
            for symbol in symbols_to_load:
                data = self.data_provider.get_historical_data(
                    symbol, start_date, end_date, self.timeframe
                )
                
                if data.empty:
                    logger.warning(f"No data loaded for {symbol}")
                    continue
                
                self.market_data[symbol] = data
                self.current_prices[symbol] = data['close'].iloc[0]
                
                logger.info(f"Loaded {len(data)} data points for {symbol}")
            
            if not self.market_data:
                logger.error("No market data loaded")
                return False
            
            # Reset to beginning
            self.reset()
            return True
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return False
    
    def step(self) -> Tuple[Dict[str, MarketTick], MarketConditions]:
        """
        Advance simulation by one step
        
        Returns:
            Tuple of (market ticks, market conditions)
        """
        try:
            if not self.market_data:
                raise ValueError("No market data loaded")
            
            # Get current data for all symbols
            ticks = {}
            for symbol, data in self.market_data.items():
                if self.current_step >= len(data):
                    continue
                
                row = data.iloc[self.current_step]
                
                # Create market tick
                tick = self._create_market_tick(symbol, row)
                ticks[symbol] = tick
                
                # Update current price
                self.current_prices[symbol] = tick.close
            
            # Update market conditions
            self._update_market_conditions()
            
            # Update timestamp
            if ticks:
                self.current_timestamp = list(ticks.values())[0].timestamp
            
            # Advance step
            self.current_step += 1
            
            return ticks, self.market_conditions
            
        except Exception as e:
            logger.error(f"Error stepping market simulator: {e}")
            return {}, self.market_conditions
    
    def get_execution_price(self, 
                          symbol: str, 
                          is_buy: bool, 
                          quantity: float,
                          order_type: str = 'market') -> float:
        """
        Get execution price for an order
        
        Args:
            symbol: Trading symbol
            is_buy: True for buy orders
            quantity: Order quantity
            order_type: Order type ('market', 'limit')
            
        Returns:
            Execution price
        """
        try:
            if symbol not in self.current_prices:
                raise ValueError(f"No price data for {symbol}")
            
            base_price = self.current_prices[symbol]
            
            # Start with current price
            execution_price = base_price
            
            # Add bid-ask spread
            if self.enable_realistic_spreads:
                spread = self._calculate_spread(symbol, base_price)
                if is_buy:
                    execution_price += spread / 2  # Pay the ask
                else:
                    execution_price -= spread / 2  # Receive the bid
            
            # Add market impact
            if self.enable_market_impact and self.impact_model:
                avg_volume = self._get_average_volume(symbol)
                impact = self.impact_model.calculate_impact(
                    quantity, base_price, avg_volume, is_buy
                )
                execution_price += impact
            
            # Ensure price is positive
            execution_price = max(execution_price, 0.01)
            
            return execution_price
            
        except Exception as e:
            logger.error(f"Error calculating execution price: {e}")
            return self.current_prices.get(symbol, 100.0)
    
    def reset(self, step: int = 0):
        """Reset simulator to specified step"""
        self.current_step = step
        
        # Reset current prices to start values
        for symbol, data in self.market_data.items():
            if not data.empty and step < len(data):
                self.current_prices[symbol] = data['close'].iloc[step]
                self.current_timestamp = data.index[step]
        
        # Reset market impact
        if self.impact_model:
            self.impact_model.recent_trades.clear()
        
        logger.info(f"MarketSimulator reset to step {step}")
    
    def is_market_open(self) -> bool:
        """Check if market is open (crypto markets are always open)"""
        return True
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        return self.current_prices.copy()
    
    def get_market_conditions(self) -> MarketConditions:
        """Get current market conditions"""
        return self.market_conditions
    
    def get_available_steps(self) -> int:
        """Get number of available steps"""
        if not self.market_data:
            return 0
        
        return min(len(data) for data in self.market_data.values()) - self.current_step
    
    def _create_market_tick(self, symbol: str, row: pd.Series) -> MarketTick:
        """Create market tick from data row"""
        try:
            # Calculate bid/ask from mid price
            mid_price = row['close']
            spread = self._calculate_spread(symbol, mid_price)
            
            bid = mid_price - spread / 2
            ask = mid_price + spread / 2
            
            # Estimate sizes (in a real implementation, this would come from order book)
            avg_volume = self._get_average_volume(symbol)
            bid_size = np.random.exponential(avg_volume * 0.001)
            ask_size = np.random.exponential(avg_volume * 0.001)
            
            tick = MarketTick(
                timestamp=row.name,
                symbol=symbol,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                last_trade_price=row['close'],
                last_trade_size=np.random.exponential(avg_volume * 0.0001)
            )
            
            return tick
            
        except Exception as e:
            logger.error(f"Error creating market tick: {e}")
            # Return a basic tick
            return MarketTick(
                timestamp=datetime.now(),
                symbol=symbol,
                open=100.0, high=100.0, low=100.0, close=100.0, volume=1000.0,
                bid=99.5, ask=100.5, bid_size=10.0, ask_size=10.0,
                last_trade_price=100.0, last_trade_size=1.0
            )
    
    def _calculate_spread(self, symbol: str, price: float) -> float:
        """Calculate realistic bid-ask spread"""
        try:
            # Base spread as percentage of price
            if 'BTC' in symbol:
                base_spread_pct = 0.001  # 0.1%
            elif 'ETH' in symbol:
                base_spread_pct = 0.0015  # 0.15%
            else:
                base_spread_pct = 0.002  # 0.2% for other pairs
            
            # Adjust for volatility
            volatility_adjustment = self.market_conditions.volatility * 0.5
            total_spread_pct = base_spread_pct * (1 + volatility_adjustment)
            
            return price * total_spread_pct
            
        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return price * 0.001  # 0.1% default
    
    def _get_average_volume(self, symbol: str) -> float:
        """Get average volume for a symbol"""
        try:
            if symbol not in self.market_data:
                return 1000000.0  # Default volume
            
            data = self.market_data[symbol]
            if data.empty:
                return 1000000.0
            
            # Use recent average volume
            recent_data = data.tail(min(24, len(data)))  # Last 24 periods
            return recent_data['volume'].mean()
            
        except Exception as e:
            logger.error(f"Error getting average volume: {e}")
            return 1000000.0
    
    def _update_market_conditions(self):
        """Update market conditions based on recent price action"""
        try:
            if not self.market_data or self.current_step < 20:
                return
            
            # Analyze recent price action for primary symbol
            primary_symbol = self.symbols[0] if self.symbols else list(self.market_data.keys())[0]
            
            if primary_symbol not in self.market_data:
                return
            
            data = self.market_data[primary_symbol]
            
            # Get recent data
            start_idx = max(0, self.current_step - 20)
            end_idx = self.current_step
            recent_data = data.iloc[start_idx:end_idx]
            
            if len(recent_data) < 2:
                return
            
            # Calculate volatility
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() * self.volatility_factor
            
            # Determine trend
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            trend_strength = abs(price_change)
            
            # Determine regime
            if price_change > 0.05:  # 5% up
                regime = MarketRegime.BULL
            elif price_change < -0.05:  # 5% down
                regime = MarketRegime.BEAR
            else:
                regime = MarketRegime.SIDEWAYS
            
            if volatility > 0.03:  # High volatility
                regime = MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.01:  # Low volatility
                regime = MarketRegime.LOW_VOLATILITY
            
            # Calculate volume profile
            avg_volume = recent_data['volume'].mean()
            current_volume = recent_data['volume'].iloc[-1] if not recent_data.empty else avg_volume
            volume_profile = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Update market conditions
            self.market_conditions = MarketConditions(
                regime=regime,
                volatility=volatility,
                trend_strength=trend_strength,
                volume_profile=volume_profile,
                bid_ask_spread=self._calculate_spread(primary_symbol, recent_data['close'].iloc[-1]),
                market_impact_factor=self.market_conditions.market_impact_factor,
                timestamp=recent_data.index[-1] if not recent_data.empty else datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error updating market conditions: {e}")


if __name__ == "__main__":
    # Example usage
    simulator = MarketSimulator(
        mode=DataMode.HISTORICAL,
        symbols=['BTC/USD', 'ETH/USD'],
        timeframe='1h'
    )
    
    # Load data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    
    if simulator.load_data(start_date, end_date):
        print(f"Data loaded successfully")
        print(f"Available steps: {simulator.get_available_steps()}")
        
        # Simulate a few steps
        for i in range(5):
            ticks, conditions = simulator.step()
            print(f"Step {i}: Prices = {simulator.get_current_prices()}")
            print(f"Market Regime: {conditions.regime.value}, Volatility: {conditions.volatility:.4f}")
    else:
        print("Failed to load data")