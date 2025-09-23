"""
Portfolio Manager for RL Trading Environment
Handles position tracking, P&L calculation, and risk management
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import copy

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for trading"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    commission: float = 0.0
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is active (pending or partially filled)"""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    market_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.market_price
    
    @property
    def cost_basis(self) -> float:
        """Cost basis of position"""
        return abs(self.quantity) * self.average_price
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)"""
        return abs(self.quantity) < 1e-8
    
    def update_market_price(self, new_price: float):
        """Update market price and recalculate unrealized P&L"""
        self.market_price = new_price
        if not self.is_flat:
            self.unrealized_pnl = (new_price - self.average_price) * self.quantity


@dataclass
class Trade:
    """Represents a completed trade"""
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    order_id: str
    pnl: float = 0.0
    
    @property
    def gross_amount(self) -> float:
        """Gross trade amount"""
        return self.quantity * self.price
    
    @property
    def net_amount(self) -> float:
        """Net trade amount after commission"""
        return self.gross_amount - self.commission


@dataclass
class PortfolioState:
    """Current portfolio state"""
    cash_balance: float
    total_equity: float
    portfolio_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    positions: Dict[str, Position]
    active_orders: List[Order]
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for observation space"""
        return {
            'cash_balance': self.cash_balance,
            'total_equity': self.total_equity,
            'portfolio_value': self.portfolio_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'num_positions': len([p for p in self.positions.values() if not p.is_flat]),
            'num_active_orders': len(self.active_orders)
        }


class PortfolioManager:
    """Manages portfolio state, positions, and orders for RL trading environment"""
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 max_position_size: float = 1.0):
        """
        Initialize portfolio manager
        
        Args:
            initial_balance: Starting cash balance
            commission_rate: Commission rate (e.g., 0.001 = 0.1%)
            slippage_rate: Slippage rate for market orders
            max_position_size: Maximum position size as fraction of portfolio
        """
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        
        # Portfolio state
        self.cash_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.active_orders: List[Order] = []
        self.completed_trades: List[Trade] = []
        
        # Performance tracking
        self.equity_curve: List[float] = [initial_balance]
        self.timestamps: List[datetime] = [datetime.now()]
        self.peak_equity = initial_balance
        self.max_drawdown = 0.0
        self.daily_pnl_history: List[float] = []
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        
        # Order management
        self.next_order_id = 1
        self.next_trade_id = 1
        
        logger.info(f"PortfolioManager initialized with ${initial_balance:,.2f}")
    
    def place_order(self, 
                   symbol: str,
                   side: OrderSide,
                   quantity: float,
                   order_type: OrderType = OrderType.MARKET,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> Optional[Order]:
        """
        Place a trading order
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Order type
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            
        Returns:
            Order object if successful, None if rejected
        """
        try:
            # Validate order
            if not self._validate_order(symbol, side, quantity, order_type, price):
                return None
            
            # Create order
            order = Order(
                id=self._generate_order_id(),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                timestamp=datetime.now()
            )
            
            # Add to active orders
            self.active_orders.append(order)
            
            logger.info(f"Order placed: {order.id} - {side.value} {quantity} {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def execute_order(self, order: Order, execution_price: float) -> bool:
        """
        Execute an order at given price
        
        Args:
            order: Order to execute
            execution_price: Price at which to execute
            
        Returns:
            True if execution successful
        """
        try:
            if not order.is_active:
                logger.warning(f"Order {order.id} is not active")
                return False
            
            # Apply slippage for market orders
            if order.order_type == OrderType.MARKET:
                if order.side == OrderSide.BUY:
                    execution_price *= (1 + self.slippage_rate)
                else:
                    execution_price *= (1 - self.slippage_rate)
            
            # Calculate commission
            commission = order.quantity * execution_price * self.commission_rate
            
            # Check if we have sufficient funds/position
            if not self._check_execution_feasibility(order, execution_price, commission):
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order {order.id} rejected: insufficient funds/position")
                return False
            
            # Execute the order
            self._execute_trade(order, execution_price, commission)
            
            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = execution_price
            order.commission = commission
            
            # Remove from active orders
            self.active_orders = [o for o in self.active_orders if o.id != order.id]
            
            logger.info(f"Order executed: {order.id} at ${execution_price:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing order {order.id}: {e}")
            return False
    
    def update_market_prices(self, prices: Dict[str, float]):
        """
        Update market prices for all positions
        
        Args:
            prices: Dictionary of symbol -> price
        """
        try:
            for symbol, price in prices.items():
                if symbol in self.positions:
                    self.positions[symbol].update_market_price(price)
            
            # Update equity curve
            current_equity = self.get_total_equity()
            self.equity_curve.append(current_equity)
            self.timestamps.append(datetime.now())
            
            # Update peak equity and drawdown
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
                
        except Exception as e:
            logger.error(f"Error updating market prices: {e}")
    
    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state"""
        try:
            total_equity = self.get_total_equity()
            unrealized_pnl = self.get_unrealized_pnl()
            realized_pnl = self.get_realized_pnl()
            
            return PortfolioState(
                cash_balance=self.cash_balance,
                total_equity=total_equity,
                portfolio_value=total_equity,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_pnl=unrealized_pnl + realized_pnl,
                positions=copy.deepcopy(self.positions),
                active_orders=copy.deepcopy(self.active_orders),
                daily_pnl=self._calculate_daily_pnl(),
                max_drawdown=self.max_drawdown,
                peak_equity=self.peak_equity
            )
            
        except Exception as e:
            logger.error(f"Error getting portfolio state: {e}")
            return PortfolioState(
                cash_balance=0.0, total_equity=0.0, portfolio_value=0.0,
                unrealized_pnl=0.0, realized_pnl=0.0, total_pnl=0.0,
                positions={}, active_orders=[]
            )
    
    def get_position(self, symbol: str) -> Position:
        """Get position for a symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def get_total_equity(self) -> float:
        """Calculate total portfolio equity"""
        try:
            equity = self.cash_balance
            for position in self.positions.values():
                if not position.is_flat:
                    equity += position.market_value
            return equity
        except Exception as e:
            logger.error(f"Error calculating total equity: {e}")
            return self.cash_balance
    
    def get_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L"""
        try:
            return sum(pos.unrealized_pnl for pos in self.positions.values() if not pos.is_flat)
        except Exception as e:
            logger.error(f"Error calculating unrealized P&L: {e}")
            return 0.0
    
    def get_realized_pnl(self) -> float:
        """Calculate total realized P&L"""
        try:
            return sum(trade.pnl for trade in self.completed_trades)
        except Exception as e:
            logger.error(f"Error calculating realized P&L: {e}")
            return 0.0
    
    def get_position_allocation(self, symbol: str) -> float:
        """Get position allocation as fraction of portfolio"""
        try:
            position = self.get_position(symbol)
            if position.is_flat:
                return 0.0
            
            total_equity = self.get_total_equity()
            if total_equity <= 0:
                return 0.0
            
            return abs(position.market_value) / total_equity
        except Exception as e:
            logger.error(f"Error calculating position allocation: {e}")
            return 0.0
    
    def get_available_buying_power(self, symbol: str, price: float) -> float:
        """Calculate available buying power for a symbol"""
        try:
            # For simplicity, assume no margin trading
            return self.cash_balance / price
        except Exception as e:
            logger.error(f"Error calculating buying power: {e}")
            return 0.0
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        try:
            if len(self.equity_curve) < 2:
                return {}
            
            returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            
            # Filter out invalid returns
            valid_returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
            
            if len(valid_returns) == 0:
                return {}
            
            total_return = (self.equity_curve[-1] - self.initial_balance) / self.initial_balance
            
            metrics = {
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'max_drawdown': self.max_drawdown,
                'max_drawdown_pct': self.max_drawdown * 100,
                'sharpe_ratio': self._calculate_sharpe_ratio(valid_returns),
                'win_rate': self._calculate_win_rate(),
                'profit_factor': self._calculate_profit_factor(),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'total_commission': self.total_commission,
                'avg_trade_pnl': self._calculate_avg_trade_pnl()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def reset(self, initial_balance: Optional[float] = None):
        """Reset portfolio to initial state"""
        if initial_balance is not None:
            self.initial_balance = initial_balance
        
        self.cash_balance = self.initial_balance
        self.positions.clear()
        self.active_orders.clear()
        self.completed_trades.clear()
        
        self.equity_curve = [self.initial_balance]
        self.timestamps = [datetime.now()]
        self.peak_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.daily_pnl_history.clear()
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        
        self.next_order_id = 1
        self.next_trade_id = 1
        
        logger.info(f"Portfolio reset with ${self.initial_balance:,.2f}")
    
    def _validate_order(self, symbol: str, side: OrderSide, quantity: float, 
                       order_type: OrderType, price: Optional[float]) -> bool:
        """Validate order parameters"""
        try:
            if quantity <= 0:
                logger.warning(f"Invalid quantity: {quantity}")
                return False
            
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
                logger.warning("Price required for limit orders")
                return False
            
            # Check position size limits
            if side == OrderSide.BUY:
                current_allocation = self.get_position_allocation(symbol)
                estimated_price = price or self.positions.get(symbol, Position(symbol)).market_price
                if estimated_price > 0:
                    trade_value = quantity * estimated_price
                    new_allocation = trade_value / max(self.get_total_equity(), self.cash_balance)
                    if current_allocation + new_allocation > self.max_position_size:
                        logger.warning(f"Order exceeds max position size: {current_allocation + new_allocation:.2%}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False
    
    def _check_execution_feasibility(self, order: Order, price: float, commission: float) -> bool:
        """Check if order execution is feasible"""
        try:
            if order.side == OrderSide.BUY:
                required_cash = order.quantity * price + commission
                return self.cash_balance >= required_cash
            else:  # SELL
                position = self.get_position(order.symbol)
                return position.quantity >= order.quantity
                
        except Exception as e:
            logger.error(f"Error checking execution feasibility: {e}")
            return False
    
    def _execute_trade(self, order: Order, execution_price: float, commission: float):
        """Execute the actual trade"""
        try:
            position = self.get_position(order.symbol)
            
            if order.side == OrderSide.BUY:
                # Update cash
                self.cash_balance -= (order.quantity * execution_price + commission)
                
                # Update position
                if position.is_flat:
                    position.quantity = order.quantity
                    position.average_price = execution_price
                else:
                    # Calculate new average price
                    total_cost = (position.quantity * position.average_price + 
                                order.quantity * execution_price)
                    total_quantity = position.quantity + order.quantity
                    position.average_price = total_cost / total_quantity
                    position.quantity = total_quantity
                    
            else:  # SELL
                # Calculate realized P&L
                realized_pnl = order.quantity * (execution_price - position.average_price)
                
                # Update cash
                self.cash_balance += (order.quantity * execution_price - commission)
                
                # Update position
                position.quantity -= order.quantity
                position.realized_pnl += realized_pnl
                
                # If position is closed, reset average price
                if position.is_flat:
                    position.average_price = 0.0
            
            # Create trade record
            trade = Trade(
                id=self._generate_trade_id(),
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                timestamp=datetime.now(),
                commission=commission,
                order_id=order.id,
                pnl=realized_pnl if order.side == OrderSide.SELL else 0.0
            )
            
            self.completed_trades.append(trade)
            self.total_commission += commission
            self.total_trades += 1
            
            # Update trade statistics
            if order.side == OrderSide.SELL:
                if trade.pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                    
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            raise
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        order_id = f"ORD_{self.next_order_id:06d}"
        self.next_order_id += 1
        return order_id
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        trade_id = f"TRD_{self.next_trade_id:06d}"
        self.next_trade_id += 1
        return trade_id
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        try:
            if len(self.equity_curve) < 2:
                return 0.0
            return self.equity_curve[-1] - self.equity_curve[-2]
        except Exception as e:
            logger.error(f"Error calculating daily P&L: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) == 0:
                return 0.0
            
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            if np.std(excess_returns) == 0:
                return 0.0
            
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        try:
            if self.total_trades == 0:
                return 0.0
            return self.winning_trades / self.total_trades
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        try:
            winning_trades_pnl = sum(trade.pnl for trade in self.completed_trades if trade.pnl > 0)
            losing_trades_pnl = abs(sum(trade.pnl for trade in self.completed_trades if trade.pnl < 0))
            
            if losing_trades_pnl == 0:
                return float('inf') if winning_trades_pnl > 0 else 0.0
            
            return winning_trades_pnl / losing_trades_pnl
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 0.0
    
    def _calculate_avg_trade_pnl(self) -> float:
        """Calculate average trade P&L"""
        try:
            if not self.completed_trades:
                return 0.0
            return sum(trade.pnl for trade in self.completed_trades) / len(self.completed_trades)
        except Exception as e:
            logger.error(f"Error calculating average trade P&L: {e}")
            return 0.0


if __name__ == "__main__":
    # Example usage
    portfolio = PortfolioManager(initial_balance=10000.0)
    
    # Place and execute a buy order
    order = portfolio.place_order("BTC/USD", OrderSide.BUY, 0.1, OrderType.MARKET)
    if order:
        portfolio.execute_order(order, 50000.0)
    
    # Update market price
    portfolio.update_market_prices({"BTC/USD": 52000.0})
    
    # Get portfolio state
    state = portfolio.get_portfolio_state()
    print(f"Portfolio Value: ${state.total_equity:,.2f}")
    print(f"Unrealized P&L: ${state.unrealized_pnl:,.2f}")
    
    # Get performance metrics
    metrics = portfolio.get_performance_metrics()
    print(f"Performance Metrics: {metrics}")