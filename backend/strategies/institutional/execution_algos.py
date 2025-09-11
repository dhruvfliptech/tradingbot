"""
Smart Execution Algorithms for Liquidity-Based Trading
Implements various execution strategies optimized for different market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ExecutionStyle(Enum):
    """Execution style categories"""
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    ADAPTIVE = "adaptive"
    STEALTH = "stealth"
    OPPORTUNISTIC = "opportunistic"


@dataclass
class ExecutionOrder:
    """Order for execution"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    total_quantity: float
    executed_quantity: float = 0.0
    remaining_quantity: float = 0.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_limit: Optional[datetime] = None
    execution_style: ExecutionStyle = ExecutionStyle.ADAPTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.remaining_quantity = self.total_quantity - self.executed_quantity
    
    @property
    def is_complete(self) -> bool:
        return self.remaining_quantity <= 0
    
    @property
    def fill_rate(self) -> float:
        if self.total_quantity > 0:
            return self.executed_quantity / self.total_quantity
        return 0.0


@dataclass
class ExecutionSlice:
    """Single execution slice"""
    slice_id: str
    parent_order_id: str
    quantity: float
    price: Optional[float]
    order_type: str  # 'market', 'limit', 'stop', 'stop_limit'
    time_in_force: str  # 'IOC', 'FOK', 'GTC', 'GTD'
    execute_at: datetime
    expire_at: Optional[datetime] = None
    
    
@dataclass 
class ExecutionResult:
    """Execution result"""
    order_id: str
    success: bool
    executed_quantity: float
    average_price: float
    slippage: float
    market_impact: float
    total_cost: float
    execution_time: float  # seconds
    fills: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms"""
    
    def __init__(self, symbol: str, config: Dict[str, Any]):
        self.symbol = symbol
        self.config = config
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.execution_history = deque(maxlen=100)
        
    @abstractmethod
    async def execute(self, 
                      order: ExecutionOrder,
                      market_data: Dict[str, Any],
                      order_book: Dict[str, Any]) -> ExecutionResult:
        """Execute order using specific algorithm"""
        pass
    
    @abstractmethod
    def generate_slices(self,
                       order: ExecutionOrder,
                       market_data: Dict[str, Any]) -> List[ExecutionSlice]:
        """Generate execution slices for order"""
        pass
    
    def calculate_slippage(self, 
                          executed_price: float,
                          reference_price: float,
                          side: str) -> float:
        """Calculate execution slippage"""
        if side == 'buy':
            return (executed_price - reference_price) / reference_price
        else:
            return (reference_price - executed_price) / reference_price


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price execution algorithm"""
    
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.time_slices = config.get('time_slices', 10)
        self.randomize = config.get('randomize', True)
        self.min_slice_size = config.get('min_slice_size', 0.01)
    
    async def execute(self,
                     order: ExecutionOrder,
                     market_data: Dict[str, Any],
                     order_book: Dict[str, Any]) -> ExecutionResult:
        """Execute order using TWAP algorithm"""
        try:
            start_time = datetime.now()
            slices = self.generate_slices(order, market_data)
            
            fills = []
            total_executed = 0
            total_cost = 0
            
            for slice in slices:
                # Wait until slice execution time
                wait_time = (slice.execute_at - datetime.now()).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                # Execute slice
                fill = await self._execute_slice(slice, order_book)
                if fill:
                    fills.append(fill)
                    total_executed += fill['quantity']
                    total_cost += fill['quantity'] * fill['price']
                
                # Check if order complete
                if total_executed >= order.total_quantity * 0.99:  # 99% filled
                    break
            
            # Calculate results
            avg_price = total_cost / total_executed if total_executed > 0 else 0
            reference_price = market_data.get('mid_price', avg_price)
            slippage = self.calculate_slippage(avg_price, reference_price, order.side)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                order_id=order.order_id,
                success=total_executed >= order.total_quantity * 0.95,
                executed_quantity=total_executed,
                average_price=avg_price,
                slippage=slippage,
                market_impact=self._estimate_market_impact(total_executed, market_data),
                total_cost=total_cost,
                execution_time=execution_time,
                fills=fills,
                metadata={'algorithm': 'TWAP', 'slices': len(slices)}
            )
            
        except Exception as e:
            logger.error(f"TWAP execution error: {e}")
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                executed_quantity=0,
                average_price=0,
                slippage=0,
                market_impact=0,
                total_cost=0,
                execution_time=0,
                fills=[]
            )
    
    def generate_slices(self,
                       order: ExecutionOrder,
                       market_data: Dict[str, Any]) -> List[ExecutionSlice]:
        """Generate TWAP execution slices"""
        slices = []
        
        # Calculate time interval
        if order.time_limit:
            total_time = (order.time_limit - datetime.now()).total_seconds()
        else:
            total_time = 3600  # Default 1 hour
        
        time_per_slice = total_time / self.time_slices
        
        # Calculate quantity per slice
        base_quantity = order.total_quantity / self.time_slices
        
        for i in range(self.time_slices):
            # Randomize quantity if enabled
            if self.randomize:
                quantity = base_quantity * np.random.uniform(0.8, 1.2)
            else:
                quantity = base_quantity
            
            # Ensure minimum slice size
            quantity = max(quantity, self.min_slice_size)
            
            # Calculate execution time
            if self.randomize:
                # Add random offset within slice window
                offset = np.random.uniform(0, time_per_slice * 0.5)
            else:
                offset = 0
            
            execute_at = datetime.now() + timedelta(seconds=i * time_per_slice + offset)
            
            slice = ExecutionSlice(
                slice_id=f"{order.order_id}_slice_{i}",
                parent_order_id=order.order_id,
                quantity=quantity,
                price=order.limit_price,
                order_type='limit' if order.limit_price else 'market',
                time_in_force='IOC',
                execute_at=execute_at
            )
            slices.append(slice)
        
        return slices
    
    async def _execute_slice(self,
                           slice: ExecutionSlice,
                           order_book: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a single slice"""
        # Simulated execution - replace with actual exchange API
        try:
            if slice.order_type == 'market':
                # Take liquidity from order book
                price = order_book['asks'][0][0] if slice.quantity > 0 else order_book['bids'][0][0]
            else:
                price = slice.price
            
            return {
                'slice_id': slice.slice_id,
                'quantity': slice.quantity,
                'price': price,
                'timestamp': datetime.now(),
                'order_type': slice.order_type
            }
        except Exception as e:
            logger.error(f"Slice execution error: {e}")
            return None
    
    def _estimate_market_impact(self, quantity: float, market_data: Dict[str, Any]) -> float:
        """Estimate market impact of execution"""
        # Simple linear impact model
        avg_volume = market_data.get('avg_volume_1h', 1000)
        participation_rate = quantity / avg_volume
        
        # Impact increases with participation rate
        impact = participation_rate * 0.001  # 10 bps per 1% participation
        return min(impact, 0.01)  # Cap at 100 bps


class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution algorithm"""
    
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.volume_profile = config.get('volume_profile', None)
        self.participation_rate = config.get('participation_rate', 0.1)
        self.min_slice_size = config.get('min_slice_size', 0.01)
    
    async def execute(self,
                     order: ExecutionOrder,
                     market_data: Dict[str, Any],
                     order_book: Dict[str, Any]) -> ExecutionResult:
        """Execute order using VWAP algorithm"""
        try:
            start_time = datetime.now()
            slices = self.generate_slices(order, market_data)
            
            fills = []
            total_executed = 0
            total_cost = 0
            
            for slice in slices:
                # Wait for market volume
                await self._wait_for_volume(slice, market_data)
                
                # Execute slice
                fill = await self._execute_slice(slice, order_book)
                if fill:
                    fills.append(fill)
                    total_executed += fill['quantity']
                    total_cost += fill['quantity'] * fill['price']
                
                if total_executed >= order.total_quantity * 0.99:
                    break
            
            avg_price = total_cost / total_executed if total_executed > 0 else 0
            vwap = market_data.get('vwap', avg_price)
            slippage = self.calculate_slippage(avg_price, vwap, order.side)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                order_id=order.order_id,
                success=total_executed >= order.total_quantity * 0.95,
                executed_quantity=total_executed,
                average_price=avg_price,
                slippage=slippage,
                market_impact=self._estimate_market_impact(total_executed, market_data),
                total_cost=total_cost,
                execution_time=execution_time,
                fills=fills,
                metadata={'algorithm': 'VWAP', 'vwap_tracking_error': abs(avg_price - vwap) / vwap}
            )
            
        except Exception as e:
            logger.error(f"VWAP execution error: {e}")
            return self._create_failed_result(order.order_id)
    
    def generate_slices(self,
                       order: ExecutionOrder,
                       market_data: Dict[str, Any]) -> List[ExecutionSlice]:
        """Generate VWAP execution slices based on volume profile"""
        slices = []
        
        # Get or estimate volume profile
        if self.volume_profile:
            profile = self.volume_profile
        else:
            profile = self._estimate_volume_profile(market_data)
        
        # Distribute quantity according to volume profile
        for i, volume_weight in enumerate(profile):
            quantity = order.total_quantity * volume_weight
            quantity = max(quantity, self.min_slice_size)
            
            slice = ExecutionSlice(
                slice_id=f"{order.order_id}_vwap_{i}",
                parent_order_id=order.order_id,
                quantity=quantity,
                price=None,  # Will be determined dynamically
                order_type='limit',
                time_in_force='IOC',
                execute_at=datetime.now() + timedelta(minutes=i * 5)
            )
            slices.append(slice)
        
        return slices
    
    def _estimate_volume_profile(self, market_data: Dict[str, Any]) -> List[float]:
        """Estimate intraday volume profile"""
        # Simple U-shaped profile (high at open/close, low midday)
        hours = 24
        profile = []
        
        for hour in range(hours):
            if hour < 4 or hour > 20:  # Early morning or late evening
                weight = 0.08
            elif 9 <= hour <= 11 or 14 <= hour <= 16:  # Active hours
                weight = 0.06
            else:  # Midday
                weight = 0.03
            profile.append(weight)
        
        # Normalize
        total = sum(profile)
        return [w / total for w in profile]
    
    async def _wait_for_volume(self,
                              slice: ExecutionSlice,
                              market_data: Dict[str, Any]):
        """Wait for appropriate market volume before executing"""
        # Simulate waiting for volume conditions
        await asyncio.sleep(0.1)
    
    async def _execute_slice(self,
                           slice: ExecutionSlice,
                           order_book: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute slice at VWAP"""
        try:
            # Calculate VWAP price
            bids = np.array(order_book['bids'])
            asks = np.array(order_book['asks'])
            
            # Simple VWAP calculation from order book
            total_bid_value = np.sum(bids[:, 0] * bids[:, 1])
            total_bid_volume = np.sum(bids[:, 1])
            total_ask_value = np.sum(asks[:, 0] * asks[:, 1])
            total_ask_volume = np.sum(asks[:, 1])
            
            vwap = (total_bid_value + total_ask_value) / (total_bid_volume + total_ask_volume)
            
            return {
                'slice_id': slice.slice_id,
                'quantity': slice.quantity,
                'price': vwap,
                'timestamp': datetime.now(),
                'order_type': 'vwap'
            }
        except Exception as e:
            logger.error(f"VWAP slice execution error: {e}")
            return None
    
    def _estimate_market_impact(self, quantity: float, market_data: Dict[str, Any]) -> float:
        """Estimate market impact using square-root model"""
        avg_volume = market_data.get('avg_volume_1h', 1000)
        participation_rate = quantity / avg_volume
        
        # Square-root impact model
        impact = 0.01 * np.sqrt(participation_rate)  # Impact in percentage
        return min(impact, 0.02)  # Cap at 200 bps
    
    def _create_failed_result(self, order_id: str) -> ExecutionResult:
        """Create failed execution result"""
        return ExecutionResult(
            order_id=order_id,
            success=False,
            executed_quantity=0,
            average_price=0,
            slippage=0,
            market_impact=0,
            total_cost=0,
            execution_time=0,
            fills=[]
        )


class IcebergAlgorithm(ExecutionAlgorithm):
    """Iceberg order execution algorithm"""
    
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.visible_quantity = config.get('visible_quantity', 0.1)
        self.reload_threshold = config.get('reload_threshold', 0.2)
        self.price_tolerance = config.get('price_tolerance', 0.001)
    
    async def execute(self,
                     order: ExecutionOrder,
                     market_data: Dict[str, Any],
                     order_book: Dict[str, Any]) -> ExecutionResult:
        """Execute order as iceberg"""
        try:
            start_time = datetime.now()
            fills = []
            total_executed = 0
            total_cost = 0
            
            # Calculate visible and hidden quantities
            visible_qty = min(order.total_quantity * self.visible_quantity, order.total_quantity)
            remaining_hidden = order.total_quantity - visible_qty
            
            while total_executed < order.total_quantity:
                # Place visible order
                visible_order = await self._place_visible_order(
                    visible_qty, order, order_book
                )
                
                if visible_order:
                    fills.append(visible_order)
                    total_executed += visible_order['quantity']
                    total_cost += visible_order['quantity'] * visible_order['price']
                    
                    # Check if need to reload
                    if visible_order['quantity'] >= visible_qty * (1 - self.reload_threshold):
                        # Reload from hidden quantity
                        reload_qty = min(visible_qty, remaining_hidden)
                        remaining_hidden -= reload_qty
                        visible_qty = reload_qty
                        
                        if visible_qty == 0:
                            break
                        
                        # Add small delay to avoid detection
                        await asyncio.sleep(np.random.uniform(0.5, 2.0))
                else:
                    break
            
            avg_price = total_cost / total_executed if total_executed > 0 else 0
            reference_price = market_data.get('mid_price', avg_price)
            slippage = self.calculate_slippage(avg_price, reference_price, order.side)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                order_id=order.order_id,
                success=total_executed >= order.total_quantity * 0.95,
                executed_quantity=total_executed,
                average_price=avg_price,
                slippage=slippage,
                market_impact=self._estimate_market_impact(total_executed, market_data),
                total_cost=total_cost,
                execution_time=execution_time,
                fills=fills,
                metadata={
                    'algorithm': 'Iceberg',
                    'reload_count': len(fills),
                    'avg_visible_size': visible_qty
                }
            )
            
        except Exception as e:
            logger.error(f"Iceberg execution error: {e}")
            return self._create_failed_result(order.order_id)
    
    def generate_slices(self,
                       order: ExecutionOrder,
                       market_data: Dict[str, Any]) -> List[ExecutionSlice]:
        """Generate iceberg slices"""
        slices = []
        
        remaining = order.total_quantity
        slice_count = 0
        
        while remaining > 0:
            visible_qty = min(
                order.total_quantity * self.visible_quantity,
                remaining
            )
            
            slice = ExecutionSlice(
                slice_id=f"{order.order_id}_iceberg_{slice_count}",
                parent_order_id=order.order_id,
                quantity=visible_qty,
                price=order.limit_price,
                order_type='limit',
                time_in_force='GTC',
                execute_at=datetime.now() + timedelta(seconds=slice_count * 30)
            )
            slices.append(slice)
            
            remaining -= visible_qty
            slice_count += 1
        
        return slices
    
    async def _place_visible_order(self,
                                  quantity: float,
                                  order: ExecutionOrder,
                                  order_book: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Place visible portion of iceberg order"""
        try:
            # Determine price
            if order.limit_price:
                price = order.limit_price
            else:
                # Place at best bid/ask
                if order.side == 'buy':
                    price = order_book['bids'][0][0]
                else:
                    price = order_book['asks'][0][0]
            
            # Simulate execution
            return {
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now(),
                'order_type': 'iceberg_visible'
            }
        except Exception as e:
            logger.error(f"Error placing visible order: {e}")
            return None
    
    def _estimate_market_impact(self, quantity: float, market_data: Dict[str, Any]) -> float:
        """Estimate reduced market impact due to iceberg execution"""
        avg_volume = market_data.get('avg_volume_1h', 1000)
        visible_participation = (quantity * self.visible_quantity) / avg_volume
        
        # Reduced impact due to hidden quantity
        impact = visible_participation * 0.0005  # 5 bps per 1% visible participation
        return min(impact, 0.005)  # Cap at 50 bps
    
    def _create_failed_result(self, order_id: str) -> ExecutionResult:
        """Create failed execution result"""
        return ExecutionResult(
            order_id=order_id,
            success=False,
            executed_quantity=0,
            average_price=0,
            slippage=0,
            market_impact=0,
            total_cost=0,
            execution_time=0,
            fills=[]
        )


class AdaptiveAlgorithm(ExecutionAlgorithm):
    """Adaptive execution algorithm that adjusts to market conditions"""
    
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.urgency_threshold = config.get('urgency_threshold', 0.7)
        self.impact_threshold = config.get('impact_threshold', 0.002)
        self.algorithms = {
            'twap': TWAPAlgorithm(symbol, config),
            'vwap': VWAPAlgorithm(symbol, config),
            'iceberg': IcebergAlgorithm(symbol, config)
        }
    
    async def execute(self,
                     order: ExecutionOrder,
                     market_data: Dict[str, Any],
                     order_book: Dict[str, Any]) -> ExecutionResult:
        """Execute order using adaptive algorithm selection"""
        try:
            # Select best algorithm based on conditions
            selected_algo = self._select_algorithm(order, market_data, order_book)
            
            # Execute using selected algorithm
            result = await self.algorithms[selected_algo].execute(
                order, market_data, order_book
            )
            
            # Add adaptive metadata
            result.metadata['adaptive_selection'] = selected_algo
            result.metadata['selection_reason'] = self._get_selection_reason(
                selected_algo, order, market_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Adaptive execution error: {e}")
            return self._create_failed_result(order.order_id)
    
    def generate_slices(self,
                       order: ExecutionOrder,
                       market_data: Dict[str, Any]) -> List[ExecutionSlice]:
        """Generate slices using selected algorithm"""
        selected_algo = self._select_algorithm(order, market_data, {})
        return self.algorithms[selected_algo].generate_slices(order, market_data)
    
    def _select_algorithm(self,
                         order: ExecutionOrder,
                         market_data: Dict[str, Any],
                         order_book: Dict[str, Any]) -> str:
        """Select best algorithm for current conditions"""
        
        # Check urgency
        urgency = order.metadata.get('urgency', 0.5)
        if urgency > self.urgency_threshold:
            # High urgency - use TWAP for quick execution
            return 'twap'
        
        # Check market impact sensitivity
        expected_impact = self._estimate_impact(order.total_quantity, market_data)
        if expected_impact > self.impact_threshold:
            # High impact - use iceberg to hide size
            return 'iceberg'
        
        # Check volume profile availability
        if market_data.get('volume_profile') or market_data.get('vwap'):
            # VWAP data available - use VWAP algorithm
            return 'vwap'
        
        # Default to TWAP
        return 'twap'
    
    def _estimate_impact(self, quantity: float, market_data: Dict[str, Any]) -> float:
        """Estimate market impact"""
        avg_volume = market_data.get('avg_volume_1h', 1000)
        participation = quantity / avg_volume
        return participation * 0.001  # Simple linear model
    
    def _get_selection_reason(self,
                             algorithm: str,
                             order: ExecutionOrder,
                             market_data: Dict[str, Any]) -> str:
        """Get reason for algorithm selection"""
        urgency = order.metadata.get('urgency', 0.5)
        
        if algorithm == 'twap':
            if urgency > self.urgency_threshold:
                return f"High urgency ({urgency:.2f})"
            else:
                return "Default selection"
        elif algorithm == 'vwap':
            return "Volume profile available"
        elif algorithm == 'iceberg':
            return "High expected market impact"
        else:
            return "Unknown"
    
    def _create_failed_result(self, order_id: str) -> ExecutionResult:
        """Create failed execution result"""
        return ExecutionResult(
            order_id=order_id,
            success=False,
            executed_quantity=0,
            average_price=0,
            slippage=0,
            market_impact=0,
            total_cost=0,
            execution_time=0,
            fills=[]
        )


class ExecutionEngine:
    """Main execution engine that manages all algorithms"""
    
    def __init__(self, symbol: str, config: Optional[Dict[str, Any]] = None):
        self.symbol = symbol
        self.config = config or {}
        
        # Initialize algorithms
        self.algorithms = {
            'twap': TWAPAlgorithm(symbol, self.config),
            'vwap': VWAPAlgorithm(symbol, self.config),
            'iceberg': IcebergAlgorithm(symbol, self.config),
            'adaptive': AdaptiveAlgorithm(symbol, self.config)
        }
        
        # Execution tracking
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.execution_results: deque = deque(maxlen=100)
        
        logger.info(f"ExecutionEngine initialized for {symbol}")
    
    async def execute_order(self,
                           order: ExecutionOrder,
                           market_data: Dict[str, Any],
                           order_book: Dict[str, Any],
                           algorithm: Optional[str] = None) -> ExecutionResult:
        """Execute order using specified or adaptive algorithm"""
        try:
            # Store active order
            self.active_orders[order.order_id] = order
            
            # Select algorithm
            if algorithm is None:
                algorithm = 'adaptive'
            
            if algorithm not in self.algorithms:
                logger.error(f"Unknown algorithm: {algorithm}")
                algorithm = 'adaptive'
            
            # Execute
            result = await self.algorithms[algorithm].execute(
                order, market_data, order_book
            )
            
            # Update order status
            if result.success:
                order.executed_quantity = result.executed_quantity
                order.remaining_quantity = order.total_quantity - result.executed_quantity
                
                if order.is_complete:
                    del self.active_orders[order.order_id]
            
            # Store result
            self.execution_results.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Execution engine error: {e}")
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                executed_quantity=0,
                average_price=0,
                slippage=0,
                market_impact=0,
                total_cost=0,
                execution_time=0,
                fills=[]
            )
    
    def get_active_orders(self) -> List[ExecutionOrder]:
        """Get list of active orders"""
        return list(self.active_orders.values())
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel active order"""
        if order_id in self.active_orders:
            del self.active_orders[order_id]
            logger.info(f"Order {order_id} cancelled")
            return True
        return False
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_results:
            return {}
        
        results = list(self.execution_results)
        
        successful = [r for r in results if r.success]
        
        if successful:
            avg_slippage = np.mean([r.slippage for r in successful])
            avg_impact = np.mean([r.market_impact for r in successful])
            avg_time = np.mean([r.execution_time for r in successful])
            success_rate = len(successful) / len(results)
        else:
            avg_slippage = 0
            avg_impact = 0
            avg_time = 0
            success_rate = 0
        
        return {
            'total_orders': len(results),
            'success_rate': success_rate,
            'avg_slippage': avg_slippage,
            'avg_market_impact': avg_impact,
            'avg_execution_time': avg_time,
            'active_orders': len(self.active_orders)
        }


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Initialize execution engine
        engine = ExecutionEngine(
            symbol="BTC/USDT",
            config={
                'time_slices': 5,
                'participation_rate': 0.1,
                'visible_quantity': 0.2
            }
        )
        
        # Create order
        order = ExecutionOrder(
            order_id="test_order_1",
            symbol="BTC/USDT",
            side="buy",
            total_quantity=10.0,
            limit_price=50000,
            execution_style=ExecutionStyle.ADAPTIVE
        )
        
        # Mock market data
        market_data = {
            'mid_price': 50000,
            'vwap': 49995,
            'avg_volume_1h': 100,
            'volume_profile': None
        }
        
        # Mock order book
        order_book = {
            'bids': [[49999, 1.0], [49998, 1.5], [49997, 2.0]],
            'asks': [[50001, 1.0], [50002, 1.5], [50003, 2.0]]
        }
        
        # Execute order
        result = await engine.execute_order(
            order, market_data, order_book, algorithm='adaptive'
        )
        
        print(f"Execution Result:")
        print(f"  Success: {result.success}")
        print(f"  Executed Quantity: {result.executed_quantity}")
        print(f"  Average Price: {result.average_price}")
        print(f"  Slippage: {result.slippage:.4f}")
        print(f"  Market Impact: {result.market_impact:.4f}")
        print(f"  Execution Time: {result.execution_time:.2f}s")
        print(f"  Algorithm Used: {result.metadata.get('adaptive_selection', 'N/A')}")
        
        # Get statistics
        stats = engine.get_execution_stats()
        print(f"\nExecution Statistics:")
        print(f"  {stats}")
    
    # Run example
    asyncio.run(main())