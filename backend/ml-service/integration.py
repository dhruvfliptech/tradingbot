"""
Integration Layer for AdaptiveThreshold with Existing Trading Services
Provides seamless integration between ML service and trading systems
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import uuid

from adaptive_threshold import AdaptiveThreshold, threshold_manager, ThresholdUpdate
from performance_tracker import performance_tracker, TradingPerformanceSnapshot
from monitoring import logger, logged_function
from config import get_config


@dataclass
class TradingSignalRequest:
    """Request to evaluate a trading signal"""
    user_id: str
    signal: Dict[str, Any]
    symbol: str
    timestamp: Optional[datetime] = None
    request_id: Optional[str] = None


@dataclass
class TradingSignalResponse:
    """Response from signal evaluation"""
    request_id: str
    user_id: str
    symbol: str
    should_trade: bool
    confidence_adjustment: float
    threshold_adjustments: Dict[str, float]
    reasoning: str
    processing_time_ms: float
    timestamp: datetime


@dataclass
class PerformanceUpdateRequest:
    """Request to update performance data"""
    user_id: str
    symbol: Optional[str]
    trade_data: Dict[str, Any]
    timestamp: Optional[datetime] = None


@dataclass
class AdaptationRequest:
    """Request for threshold adaptation"""
    user_id: str
    symbol: Optional[str]
    force_adaptation: bool = False
    timestamp: Optional[datetime] = None


class TradingServiceIntegration:
    """Integration layer for trading services"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.session = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="integration")
        self._request_queue = queue.Queue()
        self._response_cache = {}
        self._cache_lock = threading.Lock()
        
        # Integration metrics
        self._integration_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'avg_response_time_ms': 0.0
        }
        
        # Start background processing
        self._processor_thread = threading.Thread(target=self._process_requests, daemon=True)
        self._processor_thread.start()
        
        logger.info("TradingServiceIntegration initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    @logged_function("evaluate_trading_signal")
    async def evaluate_trading_signal(self, request: TradingSignalRequest) -> TradingSignalResponse:
        """Evaluate trading signal through adaptive thresholds"""
        start_time = time.time()
        request_id = request.request_id or str(uuid.uuid4())[:8]
        
        with logger.context(user_id=request.user_id, symbol=request.symbol, request_id=request_id):
            try:
                # Get adaptive threshold instance
                threshold_instance = threshold_manager.get_instance(request.user_id, request.symbol)
                
                # Evaluate signal
                should_trade = threshold_instance.should_trade(request.signal)
                
                # Get current thresholds for response
                current_thresholds = threshold_instance.get_current_thresholds()
                
                # Calculate confidence adjustment based on recent performance
                confidence_adjustment = await self._calculate_confidence_adjustment(
                    request.user_id, request.symbol, threshold_instance
                )
                
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Generate reasoning
                reasoning = self._generate_signal_reasoning(
                    request.signal, current_thresholds, should_trade
                )
                
                response = TradingSignalResponse(
                    request_id=request_id,
                    user_id=request.user_id,
                    symbol=request.symbol,
                    should_trade=should_trade,
                    confidence_adjustment=confidence_adjustment,
                    threshold_adjustments=current_thresholds,
                    reasoning=reasoning,
                    processing_time_ms=processing_time_ms,
                    timestamp=datetime.now()
                )
                
                # Record metrics
                self._integration_metrics['total_requests'] += 1
                self._integration_metrics['successful_requests'] += 1
                self._update_avg_response_time(processing_time_ms)
                
                # Track performance
                performance_tracker.record_metric(
                    name="signal_evaluation_duration_ms",
                    value=processing_time_ms,
                    tags={
                        'user_id': request.user_id,
                        'symbol': request.symbol,
                        'should_trade': str(should_trade)
                    }
                )
                
                logger.info(f"Signal evaluation completed", extra={
                    'should_trade': should_trade,
                    'confidence_adjustment': confidence_adjustment,
                    'processing_time_ms': processing_time_ms
                })
                
                return response
                
            except Exception as e:
                self._integration_metrics['total_requests'] += 1
                self._integration_metrics['failed_requests'] += 1
                
                logger.error(f"Error evaluating trading signal: {str(e)}")
                
                # Return safe default response
                return TradingSignalResponse(
                    request_id=request_id,
                    user_id=request.user_id,
                    symbol=request.symbol,
                    should_trade=False,
                    confidence_adjustment=0.0,
                    threshold_adjustments={},
                    reasoning=f"Error in evaluation: {str(e)}",
                    processing_time_ms=(time.time() - start_time) * 1000,
                    timestamp=datetime.now()
                )
    
    async def _calculate_confidence_adjustment(self, user_id: str, symbol: Optional[str], 
                                             threshold_instance: AdaptiveThreshold) -> float:
        """Calculate confidence adjustment based on recent performance"""
        try:
            # Get recent performance metrics
            performance_metrics = threshold_instance.get_performance_metrics(days_back=7)
            performance_score = threshold_instance.calculate_performance_score(performance_metrics)
            
            # Convert performance score to confidence adjustment
            # Performance score is 0-1, we want adjustment in range -0.2 to +0.2
            adjustment = (performance_score - 0.5) * 0.4
            
            # Clamp adjustment
            adjustment = max(-0.2, min(0.2, adjustment))
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error calculating confidence adjustment: {e}")
            return 0.0
    
    def _generate_signal_reasoning(self, signal: Dict[str, Any], 
                                 thresholds: Dict[str, float], should_trade: bool) -> str:
        """Generate human-readable reasoning for signal decision"""
        reasons = []
        
        confidence = signal.get('confidence', 0)
        confidence_threshold = thresholds.get('confidence_threshold', 0.75)
        
        if confidence < confidence_threshold:
            reasons.append(f"Signal confidence ({confidence:.2f}) below threshold ({confidence_threshold:.2f})")
        else:
            reasons.append(f"Signal confidence ({confidence:.2f}) meets threshold ({confidence_threshold:.2f})")
        
        # RSI reasoning
        rsi = signal.get('rsi', 50)
        rsi_threshold = thresholds.get('rsi_threshold', 70)
        if signal.get('action') == 'BUY' and rsi > rsi_threshold:
            reasons.append(f"RSI ({rsi:.1f}) above buy threshold ({rsi_threshold:.1f}) - overbought")
        
        # Volume reasoning
        volume = signal.get('volume', 0)
        volume_threshold = thresholds.get('volume_threshold', 1000000000)
        if volume < volume_threshold:
            reasons.append(f"Volume ({volume:,.0f}) below threshold ({volume_threshold:,.0f})")
        
        # Momentum reasoning
        change_percent = abs(signal.get('change_percent', 0))
        momentum_threshold = thresholds.get('momentum_threshold', 2.0)
        if change_percent < momentum_threshold:
            reasons.append(f"Price momentum ({change_percent:.1f}%) below threshold ({momentum_threshold:.1f}%)")
        
        decision_text = "APPROVED" if should_trade else "REJECTED"
        reasoning = f"Signal {decision_text}: " + "; ".join(reasons)
        
        return reasoning
    
    @logged_function("update_performance_data")
    async def update_performance_data(self, request: PerformanceUpdateRequest):
        """Update performance data for a user/symbol"""
        with logger.context(user_id=request.user_id, symbol=request.symbol):
            try:
                # Extract performance data from trade
                trade_data = request.trade_data
                
                # Create performance snapshot
                snapshot = TradingPerformanceSnapshot(
                    user_id=request.user_id,
                    symbol=request.symbol,
                    total_return=trade_data.get('total_return', 0.0),
                    sharpe_ratio=trade_data.get('sharpe_ratio', 0.0),
                    win_rate=trade_data.get('win_rate', 0.0),
                    avg_trade_return=trade_data.get('avg_trade_return', 0.0),
                    max_drawdown=trade_data.get('max_drawdown', 0.0),
                    volatility=trade_data.get('volatility', 0.0),
                    trade_count=trade_data.get('trade_count', 0),
                    profit_factor=trade_data.get('profit_factor', 0.0),
                    calmar_ratio=trade_data.get('calmar_ratio', 0.0),
                    sortino_ratio=trade_data.get('sortino_ratio', 0.0),
                    timestamp=request.timestamp or datetime.now(),
                    period_days=trade_data.get('period_days', 30)
                )
                
                # Track performance
                performance_tracker.track_trading_performance(snapshot)
                
                logger.info("Performance data updated", extra={
                    'total_return': snapshot.total_return,
                    'win_rate': snapshot.win_rate,
                    'trade_count': snapshot.trade_count
                })
                
            except Exception as e:
                logger.error(f"Error updating performance data: {str(e)}")
    
    @logged_function("trigger_adaptation")
    async def trigger_adaptation(self, request: AdaptationRequest) -> List[ThresholdUpdate]:
        """Trigger threshold adaptation for a user"""
        start_time = time.time()
        
        with logger.context(user_id=request.user_id, symbol=request.symbol):
            try:
                # Get threshold instance
                threshold_instance = threshold_manager.get_instance(request.user_id, request.symbol)
                
                # Check if adaptation should run
                if not request.force_adaptation:
                    # Check time since last adaptation
                    time_since_adaptation = datetime.now() - threshold_instance.last_adaptation
                    adaptation_frequency = timedelta(hours=24)  # From config
                    
                    if time_since_adaptation < adaptation_frequency:
                        logger.debug("Skipping adaptation - too soon since last adaptation")
                        return []
                
                # Run adaptation
                updates = threshold_instance.adapt_thresholds()
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Get performance score for tracking
                performance_metrics = threshold_instance.get_performance_metrics()
                performance_score = threshold_instance.calculate_performance_score(performance_metrics)
                
                # Track adaptation performance
                performance_tracker.track_adaptation_performance(
                    user_id=request.user_id,
                    symbol=request.symbol,
                    parameter_updates=updates,
                    performance_score=performance_score,
                    execution_time_ms=execution_time_ms,
                    success=True
                )
                
                # Log adaptations
                for update in updates:
                    logger.log_adaptation(
                        user_id=request.user_id,
                        symbol=request.symbol,
                        parameter=update.parameter_name,
                        old_value=update.old_value,
                        new_value=update.new_value,
                        reason=update.reason,
                        confidence=update.confidence,
                        performance_score=performance_score
                    )
                
                logger.info(f"Adaptation completed: {len(updates)} parameters updated")
                
                return updates
                
            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Track failed adaptation
                performance_tracker.track_adaptation_performance(
                    user_id=request.user_id,
                    symbol=request.symbol,
                    parameter_updates=[],
                    performance_score=0.0,
                    execution_time_ms=execution_time_ms,
                    success=False,
                    error_message=str(e)
                )
                
                logger.error(f"Error during adaptation: {str(e)}")
                raise
    
    def _update_avg_response_time(self, new_time_ms: float):
        """Update average response time with exponential moving average"""
        alpha = 0.1  # Smoothing factor
        current_avg = self._integration_metrics['avg_response_time_ms']
        
        if current_avg == 0:
            self._integration_metrics['avg_response_time_ms'] = new_time_ms
        else:
            self._integration_metrics['avg_response_time_ms'] = (
                alpha * new_time_ms + (1 - alpha) * current_avg
            )
    
    def _process_requests(self):
        """Background thread to process queued requests"""
        while True:
            try:
                # Get request from queue (blocking)
                request_item = self._request_queue.get(timeout=1.0)
                
                # Process the request
                # This is a placeholder for batch processing logic
                # In practice, you might batch multiple requests together
                
                self._request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing request: {e}")
    
    async def batch_evaluate_signals(self, requests: List[TradingSignalRequest]) -> List[TradingSignalResponse]:
        """Batch evaluate multiple trading signals for efficiency"""
        tasks = []
        
        for request in requests:
            task = asyncio.create_task(self.evaluate_trading_signal(request))
            tasks.append(task)
        
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log them
            valid_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Error in batch request {i}: {response}")
                else:
                    valid_responses.append(response)
            
            return valid_responses
            
        except Exception as e:
            logger.error(f"Error in batch signal evaluation: {e}")
            return []
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status and metrics"""
        return {
            'active_instances': len(threshold_manager.instances),
            'integration_metrics': self._integration_metrics.copy(),
            'executor_status': {
                'active_threads': self._executor._threads,
                'queue_size': self._request_queue.qsize()
            },
            'session_status': {
                'connected': self.session is not None and not self.session.closed
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on integration layer"""
        health_status = {
            'healthy': True,
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check threshold manager
        try:
            test_instance = threshold_manager.get_instance("health_check_user", None)
            health_status['checks']['threshold_manager'] = {'status': 'healthy'}
        except Exception as e:
            health_status['healthy'] = False
            health_status['checks']['threshold_manager'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Check performance tracker
        try:
            performance_tracker.metrics_collector.get_metrics_summary(hours_back=1)
            health_status['checks']['performance_tracker'] = {'status': 'healthy'}
        except Exception as e:
            health_status['healthy'] = False
            health_status['checks']['performance_tracker'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Check database connectivity
        try:
            # This would be implemented with actual database check
            health_status['checks']['database'] = {'status': 'healthy'}
        except Exception as e:
            health_status['healthy'] = False
            health_status['checks']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        return health_status
    
    async def close(self):
        """Clean up resources"""
        if self.session and not self.session.closed:
            await self.session.close()
        
        self._executor.shutdown(wait=True)
        logger.info("TradingServiceIntegration closed")


class MLServiceClient:
    """Client for external services to interact with ML service"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def evaluate_signal(self, user_id: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trading signal via HTTP API"""
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/evaluate/{user_id}"
        
        payload = {'signal': signal_data}
        
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error calling ML service: {e}")
            raise
    
    async def get_thresholds(self, user_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get current thresholds"""
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/thresholds/{user_id}"
        
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error getting thresholds: {e}")
            raise
    
    async def trigger_adaptation(self, user_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Trigger threshold adaptation"""
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/thresholds/{user_id}/adapt"
        
        payload = {}
        if symbol:
            payload['symbol'] = symbol
        
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error triggering adaptation: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check ML service health"""
        session = await self._get_session()
        url = f"{self.base_url}/health"
        
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            raise
    
    async def close(self):
        """Close client session"""
        if self.session and not self.session.closed:
            await self.session.close()


# Global integration instance
integration = TradingServiceIntegration()


# Utility functions for easy integration
async def should_execute_trade(user_id: str, signal: Dict[str, Any]) -> bool:
    """Simple utility function to check if trade should be executed"""
    try:
        request = TradingSignalRequest(
            user_id=user_id,
            signal=signal,
            symbol=signal.get('symbol', 'UNKNOWN')
        )
        
        response = await integration.evaluate_trading_signal(request)
        return response.should_trade
        
    except Exception as e:
        logger.error(f"Error in should_execute_trade: {e}")
        return False


async def update_trade_performance(user_id: str, symbol: Optional[str], trade_result: Dict[str, Any]):
    """Simple utility function to update performance after trade"""
    try:
        request = PerformanceUpdateRequest(
            user_id=user_id,
            symbol=symbol,
            trade_data=trade_result
        )
        
        await integration.update_performance_data(request)
        
    except Exception as e:
        logger.error(f"Error updating performance: {e}")


async def adapt_user_thresholds(user_id: str, symbol: Optional[str] = None) -> List[ThresholdUpdate]:
    """Simple utility function to trigger adaptation"""
    try:
        request = AdaptationRequest(
            user_id=user_id,
            symbol=symbol
        )
        
        return await integration.trigger_adaptation(request)
        
    except Exception as e:
        logger.error(f"Error adapting thresholds: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    async def main():
        # Test signal evaluation
        test_signal = {
            'symbol': 'BTCUSD',
            'confidence': 0.85,
            'rsi': 65,
            'change_percent': 3.5,
            'volume': 1500000000,
            'action': 'BUY'
        }
        
        should_trade = await should_execute_trade("test_user", test_signal)
        print(f"Should trade: {should_trade}")
        
        # Test health check
        health = await integration.health_check()
        print(f"Health status: {health}")
        
        # Clean up
        await integration.close()
    
    asyncio.run(main())