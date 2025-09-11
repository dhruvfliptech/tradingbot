"""
Decision Server - Real-time RL Decision Engine
============================================

This module implements the real-time decision serving engine for the RL system,
providing low-latency prediction serving, agent state management, and
WebSocket support for live trading decisions.

Features:
- Real-time RL model inference
- Agent state management per user
- WebSocket streaming for live predictions
- Prediction caching and batching
- Load balancing across multiple agents
- Performance monitoring and metrics
- Graceful fallback handling
"""

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import threading
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class AgentState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class PredictionStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class PredictionRequest:
    """Individual prediction request"""
    request_id: str
    user_id: str
    symbol: str
    observation: np.ndarray
    market_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1=high, 2=medium, 3=low
    timeout_seconds: float = 5.0

@dataclass
class PredictionResult:
    """Prediction result with metadata"""
    request_id: str
    user_id: str
    symbol: str
    action: np.ndarray
    confidence: float
    reasoning: str
    model_info: Dict[str, Any]
    processing_time_ms: float
    timestamp: datetime
    status: PredictionStatus

@dataclass
class AgentInstance:
    """Agent instance for a specific user"""
    user_id: str
    state: AgentState
    ensemble_agent: Optional[Any] = None  # EnsembleAgent instance
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    prediction_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

class DecisionServer:
    """
    Real-time decision serving engine for RL models
    
    Provides:
    - Low-latency prediction serving
    - Agent state management
    - Request queuing and prioritization
    - Performance monitoring
    - WebSocket streaming
    - Load balancing
    """
    
    def __init__(self, max_concurrent_predictions: int = 50):
        self.max_concurrent_predictions = max_concurrent_predictions
        
        # Agent instances per user
        self.agent_instances: Dict[str, AgentInstance] = {}
        
        # Prediction queues
        self.prediction_queue = asyncio.Queue()
        self.pending_predictions: Dict[str, PredictionRequest] = {}
        self.completed_predictions: Dict[str, PredictionResult] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'avg_processing_time_ms': 0.0,
            'current_queue_size': 0,
            'active_agents': 0
        }
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_predictions)
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Caching
        self.prediction_cache: Dict[str, Tuple[PredictionResult, datetime]] = {}
        self.cache_ttl_seconds = 30  # Cache predictions for 30 seconds
        
        # WebSocket connections for streaming
        self.websocket_connections: Dict[str, List[Any]] = defaultdict(list)
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        logger.info("DecisionServer initialized")
    
    async def start(self):
        """Start the decision server"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._prediction_processor()),
            asyncio.create_task(self._cleanup_task()),
            asyncio.create_task(self._metrics_collector())
        ]
        
        logger.info("DecisionServer started")
    
    async def stop(self):
        """Stop the decision server"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Stop all agent instances
        for user_id in list(self.agent_instances.keys()):
            await self.stop_agent(user_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("DecisionServer stopped")
    
    async def start_agent(self, user_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start RL agent for a user"""
        try:
            if user_id in self.agent_instances:
                instance = self.agent_instances[user_id]
                if instance.state == AgentState.RUNNING:
                    return {"status": "already_running", "user_id": user_id}
                elif instance.state == AgentState.PAUSED:
                    # Resume paused agent
                    instance.state = AgentState.RUNNING
                    instance.last_activity = datetime.now()
                    return {"status": "resumed", "user_id": user_id}
            
            # Create new agent instance
            instance = AgentInstance(
                user_id=user_id,
                state=AgentState.STARTING,
                config=config or {}
            )
            
            self.agent_instances[user_id] = instance
            
            # Initialize ensemble agent (would be loaded from models)
            # For now, we'll mark as running without actually loading
            # In production, this would load the pre-trained ensemble
            instance.state = AgentState.RUNNING
            instance.last_activity = datetime.now()
            
            self._update_metrics()
            
            logger.info(f"Started RL agent for user {user_id}")
            
            return {
                "status": "started",
                "user_id": user_id,
                "config": instance.config,
                "created_at": instance.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to start agent for user {user_id}: {e}")
            
            if user_id in self.agent_instances:
                self.agent_instances[user_id].state = AgentState.ERROR
                self.agent_instances[user_id].last_error = str(e)
            
            return {"status": "error", "user_id": user_id, "error": str(e)}
    
    async def stop_agent(self, user_id: str) -> Dict[str, Any]:
        """Stop RL agent for a user"""
        try:
            if user_id not in self.agent_instances:
                return {"status": "not_found", "user_id": user_id}
            
            instance = self.agent_instances[user_id]
            
            # Cancel any pending predictions for this user
            await self._cancel_user_predictions(user_id)
            
            # Update state
            instance.state = AgentState.STOPPED
            instance.last_activity = datetime.now()
            
            # Remove instance after a delay (keep for historical data)
            # In production, you might want to persist this data
            
            self._update_metrics()
            
            logger.info(f"Stopped RL agent for user {user_id}")
            
            return {
                "status": "stopped",
                "user_id": user_id,
                "final_stats": {
                    "prediction_count": instance.prediction_count,
                    "error_count": instance.error_count,
                    "uptime_seconds": (datetime.now() - instance.created_at).total_seconds()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to stop agent for user {user_id}: {e}")
            return {"status": "error", "user_id": user_id, "error": str(e)}
    
    async def pause_agent(self, user_id: str) -> Dict[str, Any]:
        """Pause RL agent for a user"""
        if user_id not in self.agent_instances:
            return {"status": "not_found", "user_id": user_id}
        
        instance = self.agent_instances[user_id]
        if instance.state != AgentState.RUNNING:
            return {"status": "not_running", "user_id": user_id, "current_state": instance.state.value}
        
        instance.state = AgentState.PAUSED
        instance.last_activity = datetime.now()
        
        self._update_metrics()
        
        logger.info(f"Paused RL agent for user {user_id}")
        
        return {"status": "paused", "user_id": user_id}
    
    async def resume_agent(self, user_id: str) -> Dict[str, Any]:
        """Resume paused RL agent for a user"""
        if user_id not in self.agent_instances:
            return {"status": "not_found", "user_id": user_id}
        
        instance = self.agent_instances[user_id]
        if instance.state != AgentState.PAUSED:
            return {"status": "not_paused", "user_id": user_id, "current_state": instance.state.value}
        
        instance.state = AgentState.RUNNING
        instance.last_activity = datetime.now()
        
        self._update_metrics()
        
        logger.info(f"Resumed RL agent for user {user_id}")
        
        return {"status": "resumed", "user_id": user_id}
    
    async def get_agent_status(self, user_id: str) -> Dict[str, Any]:
        """Get status of RL agent for a user"""
        if user_id not in self.agent_instances:
            return {"status": "not_found", "user_id": user_id}
        
        instance = self.agent_instances[user_id]
        
        return {
            "user_id": user_id,
            "state": instance.state.value,
            "created_at": instance.created_at.isoformat(),
            "last_activity": instance.last_activity.isoformat(),
            "prediction_count": instance.prediction_count,
            "error_count": instance.error_count,
            "last_error": instance.last_error,
            "config": instance.config,
            "uptime_seconds": (datetime.now() - instance.created_at).total_seconds(),
            "is_active": instance.state == AgentState.RUNNING
        }
    
    async def predict_async(self, user_id: str, symbol: str, observation: np.ndarray,
                          market_data: Optional[Dict[str, Any]] = None,
                          priority: int = 1, timeout_seconds: float = 5.0) -> str:
        """Submit async prediction request and return request ID"""
        
        # Check if agent is running
        if user_id not in self.agent_instances:
            raise ValueError(f"No agent instance for user {user_id}")
        
        instance = self.agent_instances[user_id]
        if instance.state != AgentState.RUNNING:
            raise ValueError(f"Agent for user {user_id} is not running (state: {instance.state.value})")
        
        # Check cache first
        cache_key = self._generate_cache_key(user_id, symbol, observation)
        cached_result = self._get_cached_prediction(cache_key)
        if cached_result:
            return cached_result.request_id
        
        # Create prediction request
        request_id = str(uuid.uuid4())
        request = PredictionRequest(
            request_id=request_id,
            user_id=user_id,
            symbol=symbol,
            observation=observation,
            market_data=market_data,
            priority=priority,
            timeout_seconds=timeout_seconds
        )
        
        # Add to pending predictions
        self.pending_predictions[request_id] = request
        
        # Queue for processing
        await self.prediction_queue.put(request)
        
        # Update metrics
        self.performance_metrics['current_queue_size'] = self.prediction_queue.qsize()
        
        logger.debug(f"Queued prediction request {request_id} for user {user_id}")
        
        return request_id
    
    async def get_prediction_result(self, request_id: str, timeout_seconds: float = 10.0) -> Optional[PredictionResult]:
        """Get prediction result by request ID"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            # Check completed predictions
            if request_id in self.completed_predictions:
                result = self.completed_predictions[request_id]
                
                # Clean up after returning
                if request_id in self.pending_predictions:
                    del self.pending_predictions[request_id]
                
                return result
            
            # Check if request is still pending
            if request_id not in self.pending_predictions:
                break
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        # Timeout or request not found
        if request_id in self.pending_predictions:
            logger.warning(f"Prediction request {request_id} timed out")
            del self.pending_predictions[request_id]
        
        return None
    
    async def predict_sync(self, user_id: str, symbol: str, observation: np.ndarray,
                         market_data: Optional[Dict[str, Any]] = None,
                         timeout_seconds: float = 5.0) -> Optional[PredictionResult]:
        """Submit prediction request and wait for result"""
        request_id = await self.predict_async(user_id, symbol, observation, market_data, 
                                            priority=1, timeout_seconds=timeout_seconds)
        return await self.get_prediction_result(request_id, timeout_seconds)
    
    def _generate_cache_key(self, user_id: str, symbol: str, observation: np.ndarray) -> str:
        """Generate cache key for prediction"""
        obs_hash = hash(observation.tobytes())
        return f"{user_id}_{symbol}_{obs_hash}"
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[PredictionResult]:
        """Get cached prediction if valid"""
        if cache_key in self.prediction_cache:
            result, timestamp = self.prediction_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl_seconds):
                return result
            else:
                # Remove expired cache entry
                del self.prediction_cache[cache_key]
        return None
    
    def _cache_prediction(self, cache_key: str, result: PredictionResult):
        """Cache prediction result"""
        self.prediction_cache[cache_key] = (result, datetime.now())
        
        # Limit cache size
        if len(self.prediction_cache) > 1000:
            # Remove oldest entries
            sorted_items = sorted(self.prediction_cache.items(), 
                                key=lambda x: x[1][1])
            for i in range(100):  # Remove 100 oldest entries
                del self.prediction_cache[sorted_items[i][0]]
    
    async def _prediction_processor(self):
        """Background task to process prediction requests"""
        while self.is_running:
            try:
                # Get request from queue with timeout
                try:
                    request = await asyncio.wait_for(self.prediction_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Check if request is still valid
                if request.request_id not in self.pending_predictions:
                    continue
                
                # Check timeout
                if (datetime.now() - request.timestamp).total_seconds() > request.timeout_seconds:
                    logger.warning(f"Prediction request {request.request_id} expired")
                    self._complete_prediction_with_error(request, "Request timeout")
                    continue
                
                # Process the prediction
                task = asyncio.create_task(self._process_single_prediction(request))
                self.processing_tasks[request.request_id] = task
                
                # Update queue size metric
                self.performance_metrics['current_queue_size'] = self.prediction_queue.qsize()
                
            except Exception as e:
                logger.error(f"Error in prediction processor: {e}")
                await asyncio.sleep(1)
    
    async def _process_single_prediction(self, request: PredictionRequest):
        """Process a single prediction request"""
        start_time = time.time()
        
        try:
            # Get agent instance
            instance = self.agent_instances.get(request.user_id)
            if not instance or instance.state != AgentState.RUNNING:
                self._complete_prediction_with_error(request, "Agent not available")
                return
            
            # Simulate RL prediction (in production, use actual ensemble agent)
            action, confidence, reasoning, model_info = await self._run_rl_inference(
                request.observation, request.market_data
            )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = PredictionResult(
                request_id=request.request_id,
                user_id=request.user_id,
                symbol=request.symbol,
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                model_info=model_info,
                processing_time_ms=processing_time_ms,
                timestamp=datetime.now(),
                status=PredictionStatus.COMPLETED
            )
            
            # Cache result
            cache_key = self._generate_cache_key(request.user_id, request.symbol, request.observation)
            self._cache_prediction(cache_key, result)
            
            # Store completed prediction
            self.completed_predictions[request.request_id] = result
            
            # Update instance stats
            instance.prediction_count += 1
            instance.last_activity = datetime.now()
            
            # Update metrics
            self._update_prediction_metrics(True, processing_time_ms)
            
            # Notify WebSocket connections
            await self._notify_websocket_connections(request.user_id, result)
            
            logger.debug(f"Completed prediction {request.request_id} in {processing_time_ms:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error processing prediction {request.request_id}: {e}")
            self._complete_prediction_with_error(request, str(e))
        
        finally:
            # Clean up processing task
            if request.request_id in self.processing_tasks:
                del self.processing_tasks[request.request_id]
    
    async def _run_rl_inference(self, observation: np.ndarray, 
                              market_data: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, float, str, Dict[str, Any]]:
        """Run RL model inference (simplified simulation)"""
        
        # In production, this would use the actual ensemble agent
        # For now, simulate the prediction
        
        await asyncio.sleep(0.01)  # Simulate model inference time
        
        # Simulate ensemble prediction
        action_value = np.tanh(np.mean(observation) * 10)  # Convert to action between -1 and 1
        action = np.array([action_value])
        
        # Simulate confidence based on observation variance
        confidence = max(0.1, min(0.9, 1.0 - np.std(observation)))
        
        # Simulate reasoning
        if action_value > 0.3:
            reasoning = f"RL ensemble recommends BUY with {confidence:.2f} confidence based on favorable market signals"
        elif action_value < -0.3:
            reasoning = f"RL ensemble recommends SELL with {confidence:.2f} confidence based on bearish indicators"
        else:
            reasoning = f"RL ensemble recommends HOLD with {confidence:.2f} confidence in uncertain market conditions"
        
        # Simulate model info
        model_info = {
            "ensemble_agents": 4,
            "dominant_agent": "bull_specialist" if action_value > 0 else "bear_specialist" if action_value < 0 else "sideways_specialist",
            "regime_detected": "bull" if action_value > 0.2 else "bear" if action_value < -0.2 else "sideways",
            "observation_features": len(observation)
        }
        
        return action, confidence, reasoning, model_info
    
    def _complete_prediction_with_error(self, request: PredictionRequest, error_message: str):
        """Complete prediction request with error"""
        result = PredictionResult(
            request_id=request.request_id,
            user_id=request.user_id,
            symbol=request.symbol,
            action=np.array([0.0]),  # Hold action
            confidence=0.0,
            reasoning=f"Error: {error_message}",
            model_info={"error": error_message},
            processing_time_ms=0.0,
            timestamp=datetime.now(),
            status=PredictionStatus.FAILED
        )
        
        self.completed_predictions[request.request_id] = result
        
        # Update instance error count
        if request.user_id in self.agent_instances:
            instance = self.agent_instances[request.user_id]
            instance.error_count += 1
            instance.last_error = error_message
        
        # Update metrics
        self._update_prediction_metrics(False, 0)
    
    def _update_prediction_metrics(self, success: bool, processing_time_ms: float):
        """Update prediction performance metrics"""
        self.performance_metrics['total_predictions'] += 1
        
        if success:
            self.performance_metrics['successful_predictions'] += 1
            
            # Update average processing time
            total_successful = self.performance_metrics['successful_predictions']
            current_avg = self.performance_metrics['avg_processing_time_ms']
            self.performance_metrics['avg_processing_time_ms'] = (
                (current_avg * (total_successful - 1) + processing_time_ms) / total_successful
            )
        else:
            self.performance_metrics['failed_predictions'] += 1
    
    def _update_metrics(self):
        """Update general metrics"""
        active_agents = sum(1 for instance in self.agent_instances.values() 
                          if instance.state == AgentState.RUNNING)
        self.performance_metrics['active_agents'] = active_agents
    
    async def _cancel_user_predictions(self, user_id: str):
        """Cancel all pending predictions for a user"""
        to_remove = []
        
        for request_id, request in self.pending_predictions.items():
            if request.user_id == user_id:
                to_remove.append(request_id)
        
        for request_id in to_remove:
            if request_id in self.pending_predictions:
                request = self.pending_predictions[request_id]
                self._complete_prediction_with_error(request, "Agent stopped")
                del self.pending_predictions[request_id]
            
            # Cancel processing task if exists
            if request_id in self.processing_tasks:
                self.processing_tasks[request_id].cancel()
                del self.processing_tasks[request_id]
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Clean up old completed predictions (keep for 10 minutes)
                to_remove = []
                for request_id, result in self.completed_predictions.items():
                    if (current_time - result.timestamp).total_seconds() > 600:
                        to_remove.append(request_id)
                
                for request_id in to_remove:
                    del self.completed_predictions[request_id]
                
                # Clean up old cache entries
                cache_to_remove = []
                for cache_key, (result, timestamp) in self.prediction_cache.items():
                    if (current_time - timestamp).total_seconds() > self.cache_ttl_seconds:
                        cache_to_remove.append(cache_key)
                
                for cache_key in cache_to_remove:
                    del self.prediction_cache[cache_key]
                
                # Clean up inactive agents (optional)
                inactive_threshold = timedelta(hours=1)
                agents_to_cleanup = []
                for user_id, instance in self.agent_instances.items():
                    if (instance.state == AgentState.STOPPED and 
                        current_time - instance.last_activity > inactive_threshold):
                        agents_to_cleanup.append(user_id)
                
                for user_id in agents_to_cleanup:
                    del self.agent_instances[user_id]
                    logger.info(f"Cleaned up inactive agent for user {user_id}")
                
                await asyncio.sleep(60)  # Run cleanup every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collector(self):
        """Background metrics collection task"""
        while self.is_running:
            try:
                self._update_metrics()
                await asyncio.sleep(30)  # Update metrics every 30 seconds
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(30)
    
    async def _notify_websocket_connections(self, user_id: str, result: PredictionResult):
        """Notify WebSocket connections of new prediction result"""
        if user_id in self.websocket_connections:
            message = {
                "type": "prediction_result",
                "data": asdict(result)
            }
            
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(message["data"]["action"], np.ndarray):
                message["data"]["action"] = message["data"]["action"].tolist()
            
            disconnected_connections = []
            
            for websocket in self.websocket_connections[user_id]:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send WebSocket message to user {user_id}: {e}")
                    disconnected_connections.append(websocket)
            
            # Remove disconnected connections
            for websocket in disconnected_connections:
                self.websocket_connections[user_id].remove(websocket)
            
            # Clean up empty lists
            if not self.websocket_connections[user_id]:
                del self.websocket_connections[user_id]
    
    def add_websocket_connection(self, user_id: str, websocket: Any):
        """Add WebSocket connection for user"""
        self.websocket_connections[user_id].append(websocket)
        logger.info(f"Added WebSocket connection for user {user_id}")
    
    def remove_websocket_connection(self, user_id: str, websocket: Any):
        """Remove WebSocket connection for user"""
        if user_id in self.websocket_connections:
            try:
                self.websocket_connections[user_id].remove(websocket)
                if not self.websocket_connections[user_id]:
                    del self.websocket_connections[user_id]
                logger.info(f"Removed WebSocket connection for user {user_id}")
            except ValueError:
                pass  # Connection already removed
    
    async def health_check(self) -> str:
        """Check health of DecisionServer"""
        try:
            if not self.is_running:
                return "unhealthy"
            
            # Check queue size
            queue_size = self.prediction_queue.qsize()
            if queue_size > self.max_concurrent_predictions * 2:
                return "degraded"
            
            # Check error rate
            total_predictions = self.performance_metrics['total_predictions']
            if total_predictions > 100:  # Only check if we have enough data
                error_rate = self.performance_metrics['failed_predictions'] / total_predictions
                if error_rate > 0.1:  # More than 10% error rate
                    return "degraded"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"DecisionServer health check failed: {e}")
            return "unhealthy"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        queue_size = self.prediction_queue.qsize()
        
        return {
            **self.performance_metrics,
            'current_queue_size': queue_size,
            'pending_predictions': len(self.pending_predictions),
            'completed_predictions': len(self.completed_predictions),
            'cache_size': len(self.prediction_cache),
            'active_websockets': sum(len(conns) for conns in self.websocket_connections.values()),
            'processing_tasks': len(self.processing_tasks)
        }
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a specific user"""
        if user_id not in self.agent_instances:
            return {"error": "User not found"}
        
        instance = self.agent_instances[user_id]
        
        # Calculate recent performance
        user_predictions = [result for result in self.completed_predictions.values() 
                          if result.user_id == user_id]
        
        recent_predictions = [p for p in user_predictions 
                            if (datetime.now() - p.timestamp).total_seconds() < 3600]  # Last hour
        
        recent_success_rate = 0.0
        if recent_predictions:
            successful = len([p for p in recent_predictions if p.status == PredictionStatus.COMPLETED])
            recent_success_rate = successful / len(recent_predictions)
        
        return {
            "user_id": user_id,
            "state": instance.state.value,
            "total_predictions": instance.prediction_count,
            "total_errors": instance.error_count,
            "recent_predictions_1h": len(recent_predictions),
            "recent_success_rate": recent_success_rate,
            "uptime_seconds": (datetime.now() - instance.created_at).total_seconds(),
            "last_activity": instance.last_activity.isoformat(),
            "websocket_connections": len(self.websocket_connections.get(user_id, []))
        }


# Factory function
def create_decision_server(max_concurrent_predictions: int = 50) -> DecisionServer:
    """Create DecisionServer instance"""
    return DecisionServer(max_concurrent_predictions)


if __name__ == "__main__":
    # Test the decision server
    async def test_decision_server():
        server = create_decision_server()
        
        try:
            await server.start()
            
            # Start an agent
            result = await server.start_agent("test_user", {"model": "ensemble"})
            print(f"Start agent result: {result}")
            
            # Submit a prediction
            observation = np.random.randn(50).astype(np.float32)
            request_id = await server.predict_async("test_user", "BTC", observation)
            print(f"Prediction request ID: {request_id}")
            
            # Wait for result
            result = await server.get_prediction_result(request_id, timeout_seconds=10)
            if result:
                print(f"Prediction result: action={result.action}, confidence={result.confidence}")
            else:
                print("Prediction timed out")
            
            # Get metrics
            metrics = server.get_performance_metrics()
            print(f"Performance metrics: {json.dumps(metrics, indent=2)}")
            
            # Stop agent
            stop_result = await server.stop_agent("test_user")
            print(f"Stop agent result: {stop_result}")
            
        finally:
            await server.stop()
    
    # Run test
    asyncio.run(test_decision_server())