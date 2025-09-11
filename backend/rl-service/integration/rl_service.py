"""
RL Service - Main Integration Service
====================================

Main FastAPI service that provides the integration layer between the RL system
and existing trading infrastructure. Features:

- REST API for RL agent management
- Real-time decision serving
- WebSocket support for live predictions
- A/B testing capabilities
- Fallback mechanisms
- Performance monitoring
- Health checks and metrics
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .trading_bridge import TradingBridge
from .data_connector import DataConnector
from .decision_server import DecisionServer
from .monitoring import RLMonitor, PerformanceTracker
from ..agents.ensemble_agent import EnsembleAgent, EnsembleConfig
from ..environment.trading_env import TradingEnvironment
from ..rl_config import get_rl_config

logger = logging.getLogger(__name__)

# Pydantic models for API
class TradingSignalRequest(BaseModel):
    user_id: str
    symbol: str
    signal_data: Dict[str, Any]
    market_data: Optional[Dict[str, Any]] = None

class TradingSignalResponse(BaseModel):
    should_trade: bool
    confidence: float
    action: str
    reasoning: str
    model_used: str
    fallback_used: bool = False

class AgentControlRequest(BaseModel):
    user_id: str
    action: str  # 'start', 'stop', 'pause', 'resume'
    config: Optional[Dict[str, Any]] = None

class ModelUpdateRequest(BaseModel):
    model_type: str  # 'ensemble', 'individual'
    agent_name: Optional[str] = None
    training_data: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None

class PerformanceMetrics(BaseModel):
    total_predictions: int
    accuracy: float
    sharpe_ratio: float
    total_return: float
    win_rate: float
    avg_response_time_ms: float
    model_uptime_percent: float
    last_updated: datetime

class HealthStatus(BaseModel):
    status: str  # 'healthy', 'degraded', 'unhealthy'
    services: Dict[str, str]
    metrics: PerformanceMetrics
    alerts: List[str]
    timestamp: datetime

@dataclass
class RLServiceConfig:
    """Configuration for RL service"""
    # Service settings
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = False
    
    # Model settings
    models_path: str = "/app/models"
    default_ensemble_config: Dict[str, Any] = None
    
    # Performance settings
    max_concurrent_requests: int = 100
    prediction_timeout_seconds: int = 5.0
    
    # A/B testing
    ab_testing_enabled: bool = True
    rl_traffic_percentage: float = 0.1  # Start with 10% traffic to RL
    
    # Fallback settings
    fallback_to_adaptive_threshold: bool = True
    adaptive_threshold_url: str = "http://ml-service:5000"
    
    # Monitoring
    metrics_retention_hours: int = 24
    performance_check_interval_minutes: int = 5
    
    # External services
    trading_service_url: str = "http://backend:3000"
    data_aggregator_url: str = "http://backend:3000"


class RLService:
    """
    Main RL Service class that orchestrates the integration layer
    """
    
    def __init__(self, config: RLServiceConfig):
        self.config = config
        self.app = FastAPI(
            title="RL Trading Service",
            description="Reinforcement Learning integration for trading system",
            version="1.0.0"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self.rl_config = get_rl_config()
        self.trading_bridge = TradingBridge(config.trading_service_url)
        self.data_connector = DataConnector(config.data_aggregator_url)
        self.decision_server = DecisionServer()
        self.monitor = RLMonitor()
        self.performance_tracker = PerformanceTracker()
        
        # RL Environment and Agent
        self.env = None
        self.ensemble_agent: Optional[EnsembleAgent] = None
        
        # WebSocket connections
        self.websocket_connections: Dict[str, WebSocket] = {}
        
        # Service state
        self.is_ready = False
        self.last_health_check = datetime.now()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("RL Service initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.on_event("startup")
        async def startup():
            await self._initialize_service()
        
        @self.app.on_event("shutdown")
        async def shutdown():
            await self._cleanup_service()
        
        # Health and status endpoints
        @self.app.get("/health", response_model=HealthStatus)
        async def health_check():
            return await self._get_health_status()
        
        @self.app.get("/status")
        async def service_status():
            return {
                "service": "rl-service",
                "version": "1.0.0",
                "is_ready": self.is_ready,
                "uptime_seconds": (datetime.now() - self.last_health_check).total_seconds(),
                "ensemble_loaded": self.ensemble_agent is not None,
                "active_connections": len(self.websocket_connections)
            }
        
        # Main prediction endpoint
        @self.app.post("/api/v1/predict", response_model=TradingSignalResponse)
        async def predict_trading_signal(request: TradingSignalRequest):
            return await self._handle_prediction_request(request)
        
        # Agent control endpoints
        @self.app.post("/api/v1/agents/control")
        async def control_agent(request: AgentControlRequest):
            return await self._handle_agent_control(request)
        
        @self.app.get("/api/v1/agents/status/{user_id}")
        async def get_agent_status(user_id: str):
            return await self._get_agent_status(user_id)
        
        # Model management endpoints
        @self.app.post("/api/v1/models/update")
        async def update_model(request: ModelUpdateRequest, background_tasks: BackgroundTasks):
            background_tasks.add_task(self._handle_model_update, request)
            return {"message": "Model update initiated", "request_id": int(time.time())}
        
        @self.app.get("/api/v1/models/performance")
        async def get_model_performance():
            return await self._get_model_performance()
        
        # A/B testing endpoints
        @self.app.post("/api/v1/ab-test/configure")
        async def configure_ab_test(traffic_percentage: float):
            if not 0 <= traffic_percentage <= 1:
                raise HTTPException(status_code=400, detail="Traffic percentage must be between 0 and 1")
            self.config.rl_traffic_percentage = traffic_percentage
            return {"message": f"A/B test configured: {traffic_percentage*100}% traffic to RL"}
        
        @self.app.get("/api/v1/ab-test/status")
        async def get_ab_test_status():
            return {
                "enabled": self.config.ab_testing_enabled,
                "rl_traffic_percentage": self.config.rl_traffic_percentage,
                "total_requests": self.performance_tracker.get_total_requests(),
                "rl_requests": self.performance_tracker.get_rl_requests()
            }
        
        # Metrics and monitoring
        @self.app.get("/api/v1/metrics")
        async def get_metrics():
            return await self._get_comprehensive_metrics()
        
        @self.app.get("/api/v1/metrics/performance")
        async def get_performance_metrics():
            return self.performance_tracker.get_metrics()
        
        # WebSocket endpoint for real-time predictions
        @self.app.websocket("/ws/predictions/{user_id}")
        async def websocket_predictions(websocket: WebSocket, user_id: str):
            await self._handle_websocket_connection(websocket, user_id)
    
    async def _initialize_service(self):
        """Initialize the RL service components"""
        try:
            logger.info("Initializing RL service...")
            
            # Initialize environment
            self.env = TradingEnvironment(config=self.rl_config, mode='live')
            
            # Load pre-trained ensemble agent
            await self._load_ensemble_agent()
            
            # Initialize monitoring
            await self.monitor.initialize()
            
            # Start background tasks
            asyncio.create_task(self._background_health_monitor())
            asyncio.create_task(self._background_performance_tracker())
            
            self.is_ready = True
            logger.info("RL service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RL service: {e}")
            raise
    
    async def _load_ensemble_agent(self):
        """Load pre-trained ensemble agent"""
        try:
            ensemble_config = EnsembleConfig(**(self.config.default_ensemble_config or {}))
            self.ensemble_agent = EnsembleAgent(self.env, ensemble_config)
            
            # Try to load pre-trained models
            models_path = Path(self.config.models_path)
            if models_path.exists():
                self.ensemble_agent.load_ensemble(str(models_path))
                logger.info("Pre-trained ensemble agent loaded")
            else:
                logger.warning("No pre-trained models found, using default agent")
                
        except Exception as e:
            logger.error(f"Failed to load ensemble agent: {e}")
            # Continue without ensemble agent - will fallback to adaptive threshold
    
    async def _handle_prediction_request(self, request: TradingSignalRequest) -> TradingSignalResponse:
        """Handle trading signal prediction request"""
        start_time = time.time()
        
        try:
            # A/B testing decision
            use_rl = self._should_use_rl(request.user_id)
            
            if use_rl and self.ensemble_agent is not None:
                # Use RL model
                response = await self._get_rl_prediction(request)
                response.model_used = "rl_ensemble"
                self.performance_tracker.record_request('rl')
            else:
                # Fallback to adaptive threshold
                response = await self._get_fallback_prediction(request)
                response.model_used = "adaptive_threshold"
                response.fallback_used = True
                self.performance_tracker.record_request('fallback')
            
            # Record performance metrics
            response_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_response_time(response_time)
            
            # Send to monitoring
            await self.monitor.log_prediction(request, response, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling prediction request: {e}")
            
            # Emergency fallback
            response = TradingSignalResponse(
                should_trade=False,
                confidence=0.0,
                action="HOLD",
                reasoning=f"Error in prediction: {str(e)}",
                model_used="error_fallback",
                fallback_used=True
            )
            
            response_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_response_time(response_time)
            self.performance_tracker.record_error()
            
            return response
    
    def _should_use_rl(self, user_id: str) -> bool:
        """A/B testing logic to determine if RL should be used"""
        if not self.config.ab_testing_enabled:
            return True
        
        # Simple hash-based A/B testing
        user_hash = hash(user_id) % 100
        return user_hash < (self.config.rl_traffic_percentage * 100)
    
    async def _get_rl_prediction(self, request: TradingSignalRequest) -> TradingSignalResponse:
        """Get prediction from RL ensemble agent"""
        try:
            # Prepare observation from signal data
            observation = self._prepare_observation(request.signal_data)
            
            # Get market data for regime detection
            market_data = await self.data_connector.get_recent_market_data(
                request.symbol, window_hours=24
            )
            
            # Get ensemble prediction
            action, info = self.ensemble_agent.predict(observation, market_data)
            
            # Convert to trading signal response
            action_value = action[0] if isinstance(action, np.ndarray) else action
            
            # Map action to trading decision
            if action_value > 0.6:
                trade_action = "BUY"
                should_trade = True
            elif action_value < -0.6:
                trade_action = "SELL"
                should_trade = True
            else:
                trade_action = "HOLD"
                should_trade = False
            
            # Calculate confidence based on regime confidence and ensemble agreement
            regime_confidence = info.get('regime_info', {}).get('confidence', 0.5)
            ensemble_confidence = self._calculate_ensemble_confidence(info)
            confidence = (regime_confidence + ensemble_confidence) / 2
            
            # Generate reasoning
            regime_name = info.get('regime_info', {}).get('regime_name', 'unknown')
            agent_weights = info.get('agent_weights', [])
            dominant_agent = self.ensemble_agent.config.agent_types[np.argmax(agent_weights)] if agent_weights else 'unknown'
            
            reasoning = f"RL ensemble prediction: {trade_action} with {confidence:.2f} confidence. " \
                       f"Market regime: {regime_name}, dominant agent: {dominant_agent}"
            
            return TradingSignalResponse(
                should_trade=should_trade,
                confidence=confidence,
                action=trade_action,
                reasoning=reasoning,
                model_used="rl_ensemble"
            )
            
        except Exception as e:
            logger.error(f"RL prediction error: {e}")
            raise
    
    async def _get_fallback_prediction(self, request: TradingSignalRequest) -> TradingSignalResponse:
        """Get prediction from adaptive threshold service"""
        try:
            # Call adaptive threshold service
            prediction = await self.trading_bridge.get_adaptive_threshold_prediction(
                request.user_id, request.signal_data
            )
            
            return TradingSignalResponse(
                should_trade=prediction.get('should_trade', False),
                confidence=prediction.get('confidence', 0.5),
                action=prediction.get('action', 'HOLD'),
                reasoning=prediction.get('reasoning', 'Adaptive threshold decision'),
                model_used="adaptive_threshold",
                fallback_used=True
            )
            
        except Exception as e:
            logger.error(f"Fallback prediction error: {e}")
            # Final fallback - conservative decision
            return TradingSignalResponse(
                should_trade=False,
                confidence=0.0,
                action="HOLD",
                reasoning="Error fallback - hold position",
                model_used="error_fallback",
                fallback_used=True
            )
    
    def _prepare_observation(self, signal_data: Dict[str, Any]) -> np.ndarray:
        """Convert signal data to environment observation"""
        # Extract key features for RL model
        features = [
            signal_data.get('confidence', 0.5),
            signal_data.get('rsi', 50.0) / 100.0,  # Normalize RSI
            signal_data.get('change_percent', 0.0) / 100.0,  # Normalize percentage
            signal_data.get('volume', 0.0) / 1e9,  # Normalize volume
            signal_data.get('macd', 0.0),
            signal_data.get('sma_ratio', 1.0),
            signal_data.get('bollinger_position', 0.5),
            signal_data.get('atr', 0.0) / 100.0
        ]
        
        # Pad or truncate to expected observation size
        observation_size = 50  # Assuming 50-dimensional observation space
        if len(features) < observation_size:
            features.extend([0.0] * (observation_size - len(features)))
        elif len(features) > observation_size:
            features = features[:observation_size]
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_ensemble_confidence(self, ensemble_info: Dict[str, Any]) -> float:
        """Calculate confidence based on ensemble agreement"""
        agent_weights = ensemble_info.get('agent_weights', [])
        if not agent_weights:
            return 0.5
        
        # Confidence based on weight distribution
        # High entropy (even distribution) = low confidence
        # Low entropy (concentrated weights) = high confidence
        weights = np.array(agent_weights)
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(len(weights))
        
        # Convert entropy to confidence (0 = max entropy/low confidence, 1 = min entropy/high confidence)
        confidence = 1.0 - (entropy / max_entropy)
        return max(0.0, min(1.0, confidence))
    
    async def _handle_agent_control(self, request: AgentControlRequest) -> Dict[str, Any]:
        """Handle agent control requests"""
        try:
            if request.action == "start":
                # Start RL agent for user
                result = await self.decision_server.start_agent(request.user_id, request.config)
                
            elif request.action == "stop":
                # Stop RL agent for user
                result = await self.decision_server.stop_agent(request.user_id)
                
            elif request.action == "pause":
                # Pause RL agent
                result = await self.decision_server.pause_agent(request.user_id)
                
            elif request.action == "resume":
                # Resume RL agent
                result = await self.decision_server.resume_agent(request.user_id)
                
            else:
                raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
            
            return {
                "success": True,
                "action": request.action,
                "user_id": request.user_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Agent control error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _get_agent_status(self, user_id: str) -> Dict[str, Any]:
        """Get agent status for user"""
        return await self.decision_server.get_agent_status(user_id)
    
    async def _handle_model_update(self, request: ModelUpdateRequest):
        """Handle model update requests"""
        try:
            logger.info(f"Starting model update: {request.model_type}")
            
            if request.model_type == "ensemble":
                # Retrain ensemble
                await self._retrain_ensemble(request.training_data, request.parameters)
                
            elif request.model_type == "individual" and request.agent_name:
                # Retrain specific agent
                await self._retrain_individual_agent(request.agent_name, request.training_data, request.parameters)
                
            else:
                logger.error(f"Invalid model update request: {request}")
                return
            
            logger.info("Model update completed successfully")
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    async def _retrain_ensemble(self, training_data: Optional[Dict], parameters: Optional[Dict]):
        """Retrain the ensemble agent"""
        # This would implement actual retraining logic
        # For now, just log the request
        logger.info("Ensemble retraining requested - implementing in future version")
    
    async def _retrain_individual_agent(self, agent_name: str, training_data: Optional[Dict], parameters: Optional[Dict]):
        """Retrain a specific agent"""
        logger.info(f"Individual agent retraining requested for {agent_name} - implementing in future version")
    
    async def _get_model_performance(self) -> Dict[str, Any]:
        """Get comprehensive model performance metrics"""
        return {
            "overall_performance": self.performance_tracker.get_metrics(),
            "ensemble_metrics": await self._get_ensemble_metrics(),
            "fallback_metrics": await self._get_fallback_metrics(),
            "comparison": await self._get_model_comparison()
        }
    
    async def _get_ensemble_metrics(self) -> Dict[str, Any]:
        """Get ensemble-specific metrics"""
        if not self.ensemble_agent:
            return {"error": "Ensemble agent not loaded"}
        
        # Return recent performance data
        return {
            "agent_weights": self.ensemble_agent.agent_weights.tolist(),
            "current_regime": self.ensemble_agent.current_regime,
            "regime_confidence": self.ensemble_agent.regime_confidence,
            "steps_since_rebalance": self.ensemble_agent.steps_since_rebalance
        }
    
    async def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Get fallback model metrics"""
        return await self.trading_bridge.get_adaptive_threshold_metrics()
    
    async def _get_model_comparison(self) -> Dict[str, Any]:
        """Compare RL vs fallback performance"""
        rl_metrics = self.performance_tracker.get_rl_metrics()
        fallback_metrics = self.performance_tracker.get_fallback_metrics()
        
        return {
            "rl_model": rl_metrics,
            "fallback_model": fallback_metrics,
            "rl_advantage": {
                "accuracy_diff": rl_metrics.get('accuracy', 0) - fallback_metrics.get('accuracy', 0),
                "response_time_diff": rl_metrics.get('avg_response_time', 0) - fallback_metrics.get('avg_response_time', 0)
            }
        }
    
    async def _get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get all service metrics"""
        return {
            "service_metrics": {
                "uptime_seconds": (datetime.now() - self.last_health_check).total_seconds(),
                "total_requests": self.performance_tracker.get_total_requests(),
                "active_websockets": len(self.websocket_connections),
                "is_ready": self.is_ready
            },
            "performance_metrics": self.performance_tracker.get_metrics(),
            "model_metrics": await self._get_model_performance(),
            "health_status": await self._get_health_status()
        }
    
    async def _get_health_status(self) -> HealthStatus:
        """Get comprehensive health status"""
        try:
            # Check service health
            services_status = {
                "rl_service": "healthy" if self.is_ready else "unhealthy",
                "ensemble_agent": "healthy" if self.ensemble_agent else "unhealthy",
                "trading_bridge": await self.trading_bridge.health_check(),
                "data_connector": await self.data_connector.health_check(),
                "decision_server": await self.decision_server.health_check()
            }
            
            # Calculate overall status
            healthy_services = sum(1 for status in services_status.values() if status == "healthy")
            total_services = len(services_status)
            
            if healthy_services == total_services:
                overall_status = "healthy"
            elif healthy_services > total_services / 2:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"
            
            # Get performance metrics
            perf_metrics = self.performance_tracker.get_metrics()
            metrics = PerformanceMetrics(
                total_predictions=perf_metrics.get('total_requests', 0),
                accuracy=perf_metrics.get('accuracy', 0.0),
                sharpe_ratio=perf_metrics.get('sharpe_ratio', 0.0),
                total_return=perf_metrics.get('total_return', 0.0),
                win_rate=perf_metrics.get('win_rate', 0.0),
                avg_response_time_ms=perf_metrics.get('avg_response_time', 0.0),
                model_uptime_percent=perf_metrics.get('uptime_percent', 0.0),
                last_updated=datetime.now()
            )
            
            # Check for alerts
            alerts = []
            if overall_status != "healthy":
                alerts.append(f"Service status: {overall_status}")
            if metrics.avg_response_time_ms > 1000:
                alerts.append("High response time detected")
            if metrics.accuracy < 0.6:
                alerts.append("Low prediction accuracy")
            
            return HealthStatus(
                status=overall_status,
                services=services_status,
                metrics=metrics,
                alerts=alerts,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return HealthStatus(
                status="unhealthy",
                services={"error": str(e)},
                metrics=PerformanceMetrics(
                    total_predictions=0, accuracy=0.0, sharpe_ratio=0.0,
                    total_return=0.0, win_rate=0.0, avg_response_time_ms=0.0,
                    model_uptime_percent=0.0, last_updated=datetime.now()
                ),
                alerts=[f"Health check failed: {str(e)}"],
                timestamp=datetime.now()
            )
    
    async def _handle_websocket_connection(self, websocket: WebSocket, user_id: str):
        """Handle WebSocket connection for real-time predictions"""
        await websocket.accept()
        self.websocket_connections[user_id] = websocket
        
        try:
            logger.info(f"WebSocket connection established for user {user_id}")
            
            while True:
                # Wait for prediction request
                data = await websocket.receive_json()
                
                # Create prediction request
                request = TradingSignalRequest(
                    user_id=user_id,
                    symbol=data.get('symbol', 'BTC'),
                    signal_data=data.get('signal_data', {}),
                    market_data=data.get('market_data')
                )
                
                # Get prediction
                response = await self._handle_prediction_request(request)
                
                # Send response
                await websocket.send_json(asdict(response))
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for user {user_id}")
        except Exception as e:
            logger.error(f"WebSocket error for user {user_id}: {e}")
        finally:
            # Clean up connection
            if user_id in self.websocket_connections:
                del self.websocket_connections[user_id]
    
    async def _background_health_monitor(self):
        """Background task for health monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                health_status = await self._get_health_status()
                
                # Log health status
                if health_status.status != "healthy":
                    logger.warning(f"Service health degraded: {health_status.status}")
                    for alert in health_status.alerts:
                        logger.warning(f"Alert: {alert}")
                
                # Update last health check time
                self.last_health_check = datetime.now()
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _background_performance_tracker(self):
        """Background task for performance tracking"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Update performance metrics
                await self.performance_tracker.update_metrics()
                
                # Log performance summary
                metrics = self.performance_tracker.get_metrics()
                logger.info(f"Performance summary: {metrics.get('total_requests', 0)} requests, "
                           f"{metrics.get('accuracy', 0):.2f} accuracy, "
                           f"{metrics.get('avg_response_time', 0):.2f}ms avg response time")
                
            except Exception as e:
                logger.error(f"Performance tracker error: {e}")
    
    async def _cleanup_service(self):
        """Cleanup service on shutdown"""
        logger.info("Shutting down RL service...")
        
        # Close WebSocket connections
        for user_id, websocket in self.websocket_connections.items():
            try:
                await websocket.close()
            except:
                pass
        
        # Cleanup components
        await self.monitor.cleanup()
        await self.performance_tracker.cleanup()
        
        logger.info("RL service shutdown complete")
    
    def run(self):
        """Run the RL service"""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug,
            log_level="info"
        )


def create_rl_service(config_dict: Optional[Dict[str, Any]] = None) -> RLService:
    """Factory function to create RL service"""
    config = RLServiceConfig(**config_dict) if config_dict else RLServiceConfig()
    return RLService(config)


if __name__ == "__main__":
    # Run the service
    service = create_rl_service()
    service.run()