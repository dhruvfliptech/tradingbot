"""
API Routes - REST API endpoints for RL Service
==============================================

This module defines all REST API routes for the RL service, providing
comprehensive control over RL agents, predictions, monitoring, and
system management.

Features:
- Agent lifecycle management (start/stop/pause/resume)
- Real-time prediction endpoints
- A/B testing configuration
- Performance monitoring and metrics
- Health checks and diagnostics
- Model management and updates
- Administrative controls
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import numpy as np

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, validator
import uvicorn

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class AgentStartRequest(BaseModel):
    user_id: str = Field(..., description="User ID for the agent")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "config": {
                    "model_type": "ensemble",
                    "confidence_threshold": 0.7,
                    "enable_regime_detection": True
                }
            }
        }

class AgentControlRequest(BaseModel):
    user_id: str = Field(..., description="User ID for the agent")
    action: str = Field(..., description="Action to perform")
    
    @validator('action')
    def validate_action(cls, v):
        allowed_actions = ['start', 'stop', 'pause', 'resume']
        if v not in allowed_actions:
            raise ValueError(f'Action must be one of {allowed_actions}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "action": "pause"
            }
        }

class PredictionRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    symbol: str = Field(..., description="Trading symbol")
    signal_data: Dict[str, Any] = Field(..., description="Trading signal data")
    market_data: Optional[Dict[str, Any]] = Field(None, description="Market data")
    priority: int = Field(1, ge=1, le=3, description="Priority (1=high, 2=medium, 3=low)")
    timeout_seconds: float = Field(5.0, gt=0, le=30, description="Timeout in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "symbol": "BTC",
                "signal_data": {
                    "confidence": 0.8,
                    "rsi": 45.2,
                    "change_percent": 2.5,
                    "volume": 1500000,
                    "action": "BUY"
                },
                "priority": 1,
                "timeout_seconds": 5.0
            }
        }

class PredictionResponse(BaseModel):
    request_id: Optional[str] = Field(None, description="Request ID for async predictions")
    should_trade: bool = Field(..., description="Whether to execute trade")
    confidence: float = Field(..., description="Confidence score (0-1)")
    action: str = Field(..., description="Recommended action")
    reasoning: str = Field(..., description="Reasoning for the decision")
    model_used: str = Field(..., description="Model that made the prediction")
    fallback_used: bool = Field(False, description="Whether fallback was used")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class ABTestConfig(BaseModel):
    enabled: bool = Field(..., description="Enable A/B testing")
    rl_traffic_percentage: float = Field(..., ge=0, le=1, description="Percentage of traffic to RL model")
    
    class Config:
        schema_extra = {
            "example": {
                "enabled": True,
                "rl_traffic_percentage": 0.2
            }
        }

class ModelUpdateRequest(BaseModel):
    model_type: str = Field(..., description="Type of model to update")
    agent_name: Optional[str] = Field(None, description="Specific agent name for individual updates")
    training_data: Optional[Dict[str, Any]] = Field(None, description="Training data")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model parameters")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_types = ['ensemble', 'individual', 'regime_detector']
        if v not in allowed_types:
            raise ValueError(f'Model type must be one of {allowed_types}')
        return v

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    services: Dict[str, str] = Field(..., description="Status of individual services")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    alerts: List[Dict[str, Any]] = Field(..., description="Active alerts")
    timestamp: datetime = Field(..., description="Timestamp of health check")

class MetricsResponse(BaseModel):
    service_metrics: Dict[str, Any] = Field(..., description="Service-level metrics")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    model_metrics: Dict[str, Any] = Field(..., description="Model performance metrics")
    timestamp: datetime = Field(..., description="Timestamp of metrics collection")


def create_api_routes(rl_service) -> APIRouter:
    """
    Create API routes for the RL service
    
    Args:
        rl_service: Instance of RLService
    
    Returns:
        FastAPI router with all routes
    """
    
    router = APIRouter(prefix="/api/v1", tags=["RL Service"])
    
    # Health and Status Endpoints
    
    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """Get comprehensive health status of the RL service"""
        try:
            health_status = await rl_service._get_health_status()
            return health_status
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
    
    @router.get("/status")
    async def service_status():
        """Get basic service status information"""
        try:
            return {
                "service": "rl-service",
                "version": "1.0.0",
                "is_ready": rl_service.is_ready,
                "uptime_seconds": (datetime.now() - rl_service.last_health_check).total_seconds(),
                "ensemble_loaded": rl_service.ensemble_agent is not None,
                "active_connections": len(rl_service.websocket_connections),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
    
    # Agent Management Endpoints
    
    @router.post("/agents/start")
    async def start_agent(request: AgentStartRequest):
        """Start RL agent for a user"""
        try:
            result = await rl_service.decision_server.start_agent(
                request.user_id, 
                request.config
            )
            
            if result.get('status') == 'error':
                raise HTTPException(status_code=400, detail=result.get('error'))
            
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to start agent: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start agent: {str(e)}")
    
    @router.post("/agents/control")
    async def control_agent(request: AgentControlRequest):
        """Control agent (stop, pause, resume)"""
        try:
            if request.action == "stop":
                result = await rl_service.decision_server.stop_agent(request.user_id)
            elif request.action == "pause":
                result = await rl_service.decision_server.pause_agent(request.user_id)
            elif request.action == "resume":
                result = await rl_service.decision_server.resume_agent(request.user_id)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
            
            if result.get('status') == 'error':
                raise HTTPException(status_code=400, detail=result.get('error'))
            
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to control agent: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to control agent: {str(e)}")
    
    @router.get("/agents/{user_id}/status")
    async def get_agent_status(user_id: str = Path(..., description="User ID")):
        """Get agent status for a specific user"""
        try:
            status = await rl_service.decision_server.get_agent_status(user_id)
            
            if "error" in status or status.get('status') == 'not_found':
                raise HTTPException(status_code=404, detail="Agent not found")
            
            return status
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get agent status: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")
    
    @router.get("/agents")
    async def list_agents():
        """List all active agents"""
        try:
            agents = []
            for user_id, instance in rl_service.decision_server.agent_instances.items():
                agents.append({
                    "user_id": user_id,
                    "state": instance.state.value,
                    "created_at": instance.created_at.isoformat(),
                    "last_activity": instance.last_activity.isoformat(),
                    "prediction_count": instance.prediction_count,
                    "error_count": instance.error_count
                })
            
            return {
                "total_agents": len(agents),
                "agents": agents,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")
    
    # Prediction Endpoints
    
    @router.post("/predict", response_model=PredictionResponse)
    async def predict_sync(request: PredictionRequest):
        """Get synchronous prediction from RL model"""
        try:
            # Convert request to internal format
            internal_request = type('Request', (), {
                'user_id': request.user_id,
                'symbol': request.symbol,
                'signal_data': request.signal_data,
                'market_data': request.market_data
            })()
            
            # Get prediction
            response = await rl_service._handle_prediction_request(internal_request)
            
            return PredictionResponse(
                should_trade=response.should_trade,
                confidence=response.confidence,
                action=response.action,
                reasoning=response.reasoning,
                model_used=response.model_used,
                fallback_used=response.fallback_used,
                metadata={
                    "symbol": request.symbol,
                    "priority": request.priority,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    @router.post("/predict/async")
    async def predict_async(request: PredictionRequest):
        """Submit async prediction request"""
        try:
            # Prepare observation from signal data
            observation = rl_service._prepare_observation(request.signal_data)
            
            # Submit async prediction
            request_id = await rl_service.decision_server.predict_async(
                user_id=request.user_id,
                symbol=request.symbol,
                observation=observation,
                market_data=request.market_data,
                priority=request.priority,
                timeout_seconds=request.timeout_seconds
            )
            
            return {
                "request_id": request_id,
                "status": "submitted",
                "estimated_time_seconds": request.timeout_seconds,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Async prediction submission failed: {e}")
            raise HTTPException(status_code=500, detail=f"Async prediction failed: {str(e)}")
    
    @router.get("/predict/{request_id}/result")
    async def get_prediction_result(
        request_id: str = Path(..., description="Prediction request ID"),
        timeout: float = Query(10.0, ge=0.1, le=30.0, description="Timeout in seconds")
    ):
        """Get result of async prediction"""
        try:
            result = await rl_service.decision_server.get_prediction_result(
                request_id, 
                timeout_seconds=timeout
            )
            
            if result is None:
                raise HTTPException(status_code=404, detail="Prediction result not found or timed out")
            
            # Convert action array to list for JSON serialization
            action_list = result.action.tolist() if hasattr(result.action, 'tolist') else [result.action]
            
            return PredictionResponse(
                request_id=result.request_id,
                should_trade=result.confidence > 0.5,  # Simple threshold
                confidence=result.confidence,
                action=action_list[0] if action_list else 0,
                reasoning=result.reasoning,
                model_used=result.model_info.get('dominant_agent', 'unknown'),
                processing_time_ms=result.processing_time_ms,
                metadata={
                    "model_info": result.model_info,
                    "status": result.status.value,
                    "timestamp": result.timestamp.isoformat()
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get prediction result: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get prediction result: {str(e)}")
    
    # A/B Testing Endpoints
    
    @router.post("/ab-test/configure")
    async def configure_ab_test(config: ABTestConfig):
        """Configure A/B testing parameters"""
        try:
            rl_service.config.ab_testing_enabled = config.enabled
            rl_service.config.rl_traffic_percentage = config.rl_traffic_percentage
            
            return {
                "status": "configured",
                "config": asdict(config),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to configure A/B test: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to configure A/B test: {str(e)}")
    
    @router.get("/ab-test/status")
    async def get_ab_test_status():
        """Get current A/B testing status"""
        try:
            total_requests = rl_service.performance_tracker.get_total_requests()
            rl_requests = rl_service.performance_tracker.get_rl_requests()
            
            return {
                "enabled": rl_service.config.ab_testing_enabled,
                "rl_traffic_percentage": rl_service.config.rl_traffic_percentage,
                "statistics": {
                    "total_requests": total_requests,
                    "rl_requests": rl_requests,
                    "fallback_requests": total_requests - rl_requests,
                    "actual_rl_percentage": (rl_requests / max(1, total_requests))
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get A/B test status: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get A/B test status: {str(e)}")
    
    # Model Management Endpoints
    
    @router.post("/models/update")
    async def update_model(request: ModelUpdateRequest, background_tasks: BackgroundTasks):
        """Trigger model update/retraining"""
        try:
            # Add to background tasks
            background_tasks.add_task(
                rl_service._handle_model_update, 
                request
            )
            
            return {
                "status": "initiated",
                "model_type": request.model_type,
                "agent_name": request.agent_name,
                "request_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to initiate model update: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initiate model update: {str(e)}")
    
    @router.get("/models/performance")
    async def get_model_performance():
        """Get comprehensive model performance metrics"""
        try:
            performance_data = await rl_service._get_model_performance()
            return {
                "performance": performance_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")
    
    @router.get("/models/comparison")
    async def compare_models():
        """Compare RL model vs fallback performance"""
        try:
            comparison = await rl_service._get_model_comparison()
            return {
                "comparison": comparison,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to compare models: {str(e)}")
    
    # Monitoring and Metrics Endpoints
    
    @router.get("/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """Get comprehensive service metrics"""
        try:
            metrics = await rl_service._get_comprehensive_metrics()
            
            return MetricsResponse(
                service_metrics=metrics.get("service_metrics", {}),
                performance_metrics=metrics.get("performance_metrics", {}),
                model_metrics=metrics.get("model_metrics", {}),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
    
    @router.get("/metrics/prometheus", response_class=PlainTextResponse)
    async def get_prometheus_metrics():
        """Get metrics in Prometheus format"""
        try:
            prometheus_metrics = rl_service.monitor.export_prometheus_metrics()
            return PlainTextResponse(content=prometheus_metrics)
        except Exception as e:
            logger.error(f"Failed to export Prometheus metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to export Prometheus metrics: {str(e)}")
    
    @router.get("/metrics/dashboard")
    async def get_dashboard_metrics():
        """Get metrics formatted for dashboard display"""
        try:
            dashboard_data = rl_service.monitor.get_dashboard_data()
            return {
                "dashboard": dashboard_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get dashboard metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get dashboard metrics: {str(e)}")
    
    @router.get("/alerts")
    async def get_alerts(
        severity: Optional[str] = Query(None, description="Filter by severity"),
        active_only: bool = Query(True, description="Show only active alerts")
    ):
        """Get system alerts"""
        try:
            from .monitoring import AlertSeverity
            
            severity_filter = None
            if severity:
                try:
                    severity_filter = AlertSeverity(severity.lower())
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
            
            if active_only:
                alerts = rl_service.monitor.get_active_alerts(severity_filter)
            else:
                # Get all alerts (would need additional method)
                alerts = list(rl_service.monitor.alerts.values())
                if severity_filter:
                    alerts = [a for a in alerts if a.severity == severity_filter]
            
            return {
                "alerts": [asdict(alert) for alert in alerts],
                "total_count": len(alerts),
                "timestamp": datetime.now().isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")
    
    @router.post("/alerts/{alert_id}/resolve")
    async def resolve_alert(alert_id: str = Path(..., description="Alert ID")):
        """Resolve an active alert"""
        try:
            success = rl_service.monitor.resolve_alert(alert_id)
            
            if not success:
                raise HTTPException(status_code=404, detail="Alert not found")
            
            return {
                "status": "resolved",
                "alert_id": alert_id,
                "timestamp": datetime.now().isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")
    
    # Administrative Endpoints
    
    @router.get("/stats/users")
    async def get_user_statistics():
        """Get statistics for all users"""
        try:
            user_stats = {}
            for user_id in rl_service.decision_server.agent_instances.keys():
                stats = rl_service.decision_server.get_user_statistics(user_id)
                user_stats[user_id] = stats
            
            return {
                "user_statistics": user_stats,
                "total_users": len(user_stats),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get user statistics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get user statistics: {str(e)}")
    
    @router.get("/stats/performance")
    async def get_performance_statistics():
        """Get detailed performance statistics"""
        try:
            performance_stats = rl_service.performance_tracker.get_metrics()
            decision_server_stats = rl_service.decision_server.get_performance_metrics()
            
            return {
                "performance_tracker": performance_stats,
                "decision_server": decision_server_stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get performance statistics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get performance statistics: {str(e)}")
    
    @router.post("/system/restart")
    async def restart_service(background_tasks: BackgroundTasks):
        """Restart the RL service (graceful restart)"""
        try:
            # Add restart to background tasks
            background_tasks.add_task(_restart_service, rl_service)
            
            return {
                "status": "restart_initiated",
                "message": "Service restart has been initiated",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to initiate restart: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initiate restart: {str(e)}")
    
    @router.get("/system/config")
    async def get_system_config():
        """Get current system configuration"""
        try:
            config_dict = {
                "host": rl_service.config.host,
                "port": rl_service.config.port,
                "debug": rl_service.config.debug,
                "max_concurrent_requests": rl_service.config.max_concurrent_requests,
                "prediction_timeout_seconds": rl_service.config.prediction_timeout_seconds,
                "ab_testing_enabled": rl_service.config.ab_testing_enabled,
                "rl_traffic_percentage": rl_service.config.rl_traffic_percentage,
                "fallback_to_adaptive_threshold": rl_service.config.fallback_to_adaptive_threshold,
                "models_path": rl_service.config.models_path
            }
            
            return {
                "config": config_dict,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get system config: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get system config: {str(e)}")
    
    return router


async def _restart_service(rl_service):
    """Background task to restart the service"""
    try:
        logger.info("Initiating graceful service restart...")
        
        # Give some time for the response to be sent
        await asyncio.sleep(2)
        
        # Cleanup current service
        await rl_service._cleanup_service()
        
        # Reinitialize service
        await rl_service._initialize_service()
        
        logger.info("Service restart completed")
        
    except Exception as e:
        logger.error(f"Service restart failed: {e}")


if __name__ == "__main__":
    # Test the API routes
    from fastapi import FastAPI
    from .rl_service import create_rl_service
    
    # Create a test RL service
    rl_service = create_rl_service()
    
    # Create FastAPI app with routes
    app = FastAPI(title="RL Service API Test")
    api_routes = create_api_routes(rl_service)
    app.include_router(api_routes)
    
    # Add startup/shutdown events
    @app.on_event("startup")
    async def startup():
        await rl_service._initialize_service()
    
    @app.on_event("shutdown") 
    async def shutdown():
        await rl_service._cleanup_service()
    
    # Run the test server
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8001)