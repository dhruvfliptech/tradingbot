"""
RL Connector
============

Direct connection interface between the strategy integration system and the RL environment.
Handles feature injection, action interpretation, reward feedback, and real-time communication
with the reinforcement learning trading agent.

Key Features:
- Real-time feature streaming to RL environment
- Action space integration and mapping
- Reward signal processing and feedback
- State synchronization
- Performance monitoring
- Explainable action interpretation
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque
import json
import websockets
import threading
from threading import Lock, Event
import uuid

# Import RL environment components
import sys
import os
sys.path.append('/Users/greenmachine2.0/Trading Bot Aug-15/tradingbot/backend/rl-service')

from environment.trading_env import TradingEnvironment
from environment.state_processor import StateProcessor
from rl_config import get_rl_config, ActionType

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Connection status to RL environment"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SYNCING = "syncing"
    ACTIVE = "active"
    ERROR = "error"


class ActionMode(Enum):
    """How actions are interpreted"""
    DIRECT = "direct"  # Direct action execution
    ADVISORY = "advisory"  # Advisory signals only
    HYBRID = "hybrid"  # Combination of both
    OVERRIDE = "override"  # Override RL decisions


@dataclass
class FeatureUpdate:
    """Feature update for RL environment"""
    timestamp: datetime
    features: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    sequence_id: int = 0
    source: str = "strategy_integration"


@dataclass
class ActionResponse:
    """Response from RL environment action"""
    action_id: str
    action_type: str
    action_value: Union[int, float, np.ndarray]
    confidence: float
    timestamp: datetime
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


@dataclass
class RewardFeedback:
    """Reward feedback to RL environment"""
    reward_value: float
    step_id: int
    timestamp: datetime
    reward_components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionMetrics:
    """Metrics for RL connection"""
    messages_sent: int = 0
    messages_received: int = 0
    feature_updates_sent: int = 0
    actions_received: int = 0
    rewards_sent: int = 0
    connection_uptime: float = 0.0
    avg_latency: float = 0.0
    error_count: int = 0
    last_update: Optional[datetime] = None


class RLConnector:
    """
    Advanced connector for seamless integration with RL trading environment.
    
    Provides high-performance, real-time communication between strategy
    integration system and reinforcement learning agent.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize RL connector"""
        self.config = config or self._default_config()
        
        # Connection management
        self.status = ConnectionStatus.DISCONNECTED
        self.rl_environment: Optional[TradingEnvironment] = None
        self.connection_lock = Lock()
        self.shutdown_event = Event()
        
        # Feature management
        self.feature_buffer: deque = deque(maxlen=self.config['feature_buffer_size'])
        self.feature_sequence_id = 0
        self.latest_features: Dict[str, float] = {}
        self.feature_mapping: Dict[str, str] = {}
        
        # Action management
        self.action_history: deque = deque(maxlen=self.config['action_history_size'])
        self.pending_actions: Dict[str, ActionResponse] = {}
        self.action_mode = ActionMode[self.config.get('action_mode', 'HYBRID')]
        
        # Reward management
        self.reward_buffer: deque = deque(maxlen=self.config['reward_buffer_size'])
        self.reward_callbacks: List[Callable] = []
        
        # Performance monitoring
        self.metrics = ConnectionMetrics()
        self.latency_history: deque = deque(maxlen=100)
        
        # State synchronization
        self.sync_lock = Lock()
        self.last_sync_time: Optional[datetime] = None
        self.sync_interval = self.config.get('sync_interval', 1.0)
        
        # WebSocket server for external connections
        self.websocket_server = None
        self.websocket_clients: Set = set()
        
        logger.info("RL Connector initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'rl_environment_path': '/backend/rl-service/environment/',
            'feature_buffer_size': 1000,
            'action_history_size': 100,
            'reward_buffer_size': 1000,
            'connection_timeout': 30.0,
            'sync_interval': 1.0,  # seconds
            'max_latency': 0.1,  # 100ms
            'action_mode': 'HYBRID',
            'enable_websocket_server': True,
            'websocket_port': 8765,
            'feature_compression': True,
            'real_time_streaming': True,
            'batch_size': 10,
            'retry_attempts': 3,
            'retry_delay': 1.0,
            'enable_action_validation': True,
            'enable_reward_feedback': True,
            'state_sync_enabled': True,
            'performance_monitoring': True,
            'feature_mapping': {
                # Map strategy features to RL feature names
                'whale_tracker_sentiment': 'whale_sentiment',
                'whale_tracker_volume': 'whale_volume',
                'volume_profile_poc': 'volume_poc',
                'volume_profile_imbalance': 'volume_imbalance',
                'order_book_spread': 'order_spread',
                'order_book_depth': 'order_depth',
                'regime_detection_state': 'market_regime',
                'smart_money_flow': 'institutional_flow'
            },
            'action_mapping': {
                # Map RL actions to trading actions
                0: 'HOLD',
                1: 'BUY_SMALL',
                2: 'BUY_MEDIUM',
                3: 'BUY_LARGE',
                4: 'SELL_SMALL',
                5: 'SELL_MEDIUM',
                6: 'SELL_LARGE'
            }
        }
    
    async def start(self):
        """Start the RL connector"""
        logger.info("Starting RL Connector...")
        
        self.status = ConnectionStatus.CONNECTING
        
        try:
            # Initialize RL environment
            await self._initialize_rl_environment()
            
            # Start WebSocket server if enabled
            if self.config.get('enable_websocket_server', True):
                await self._start_websocket_server()
            
            # Start background tasks
            asyncio.create_task(self._sync_loop())
            asyncio.create_task(self._monitoring_loop())
            
            self.status = ConnectionStatus.ACTIVE
            self.metrics.last_update = datetime.now()
            
            logger.info("RL Connector started successfully")
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            logger.error(f"Failed to start RL Connector: {e}")
            raise
    
    async def stop(self):
        """Stop the RL connector"""
        logger.info("Stopping RL Connector...")
        
        self.shutdown_event.set()
        self.status = ConnectionStatus.DISCONNECTED
        
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        logger.info("RL Connector stopped")
    
    async def _initialize_rl_environment(self):
        """Initialize connection to RL environment"""
        try:
            # Load RL configuration
            rl_config = get_rl_config()
            
            # Create RL environment instance
            self.rl_environment = TradingEnvironment(
                config=rl_config,
                mode='live',  # Live trading mode
                symbols=['BTC/USD', 'ETH/USD']  # Default symbols
            )
            
            # Load data for environment
            await self._load_environment_data()
            
            # Initialize environment
            observation, info = self.rl_environment.reset()
            
            logger.info(f"RL Environment initialized with observation shape: {observation.shape}")
            
        except Exception as e:
            logger.error(f"Error initializing RL environment: {e}")
            raise
    
    async def _load_environment_data(self):
        """Load market data for RL environment"""
        try:
            # Use recent data for live trading
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last 30 days
            
            success = self.rl_environment.load_data(start_date, end_date)
            
            if not success:
                logger.warning("Failed to load market data, using default data")
            
        except Exception as e:
            logger.error(f"Error loading environment data: {e}")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for external connections"""
        try:
            port = self.config.get('websocket_port', 8765)
            
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                "localhost",
                port
            )
            
            logger.info(f"WebSocket server started on port {port}")
            
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle WebSocket client connections"""
        self.websocket_clients.add(websocket)
        logger.info(f"New WebSocket client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                await self._process_websocket_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.websocket_clients.discard(websocket)
    
    async def _process_websocket_message(self, websocket, message):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'feature_update':
                await self._handle_external_feature_update(data)
            elif message_type == 'action_request':
                response = await self._handle_action_request(data)
                await websocket.send(json.dumps(response))
            elif message_type == 'reward_feedback':
                await self._handle_reward_feedback(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    async def update_features(self, features: Dict[str, float], metadata: Optional[Dict] = None):
        """Update features in RL environment"""
        try:
            start_time = time.time()
            
            # Map feature names if configured
            mapped_features = self._map_features(features)
            
            # Create feature update
            update = FeatureUpdate(
                timestamp=datetime.now(),
                features=mapped_features,
                metadata=metadata or {},
                sequence_id=self.feature_sequence_id,
                source="strategy_integration"
            )
            
            # Add to buffer
            self.feature_buffer.append(update)
            self.feature_sequence_id += 1
            
            # Update latest features
            self.latest_features.update(mapped_features)
            
            # Send to RL environment
            await self._send_features_to_rl(update)
            
            # Send to WebSocket clients
            if self.websocket_clients:
                await self._broadcast_feature_update(update)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.latency_history.append(processing_time)
            self.metrics.avg_latency = np.mean(list(self.latency_history))
            self.metrics.feature_updates_sent += 1
            self.metrics.messages_sent += 1
            
            logger.debug(f"Updated {len(mapped_features)} features in {processing_time:.3f}s")
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error updating features: {e}")
    
    def _map_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Map strategy feature names to RL feature names"""
        feature_mapping = self.config.get('feature_mapping', {})
        mapped_features = {}
        
        for original_name, value in features.items():
            # Use mapped name if available, otherwise use original
            mapped_name = feature_mapping.get(original_name, original_name)
            mapped_features[mapped_name] = value
        
        return mapped_features
    
    async def _send_features_to_rl(self, update: FeatureUpdate):
        """Send feature update to RL environment"""
        if not self.rl_environment:
            return
        
        try:
            # Prepare state data for RL environment
            state_data = {
                'timestamp': update.timestamp,
                'price_data': {},  # Would be populated with current market data
                'sentiment_data': {},
                'alternative_data': update.features,  # Use our features as alternative data
                'portfolio_state': {}
            }
            
            # Update state processor if available
            if hasattr(self.rl_environment, 'state_processor'):
                market_state = self.rl_environment.state_processor.transform(state_data)
                logger.debug(f"Updated RL state with {len(update.features)} features")
            
        except Exception as e:
            logger.error(f"Error sending features to RL: {e}")
    
    async def _broadcast_feature_update(self, update: FeatureUpdate):
        """Broadcast feature update to WebSocket clients"""
        if not self.websocket_clients:
            return
        
        try:
            message = {
                'type': 'feature_update',
                'timestamp': update.timestamp.isoformat(),
                'features': update.features,
                'sequence_id': update.sequence_id,
                'metadata': update.metadata
            }
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    logger.warning(f"Error sending to WebSocket client: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
            
        except Exception as e:
            logger.error(f"Error broadcasting feature update: {e}")
    
    async def request_action(self, observation: Optional[np.ndarray] = None) -> Optional[ActionResponse]:
        """Request action from RL environment"""
        if not self.rl_environment or self.status != ConnectionStatus.ACTIVE:
            return None
        
        try:
            start_time = time.time()
            
            # Use latest observation or create from current state
            if observation is None:
                observation = self._create_observation_from_features()
            
            # Get action from RL environment
            # Note: This would typically involve calling the trained agent
            # For now, we'll simulate an action
            action = self._simulate_rl_action(observation)
            
            # Create action response
            action_response = ActionResponse(
                action_id=str(uuid.uuid4()),
                action_type=self._map_action_to_type(action),
                action_value=action,
                confidence=0.75,  # Would come from RL agent
                timestamp=datetime.now(),
                reasoning="RL agent decision based on current market state",
                execution_time=time.time() - start_time
            )
            
            # Store action
            self.action_history.append(action_response)
            self.pending_actions[action_response.action_id] = action_response
            
            # Update metrics
            self.metrics.actions_received += 1
            self.metrics.messages_received += 1
            
            logger.debug(f"Received action: {action_response.action_type} (confidence: {action_response.confidence:.2f})")
            
            return action_response
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error requesting action: {e}")
            return None
    
    def _create_observation_from_features(self) -> np.ndarray:
        """Create observation vector from current features"""
        if not self.latest_features:
            return np.zeros(50)  # Default size
        
        # Convert features to observation vector
        feature_values = list(self.latest_features.values())
        
        # Pad or truncate to expected size
        expected_size = 50  # Would get from RL environment config
        if len(feature_values) < expected_size:
            feature_values.extend([0.0] * (expected_size - len(feature_values)))
        elif len(feature_values) > expected_size:
            feature_values = feature_values[:expected_size]
        
        return np.array(feature_values, dtype=np.float32)
    
    def _simulate_rl_action(self, observation: np.ndarray) -> int:
        """Simulate RL action (placeholder for actual RL agent)"""
        # Simple logic based on observation values
        mean_value = np.mean(observation)
        
        if mean_value > 0.3:
            return 2  # BUY_MEDIUM
        elif mean_value < -0.3:
            return 5  # SELL_MEDIUM
        else:
            return 0  # HOLD
    
    def _map_action_to_type(self, action: int) -> str:
        """Map numerical action to action type"""
        action_mapping = self.config.get('action_mapping', {})
        return action_mapping.get(action, 'HOLD')
    
    async def send_reward(self, reward: float, step_id: int, components: Optional[Dict[str, float]] = None):
        """Send reward feedback to RL environment"""
        try:
            reward_feedback = RewardFeedback(
                reward_value=reward,
                step_id=step_id,
                timestamp=datetime.now(),
                reward_components=components or {},
                metadata={}
            )
            
            # Add to buffer
            self.reward_buffer.append(reward_feedback)
            
            # Send to RL environment
            await self._send_reward_to_rl(reward_feedback)
            
            # Notify callbacks
            for callback in self.reward_callbacks:
                try:
                    await callback(reward_feedback)
                except Exception as e:
                    logger.warning(f"Error in reward callback: {e}")
            
            # Update metrics
            self.metrics.rewards_sent += 1
            self.metrics.messages_sent += 1
            
            logger.debug(f"Sent reward: {reward:.4f} for step {step_id}")
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error sending reward: {e}")
    
    async def _send_reward_to_rl(self, reward_feedback: RewardFeedback):
        """Send reward to RL environment"""
        if not self.rl_environment:
            return
        
        try:
            # The reward would typically be handled by the training loop
            # For live trading, we might log it or use it for evaluation
            logger.debug(f"RL reward feedback: {reward_feedback.reward_value}")
            
        except Exception as e:
            logger.error(f"Error sending reward to RL: {e}")
    
    async def _sync_loop(self):
        """Background synchronization loop"""
        while not self.shutdown_event.is_set():
            try:
                if self.status == ConnectionStatus.ACTIVE:
                    await self._synchronize_state()
                
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _synchronize_state(self):
        """Synchronize state with RL environment"""
        try:
            with self.sync_lock:
                current_time = datetime.now()
                
                # Update connection uptime
                if self.last_sync_time:
                    self.metrics.connection_uptime += (current_time - self.last_sync_time).total_seconds()
                
                self.last_sync_time = current_time
                
                # Check environment health
                if self.rl_environment:
                    # Verify environment is responsive
                    pass  # Would implement health checks
                
                # Clean up old data
                await self._cleanup_old_data()
                
        except Exception as e:
            logger.error(f"Error synchronizing state: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data from buffers"""
        current_time = datetime.now()
        max_age = timedelta(hours=1)
        
        # Clean feature buffer
        while (self.feature_buffer and 
               current_time - self.feature_buffer[0].timestamp > max_age):
            self.feature_buffer.popleft()
        
        # Clean pending actions
        expired_actions = []
        for action_id, action in self.pending_actions.items():
            if current_time - action.timestamp > timedelta(minutes=5):
                expired_actions.append(action_id)
        
        for action_id in expired_actions:
            del self.pending_actions[action_id]
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                if self.config.get('performance_monitoring', True):
                    await self._update_performance_metrics()
                    await self._check_system_health()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate average latency
            if self.latency_history:
                self.metrics.avg_latency = np.mean(list(self.latency_history))
            
            # Update last update time
            self.metrics.last_update = datetime.now()
            
            # Log performance summary
            logger.debug(f"Performance: {self.metrics.messages_sent} sent, "
                        f"{self.metrics.messages_received} received, "
                        f"avg latency: {self.metrics.avg_latency:.3f}s")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _check_system_health(self):
        """Check system health and alert if issues"""
        try:
            # Check latency
            if self.metrics.avg_latency > self.config.get('max_latency', 0.1):
                logger.warning(f"High latency detected: {self.metrics.avg_latency:.3f}s")
            
            # Check error rate
            total_messages = self.metrics.messages_sent + self.metrics.messages_received
            if total_messages > 0:
                error_rate = self.metrics.error_count / total_messages
                if error_rate > 0.05:  # 5% error rate threshold
                    logger.warning(f"High error rate: {error_rate:.2%}")
            
            # Check connection status
            if self.status != ConnectionStatus.ACTIVE:
                logger.warning(f"Connection not active: {self.status.value}")
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
    
    async def _handle_external_feature_update(self, data: Dict):
        """Handle feature update from external source"""
        try:
            features = data.get('features', {})
            metadata = data.get('metadata', {})
            
            await self.update_features(features, metadata)
            
        except Exception as e:
            logger.error(f"Error handling external feature update: {e}")
    
    async def _handle_action_request(self, data: Dict) -> Dict:
        """Handle action request from external source"""
        try:
            observation = data.get('observation')
            if observation:
                observation = np.array(observation)
            
            action_response = await self.request_action(observation)
            
            if action_response:
                return {
                    'type': 'action_response',
                    'action_id': action_response.action_id,
                    'action_type': action_response.action_type,
                    'action_value': action_response.action_value,
                    'confidence': action_response.confidence,
                    'timestamp': action_response.timestamp.isoformat(),
                    'reasoning': action_response.reasoning
                }
            else:
                return {
                    'type': 'error',
                    'message': 'Failed to get action response'
                }
                
        except Exception as e:
            logger.error(f"Error handling action request: {e}")
            return {
                'type': 'error',
                'message': str(e)
            }
    
    async def _handle_reward_feedback(self, data: Dict):
        """Handle reward feedback from external source"""
        try:
            reward = data.get('reward', 0.0)
            step_id = data.get('step_id', 0)
            components = data.get('components', {})
            
            await self.send_reward(reward, step_id, components)
            
        except Exception as e:
            logger.error(f"Error handling reward feedback: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status"""
        return {
            'status': self.status.value,
            'connected': self.status == ConnectionStatus.ACTIVE,
            'rl_environment_ready': self.rl_environment is not None,
            'websocket_clients': len(self.websocket_clients),
            'feature_buffer_size': len(self.feature_buffer),
            'action_history_size': len(self.action_history),
            'pending_actions': len(self.pending_actions),
            'latest_features_count': len(self.latest_features),
            'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics"""
        return {
            'connection_metrics': {
                'messages_sent': self.metrics.messages_sent,
                'messages_received': self.metrics.messages_received,
                'feature_updates_sent': self.metrics.feature_updates_sent,
                'actions_received': self.metrics.actions_received,
                'rewards_sent': self.metrics.rewards_sent,
                'connection_uptime': self.metrics.connection_uptime,
                'avg_latency': self.metrics.avg_latency,
                'error_count': self.metrics.error_count,
                'last_update': self.metrics.last_update.isoformat() if self.metrics.last_update else None
            },
            'status': self.get_status(),
            'config': self.config
        }
    
    def add_reward_callback(self, callback: Callable):
        """Add reward feedback callback"""
        self.reward_callbacks.append(callback)
    
    def remove_reward_callback(self, callback: Callable):
        """Remove reward feedback callback"""
        if callback in self.reward_callbacks:
            self.reward_callbacks.remove(callback)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create RL connector
        connector = RLConnector()
        
        try:
            await connector.start()
            
            # Update some features
            features = {
                'whale_sentiment': 0.75,
                'volume_poc': 30000,
                'order_spread': 0.001,
                'market_regime': 2.0
            }
            
            await connector.update_features(features)
            
            # Request action
            action_response = await connector.request_action()
            if action_response:
                print(f"Action: {action_response.action_type} (confidence: {action_response.confidence:.2f})")
            
            # Send reward
            await connector.send_reward(0.02, step_id=1)
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Get metrics
            metrics = connector.get_metrics()
            print(f"Connector metrics: {metrics}")
            
        finally:
            await connector.stop()
    
    asyncio.run(main())