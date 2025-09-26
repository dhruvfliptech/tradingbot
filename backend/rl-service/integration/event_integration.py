"""
RL Service Event Integration
Connects Reinforcement Learning Decision Server to Event Bus
"""

import asyncio
import json
import logging
import redis
import time
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from threading import Thread, Lock
import queue

from decision_server import DecisionServer, PredictionRequest
from data_connector import DataConnector
from monitoring import MonitoringService

logger = logging.getLogger(__name__)

# Event types matching TypeScript definitions
class MLEventType:
    # Market Events
    MARKET_DATA_UPDATE = 'market.data.update'
    TRADE_EXECUTED = 'trade.executed'
    POSITION_OPENED = 'position.opened'
    POSITION_CLOSED = 'position.closed'
    PERFORMANCE_UPDATE = 'performance.update'

    # ML Service Events
    THRESHOLD_ADJUSTED = 'ml.threshold.adjusted'
    ML_PREDICTION_READY = 'ml.prediction.ready'

    # RL Service Events
    RL_SIGNAL_GENERATED = 'rl.signal.generated'
    RL_CONFIDENCE_UPDATE = 'rl.confidence.update'
    RL_ACTION_RECOMMENDED = 'rl.action.recommended'
    RL_MODEL_UPDATED = 'rl.model.updated'

    # Feedback Events
    PREDICTION_OUTCOME = 'feedback.prediction.outcome'


@dataclass
class RLEvent:
    """RL Event structure"""
    id: str
    type: str
    timestamp: datetime
    userId: str
    version: str = "1.0.0"
    correlationId: Optional[str] = None
    data: Dict[str, Any] = None


class RLServiceEventIntegration:
    """
    Event-driven integration for RL Decision Server
    Subscribes to market events and publishes RL signals
    """

    def __init__(self, decision_server: DecisionServer):
        self.decision_server = decision_server
        self.data_connector = DataConnector()
        self.monitoring = MonitoringService()

        # Redis configuration
        redis_config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'decode_responses': True
        }
        self.redis_client = redis.Redis(**redis_config)

        self.consumer_group = "rl-service"
        self.consumer_id = f"rl-service-{int(time.time())}"

        # Stream names
        self.STREAMS = {
            'MARKET': 'ml:stream:market',
            'ML_SERVICE': 'ml:stream:ml',
            'RL_SERVICE': 'ml:stream:rl',
            'FEEDBACK': 'ml:stream:feedback'
        }

        # Processing
        self.event_queue = queue.Queue(maxsize=500)
        self.state_buffer = {}  # Store recent states per symbol
        self.is_running = False

        # Performance tracking
        self.signals_generated = 0
        self.predictions_made = 0
        self.feedback_processed = 0

        # Event handlers
        self.event_handlers = {
            MLEventType.MARKET_DATA_UPDATE: self.handle_market_data,
            MLEventType.THRESHOLD_ADJUSTED: self.handle_threshold_adjusted,
            MLEventType.ML_PREDICTION_READY: self.handle_ml_prediction,
            MLEventType.POSITION_CLOSED: self.handle_position_closed,
            MLEventType.PREDICTION_OUTCOME: self.handle_prediction_outcome,
            MLEventType.PERFORMANCE_UPDATE: self.handle_performance_update
        }

        logger.info("RLServiceEventIntegration initialized")

    def start(self):
        """Start event processing"""
        if self.is_running:
            logger.warning("RLServiceEventIntegration already running")
            return

        self.is_running = True

        # Initialize consumer groups
        self._init_consumer_groups()

        # Start consumer threads
        market_thread = Thread(target=self._consume_stream, args=(self.STREAMS['MARKET'],))
        ml_thread = Thread(target=self._consume_stream, args=(self.STREAMS['ML_SERVICE'],))
        feedback_thread = Thread(target=self._consume_stream, args=(self.STREAMS['FEEDBACK'],))
        processor_thread = Thread(target=self._process_events)

        for thread in [market_thread, ml_thread, feedback_thread, processor_thread]:
            thread.daemon = True
            thread.start()

        logger.info("RLServiceEventIntegration started")

    def _init_consumer_groups(self):
        """Initialize Redis consumer groups"""
        streams = [self.STREAMS['MARKET'], self.STREAMS['ML_SERVICE'], self.STREAMS['FEEDBACK']]
        for stream_name in streams:
            try:
                self.redis_client.xgroup_create(stream_name, self.consumer_group, id='$', mkstream=True)
                logger.info(f"Created consumer group {self.consumer_group} for stream {stream_name}")
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    logger.error(f"Error creating consumer group: {e}")

    def _consume_stream(self, stream_name: str):
        """Consume events from Redis stream"""
        while self.is_running:
            try:
                messages = self.redis_client.xreadgroup(
                    self.consumer_group,
                    self.consumer_id,
                    {stream_name: '>'},
                    count=5,
                    block=5000
                )

                if messages:
                    for stream, stream_messages in messages:
                        for msg_id, data in stream_messages:
                            event = self._parse_event(msg_id, data)

                            try:
                                self.event_queue.put(event, timeout=1)
                                self.redis_client.xack(stream_name, self.consumer_group, msg_id)
                            except queue.Full:
                                logger.warning(f"Event queue full, dropping event {msg_id}")

            except Exception as e:
                logger.error(f"Error consuming from stream {stream_name}: {e}")
                time.sleep(1)

    def _parse_event(self, msg_id: str, data: Dict[str, Any]) -> RLEvent:
        """Parse event from Redis stream data"""
        event_data = {}
        nested_data = {}

        for key, value in data.items():
            if '.' in key:
                parts = key.split('.')
                current = nested_data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Parse value
                if value == 'true':
                    current[parts[-1]] = True
                elif value == 'false':
                    current[parts[-1]] = False
                elif value.replace('.', '').replace('-', '').isdigit():
                    try:
                        current[parts[-1]] = float(value) if '.' in value else int(value)
                    except:
                        current[parts[-1]] = value
                else:
                    current[parts[-1]] = value
            else:
                event_data[key] = value

        if 'data' in nested_data:
            event_data['data'] = nested_data['data']

        return RLEvent(
            id=event_data.get('id', msg_id),
            type=event_data.get('type', ''),
            timestamp=datetime.fromisoformat(event_data.get('timestamp', datetime.now().isoformat())),
            userId=event_data.get('userId', ''),
            version=event_data.get('version', '1.0.0'),
            correlationId=event_data.get('correlationId'),
            data=event_data.get('data', {})
        )

    def _process_events(self):
        """Process events from queue"""
        while self.is_running:
            try:
                event = self.event_queue.get(timeout=1)

                handler = self.event_handlers.get(event.type)
                if handler:
                    try:
                        asyncio.run(handler(event))
                    except Exception as e:
                        logger.error(f"Error processing event {event.id}: {e}")
                        self.monitoring.record_error('event_processing', str(e))

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in event processor: {e}")

    async def handle_market_data(self, event: RLEvent):
        """Handle market data update and generate RL signals"""
        try:
            data = event.data
            symbol = data.get('symbol')
            user_id = event.userId

            # Update state buffer
            self._update_state_buffer(symbol, data)

            # Prepare observation for RL agent
            observation = await self._prepare_observation(symbol, data)

            if observation is not None:
                # Create prediction request
                request = PredictionRequest(
                    request_id=f"req_{event.id}",
                    user_id=user_id,
                    symbol=symbol,
                    observation=observation,
                    market_data=data,
                    priority=1,
                    timeout_seconds=2.0
                )

                # Get RL prediction
                result = await self.decision_server.get_prediction(request)

                if result and result.confidence > 0.6:  # Confidence threshold
                    # Generate and publish RL signal
                    await self.publish_rl_signal(
                        user_id=user_id,
                        symbol=symbol,
                        action=self._convert_action(result.action),
                        confidence=result.confidence,
                        reasoning=result.reasoning,
                        model_info=result.model_info
                    )

                    self.signals_generated += 1

        except Exception as e:
            logger.error(f"Error handling market data: {e}")
            self.monitoring.record_error('market_data_handler', str(e))

    async def handle_threshold_adjusted(self, event: RLEvent):
        """Handle ML threshold adjustments"""
        try:
            # Update agent's understanding of thresholds
            data = event.data
            parameter = data.get('parameter')
            new_value = data.get('newValue')

            # Inform decision server about threshold changes
            await self.decision_server.update_thresholds({
                parameter: new_value
            })

            # May trigger confidence recalculation
            await self.publish_confidence_update(
                user_id=event.userId,
                reason="ML threshold adjusted"
            )

        except Exception as e:
            logger.error(f"Error handling threshold adjustment: {e}")

    async def handle_ml_prediction(self, event: RLEvent):
        """Enhance ML predictions with RL insights"""
        try:
            data = event.data
            ml_prediction = data.get('prediction', {})
            symbol = data.get('symbol')

            # Get current RL assessment
            observation = await self._prepare_observation(symbol, {})

            if observation is not None:
                request = PredictionRequest(
                    request_id=f"ml_enhance_{event.id}",
                    user_id=event.userId,
                    symbol=symbol,
                    observation=observation,
                    priority=2
                )

                rl_result = await self.decision_server.get_prediction(request)

                # If RL disagrees strongly, publish a counter-signal
                if rl_result and abs(rl_result.confidence - ml_prediction.get('confidence', 0)) > 0.3:
                    await self.publish_rl_signal(
                        user_id=event.userId,
                        symbol=symbol,
                        action=self._convert_action(rl_result.action),
                        confidence=rl_result.confidence,
                        reasoning=f"RL assessment differs from ML: {rl_result.reasoning}",
                        model_info=rl_result.model_info
                    )

        except Exception as e:
            logger.error(f"Error handling ML prediction: {e}")

    async def handle_position_closed(self, event: RLEvent):
        """Learn from closed positions"""
        try:
            data = event.data

            # Update agent's reward model
            await self.decision_server.process_trade_outcome({
                'symbol': data.get('symbol'),
                'pnl': data.get('pnl'),
                'pnl_percent': data.get('pnlPercent'),
                'holding_period': data.get('holdingPeriod'),
                'close_reason': data.get('closeReason')
            })

            self.feedback_processed += 1

        except Exception as e:
            logger.error(f"Error handling position closed: {e}")

    async def handle_prediction_outcome(self, event: RLEvent):
        """Process prediction outcome feedback"""
        try:
            data = event.data

            # Update agent based on prediction accuracy
            await self.decision_server.update_from_feedback({
                'signal_id': data.get('signalId'),
                'predicted_action': data.get('predicted', {}).get('action'),
                'actual_return': data.get('actual', {}).get('return'),
                'accuracy': data.get('accuracy')
            })

            # Track performance
            self.monitoring.record_metric('prediction_accuracy', data.get('accuracy', 0))

        except Exception as e:
            logger.error(f"Error handling prediction outcome: {e}")

    async def handle_performance_update(self, event: RLEvent):
        """Adjust RL strategy based on performance"""
        try:
            metrics = event.data.get('metrics', {})

            # Inform decision server about performance
            await self.decision_server.update_performance_context(metrics)

            # Adjust exploration vs exploitation
            if metrics.get('sharpeRatio', 0) < 0.5:
                # Poor performance, increase exploration
                await self.decision_server.set_exploration_rate(0.2)
            else:
                # Good performance, decrease exploration
                await self.decision_server.set_exploration_rate(0.05)

        except Exception as e:
            logger.error(f"Error handling performance update: {e}")

    def _update_state_buffer(self, symbol: str, data: Dict[str, Any]):
        """Update state buffer with latest data"""
        if symbol not in self.state_buffer:
            self.state_buffer[symbol] = []

        self.state_buffer[symbol].append({
            'timestamp': time.time(),
            'data': data
        })

        # Keep only last 100 states
        if len(self.state_buffer[symbol]) > 100:
            self.state_buffer[symbol] = self.state_buffer[symbol][-100:]

    async def _prepare_observation(self, symbol: str, current_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare observation vector for RL agent"""
        try:
            # Get historical states
            states = self.state_buffer.get(symbol, [])

            if len(states) < 10:
                return None  # Not enough data

            # Extract features
            features = []

            # Price features
            prices = [s['data'].get('price', 0) for s in states[-20:]]
            if prices:
                features.extend([
                    np.mean(prices),
                    np.std(prices),
                    prices[-1] / prices[0] if prices[0] > 0 else 1,
                    max(prices) - min(prices) if prices else 0
                ])

            # Volume features
            volumes = [s['data'].get('volume', 0) for s in states[-20:]]
            if volumes:
                features.extend([
                    np.mean(volumes),
                    np.std(volumes),
                    volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1
                ])

            # Technical indicators from current data
            indicators = current_data.get('indicators', {})
            features.extend([
                indicators.get('rsi', 50) / 100,
                indicators.get('macd', 0),
                (current_data.get('bid', 0) + current_data.get('ask', 0)) / 2 if current_data.get('bid') else 0
            ])

            # Normalize to fixed size
            observation = np.array(features[:50])  # Take first 50 features
            if len(observation) < 50:
                observation = np.pad(observation, (0, 50 - len(observation)))

            return observation

        except Exception as e:
            logger.error(f"Error preparing observation: {e}")
            return None

    def _convert_action(self, action: np.ndarray) -> str:
        """Convert RL action to trading action"""
        if isinstance(action, np.ndarray):
            action_idx = np.argmax(action)
            actions = ['hold', 'buy', 'sell']
            return actions[min(action_idx, 2)]
        return 'hold'

    async def publish_rl_signal(
        self,
        user_id: str,
        symbol: str,
        action: str,
        confidence: float,
        reasoning: str,
        model_info: Dict[str, Any]
    ):
        """Publish RL signal to event bus"""
        try:
            signal_id = f"rl_sig_{int(time.time())}_{symbol}"

            event_data = {
                'id': f"rls_{int(time.time())}_{user_id[:8]}",
                'type': MLEventType.RL_SIGNAL_GENERATED,
                'timestamp': datetime.now().isoformat(),
                'userId': user_id,
                'version': '1.0.0',
                'data.signalId': signal_id,
                'data.symbol': symbol,
                'data.action': action,
                'data.confidence': str(confidence),
                'data.reasoning': reasoning,
                'data.expectedReturn': str(model_info.get('expected_return', 0)),
                'data.riskScore': str(1 - confidence),
                'data.modelInfo.agentType': model_info.get('agent_type', 'ensemble'),
                'data.modelInfo.version': model_info.get('version', '1.0.0'),
                'data.modelInfo.trainingEpisodes': str(model_info.get('training_episodes', 0))
            }

            self.redis_client.xadd(
                self.STREAMS['RL_SERVICE'],
                event_data,
                maxlen=10000,
                approximate=True
            )

            logger.info(f"Published RL signal for {symbol}: {action} (confidence: {confidence:.2f})")
            self.predictions_made += 1

        except Exception as e:
            logger.error(f"Error publishing RL signal: {e}")

    async def publish_confidence_update(self, user_id: str, reason: str):
        """Publish confidence update event"""
        try:
            event_data = {
                'id': f"rcu_{int(time.time())}_{user_id[:8]}",
                'type': MLEventType.RL_CONFIDENCE_UPDATE,
                'timestamp': datetime.now().isoformat(),
                'userId': user_id,
                'version': '1.0.0',
                'data.reason': reason
            }

            self.redis_client.xadd(
                self.STREAMS['RL_SERVICE'],
                event_data,
                maxlen=10000,
                approximate=True
            )

        except Exception as e:
            logger.error(f"Error publishing confidence update: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            'running': self.is_running,
            'signals_generated': self.signals_generated,
            'predictions_made': self.predictions_made,
            'feedback_processed': self.feedback_processed,
            'queue_size': self.event_queue.qsize(),
            'buffered_symbols': len(self.state_buffer),
            'redis_connected': self.redis_client.ping()
        }

    def stop(self):
        """Stop event processing"""
        self.is_running = False
        logger.info("RLServiceEventIntegration stopped")


# Global instance (initialized with DecisionServer)
rl_event_integration = None

def initialize_rl_integration(decision_server: DecisionServer):
    """Initialize RL event integration with decision server"""
    global rl_event_integration
    rl_event_integration = RLServiceEventIntegration(decision_server)
    return rl_event_integration