"""
ML Service Event Integration
Connects Adaptive Threshold ML Service to Event Bus
"""

import asyncio
import json
import logging
import redis
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from threading import Thread, Lock
import queue

from adaptive_threshold import AdaptiveThreshold, threshold_manager
from performance_tracker import performance_tracker
from monitoring import logger
from config import get_config

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
    RISK_LIMIT_UPDATE = 'ml.risk.limit.update'
    ADAPTATION_TRIGGERED = 'ml.adaptation.triggered'
    ML_PREDICTION_READY = 'ml.prediction.ready'

    # Feedback Events
    PREDICTION_OUTCOME = 'feedback.prediction.outcome'
    PERFORMANCE_FEEDBACK = 'feedback.performance'


@dataclass
class MLEvent:
    """Base event structure"""
    id: str
    type: str
    timestamp: datetime
    userId: str
    version: str = "1.0.0"
    correlationId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    data: Dict[str, Any] = None


class MLServiceEventIntegration:
    """
    Event-driven integration for ML Service
    Subscribes to market/trading events and publishes ML decisions
    """

    def __init__(self):
        self.config = get_config()
        self.redis_client = self._create_redis_client()
        self.consumer_group = "ml-service"
        self.consumer_id = f"ml-service-{int(time.time())}"

        # Stream names
        self.STREAMS = {
            'MARKET': 'ml:stream:market',
            'ML_SERVICE': 'ml:stream:ml',
            'FEEDBACK': 'ml:stream:feedback',
            'SYSTEM': 'ml:stream:system'
        }

        # Processing queues
        self.event_queue = queue.Queue(maxsize=1000)
        self.processing_lock = Lock()
        self.is_running = False

        # Performance tracking
        self.processed_events = 0
        self.failed_events = 0
        self.last_adaptation_time = {}

        # Handlers
        self.event_handlers = {
            MLEventType.MARKET_DATA_UPDATE: self.handle_market_data,
            MLEventType.TRADE_EXECUTED: self.handle_trade_executed,
            MLEventType.POSITION_CLOSED: self.handle_position_closed,
            MLEventType.PERFORMANCE_UPDATE: self.handle_performance_update,
            MLEventType.PREDICTION_OUTCOME: self.handle_prediction_outcome
        }

        logger.info("MLServiceEventIntegration initialized")

    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client for streams"""
        return redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            password=self.config.redis.password if hasattr(self.config.redis, 'password') else None,
            db=0,
            decode_responses=True
        )

    def start(self):
        """Start event processing"""
        if self.is_running:
            logger.warning("MLServiceEventIntegration already running")
            return

        self.is_running = True

        # Initialize consumer groups
        self._init_consumer_groups()

        # Start consumer threads
        market_thread = Thread(target=self._consume_stream, args=(self.STREAMS['MARKET'],))
        feedback_thread = Thread(target=self._consume_stream, args=(self.STREAMS['FEEDBACK'],))
        processor_thread = Thread(target=self._process_events)

        market_thread.daemon = True
        feedback_thread.daemon = True
        processor_thread.daemon = True

        market_thread.start()
        feedback_thread.start()
        processor_thread.start()

        logger.info("MLServiceEventIntegration started")

    def _init_consumer_groups(self):
        """Initialize Redis consumer groups"""
        for stream_name in [self.STREAMS['MARKET'], self.STREAMS['FEEDBACK']]:
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
                # Read from stream with blocking
                messages = self.redis_client.xreadgroup(
                    self.consumer_group,
                    self.consumer_id,
                    {stream_name: '>'},
                    count=10,
                    block=5000  # 5 second timeout
                )

                if messages:
                    for stream, stream_messages in messages:
                        for msg_id, data in stream_messages:
                            # Parse event
                            event = self._parse_event(msg_id, data)

                            # Add to processing queue
                            try:
                                self.event_queue.put(event, timeout=1)

                                # Acknowledge message
                                self.redis_client.xack(stream_name, self.consumer_group, msg_id)
                            except queue.Full:
                                logger.warning(f"Event queue full, dropping event {msg_id}")

            except Exception as e:
                logger.error(f"Error consuming from stream {stream_name}: {e}")
                time.sleep(1)

    def _parse_event(self, msg_id: str, data: Dict[str, Any]) -> MLEvent:
        """Parse event from Redis stream data"""
        # Reconstruct nested structure from flat Redis fields
        event_data = {}
        nested_data = {}

        for key, value in data.items():
            if '.' in key:
                # Handle nested fields
                parts = key.split('.')
                current = nested_data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Parse value type
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
                # Top-level field
                event_data[key] = value

        # Merge nested data
        if 'data' in nested_data:
            event_data['data'] = nested_data['data']

        return MLEvent(
            id=event_data.get('id', msg_id),
            type=event_data.get('type', ''),
            timestamp=datetime.fromisoformat(event_data.get('timestamp', datetime.now().isoformat())),
            userId=event_data.get('userId', ''),
            version=event_data.get('version', '1.0.0'),
            correlationId=event_data.get('correlationId'),
            metadata=event_data.get('metadata'),
            data=event_data.get('data', {})
        )

    def _process_events(self):
        """Process events from queue"""
        while self.is_running:
            try:
                # Get event from queue
                event = self.event_queue.get(timeout=1)

                # Process based on type
                handler = self.event_handlers.get(event.type)
                if handler:
                    try:
                        handler(event)
                        self.processed_events += 1
                    except Exception as e:
                        logger.error(f"Error processing event {event.id}: {e}")
                        self.failed_events += 1
                else:
                    logger.debug(f"No handler for event type {event.type}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in event processor: {e}")

    def handle_market_data(self, event: MLEvent):
        """Handle market data update event"""
        try:
            data = event.data
            symbol = data.get('symbol')

            # Extract indicators
            indicators = data.get('indicators', {})

            # Get adaptive threshold instance
            threshold_instance = threshold_manager.get_instance(event.userId, symbol)

            # Check if adaptation needed based on market conditions
            if self._should_adapt(event.userId, symbol):
                # Collect recent performance
                performance_metrics = performance_tracker.get_recent_metrics(
                    event.userId,
                    symbol,
                    lookback_hours=24
                )

                # Trigger adaptation
                updates = threshold_instance.adapt_thresholds()

                # Publish threshold updates
                for update in updates:
                    self.publish_threshold_adjusted(
                        event.userId,
                        symbol,
                        update
                    )

                self.last_adaptation_time[f"{event.userId}:{symbol}"] = time.time()

        except Exception as e:
            logger.error(f"Error handling market data: {e}")

    def handle_trade_executed(self, event: MLEvent):
        """Handle trade execution event"""
        try:
            # Record trade for performance tracking
            performance_tracker.record_trade(
                user_id=event.userId,
                symbol=event.data.get('symbol'),
                side=event.data.get('side'),
                price=event.data.get('price'),
                quantity=event.data.get('quantity'),
                timestamp=event.timestamp
            )

        except Exception as e:
            logger.error(f"Error handling trade executed: {e}")

    def handle_position_closed(self, event: MLEvent):
        """Handle position closed event"""
        try:
            data = event.data

            # Record position outcome
            performance_tracker.record_position_outcome(
                user_id=event.userId,
                symbol=data.get('symbol'),
                pnl=data.get('pnl'),
                pnl_percent=data.get('pnlPercent'),
                holding_period=data.get('holdingPeriod'),
                close_reason=data.get('closeReason')
            )

            # Check if risk limits need adjustment
            if data.get('closeReason') == 'stop_loss':
                # Multiple stop losses might indicate need for tighter risk
                recent_stops = performance_tracker.get_recent_stop_losses(
                    event.userId,
                    lookback_hours=24
                )

                if recent_stops > 3:
                    self.publish_risk_limit_update(
                        event.userId,
                        'position_size',
                        reason="Multiple stop losses triggered"
                    )

        except Exception as e:
            logger.error(f"Error handling position closed: {e}")

    def handle_performance_update(self, event: MLEvent):
        """Handle performance update event"""
        try:
            metrics = event.data.get('metrics', {})

            # Store performance metrics
            performance_tracker.update_metrics(
                user_id=event.userId,
                metrics=metrics,
                period=event.data.get('period')
            )

            # Check if adaptation needed based on performance
            if metrics.get('winRate', 0) < 0.4 or metrics.get('sharpeRatio', 0) < 0.5:
                # Poor performance, trigger conservative adaptation
                threshold_instance = threshold_manager.get_instance(event.userId)
                threshold_instance.adapt_conservative()

                self.publish_adaptation_triggered(
                    event.userId,
                    reason="Poor performance metrics"
                )

        except Exception as e:
            logger.error(f"Error handling performance update: {e}")

    def handle_prediction_outcome(self, event: MLEvent):
        """Handle prediction outcome feedback"""
        try:
            data = event.data

            # Update model performance tracking
            performance_tracker.record_prediction_outcome(
                prediction_id=data.get('predictionId'),
                predicted_action=data.get('predicted', {}).get('action'),
                actual_return=data.get('actual', {}).get('return'),
                accuracy=data.get('accuracy')
            )

            # Use feedback to improve thresholds
            threshold_instance = threshold_manager.get_instance(event.userId)
            threshold_instance.process_feedback(
                prediction_accuracy=data.get('accuracy'),
                actual_return=data.get('actual', {}).get('return')
            )

        except Exception as e:
            logger.error(f"Error handling prediction outcome: {e}")

    def _should_adapt(self, user_id: str, symbol: str) -> bool:
        """Check if adaptation should be triggered"""
        key = f"{user_id}:{symbol}"
        last_time = self.last_adaptation_time.get(key, 0)

        # Adapt at most once per hour
        if time.time() - last_time < 3600:
            return False

        # Check if enough data collected
        recent_trades = performance_tracker.get_trade_count(
            user_id,
            symbol,
            lookback_hours=24
        )

        return recent_trades >= 10

    def publish_threshold_adjusted(self, user_id: str, symbol: str, update: Any):
        """Publish threshold adjustment event"""
        try:
            event_data = {
                'id': f"ta_{int(time.time())}_{user_id[:8]}",
                'type': MLEventType.THRESHOLD_ADJUSTED,
                'timestamp': datetime.now().isoformat(),
                'userId': user_id,
                'version': '1.0.0',
                'data.parameter': update.parameter_name,
                'data.oldValue': str(update.old_value),
                'data.newValue': str(update.new_value),
                'data.reason': update.reason,
                'data.confidence': str(update.confidence),
                'data.symbol': symbol
            }

            # Publish to ML service stream
            self.redis_client.xadd(
                self.STREAMS['ML_SERVICE'],
                event_data,
                maxlen=10000,
                approximate=True
            )

            logger.info(f"Published threshold adjustment: {update.parameter_name} for {symbol}")

        except Exception as e:
            logger.error(f"Error publishing threshold adjusted: {e}")

    def publish_risk_limit_update(self, user_id: str, limit_type: str, reason: str):
        """Publish risk limit update event"""
        try:
            event_data = {
                'id': f"rlu_{int(time.time())}_{user_id[:8]}",
                'type': MLEventType.RISK_LIMIT_UPDATE,
                'timestamp': datetime.now().isoformat(),
                'userId': user_id,
                'version': '1.0.0',
                'data.limitType': limit_type,
                'data.reason': reason
            }

            self.redis_client.xadd(
                self.STREAMS['ML_SERVICE'],
                event_data,
                maxlen=10000,
                approximate=True
            )

            logger.info(f"Published risk limit update: {limit_type} for {user_id}")

        except Exception as e:
            logger.error(f"Error publishing risk limit update: {e}")

    def publish_ml_prediction(self, user_id: str, symbol: str, prediction: Dict[str, Any]):
        """Publish ML prediction event"""
        try:
            prediction_id = f"pred_{int(time.time())}_{symbol}"

            event_data = {
                'id': f"mlp_{int(time.time())}_{user_id[:8]}",
                'type': MLEventType.ML_PREDICTION_READY,
                'timestamp': datetime.now().isoformat(),
                'userId': user_id,
                'version': '1.0.0',
                'data.predictionId': prediction_id,
                'data.symbol': symbol,
                'data.prediction.action': prediction.get('action', 'hold'),
                'data.prediction.confidence': str(prediction.get('confidence', 0)),
                'data.prediction.priceTarget': str(prediction.get('priceTarget', 0)),
                'data.prediction.stopLoss': str(prediction.get('stopLoss', 0)),
                'data.prediction.timeHorizon': str(prediction.get('timeHorizon', 60))
            }

            self.redis_client.xadd(
                self.STREAMS['ML_SERVICE'],
                event_data,
                maxlen=10000,
                approximate=True
            )

            logger.info(f"Published ML prediction for {symbol}: {prediction.get('action')}")

        except Exception as e:
            logger.error(f"Error publishing ML prediction: {e}")

    def publish_adaptation_triggered(self, user_id: str, reason: str):
        """Publish adaptation triggered event"""
        try:
            event_data = {
                'id': f"at_{int(time.time())}_{user_id[:8]}",
                'type': MLEventType.ADAPTATION_TRIGGERED,
                'timestamp': datetime.now().isoformat(),
                'userId': user_id,
                'version': '1.0.0',
                'data.reason': reason
            }

            self.redis_client.xadd(
                self.STREAMS['ML_SERVICE'],
                event_data,
                maxlen=10000,
                approximate=True
            )

            logger.info(f"Published adaptation triggered for {user_id}: {reason}")

        except Exception as e:
            logger.error(f"Error publishing adaptation triggered: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            'running': self.is_running,
            'processed_events': self.processed_events,
            'failed_events': self.failed_events,
            'queue_size': self.event_queue.qsize(),
            'redis_connected': self.redis_client.ping()
        }

    def stop(self):
        """Stop event processing"""
        self.is_running = False
        logger.info("MLServiceEventIntegration stopped")


# Global instance
ml_event_integration = MLServiceEventIntegration()