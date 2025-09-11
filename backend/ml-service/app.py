"""
ML Service Flask Application
Provides HTTP API for AdaptiveThreshold functionality with comprehensive monitoring
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from datetime import datetime
from typing import Dict, List, Optional
import traceback
import time

# Import our components
from adaptive_threshold import AdaptiveThreshold, threshold_manager, ThresholdUpdate
from performance_tracker import performance_tracker, TradingPerformanceSnapshot
from monitoring import logger, logged_function, health_check_manager, alert_manager
from config import get_config
from integration import integration, TradingSignalRequest

# Get configuration
config = get_config()

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=config.security.cors_origins)

# Configuration
app.config['SECRET_KEY'] = config.security.flask_secret_key
app.config['DEBUG'] = config.debug

# Rate limiting (if enabled)
if config.security.enable_rate_limiting:
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=[f"{config.security.rate_limit_per_minute} per minute"]
    )
else:
    limiter = None


# Request tracking middleware
@app.before_request
def before_request():
    """Setup request tracking"""
    request.start_time = time.time()
    request.request_id = f"{int(time.time())}-{os.getpid()}-{id(request)}"[-8:]
    
    logger.set_context(
        request_id=request.request_id,
        endpoint=request.endpoint,
        method=request.method,
        remote_addr=request.remote_addr
    )
    
    logger.debug(f"Request started: {request.method} {request.path}")

@app.after_request
def after_request(response):
    """Track request completion"""
    if hasattr(request, 'start_time'):
        duration_ms = (time.time() - request.start_time) * 1000
        
        # Record performance metrics
        performance_tracker.record_metric(
            name="http_request_duration_ms",
            value=duration_ms,
            tags={
                'method': request.method,
                'endpoint': request.endpoint or 'unknown',
                'status_code': str(response.status_code)
            }
        )
        
        logger.info(f"Request completed", extra={
            'duration_ms': duration_ms,
            'status_code': response.status_code,
            'response_size': len(response.get_data())
        })
    
    logger.clear_context()
    return response


@app.route('/health', methods=['GET'])
@logged_function("health_check")
def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Run health checks
        health_results = health_check_manager.run_checks()
        
        # Add ML service specific status
        health_results['ml_service'] = {
            'active_threshold_instances': len(threshold_manager.instances),
            'performance_tracker_status': 'healthy',
            'integration_status': integration.get_integration_status()
        }
        
        # Determine overall status
        status = 'healthy' if health_results['overall_healthy'] else 'degraded'
        status_code = 200 if health_results['overall_healthy'] else 503
        
        return jsonify({
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'checks': health_results
        }), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'error': str(e)
        }), 503


@app.route('/api/v1/thresholds/<user_id>', methods=['GET'])
def get_thresholds(user_id: str):
    """Get current thresholds for a user"""
    try:
        symbol = request.args.get('symbol')
        instance = threshold_manager.get_instance(user_id, symbol)
        thresholds = instance.get_current_thresholds()
        
        return jsonify({
            'success': True,
            'data': {
                'user_id': user_id,
                'symbol': symbol,
                'thresholds': thresholds,
                'last_adaptation': instance.last_adaptation.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting thresholds for user {user_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/thresholds/<user_id>/adapt', methods=['POST'])
def adapt_thresholds(user_id: str):
    """Trigger threshold adaptation for a user"""
    try:
        symbol = request.json.get('symbol') if request.json else None
        instance = threshold_manager.get_instance(user_id, symbol)
        
        updates = instance.adapt_thresholds()
        
        return jsonify({
            'success': True,
            'data': {
                'user_id': user_id,
                'symbol': symbol,
                'updates': [
                    {
                        'parameter': update.parameter_name,
                        'old_value': update.old_value,
                        'new_value': update.new_value,
                        'reason': update.reason,
                        'confidence': update.confidence
                    }
                    for update in updates
                ],
                'adapted_at': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error adapting thresholds for user {user_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/thresholds/<user_id>/reset', methods=['POST'])
def reset_thresholds(user_id: str):
    """Reset thresholds to default values"""
    try:
        symbol = request.json.get('symbol') if request.json else None
        instance = threshold_manager.get_instance(user_id, symbol)
        
        instance.reset_thresholds()
        
        return jsonify({
            'success': True,
            'data': {
                'user_id': user_id,
                'symbol': symbol,
                'message': 'Thresholds reset to default values',
                'thresholds': instance.get_current_thresholds()
            }
        })
        
    except Exception as e:
        logger.error(f"Error resetting thresholds for user {user_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/evaluate/<user_id>', methods=['POST'])
@logged_function("evaluate_signal")
def evaluate_signal(user_id: str):
    """Evaluate if a trading signal should result in a trade"""
    try:
        if not request.json:
            return jsonify({
                'success': False,
                'error': 'Request body required'
            }), 400
        
        signal_data = request.json.get('signal', {})
        symbol = signal_data.get('symbol', 'UNKNOWN')
        
        # Use integration layer for evaluation
        signal_request = TradingSignalRequest(
            user_id=user_id,
            signal=signal_data,
            symbol=symbol,
            request_id=getattr(request, 'request_id', None)
        )
        
        # For now, use the threshold manager directly (can be enhanced with async later)
        instance = threshold_manager.get_instance(user_id, symbol)
        should_trade = instance.should_trade(signal_data)
        current_thresholds = instance.get_current_thresholds()
        
        # Calculate processing time and confidence adjustment
        processing_time_ms = (time.time() - request.start_time) * 1000 if hasattr(request, 'start_time') else 0
        confidence_adjustment = 0.0  # Placeholder for future enhancement
        
        # Generate reasoning
        reasoning = f"Signal evaluation based on current thresholds. Should trade: {should_trade}"
        
        response = type('obj', (object,), {
            'user_id': user_id,
            'symbol': symbol,
            'should_trade': should_trade,
            'confidence_adjustment': confidence_adjustment,
            'threshold_adjustments': current_thresholds,
            'reasoning': reasoning,
            'processing_time_ms': processing_time_ms,
            'timestamp': datetime.utcnow()
        })()
        
        return jsonify({
            'success': True,
            'data': {
                'user_id': response.user_id,
                'symbol': response.symbol,
                'should_trade': response.should_trade,
                'confidence_adjustment': response.confidence_adjustment,
                'threshold_adjustments': response.threshold_adjustments,
                'reasoning': response.reasoning,
                'processing_time_ms': response.processing_time_ms,
                'evaluated_at': response.timestamp.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error evaluating signal for user {user_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/performance/<user_id>', methods=['GET'])
def get_performance(user_id: str):
    """Get performance metrics for a user"""
    try:
        symbol = request.args.get('symbol')
        days_back = int(request.args.get('days_back', 30))
        
        instance = threshold_manager.get_instance(user_id, symbol)
        metrics = instance.get_performance_metrics(days_back)
        performance_score = instance.calculate_performance_score(metrics)
        
        return jsonify({
            'success': True,
            'data': {
                'user_id': user_id,
                'symbol': symbol,
                'period_days': days_back,
                'metrics': {
                    'total_return': metrics.total_return,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'win_rate': metrics.win_rate,
                    'avg_trade_return': metrics.avg_trade_return,
                    'max_drawdown': metrics.max_drawdown,
                    'volatility': metrics.volatility
                },
                'performance_score': performance_score,
                'calculated_at': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting performance for user {user_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/adapt-all', methods=['POST'])
def adapt_all_users():
    """Adapt thresholds for all active users (admin endpoint)"""
    try:
        # Simple authentication check
        auth_key = request.headers.get('X-Admin-Key')
        if auth_key != os.getenv('ADMIN_API_KEY'):
            return jsonify({
                'success': False,
                'error': 'Unauthorized'
            }), 401
        
        results = threshold_manager.adapt_all_users()
        
        total_updates = sum(len(updates) for updates in results.values())
        
        return jsonify({
            'success': True,
            'data': {
                'total_users': len(results),
                'total_updates': total_updates,
                'results': {
                    user_key: [
                        {
                            'parameter': update.parameter_name,
                            'old_value': update.old_value,
                            'new_value': update.new_value,
                            'reason': update.reason,
                            'confidence': update.confidence
                        }
                        for update in updates
                    ]
                    for user_key, updates in results.items()
                },
                'adapted_at': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in adapt_all_users: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/stats', methods=['GET'])
@logged_function("get_stats")
def get_stats():
    """Get comprehensive service statistics"""
    try:
        # Get metrics summary
        metrics_summary = performance_tracker.metrics_collector.get_metrics_summary(hours_back=24)
        
        # Get integration status
        integration_status = integration.get_integration_status()
        
        # Get alert history
        alert_history = alert_manager.get_alert_history(hours_back=24)
        
        return jsonify({
            'success': True,
            'data': {
                'service_info': {
                    'active_threshold_instances': len(threshold_manager.instances),
                    'version': '1.0.0',
                    'environment': config.environment,
                    'uptime_seconds': time.time() - app.start_time if hasattr(app, 'start_time') else 0
                },
                'integration_status': integration_status,
                'metrics_summary': metrics_summary,
                'alerts': {
                    'total_alerts_24h': len(alert_history),
                    'recent_alerts': alert_history[:5]  # Last 5 alerts
                },
                'health_status': health_check_manager.overall_health,
                'generated_at': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/analytics/<user_id>', methods=['GET'])
@logged_function("get_analytics")
def get_analytics(user_id: str):
    """Get performance analytics for a user"""
    try:
        symbol = request.args.get('symbol')
        hours_back = int(request.args.get('hours_back', 24))
        
        # Get performance analytics
        analytics = performance_tracker.get_performance_analytics(
            user_id=user_id,
            symbol=symbol,
            hours_back=hours_back
        )
        
        return jsonify({
            'success': True,
            'data': analytics
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics for user {user_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/metrics/export', methods=['GET'])
@logged_function("export_metrics")
def export_metrics():
    """Export metrics for external analysis"""
    try:
        hours_back = int(request.args.get('hours_back', 24))
        format_type = request.args.get('format', 'json')
        
        if format_type == 'prometheus':
            # Export Prometheus metrics
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
        
        elif format_type == 'csv':
            # This would trigger CSV export
            # For now, return JSON with export info
            return jsonify({
                'success': True,
                'message': 'CSV export functionality available via integration layer',
                'data': {
                    'export_available': True,
                    'formats': ['json', 'prometheus', 'csv'],
                    'period_hours': hours_back
                }
            })
        
        else:
            # Default JSON export
            metrics_summary = performance_tracker.metrics_collector.get_metrics_summary(hours_back)
            return jsonify({
                'success': True,
                'data': {
                    'format': 'json',
                    'period_hours': hours_back,
                    'metrics': metrics_summary,
                    'exported_at': datetime.utcnow().isoformat()
                }
            })
        
    except Exception as e:
        logger.error(f"Error exporting metrics: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/alerts', methods=['GET'])
@logged_function("get_alerts")
def get_alerts():
    """Get alert history"""
    try:
        hours_back = int(request.args.get('hours_back', 24))
        severity = request.args.get('severity')
        
        alerts = alert_manager.get_alert_history(hours_back)
        
        # Filter by severity if specified
        if severity:
            alerts = [a for a in alerts if a.get('severity') == severity]
        
        return jsonify({
            'success': True,
            'data': {
                'alerts': alerts,
                'total': len(alerts),
                'period_hours': hours_back,
                'severity_filter': severity,
                'retrieved_at': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# Application startup
@app.before_first_request
def initialize_app():
    """Initialize application components"""
    app.start_time = time.time()
    
    logger.info("ML Service starting up", extra={
        'environment': config.environment,
        'debug': config.debug,
        'version': '1.0.0'
    })
    
    # Initialize performance tracking
    performance_tracker.record_metric(
        name="service_startup",
        value=1.0,
        tags={'version': '1.0.0', 'environment': config.environment}
    )
    
    # Force flush initial metrics
    performance_tracker.metrics_collector.force_flush()
    
    logger.info("ML Service initialization completed")


# Graceful shutdown
import atexit

def cleanup():
    """Cleanup resources on shutdown"""
    logger.info("ML Service shutting down...")
    
    try:
        # Flush all metrics
        performance_tracker.metrics_collector.force_flush()
        
        # Close integration resources (if needed)
        # await integration.close()  # Would be called in async context
        
        logger.info("ML Service shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

atexit.register(cleanup)


if __name__ == '__main__':
    port = int(os.getenv('PORT', config.port))
    debug = config.debug
    host = config.host
    
    logger.info(f"Starting ML Service on {host}:{port} (debug={debug})")
    
    app.run(host=host, port=port, debug=debug, threaded=True)