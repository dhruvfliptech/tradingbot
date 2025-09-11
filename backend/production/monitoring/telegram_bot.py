"""
Telegram Bot for Trading Bot Notifications
Sends real-time alerts and notifications about trading performance, system health, and risk events.
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
from jinja2 import Template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"

class AlertType(Enum):
    """Types of alerts"""
    TRADING_PERFORMANCE = "trading_performance"
    RISK_BREACH = "risk_breach"
    SYSTEM_ERROR = "system_error"
    HEALTH_CHECK = "health_check"
    TRADE_EXECUTION = "trade_execution"
    STRATEGY_UPDATE = "strategy_update"
    MARKET_EVENT = "market_event"

@dataclass
class Alert:
    """Alert data structure"""
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    chat_ids: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }

class TelegramNotifier:
    """
    Telegram bot for sending trading bot notifications.
    
    Features:
    - Real-time alert delivery
    - Rich message formatting
    - Multiple chat support
    - Alert throttling and deduplication
    - Performance reports
    - Interactive commands
    """
    
    def __init__(self, bot_token: str, default_chat_ids: List[str] = None):
        self.bot_token = bot_token
        self.default_chat_ids = default_chat_ids or []
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Alert management
        self.sent_alerts = {}  # For deduplication
        self.alert_queue = asyncio.Queue()
        self.rate_limits = {}  # Per chat rate limiting
        self.alert_history = []  # Keep history for analysis
        
        # Message templates
        self._init_templates()
        
        # Configuration
        self.max_message_length = 4096
        self.rate_limit_window = 60  # seconds
        self.max_messages_per_window = 20
        self.alert_retention_hours = 24
        
    def _init_templates(self):
        """Initialize message templates"""
        self.templates = {
            'trading_performance': Template("""
🚀 **Trading Performance Alert**

**Strategy:** {{ strategy }}
**Status:** {{ status }}

📊 **Performance Metrics:**
• Total Return: {{ total_return }}%
• Sharpe Ratio: {{ sharpe_ratio }}
• Max Drawdown: {{ max_drawdown }}%
• Win Rate: {{ win_rate }}%

💰 **Portfolio:**
• Value: ${{ portfolio_value | round(2) }}
• Unrealized P&L: ${{ unrealized_pnl | round(2) }}
• Open Positions: {{ open_positions }}

⏰ {{ timestamp }}
            """.strip()),
            
            'risk_breach': Template("""
⚠️ **RISK BREACH ALERT**

**Risk Type:** {{ risk_type }}
**Severity:** {{ severity }}

🎯 **Details:**
• Current Value: {{ current_value }}
• Threshold: {{ threshold }}
• Breach Margin: {{ breach_margin }}%

📋 **Action Taken:** {{ action_taken }}

⏰ {{ timestamp }}
            """.strip()),
            
            'system_error': Template("""
🔴 **System Error Alert**

**Service:** {{ service }}
**Error Type:** {{ error_type }}

❌ **Error Details:**
{{ error_message }}

🔧 **Status:** {{ status }}
{% if recovery_action %}
**Recovery Action:** {{ recovery_action }}
{% endif %}

⏰ {{ timestamp }}
            """.strip()),
            
            'trade_execution': Template("""
📈 **Trade Executed**

**Symbol:** {{ symbol }}
**Side:** {{ side }}
**Size:** {{ size }}
**Price:** ${{ price }}

💵 **Trade Value:** ${{ trade_value | round(2) }}
**Strategy:** {{ strategy }}
**Execution Time:** {{ execution_time }}ms

{% if profit_loss %}
**P&L:** ${{ profit_loss | round(2) }}
{% endif %}

⏰ {{ timestamp }}
            """.strip()),
            
            'health_check': Template("""
{% if status == 'healthy' %}🟢{% elif status == 'warning' %}🟡{% else %}🔴{% endif %} **System Health Update**

**Overall Status:** {{ status | upper }}
**Uptime:** {{ uptime_hours }}h {{ uptime_minutes }}m

📊 **Component Status:**
{% for component in components %}
{% if component.status == 'healthy' %}✅{% elif component.status == 'warning' %}⚠️{% else %}❌{% endif %} {{ component.name }}: {{ component.status }}
{% endfor %}

{% if issues %}
**Issues:**
{% for issue in issues %}
• {{ issue }}
{% endfor %}
{% endif %}

⏰ {{ timestamp }}
            """.strip()),
            
            'market_event': Template("""
📢 **Market Event Alert**

**Event:** {{ event_type }}
**Market:** {{ market }}

📊 **Details:**
{{ event_details }}

📈 **Impact Assessment:**
• Volatility: {{ volatility_impact }}
• Volume: {{ volume_impact }}
• Strategy Adjustment: {{ strategy_adjustment }}

⏰ {{ timestamp }}
            """.strip()),
            
            'daily_summary': Template("""
📊 **Daily Trading Summary**

**Date:** {{ date }}

💰 **Performance:**
• Daily Return: {{ daily_return }}%
• Total Return: {{ total_return }}%
• Trades Executed: {{ trades_count }}
• Win Rate: {{ win_rate }}%

📈 **Best Performer:** {{ best_strategy }} ({{ best_return }}%)
📉 **Worst Performer:** {{ worst_strategy }} ({{ worst_return }}%)

🎯 **Portfolio:**
• Value: ${{ portfolio_value | round(2) }}
• Available Cash: ${{ available_cash | round(2) }}
• Open Positions: {{ open_positions }}

⚠️ **Risk Metrics:**
• VaR (95%): {{ var_95 }}%
• Max Drawdown: {{ max_drawdown }}%
• Position Concentration: {{ concentration }}%

🔧 **System:**
• Uptime: {{ uptime }}%
• Errors: {{ error_count }}
• Avg Response Time: {{ avg_response_time }}ms

---
Generated by Trading Bot Monitor
            """.strip())
        }
    
    async def send_message(self, chat_id: str, message: str, parse_mode: str = "Markdown") -> bool:
        """Send message to Telegram chat"""
        try:
            # Check rate limiting
            if not self._check_rate_limit(chat_id):
                logger.warning(f"Rate limit exceeded for chat {chat_id}")
                return False
            
            # Truncate message if too long
            if len(message) > self.max_message_length:
                message = message[:self.max_message_length - 100] + "\n\n... (message truncated)"
            
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self._update_rate_limit(chat_id)
                        logger.info(f"Message sent successfully to chat {chat_id}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to send message to {chat_id}: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending message to {chat_id}: {e}")
            return False
    
    def _check_rate_limit(self, chat_id: str) -> bool:
        """Check if message can be sent without hitting rate limit"""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.rate_limit_window)
        
        if chat_id not in self.rate_limits:
            self.rate_limits[chat_id] = []
        
        # Remove old messages from window
        self.rate_limits[chat_id] = [
            msg_time for msg_time in self.rate_limits[chat_id]
            if msg_time > window_start
        ]
        
        return len(self.rate_limits[chat_id]) < self.max_messages_per_window
    
    def _update_rate_limit(self, chat_id: str):
        """Update rate limit counter"""
        if chat_id not in self.rate_limits:
            self.rate_limits[chat_id] = []
        self.rate_limits[chat_id].append(datetime.now())
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to configured chats"""
        try:
            # Check for duplicate alerts
            alert_key = f"{alert.alert_type.value}_{alert.title}_{alert.severity.value}"
            if self._is_duplicate_alert(alert_key, alert.message):
                logger.info(f"Skipping duplicate alert: {alert_key}")
                return True
            
            # Determine chat IDs
            chat_ids = alert.chat_ids or self.default_chat_ids
            if not chat_ids:
                logger.warning("No chat IDs configured for alert")
                return False
            
            # Format message
            message = self._format_alert_message(alert)
            
            # Send to all configured chats
            success_count = 0
            for chat_id in chat_ids:
                if await self.send_message(chat_id, message):
                    success_count += 1
            
            # Mark as sent and add to history
            self._mark_alert_sent(alert_key, alert.message)
            self.alert_history.append(alert)
            self._cleanup_alert_history()
            
            logger.info(f"Alert sent to {success_count}/{len(chat_ids)} chats")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
    
    def _is_duplicate_alert(self, alert_key: str, message: str) -> bool:
        """Check if alert is duplicate within time window"""
        if alert_key in self.sent_alerts:
            last_sent, last_message = self.sent_alerts[alert_key]
            time_diff = datetime.now() - last_sent
            
            # Same message within 5 minutes is considered duplicate
            if time_diff < timedelta(minutes=5) and last_message == message:
                return True
                
        return False
    
    def _mark_alert_sent(self, alert_key: str, message: str):
        """Mark alert as sent"""
        self.sent_alerts[alert_key] = (datetime.now(), message)
        
        # Cleanup old sent alerts
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.sent_alerts = {
            key: (time, msg) for key, (time, msg) in self.sent_alerts.items()
            if time > cutoff_time
        }
    
    def _cleanup_alert_history(self):
        """Remove old alerts from history"""
        cutoff_time = datetime.now() - timedelta(hours=self.alert_retention_hours)
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
    
    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert using appropriate template"""
        try:
            template_name = alert.alert_type.value
            if template_name in self.templates:
                template = self.templates[template_name]
                
                # Prepare template context
                context = {
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'severity': alert.severity.value.upper(),
                    **(alert.metadata or {})
                }
                
                return template.render(**context)
            else:
                # Fallback to simple format
                severity_emoji = {
                    AlertSeverity.INFO: "ℹ️",
                    AlertSeverity.WARNING: "⚠️", 
                    AlertSeverity.CRITICAL: "🚨",
                    AlertSeverity.SUCCESS: "✅"
                }.get(alert.severity, "📢")
                
                return f"{severity_emoji} **{alert.title}**\n\n{alert.message}\n\n⏰ {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                
        except Exception as e:
            logger.error(f"Error formatting alert message: {e}")
            return f"🚨 **Alert**\n\n{alert.title}\n\n{alert.message}\n\n⏰ {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    
    async def send_trading_performance_alert(self, strategy: str, metrics: Dict[str, Any], chat_ids: List[str] = None):
        """Send trading performance alert"""
        severity = AlertSeverity.INFO
        if metrics.get('max_drawdown', 0) > 10:
            severity = AlertSeverity.WARNING
        if metrics.get('max_drawdown', 0) > 20:
            severity = AlertSeverity.CRITICAL
            
        alert = Alert(
            alert_type=AlertType.TRADING_PERFORMANCE,
            severity=severity,
            title=f"Trading Performance Update - {strategy}",
            message="Performance metrics updated",
            timestamp=datetime.now(),
            metadata={'strategy': strategy, **metrics},
            chat_ids=chat_ids
        )
        
        return await self.send_alert(alert)
    
    async def send_risk_breach_alert(self, risk_type: str, current_value: float, 
                                   threshold: float, action_taken: str, chat_ids: List[str] = None):
        """Send risk breach alert"""
        breach_margin = ((current_value - threshold) / threshold) * 100
        severity = AlertSeverity.WARNING if breach_margin < 50 else AlertSeverity.CRITICAL
        
        alert = Alert(
            alert_type=AlertType.RISK_BREACH,
            severity=severity,
            title=f"Risk Breach: {risk_type}",
            message=f"Risk threshold exceeded by {breach_margin:.1f}%",
            timestamp=datetime.now(),
            metadata={
                'risk_type': risk_type,
                'current_value': current_value,
                'threshold': threshold,
                'breach_margin': breach_margin,
                'action_taken': action_taken
            },
            chat_ids=chat_ids
        )
        
        return await self.send_alert(alert)
    
    async def send_system_error_alert(self, service: str, error_type: str, error_message: str,
                                    recovery_action: str = None, chat_ids: List[str] = None):
        """Send system error alert"""
        alert = Alert(
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.CRITICAL,
            title=f"System Error: {service}",
            message=f"{error_type}: {error_message}",
            timestamp=datetime.now(),
            metadata={
                'service': service,
                'error_type': error_type,
                'error_message': error_message,
                'recovery_action': recovery_action,
                'status': 'investigating'
            },
            chat_ids=chat_ids
        )
        
        return await self.send_alert(alert)
    
    async def send_trade_execution_alert(self, symbol: str, side: str, size: float, price: float,
                                       strategy: str, execution_time: float, profit_loss: float = None,
                                       chat_ids: List[str] = None):
        """Send trade execution alert"""
        trade_value = size * price
        
        alert = Alert(
            alert_type=AlertType.TRADE_EXECUTION,
            severity=AlertSeverity.SUCCESS,
            title=f"Trade Executed: {symbol}",
            message=f"{side.upper()} {size} {symbol} @ ${price}",
            timestamp=datetime.now(),
            metadata={
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': price,
                'trade_value': trade_value,
                'strategy': strategy,
                'execution_time': execution_time,
                'profit_loss': profit_loss
            },
            chat_ids=chat_ids
        )
        
        return await self.send_alert(alert)
    
    async def send_health_status_alert(self, overall_status: str, components: List[Dict], 
                                     uptime_seconds: float, issues: List[str] = None,
                                     chat_ids: List[str] = None):
        """Send system health status alert"""
        severity = AlertSeverity.INFO
        if overall_status == 'warning':
            severity = AlertSeverity.WARNING
        elif overall_status == 'critical':
            severity = AlertSeverity.CRITICAL
            
        uptime_hours = int(uptime_seconds // 3600)
        uptime_minutes = int((uptime_seconds % 3600) // 60)
        
        alert = Alert(
            alert_type=AlertType.HEALTH_CHECK,
            severity=severity,
            title="System Health Status",
            message=f"Overall status: {overall_status.upper()}",
            timestamp=datetime.now(),
            metadata={
                'status': overall_status,
                'components': components,
                'uptime_hours': uptime_hours,
                'uptime_minutes': uptime_minutes,
                'issues': issues or []
            },
            chat_ids=chat_ids
        )
        
        return await self.send_alert(alert)
    
    async def send_daily_summary(self, summary_data: Dict[str, Any], chat_ids: List[str] = None):
        """Send daily trading summary"""
        alert = Alert(
            alert_type=AlertType.STRATEGY_UPDATE,
            severity=AlertSeverity.INFO,
            title="Daily Trading Summary",
            message="Daily performance and system summary",
            timestamp=datetime.now(),
            metadata=summary_data,
            chat_ids=chat_ids
        )
        
        # Use daily summary template
        template = self.templates['daily_summary']
        context = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            **summary_data
        }
        
        message = template.render(**context)
        
        # Send directly without using alert formatting
        chat_ids = chat_ids or self.default_chat_ids
        for chat_id in chat_ids:
            await self.send_message(chat_id, message)
    
    async def start_alert_processor(self):
        """Start background alert processing"""
        while True:
            try:
                # Process alerts from queue
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                await self.send_alert(alert)
                self.alert_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    def queue_alert(self, alert: Alert):
        """Add alert to processing queue"""
        self.alert_queue.put_nowait(alert)
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        recent_alerts = [a for a in self.alert_history if a.timestamp > last_hour]
        daily_alerts = [a for a in self.alert_history if a.timestamp > last_day]
        
        return {
            'total_alerts_24h': len(daily_alerts),
            'alerts_last_hour': len(recent_alerts),
            'critical_alerts_24h': len([a for a in daily_alerts if a.severity == AlertSeverity.CRITICAL]),
            'most_common_alert_type': max(
                [a.alert_type.value for a in daily_alerts],
                key=[a.alert_type.value for a in daily_alerts].count
            ) if daily_alerts else None,
            'alert_history_size': len(self.alert_history)
        }

# Factory function for creating notifier instances
def create_telegram_notifier(bot_token: str = None, chat_ids: List[str] = None) -> TelegramNotifier:
    """Create Telegram notifier instance"""
    if not bot_token:
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable or bot_token parameter required")
    
    if not chat_ids:
        chat_ids_str = os.getenv('TELEGRAM_CHAT_IDS', '')
        chat_ids = [cid.strip() for cid in chat_ids_str.split(',') if cid.strip()]
    
    return TelegramNotifier(bot_token, chat_ids)

# Example usage and testing
async def main():
    """Example usage of Telegram notifier"""
    # Initialize notifier (requires environment variables)
    try:
        notifier = create_telegram_notifier()
        
        # Test trading performance alert
        await notifier.send_trading_performance_alert(
            strategy="Momentum Strategy",
            metrics={
                'total_return': 12.5,
                'sharpe_ratio': 1.8,
                'max_drawdown': -3.2,
                'win_rate': 68.5,
                'portfolio_value': 125000.0,
                'unrealized_pnl': 2500.0,
                'open_positions': 4
            }
        )
        
        # Test risk breach alert
        await notifier.send_risk_breach_alert(
            risk_type="Position Concentration",
            current_value=35.0,
            threshold=30.0,
            action_taken="Reduced position size by 20%"
        )
        
        # Test system error alert
        await notifier.send_system_error_alert(
            service="ML Service",
            error_type="Connection Timeout",
            error_message="Failed to connect to ML inference endpoint after 3 retries",
            recovery_action="Fallback to cached predictions activated"
        )
        
        print("Test alerts sent successfully!")
        
    except Exception as e:
        logger.error(f"Failed to send test alerts: {e}")

if __name__ == "__main__":
    asyncio.run(main())