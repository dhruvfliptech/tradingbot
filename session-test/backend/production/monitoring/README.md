# Trading Bot Monitoring Stack

A comprehensive observability solution for the trading bot with Prometheus, Grafana, and intelligent alerting.

## Overview

This monitoring stack provides:

- **Real-time metrics collection** with Prometheus
- **Rich visualization** with Grafana dashboards
- **Intelligent alerting** with Telegram notifications
- **Log aggregation** with Loki
- **Distributed tracing** with Jaeger
- **Health monitoring** for all components
- **Container monitoring** with cAdvisor

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Trading Bot   │────│ Metrics Exporter│────│   Prometheus    │
│                 │    │   (Port 8000)   │    │   (Port 9090)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│  Health Checks  │    │  System Metrics │             │
│   (Port 8080)   │────│ Node Exporter   │─────────────┤
└─────────────────┘    │   (Port 9100)   │             │
                       └─────────────────┘             │
┌─────────────────┐    ┌─────────────────┐             │
│   Log Files     │────│    Promtail     │             │
│                 │    │                 │             │
└─────────────────┘    └─────────────────┘             │
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Grafana     │────│   Alertmanager  │────│ Telegram Bot    │
│   (Port 3000)   │    │   (Port 9093)   │    │  Notifications  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Environment Setup

Create a `.env` file in the monitoring directory:

```bash
# Grafana Admin Credentials
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your_secure_password

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CRITICAL_CHAT_ID=your_critical_alerts_chat_id
TELEGRAM_WARNING_CHAT_ID=your_warning_alerts_chat_id

# Database Connection
DATABASE_URL=postgresql://user:password@postgres:5432/tradingbot

# Optional: Email Configuration
SMTP_HOST=smtp.gmail.com:587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM=tradingbot@yourcompany.com
```

### 2. Launch Monitoring Stack

```bash
# Start all monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Check service status
docker-compose -f docker-compose.monitoring.yml ps

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f
```

### 3. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/your_password)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093
- **Health Checks**: http://localhost:8080/health/detailed
- **Trading Metrics**: http://localhost:8000/metrics

## Dashboards

### Trading Performance Dashboard
- Portfolio value and P&L tracking
- Performance metrics (Sharpe ratio, drawdown, win rate)
- Trade execution analysis
- Strategy comparison
- Real-time trading activity

### System Health Dashboard
- CPU, memory, and disk usage
- Network latency monitoring
- API response times
- Service health status
- Error rate tracking

### Risk Metrics Dashboard
- Value at Risk (VaR) monitoring
- Position concentration analysis
- Leverage tracking
- Correlation and liquidity risk
- Risk breach timeline

### Agent Performance Dashboard
- ML model accuracy tracking
- Ensemble consensus monitoring
- Agent reward analysis
- Model inference performance
- Training progress

## Key Metrics Tracked

### Trading Metrics
- `trading_portfolio_value_usd` - Current portfolio value
- `trading_total_return` - Total return percentage
- `trading_sharpe_ratio` - Risk-adjusted return metric
- `trading_max_drawdown` - Maximum portfolio decline
- `trading_win_rate` - Percentage of profitable trades
- `trading_open_positions` - Number of active positions

### Risk Metrics
- `risk_var_95_percent` - Value at Risk (95% confidence)
- `risk_var_99_percent` - Value at Risk (99% confidence)
- `risk_position_concentration` - Portfolio concentration risk
- `risk_leverage_ratio` - Current leverage multiplier
- `risk_correlation_score` - Portfolio correlation risk

### System Metrics
- `system_cpu_usage_percent` - CPU utilization
- `system_memory_usage_percent` - Memory utilization
- `system_disk_usage_percent` - Disk utilization
- `system_network_latency_ms` - Network response time
- `api_request_duration_seconds` - API performance

### Agent Metrics
- `agent_prediction_accuracy` - ML model accuracy
- `agent_cumulative_reward` - RL agent performance
- `ensemble_consensus_score` - Agent agreement level
- `model_inference_duration_seconds` - Prediction speed

## Alerting Rules

### Critical Alerts (Immediate notification)
- Portfolio drawdown > 25%
- VaR 95% > 10%
- System health failures
- Trading execution failures
- Model accuracy < 50%

### Warning Alerts (Monitored closely)
- Portfolio drawdown > 15%
- High CPU/memory usage
- Slow API responses
- Low model accuracy
- Risk limit approaches

### Telegram Notifications

Alerts are sent to Telegram with rich formatting:

```
🚨 CRITICAL TRADING BOT ALERT

Alert: HighDrawdown
Severity: CRITICAL
Component: TRADING

Summary: High portfolio drawdown detected
Description: Portfolio drawdown is 18.5%, exceeding the 15% warning threshold
Action Required: Review strategy performance and consider position sizing adjustments
Time: 2024-01-15 14:30:45 UTC

🔗 View in Grafana
🔗 View in Prometheus
```

## Health Checks

### Available Endpoints

- `GET /health` - Basic health check
- `GET /health/live` - Kubernetes liveness probe
- `GET /health/ready` - Kubernetes readiness probe
- `GET /health/detailed` - Comprehensive system health
- `GET /health/component/{name}` - Specific component health
- `GET /health/metrics` - Health metrics in Prometheus format

### Health Check Response Example

```json
{
  "status": "healthy",
  "components": [
    {
      "name": "database",
      "status": "healthy",
      "last_check": "2024-01-15T14:30:45Z",
      "response_time_ms": 45.2,
      "metadata": {
        "connection_pool_size": 10,
        "active_connections": 3
      }
    }
  ],
  "check_time": "2024-01-15T14:30:45Z",
  "uptime_seconds": 86400,
  "version": "1.0.0"
}
```

## Integration Guide

### Adding Custom Metrics

1. **Update Prometheus Exporter**:
```python
from prometheus_exporter import get_exporter

exporter = get_exporter()

# Record a custom metric
exporter.custom_metric = Gauge('my_custom_metric', 'Description')
exporter.custom_metric.set(42)
```

2. **Create Alert Rules**:
```yaml
# Add to alert_rules.yaml
- alert: CustomMetricHigh
  expr: my_custom_metric > 100
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Custom metric is high"
```

3. **Add Dashboard Panel**:
```json
{
  "targets": [
    {
      "expr": "my_custom_metric",
      "legendFormat": "Custom Metric"
    }
  ]
}
```

### Telegram Bot Setup

1. Create a Telegram bot:
   - Message @BotFather on Telegram
   - Use `/newbot` command
   - Get bot token

2. Get chat ID:
   - Add bot to chat/channel
   - Send a message
   - Visit: `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Copy chat ID from response

3. Configure environment variables:
```bash
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CRITICAL_CHAT_ID=-1001234567890
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing**:
   - Check service connectivity: `docker-compose logs trading-metrics-exporter`
   - Verify Prometheus targets: http://localhost:9090/targets
   - Check metric names in exporter

2. **Alerts not firing**:
   - Verify alert rules: http://localhost:9090/rules
   - Check Alertmanager: http://localhost:9093
   - Test Telegram bot token

3. **Grafana dashboards empty**:
   - Confirm Prometheus datasource connection
   - Check metric queries in dashboard
   - Verify time range selection

4. **High resource usage**:
   - Adjust scrape intervals in prometheus.yml
   - Reduce metric retention period
   - Scale monitoring services

### Performance Tuning

1. **Prometheus Optimization**:
```yaml
# prometheus.yml
global:
  scrape_interval: 30s  # Increase for less frequent collection
  evaluation_interval: 30s

# Reduce retention
storage:
  tsdb:
    retention.time: 15d
    retention.size: 5GB
```

2. **Grafana Optimization**:
```bash
# Increase memory limit
GF_DATABASE_MAX_OPEN_CONNS=25
GF_DATABASE_MAX_IDLE_CONNS=5
```

3. **Alert Tuning**:
```yaml
# Reduce notification frequency
route:
  repeat_interval: 4h  # Instead of 1h
  group_interval: 5m   # Group related alerts
```

## Maintenance

### Regular Tasks

1. **Weekly**:
   - Review alert noise and tune thresholds
   - Check disk usage for metrics storage
   - Verify backup of dashboard configurations

2. **Monthly**:
   - Update monitoring stack images
   - Review and optimize slow queries
   - Clean up old metric data if needed

3. **Quarterly**:
   - Performance review of monitoring overhead
   - Update alert rules based on trading patterns
   - Security review of credentials and access

### Backup and Recovery

1. **Export Grafana Dashboards**:
```bash
# Export all dashboards
curl -s "http://admin:password@localhost:3000/api/search?query=&" | \
jq -r '.[].uri' | \
xargs -I {} curl -s "http://admin:password@localhost:3000/api/dashboards/{}" > backup.json
```

2. **Backup Prometheus Data**:
```bash
# Create snapshot
curl -XPOST http://localhost:9090/api/v1/admin/tsdb/snapshot
# Copy snapshot from /prometheus/snapshots/
```

3. **Alert Rules Backup**:
```bash
# Rules are in version control
git add alert_rules.yaml
git commit -m "Update alert rules"
```

## Security Considerations

1. **Credential Management**:
   - Use environment variables for secrets
   - Rotate passwords regularly
   - Implement least privilege access

2. **Network Security**:
   - Use internal Docker networks
   - Consider VPN for external access
   - Enable HTTPS for production

3. **Data Protection**:
   - Encrypt metrics at rest
   - Secure backup storage
   - Audit access logs

## Support

For issues or questions:
1. Check logs: `docker-compose -f docker-compose.monitoring.yml logs`
2. Review Prometheus targets: http://localhost:9090/targets
3. Test health endpoints: http://localhost:8080/health/detailed
4. Monitor Grafana datasource status

## Contributing

To add new monitoring features:
1. Update metric definitions in `prometheus_exporter.py`
2. Add corresponding alert rules in `alert_rules.yaml`
3. Create dashboard panels in Grafana
4. Update documentation

The monitoring stack is designed to be extensible and can be customized based on specific trading strategies and risk requirements.