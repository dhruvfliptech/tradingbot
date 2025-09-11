# API Endpoints Documentation

## Base URL
```
Production: https://api.tradingbot.com/api/v1
Development: http://localhost:3000/api/v1
```

## Authentication
All endpoints require JWT token in Authorization header:
```
Authorization: Bearer <jwt_token>
```

## Market Data Endpoints

### GET /market/prices
Get real-time crypto prices
```json
{
  "symbols": ["BTC", "ETH", "SOL"],
  "include_indicators": true
}
```

Response:
```json
{
  "success": true,
  "data": [
    {
      "symbol": "BTC",
      "price": 43250.00,
      "change_24h": 2.5,
      "volume_24h": 15000000000,
      "indicators": {
        "rsi": 65.2,
        "macd": 0.045,
        "ma_20": 42800,
        "ma_50": 41500
      }
    }
  ]
}
```

### GET /market/watchlist
Get user's watchlist with live data
```json
{
  "success": true,
  "data": {
    "symbols": ["BTC", "ETH", "SOL", "ADA", "DOT"],
    "prices": [...]
  }
}
```

### PUT /market/watchlist
Update user's watchlist
```json
{
  "symbols": ["BTC", "ETH", "SOL", "MATIC", "LINK"]
}
```

## Trading Endpoints

### GET /trading/status
Get trading bot status
```json
{
  "success": true,
  "data": {
    "active": true,
    "uptime": 86400,
    "last_trade": "2024-01-15T10:30:00Z",
    "active_pairs": 5,
    "pending_orders": 2
  }
}
```

### POST /trading/start
Start trading bot
```json
{
  "watchlist": ["BTC", "ETH", "SOL"],
  "settings": {
    "confidence_threshold": 0.75,
    "max_position_size": 1000,
    "risk_percentage": 2.0,
    "cooldown_minutes": 30
  }
}
```

### POST /trading/stop
Stop trading bot
```json
{
  "success": true,
  "message": "Trading bot stopped successfully"
}
```

### GET /trading/signals
Get trading signals
Query params: `limit`, `symbol`, `confidence_min`
```json
{
  "success": true,
  "data": [
    {
      "symbol": "BTC",
      "action": "BUY",
      "confidence": 0.85,
      "reason": "Technical breakout above resistance",
      "price_target": 45000,
      "stop_loss": 41000,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### POST /trading/orders
Place manual order
```json
{
  "symbol": "BTC",
  "side": "buy",
  "quantity": "0.1",
  "order_type": "market",
  "limit_price": 43000
}
```

### GET /trading/orders
Get order history
Query params: `status`, `symbol`, `limit`, `offset`
```json
{
  "success": true,
  "data": [
    {
      "id": "order_123",
      "symbol": "BTC",
      "side": "buy",
      "quantity": "0.1",
      "filled_quantity": "0.1",
      "avg_price": 43250,
      "status": "filled",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### DELETE /trading/orders/:orderId
Cancel pending order
```json
{
  "success": true,
  "message": "Order cancelled successfully"
}
```

### GET /trading/positions
Get current positions
```json
{
  "success": true,
  "data": [
    {
      "symbol": "BTC",
      "quantity": "0.5",
      "avg_cost": 42000,
      "current_price": 43250,
      "unrealized_pnl": 625,
      "unrealized_pnl_percent": 2.98
    }
  ]
}
```

### POST /trading/positions/:symbol/close
Close position
```json
{
  "percentage": 100
}
```

### PUT /trading/settings
Update trading settings
```json
{
  "confidence_threshold": 0.80,
  "max_position_size": 2000,
  "risk_percentage": 1.5,
  "cooldown_minutes": 45,
  "adaptive_threshold_enabled": true
}
```

### GET /trading/performance
Get performance metrics
Query params: `period` (1h, 24h, 7d, 30d, all)
```json
{
  "success": true,
  "data": {
    "period": "24h",
    "total_return_percent": 2.5,
    "sharpe_ratio": 1.2,
    "max_drawdown": -1.8,
    "win_rate": 65.5,
    "total_trades": 12,
    "avg_trade_return": 0.8,
    "best_trade": 5.2,
    "worst_trade": -2.1
  }
}
```

## Analytics Endpoints

### GET /analytics/dashboard
Get dashboard analytics
```json
{
  "success": true,
  "data": {
    "portfolio_value": 10500,
    "day_pnl": 250,
    "day_pnl_percent": 2.4,
    "active_positions": 3,
    "total_trades_today": 5,
    "win_rate_today": 80
  }
}
```

### GET /analytics/charts/performance
Get performance chart data
Query params: `period`, `granularity`
```json
{
  "success": true,
  "data": {
    "timestamps": [...],
    "portfolio_values": [...],
    "returns": [...]
  }
}
```

### GET /analytics/charts/trades
Get trades chart data
```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "symbol": "BTC",
      "side": "buy",
      "price": 43250,
      "quantity": 0.1,
      "pnl": 125
    }
  ]
}
```

## User Management Endpoints

### GET /users/profile
Get user profile
```json
{
  "success": true,
  "data": {
    "id": "user_123",
    "email": "user@example.com",
    "created_at": "2024-01-01T00:00:00Z",
    "subscription_tier": "pro",
    "api_keys_configured": true
  }
}
```

### PUT /users/profile
Update user profile
```json
{
  "notification_preferences": {
    "email_alerts": true,
    "trade_notifications": true,
    "performance_reports": "daily"
  }
}
```

### POST /users/api-keys
Add/update API keys
```json
{
  "provider": "alpaca",
  "api_key": "encrypted_key",
  "secret_key": "encrypted_secret",
  "paper_trading": true
}
```

## WebSocket Events

### Connection
```javascript
const socket = io('wss://api.tradingbot.com', {
  auth: {
    token: 'jwt_token'
  }
});
```

### Events Received

#### price_update
```json
{
  "event": "price_update",
  "data": {
    "symbol": "BTC",
    "price": 43275,
    "change": 0.25,
    "timestamp": "2024-01-15T10:30:15Z"
  }
}
```

#### signal_generated
```json
{
  "event": "signal_generated",
  "data": {
    "symbol": "BTC",
    "action": "BUY",
    "confidence": 0.85,
    "reason": "Technical breakout",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### order_filled
```json
{
  "event": "order_filled",
  "data": {
    "order_id": "order_123",
    "symbol": "BTC",
    "side": "buy",
    "quantity": 0.1,
    "price": 43250,
    "timestamp": "2024-01-15T10:30:30Z"
  }
}
```

#### bot_status_changed
```json
{
  "event": "bot_status_changed",
  "data": {
    "active": true,
    "reason": "Started by user",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Error Responses

All errors follow this format:
```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid request parameters",
    "details": {
      "field": "symbol",
      "reason": "Symbol is required"
    }
  }
}
```

### Common Error Codes
- `UNAUTHORIZED`: Invalid or missing authentication
- `FORBIDDEN`: Insufficient permissions
- `INVALID_REQUEST`: Bad request parameters
- `NOT_FOUND`: Resource not found
- `RATE_LIMITED`: Too many requests
- `INTERNAL_ERROR`: Server error
- `TRADING_DISABLED`: Trading is currently disabled
- `INSUFFICIENT_FUNDS`: Not enough balance for trade