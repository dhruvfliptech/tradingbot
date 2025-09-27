#!/bin/bash

# System Monitoring Script for Trading Bot
# Provides real-time monitoring of the trading system

echo "📊 Trading Bot System Monitor"
echo "============================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Backend API base URL
API_URL="http://localhost:3001/api/v1"

# Function to check service status
check_service() {
    if curl -s "$1" > /dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
}

# Function to format JSON response
format_json() {
    if command -v jq &> /dev/null; then
        echo "$1" | jq '.'
    else
        echo "$1"
    fi
}

while true; do
    clear
    echo "📊 Trading Bot System Monitor"
    echo "============================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Check services
    echo -e "${BLUE}Service Status:${NC}"
    echo -n "  • Backend API: "
    check_service "http://localhost:3001/health"
    echo -n "  • Frontend: "
    check_service "http://localhost:5173"
    echo ""

    # Get monitoring dashboard if backend is running
    if curl -s "http://localhost:3001/health" > /dev/null; then

        # Get system health
        echo -e "${BLUE}System Health:${NC}"
        HEALTH=$(curl -s "$API_URL/monitoring/health" 2>/dev/null)
        if [ ! -z "$HEALTH" ]; then
            echo "$HEALTH" | jq -r '.data | "  • Overall: \(.overall)\n  • Uptime: \(.uptime)s\n  • CPU: \(.systemMetrics.cpu.usage)%\n  • Memory: \(.systemMetrics.memory.usage)%"' 2>/dev/null || echo "  Unable to parse health data"
        else
            echo "  Unable to fetch health data"
        fi
        echo ""

        # Get recent alerts
        echo -e "${BLUE}Recent Alerts:${NC}"
        ALERTS=$(curl -s "$API_URL/monitoring/alerts?limit=3" 2>/dev/null)
        if [ ! -z "$ALERTS" ]; then
            echo "$ALERTS" | jq -r '.data[] | "  [\(.severity)] \(.title)"' 2>/dev/null || echo "  No recent alerts"
        else
            echo "  No alerts available"
        fi
        echo ""

        # Get trading bot status
        echo -e "${BLUE}Trading Bot Status:${NC}"
        BOT_STATUS=$(curl -s "$API_URL/trading/bot/status" 2>/dev/null)
        if [ ! -z "$BOT_STATUS" ]; then
            echo "$BOT_STATUS" | jq -r '.data | "  • Status: \(.isRunning)\n  • Cycles: \(.cyclesExecuted)\n  • Positions: \(.openPositions)"' 2>/dev/null || echo "  Bot not initialized"
        else
            echo "  Bot status unavailable"
        fi
        echo ""

        # Get account summary
        echo -e "${BLUE}Account Summary:${NC}"
        ACCOUNT=$(curl -s "$API_URL/brokers/account" 2>/dev/null)
        if [ ! -z "$ACCOUNT" ]; then
            echo "$ACCOUNT" | jq -r '.data | "  • Broker: \(.broker)\n  • Cash: $\(.cashBalance)\n  • Buying Power: $\(.buyingPower)"' 2>/dev/null || echo "  Account data unavailable"
        else
            echo "  Unable to fetch account data"
        fi
        echo ""

        # Performance metrics
        echo -e "${BLUE}Performance Metrics:${NC}"
        METRICS=$(curl -s "$API_URL/metrics/performance?period=1d" 2>/dev/null)
        if [ ! -z "$METRICS" ]; then
            echo "$METRICS" | jq -r '.data | "  • Total P&L: $\(.totalPnL)\n  • Win Rate: \(.winRate)%\n  • Sharpe Ratio: \(.sharpeRatio)"' 2>/dev/null || echo "  Metrics unavailable"
        else
            echo "  Performance data unavailable"
        fi

    else
        echo -e "${RED}Backend is not running!${NC}"
        echo "Run ./start-local.sh to start the system"
    fi

    echo ""
    echo "================================"
    echo "Press Ctrl+C to exit | Refreshing every 5 seconds..."

    sleep 5
done