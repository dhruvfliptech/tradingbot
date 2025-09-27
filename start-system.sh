#!/bin/bash

# Complete System Startup Script
# This starts both frontend and backend with all services

echo "🚀 Starting Complete Trading System"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Kill any existing processes
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
pkill -f "ts-node-dev" 2>/dev/null
pkill -f "vite" 2>/dev/null
pkill -f "node simple-server.js" 2>/dev/null
sleep 2

# Start backend
echo -e "${BLUE}Starting Backend Server...${NC}"
cd backend
npm run dev > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to initialize...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:3001/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is running${NC}"
        break
    fi
    sleep 1
done

# Start frontend
echo -e "${BLUE}Starting Frontend...${NC}"
npm run dev > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend
echo -e "${YELLOW}Waiting for frontend...${NC}"
sleep 3

# Check if everything is running
BACKEND_STATUS="${RED}✗${NC}"
FRONTEND_STATUS="${RED}✗${NC}"

if curl -s http://localhost:3001/health > /dev/null 2>&1; then
    BACKEND_STATUS="${GREEN}✓${NC}"
fi

if curl -s http://localhost:5173 > /dev/null 2>&1; then
    FRONTEND_STATUS="${GREEN}✓${NC}"
fi

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}    System Status${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "Backend API: $BACKEND_STATUS http://localhost:3001"
echo -e "Frontend:    $FRONTEND_STATUS http://localhost:5173"
echo ""
echo -e "${YELLOW}Services:${NC}"
echo "  • Binance.US API: Configured"
echo "  • Alpaca API: Configured (Paper)"
echo "  • WebSocket: ws://localhost:3001"
echo "  • Supabase: Connected"
echo ""
echo -e "${YELLOW}Account Access:${NC}"
echo "  • Use Demo Mode (no login required)"
echo "  • Or create a new account"
echo ""
echo -e "${YELLOW}Safety Settings:${NC}"
echo "  • Max Position: \$1,000"
echo "  • Max Daily Loss: \$500"
echo "  • Max Open Positions: 5"
echo ""
echo -e "${BLUE}📊 Open Dashboard: http://localhost:5173${NC}"
echo ""
echo -e "${YELLOW}Monitoring:${NC}"
echo "  • Backend logs: tail -f /tmp/backend.log"
echo "  • Frontend logs: tail -f /tmp/frontend.log"
echo "  • System monitor: ./monitor-system.sh"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $FRONTEND_PID 2>/dev/null
    kill $BACKEND_PID 2>/dev/null
    pkill -f "ts-node-dev" 2>/dev/null
    pkill -f "vite" 2>/dev/null
    echo -e "${GREEN}✓ System stopped${NC}"
}

# Set up trap to cleanup on Ctrl+C
trap cleanup EXIT

# Keep running
wait