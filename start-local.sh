#!/bin/bash

# Trading Bot Local Development Startup Script
# This script starts both frontend and backend for local testing

echo "🚀 Starting Trading Bot System (Local Development)"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}❌ Port $1 is already in use${NC}"
        echo "Please stop the process using port $1 and try again"
        exit 1
    fi
}

# Function to check if .env file exists
check_env() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}❌ Missing $1 file${NC}"
        echo "Please ensure you have created the environment file with your API keys"
        exit 1
    fi
}

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js found:${NC} $(node --version)"

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ npm found:${NC} $(npm --version)"

# Check environment files
echo -e "${BLUE}🔐 Checking environment files...${NC}"
check_env ".env"
check_env "backend/.env"
echo -e "${GREEN}✓ Environment files found${NC}"

# Check ports
echo -e "${BLUE}🔌 Checking port availability...${NC}"
check_port 3001  # Backend
check_port 5173  # Frontend
echo -e "${GREEN}✓ Ports are available${NC}"

# Install dependencies if needed
echo -e "${BLUE}📦 Checking dependencies...${NC}"

# Frontend dependencies
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

# Backend dependencies
if [ ! -d "backend/node_modules" ]; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    cd backend && npm install && cd ..
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Display safety warnings for live trading
echo ""
echo -e "${YELLOW}⚠️  LIVE TRADING WARNING ⚠️${NC}"
echo "================================================"
echo -e "${YELLOW}You are about to start the system with LIVE BINANCE.US trading enabled!${NC}"
echo ""
echo "Current Safety Settings (from backend/.env):"
grep -E "^MAX_POSITION_SIZE_USD|^MAX_DAILY_LOSS_USD|^MAX_OPEN_POSITIONS|^ENABLE_LIVE_TRADING" backend/.env | while read line; do
    echo "  • $line"
done
echo ""
echo -e "${RED}IMPORTANT:${NC}"
echo "  1. This will use REAL MONEY on Binance.US"
echo "  2. Start with small position sizes"
echo "  3. Monitor closely via the dashboard"
echo "  4. Use the emergency stop button if needed (Ctrl+Shift+E)"
echo ""
read -p "Do you want to continue? (yes/no): " -n 3 -r
echo
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo "Startup cancelled for safety"
    exit 1
fi

# Start the backend
echo ""
echo -e "${BLUE}🚀 Starting Backend Server...${NC}"
echo "================================================"
cd backend
npm run dev &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to initialize...${NC}"
sleep 5

# Check if backend is running
if ! curl -s http://localhost:3001/health > /dev/null; then
    echo -e "${RED}❌ Backend failed to start${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi
echo -e "${GREEN}✓ Backend is running on http://localhost:3001${NC}"

# Start the frontend
echo ""
echo -e "${BLUE}🚀 Starting Frontend...${NC}"
echo "================================================"
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
echo -e "${YELLOW}Waiting for frontend to initialize...${NC}"
sleep 5

# Display status
echo ""
echo -e "${GREEN}✅ System Started Successfully!${NC}"
echo "================================================"
echo ""
echo "📊 Dashboard: http://localhost:5173"
echo "🔧 Backend API: http://localhost:3001"
echo "📡 WebSocket: ws://localhost:3001"
echo ""
echo "📌 Useful Commands:"
echo "  • View logs: Check terminal output"
echo "  • Stop all: Press Ctrl+C"
echo "  • Emergency stop in app: Ctrl+Shift+E"
echo ""
echo -e "${YELLOW}💡 Tips for Live Trading:${NC}"
echo "  1. Start bot with conservative settings"
echo "  2. Monitor the Binance Account Summary widget"
echo "  3. Check Performance Metrics regularly"
echo "  4. Use stop-loss on all positions"
echo ""
echo -e "${GREEN}Happy Trading! 🚀${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $FRONTEND_PID 2>/dev/null
    kill $BACKEND_PID 2>/dev/null
    echo -e "${GREEN}✓ System stopped${NC}"
}

# Set up trap to cleanup on Ctrl+C
trap cleanup EXIT

# Wait for processes
wait $FRONTEND_PID
wait $BACKEND_PID