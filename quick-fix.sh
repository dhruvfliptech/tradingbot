#!/bin/bash

echo "🔧 Quick Fix Script - Getting the app running"
echo "============================================"

# Kill any running processes
echo "Stopping existing processes..."
pkill -f "ts-node-dev" 2>/dev/null
pkill -f "vite" 2>/dev/null
sleep 2

# Start frontend
echo "Starting frontend..."
npm run dev > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!

# Start a minimal backend server for testing
echo "Creating minimal backend server..."
cat > backend/simple-server.js << 'EOF'
const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json());

// Health endpoint
app.get('/health', (req, res) => {
  res.json({
    success: true,
    message: 'Backend is running',
    timestamp: new Date()
  });
});

// Minimal broker account endpoint
app.get('/api/v1/brokers/account', (req, res) => {
  res.json({
    success: true,
    data: {
      broker: 'binance',
      cashBalance: 10000,
      buyingPower: 10000,
      balances: {
        USDT: { free: "10000", locked: "0" },
        BTC: { free: "0.5", locked: "0" },
        ETH: { free: "2", locked: "0" }
      }
    }
  });
});

// Market data endpoint
app.post('/api/v1/brokers/market-data', (req, res) => {
  res.json({
    success: true,
    data: [
      { symbol: 'BTCUSDT', price: 65000 }
    ]
  });
});

const PORT = 3001;
app.listen(PORT, () => {
  console.log(`Simple backend running on http://localhost:${PORT}`);
});
EOF

# Start simple backend
echo "Starting simple backend..."
cd backend && node simple-server.js > /tmp/simple-backend.log 2>&1 &
BACKEND_PID=$!
cd ..

sleep 3

echo ""
echo "✅ Services Started!"
echo "===================="
echo "Frontend: http://localhost:5173"
echo "Backend: http://localhost:3001"
echo ""
echo "This is a minimal setup to test the frontend."
echo "The backend is running with mock data."
echo ""
echo "To stop: Press Ctrl+C"

# Cleanup function
cleanup() {
  echo "Stopping services..."
  kill $FRONTEND_PID 2>/dev/null
  kill $BACKEND_PID 2>/dev/null
  rm -f backend/simple-server.js
  echo "Stopped"
}

trap cleanup EXIT

wait