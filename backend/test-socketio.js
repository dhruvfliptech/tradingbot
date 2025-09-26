const io = require('socket.io-client');

// Connect to Socket.IO server
const socket = io('http://localhost:3001', {
  auth: {
    token: 'test-token' // You'll need a valid JWT token for authenticated features
  },
  transports: ['websocket', 'polling']
});

console.log('🔌 Attempting to connect to Socket.IO server...');

socket.on('connect', () => {
  console.log('✅ Connected to Socket.IO server');
  console.log('📝 Socket ID:', socket.id);

  // Emit connection_success event
  socket.emit('connection_success', {
    authenticated: false,
    userId: 'test-user'
  });

  // Subscribe to channels
  console.log('📊 Subscribing to market data channels...');
  socket.emit('subscribe', ['market:prices', 'price:BTC', 'price:ETH'], (response) => {
    console.log('📊 Subscription response:', response);
  });

  // Test getting positions
  console.log('📋 Getting positions...');
  socket.emit('get_positions', (response) => {
    console.log('📋 Positions response:', response);
  });

  // Test getting market data
  console.log('💹 Getting market data...');
  socket.emit('get_market_data', ['BTC', 'ETH'], (response) => {
    console.log('💹 Market data response:', response);
  });
});

socket.on('disconnect', (reason) => {
  console.log('❌ Disconnected from Socket.IO server:', reason);
});

socket.on('connect_error', (error) => {
  console.error('🚫 Connection error:', error.message);
});

// Listen for market data updates
socket.on('market_data', (data) => {
  console.log('📈 Market data update:', {
    timestamp: data.timestamp,
    priceCount: data.prices?.length || 0,
    firstPrice: data.prices?.[0]
  });
});

socket.on('price_update', (data) => {
  console.log('💰 Price update:', {
    symbol: data.symbol,
    price: data.price,
    change: data.change
  });
});

socket.on('order_update', (data) => {
  console.log('📝 Order update:', data);
});

socket.on('position_update', (data) => {
  console.log('💼 Position update:', data);
});

socket.on('alert', (data) => {
  console.log('🚨 Alert:', data);
});

socket.on('emergency_stop', (data) => {
  console.log('🛑 Emergency stop:', data);
});

// Keep the script running
console.log('🎧 Listening for Socket.IO events... Press Ctrl+C to exit');

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n👋 Closing Socket.IO connection...');
  socket.close();
  process.exit(0);
});