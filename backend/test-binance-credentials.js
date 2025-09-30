require('dotenv').config({ path: '../.env' });
const axios = require('axios');
const crypto = require('crypto');

const BINANCE_API_KEY = process.env.VITE_BINANCE_API_KEY;
const BINANCE_SECRET_KEY = process.env.VITE_BINANCE_SECRET_KEY;
const BINANCE_BASE_URL = process.env.VITE_BINANCE_BASE_URL || 'https://api.binance.com';

console.log('Testing Binance credentials...');
console.log('API Key:', BINANCE_API_KEY ? `${BINANCE_API_KEY.substring(0, 8)}...` : 'NOT SET');
console.log('Secret Key:', BINANCE_SECRET_KEY ? `${BINANCE_SECRET_KEY.substring(0, 8)}...` : 'NOT SET');
console.log('Base URL:', BINANCE_BASE_URL);

async function testBinanceConnection() {
  try {
    // Test 1: Simple public endpoint (no auth needed)
    console.log('\n1. Testing public endpoint...');
    const publicResponse = await axios.get(`${BINANCE_BASE_URL}/api/v3/ping`);
    console.log('✅ Public endpoint works:', publicResponse.data);

    // Test 2: Get server time
    console.log('\n2. Testing server time...');
    const timeResponse = await axios.get(`${BINANCE_BASE_URL}/api/v3/time`);
    console.log('✅ Server time:', new Date(timeResponse.data.serverTime));

    // Test 3: Test API key (without signature)
    console.log('\n3. Testing API key...');
    const keyTestResponse = await axios.get(`${BINANCE_BASE_URL}/api/v3/account`, {
      headers: {
        'X-MBX-APIKEY': BINANCE_API_KEY
      },
      params: {
        timestamp: Date.now()
      }
    });
    console.log('✅ API key works:', keyTestResponse.data);

  } catch (error) {
    console.error('❌ Error:', error.response?.status, error.response?.data || error.message);
    
    if (error.response?.status === 401) {
      console.log('\n💡 401 Error suggests:');
      console.log('   - API key is invalid or expired');
      console.log('   - API key doesn\'t have required permissions');
      console.log('   - Wrong Binance API endpoint (try https://api.binance.us for US)');
    }
  }
}

testBinanceConnection();
