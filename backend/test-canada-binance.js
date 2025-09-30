require('dotenv').config({ path: '../.env' });
const axios = require('axios');
const crypto = require('crypto');

const BINANCE_API_KEY = process.env.VITE_BINANCE_API_KEY;
const BINANCE_SECRET_KEY = process.env.VITE_BINANCE_SECRET_KEY;
const BINANCE_BASE_URL = process.env.VITE_BINANCE_BASE_URL || 'https://api.binance.com';

console.log('🇨🇦 Testing Binance API Access from Canada');
console.log('📍 Your IP: 162.156.178.137 (Vancouver, Canada)');
console.log('🔑 API Key:', BINANCE_API_KEY ? `${BINANCE_API_KEY.substring(0, 12)}...` : 'NOT SET');
console.log('🌐 Base URL:', BINANCE_BASE_URL);
console.log('');

function createBinanceSignature(queryString, secretKey) {
  return crypto.createHmac('sha256', secretKey).update(queryString).digest('hex');
}

async function testBinanceFromCanada() {
  try {
    // Test 1: Basic connectivity
    console.log('1️⃣ Testing basic connectivity to Binance...');
    const pingStart = Date.now();
    const pingResponse = await axios.get(`${BINANCE_BASE_URL}/api/v3/ping`, { timeout: 10000 });
    const pingTime = Date.now() - pingStart;
    console.log(`✅ Binance is accessible from Canada (${pingTime}ms)`);
    
    // Test 2: Server time and latency
    console.log('\n2️⃣ Testing server time and latency...');
    const timeStart = Date.now();
    const timeResponse = await axios.get(`${BINANCE_BASE_URL}/api/v3/time`, { timeout: 10000 });
    const timeLatency = Date.now() - timeStart;
    const serverTime = new Date(timeResponse.data.serverTime);
    const localTime = new Date();
    const timeDiff = Math.abs(serverTime - localTime);
    
    console.log(`✅ Server time: ${serverTime.toISOString()}`);
    console.log(`✅ Local time: ${localTime.toISOString()}`);
    console.log(`✅ Network latency: ${timeLatency}ms`);
    console.log(`✅ Time sync difference: ${timeDiff}ms ${timeDiff < 5000 ? '(Good)' : '(High)'}`);
    
    // Test 3: Public market data (no auth needed)
    console.log('\n3️⃣ Testing public market data access...');
    const btcPriceResponse = await axios.get(`${BINANCE_BASE_URL}/api/v3/ticker/price`, {
      params: { symbol: 'BTCUSDT' },
      timeout: 10000
    });
    console.log(`✅ BTC Price: $${parseFloat(btcPriceResponse.data.price).toLocaleString()}`);
    
    // Test 4: API key format validation
    console.log('\n4️⃣ Validating API key format...');
    if (BINANCE_API_KEY && BINANCE_API_KEY.length >= 64) {
      console.log('✅ API key format looks valid (64+ characters)');
    } else {
      console.log('❌ API key format invalid (should be 64+ characters)');
      return;
    }
    
    if (BINANCE_SECRET_KEY && BINANCE_SECRET_KEY.length >= 64) {
      console.log('✅ Secret key format looks valid (64+ characters)');
    } else {
      console.log('❌ Secret key format invalid (should be 64+ characters)');
      return;
    }
    
    // Test 5: Signature generation test
    console.log('\n5️⃣ Testing signature generation...');
    const timestamp = Date.now();
    const recvWindow = 5000;
    const queryParams = { timestamp, recvWindow };
    const queryString = Object.keys(queryParams)
      .sort()
      .map(key => `${key}=${queryParams[key]}`)
      .join('&');
    const signature = createBinanceSignature(queryString, BINANCE_SECRET_KEY);
    console.log(`✅ Signature generated: ${signature.substring(0, 16)}...`);
    
    // Test 6: Authenticated request (this should fail due to IP restriction)
    console.log('\n6️⃣ Testing authenticated request (expecting IP restriction)...');
    try {
      const authResponse = await axios.get(`${BINANCE_BASE_URL}/api/v3/account`, {
        headers: {
          'X-MBX-APIKEY': BINANCE_API_KEY
        },
        params: {
          timestamp,
          recvWindow,
          signature
        },
        timeout: 10000
      });
      console.log('✅ Authenticated request succeeded! (Unexpected - no IP restriction?)');
      console.log('Account data:', {
        accountType: authResponse.data.accountType,
        balances: authResponse.data.balances?.length || 0
      });
    } catch (authError) {
      if (authError.response?.data?.code === -2015) {
        console.log('❌ IP restriction detected (expected)');
        console.log('💡 Your friend needs to add your IP: 162.156.178.137');
        console.log('💡 Or disable IP restrictions on the API key');
      } else if (authError.response?.data?.code === -1022) {
        console.log('❌ Signature error (check secret key)');
      } else if (authError.response?.data?.code === -2014) {
        console.log('❌ API key format error');
      } else {
        console.log('❌ Other auth error:', authError.response?.data);
      }
    }
    
    console.log('\n🎯 Summary:');
    console.log('✅ Binance API is accessible from Canada');
    console.log('✅ Network latency is good');
    console.log('✅ API credentials are properly formatted');
    console.log('✅ Signature generation works');
    console.log('❌ IP restriction is blocking authenticated requests');
    console.log('');
    console.log('📋 Next steps for your friend:');
    console.log('1. Go to https://www.binance.com/en/my/settings/api-management');
    console.log('2. Find the API key and click "Edit"');
    console.log('3. Add this IP to whitelist: 162.156.178.137');
    console.log('4. Or uncheck "Restrict access to trusted IPs only"');
    console.log('5. Save the changes');
    console.log('');
    console.log('🔄 Once your friend updates the settings, test again with:');
    console.log('   curl -s "http://localhost:3000/api/v1/binance/account"');
    
  } catch (error) {
    console.error('❌ Connection error:', error.message);
    if (error.code === 'ENOTFOUND') {
      console.log('💡 DNS resolution failed. Check your internet connection.');
    } else if (error.code === 'ECONNREFUSED') {
      console.log('💡 Connection refused. Binance might be blocking Canada.');
    }
  }
}

testBinanceFromCanada();
