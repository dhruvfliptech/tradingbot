require('dotenv').config({ path: '../.env' });
const axios = require('axios');

const BINANCE_API_KEY = process.env.VITE_BINANCE_API_KEY;
const BINANCE_SECRET_KEY = process.env.VITE_BINANCE_SECRET_KEY;
const BINANCE_BASE_URL = process.env.VITE_BINANCE_BASE_URL || 'https://api.binance.com';

console.log('🌍 Testing from Vancouver, Canada');
console.log('📍 Your IP: 162.156.178.137');
console.log('🔑 API Key:', BINANCE_API_KEY ? `${BINANCE_API_KEY.substring(0, 12)}...` : 'NOT SET');
console.log('🌐 Base URL:', BINANCE_BASE_URL);
console.log('');

async function testBinanceAccess() {
  try {
    // Test 1: Check if Binance is accessible from Canada
    console.log('1️⃣ Testing Binance connectivity from Canada...');
    const pingResponse = await axios.get(`${BINANCE_BASE_URL}/api/v3/ping`, { timeout: 5000 });
    console.log('✅ Binance is accessible from Canada');
    
    // Test 2: Check server time
    console.log('\n2️⃣ Getting Binance server time...');
    const timeResponse = await axios.get(`${BINANCE_BASE_URL}/api/v3/time`, { timeout: 5000 });
    const serverTime = new Date(timeResponse.data.serverTime);
    const localTime = new Date();
    const timeDiff = Math.abs(serverTime - localTime);
    console.log(`✅ Server time: ${serverTime.toISOString()}`);
    console.log(`✅ Local time: ${localTime.toISOString()}`);
    console.log(`✅ Time difference: ${timeDiff}ms (${timeDiff < 5000 ? 'Good' : 'High latency'})`);
    
    // Test 3: Test API key without signature (should fail but give us info)
    console.log('\n3️⃣ Testing API key validity...');
    try {
      const keyTestResponse = await axios.get(`${BINANCE_BASE_URL}/api/v3/account`, {
        headers: {
          'X-MBX-APIKEY': BINANCE_API_KEY
        },
        params: {
          timestamp: Date.now()
        },
        timeout: 5000
      });
      console.log('✅ API key works without signature (unexpected!)');
    } catch (keyError) {
      if (keyError.response?.data?.code === -1102) {
        console.log('✅ API key is valid (signature required as expected)');
      } else if (keyError.response?.data?.code === -2015) {
        console.log('❌ API key issue: Invalid key, IP restrictions, or permissions');
        console.log('💡 Solutions:');
        console.log('   • Check if IP restrictions are enabled in Binance');
        console.log('   • Add 162.156.178.137 to your IP whitelist');
        console.log('   • Or disable IP restrictions');
        console.log('   • Verify API key has "Enable Reading" permission');
      } else {
        console.log('❌ API key error:', keyError.response?.data);
      }
    }
    
    console.log('\n🎯 Next steps:');
    console.log('1. Go to https://www.binance.com/en/my/settings/api-management');
    console.log('2. Find your API key and click "Edit"');
    console.log('3. Check "Restrict access to trusted IPs only"');
    console.log('4. Either:');
    console.log('   • Add IP: 162.156.178.137');
    console.log('   • OR uncheck the IP restriction option');
    console.log('5. Save and test again');
    
  } catch (error) {
    console.error('❌ Connection error:', error.message);
    if (error.code === 'ENOTFOUND') {
      console.log('💡 DNS resolution failed. Try using a VPN or check your internet connection.');
    }
  }
}

testBinanceAccess();
