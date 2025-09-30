require('dotenv').config({ path: '../.env' });
const axios = require('axios');
const crypto = require('crypto');

const BINANCE_API_KEY = process.env.VITE_BINANCE_API_KEY;
const BINANCE_SECRET_KEY = process.env.VITE_BINANCE_SECRET_KEY;
const BINANCE_BASE_URL = process.env.VITE_BINANCE_BASE_URL || 'https://api.binance.com';

function createBinanceSignature(queryString, secretKey) {
  return crypto.createHmac('sha256', secretKey).update(queryString).digest('hex');
}

async function testSignedRequest() {
  try {
    const timestamp = Date.now();
    const recvWindow = 5000;
    
    const queryParams = {
      timestamp,
      recvWindow
    };

    const queryString = Object.keys(queryParams)
      .sort()
      .map(key => `${key}=${queryParams[key]}`)
      .join('&');

    const signature = createBinanceSignature(queryString, BINANCE_SECRET_KEY);
    const url = `${BINANCE_BASE_URL}/api/v3/account?${queryString}&signature=${signature}`;

    console.log('Query string:', queryString);
    console.log('Signature:', signature);
    console.log('Full URL:', url);

    const response = await axios.get(url, {
      headers: {
        'X-MBX-APIKEY': BINANCE_API_KEY,
        'User-Agent': 'TradingBot/1.0'
      },
      timeout: 10000
    });

    console.log('✅ Success! Account data:', {
      accountType: response.data.accountType,
      balances: response.data.balances?.length || 0,
      totalTradeCount: response.data.totalTradeCount
    });

  } catch (error) {
    console.error('❌ Error:', error.response?.status, error.response?.data || error.message);
  }
}

testSignedRequest();
