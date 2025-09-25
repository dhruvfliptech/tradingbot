import axios, { AxiosRequestConfig } from 'axios';
import CryptoJS from 'crypto-js';
import * as dotenv from 'dotenv';

// Load environment variables from parent directory
dotenv.config({ path: '../.env' });

async function testBinanceAccount() {
  // Try to load .env file directly
  const fs = require('fs');
  const pathModule = require('path');

  console.log('🔐 Current working directory:', process.cwd());
  console.log('🔐 Looking for .env file at:', pathModule.resolve('../.env'));

  try {
    const envPath = pathModule.resolve('../.env');
    if (fs.existsSync(envPath)) {
      console.log('✅ .env file found at:', envPath);
      const envContent = fs.readFileSync(envPath, 'utf8');
      console.log('🔐 .env content preview:', envContent.substring(0, 200) + '...');

      // Parse environment variables manually
      const lines = envContent.split('\n');
      for (const line of lines) {
        if (line.startsWith('VITE_BINANCE_API_KEY=')) {
          const apiKey = line.split('=')[1];
          console.log('✅ Found API Key:', apiKey ? 'Loaded' : 'Empty');
          process.env.BINANCE_API_KEY = apiKey;
        }
        if (line.startsWith('VITE_BINANCE_SECRET_KEY=')) {
          const secretKey = line.split('=')[1];
          console.log('✅ Found Secret Key:', secretKey ? 'Loaded' : 'Empty');
          process.env.BINANCE_SECRET_KEY = secretKey;
        }
      }
    } else {
      console.error('❌ .env file not found at:', envPath);
    }
  } catch (error) {
    console.error('❌ Error reading .env file:', error);
  }

  const apiKey = process.env.BINANCE_API_KEY;
  const secretKey = process.env.BINANCE_SECRET_KEY;

  console.log('🔐 Testing Binance API keys:');
  console.log('API Key:', apiKey ? '✅ Loaded' : '❌ Missing');
  console.log('Secret Key:', secretKey ? '✅ Loaded' : '❌ Missing');

  if (!apiKey || !secretKey) {
    console.error('❌ API keys not found in environment variables');
    return;
  }

  const baseUrl = 'https://api.binance.com';
  const path = '/api/v3/account';

  // Create parameters for signature
  const params = {
    timestamp: Date.now(),
    recvWindow: 5000
  };

  // Create query string
  const queryString = Object.keys(params)
    .sort()
    .map(key => `${key}=${(params as any)[key]}`)
    .join('&');

  // Create signature
  const signature = CryptoJS.HmacSHA256(queryString, secretKey).toString(CryptoJS.enc.Hex);
  const fullQueryString = `${queryString}&signature=${signature}`;

  const config: AxiosRequestConfig = {
    method: 'GET',
    url: `${baseUrl}${path}?${fullQueryString}`,
    headers: {
      'X-MBX-APIKEY': apiKey,
      'User-Agent': 'TradingBot/1.0',
    },
    timeout: 10000,
  };

  console.log('🔐 Making signed request to:', config.url);

  try {
    const response = await axios(config);
    console.log('✅ Binance account response:', JSON.stringify(response.data, null, 2));
  } catch (error: any) {
    console.error('❌ Binance API error:', error.response?.data || error.message);
    console.error('Status:', error.response?.status);
    console.error('Status Text:', error.response?.statusText);
  }
}

testBinanceAccount();
