const { Handler } = require('@netlify/functions');

exports.handler = async (event, context) => {
  // Handle CORS preflight requests
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      },
      body: '',
    };
  }

  try {
    const { start, end, min_value = 500000, limit = 100 } = event.queryStringParameters || {};
    
    // Use default API key if none provided
    const apiKey = process.env.WHALE_ALERT_API_KEY || 'default_key';
    
    // Build WhaleAlert API URL
    const whaleAlertUrl = new URL('https://api.whale-alert.io/v1/transactions');
    whaleAlertUrl.searchParams.set('start', start || Math.floor(Date.now() / 1000) - 86400);
    whaleAlertUrl.searchParams.set('end', end || Math.floor(Date.now() / 1000));
    whaleAlertUrl.searchParams.set('min_value', min_value);
    whaleAlertUrl.searchParams.set('limit', limit);
    whaleAlertUrl.searchParams.set('api_key', apiKey);

    // Make request to WhaleAlert API
    const response = await fetch(whaleAlertUrl.toString());
    
    if (!response.ok) {
      throw new Error(`WhaleAlert API error: ${response.status}`);
    }
    
    const data = await response.json();

    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    };
  } catch (error) {
    console.error('Error in whale-alert-proxy function:', error);
    
    // Return mock data on error
    const mockData = {
      count: 0,
      transactions: [],
      message: 'WhaleAlert proxy error - using mock data'
    };
    
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(mockData),
    };
  }
};
