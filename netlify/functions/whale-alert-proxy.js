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
    const { start, end, min_value = 500000, limit = 100, endpoint = 'transactions' } = event.queryStringParameters || {};
    
    // Use default API key if none provided
    const apiKey = process.env.WHALE_ALERT_API_KEY || 'default_key';
    
    // Check if we have a valid API key
    if (!apiKey || apiKey === 'default_key') {
      console.warn('WhaleAlert API key not configured, using mock data');
      return {
        statusCode: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          result: 'success',
          count: 0,
          transactions: [],
          message: 'WhaleAlert API key not configured - using mock data'
        }),
      };
    }
    
    // Build WhaleAlert API URL based on endpoint
    const whaleAlertUrl = new URL(`https://api.whale-alert.io/v1/${endpoint}`);
    
    // Add parameters based on endpoint
    if (endpoint === 'transactions') {
      whaleAlertUrl.searchParams.set('start', start || Math.floor(Date.now() / 1000) - 86400);
      whaleAlertUrl.searchParams.set('end', end || Math.floor(Date.now() / 1000));
      whaleAlertUrl.searchParams.set('min_value', min_value);
      whaleAlertUrl.searchParams.set('limit', limit);
    }
    
    whaleAlertUrl.searchParams.set('api_key', apiKey);

    console.log(`Making request to WhaleAlert API: ${whaleAlertUrl.toString().replace(apiKey, '***')}`);

    // Make request to WhaleAlert API with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

    const response = await fetch(whaleAlertUrl.toString(), {
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`WhaleAlert API error: ${response.status} - ${errorText}`);
      throw new Error(`WhaleAlert API error: ${response.status} - ${errorText}`);
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
    
    // Return structured error response
    const errorResponse = {
      result: 'error',
      count: 0,
      transactions: [],
      message: `WhaleAlert proxy error: ${error.message}`,
      error: error.name || 'UnknownError'
    };
    
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(errorResponse),
    };
  }
};
