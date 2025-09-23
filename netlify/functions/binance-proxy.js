const CryptoJS = require('crypto-js');

exports.handler = async (event, context) => {
  // Handle CORS preflight requests
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-MBX-APIKEY',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      },
      body: '',
    };
  }

  try {
    const { httpMethod, path, body, queryStringParameters, headers } = event;
    
    // Extract the API path from the request
    const apiPath = path.replace('/.netlify/functions/binance-proxy', '');
    
    // Get API credentials from environment variables or request headers
    const apiKey = process.env.VITE_BINANCE_API_KEY || headers['x-mbx-apikey'];
    const secretKey = process.env.VITE_BINANCE_SECRET_KEY;
    
    console.log('API Key present:', !!apiKey);
    console.log('Secret Key present:', !!secretKey);
    console.log('API Key (first 10 chars):', apiKey ? apiKey.substring(0, 10) + '...' : 'none');
    console.log('Request method:', httpMethod);
    console.log('Request path:', apiPath);
    console.log('Query params:', queryStringParameters);
    console.log('Body:', body);
    console.log('Headers:', headers);
    
    if (!apiKey || !secretKey) {
      return {
        statusCode: 400,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ error: 'Binance API credentials not configured', apiKeyPresent: !!apiKey, secretKeyPresent: !!secretKey }),
      };
    }

    // Build the request URL
    const baseUrl = process.env.VITE_BINANCE_BASE_URL || 'https://api.binance.com';
    let url = `${baseUrl}${apiPath}`;
    
    // Handle query parameters - but exclude body parameters for POST requests
    let queryParams = new URLSearchParams();
    let bodyParams = new URLSearchParams();
    
    // Also handle body parameters if they exist
    if (body) {
      try {
        const bodyParamsFromBody = new URLSearchParams(body);
        for (const [key, value] of bodyParamsFromBody.entries()) {
          bodyParams.append(key, value);
        }
      } catch (error) {
        console.log('Could not parse body parameters:', error);
      }
    }
    
    if (queryStringParameters) {
      for (const [key, value] of Object.entries(queryStringParameters)) {
        if (httpMethod === 'POST') {
          // For POST requests, put parameters in body instead of query string
          bodyParams.append(key, value);
        } else {
          queryParams.append(key, value);
        }
      }
    }

    // Handle signed requests
    let requestHeaders = {
      'Content-Type': 'application/x-www-form-urlencoded',
      'X-MBX-APIKEY': apiKey,
    };

    // Add timestamp and signature for signed endpoints
    if (apiPath.includes('/api/v3/account') || 
        apiPath.includes('/api/v3/order') || 
        apiPath.includes('/api/v3/allOrders') ||
        apiPath.includes('/api/v3/myTrades')) {
      
      const timestamp = Date.now();
      const recvWindow = process.env.VITE_BINANCE_RECV_WINDOW || '5000';
      
      // Add timestamp and recvWindow to the appropriate params
      if (httpMethod === 'POST') {
        bodyParams.append('timestamp', timestamp.toString());
        bodyParams.append('recvWindow', recvWindow);
        console.log('Added timestamp and recvWindow to bodyParams for POST request');
      } else {
        queryParams.append('timestamp', timestamp.toString());
        queryParams.append('recvWindow', recvWindow);
        console.log('Added timestamp and recvWindow to queryParams for GET request');
      }
      
      // Create signature using the appropriate parameter set
      const signatureParams = httpMethod === 'POST' ? bodyParams : queryParams;
      const queryString = signatureParams.toString();
      const signature = CryptoJS.HmacSHA256(queryString, secretKey).toString(CryptoJS.enc.Hex);
      
      if (httpMethod === 'POST') {
        bodyParams.append('signature', signature);
      } else {
        queryParams.append('signature', signature);
      }
    }

    // Build final URL
    const finalQueryString = queryParams.toString();
    if (finalQueryString) {
      url += `?${finalQueryString}`;
    }

    // Prepare request options
    const requestOptions = {
      method: httpMethod,
      headers: requestHeaders,
    };

    // Add body for POST requests
    if (httpMethod === 'POST') {
      // Use bodyParams if we have them, otherwise use the original body
      const requestBody = bodyParams.toString() || body || '';
      console.log('Final bodyParams for POST:', bodyParams.toString());
      console.log('Final requestBody:', requestBody);
      if (requestBody) {
        requestOptions.body = requestBody;
      }
    }

    // Log the final request details
    console.log('Final Binance URL:', url);
    console.log('Final request options:', {
      method: requestOptions.method,
      headers: requestOptions.headers,
      body: requestOptions.body
    });

    // Make the request to Binance API
    const response = await fetch(url, requestOptions);
    
    // Log the response details
    console.log('Binance response status:', response.status);
    console.log('Binance response headers:', Object.fromEntries(response.headers.entries()));
    
    // Get response data
    let responseData;
    const contentType = response.headers.get('content-type');
    
    if (contentType && contentType.includes('application/json')) {
      responseData = await response.json();
    } else {
      responseData = await response.text();
    }
    
    // Log response data for debugging
    console.log('Binance response data:', responseData);

    // Return the response
    return {
      statusCode: response.status,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': contentType || 'application/json',
      },
      body: typeof responseData === 'string' ? responseData : JSON.stringify(responseData),
    };

  } catch (error) {
    console.error('Binance proxy error:', error);
    
    return {
      statusCode: 500,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        error: 'Internal server error', 
        message: error.message 
      }),
    };
  }
};
