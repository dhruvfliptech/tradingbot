const { Handler } = require('@netlify/functions');

exports.handler = async (event, context) => {
  // Handle CORS preflight requests
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
      },
      body: '',
    };
  }

  // Only allow POST requests
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers: {
        'Access-Control-Allow-Origin': '*',
      },
      body: JSON.stringify({ error: 'Method not allowed' }),
    };
  }

  try {
    const { symbols = [], limit = 5 } = JSON.parse(event.body || '{}');
    
    // For now, return mock data since we don't have news API keys configured
    // In production, this would integrate with news APIs like NewsAPI, Tavily, etc.
    const mockInsights = symbols.slice(0, limit).map((symbol, index) => ({
      id: `insight-${symbol}-${index}`,
      title: `${symbol} Market Analysis`,
      summary: `Recent market activity for ${symbol} shows mixed signals. Technical indicators suggest potential volatility ahead.`,
      sentiment: ['bullish', 'bearish', 'neutral'][index % 3],
      relevantCoins: [symbol],
      confidence: 0.7 + (index * 0.1),
      timestamp: new Date().toISOString(),
      sources: [
        {
          title: `Crypto News - ${symbol}`,
          url: `https://example.com/news/${symbol.toLowerCase()}`
        }
      ]
    }));

    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        insights: mockInsights,
        message: 'Mock news insights - configure news API for real data'
      }),
    };
  } catch (error) {
    console.error('Error in search-news function:', error);
    
    return {
      statusCode: 500,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        error: 'Internal server error',
        insights: []
      }),
    };
  }
};
