# WhaleAlert API Setup

## Issue Description
The WhaleAlert API is returning HTML instead of JSON, causing the OnChainMetrics component to fail. This typically happens when:

1. **Netlify function not deployed**: The proxy function isn't available
2. **Missing API key**: No valid WhaleAlert API key configured
3. **CORS issues**: Direct API calls blocked by browser

## Quick Fix ✅

The service now includes robust fallback mechanisms and will automatically use mock data when the API fails. 

**API Key Configured**: The provided WhaleAlert API key has been configured in the service.

**Development Mode**: In development, the service now attempts direct API calls first, then falls back to mock data if CORS issues occur.

## Environment Variables

### For Local Development
Create a `.env.local` file in the project root:

```bash
# WhaleAlert API Key (get from https://whale-alert.io/)
WHALE_ALERT_API_KEY=your_api_key_here

# Other required keys
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_ALPACA_API_KEY=your_alpaca_key
VITE_ALPACA_SECRET_KEY=your_alpaca_secret
VITE_GROQ_API_KEY=your_groq_key
```

### For Netlify Deployment
Add environment variables in Netlify dashboard:
- Go to Site Settings → Environment Variables
- Add `WHALE_ALERT_API_KEY` with your API key

## Testing the Connection

Open browser console and run:

```javascript
// Test direct API connection (recommended for development)
await testWhaleAlert.testDirect()

// Test proxy connection (for production)
await testWhaleAlert.testProxy()

// Test both connections
await testWhaleAlert.testBoth()

// Get real data (should work now with configured API key)
await whaleAlertService.getLargeTransactions(500000)

// Get mock data (fallback)
await testWhaleAlert.getMockData()
```

**Expected Results**: With the configured API key, you should now see real whale transaction data instead of mock data.

## Getting a WhaleAlert API Key

1. Visit [whale-alert.io](https://whale-alert.io/)
2. Sign up for an account
3. Go to API section
4. Generate an API key
5. Add it to your environment variables

## Fallback Behavior

When the API fails, the service automatically:
- Logs detailed error information
- Returns realistic mock data
- Continues to function normally
- Shows appropriate error messages in console

## Debugging Steps

1. **Check console logs** for detailed error messages
2. **Test proxy function** using browser console commands
3. **Verify API key** is correctly configured
4. **Check Netlify deployment** if using hosted version
5. **Use mock data** as temporary solution

The application will continue to work with mock data while you resolve the API configuration.
