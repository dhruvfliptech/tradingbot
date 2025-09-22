# Supabase Database Setup

The application requires several database tables that need to be created in your Supabase project. Follow these steps:

## 1. Access Supabase Dashboard

1. Go to [supabase.com](https://supabase.com)
2. Sign in to your account
3. Select your project: `ewezuuywerilgnhhzzho`

## 2. Run SQL Migrations

Go to the SQL Editor in your Supabase dashboard and run these migrations in order:

### Step 1: API Keys Table
```sql
-- Copy and paste the contents of: supabase/migrations/20250815000000_api_keys.sql
```

### Step 2: Trading Persistence Tables
```sql
-- Copy and paste the contents of: supabase/migrations/001_trading_persistence.sql
```

### Step 3: Performance Tracking Tables
```sql
-- Copy and paste the contents of: src/migrations/supabase-performance-tables.sql
```

## 3. Verify Tables Created

After running the migrations, you should see these tables in your Supabase dashboard:
- `api_keys`
- `trade_history`
- `audit_logs`
- `user_settings`
- `virtual_portfolios`
- `daily_performance`
- `strategy_performance`
- `trade_attribution`
- `performance_snapshots`
- `position_rules`
- `dca_orders`
- `correlation_data`
- `news_sentiment`

## 4. Environment Variables

Create a `.env` file in your project root with:

```env
# Supabase Configuration
VITE_SUPABASE_URL=https://ewezuuywerilgnhhzzho.supabase.co
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key_here

# Trading APIs
VITE_ALPACA_API_KEY=your_alpaca_api_key_here
VITE_ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# AI Services
VITE_GROQ_API_KEY=your_groq_api_key_here

# Optional APIs
VITE_COINGECKO_API_KEY=your_coingecko_api_key_here
VITE_WHALE_ALERT_API_KEY=your_whale_alert_api_key_here
VITE_ETHERSCAN_API_KEY=your_etherscan_api_key_here
VITE_BITQUERY_API_KEY=your_bitquery_api_key_here
```

## 5. Get Your Supabase Keys

1. In your Supabase dashboard, go to Settings > API
2. Copy the "Project URL" and "anon public" key
3. Replace the placeholder values in your `.env` file

## 6. Restart Development Server

After setting up the database and environment variables:

```bash
npm run dev
```

The application should now work without the 404 errors for missing tables.
