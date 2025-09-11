/*
  # Fix RLS policies for demo trading

  1. Security Changes
    - Update RLS policies to work with demo users
    - Allow anonymous users to create and manage demo portfolios
    - Add proper policies for positions and orders tables

  2. Notes
    - This is for demo purposes only
    - In production, use proper Supabase Auth
*/

-- Drop existing policies
DROP POLICY IF EXISTS "Users can insert own portfolios" ON portfolios;
DROP POLICY IF EXISTS "Users can read own portfolios" ON portfolios;
DROP POLICY IF EXISTS "Users can update own portfolios" ON portfolios;

DROP POLICY IF EXISTS "Users can insert own positions" ON positions;
DROP POLICY IF EXISTS "Users can read own positions" ON positions;
DROP POLICY IF EXISTS "Users can update own positions" ON positions;
DROP POLICY IF EXISTS "Users can delete own positions" ON positions;

DROP POLICY IF EXISTS "Users can insert own orders" ON orders;
DROP POLICY IF EXISTS "Users can read own orders" ON orders;
DROP POLICY IF EXISTS "Users can update own orders" ON orders;

-- Create new policies that work with demo users
CREATE POLICY "Allow demo portfolio operations"
  ON portfolios
  FOR ALL
  TO anon, authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Allow demo position operations"
  ON positions
  FOR ALL
  TO anon, authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Allow demo order operations"
  ON orders
  FOR ALL
  TO anon, authenticated
  USING (true)
  WITH CHECK (true);