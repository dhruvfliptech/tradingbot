/*
  # Update authentication and RLS policies

  1. Security Updates
    - Remove demo policies that allowed anonymous access
    - Add proper RLS policies for authenticated users only
    - Ensure users can only access their own data

  2. Policy Changes
    - Portfolios: Users can only manage their own portfolios
    - Positions: Users can only see positions in their portfolios
    - Orders: Users can only see their own orders
*/

-- Drop existing demo policies
DROP POLICY IF EXISTS "Allow demo portfolio operations" ON portfolios;
DROP POLICY IF EXISTS "Allow demo position operations" ON positions;
DROP POLICY IF EXISTS "Allow demo order operations" ON orders;

-- Create proper RLS policies for authenticated users
CREATE POLICY "Users can manage own portfolios"
  ON portfolios
  FOR ALL
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage positions in own portfolios"
  ON positions
  FOR ALL
  TO authenticated
  USING (
    portfolio_id IN (
      SELECT id FROM portfolios WHERE user_id = auth.uid()
    )
  )
  WITH CHECK (
    portfolio_id IN (
      SELECT id FROM portfolios WHERE user_id = auth.uid()
    )
  );

CREATE POLICY "Users can manage orders in own portfolios"
  ON orders
  FOR ALL
  TO authenticated
  USING (
    portfolio_id IN (
      SELECT id FROM portfolios WHERE user_id = auth.uid()
    )
  )
  WITH CHECK (
    portfolio_id IN (
      SELECT id FROM portfolios WHERE user_id = auth.uid()
    )
  );