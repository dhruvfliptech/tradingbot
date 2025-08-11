import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables. Please set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Database types
export interface Database {
  public: {
    Tables: {
      portfolios: {
        Row: {
          id: string;
          user_id: string;
          balance_usd: number;
          balance_btc: number;
          total_trades: number;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          balance_usd?: number;
          balance_btc?: number;
          total_trades?: number;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          balance_usd?: number;
          balance_btc?: number;
          total_trades?: number;
          created_at?: string;
          updated_at?: string;
        };
      };
      positions: {
        Row: {
          id: string;
          portfolio_id: string;
          symbol: string;
          name: string;
          image: string;
          quantity: number;
          avg_cost_basis: number;
          side: string;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          portfolio_id: string;
          symbol: string;
          name: string;
          image?: string;
          quantity: number;
          avg_cost_basis: number;
          side?: string;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          portfolio_id?: string;
          symbol?: string;
          name?: string;
          image?: string;
          quantity?: number;
          avg_cost_basis?: number;
          side?: string;
          created_at?: string;
          updated_at?: string;
        };
      };
      orders: {
        Row: {
          id: string;
          portfolio_id: string;
          symbol: string;
          quantity: number;
          side: string;
          order_type: string;
          status: string;
          filled_quantity: number;
          filled_avg_price: number;
          limit_price: number | null;
          submitted_at: string;
          filled_at: string | null;
        };
        Insert: {
          id?: string;
          portfolio_id: string;
          symbol: string;
          quantity: number;
          side: string;
          order_type?: string;
          status?: string;
          filled_quantity?: number;
          filled_avg_price?: number;
          limit_price?: number | null;
          submitted_at?: string;
          filled_at?: string | null;
        };
        Update: {
          id?: string;
          portfolio_id?: string;
          symbol?: string;
          quantity?: number;
          side?: string;
          order_type?: string;
          status?: string;
          filled_quantity?: number;
          filled_avg_price?: number;
          limit_price?: number | null;
          submitted_at?: string;
          filled_at?: string | null;
        };
      };
    };
  };
}