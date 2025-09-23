import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

// Check if we're in demo mode (placeholder URLs)
const isDemoMode = supabaseUrl?.includes('placeholder') || supabaseAnonKey?.includes('placeholder');

if ((!supabaseUrl || !supabaseAnonKey) && !isDemoMode) {
  throw new Error('Missing Supabase environment variables. Please set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY');
}

// Create a mock client for demo mode or real client for production
export const supabase = isDemoMode 
  ? createMockSupabaseClient()
  : createClient(supabaseUrl, supabaseAnonKey);

// Mock Supabase client for demo mode
function createMockSupabaseClient() {
  // Create chainable query builder
  const createQueryBuilder = (initialData: any = []) => {
    let data = initialData;
    let error = null;
    
    const builder = {
      select: (columns?: string) => {
        return createQueryBuilder(data);
      },
      insert: (values: any) => {
        return Promise.resolve({ data: values, error: null });
      },
      update: (values: any) => {
        return Promise.resolve({ data: values, error: null });
      },
      delete: () => {
        return Promise.resolve({ data: null, error: null });
      },
      upsert: (values: any) => {
        return Promise.resolve({ data: values, error: null });
      },
      eq: function(column: string, value: any) {
        return this;
      },
      neq: function(column: string, value: any) {
        return this;
      },
      gt: function(column: string, value: any) {
        return this;
      },
      gte: function(column: string, value: any) {
        return this;
      },
      lt: function(column: string, value: any) {
        return this;
      },
      lte: function(column: string, value: any) {
        return this;
      },
      like: function(column: string, value: any) {
        return this;
      },
      ilike: function(column: string, value: any) {
        return this;
      },
      is: function(column: string, value: any) {
        return this;
      },
      in: function(column: string, values: any[]) {
        return this;
      },
      contains: function(column: string, value: any) {
        return this;
      },
      containedBy: function(column: string, value: any) {
        return this;
      },
      limit: function(count: number) {
        return this;
      },
      order: function(column: string, options?: any) {
        return this;
      },
      range: function(from: number, to: number) {
        return this;
      },
      single: function() {
        // Return single item promise
        return Promise.resolve({ 
          data: {
            id: 'demo-portfolio',
            user_id: 'demo-user',
            balance_usd: 50000,
            balance_btc: 0,
            total_trades: 0,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          }, 
          error: null 
        });
      },
      maybeSingle: function() {
        return Promise.resolve({ data: data[0] || null, error: null });
      },
      then: function(resolve: any, reject: any) {
        return Promise.resolve({ data, error }).then(resolve, reject);
      }
    };
    
    return builder;
  };

  const mockClient = {
    from: (table: string) => createQueryBuilder(),
    auth: {
      signInWithPassword: () => Promise.resolve({ 
        data: { 
          user: { 
            id: 'demo-user', 
            email: 'demo@tradingbot.com',
            email_confirmed_at: new Date().toISOString(),
            created_at: new Date().toISOString()
          }, 
          session: { access_token: 'demo-token' } 
        }, 
        error: null 
      }),
      signUp: () => Promise.resolve({ 
        data: { 
          user: { 
            id: 'demo-user', 
            email: 'demo@tradingbot.com',
            email_confirmed_at: new Date().toISOString(),
            created_at: new Date().toISOString()
          }, 
          session: null 
        }, 
        error: null 
      }),
      signOut: () => Promise.resolve({ error: null }),
      getSession: () => Promise.resolve({ 
        data: { 
          session: { 
            access_token: 'demo-token',
            user: { 
              id: 'demo-user', 
              email: 'demo@tradingbot.com',
              email_confirmed_at: new Date().toISOString(),
              created_at: new Date().toISOString()
            }
          } 
        }, 
        error: null 
      }),
      getUser: () => Promise.resolve({ 
        data: { 
          user: { 
            id: 'demo-user', 
            email: 'demo@tradingbot.com',
            email_confirmed_at: new Date().toISOString(),
            created_at: new Date().toISOString()
          } 
        }, 
        error: null 
      }),
      onAuthStateChange: (callback: any) => {
        // Simulate logged in state for demo
        setTimeout(() => {
          callback('SIGNED_IN', { 
            access_token: 'demo-token',
            user: { 
              id: 'demo-user', 
              email: 'demo@tradingbot.com',
              email_confirmed_at: new Date().toISOString(),
              created_at: new Date().toISOString()
            }
          });
        }, 100);
        return { data: { subscription: { unsubscribe: () => {} } } };
      }
    },
    storage: {
      from: () => ({
        upload: () => Promise.resolve({ data: null, error: null }),
        download: () => Promise.resolve({ data: null, error: null }),
        remove: () => Promise.resolve({ data: null, error: null })
      })
    },
    channel: (name: string) => ({
      on: (event: string, filter: any, callback: any) => {
        // Mock realtime subscription
        return {
          subscribe: () => {
            console.log(`📡 Mock subscription to channel: ${name}`);
            return {
              unsubscribe: () => {}
            };
          }
        };
      },
      subscribe: (callback?: any) => {
        if (callback) callback('SUBSCRIBED');
        return {
          unsubscribe: () => {}
        };
      }
    }),
    removeChannel: (channel: any) => Promise.resolve({ error: null })
  };
  
  console.log('🎭 Running in demo mode - using mock Supabase client');
  return mockClient as any;
}

// Database types
export interface Database {
  public: {
    Tables: {
      api_keys: {
        Row: {
          id: string;
          user_id: string;
          provider: string;
          key_name: string;
          encrypted_value: string;
          is_active: boolean;
          last_validated_at: string | null;
          validation_status: 'pending' | 'valid' | 'invalid' | 'error';
          validation_error: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          provider: string;
          key_name: string;
          encrypted_value: string;
          is_active?: boolean;
          last_validated_at?: string | null;
          validation_status?: 'pending' | 'valid' | 'invalid' | 'error';
          validation_error?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          provider?: string;
          key_name?: string;
          encrypted_value?: string;
          is_active?: boolean;
          last_validated_at?: string | null;
          validation_status?: 'pending' | 'valid' | 'invalid' | 'error';
          validation_error?: string | null;
          created_at?: string;
          updated_at?: string;
        };
      };
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