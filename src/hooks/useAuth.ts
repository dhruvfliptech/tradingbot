import { useState, useEffect } from 'react';
import { User } from '@supabase/supabase-js';
import { supabase } from '../lib/supabase';

// Demo user for testing
const DEMO_USER: User = {
  id: 'demo-user-001',
  email: 'demo@tradingbot.local',
  app_metadata: {},
  user_metadata: {},
  aud: 'authenticated',
  created_at: new Date().toISOString(),
} as User;

export const useAuth = () => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if we're in demo mode
    const isDemoMode = import.meta.env.VITE_SUPABASE_URL === 'https://placeholder.supabase.co';
    
    if (isDemoMode) {
      // Auto-login with demo user in demo mode
      console.log('🎮 Demo mode activated - using demo user');
      setUser(DEMO_USER);
      setLoading(false);
      return;
    }

    // Get initial session for real Supabase
    const getSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setUser(session?.user ?? null);
      setLoading(false);
    };

    getSession();

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        setUser(session?.user ?? null);
        setLoading(false);
      }
    );

    return () => subscription.unsubscribe();
  }, []);

  const signOut = async () => {
    await supabase.auth.signOut();
  };

  return {
    user,
    loading,
    signOut,
  };
};