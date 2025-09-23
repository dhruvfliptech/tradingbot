import { supabase } from './supabase';

let cachedUserId: string | null = null;

export async function requireUserId(): Promise<string> {
  if (cachedUserId) {
    return cachedUserId;
  }

  const { data, error } = await supabase.auth.getUser();
  if (error) {
    throw new Error(`Failed to retrieve Supabase user: ${error.message}`);
  }

  const user = data.user;
  if (!user) {
    throw new Error('Supabase user not authenticated');
  }

  cachedUserId = user.id;
  return cachedUserId;
}

export function clearCachedUserId(): void {
  cachedUserId = null;
}
