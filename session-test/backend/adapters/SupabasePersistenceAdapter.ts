import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { PersistenceAdapter } from '../../core/adapters';

interface SupabasePersistenceOptions {
  url: string;
  serviceRoleKey: string;
  userId: string;
  stateTable?: string;
  auditLogTable?: string;
  tradeTable?: string;
}

export class SupabasePersistenceAdapter implements PersistenceAdapter {
  private client: SupabaseClient;
  private userId: string;
  private stateTable: string;
  private auditLogTable: string;
  private tradeTable: string;

  constructor(options: SupabasePersistenceOptions) {
    this.client = createClient(options.url, options.serviceRoleKey, {
      auth: { persistSession: false },
    });
    this.userId = options.userId;
    this.stateTable = options.stateTable ?? 'agent_state';
    this.auditLogTable = options.auditLogTable ?? 'audit_logs';
    this.tradeTable = options.tradeTable ?? 'trade_history';
  }

  async loadState<T = any>(key: string): Promise<T | null> {
    const { data, error } = await this.client
      .from(this.stateTable)
      .select('value')
      .eq('user_id', this.userId)
      .eq('key', key)
      .maybeSingle();

    if (error) {
      console.error('[SupabasePersistence] loadState failed', { key, error });
      return null;
    }

    return data?.value ?? null;
  }

  async saveState<T = any>(key: string, value: T): Promise<void> {
    const payload = {
      user_id: this.userId,
      key,
      value,
      updated_at: new Date().toISOString(),
    };

    const { error } = await this.client
      .from(this.stateTable)
      .upsert(payload, { onConflict: 'user_id,key' });

    if (error) {
      console.error('[SupabasePersistence] saveState failed', { key, error });
      throw error;
    }
  }

  async appendAuditLog(entry: Record<string, any>): Promise<void> {
    const payload = {
      ...entry,
      user_id: this.userId,
      created_at: new Date().toISOString(),
    };

    const { error } = await this.client.from(this.auditLogTable).insert(payload);

    if (error) {
      console.error('[SupabasePersistence] appendAuditLog failed', error);
    }
  }

  async recordTrade(trade: Record<string, any>): Promise<void> {
    const payload = {
      ...trade,
      user_id: this.userId,
      created_at: trade.created_at ?? new Date().toISOString(),
    };

    const { error } = await this.client.from(this.tradeTable).insert(payload);

    if (error) {
      console.error('[SupabasePersistence] recordTrade failed', error);
    }
  }
}
