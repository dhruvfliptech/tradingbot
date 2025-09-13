import { supabase } from '../../lib/supabase';

export type EventType = 
  | 'ai_decision' 
  | 'user_override' 
  | 'setting_change' 
  | 'trade_execution' 
  | 'agent_pause'
  | 'agent_resume'
  | 'portfolio_reset'
  | 'api_error'
  | 'system_alert';

export type EventCategory = 'trading' | 'risk' | 'settings' | 'system';

export interface AIReasoning {
  signals?: {
    technical?: {
      rsi?: number;
      macd?: string;
      ma_trend?: string;
      bollinger?: string;
    };
    momentum?: {
      volume?: number;
      change_24h?: number;
      volume_trend?: string;
    };
    sentiment?: {
      fear_greed?: number;
      whale_activity?: string;
      news_sentiment?: string;
    };
    funding?: {
      rate?: number;
      trend?: string;
    };
  };
  risk_assessment?: {
    volatility?: string;
    drawdown_risk?: number;
    position_correlation?: number;
    portfolio_exposure?: number;
  };
  decision_factors?: Array<{
    factor: string;
    weight: number;
    contribution: number;
    reasoning?: string;
  }>;
  final_decision?: string;
  confidence?: number;
  expected_risk_reward?: number;
}

export interface AuditLog {
  id?: string;
  user_id?: string;
  event_type: EventType;
  event_category?: EventCategory;
  symbol?: string;
  action?: 'buy' | 'sell' | 'hold' | 'skip' | 'short' | 'cover';
  confidence_score?: number;
  ai_reasoning?: AIReasoning;
  old_value?: any;
  new_value?: any;
  user_reason?: string;
  market_conditions?: {
    btc_price?: number;
    eth_price?: number;
    total_market_cap?: number;
    market_trend?: string;
    volatility_index?: number;
  };
  portfolio_state?: {
    total_value?: number;
    cash_balance?: number;
    positions_count?: number;
    daily_pnl?: number;
    total_pnl?: number;
  };
  created_at?: string;
  session_id?: string;
  ip_address?: string;
}

export interface AuditLogFilters {
  event_type?: EventType;
  event_category?: EventCategory;
  symbol?: string;
  startDate?: Date;
  endDate?: Date;
  limit?: number;
}

class AuditLogService {
  private readonly TABLE_NAME = 'audit_logs';
  private sessionId: string;
  private buffer: AuditLog[] = [];
  private flushInterval: number | null = null;
  private readonly BUFFER_SIZE = 10;
  private readonly FLUSH_INTERVAL = 5000; // 5 seconds

  constructor() {
    this.sessionId = this.generateSessionId();
    this.startBufferFlush();
  }

  /**
   * Log an AI trading decision
   */
  async logAIDecision(
    symbol: string,
    action: 'buy' | 'sell' | 'hold' | 'skip',
    confidence: number,
    reasoning: AIReasoning,
    marketConditions?: any,
    portfolioState?: any
  ): Promise<void> {
    const log: AuditLog = {
      event_type: 'ai_decision',
      event_category: 'trading',
      symbol,
      action,
      confidence_score: confidence,
      ai_reasoning: reasoning,
      market_conditions: marketConditions,
      portfolio_state: portfolioState,
      session_id: this.sessionId
    };

    await this.addToBuffer(log);
  }

  /**
   * Log a user override of AI decision
   */
  async logUserOverride(
    symbol: string,
    aiAction: string,
    userAction: string,
    reason?: string,
    aiConfidence?: number
  ): Promise<void> {
    const log: AuditLog = {
      event_type: 'user_override',
      event_category: 'trading',
      symbol,
      old_value: { action: aiAction, confidence: aiConfidence },
      new_value: { action: userAction },
      user_reason: reason,
      session_id: this.sessionId
    };

    await this.addToBuffer(log);
  }

  /**
   * Log a settings change
   */
  async logSettingChange(
    settingName: string,
    oldValue: any,
    newValue: any,
    reason?: string
  ): Promise<void> {
    const log: AuditLog = {
      event_type: 'setting_change',
      event_category: 'settings',
      old_value: { [settingName]: oldValue },
      new_value: { [settingName]: newValue },
      user_reason: reason,
      session_id: this.sessionId
    };

    await this.addToBuffer(log);
  }

  /**
   * Log trade execution
   */
  async logTradeExecution(
    symbol: string,
    action: 'buy' | 'sell' | 'short' | 'cover',
    quantity: number,
    price: number,
    orderId?: string,
    reasoning?: AIReasoning
  ): Promise<void> {
    const log: AuditLog = {
      event_type: 'trade_execution',
      event_category: 'trading',
      symbol,
      action,
      new_value: {
        quantity,
        price,
        order_id: orderId,
        total_value: quantity * price
      },
      ai_reasoning: reasoning,
      session_id: this.sessionId
    };

    await this.addToBuffer(log);
  }

  /**
   * Log agent pause/resume
   */
  async logAgentControl(
    action: 'pause' | 'resume',
    reason?: string,
    additionalInfo?: any
  ): Promise<void> {
    const log: AuditLog = {
      event_type: action === 'pause' ? 'agent_pause' : 'agent_resume',
      event_category: 'system',
      user_reason: reason,
      new_value: additionalInfo,
      session_id: this.sessionId
    };

    await this.addToBuffer(log);
  }

  /**
   * Log system alerts and errors
   */
  async logSystemAlert(
    alertType: string,
    message: string,
    details?: any
  ): Promise<void> {
    const log: AuditLog = {
      event_type: 'system_alert',
      event_category: 'system',
      new_value: {
        alert_type: alertType,
        message,
        details
      },
      session_id: this.sessionId
    };

    await this.addToBuffer(log);
  }

  /**
   * Get audit logs with filters
   */
  async getAuditLogs(filters?: AuditLogFilters): Promise<AuditLog[]> {
    try {
      // Flush buffer before querying
      await this.flushBuffer();

      let query = supabase
        .from(this.TABLE_NAME)
        .select('*')
        .order('created_at', { ascending: false });

      if (filters?.event_type) {
        query = query.eq('event_type', filters.event_type);
      }
      if (filters?.event_category) {
        query = query.eq('event_category', filters.event_category);
      }
      if (filters?.symbol) {
        query = query.eq('symbol', filters.symbol);
      }
      if (filters?.startDate) {
        query = query.gte('created_at', filters.startDate.toISOString());
      }
      if (filters?.endDate) {
        query = query.lte('created_at', filters.endDate.toISOString());
      }
      if (filters?.limit) {
        query = query.limit(filters.limit);
      } else {
        query = query.limit(100); // Default limit
      }

      const { data, error } = await query;

      if (error) {
        console.error('Error fetching audit logs:', error);
        return [];
      }

      return data || [];
    } catch (error) {
      console.error('Failed to fetch audit logs:', error);
      return [];
    }
  }

  /**
   * Get decision history for a specific symbol
   */
  async getSymbolDecisionHistory(symbol: string, limit: number = 50): Promise<AuditLog[]> {
    return this.getAuditLogs({
      symbol,
      event_type: 'ai_decision',
      limit
    });
  }

  /**
   * Get user intervention history
   */
  async getUserInterventions(limit: number = 50): Promise<AuditLog[]> {
    try {
      const { data, error } = await supabase
        .from(this.TABLE_NAME)
        .select('*')
        .in('event_type', ['user_override', 'agent_pause', 'setting_change'])
        .order('created_at', { ascending: false })
        .limit(limit);

      if (error) {
        console.error('Error fetching user interventions:', error);
        return [];
      }

      return data || [];
    } catch (error) {
      console.error('Failed to fetch user interventions:', error);
      return [];
    }
  }

  /**
   * Calculate user impact on AI performance
   */
  async calculateUserImpact(startDate: Date, endDate: Date): Promise<{
    totalInterventions: number;
    overrideCount: number;
    pauseCount: number;
    settingChanges: number;
    impactOnReturns: number;
  }> {
    try {
      const logs = await this.getAuditLogs({
        startDate,
        endDate
      });

      const interventions = logs.filter(l => 
        ['user_override', 'agent_pause', 'setting_change'].includes(l.event_type)
      );

      const overrides = interventions.filter(l => l.event_type === 'user_override');
      const pauses = interventions.filter(l => l.event_type === 'agent_pause');
      const settings = interventions.filter(l => l.event_type === 'setting_change');

      // TODO: Calculate actual impact on returns by comparing with shadow portfolio
      const impactOnReturns = 0; // Placeholder

      return {
        totalInterventions: interventions.length,
        overrideCount: overrides.length,
        pauseCount: pauses.length,
        settingChanges: settings.length,
        impactOnReturns
      };
    } catch (error) {
      console.error('Failed to calculate user impact:', error);
      return {
        totalInterventions: 0,
        overrideCount: 0,
        pauseCount: 0,
        settingChanges: 0,
        impactOnReturns: 0
      };
    }
  }

  /**
   * Export audit logs to CSV
   */
  async exportToCSV(filters?: AuditLogFilters): Promise<string> {
    const logs = await this.getAuditLogs(filters);
    
    if (logs.length === 0) {
      return '';
    }

    const headers = [
      'Timestamp',
      'Event Type',
      'Category',
      'Symbol',
      'Action',
      'Confidence',
      'User Reason',
      'Session ID'
    ];

    const rows = logs.map(l => [
      new Date(l.created_at || '').toLocaleString(),
      l.event_type,
      l.event_category || '',
      l.symbol || '',
      l.action || '',
      l.confidence_score?.toString() || '',
      l.user_reason || '',
      l.session_id || ''
    ]);

    const csv = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');

    return csv;
  }

  /**
   * Subscribe to real-time audit log updates
   */
  subscribeToAuditLogs(callback: (log: AuditLog) => void) {
    const subscription = supabase
      .channel('audit_log_updates')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: this.TABLE_NAME
        },
        (payload) => {
          callback(payload.new as AuditLog);
        }
      )
      .subscribe();

    return () => {
      subscription.unsubscribe();
    };
  }

  // Private helper methods
  private async addToBuffer(log: AuditLog): Promise<void> {
    this.buffer.push(log);
    
    if (this.buffer.length >= this.BUFFER_SIZE) {
      await this.flushBuffer();
    }
  }

  private async flushBuffer(): Promise<void> {
    if (this.buffer.length === 0) return;

    const logsToFlush = [...this.buffer];
    this.buffer = [];

    try {
      const { error } = await supabase
        .from(this.TABLE_NAME)
        .insert(logsToFlush);

      if (error) {
        console.error('Error flushing audit logs:', error);
        // Re-add to buffer on error
        this.buffer.unshift(...logsToFlush);
      }
    } catch (error) {
      console.error('Failed to flush audit logs:', error);
      // Re-add to buffer on error
      this.buffer.unshift(...logsToFlush);
    }
  }

  private startBufferFlush(): void {
    this.flushInterval = window.setInterval(() => {
      this.flushBuffer();
    }, this.FLUSH_INTERVAL);
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Clean up on unmount
   */
  destroy(): void {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    this.flushBuffer();
  }
}

// Export singleton instance
export const auditLogService = new AuditLogService();

// Clean up on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    auditLogService.destroy();
  });
}