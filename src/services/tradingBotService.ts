/**
 * Trading Bot Service - Frontend API client for controlling backend trading bot
 * Replaces direct trading execution with API calls to backend
 */

import { io, Socket } from 'socket.io-client';
import { supabase } from '../lib/supabase';

export type BotStatus = 'not_initialized' | 'active' | 'stopped' | 'error';

export interface BotConfig {
  watchlist: string[];
  cycleIntervalMs: number;
  cooldownMinutes: number;
  maxOpenPositions: number;
  riskBudgetUsd: number;
  confidenceThreshold: number;
  settings: {
    stopLossPercent?: number;
    takeProfitPercent?: number;
    emergencyCloseAll?: boolean;
    enabledStrategies?: string[];
  };
}

export interface BotState {
  status: BotStatus;
  isActive: boolean;
  cyclesExecuted: number;
  signalsGenerated: number;
  tradesExecuted: number;
  errors: number;
  lastError?: string;
  positions: number;
  account?: any;
  config?: BotConfig;
}

export interface TradingSignal {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  timestamp: string;
  reasoning?: string;
}

export interface ManualOrder {
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  orderType: 'market' | 'limit';
  limitPrice?: number;
}

export type TradingEvent =
  | { type: 'status'; data: { active: boolean; account?: any; positions?: number }; timestamp: string }
  | { type: 'analysis'; data: { evaluated: number; signals: number; topSignals: any[] }; timestamp: string }
  | { type: 'decision'; data: { symbol: string; action: string; confidence: number; price: number; quantity: number; reason: string }; timestamp: string }
  | { type: 'order_submitted'; data: { symbol: string; side: string; quantity: number; orderId: string; price: number }; timestamp: string }
  | { type: 'order_error'; data: { symbol: string; message: string }; timestamp: string }
  | { type: 'cycle_complete'; data: { duration: number; signalsGenerated: number; cyclesExecuted: number }; timestamp: string }
  | { type: 'emergency_stop'; data: { reason: string; positionsClosed: boolean }; timestamp: string }
  | { type: 'error'; data: { message: string; cycle: number }; timestamp: string };

class TradingBotService {
  private apiUrl: string;
  private socket: Socket | null = null;
  private eventHandlers: Map<string, Set<(event: TradingEvent) => void>> = new Map();
  private currentState: BotState = {
    status: 'not_initialized',
    isActive: false,
    cyclesExecuted: 0,
    signalsGenerated: 0,
    tradesExecuted: 0,
    errors: 0,
    positions: 0
  };

  constructor() {
    this.apiUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3001';
    this.initializeSocket();
  }

  /**
   * Initialize Socket.IO connection for real-time updates
   */
  private async initializeSocket(): Promise<void> {
    try {
      // Get auth token
      const { data: { session } } = await supabase.auth.getSession();
      const token = session?.access_token;

      // Connect to Socket.IO
      this.socket = io(this.apiUrl, {
        auth: { token },
        transports: ['websocket', 'polling']
      });

      // Listen for trading events
      this.socket.on('trading_event', (event: TradingEvent) => {
        this.handleTradingEvent(event);
      });

      this.socket.on('bot_status_update', (state: BotState) => {
        this.currentState = state;
      });

    } catch (error) {
      console.error('Failed to initialize Socket.IO:', error);
    }
  }

  /**
   * Handle incoming trading events
   */
  private handleTradingEvent(event: TradingEvent): void {
    // Update internal state based on event
    switch (event.type) {
      case 'status':
        this.currentState.isActive = event.data.active;
        this.currentState.status = event.data.active ? 'active' : 'stopped';
        break;
      case 'cycle_complete':
        this.currentState.cyclesExecuted = event.data.cyclesExecuted;
        this.currentState.signalsGenerated += event.data.signalsGenerated;
        break;
      case 'order_submitted':
        this.currentState.tradesExecuted++;
        break;
      case 'error':
        this.currentState.errors++;
        this.currentState.lastError = event.data.message;
        break;
    }

    // Notify all registered handlers
    const handlers = this.eventHandlers.get(event.type) || new Set();
    handlers.forEach(handler => handler(event));

    // Also notify generic handlers
    const allHandlers = this.eventHandlers.get('*') || new Set();
    allHandlers.forEach(handler => handler(event));
  }

  /**
   * Start the trading bot
   */
  async start(config?: Partial<BotConfig>): Promise<void> {
    try {
      const response = await this.apiCall('/api/v1/trading/bot/start', 'POST', config || {});

      if (response.success) {
        this.currentState.status = 'active';
        this.currentState.isActive = true;
        this.currentState.config = response.data.config;
      } else {
        throw new Error(response.error || 'Failed to start bot');
      }
    } catch (error) {
      console.error('Error starting bot:', error);
      throw error;
    }
  }

  /**
   * Stop the trading bot
   */
  async stop(reason?: string): Promise<void> {
    try {
      const response = await this.apiCall('/api/v1/trading/bot/stop', 'POST', { reason });

      if (response.success) {
        this.currentState.status = 'stopped';
        this.currentState.isActive = false;
      } else {
        throw new Error(response.error || 'Failed to stop bot');
      }
    } catch (error) {
      console.error('Error stopping bot:', error);
      throw error;
    }
  }

  /**
   * Get bot status
   */
  async getStatus(): Promise<BotState> {
    try {
      const response = await this.apiCall('/api/v1/trading/bot/status', 'GET');

      if (response.success) {
        this.currentState = response.data;
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to get status');
      }
    } catch (error) {
      console.error('Error getting status:', error);
      throw error;
    }
  }

  /**
   * Update bot configuration
   */
  async updateConfig(config: Partial<BotConfig>): Promise<void> {
    try {
      const response = await this.apiCall('/api/v1/trading/bot/config', 'PUT', config);

      if (response.success) {
        this.currentState.config = response.data.config;
      } else {
        throw new Error(response.error || 'Failed to update config');
      }
    } catch (error) {
      console.error('Error updating config:', error);
      throw error;
    }
  }

  /**
   * Emergency stop the bot
   */
  async emergencyStop(reason?: string): Promise<void> {
    try {
      const response = await this.apiCall('/api/v1/trading/bot/emergency-stop', 'POST', { reason });

      if (response.success) {
        this.currentState.status = 'stopped';
        this.currentState.isActive = false;
      } else {
        throw new Error(response.error || 'Failed to emergency stop');
      }
    } catch (error) {
      console.error('Error during emergency stop:', error);
      throw error;
    }
  }

  /**
   * Place a manual order
   */
  async placeManualOrder(order: ManualOrder): Promise<any> {
    try {
      const response = await this.apiCall('/api/v1/trading/manual/order', 'POST', order);

      if (response.success) {
        return response.data.order;
      } else {
        throw new Error(response.error || 'Failed to place order');
      }
    } catch (error) {
      console.error('Error placing manual order:', error);
      throw error;
    }
  }

  /**
   * Get recent signals
   */
  async getSignals(limit: number = 10): Promise<TradingSignal[]> {
    try {
      const response = await this.apiCall(`/api/v1/trading/signals?limit=${limit}`, 'GET');

      if (response.success) {
        return response.data.signals;
      } else {
        throw new Error(response.error || 'Failed to get signals');
      }
    } catch (error) {
      console.error('Error getting signals:', error);
      throw error;
    }
  }

  /**
   * Get bot performance metrics
   */
  async getPerformance(period: 'hourly' | 'daily' | 'weekly' = 'daily'): Promise<any> {
    try {
      const response = await this.apiCall(`/api/v1/trading/bot/performance?period=${period}`, 'GET');

      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to get performance');
      }
    } catch (error) {
      console.error('Error getting performance:', error);
      throw error;
    }
  }

  /**
   * Subscribe to trading events
   */
  subscribe(eventType: string, handler: (event: TradingEvent) => void): () => void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, new Set());
    }

    this.eventHandlers.get(eventType)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = this.eventHandlers.get(eventType);
      if (handlers) {
        handlers.delete(handler);
      }
    };
  }

  /**
   * Check if bot is currently running
   */
  isRunning(): boolean {
    return this.currentState.isActive;
  }

  /**
   * Get current state (cached)
   */
  getCurrentState(): BotState {
    return this.currentState;
  }

  /**
   * Make API call to backend
   */
  private async apiCall(endpoint: string, method: string = 'GET', body?: any): Promise<any> {
    try {
      // Get auth token
      const { data: { session } } = await supabase.auth.getSession();
      const token = session?.access_token;

      const options: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` })
        }
      };

      if (body && method !== 'GET') {
        options.body = JSON.stringify(body);
      }

      const response = await fetch(`${this.apiUrl}${endpoint}`, options);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      return data;

    } catch (error) {
      console.error(`API call failed: ${endpoint}`, error);
      throw error;
    }
  }

  /**
   * Cleanup resources
   */
  cleanup(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.eventHandlers.clear();
  }
}

// Export singleton instance
export const tradingBotService = new TradingBotService();

// Export for backward compatibility
export const tradingAgentV2 = {
  start: () => tradingBotService.start(),
  stop: () => tradingBotService.stop(),
  isRunning: () => tradingBotService.isRunning(),
  subscribe: (handler: (event: any) => void) => tradingBotService.subscribe('*', handler),
  getSignals: () => tradingBotService.getSignals(),
  getStatus: () => tradingBotService.getStatus()
};