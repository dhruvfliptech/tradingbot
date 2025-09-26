import { io, Socket } from 'socket.io-client';
import { supabase } from './supabase';

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface SocketEventHandlers {
  onOrderUpdate?: (data: any) => void;
  onPositionUpdate?: (data: any) => void;
  onMarketData?: (data: any) => void;
  onPriceUpdate?: (data: any) => void;
  onOrderBookUpdate?: (data: any) => void;
  onAlert?: (data: any) => void;
  onTradingCycle?: (data: any) => void;
  onConnectionStatus?: (status: ConnectionStatus) => void;
  onEmergencyStop?: (data: any) => void;
  onTradingEvent?: (data: any) => void;
  onBotStatus?: (data: any) => void;
  onSignalUpdate?: (data: any) => void;
}

class SocketService {
  private socket: Socket | null = null;
  private eventHandlers: Map<string, Set<Function>> = new Map();
  private connectionStatus: ConnectionStatus = 'disconnected';
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private subscribedChannels: Set<string> = new Set();
  private isAuthenticated = false;

  constructor() {
    // Bind methods to preserve 'this' context
    this.connect = this.connect.bind(this);
    this.disconnect = this.disconnect.bind(this);
    this.emit = this.emit.bind(this);
  }

  /**
   * Connect to Socket.IO server
   */
  async connect(): Promise<void> {
    if (this.socket?.connected) {
      console.log('Socket already connected');
      return;
    }

    try {
      // Get auth token from Supabase
      const { data: { session } } = await supabase.auth.getSession();
      const token = session?.access_token;

      if (!token) {
        console.error('No auth token available');
        this.updateConnectionStatus('error');
        return;
      }

      // Create Socket.IO connection
      this.socket = io(import.meta.env.VITE_BACKEND_URL || 'http://localhost:3001', {
        auth: {
          token
        },
        transports: ['websocket', 'polling'], // Prefer WebSocket but allow fallback
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: this.maxReconnectAttempts,
        timeout: 20000,
      });

      this.setupEventHandlers();
      this.updateConnectionStatus('connecting');

    } catch (error) {
      console.error('Failed to connect to Socket.IO:', error);
      this.updateConnectionStatus('error');
    }
  }

  /**
   * Set up Socket.IO event handlers
   */
  private setupEventHandlers(): void {
    if (!this.socket) return;

    // Connection events
    this.socket.on('connect', () => {
      console.log('Socket.IO connected:', this.socket?.id);
      this.reconnectAttempts = 0;
      this.updateConnectionStatus('connected');

      // Re-subscribe to channels after reconnection
      if (this.subscribedChannels.size > 0) {
        this.subscribeToChannels(Array.from(this.subscribedChannels));
      }
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Socket.IO disconnected:', reason);
      this.isAuthenticated = false;
      this.updateConnectionStatus('disconnected');
    });

    this.socket.on('connect_error', (error) => {
      console.error('Socket.IO connection error:', error.message);
      this.reconnectAttempts++;

      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        this.updateConnectionStatus('error');
      }
    });

    // Authentication events
    this.socket.on('connection_success', (data) => {
      console.log('Socket connection success:', data);

      if (data.authenticated) {
        this.isAuthenticated = true;
        // Auto-subscribe to user channels
        this.subscribeToUserChannels(data.userId);
      } else {
        // Send authentication if not auto-authenticated
        this.authenticate();
      }
    });

    // Trading events
    this.socket.on('order_update', (data) => {
      console.log('Order update:', data);
      this.emitToHandlers('onOrderUpdate', data);
    });

    this.socket.on('position_update', (data) => {
      console.log('Position update:', data);
      this.emitToHandlers('onPositionUpdate', data);
    });

    this.socket.on('market_data', (data) => {
      this.emitToHandlers('onMarketData', data);
    });

    this.socket.on('price_update', (data) => {
      this.emitToHandlers('onPriceUpdate', data);
    });

    this.socket.on('orderbook_update', (data) => {
      this.emitToHandlers('onOrderBookUpdate', data);
    });

    this.socket.on('alert', (data) => {
      console.log('Alert received:', data);
      this.emitToHandlers('onAlert', data);
    });

    this.socket.on('trading_cycle', (data) => {
      console.log('Trading cycle complete:', data);
      this.emitToHandlers('onTradingCycle', data);
    });

    this.socket.on('emergency_stop', (data) => {
      console.error('Emergency stop triggered:', data);
      this.emitToHandlers('onEmergencyStop', data);
    });

    this.socket.on('emergency_stop_executed', (data) => {
      console.error('Emergency stop executed:', data);
      this.emitToHandlers('onEmergencyStop', data);
    });

    // Trading bot events
    this.socket.on('trading_event', (data) => {
      console.log('Trading event:', data.type);
      this.emitToHandlers('onTradingEvent', data);
    });

    this.socket.on('bot_status', (data) => {
      console.log('Bot status update:', data);
      this.emitToHandlers('onBotStatus', data);
    });

    this.socket.on('signal_update', (data) => {
      console.log('Signal update:', data);
      this.emitToHandlers('onSignalUpdate', data);
    });

    // Ping/Pong for connection health
    this.socket.on('pong', (data) => {
      // Connection is healthy
    });
  }

  /**
   * Authenticate with the server
   */
  private async authenticate(): Promise<void> {
    if (!this.socket) return;

    const { data: { session } } = await supabase.auth.getSession();
    const token = session?.access_token;

    if (!token) {
      console.error('No auth token for authentication');
      return;
    }

    this.socket.emit('authenticate', { token }, (response: any) => {
      if (response.success) {
        console.log('Socket authenticated:', response.userId);
        this.isAuthenticated = true;
        this.subscribeToUserChannels(response.userId);
      } else {
        console.error('Socket authentication failed:', response.error);
        this.updateConnectionStatus('error');
      }
    });
  }

  /**
   * Subscribe to user-specific channels
   */
  private subscribeToUserChannels(userId: string): void {
    const channels = [
      `user:${userId}`,
      `user:${userId}:orders`,
      `user:${userId}:positions`,
      `user:${userId}:alerts`,
      `user:${userId}:bot`,
      `user:${userId}:signals`,
      'market:prices'
    ];

    this.subscribeToChannels(channels);
  }

  /**
   * Subscribe to channels
   */
  subscribeToChannels(channels: string[]): void {
    if (!this.socket?.connected) {
      console.warn('Socket not connected, cannot subscribe to channels');
      return;
    }

    this.socket.emit('subscribe', channels, (response: any) => {
      if (response.success) {
        console.log('Subscribed to channels:', channels);
        channels.forEach(channel => this.subscribedChannels.add(channel));
      } else {
        console.error('Failed to subscribe:', response.error);
      }
    });
  }

  /**
   * Unsubscribe from channels
   */
  unsubscribeFromChannels(channels: string[]): void {
    if (!this.socket?.connected) return;

    this.socket.emit('unsubscribe', channels, (response: any) => {
      if (response.success) {
        console.log('Unsubscribed from channels:', channels);
        channels.forEach(channel => this.subscribedChannels.delete(channel));
      }
    });
  }

  /**
   * Subscribe to order book updates for a symbol
   */
  subscribeToOrderBook(symbol: string): void {
    const channel = `orderbook:${symbol}`;
    this.subscribeToChannels([channel]);
  }

  /**
   * Unsubscribe from order book updates
   */
  unsubscribeFromOrderBook(symbol: string): void {
    const channel = `orderbook:${symbol}`;
    this.unsubscribeFromChannels([channel]);
  }

  /**
   * Subscribe to price updates for specific symbols
   */
  subscribeToPrices(symbols: string[]): void {
    const channels = symbols.map(symbol => `price:${symbol}`);
    this.subscribeToChannels(channels);
  }

  /**
   * Register event handlers
   */
  on(event: keyof SocketEventHandlers, handler: Function): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler);
  }

  /**
   * Remove event handler
   */
  off(event: keyof SocketEventHandlers, handler: Function): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  /**
   * Emit event to registered handlers
   */
  private emitToHandlers(event: string, data: any): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in handler for ${event}:`, error);
        }
      });
    }
  }

  /**
   * Update connection status
   */
  private updateConnectionStatus(status: ConnectionStatus): void {
    this.connectionStatus = status;
    this.emitToHandlers('onConnectionStatus', status);
  }

  /**
   * Get current connection status
   */
  getConnectionStatus(): ConnectionStatus {
    return this.connectionStatus;
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  /**
   * Send a ping to check connection health
   */
  ping(): void {
    if (this.socket?.connected) {
      this.socket.emit('ping');
    }
  }

  /**
   * Emit an event to the server with acknowledgment
   */
  emit(event: string, data?: any): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.socket?.connected) {
        reject(new Error('Socket not connected'));
        return;
      }

      // If no data provided, just emit the event
      if (data === undefined) {
        this.socket.emit(event, (response: any) => {
          if (response?.success === false) {
            reject(new Error(response.error || 'Request failed'));
          } else {
            resolve(response);
          }
        });
      } else {
        // Emit with data
        this.socket.emit(event, data, (response: any) => {
          if (response?.success === false) {
            reject(new Error(response.error || 'Request failed'));
          } else {
            resolve(response);
          }
        });
      }
    });
  }

  /**
   * Trading-specific methods
   */
  async getPositions(): Promise<any> {
    return this.emit('get_positions');
  }

  async getOrders(status?: string): Promise<any> {
    return this.emit('get_orders', { status });
  }

  async getPerformance(period?: string): Promise<any> {
    return this.emit('get_performance', { period });
  }

  async startTrading(settings?: any): Promise<any> {
    return this.emit('start_trading', { settings });
  }

  async stopTrading(): Promise<any> {
    return this.emit('stop_trading');
  }

  async emergencyStop(): Promise<any> {
    return this.emit('emergency_stop');
  }

  async getMarketData(symbols: string[]): Promise<any> {
    return this.emit('get_market_data', symbols);
  }

  /**
   * Disconnect from Socket.IO server
   */
  disconnect(): void {
    if (this.socket) {
      console.log('Disconnecting Socket.IO');
      this.socket.disconnect();
      this.socket = null;
      this.subscribedChannels.clear();
      this.isAuthenticated = false;
      this.updateConnectionStatus('disconnected');
    }
  }

  /**
   * Clean up resources
   */
  cleanup(): void {
    this.eventHandlers.clear();
    this.disconnect();
  }
}

// Create singleton instance
const socketService = new SocketService();

// Auto-connect when authenticated
supabase.auth.onAuthStateChange((event, session) => {
  if (event === 'SIGNED_IN' && session) {
    console.log('Auth state changed: signed in, connecting socket');
    socketService.connect();
  } else if (event === 'SIGNED_OUT') {
    console.log('Auth state changed: signed out, disconnecting socket');
    socketService.disconnect();
  }
});

export default socketService;