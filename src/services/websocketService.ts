import React from 'react';
import { statePersistenceService } from './persistence/statePersistenceService';

export type WSMessage = 
  | { type: 'price_update'; data: { symbol: string; price: number; change: number } }
  | { type: 'trade_executed'; data: { symbol: string; side: string; quantity: number; price: number } }
  | { type: 'position_update'; data: { positions: any[] } }
  | { type: 'agent_status'; data: { active: boolean; nextCycle: number } }
  | { type: 'heartbeat'; data: { timestamp: number } };

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectTimer: number | null = null;
  private heartbeatTimer: number | null = null;
  private subscribers: Map<string, Set<(message: WSMessage) => void>> = new Map();
  private url: string;
  private isConnecting: boolean = false;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectDelay: number = 1000; // Start with 1 second
  private maxReconnectDelay: number = 30000; // Max 30 seconds
  
  constructor() {
    // In production, this would be your WebSocket server URL
    // For now, we'll simulate WebSocket behavior
    this.url = import.meta.env.VITE_WS_URL || 'ws://localhost:8080';
    
    // Auto-connect on instantiation
    this.connect();
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', this.handleVisibilityChange.bind(this));
    
    // Handle online/offline
    window.addEventListener('online', this.handleOnline.bind(this));
    window.addEventListener('offline', this.handleOffline.bind(this));
  }

  /**
   * Connect to WebSocket server
   */
  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return;
    }
    
    this.isConnecting = true;
    
    try {
      // For demo purposes, we'll simulate WebSocket behavior
      // In production, uncomment the WebSocket connection
      /*
      this.ws = new WebSocket(this.url);
      
      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      */
      
      // Simulate connection for demo
      this.simulateConnection();
    } catch (error) {
      console.error('WebSocket connection error:', error);
      this.isConnecting = false;
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    this.clearTimers();
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    
    this.reconnectAttempts = 0;
  }

  /**
   * Subscribe to specific message types
   */
  subscribe(type: string, callback: (message: WSMessage) => void): () => void {
    if (!this.subscribers.has(type)) {
      this.subscribers.set(type, new Set());
    }
    
    this.subscribers.get(type)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      const typeSubscribers = this.subscribers.get(type);
      if (typeSubscribers) {
        typeSubscribers.delete(callback);
        if (typeSubscribers.size === 0) {
          this.subscribers.delete(type);
        }
      }
    };
  }

  /**
   * Send message through WebSocket
   */
  send(message: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, queuing message');
      // In production, implement a message queue
    }
  }

  // Private methods
  private handleOpen(): void {
    console.log('WebSocket connected');
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    this.reconnectDelay = 1000;
    
    // Start heartbeat
    this.startHeartbeat();
    
    // Notify subscribers
    this.emit({
      type: 'agent_status',
      data: { active: true, nextCycle: Date.now() + 45000 }
    });
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data) as WSMessage;
      this.emit(message);
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  private handleError(event: Event): void {
    console.error('WebSocket error:', event);
    this.isConnecting = false;
  }

  private handleClose(event: CloseEvent): void {
    console.log('WebSocket closed:', event.code, event.reason);
    this.isConnecting = false;
    this.clearTimers();
    
    // Only reconnect if not a normal closure
    if (event.code !== 1000) {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      return;
    }
    
    this.reconnectAttempts++;
    
    // Exponential backoff with jitter
    const jitter = Math.random() * 1000;
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1) + jitter,
      this.maxReconnectDelay
    );
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    this.reconnectTimer = window.setTimeout(() => {
      this.connect();
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = window.setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping', timestamp: Date.now() });
        
        // Also emit local heartbeat
        this.emit({
          type: 'heartbeat',
          data: { timestamp: Date.now() }
        });
      }
    }, 30000); // Every 30 seconds
  }

  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private emit(message: WSMessage): void {
    // Emit to all subscribers
    const allSubscribers = this.subscribers.get('*') || new Set();
    allSubscribers.forEach(callback => {
      try {
        callback(message);
      } catch (error) {
        console.error('Subscriber error:', error);
      }
    });
    
    // Emit to type-specific subscribers
    const typeSubscribers = this.subscribers.get(message.type) || new Set();
    typeSubscribers.forEach(callback => {
      try {
        callback(message);
      } catch (error) {
        console.error('Subscriber error:', error);
      }
    });
  }

  private handleVisibilityChange(): void {
    if (document.hidden) {
      // Page is hidden, reduce activity
      this.clearTimers();
    } else {
      // Page is visible, ensure connection
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        this.connect();
      } else {
        this.startHeartbeat();
      }
    }
  }

  private handleOnline(): void {
    console.log('Network online, reconnecting WebSocket');
    this.connect();
  }

  private handleOffline(): void {
    console.log('Network offline');
    this.disconnect();
  }

  // Simulation methods for demo (remove in production)
  private simulateConnection(): void {
    // Simulate successful connection after a short delay
    setTimeout(() => {
      this.isConnecting = false;
      this.reconnectAttempts = 0;
      
      // Start simulating real-time updates
      this.simulateRealTimeUpdates();
    }, 500);
  }

  private simulateRealTimeUpdates(): void {
    // Simulate price updates every 5 seconds
    setInterval(() => {
      const symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA'];
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      const basePrice = symbol === 'BTC' ? 65000 : symbol === 'ETH' ? 3500 : 300;
      const variation = (Math.random() - 0.5) * 0.02; // ±1% variation
      
      this.emit({
        type: 'price_update',
        data: {
          symbol,
          price: basePrice * (1 + variation),
          change: variation * 100
        }
      });
    }, 5000);
    
    // Simulate trade executions occasionally
    setInterval(() => {
      if (Math.random() > 0.7) { // 30% chance
        const symbols = ['BTC', 'ETH', 'SOL'];
        const symbol = symbols[Math.floor(Math.random() * symbols.length)];
        
        this.emit({
          type: 'trade_executed',
          data: {
            symbol,
            side: Math.random() > 0.5 ? 'buy' : 'sell',
            quantity: Math.random() * 0.1,
            price: symbol === 'BTC' ? 65000 : symbol === 'ETH' ? 3500 : 150
          }
        });
      }
    }, 15000);
    
    // Simulate agent status updates
    setInterval(() => {
      this.emit({
        type: 'agent_status',
        data: {
          active: true,
          nextCycle: Date.now() + 45000
        }
      });
    }, 45000);
  }

  /**
   * Get connection status
   */
  getStatus(): 'connected' | 'connecting' | 'disconnected' {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return 'connected';
    }
    if (this.isConnecting) {
      return 'connecting';
    }
    return 'disconnected';
  }
}

// Export singleton instance
export const websocketService = new WebSocketService();

// React hook for WebSocket
export const useWebSocket = (messageType?: string) => {
  const [lastMessage, setLastMessage] = React.useState<WSMessage | null>(null);
  const [isConnected, setIsConnected] = React.useState(false);
  
  React.useEffect(() => {
    // Subscribe to messages
    const unsubscribe = websocketService.subscribe(messageType || '*', (message) => {
      setLastMessage(message);
    });
    
    // Check connection status
    const checkStatus = () => {
      setIsConnected(websocketService.getStatus() === 'connected');
    };
    
    checkStatus();
    const statusInterval = setInterval(checkStatus, 1000);
    
    return () => {
      unsubscribe();
      clearInterval(statusInterval);
    };
  }, [messageType]);
  
  return {
    lastMessage,
    isConnected,
    send: websocketService.send.bind(websocketService)
  };
};