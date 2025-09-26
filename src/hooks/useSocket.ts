import { useState, useEffect, useCallback } from 'react';
import socketService, { ConnectionStatus, SocketEventHandlers } from '../services/socketService';
import { Position, Order } from '../types/trading';

interface UseSocketReturn {
  isConnected: boolean;
  connectionStatus: ConnectionStatus;
  connect: () => Promise<void>;
  disconnect: () => void;
  emit: (event: string, data?: any) => Promise<any>;
  subscribe: (channels: string[]) => void;
  unsubscribe: (channels: string[]) => void;
  subscribeToOrderBook: (symbol: string) => void;
  unsubscribeFromOrderBook: (symbol: string) => void;
  subscribeToPrices: (symbols: string[]) => void;
  // Trading operations
  getPositions: () => Promise<any>;
  getOrders: (status?: string) => Promise<any>;
  getPerformance: (period?: string) => Promise<any>;
  startTrading: (settings?: any) => Promise<any>;
  stopTrading: () => Promise<any>;
  emergencyStop: () => Promise<any>;
  getMarketData: (symbols: string[]) => Promise<any>;
}

export function useSocket(handlers?: SocketEventHandlers): UseSocketReturn {
  const [isConnected, setIsConnected] = useState(socketService.isConnected());
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>(
    socketService.getConnectionStatus()
  );

  useEffect(() => {
    // Register event handlers
    if (handlers) {
      Object.entries(handlers).forEach(([event, handler]) => {
        if (handler) {
          socketService.on(event as keyof SocketEventHandlers, handler);
        }
      });
    }

    // Connection status handler
    const handleConnectionStatus = (status: ConnectionStatus) => {
      setConnectionStatus(status);
      setIsConnected(status === 'connected');
    };

    socketService.on('onConnectionStatus', handleConnectionStatus);

    // Check initial connection status
    setIsConnected(socketService.isConnected());
    setConnectionStatus(socketService.getConnectionStatus());

    // Cleanup
    return () => {
      if (handlers) {
        Object.entries(handlers).forEach(([event, handler]) => {
          if (handler) {
            socketService.off(event as keyof SocketEventHandlers, handler);
          }
        });
      }
      socketService.off('onConnectionStatus', handleConnectionStatus);
    };
  }, [handlers]);

  const connect = useCallback(async () => {
    await socketService.connect();
  }, []);

  const disconnect = useCallback(() => {
    socketService.disconnect();
  }, []);

  const emit = useCallback((event: string, data?: any) => {
    return socketService.emit(event, data);
  }, []);

  const subscribe = useCallback((channels: string[]) => {
    socketService.subscribeToChannels(channels);
  }, []);

  const unsubscribe = useCallback((channels: string[]) => {
    socketService.unsubscribeFromChannels(channels);
  }, []);

  const subscribeToOrderBook = useCallback((symbol: string) => {
    socketService.subscribeToOrderBook(symbol);
  }, []);

  const unsubscribeFromOrderBook = useCallback((symbol: string) => {
    socketService.unsubscribeFromOrderBook(symbol);
  }, []);

  const subscribeToPrices = useCallback((symbols: string[]) => {
    socketService.subscribeToPrices(symbols);
  }, []);

  // Trading operations
  const getPositions = useCallback(() => {
    return socketService.getPositions();
  }, []);

  const getOrders = useCallback((status?: string) => {
    return socketService.getOrders(status);
  }, []);

  const getPerformance = useCallback((period?: string) => {
    return socketService.getPerformance(period);
  }, []);

  const startTrading = useCallback((settings?: any) => {
    return socketService.startTrading(settings);
  }, []);

  const stopTrading = useCallback(() => {
    return socketService.stopTrading();
  }, []);

  const emergencyStop = useCallback(() => {
    return socketService.emergencyStop();
  }, []);

  const getMarketData = useCallback((symbols: string[]) => {
    return socketService.getMarketData(symbols);
  }, []);

  return {
    isConnected,
    connectionStatus,
    connect,
    disconnect,
    emit,
    subscribe,
    unsubscribe,
    subscribeToOrderBook,
    unsubscribeFromOrderBook,
    subscribeToPrices,
    getPositions,
    getOrders,
    getPerformance,
    startTrading,
    stopTrading,
    emergencyStop,
    getMarketData,
  };
}

// Hook for real-time positions
export function useRealtimePositions(initialPositions?: Position[]) {
  const [positions, setPositions] = useState<Position[]>(initialPositions || []);

  const handlers: SocketEventHandlers = {
    onPositionUpdate: useCallback((data: any) => {
      console.log('Position update received:', data);

      if (data.type === 'opened') {
        setPositions(prev => [...prev, data.position]);
      } else if (data.type === 'closed') {
        setPositions(prev => prev.filter(p => p.id !== data.position.id));
      } else if (data.type === 'updated') {
        setPositions(prev => prev.map(p =>
          p.id === data.position.id ? data.position : p
        ));
      }
    }, []),
  };

  const socket = useSocket(handlers);

  // Fetch initial positions when connected
  useEffect(() => {
    if (socket.isConnected) {
      socket.getPositions().then(response => {
        if (response.success) {
          setPositions(response.data);
        }
      }).catch(console.error);
    }
  }, [socket.isConnected]);

  return { positions, ...socket };
}

// Hook for real-time orders
export function useRealtimeOrders(initialOrders?: Order[]) {
  const [orders, setOrders] = useState<Order[]>(initialOrders || []);

  const handlers: SocketEventHandlers = {
    onOrderUpdate: useCallback((data: any) => {
      console.log('Order update received:', data);

      const { type, order } = data;

      if (type === 'executed' || type === 'created') {
        setOrders(prev => [order, ...prev]);
      } else if (type === 'filled') {
        setOrders(prev => prev.map(o =>
          o.id === order.id ? { ...o, status: 'filled' } : o
        ));
      } else if (type === 'failed') {
        setOrders(prev => prev.map(o =>
          o.id === order.id ? { ...o, status: 'failed' } : o
        ));
      }
    }, []),
  };

  const socket = useSocket(handlers);

  // Fetch initial orders when connected
  useEffect(() => {
    if (socket.isConnected) {
      socket.getOrders().then(response => {
        if (response.success) {
          setOrders(response.data);
        }
      }).catch(console.error);
    }
  }, [socket.isConnected]);

  return { orders, ...socket };
}

// Hook for real-time market data
export function useRealtimeMarketData(symbols: string[]) {
  const [marketData, setMarketData] = useState<Map<string, any>>(new Map());
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const handlers: SocketEventHandlers = {
    onMarketData: useCallback((data: any) => {
      if (data.prices) {
        const newData = new Map(marketData);
        data.prices.forEach((item: any) => {
          newData.set(item.symbol, item);
        });
        setMarketData(newData);
        setLastUpdate(new Date());
      }
    }, [marketData]),

    onPriceUpdate: useCallback((data: any) => {
      setMarketData(prev => {
        const newData = new Map(prev);
        newData.set(data.symbol, data);
        return newData;
      });
      setLastUpdate(new Date());
    }, []),
  };

  const socket = useSocket(handlers);

  // Subscribe to price updates for symbols
  useEffect(() => {
    if (socket.isConnected && symbols.length > 0) {
      socket.subscribeToPrices(symbols);

      // Fetch initial market data
      socket.getMarketData(symbols).then(response => {
        if (response.success && response.data) {
          const newData = new Map();
          response.data.forEach((item: any) => {
            newData.set(item.symbol, item);
          });
          setMarketData(newData);
        }
      }).catch(console.error);
    }

    return () => {
      // Unsubscribe when unmounting
      if (socket.isConnected && symbols.length > 0) {
        const channels = symbols.map(s => `price:${s}`);
        socket.unsubscribe(channels);
      }
    };
  }, [socket.isConnected, symbols.join(',')]);

  return { marketData, lastUpdate, ...socket };
}

// Hook for real-time order book
export function useRealtimeOrderBook(symbol: string) {
  const [orderBook, setOrderBook] = useState<any>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const handlers: SocketEventHandlers = {
    onOrderBookUpdate: useCallback((data: any) => {
      if (data.symbol === symbol) {
        setOrderBook(data.orderBook);
        setLastUpdate(new Date(data.timestamp));
      }
    }, [symbol]),
  };

  const socket = useSocket(handlers);

  // Subscribe to order book updates
  useEffect(() => {
    if (socket.isConnected && symbol) {
      socket.subscribeToOrderBook(symbol);
    }

    return () => {
      if (socket.isConnected && symbol) {
        socket.unsubscribeFromOrderBook(symbol);
      }
    };
  }, [socket.isConnected, symbol]);

  return { orderBook, lastUpdate, ...socket };
}

// Hook for alerts
export function useRealtimeAlerts() {
  const [alerts, setAlerts] = useState<any[]>([]);

  const handlers: SocketEventHandlers = {
    onAlert: useCallback((alert: any) => {
      console.log('Alert received:', alert);
      setAlerts(prev => [alert, ...prev].slice(0, 50)); // Keep last 50 alerts
    }, []),

    onEmergencyStop: useCallback((data: any) => {
      console.error('Emergency stop:', data);
      setAlerts(prev => [{
        id: `emergency_${Date.now()}`,
        type: 'system',
        severity: 'critical',
        title: 'Emergency Stop Triggered',
        message: data.message || 'Trading has been stopped',
        timestamp: new Date()
      }, ...prev]);
    }, []),
  };

  const socket = useSocket(handlers);

  return { alerts, ...socket };
}

// Hook for trading bot status
export function useRealtimeBotStatus() {
  const [botStatus, setBotStatus] = useState<any>(null);
  const [tradingEvents, setTradingEvents] = useState<any[]>([]);
  const [signals, setSignals] = useState<any[]>([]);

  const handlers: SocketEventHandlers = {
    onBotStatus: useCallback((data: any) => {
      console.log('Bot status update:', data);
      setBotStatus(data);
    }, []),

    onTradingEvent: useCallback((event: any) => {
      console.log('Trading event:', event);
      setTradingEvents(prev => [event, ...prev].slice(0, 100)); // Keep last 100 events

      // Update bot status based on event type
      if (event.type === 'status') {
        setBotStatus(prev => ({ ...prev, ...event.data }));
      }
    }, []),

    onSignalUpdate: useCallback((signal: any) => {
      console.log('Signal update:', signal);
      setSignals(prev => [signal, ...prev].slice(0, 50)); // Keep last 50 signals
    }, []),

    onTradingCycle: useCallback((data: any) => {
      console.log('Trading cycle complete:', data);
      setBotStatus(prev => ({
        ...prev,
        cyclesExecuted: data.cyclesExecuted || (prev?.cyclesExecuted || 0) + 1,
        signalsGenerated: data.signalsGenerated || prev?.signalsGenerated || 0
      }));
    }, []),
  };

  const socket = useSocket(handlers);

  return { botStatus, tradingEvents, signals, ...socket };
}