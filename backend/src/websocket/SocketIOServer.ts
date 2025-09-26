import { Server as HTTPServer } from 'http';
import { Server, Socket } from 'socket.io';
import jwt from 'jsonwebtoken';
import logger from '../utils/logger';
import { TradingEngineService } from '../services/trading/TradingEngineService';
import { StateManager } from '../services/trading/StateManager';
import { MarketDataService } from '../services/trading/MarketDataService';
import { Alert } from '../types/trading';

interface AuthenticatedSocket extends Socket {
  userId?: string;
  isAuthenticated?: boolean;
}

interface SocketData {
  userId?: string;
  isAuthenticated?: boolean;
  subscribedChannels?: Set<string>;
}

export class SocketIOServer {
  private io: Server;
  private stateManager: StateManager;
  private tradingEngine: TradingEngineService;
  private marketDataService: MarketDataService;
  private connectedClients: Map<string, Set<string>> = new Map(); // userId -> socketIds
  private marketDataInterval: NodeJS.Timeout | null = null;
  private orderBookIntervals: Map<string, NodeJS.Timeout> = new Map();

  constructor(server: HTTPServer) {
    // Initialize Socket.IO with configuration
    this.io = new Server(server, {
      cors: {
        origin: process.env.FRONTEND_URL || 'http://localhost:5173',
        credentials: true,
        methods: ['GET', 'POST']
      },
      // Socket.IO specific options for reliability
      pingTimeout: 60000, // 60 seconds
      pingInterval: 25000, // 25 seconds
      upgradeTimeout: 30000, // 30 seconds
      maxHttpBufferSize: 1e8, // 100 MB
      allowEIO3: true, // Allow different Socket.IO versions
      transports: ['websocket', 'polling'], // Allow fallback to polling
    });

    this.stateManager = new StateManager();
    this.tradingEngine = TradingEngineService.getInstance();
    this.marketDataService = new MarketDataService();

    this.initialize();
  }

  private initialize(): void {
    // Set up authentication middleware
    this.io.use(async (socket: AuthenticatedSocket, next) => {
      try {
        const token = socket.handshake.auth.token || socket.handshake.headers.authorization?.replace('Bearer ', '');

        if (!token) {
          // Allow connection but mark as unauthenticated
          socket.data.isAuthenticated = false;
          return next();
        }

        // Verify JWT token
        const decoded = jwt.verify(token, process.env.JWT_SECRET || 'secret') as any;
        const userId = decoded.userId || decoded.sub || decoded.id;

        if (userId) {
          socket.data.userId = userId;
          socket.data.isAuthenticated = true;
          logger.info(`Socket authenticated for user: ${userId}`);
        }

        next();
      } catch (error) {
        logger.error('Socket authentication error:', error);
        socket.data.isAuthenticated = false;
        next(); // Allow connection but as unauthenticated
      }
    });

    // Handle connections
    this.io.on('connection', (socket: Socket) => {
      this.handleConnection(socket);
    });

    // Subscribe to trading engine events
    this.subscribeToTradingEvents();

    // Start market data streaming
    this.startMarketDataStreaming();

    logger.info('Socket.IO Server initialized successfully');
  }

  private handleConnection(socket: Socket): void {
    const socketData = socket.data as SocketData;
    logger.info(`New socket connection: ${socket.id}, authenticated: ${socketData.isAuthenticated}`);

    // Initialize socket data
    socketData.subscribedChannels = new Set();

    // Send connection success message
    socket.emit('connection_success', {
      socketId: socket.id,
      authenticated: socketData.isAuthenticated,
      userId: socketData.userId,
      timestamp: new Date()
    });

    // If authenticated, add to user's connected clients
    if (socketData.isAuthenticated && socketData.userId) {
      this.addUserConnection(socketData.userId, socket.id);

      // Auto-join user-specific rooms
      socket.join(`user:${socketData.userId}`);
      socket.join(`user:${socketData.userId}:orders`);
      socket.join(`user:${socketData.userId}:positions`);
      socket.join(`user:${socketData.userId}:alerts`);
    }

    // Set up event handlers
    this.setupSocketEventHandlers(socket);

    // Handle disconnection
    socket.on('disconnect', (reason) => {
      logger.info(`Socket disconnected: ${socket.id}, reason: ${reason}`);

      if (socketData.userId) {
        this.removeUserConnection(socketData.userId, socket.id);
      }

      // Clean up order book subscriptions
      socketData.subscribedChannels?.forEach(channel => {
        if (channel.startsWith('orderbook:')) {
          this.unsubscribeFromOrderBook(channel);
        }
      });
    });
  }

  private setupSocketEventHandlers(socket: Socket): void {
    const socketData = socket.data as SocketData;

    // Authentication
    socket.on('authenticate', async (data, callback) => {
      try {
        const { token } = data;
        if (!token) {
          return callback({ success: false, error: 'Token required' });
        }

        const decoded = jwt.verify(token, process.env.JWT_SECRET || 'secret') as any;
        const userId = decoded.userId || decoded.sub || decoded.id;

        if (userId) {
          socketData.userId = userId;
          socketData.isAuthenticated = true;

          // Add to user connections
          this.addUserConnection(userId, socket.id);

          // Join user rooms
          socket.join(`user:${userId}`);
          socket.join(`user:${userId}:orders`);
          socket.join(`user:${userId}:positions`);
          socket.join(`user:${userId}:alerts`);

          callback({ success: true, userId });
          logger.info(`Socket ${socket.id} authenticated as user ${userId}`);
        } else {
          callback({ success: false, error: 'Invalid token' });
        }
      } catch (error) {
        logger.error('Authentication error:', error);
        callback({ success: false, error: 'Authentication failed' });
      }
    });

    // Subscribe to channels
    socket.on('subscribe', (channels: string[], callback) => {
      if (!Array.isArray(channels)) {
        return callback?.({ success: false, error: 'Invalid channels' });
      }

      channels.forEach(channel => {
        socket.join(channel);
        socketData.subscribedChannels?.add(channel);

        // Special handling for order book subscriptions
        if (channel.startsWith('orderbook:')) {
          this.subscribeToOrderBook(channel, socket);
        }
      });

      callback?.({ success: true, channels });
      logger.debug(`Socket ${socket.id} subscribed to: ${channels.join(', ')}`);
    });

    // Unsubscribe from channels
    socket.on('unsubscribe', (channels: string[], callback) => {
      if (!Array.isArray(channels)) {
        return callback?.({ success: false, error: 'Invalid channels' });
      }

      channels.forEach(channel => {
        socket.leave(channel);
        socketData.subscribedChannels?.delete(channel);

        // Clean up order book subscriptions
        if (channel.startsWith('orderbook:')) {
          this.unsubscribeFromOrderBook(channel);
        }
      });

      callback?.({ success: true, channels });
    });

    // Trading operations (require authentication)
    socket.on('get_positions', async (callback) => {
      if (!socketData.isAuthenticated || !socketData.userId) {
        return callback?.({ success: false, error: 'Authentication required' });
      }

      try {
        const positions = await this.tradingEngine.getPositions(socketData.userId);
        callback({ success: true, data: positions });
      } catch (error) {
        logger.error('Error getting positions:', error);
        callback({ success: false, error: 'Failed to get positions' });
      }
    });

    socket.on('get_orders', async (data, callback) => {
      if (!socketData.isAuthenticated || !socketData.userId) {
        return callback?.({ success: false, error: 'Authentication required' });
      }

      try {
        const orders = await this.tradingEngine.getOrders(socketData.userId, data?.status);
        callback({ success: true, data: orders });
      } catch (error) {
        logger.error('Error getting orders:', error);
        callback({ success: false, error: 'Failed to get orders' });
      }
    });

    socket.on('get_performance', async (data, callback) => {
      if (!socketData.isAuthenticated || !socketData.userId) {
        return callback?.({ success: false, error: 'Authentication required' });
      }

      try {
        const performance = await this.tradingEngine.getPerformanceMetrics(
          socketData.userId,
          data?.period || '24h'
        );
        callback({ success: true, data: performance });
      } catch (error) {
        logger.error('Error getting performance:', error);
        callback({ success: false, error: 'Failed to get performance' });
      }
    });

    socket.on('start_trading', async (data, callback) => {
      if (!socketData.isAuthenticated || !socketData.userId) {
        return callback?.({ success: false, error: 'Authentication required' });
      }

      try {
        const session = await this.tradingEngine.startSession(socketData.userId, data?.settings);
        callback({ success: true, data: { sessionId: session.id } });
      } catch (error) {
        logger.error('Error starting trading:', error);
        callback({ success: false, error: 'Failed to start trading' });
      }
    });

    socket.on('stop_trading', async (callback) => {
      if (!socketData.isAuthenticated || !socketData.userId) {
        return callback?.({ success: false, error: 'Authentication required' });
      }

      try {
        await this.tradingEngine.stopSession(socketData.userId);
        callback({ success: true });
      } catch (error) {
        logger.error('Error stopping trading:', error);
        callback({ success: false, error: 'Failed to stop trading' });
      }
    });

    socket.on('emergency_stop', async (callback) => {
      if (!socketData.isAuthenticated || !socketData.userId) {
        return callback?.({ success: false, error: 'Authentication required' });
      }

      try {
        await this.tradingEngine.stopSession(socketData.userId);
        socket.emit('emergency_stop_executed', {
          message: 'Trading stopped immediately',
          timestamp: new Date()
        });
        callback({ success: true });
      } catch (error) {
        logger.error('Error in emergency stop:', error);
        callback({ success: false, error: 'Emergency stop failed' });
      }
    });

    // Market data requests
    socket.on('get_market_data', async (symbols: string[], callback) => {
      try {
        const marketData = await this.marketDataService.getMarketData(symbols);
        callback({ success: true, data: marketData });
      } catch (error) {
        logger.error('Error getting market data:', error);
        callback({ success: false, error: 'Failed to get market data' });
      }
    });

    // Ping/Pong for connection health
    socket.on('ping', () => {
      socket.emit('pong', { timestamp: new Date() });
    });
  }

  private subscribeToTradingEvents(): void {
    // Subscribe to trading engine events
    this.tradingEngine.on('orderExecuted', (data) => {
      const { userId, order } = data;
      this.io.to(`user:${userId}:orders`).emit('order_update', {
        type: 'executed',
        order,
        timestamp: new Date()
      });
    });

    this.tradingEngine.on('orderFilled', (order) => {
      if (order.userId) {
        this.io.to(`user:${order.userId}:orders`).emit('order_update', {
          type: 'filled',
          order,
          timestamp: new Date()
        });
      }
    });

    this.tradingEngine.on('orderFailed', (data) => {
      const { order, error } = data;
      if (order.userId) {
        this.io.to(`user:${order.userId}:orders`).emit('order_update', {
          type: 'failed',
          order,
          error,
          timestamp: new Date()
        });
      }
    });

    this.tradingEngine.on('positionOpened', (position) => {
      this.io.to(`user:${position.userId}:positions`).emit('position_update', {
        type: 'opened',
        position,
        timestamp: new Date()
      });
    });

    this.tradingEngine.on('positionClosed', (position) => {
      this.io.to(`user:${position.userId}:positions`).emit('position_update', {
        type: 'closed',
        position,
        timestamp: new Date()
      });
    });

    this.tradingEngine.on('tradingCycleComplete', (data) => {
      const { userId, signalsEvaluated, tradesExecuted } = data;
      this.io.to(`user:${userId}`).emit('trading_cycle', {
        signalsEvaluated,
        tradesExecuted,
        timestamp: new Date()
      });
    });

    this.tradingEngine.on('riskLimitExceeded', (data) => {
      const { userId, reason } = data;
      const alert: Alert = {
        id: `alert_${Date.now()}`,
        userId,
        type: 'risk',
        severity: 'warning',
        title: 'Risk Limit Exceeded',
        message: reason,
        timestamp: new Date()
      };

      this.io.to(`user:${userId}:alerts`).emit('alert', alert);
    });

    this.tradingEngine.on('emergencyStop', (userId) => {
      const alert: Alert = {
        id: `alert_${Date.now()}`,
        userId,
        type: 'system',
        severity: 'critical',
        title: 'Emergency Stop',
        message: 'Trading has been stopped due to emergency conditions',
        timestamp: new Date()
      };

      this.io.to(`user:${userId}:alerts`).emit('alert', alert);
      this.io.to(`user:${userId}`).emit('emergency_stop', {
        message: 'All trading activities stopped',
        timestamp: new Date()
      });
    });

    logger.info('Subscribed to all trading engine events');
  }

  private startMarketDataStreaming(): void {
    // Stream market data every 5 seconds
    this.marketDataInterval = setInterval(async () => {
      try {
        // Get market data for popular symbols
        const symbols = ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano'];
        const marketData = await this.marketDataService.getMarketData(symbols);

        // Broadcast to all connected clients in market room
        this.io.to('market:prices').emit('market_data', {
          prices: marketData,
          timestamp: new Date()
        });

        // Also emit individual symbol updates
        marketData.forEach(data => {
          this.io.to(`price:${data.symbol}`).emit('price_update', {
            symbol: data.symbol,
            price: data.price,
            change: data.changePercent,
            volume: data.volume,
            timestamp: new Date()
          });
        });

      } catch (error) {
        logger.error('Error streaming market data:', error);
      }
    }, 5000); // Every 5 seconds

    logger.info('Market data streaming started');
  }

  private async subscribeToOrderBook(channel: string, socket: Socket): Promise<void> {
    const symbol = channel.split(':')[1];
    if (!symbol) return;

    // Check if we already have an interval for this symbol
    if (this.orderBookIntervals.has(channel)) {
      return;
    }

    // Create interval to stream order book updates
    const interval = setInterval(async () => {
      try {
        const orderBook = await this.marketDataService.getOrderBook(symbol);

        // Emit to all sockets in this channel
        this.io.to(channel).emit('orderbook_update', {
          symbol,
          orderBook,
          timestamp: new Date()
        });

      } catch (error) {
        logger.error(`Error streaming order book for ${symbol}:`, error);
      }
    }, 1000); // Every second for order book

    this.orderBookIntervals.set(channel, interval);
    logger.debug(`Started order book streaming for ${symbol}`);
  }

  private unsubscribeFromOrderBook(channel: string): void {
    const interval = this.orderBookIntervals.get(channel);
    if (interval) {
      clearInterval(interval);
      this.orderBookIntervals.delete(channel);
      logger.debug(`Stopped order book streaming for ${channel}`);
    }
  }

  private addUserConnection(userId: string, socketId: string): void {
    if (!this.connectedClients.has(userId)) {
      this.connectedClients.set(userId, new Set());
    }
    this.connectedClients.get(userId)!.add(socketId);

    // Emit user online status
    this.io.to(`user:${userId}`).emit('connection_status', {
      online: true,
      connections: this.connectedClients.get(userId)!.size
    });
  }

  private removeUserConnection(userId: string, socketId: string): void {
    const userSockets = this.connectedClients.get(userId);
    if (userSockets) {
      userSockets.delete(socketId);

      if (userSockets.size === 0) {
        this.connectedClients.delete(userId);
        // User is now offline
        this.io.to(`user:${userId}`).emit('connection_status', {
          online: false,
          connections: 0
        });
      }
    }
  }

  // Public methods for external use
  public emitToUser(userId: string, event: string, data: any): void {
    this.io.to(`user:${userId}`).emit(event, data);
  }

  public emitToRoom(room: string, event: string, data: any): void {
    this.io.to(room).emit(event, data);
  }

  public broadcast(event: string, data: any): void {
    this.io.emit(event, data);
  }

  public getMetrics(): any {
    const sockets = Array.from(this.io.sockets.sockets.values());

    return {
      totalConnections: this.io.sockets.sockets.size,
      authenticatedUsers: this.connectedClients.size,
      connectionsPerUser: Array.from(this.connectedClients.entries()).map(([userId, socketIds]) => ({
        userId,
        connections: socketIds.size
      })),
      rooms: Array.from(this.io.sockets.adapter.rooms.keys()),
      transports: sockets.map(s => (s as any).conn.transport.name),
      socketDetails: sockets.map(s => ({
        id: s.id,
        userId: s.data.userId,
        authenticated: s.data.isAuthenticated,
        rooms: Array.from(s.rooms)
      }))
    };
  }

  public async shutdown(): Promise<void> {
    logger.info('Shutting down Socket.IO server');

    // Clear intervals
    if (this.marketDataInterval) {
      clearInterval(this.marketDataInterval);
    }

    this.orderBookIntervals.forEach(interval => clearInterval(interval));
    this.orderBookIntervals.clear();

    // Disconnect all sockets
    this.io.disconnectSockets(true);

    // Close the server
    await new Promise<void>((resolve) => {
      this.io.close(() => {
        logger.info('Socket.IO server closed');
        resolve();
      });
    });
  }
}

export default SocketIOServer;