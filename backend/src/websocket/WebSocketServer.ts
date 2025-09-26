import { Server as HTTPServer } from 'http';
import { WebSocket, WebSocketServer as WSServer } from 'ws';
import { EventEmitter } from 'events';
import jwt from 'jsonwebtoken';
import logger from '../utils/logger';
import { TradingEngineService } from '../services/trading/TradingEngineService';
import { StateManager } from '../services/trading/StateManager';
import { WSEvent, WSOrderEvent, WSPositionEvent, WSMarketEvent, WSAlertEvent, Alert } from '../types/trading';

interface AuthenticatedWebSocket extends WebSocket {
  userId?: string;
  isAuthenticated?: boolean;
  subscriptions?: Set<string>;
  pingInterval?: NodeJS.Timeout;
}

interface ClientMessage {
  type: string;
  data?: any;
  token?: string;
}

export class WebSocketServer extends EventEmitter {
  private wss: WSServer;
  private clients: Map<string, Set<AuthenticatedWebSocket>> = new Map();
  private stateManager: StateManager;
  private tradingEngine: TradingEngineService;
  private heartbeatInterval: NodeJS.Timeout | null = null;

  constructor(server: HTTPServer) {
    super();

    // Initialize WebSocket server
    this.wss = new WSServer({
      server,
      path: '/ws',
      clientTracking: true,
      perMessageDeflate: {
        zlibDeflateOptions: {
          chunkSize: 1024,
          memLevel: 7,
          level: 3
        },
        zlibInflateOptions: {
          chunkSize: 10 * 1024
        },
        clientNoContextTakeover: true,
        serverNoContextTakeover: true,
        serverMaxWindowBits: 10,
        concurrencyLimit: 10,
        threshold: 1024
      }
    });

    this.stateManager = new StateManager();
    this.tradingEngine = TradingEngineService.getInstance();

    this.initialize();
  }

  private initialize(): void {
    // Set up connection handler
    this.wss.on('connection', (ws: AuthenticatedWebSocket, req) => {
      this.handleConnection(ws, req);
    });

    // Set up error handler
    this.wss.on('error', (error) => {
      logger.error('WebSocket Server error:', error);
    });

    // Start heartbeat mechanism
    this.startHeartbeat();

    // Subscribe to trading engine events
    this.subscribeToTradingEvents();

    // Subscribe to Redis pub/sub events
    this.subscribeToRedisEvents();

    logger.info('WebSocket Server initialized');
  }

  private handleConnection(ws: AuthenticatedWebSocket, req: any): void {
    const clientIp = req.socket.remoteAddress;
    logger.info(`WebSocket connection from ${clientIp}`);

    // Initialize client state
    ws.isAuthenticated = false;
    ws.subscriptions = new Set();

    // Set up ping/pong for connection health
    ws.on('pong', () => {
      (ws as any).isAlive = true;
    });

    // Handle incoming messages
    ws.on('message', async (data: Buffer) => {
      try {
        const message: ClientMessage = JSON.parse(data.toString());
        await this.handleClientMessage(ws, message);
      } catch (error) {
        logger.error('Error handling WebSocket message:', error);
        this.sendError(ws, 'Invalid message format');
      }
    });

    // Handle disconnection
    ws.on('close', (code, reason) => {
      this.handleDisconnection(ws, code, reason.toString());
    });

    // Handle errors
    ws.on('error', (error) => {
      logger.error('WebSocket client error:', error);
    });

    // Send welcome message
    this.sendMessage(ws, {
      type: 'connection',
      data: {
        status: 'connected',
        message: 'Please authenticate',
        timestamp: new Date()
      }
    });
  }

  private async handleClientMessage(ws: AuthenticatedWebSocket, message: ClientMessage): Promise<void> {
    switch (message.type) {
      case 'auth':
        await this.handleAuthentication(ws, message.data);
        break;

      case 'subscribe':
        if (!ws.isAuthenticated) {
          this.sendError(ws, 'Authentication required');
          return;
        }
        this.handleSubscription(ws, message.data);
        break;

      case 'unsubscribe':
        if (!ws.isAuthenticated) {
          this.sendError(ws, 'Authentication required');
          return;
        }
        this.handleUnsubscription(ws, message.data);
        break;

      case 'ping':
        this.sendMessage(ws, { type: 'pong', data: { timestamp: new Date() } });
        break;

      case 'command':
        if (!ws.isAuthenticated) {
          this.sendError(ws, 'Authentication required');
          return;
        }
        await this.handleCommand(ws, message.data);
        break;

      default:
        this.sendError(ws, `Unknown message type: ${message.type}`);
    }
  }

  private async handleAuthentication(ws: AuthenticatedWebSocket, data: any): Promise<void> {
    try {
      const { token } = data;
      if (!token) {
        this.sendError(ws, 'Token required');
        return;
      }

      // Verify JWT token
      const decoded = jwt.verify(token, process.env.JWT_SECRET || 'secret') as any;
      const userId = decoded.userId || decoded.sub;

      if (!userId) {
        this.sendError(ws, 'Invalid token');
        return;
      }

      // Mark as authenticated
      ws.userId = userId;
      ws.isAuthenticated = true;

      // Add to clients map
      if (!this.clients.has(userId)) {
        this.clients.set(userId, new Set());
      }
      this.clients.get(userId)!.add(ws);

      // Send authentication success
      this.sendMessage(ws, {
        type: 'auth_success',
        data: {
          userId,
          message: 'Authentication successful',
          timestamp: new Date()
        }
      });

      // Auto-subscribe to user's channels
      this.autoSubscribe(ws, userId);

      logger.info(`WebSocket client authenticated: ${userId}`);

    } catch (error) {
      logger.error('Authentication error:', error);
      this.sendError(ws, 'Authentication failed');
    }
  }

  private handleSubscription(ws: AuthenticatedWebSocket, data: any): void {
    const { channels } = data;
    if (!channels || !Array.isArray(channels)) {
      this.sendError(ws, 'Invalid subscription request');
      return;
    }

    for (const channel of channels) {
      ws.subscriptions!.add(channel);
    }

    this.sendMessage(ws, {
      type: 'subscription_success',
      data: {
        channels,
        message: 'Subscribed successfully',
        timestamp: new Date()
      }
    });
  }

  private handleUnsubscription(ws: AuthenticatedWebSocket, data: any): void {
    const { channels } = data;
    if (!channels || !Array.isArray(channels)) {
      this.sendError(ws, 'Invalid unsubscription request');
      return;
    }

    for (const channel of channels) {
      ws.subscriptions!.delete(channel);
    }

    this.sendMessage(ws, {
      type: 'unsubscription_success',
      data: {
        channels,
        message: 'Unsubscribed successfully',
        timestamp: new Date()
      }
    });
  }

  private async handleCommand(ws: AuthenticatedWebSocket, data: any): Promise<void> {
    const { command, params } = data;

    try {
      switch (command) {
        case 'get_positions':
          const positions = await this.tradingEngine.getPositions(ws.userId!);
          this.sendMessage(ws, {
            type: 'positions',
            data: positions
          });
          break;

        case 'get_orders':
          const orders = await this.tradingEngine.getOrders(ws.userId!, params?.status);
          this.sendMessage(ws, {
            type: 'orders',
            data: orders
          });
          break;

        case 'get_performance':
          const performance = await this.tradingEngine.getPerformanceMetrics(ws.userId!, params?.period);
          this.sendMessage(ws, {
            type: 'performance',
            data: performance
          });
          break;

        case 'emergency_stop':
          await this.tradingEngine.stopSession(ws.userId!);
          this.sendMessage(ws, {
            type: 'emergency_stop_success',
            data: { message: 'Trading stopped successfully' }
          });
          break;

        default:
          this.sendError(ws, `Unknown command: ${command}`);
      }
    } catch (error) {
      logger.error(`Command error for ${command}:`, error);
      this.sendError(ws, `Command failed: ${command}`);
    }
  }

  private handleDisconnection(ws: AuthenticatedWebSocket, code: number, reason: string): void {
    logger.info(`WebSocket disconnection: ${ws.userId || 'unauthenticated'}, code: ${code}, reason: ${reason}`);

    // Remove from clients map
    if (ws.userId && this.clients.has(ws.userId)) {
      const userClients = this.clients.get(ws.userId)!;
      userClients.delete(ws);
      if (userClients.size === 0) {
        this.clients.delete(ws.userId);
      }
    }

    // Clear ping interval if exists
    if (ws.pingInterval) {
      clearInterval(ws.pingInterval);
    }
  }

  private autoSubscribe(ws: AuthenticatedWebSocket, userId: string): void {
    // Auto-subscribe to user-specific channels
    const channels = [
      `user:${userId}:orders`,
      `user:${userId}:positions`,
      `user:${userId}:alerts`,
      `user:${userId}:performance`,
      'market:updates',
      'system:announcements'
    ];

    for (const channel of channels) {
      ws.subscriptions!.add(channel);
    }

    logger.debug(`Auto-subscribed user ${userId} to channels:`, channels);
  }

  private subscribeToTradingEvents(): void {
    // Subscribe to trading engine events
    this.tradingEngine.on('orderExecuted', (data) => {
      this.handleOrderEvent(data);
    });

    this.tradingEngine.on('positionOpened', (data) => {
      this.handlePositionEvent(data, 'opened');
    });

    this.tradingEngine.on('positionClosed', (data) => {
      this.handlePositionEvent(data, 'closed');
    });

    this.tradingEngine.on('tradingCycleComplete', (data) => {
      this.handleTradingCycleEvent(data);
    });

    this.tradingEngine.on('riskLimitExceeded', (data) => {
      this.handleRiskAlert(data);
    });

    this.tradingEngine.on('emergencyStop', (data) => {
      this.handleEmergencyStop(data);
    });
  }

  private async subscribeToRedisEvents(): Promise<void> {
    // This will be called when StateManager publishes events
    // Implementation depends on Redis pub/sub setup
  }

  private handleOrderEvent(data: any): void {
    const { userId, order, signal } = data;

    const event: WSOrderEvent = {
      type: 'order_update',
      data: {
        order,
        action: order.status === 'filled' ? 'filled' : 'created'
      },
      timestamp: new Date()
    };

    this.broadcastToUser(userId, `user:${userId}:orders`, event);
  }

  private handlePositionEvent(data: any, action: 'opened' | 'updated' | 'closed'): void {
    const position = data.position || data;
    const userId = position.userId;

    const event: WSPositionEvent = {
      type: 'position_update',
      data: {
        position,
        action
      },
      timestamp: new Date()
    };

    this.broadcastToUser(userId, `user:${userId}:positions`, event);
  }

  private handleTradingCycleEvent(data: any): void {
    const { userId, signalsEvaluated, tradesExecuted } = data;

    const event: WSEvent = {
      type: 'trading_cycle',
      data: {
        signalsEvaluated,
        tradesExecuted,
        timestamp: new Date()
      },
      timestamp: new Date()
    };

    this.broadcastToUser(userId, `user:${userId}:performance`, event);
  }

  private handleRiskAlert(data: any): void {
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

    const event: WSAlertEvent = {
      type: 'alert',
      data: alert,
      timestamp: new Date()
    };

    this.broadcastToUser(userId, `user:${userId}:alerts`, event);
  }

  private handleEmergencyStop(userId: string): void {
    const alert: Alert = {
      id: `alert_${Date.now()}`,
      userId,
      type: 'system',
      severity: 'critical',
      title: 'Emergency Stop Activated',
      message: 'Trading has been stopped due to emergency conditions',
      timestamp: new Date()
    };

    const event: WSAlertEvent = {
      type: 'alert',
      data: alert,
      timestamp: new Date()
    };

    this.broadcastToUser(userId, `user:${userId}:alerts`, event);
  }

  // Public methods for external use
  public broadcastToUser(userId: string, channel: string, data: any): void {
    const userClients = this.clients.get(userId);
    if (!userClients) return;

    for (const client of userClients) {
      if (client.subscriptions?.has(channel) && client.readyState === WebSocket.OPEN) {
        this.sendMessage(client, data);
      }
    }
  }

  public broadcastToAll(channel: string, data: any): void {
    this.wss.clients.forEach((client) => {
      const ws = client as AuthenticatedWebSocket;
      if (ws.subscriptions?.has(channel) && ws.readyState === WebSocket.OPEN) {
        this.sendMessage(ws, data);
      }
    });
  }

  public sendToUser(userId: string, event: string, data: any): void {
    const userClients = this.clients.get(userId);
    if (!userClients) return;

    const message = {
      type: event,
      data,
      timestamp: new Date()
    };

    for (const client of userClients) {
      if (client.readyState === WebSocket.OPEN) {
        this.sendMessage(client, message);
      }
    }
  }

  public broadcast(event: string, data: any): void {
    const message = {
      type: event,
      data,
      timestamp: new Date()
    };

    this.wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        this.sendMessage(client as AuthenticatedWebSocket, message);
      }
    });
  }

  private sendMessage(ws: AuthenticatedWebSocket, message: any): void {
    try {
      ws.send(JSON.stringify(message));
    } catch (error) {
      logger.error('Error sending WebSocket message:', error);
    }
  }

  private sendError(ws: AuthenticatedWebSocket, error: string): void {
    this.sendMessage(ws, {
      type: 'error',
      data: {
        message: error,
        timestamp: new Date()
      }
    });
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      this.wss.clients.forEach((ws) => {
        const client = ws as any;
        if (client.isAlive === false) {
          logger.debug('Terminating inactive WebSocket connection');
          return ws.terminate();
        }

        client.isAlive = false;
        ws.ping();
      });
    }, 30000); // 30 seconds
  }

  public async shutdown(): Promise<void> {
    logger.info('Shutting down WebSocket server');

    // Clear heartbeat interval
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    // Close all client connections
    this.wss.clients.forEach((client) => {
      client.close(1000, 'Server shutting down');
    });

    // Close the server
    await new Promise<void>((resolve) => {
      this.wss.close(() => {
        logger.info('WebSocket server closed');
        resolve();
      });
    });
  }

  // Metrics and monitoring
  public getMetrics(): any {
    const metrics = {
      totalConnections: this.wss.clients.size,
      authenticatedConnections: this.clients.size,
      connectionsByUser: {} as Record<string, number>
    };

    for (const [userId, clients] of this.clients) {
      metrics.connectionsByUser[userId] = clients.size;
    }

    return metrics;
  }
}

export default WebSocketServer;