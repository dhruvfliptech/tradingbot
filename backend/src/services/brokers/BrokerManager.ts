/**
 * Broker Manager - Manages multiple broker instances and routing
 */

import { BrokerService } from './BrokerService';
import { AlpacaBroker } from './AlpacaBroker';
import { BinanceBroker } from './BinanceBroker';
import { secretsManager } from '../../config/secrets';
import logger from '../../utils/logger';

export type BrokerType = 'alpaca' | 'binance';

export class BrokerManager {
  private static instance: BrokerManager;
  private brokers: Map<BrokerType, BrokerService> = new Map();
  private activeBroker: BrokerType = 'binance';

  private constructor() {
    this.initializeBrokers();
  }

  static getInstance(): BrokerManager {
    if (!BrokerManager.instance) {
      BrokerManager.instance = new BrokerManager();
    }
    return BrokerManager.instance;
  }

  private initializeBrokers(): void {
    // Initialize Alpaca Broker
    const alpacaCredentials = secretsManager.getBrokerCredentials('alpaca');
    if (alpacaCredentials) {
      this.brokers.set('alpaca', new AlpacaBroker(alpacaCredentials));
      logger.info('Alpaca broker initialized with secure credentials');
    } else {
      logger.warn('Alpaca broker not configured - missing API keys');
    }

    // Initialize Binance Broker
    const binanceCredentials = secretsManager.getBrokerCredentials('binance');
    if (binanceCredentials) {
      this.brokers.set('binance', new BinanceBroker(binanceCredentials));
      logger.info('Binance broker initialized with secure credentials');
    } else {
      logger.warn('Binance broker not configured - missing API keys');
    }

    // Set default broker
    const defaultBroker = process.env.DEFAULT_BROKER as BrokerType;
    if (defaultBroker && this.brokers.has(defaultBroker)) {
      this.activeBroker = defaultBroker;
    } else if (this.brokers.has('binance')) {
      this.activeBroker = 'binance';
    } else if (this.brokers.has('alpaca')) {
      this.activeBroker = 'alpaca';
    }

    logger.info(`Active broker set to: ${this.activeBroker}`);
  }

  /**
   * Get a specific broker instance
   */
  getBroker(type: BrokerType): BrokerService | undefined {
    return this.brokers.get(type);
  }

  /**
   * Get the active broker
   */
  getActiveBroker(): BrokerService {
    const broker = this.brokers.get(this.activeBroker);
    if (!broker) {
      throw new Error(`Active broker ${this.activeBroker} not initialized`);
    }
    return broker;
  }

  /**
   * Set the active broker
   */
  setActiveBroker(type: BrokerType): void {
    if (!this.brokers.has(type)) {
      throw new Error(`Broker ${type} not available`);
    }
    this.activeBroker = type;
    logger.info(`Active broker changed to: ${type}`);
  }

  /**
   * Get list of available brokers
   */
  getAvailableBrokers(): Array<{
    type: BrokerType;
    connected: boolean;
    name: string;
  }> {
    const available: Array<{ type: BrokerType; connected: boolean; name: string }> = [];

    for (const [type, broker] of this.brokers) {
      available.push({
        type,
        connected: broker.isConnected(),
        name: broker.getBrokerName()
      });
    }

    return available;
  }

  /**
   * Test all broker connections
   */
  async testAllConnections(): Promise<Map<BrokerType, boolean>> {
    const results = new Map<BrokerType, boolean>();

    for (const [type, broker] of this.brokers) {
      try {
        const connected = await broker.testConnection();
        results.set(type, connected);
        logger.info(`Broker ${type} connection test: ${connected ? 'SUCCESS' : 'FAILED'}`);
      } catch (error) {
        results.set(type, false);
        logger.error(`Broker ${type} connection test error:`, error);
      }
    }

    return results;
  }

  /**
   * Initialize broker with new credentials
   */
  async initializeBrokerWithCredentials(
    type: BrokerType,
    apiKey: string,
    secretKey: string,
    baseUrl?: string
  ): Promise<boolean> {
    try {
      const credentials = { apiKey, secretKey, baseUrl };

      // Update credentials in SecretsManager
      secretsManager.updateBrokerCredentials(type, credentials);

      let broker: BrokerService;
      if (type === 'alpaca') {
        broker = new AlpacaBroker(credentials);
      } else if (type === 'binance') {
        broker = new BinanceBroker(credentials);
      } else {
        throw new Error(`Unknown broker type: ${type}`);
      }

      // Test connection
      const connected = await broker.testConnection();
      if (connected) {
        // Replace the broker instance
        this.brokers.set(type, broker);
        logger.info(`Broker ${type} initialized with new credentials and stored securely`);
        return true;
      } else {
        logger.warn(`Broker ${type} failed connection test with new credentials`);
        return false;
      }
    } catch (error) {
      logger.error(`Failed to initialize broker ${type}:`, error);
      return false;
    }
  }

  /**
   * Get aggregated account data from all connected brokers
   */
  async getAggregatedAccount(): Promise<{
    total_portfolio_value: number;
    total_available_balance: number;
    accounts: Array<{
      broker: BrokerType;
      account: any;
    }>;
  }> {
    const accounts: Array<{ broker: BrokerType; account: any }> = [];
    let totalPortfolioValue = 0;
    let totalAvailableBalance = 0;

    for (const [type, broker] of this.brokers) {
      try {
        const account = await broker.getAccount();
        accounts.push({ broker: type, account });
        totalPortfolioValue += account.portfolio_value;
        totalAvailableBalance += account.available_balance;
      } catch (error) {
        logger.warn(`Failed to get account from broker ${type}:`, error);
      }
    }

    return {
      total_portfolio_value: totalPortfolioValue,
      total_available_balance: totalAvailableBalance,
      accounts
    };
  }
}