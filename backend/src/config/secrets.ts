/**
 * Secrets Configuration
 * Secure management of API keys and sensitive configuration
 */

import * as dotenv from 'dotenv';
import logger from '../utils/logger';

dotenv.config();

export interface BrokerCredentials {
  apiKey: string;
  secretKey: string;
  baseUrl?: string;
}

export class SecretsManager {
  private static instance: SecretsManager;
  private secrets: Map<string, any> = new Map();

  private constructor() {
    this.loadSecrets();
  }

  static getInstance(): SecretsManager {
    if (!SecretsManager.instance) {
      SecretsManager.instance = new SecretsManager();
    }
    return SecretsManager.instance;
  }

  private loadSecrets(): void {
    // Load broker credentials from environment
    this.loadBrokerCredentials('alpaca', {
      apiKey: process.env.ALPACA_API_KEY,
      secretKey: process.env.ALPACA_SECRET_KEY,
      baseUrl: process.env.ALPACA_BASE_URL
    });

    this.loadBrokerCredentials('binance', {
      apiKey: process.env.BINANCE_API_KEY,
      secretKey: process.env.BINANCE_SECRET_KEY,
      baseUrl: process.env.BINANCE_BASE_URL
    });

    // Load other secrets
    this.secrets.set('supabase', {
      url: process.env.SUPABASE_URL,
      anonKey: process.env.SUPABASE_ANON_KEY,
      serviceKey: process.env.SUPABASE_SERVICE_KEY
    });

    this.secrets.set('redis', {
      host: process.env.REDIS_HOST,
      port: process.env.REDIS_PORT,
      password: process.env.REDIS_PASSWORD
    });

    this.secrets.set('jwt', {
      secret: process.env.JWT_SECRET || 'your-jwt-secret-key',
      expiresIn: process.env.JWT_EXPIRES_IN || '24h'
    });

    logger.info('Secrets loaded from environment');
  }

  private loadBrokerCredentials(broker: string, credentials: any): void {
    if (credentials.apiKey && credentials.secretKey) {
      this.secrets.set(`broker.${broker}`, {
        apiKey: credentials.apiKey,
        secretKey: credentials.secretKey,
        baseUrl: credentials.baseUrl
      });
      logger.info(`${broker} broker credentials loaded`);
    } else {
      logger.warn(`${broker} broker credentials not found in environment`);
    }
  }

  /**
   * Get broker credentials
   */
  getBrokerCredentials(broker: 'alpaca' | 'binance'): BrokerCredentials | null {
    return this.secrets.get(`broker.${broker}`) || null;
  }

  /**
   * Update broker credentials (runtime update)
   */
  updateBrokerCredentials(
    broker: 'alpaca' | 'binance',
    credentials: BrokerCredentials
  ): void {
    this.secrets.set(`broker.${broker}`, credentials);
    logger.info(`${broker} broker credentials updated`);
  }

  /**
   * Get Supabase configuration
   */
  getSupabaseConfig(): any {
    return this.secrets.get('supabase');
  }

  /**
   * Get Redis configuration
   */
  getRedisConfig(): any {
    return this.secrets.get('redis');
  }

  /**
   * Get JWT configuration
   */
  getJWTConfig(): any {
    return this.secrets.get('jwt');
  }

  /**
   * Validate all required secrets are present
   */
  validateSecrets(): boolean {
    const required = [
      'broker.alpaca',
      'broker.binance',
      'supabase',
      'jwt'
    ];

    const missing = required.filter(key => !this.secrets.has(key));

    if (missing.length > 0) {
      logger.warn(`Missing secrets: ${missing.join(', ')}`);
      return false;
    }

    return true;
  }

  /**
   * Clear sensitive data from memory
   */
  clearSecrets(): void {
    this.secrets.clear();
    logger.info('Secrets cleared from memory');
  }
}

// Export singleton instance
export const secretsManager = SecretsManager.getInstance();