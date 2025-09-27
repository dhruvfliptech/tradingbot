/**
 * Keepalive system for Heroku Eco dynos
 * Prevents sleeping during trading hours
 */

import axios from 'axios';
import logger from './logger';

export class KeepaliveService {
  private interval: NodeJS.Timeout | null = null;
  private readonly APP_URL = process.env.APP_URL || 'http://localhost:3001';

  start(): void {
    // Only needed for Heroku Eco dynos
    if (process.env.DYNO_TYPE !== 'eco') {
      logger.info('Not on Eco dyno, keepalive not needed');
      return;
    }

    // Ping every 25 minutes to prevent 30-minute sleep
    this.interval = setInterval(async () => {
      try {
        await axios.get(`${this.APP_URL}/health`);
        logger.debug('Keepalive ping sent');
      } catch (error) {
        logger.error('Keepalive ping failed:', error);
      }
    }, 25 * 60 * 1000);

    // Also set up external monitoring
    this.setupExternalMonitoring();
  }

  private setupExternalMonitoring(): void {
    // Register with UptimeRobot or similar
    logger.info(`
      IMPORTANT: Set up external monitoring to prevent sleep:
      1. Go to https://uptimerobot.com
      2. Add monitor for: ${this.APP_URL}/health
      3. Set interval: 20 minutes
      4. This prevents Eco dyno from sleeping
    `);
  }

  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }
}

export const keepaliveService = new KeepaliveService();