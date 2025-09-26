/**
 * Unified Monitoring Service
 * Consolidates monitoring from RL, ML, and Production implementations
 * Provides a single interface for all monitoring needs
 */

import { EventEmitter } from 'events';
import * as os from 'os';
import logger from '../../utils/logger';

export interface MetricPoint {
  name: string;
  value: number;
  timestamp: Date;
  labels?: Record<string, string>;
  type: 'counter' | 'gauge' | 'histogram';
}

export interface Alert {
  id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  timestamp: Date;
  source: string;
  metadata?: any;
}

export interface HealthCheck {
  service: string;
  status: 'healthy' | 'unhealthy' | 'degraded';
  latency: number;
  lastCheck: Date;
  details?: any;
}

export class UnifiedMonitoringService extends EventEmitter {
  private static instance: UnifiedMonitoringService;
  private metrics: Map<string, MetricPoint[]> = new Map();
  private alerts: Alert[] = [];
  private healthChecks: Map<string, HealthCheck> = new Map();
  private metricsBuffer: MetricPoint[] = [];
  private flushInterval: NodeJS.Timer | null = null;
  private alertThresholds: Map<string, { min?: number; max?: number }> = new Map();

  private constructor() {
    super();
    this.setupDefaultThresholds();
    this.startMetricsFlush();
  }

  static getInstance(): UnifiedMonitoringService {
    if (!UnifiedMonitoringService.instance) {
      UnifiedMonitoringService.instance = new UnifiedMonitoringService();
    }
    return UnifiedMonitoringService.instance;
  }

  /**
   * Initialize monitoring service
   */
  async initialize(): Promise<void> {
    logger.info('Initializing Unified Monitoring Service');

    // Start system metrics collection
    this.startSystemMetricsCollection();

    // Setup alert thresholds
    this.setupAlertThresholds();

    logger.info('Monitoring Service initialized');
  }

  /**
   * Record a metric
   */
  recordMetric(
    name: string,
    value: number,
    type: 'counter' | 'gauge' | 'histogram' = 'gauge',
    labels?: Record<string, string>
  ): void {
    const metric: MetricPoint = {
      name,
      value,
      type,
      timestamp: new Date(),
      labels
    };

    this.metricsBuffer.push(metric);

    // Check thresholds
    this.checkThresholds(name, value);

    // Emit for real-time monitoring
    this.emit('metric', metric);
  }

  /**
   * Record trading-specific metrics
   */
  recordTradingMetrics(data: {
    ordersPlaced?: number;
    ordersExecuted?: number;
    ordersFailed?: number;
    positionsOpened?: number;
    positionsClosed?: number;
    pnl?: number;
    winRate?: number;
    sharpeRatio?: number;
    maxDrawdown?: number;
  }): void {
    Object.entries(data).forEach(([key, value]) => {
      if (value !== undefined) {
        this.recordMetric(`trading.${key}`, value);
      }
    });
  }

  /**
   * Record API metrics
   */
  recordApiMetrics(data: {
    endpoint: string;
    method: string;
    statusCode: number;
    responseTime: number;
    userId?: string;
  }): void {
    this.recordMetric('api.request.count', 1, 'counter', {
      endpoint: data.endpoint,
      method: data.method,
      status: String(data.statusCode)
    });

    this.recordMetric('api.response.time', data.responseTime, 'histogram', {
      endpoint: data.endpoint,
      method: data.method
    });

    if (data.statusCode >= 400) {
      this.recordMetric('api.error.count', 1, 'counter', {
        endpoint: data.endpoint,
        status: String(data.statusCode)
      });
    }
  }

  /**
   * Create an alert
   */
  createAlert(
    severity: Alert['severity'],
    title: string,
    message: string,
    source: string,
    metadata?: any
  ): void {
    const alert: Alert = {
      id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      severity,
      title,
      message,
      timestamp: new Date(),
      source,
      metadata
    };

    this.alerts.push(alert);

    // Keep only last 1000 alerts
    if (this.alerts.length > 1000) {
      this.alerts = this.alerts.slice(-1000);
    }

    // Log alert
    const logMethod = severity === 'critical' || severity === 'error' ? 'error' :
                     severity === 'warning' ? 'warn' : 'info';
    logger[logMethod](`[ALERT] ${title}: ${message}`, metadata);

    // Emit alert
    this.emit('alert', alert);

    // Critical alerts trigger emergency actions
    if (severity === 'critical') {
      this.handleCriticalAlert(alert);
    }
  }

  /**
   * Record health check
   */
  recordHealthCheck(
    service: string,
    status: HealthCheck['status'],
    latency: number,
    details?: any
  ): void {
    const healthCheck: HealthCheck = {
      service,
      status,
      latency,
      lastCheck: new Date(),
      details
    };

    this.healthChecks.set(service, healthCheck);

    // Alert on unhealthy services
    if (status === 'unhealthy') {
      this.createAlert(
        'error',
        `Service Unhealthy: ${service}`,
        `Service ${service} is reporting unhealthy status`,
        'health_check',
        details
      );
    }
  }

  /**
   * Get current metrics
   */
  getMetrics(name?: string, period?: number): MetricPoint[] {
    if (!name) {
      // Return all recent metrics
      const allMetrics: MetricPoint[] = [];
      this.metrics.forEach(points => {
        allMetrics.push(...points);
      });
      return this.filterByPeriod(allMetrics, period);
    }

    const metrics = this.metrics.get(name) || [];
    return this.filterByPeriod(metrics, period);
  }

  /**
   * Get alerts
   */
  getAlerts(severity?: Alert['severity'], limit: number = 100): Alert[] {
    let alerts = [...this.alerts].reverse(); // Most recent first

    if (severity) {
      alerts = alerts.filter(a => a.severity === severity);
    }

    return alerts.slice(0, limit);
  }

  /**
   * Get health status
   */
  getHealthStatus(): {
    overall: 'healthy' | 'degraded' | 'unhealthy';
    services: HealthCheck[];
    uptime: number;
    systemMetrics: any;
  } {
    const services = Array.from(this.healthChecks.values());
    const unhealthyCount = services.filter(s => s.status === 'unhealthy').length;
    const degradedCount = services.filter(s => s.status === 'degraded').length;

    let overall: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    if (unhealthyCount > 0) overall = 'unhealthy';
    else if (degradedCount > 0) overall = 'degraded';

    return {
      overall,
      services,
      uptime: process.uptime(),
      systemMetrics: this.getSystemMetrics()
    };
  }

  /**
   * Get dashboard data
   */
  getDashboardData(): {
    metrics: Record<string, number>;
    alerts: Alert[];
    health: ReturnType<typeof this.getHealthStatus>;
    recentActivity: any[];
  } {
    // Aggregate recent metrics
    const aggregatedMetrics: Record<string, number> = {};
    const recentMetrics = this.getMetrics(undefined, 300); // Last 5 minutes

    recentMetrics.forEach(metric => {
      if (metric.type === 'counter') {
        aggregatedMetrics[metric.name] = (aggregatedMetrics[metric.name] || 0) + metric.value;
      } else {
        aggregatedMetrics[metric.name] = metric.value; // Take latest for gauges
      }
    });

    return {
      metrics: aggregatedMetrics,
      alerts: this.getAlerts(undefined, 10),
      health: this.getHealthStatus(),
      recentActivity: this.getRecentActivity()
    };
  }

  /**
   * Setup default alert thresholds
   */
  private setupDefaultThresholds(): void {
    // API thresholds
    this.alertThresholds.set('api.response.time', { max: 5000 }); // 5 seconds
    this.alertThresholds.set('api.error.count', { max: 100 }); // per minute

    // Trading thresholds
    this.alertThresholds.set('trading.ordersFailed', { max: 10 }); // per minute
    this.alertThresholds.set('trading.maxDrawdown', { max: 0.20 }); // 20%
    this.alertThresholds.set('trading.pnl', { min: -1000 }); // -$1000

    // System thresholds
    this.alertThresholds.set('system.cpu.usage', { max: 90 }); // 90%
    this.alertThresholds.set('system.memory.usage', { max: 85 }); // 85%
  }

  /**
   * Setup custom alert thresholds
   */
  private setupAlertThresholds(): void {
    // Load from environment or config
    const customThresholds = process.env.MONITOR_THRESHOLDS;
    if (customThresholds) {
      try {
        const thresholds = JSON.parse(customThresholds);
        Object.entries(thresholds).forEach(([metric, limits]: [string, any]) => {
          this.alertThresholds.set(metric, limits);
        });
      } catch (error) {
        logger.error('Failed to parse custom thresholds:', error);
      }
    }
  }

  /**
   * Check if metric exceeds thresholds
   */
  private checkThresholds(name: string, value: number): void {
    const threshold = this.alertThresholds.get(name);
    if (!threshold) return;

    if (threshold.max !== undefined && value > threshold.max) {
      this.createAlert(
        'warning',
        `Threshold Exceeded: ${name}`,
        `Metric ${name} (${value}) exceeds maximum threshold (${threshold.max})`,
        'threshold_monitor'
      );
    }

    if (threshold.min !== undefined && value < threshold.min) {
      this.createAlert(
        'warning',
        `Threshold Breached: ${name}`,
        `Metric ${name} (${value}) below minimum threshold (${threshold.min})`,
        'threshold_monitor'
      );
    }
  }

  /**
   * Start system metrics collection
   */
  private startSystemMetricsCollection(): void {
    setInterval(() => {
      const cpus = os.cpus();
      const totalMemory = os.totalmem();
      const freeMemory = os.freemem();
      const usedMemory = totalMemory - freeMemory;

      // CPU usage
      const cpuUsage = cpus.reduce((acc, cpu) => {
        const total = Object.values(cpu.times).reduce((a, b) => a + b, 0);
        const idle = cpu.times.idle;
        return acc + ((total - idle) / total) * 100;
      }, 0) / cpus.length;

      this.recordMetric('system.cpu.usage', cpuUsage, 'gauge');
      this.recordMetric('system.cpu.count', cpus.length, 'gauge');

      // Memory usage
      this.recordMetric('system.memory.used', usedMemory, 'gauge');
      this.recordMetric('system.memory.free', freeMemory, 'gauge');
      this.recordMetric('system.memory.usage', (usedMemory / totalMemory) * 100, 'gauge');

      // Process metrics
      const memUsage = process.memoryUsage();
      this.recordMetric('process.memory.heap.used', memUsage.heapUsed, 'gauge');
      this.recordMetric('process.memory.heap.total', memUsage.heapTotal, 'gauge');
      this.recordMetric('process.memory.rss', memUsage.rss, 'gauge');
      this.recordMetric('process.uptime', process.uptime(), 'gauge');

    }, 30000); // Every 30 seconds
  }

  /**
   * Start metrics flush interval
   */
  private startMetricsFlush(): void {
    this.flushInterval = setInterval(() => {
      this.flushMetrics();
    }, 10000); // Flush every 10 seconds
  }

  /**
   * Flush metrics buffer
   */
  private flushMetrics(): void {
    if (this.metricsBuffer.length === 0) return;

    // Group metrics by name
    this.metricsBuffer.forEach(metric => {
      if (!this.metrics.has(metric.name)) {
        this.metrics.set(metric.name, []);
      }

      const points = this.metrics.get(metric.name)!;
      points.push(metric);

      // Keep only last hour of data points
      const oneHourAgo = new Date(Date.now() - 3600000);
      this.metrics.set(
        metric.name,
        points.filter(p => p.timestamp > oneHourAgo)
      );
    });

    // Clear buffer
    this.metricsBuffer = [];
  }

  /**
   * Filter metrics by period
   */
  private filterByPeriod(metrics: MetricPoint[], periodSeconds?: number): MetricPoint[] {
    if (!periodSeconds) return metrics;

    const cutoff = new Date(Date.now() - periodSeconds * 1000);
    return metrics.filter(m => m.timestamp > cutoff);
  }

  /**
   * Get system metrics
   */
  private getSystemMetrics(): any {
    const cpus = os.cpus();
    const totalMemory = os.totalmem();
    const freeMemory = os.freemem();

    return {
      cpu: {
        count: cpus.length,
        model: cpus[0]?.model,
        usage: this.getLatestMetricValue('system.cpu.usage')
      },
      memory: {
        total: totalMemory,
        free: freeMemory,
        used: totalMemory - freeMemory,
        usage: this.getLatestMetricValue('system.memory.usage')
      },
      process: {
        uptime: process.uptime(),
        pid: process.pid,
        memoryUsage: process.memoryUsage()
      },
      os: {
        platform: os.platform(),
        release: os.release(),
        uptime: os.uptime()
      }
    };
  }

  /**
   * Get latest value for a metric
   */
  private getLatestMetricValue(name: string): number | null {
    const points = this.metrics.get(name);
    if (!points || points.length === 0) return null;
    return points[points.length - 1].value;
  }

  /**
   * Get recent activity
   */
  private getRecentActivity(): any[] {
    const activities: any[] = [];

    // Add recent alerts
    this.alerts.slice(-5).forEach(alert => {
      activities.push({
        type: 'alert',
        timestamp: alert.timestamp,
        data: alert
      });
    });

    // Add recent health check changes
    this.healthChecks.forEach(health => {
      if (health.status !== 'healthy') {
        activities.push({
          type: 'health',
          timestamp: health.lastCheck,
          data: health
        });
      }
    });

    // Sort by timestamp
    return activities.sort((a, b) =>
      b.timestamp.getTime() - a.timestamp.getTime()
    ).slice(0, 10);
  }

  /**
   * Handle critical alerts
   */
  private handleCriticalAlert(alert: Alert): void {
    logger.error('CRITICAL ALERT - Triggering emergency procedures', alert);

    // Emit critical event for other services to handle
    this.emit('critical_alert', alert);

    // Could trigger:
    // - Emergency stop of trading
    // - Notification to administrators
    // - Automatic scaling/recovery procedures
  }

  /**
   * Cleanup
   */
  async shutdown(): Promise<void> {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }

    // Final flush
    this.flushMetrics();

    logger.info('Monitoring service shut down');
  }
}

export default UnifiedMonitoringService.getInstance();