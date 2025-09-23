export class ServiceWorkerManager {
  private registration: ServiceWorkerRegistration | null = null;
  private isSupported: boolean = false;

  constructor() {
    this.isSupported = 'serviceWorker' in navigator;
  }

  async register(): Promise<boolean> {
    if (!this.isSupported) {
      console.log('Service Worker not supported');
      return false;
    }

    try {
      // Register service worker
      this.registration = await navigator.serviceWorker.register('/service-worker.js', {
        scope: '/'
      });

      console.log('Service Worker registered:', this.registration);

      // Check for updates
      this.registration.addEventListener('updatefound', () => {
        const newWorker = this.registration!.installing;
        if (newWorker) {
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              // New service worker available
              this.notifyUpdate();
            }
          });
        }
      });

      // Handle messages from service worker
      navigator.serviceWorker.addEventListener('message', this.handleMessage.bind(this));

      // Start background sync
      this.startBackgroundSync();

      // Request notification permission
      await this.requestNotificationPermission();

      return true;
    } catch (error) {
      console.error('Service Worker registration failed:', error);
      return false;
    }
  }

  async unregister(): Promise<boolean> {
    if (!this.registration) {
      return false;
    }

    try {
      const success = await this.registration.unregister();
      if (success) {
        this.registration = null;
        console.log('Service Worker unregistered');
      }
      return success;
    } catch (error) {
      console.error('Service Worker unregistration failed:', error);
      return false;
    }
  }

  private handleMessage(event: MessageEvent): void {
    const { type, data } = event.data;

    switch (type) {
      case 'PRICE_UPDATE':
        this.handlePriceUpdate(data);
        break;
      case 'TRADE_EXECUTED':
        this.handleTradeExecuted(data);
        break;
      case 'AGENT_STATUS':
        this.handleAgentStatus(data);
        break;
      default:
        console.log('Unknown message from service worker:', type);
    }
  }

  private handlePriceUpdate(prices: any): void {
    // Dispatch custom event for price updates
    window.dispatchEvent(new CustomEvent('sw-price-update', {
      detail: prices
    }));
  }

  private handleTradeExecuted(trade: any): void {
    // Dispatch custom event for trade execution
    window.dispatchEvent(new CustomEvent('sw-trade-executed', {
      detail: trade
    }));

    // Prevent auto-scroll by saving current position
    const scrollPosition = window.scrollY;
    
    // After DOM update, restore scroll position
    requestAnimationFrame(() => {
      window.scrollTo(0, scrollPosition);
    });
  }

  private handleAgentStatus(status: any): void {
    // Dispatch custom event for agent status
    window.dispatchEvent(new CustomEvent('sw-agent-status', {
      detail: status
    }));
  }

  private notifyUpdate(): void {
    // Notify user of available update
    const shouldUpdate = confirm('A new version of the app is available. Reload to update?');
    if (shouldUpdate) {
      window.location.reload();
    }
  }

  private async requestNotificationPermission(): Promise<void> {
    if (!('Notification' in window)) {
      console.log('Notifications not supported');
      return;
    }

    if (Notification.permission === 'default') {
      const permission = await Notification.requestPermission();
      console.log('Notification permission:', permission);
    }
  }

  private startBackgroundSync(): void {
    if (!this.registration) return;

    // Send message to service worker to start sync
    if (navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'START_BACKGROUND_SYNC'
      });
    }

    // Register periodic sync if supported
    if ('periodicSync' in this.registration) {
      this.registerPeriodicSync();
    }
  }

  private async registerPeriodicSync(): Promise<void> {
    if (!this.registration) return;

    try {
      // @ts-ignore - periodicSync is experimental
      await this.registration.periodicSync.register('update-prices', {
        minInterval: 60 * 1000 // 1 minute
      });

      // @ts-ignore
      await this.registration.periodicSync.register('check-agent-status', {
        minInterval: 5 * 60 * 1000 // 5 minutes
      });

      console.log('Periodic sync registered');
    } catch (error) {
      console.log('Periodic sync not available:', error);
    }
  }

  async syncInBackground(tag: string): Promise<void> {
    if (!this.registration || !('sync' in this.registration)) {
      console.log('Background sync not supported');
      return;
    }

    try {
      // @ts-ignore - sync is experimental
      await this.registration.sync.register(tag);
      console.log(`Background sync registered: ${tag}`);
    } catch (error) {
      console.error('Background sync registration failed:', error);
    }
  }

  sendMessage(message: any): void {
    if (navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage(message);
    }
  }

  async checkForUpdates(): Promise<void> {
    if (this.registration) {
      await this.registration.update();
    }
  }
}

// Export singleton instance
export const serviceWorkerManager = new ServiceWorkerManager();

// Auto-register on load
if (typeof window !== 'undefined') {
  window.addEventListener('load', () => {
    serviceWorkerManager.register();
  });
}