// Service Worker for Trading Bot Background Processing
const CACHE_NAME = 'trading-bot-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/manifest.json'
];

// Install event - cache assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Service Worker: Caching files');
        return cache.addAll(urlsToCache);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('Service Worker: Clearing old cache');
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event - serve from cache or network
self.addEventListener('fetch', event => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') return;
  
  // Skip API requests
  if (event.request.url.includes('/api/') || 
      event.request.url.includes('supabase') ||
      event.request.url.includes('alpaca') ||
      event.request.url.includes('coingecko')) {
    return;
  }
  
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Cache hit - return response
        if (response) {
          return response;
        }
        
        // Clone the request
        const fetchRequest = event.request.clone();
        
        return fetch(fetchRequest).then(response => {
          // Check if valid response
          if (!response || response.status !== 200 || response.type !== 'basic') {
            return response;
          }
          
          // Clone the response
          const responseToCache = response.clone();
          
          caches.open(CACHE_NAME)
            .then(cache => {
              cache.put(event.request, responseToCache);
            });
          
          return response;
        });
      })
  );
});

// Background sync for trading operations
self.addEventListener('sync', event => {
  if (event.tag === 'sync-trades') {
    event.waitUntil(syncTrades());
  } else if (event.tag === 'sync-prices') {
    event.waitUntil(syncPrices());
  }
});

// Periodic background sync (if supported)
self.addEventListener('periodicsync', event => {
  if (event.tag === 'update-prices') {
    event.waitUntil(updatePrices());
  } else if (event.tag === 'check-agent-status') {
    event.waitUntil(checkAgentStatus());
  }
});

// Message handling from main thread
self.addEventListener('message', event => {
  const { type, data } = event.data;
  
  switch (type) {
    case 'START_BACKGROUND_SYNC':
      startBackgroundSync();
      break;
    case 'STOP_BACKGROUND_SYNC':
      stopBackgroundSync();
      break;
    case 'EXECUTE_TRADE':
      executeTradeInBackground(data);
      break;
    case 'UPDATE_CACHE':
      updateCache(data);
      break;
    default:
      console.log('Unknown message type:', type);
  }
});

// Background sync functions
async function syncTrades() {
  try {
    // In production, this would sync pending trades with the server
    const response = await fetch('/api/trades/sync', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ 
        timestamp: Date.now() 
      })
    });
    
    if (response.ok) {
      console.log('Trades synced successfully');
    }
  } catch (error) {
    console.error('Trade sync failed:', error);
  }
}

async function syncPrices() {
  try {
    // Fetch latest prices and notify clients
    const response = await fetch('/api/prices/latest');
    if (response.ok) {
      const prices = await response.json();
      
      // Notify all clients
      self.clients.matchAll().then(clients => {
        clients.forEach(client => {
          client.postMessage({
            type: 'PRICE_UPDATE',
            data: prices
          });
        });
      });
    }
  } catch (error) {
    console.error('Price sync failed:', error);
  }
}

async function updatePrices() {
  // Similar to syncPrices but for periodic updates
  await syncPrices();
}

async function checkAgentStatus() {
  try {
    const response = await fetch('/api/agent/status');
    if (response.ok) {
      const status = await response.json();
      
      // Notify clients of status change
      self.clients.matchAll().then(clients => {
        clients.forEach(client => {
          client.postMessage({
            type: 'AGENT_STATUS',
            data: status
          });
        });
      });
    }
  } catch (error) {
    console.error('Agent status check failed:', error);
  }
}

async function executeTradeInBackground(tradeData) {
  try {
    // Execute trade even if main app is not active
    const response = await fetch('/api/trades/execute', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(tradeData)
    });
    
    const result = await response.json();
    
    // Notify clients
    self.clients.matchAll().then(clients => {
      clients.forEach(client => {
        client.postMessage({
          type: 'TRADE_EXECUTED',
          data: result
        });
      });
    });
    
    // Show notification
    if (self.registration.showNotification) {
      self.registration.showNotification('Trade Executed', {
        body: `${tradeData.side} ${tradeData.quantity} ${tradeData.symbol} at $${tradeData.price}`,
        icon: '/icon-192x192.png',
        badge: '/badge-72x72.png',
        tag: 'trade-notification',
        requireInteraction: false
      });
    }
  } catch (error) {
    console.error('Background trade execution failed:', error);
  }
}

async function updateCache(urls) {
  const cache = await caches.open(CACHE_NAME);
  await cache.addAll(urls);
}

let syncInterval = null;

function startBackgroundSync() {
  if (syncInterval) return;
  
  // Sync every 45 seconds (matching trading cycle)
  syncInterval = setInterval(() => {
    syncPrices();
    checkAgentStatus();
  }, 45000);
  
  console.log('Background sync started');
}

function stopBackgroundSync() {
  if (syncInterval) {
    clearInterval(syncInterval);
    syncInterval = null;
    console.log('Background sync stopped');
  }
}

// Push notification handling
self.addEventListener('push', event => {
  const options = {
    body: event.data ? event.data.text() : 'New trading opportunity',
    icon: '/icon-192x192.png',
    badge: '/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'view',
        title: 'View',
        icon: '/images/view.png'
      },
      {
        action: 'dismiss',
        title: 'Dismiss',
        icon: '/images/dismiss.png'
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification('Trading Bot Alert', options)
  );
});

// Notification click handling
self.addEventListener('notificationclick', event => {
  event.notification.close();
  
  if (event.action === 'view') {
    // Open the app
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

console.log('Service Worker: Loaded');