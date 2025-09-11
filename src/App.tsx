import { useState, useEffect } from 'react';
import { Activity, Bitcoin, LogOut, User, Wifi, WifiOff, AlertCircle, Fish, X } from 'lucide-react';
import { DraggableGrid } from './components/Dashboard/DraggableGrid';
import { AccountSummary } from './components/Dashboard/AccountSummary';
import { PositionsTable } from './components/Dashboard/PositionsTable';
import { MarketWatchlist } from './components/Dashboard/MarketWatchlist';
import { TradingControls } from './components/Dashboard/TradingControls';
import { TradingSignals } from './components/Dashboard/TradingSignals';
import { OrdersTable } from './components/Dashboard/OrdersTable';
import { AutoTradeActivity } from './components/Dashboard/AutoTradeActivity';
import { FearGreedIndexWidget } from './components/Dashboard/FearGreedIndex';
import { MarketInsights } from './components/Dashboard/MarketInsights';
// Whale alerts moved to header popover
import { fetchWhaleAlerts, WhaleTx } from './services/whaleAlertsService';
import { AutoTradeSettings } from './components/Dashboard/AutoTradeSettings';
import { GlobalMarketHeader } from './components/Dashboard/GlobalMarketHeader';
import { DashboardChat } from './components/Assistant/DashboardChat';
import TradingBotReport from './components/Dashboard/TradingBotReport';
import { AuthModal } from './components/Auth/AuthModal';
import { ApiStatusModal } from './components/Settings/ApiStatusModal';
import { SettingsModal } from './components/Settings/SettingsModal';
import { useAuth } from './hooks/useAuth';
// import { portfolioService } from './services/portfolioService';
import { alpacaService } from './services/alpacaService';
import { coinGeckoService } from './services/coinGeckoService';
import { supabase } from './lib/supabase';
import { Account, Position, Order, CryptoData } from './types/trading';
import { tradingAgent, AgentEvent } from './services/tradingAgent';

function App() {
  const [account, setAccount] = useState<Account | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [cryptoData, setCryptoData] = useState<CryptoData[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showApiStatus, setShowApiStatus] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showWhalePanel, setShowWhalePanel] = useState(false);
  const [whaleEvents, setWhaleEvents] = useState<WhaleTx[]>([]);
  const [whaleLoading, setWhaleLoading] = useState(false);
  const [whaleLastFetched, setWhaleLastFetched] = useState<number>(0);
  const [apiStatuses, setApiStatuses] = useState({
    alpaca: 'checking' as 'connected' | 'error' | 'checking',
    coingecko: 'checking' as 'connected' | 'error' | 'checking',
    supabase: 'checking' as 'connected' | 'error' | 'checking',
  });
  
  const { user, loading: authLoading, signOut } = useAuth();

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        console.log('🔄 Fetching dashboard data...');
        // Temporarily bypass user check for demo
        // if (!user) {
        //   setLoading(false);
        //   setApiStatuses({
        //     alpaca: 'error',
        //     coingecko: 'error',
        //     supabase: 'error',
        //   });
        //   return;
        // }

        // First fetch crypto data to ensure we have real-time prices
        console.log('📊 Fetching real-time crypto data...');
        const cryptoData = await coinGeckoService.getCryptoData(['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano']);
        setCryptoData(cryptoData);
        console.log('✅ Crypto data updated:', cryptoData.map(c => `${c.symbol}: $${c.price.toLocaleString()}`));

        // Then fetch other data
        const [accountData, positionsData, ordersData] = await Promise.all([
          alpacaService.getAccount(),
          alpacaService.getPositions(),
          alpacaService.getOrders(),
        ]);

        setAccount(accountData);
        setPositions(positionsData);
        setOrders(ordersData);
        
        // Update API statuses based on successful data fetch
        setApiStatuses({
          alpaca: 'connected',
          coingecko: 'connected',
          supabase: 'connected',
        });
      } catch (error) {
        console.error('Failed to fetch Alpaca data:', error);
        console.warn('Using fallback data due to Alpaca API issues');
        
        // Check individual API statuses
        const newStatuses = { ...apiStatuses };
        
        try {
          await alpacaService.getAccount();
          newStatuses.alpaca = 'connected';
        } catch {
          newStatuses.alpaca = 'error';
        }
        
        try {
          await coinGeckoService.getCryptoData(['bitcoin']);
          newStatuses.coingecko = 'connected';
        } catch {
          newStatuses.coingecko = 'error';
        }
        
        try {
          await supabase.from('portfolios').select('count').limit(1);
          newStatuses.supabase = 'connected';
        } catch {
          newStatuses.supabase = 'error';
        }
        
        setApiStatuses(newStatuses);
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    // Live refresh: keep core data up to date
    const cryptoInterval = setInterval(async () => {
      try {
        const updated = await coinGeckoService.getCryptoData(['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano']);
        setCryptoData(updated);
      } catch (err) {
        console.warn('Crypto refresh failed:', err);
      }
    }, 30 * 1000); // 30s

    const refreshAlpaca = async () => {
      // Temporarily bypass user check for demo
      // if (!user) return;
      try {
        const [accountData, positionsData, ordersData] = await Promise.all([
          alpacaService.getAccount(),
          alpacaService.getPositions(),
          alpacaService.getOrders(),
        ]);
        setAccount(accountData);
        setPositions(positionsData);
        setOrders(ordersData);
      } catch (err) {
        console.warn('Alpaca refresh failed:', err);
      }
    };

    const alpacaInterval = setInterval(refreshAlpaca, 60 * 1000); // 60s

    // Subscribe to auto-trader: refresh immediately when an order is submitted
    const unsub = tradingAgent.subscribe((e: AgentEvent) => {
      if (e.type === 'order_submitted') {
        // immediate refresh to reflect new position/order
        refreshAlpaca();
      }
    });

    return () => {
      clearInterval(cryptoInterval);
      clearInterval(alpacaInterval);
      unsub();
    };
  }, [user]);

  // immediate refresh after a manual order
  const handleOrderPlaced = async () => {
    try {
      const [accountData, positionsData, ordersData] = await Promise.all([
        alpacaService.getAccount(),
        alpacaService.getPositions(),
        alpacaService.getOrders(),
      ]);
      setAccount(accountData);
      setPositions(positionsData);
      setOrders(ordersData);
    } catch (err) {
      console.warn('Post-order refresh failed:', err);
    }
  };

  const handleAuthSuccess = () => {
    setShowAuthModal(false);
    // Data will be fetched automatically when user state changes
  };

  const handleSignOut = async () => {
    await signOut();
    setAccount(null);
    setPositions([]);
    setOrders([]);
  };

  const getOverallApiStatus = () => {
    const statuses = Object.values(apiStatuses);
    const connectedCount = statuses.filter(status => status === 'connected').length;
    const errorCount = statuses.filter(status => status === 'error').length;
    
    if (connectedCount === 3) {
      return { icon: Wifi, color: 'text-green-400', text: 'All APIs Connected' };
    } else if (errorCount === 3) {
      return { icon: WifiOff, color: 'text-red-400', text: 'All APIs Disconnected' };
    } else {
      return { icon: AlertCircle, color: 'text-yellow-400', text: 'Partial API Issues' };
    }
  };

  const toggleWhalePanel = async () => {
    const now = Date.now();
    const needsRefresh = now - whaleLastFetched > 2 * 60 * 1000; // 2 minutes
    setShowWhalePanel(v => !v);
    if ((!showWhalePanel && (whaleEvents.length === 0 || needsRefresh))) {
      try {
        setWhaleLoading(true);
        const events = await fetchWhaleAlerts(['BTC','ETH','SOL'], 1_000_000);
        setWhaleEvents(events);
        setWhaleLastFetched(Date.now());
      } catch {
        setWhaleEvents([]);
      } finally {
        setWhaleLoading(false);
      }
    }
  };

  // Dashboard widgets configuration
  const dashboardWidgets = [
    {
      id: 'trading-bot-report',
      title: 'Trading Bot Performance Report',
      component: <TradingBotReport />,
    },
    {
      id: 'account-summary',
      title: 'Account Summary',
      component: <AccountSummary account={account} />,
    },
    {
      id: 'positions',
      title: 'Current Positions',
      component: <PositionsTable positions={positions} />,
    },
    {
      id: 'watchlist',
      title: 'Market Watchlist',
      component: <MarketWatchlist cryptoData={cryptoData} />,
    },
    {
      id: 'orders',
      title: 'Recent Orders',
      component: <OrdersTable orders={orders} />,
    },
    {
      id: 'fear-greed',
      title: 'Fear & Greed Index',
      component: <FearGreedIndexWidget />,
    },
    {
      id: 'trading-controls',
      title: 'Trading Controls',
      component: <TradingControls onOrderPlaced={handleOrderPlaced} />,
    },
    {
      id: 'trading-signals',
      title: 'Trading Signals',
      component: <TradingSignals cryptoData={cryptoData} />,
    },
    {
      id: 'auto-trade-activity',
      title: 'Auto-Trade Activity',
      component: <AutoTradeActivity />,
    },
    {
      id: 'market-insights',
      title: 'AI Market Insights',
      component: <MarketInsights key={`insights-${cryptoData.length}-${Date.now()}`} cryptoData={cryptoData} />,
    },
    {
      id: 'auto-trade-settings',
      title: 'Auto-Trade Settings',
      component: <AutoTradeSettings />,
    },
  ];

  if (authLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  // Temporarily bypass authentication for demo purposes
  // if (!user) {
  //   return (
  //     <div className="min-h-screen bg-gray-900 flex items-center justify-center">
  //       <div className="text-center">
  //         <div className="mb-8">
  //           <Bitcoin className="h-16 w-16 text-orange-400 mx-auto mb-4" />
  //           <h1 className="text-3xl font-bold text-white mb-2">AI Crypto Trading Agent</h1>
  //           <p className="text-gray-400">Sign in to access your crypto portfolio</p>
  //         </div>
  //         <button
  //           onClick={() => setShowAuthModal(true)}
  //           className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-semibold transition-colors"
  //         >
  //           Get Started
  //         </button>
  //       </div>
  //       <AuthModal
  //         isOpen={showAuthModal}
  //         onClose={() => setShowAuthModal(false)}
  //         onAuthSuccess={handleAuthSuccess}
  //       />
  //     </div>
  //   );
  // }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading your portfolio...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-[1600px] mx-auto px-2 sm:px-4 lg:px-8">
          <div className="flex items-center justify-between h-14 sm:h-16">
            <div className="flex items-center">
              <Bitcoin className="h-6 w-6 sm:h-8 sm:w-8 text-orange-400" />
              <h1 className="ml-2 sm:ml-3 text-lg sm:text-xl font-bold text-white hidden sm:block">AI Crypto Trading Agent</h1>
              <h1 className="ml-2 text-sm font-bold text-white sm:hidden">AI Crypto</h1>
              <div className="ml-2 sm:ml-4 px-2 sm:px-3 py-1 bg-green-900/30 text-green-400 rounded-full text-xs sm:text-sm">
                DEMO TRADING
              </div>
            </div>
            <div className="flex items-center space-x-2 sm:space-x-4 relative">
              <div className="hidden lg:flex items-center text-gray-300">
                <Activity className="h-4 w-4 mr-2" />
                <span className="text-sm">24/7 Trading</span>
              </div>
             
              {/* API Status Indicator */}
             <button
               onClick={() => setShowApiStatus(true)}
               className="flex items-center px-2 sm:px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
               title="Click to view detailed API status"
             >
               {(() => {
                 const status = getOverallApiStatus();
                 return (
                   <>
                     <status.icon className={`h-4 w-4 mr-2 ${status.color}`} />
                     <span className={`text-xs sm:text-sm ${status.color}`}>
                       <span className="hidden sm:inline">
                         {apiStatuses.alpaca === 'connected' ? '✓' : '✗'} Alpaca{' '}
                         {apiStatuses.coingecko === 'connected' ? '✓' : '✗'} CoinGecko{' '}
                         {apiStatuses.supabase === 'connected' ? '✓' : '✗'} Supabase
                       </span>
                       <span className="sm:hidden">
                         {apiStatuses.alpaca === 'connected' ? '✓' : '✗'}
                         {apiStatuses.coingecko === 'connected' ? '✓' : '✗'}
                         {apiStatuses.supabase === 'connected' ? '✓' : '✗'}
                       </span>
                     </span>
                   </>
                 );
               })()}
             </button>

              {/* Whale Alerts */}
              <div className="relative">
                <button
                  onClick={toggleWhalePanel}
                  className={`flex items-center px-2 py-1 rounded-lg transition-colors ${showWhalePanel ? 'bg-indigo-700' : 'bg-gray-700 hover:bg-gray-600'}`}
                  title="Whale activity"
                >
                  <Fish className="h-4 w-4 text-indigo-300" />
                </button>
                {showWhalePanel && (
                  <div className="absolute right-0 mt-2 w-[380px] max-h-[480px] overflow-y-auto bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-50">
                    <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
                      <div className="text-white font-semibold text-sm">Whale Activity</div>
                      <button onClick={() => setShowWhalePanel(false)} className="text-gray-400 hover:text-white"><X className="h-4 w-4" /></button>
                    </div>
                    <div className="p-3 space-y-2">
                      {whaleLoading ? (
                        <div className="text-gray-400 text-sm">Loading large transfers…</div>
                      ) : whaleEvents.length === 0 ? (
                        <div className="text-gray-400 text-sm">No recent $1M+ transfers.</div>
                      ) : (
                        whaleEvents.map(e => (
                          <div key={e.id} className="bg-gray-800 rounded p-3">
                            <div className="flex items-center justify-between">
                              <div className="text-white font-medium">{e.symbol}</div>
                              <div className="text-gray-400 text-xs">{new Date(e.timestamp).toLocaleTimeString()}</div>
                            </div>
                            <div className="text-gray-300 text-sm">
                              {e.amount.toLocaleString()} {e.symbol} {e.usdValue ? `(≈$${Math.round(e.usdValue).toLocaleString()})` : ''}
                            </div>
                            {e.note && <div className="text-gray-500 text-xs mt-1">{e.note}</div>}
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                )}
              </div>
             
              <div className="hidden md:flex items-center text-gray-300 text-sm">
                <User className="h-4 w-4 mr-2" />
                <span>{user?.email || 'demo@example.com'}</span>
              </div>
              <button
                onClick={handleSignOut}
                className="flex items-center p-2 text-gray-400 hover:text-white transition-colors"
                title="Sign Out"
              >
                <LogOut className="h-5 w-5" />
              </button>

              {/* Settings (Profile + Analytics) */}
              <button
                onClick={() => setShowSettings(true)}
                className="flex items-center px-2 sm:px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                title="Profile, Trading Settings, Analytics"
              >
                <span className="text-xs sm:text-sm text-gray-200">Settings</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Global Market Header */}
      <GlobalMarketHeader />

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto px-2 sm:px-4 lg:px-8 py-4 sm:py-8">
        {/* Draggable Dashboard Grid */}
        <DraggableGrid widgets={dashboardWidgets} />
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 mt-12">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="text-gray-400 text-sm">
              © 2024 AI Crypto Trading Agent. Secure trading platform.
            </div>
            <div className="flex items-center space-x-4 text-sm text-gray-400">
              <span>Last Update: {new Date().toLocaleTimeString()}</span>
              <div className="flex items-center">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                <span>Connected</span>
              </div>
            </div>
          </div>
        </div>
      </footer>

      {/* API Status Modal */}
      <ApiStatusModal
        isOpen={showApiStatus}
        onClose={() => setShowApiStatus(false)}
      />

      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        userEmail={user?.email ?? 'demo@example.com'}
        onSignOut={handleSignOut}
      />

      {/* Floating AI Assistant */}
      <DashboardChat
        account={account}
        positions={positions}
        orders={orders}
        cryptoData={cryptoData}
        apiStatuses={apiStatuses}
        onPostTrade={handleOrderPlaced}
      />
    </div>
  );
}

export default App;