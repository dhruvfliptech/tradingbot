import { useState, useEffect } from 'react';
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';
import { useNavigate } from 'react-router-dom';
import { Activity, Bitcoin, LogOut, User, Wifi, WifiOff, AlertCircle, Fish, X, Settings as SettingsIcon, SlidersHorizontal } from 'lucide-react';
import { DraggableGrid } from '../components/Dashboard/DraggableGrid';
import { AccountSummary } from '../components/Dashboard/AccountSummary';
import { PositionsTable } from '../components/Dashboard/PositionsTable';
import { MarketWatchlist } from '../components/Dashboard/MarketWatchlist';
import { TradingSignals } from '../components/Dashboard/TradingSignals';
import { FearGreedIndexWidget } from '../components/Dashboard/FearGreedIndex';
import { OrdersTable } from '../components/Dashboard/OrdersTable';
import { PerformanceCalendar } from '../components/Dashboard/PerformanceCalendar';
import { MarketInsights } from '../components/Dashboard/MarketInsights';
import { PortfolioAnalyticsCompact } from '../components/Dashboard/PortfolioAnalyticsCompact';
import { StrategyControlPanel } from '../components/Dashboard/StrategyControlPanel';
import { TradeManagement } from '../components/Dashboard/TradeManagement';
import { ExportReporting } from '../components/Dashboard/ExportReporting';
import StrategySignals from '../components/Dashboard/StrategySignals';
import TargetTracker from '../components/Dashboard/TargetTracker';
import OnChainMetrics from '../components/Dashboard/OnChainMetrics';
import { WidgetErrorBoundary } from '../components/ErrorBoundary';
import { tradingAgentV2 as tradingAgent } from '../services/tradingAgentV2';
import { fetchWhaleAlerts, WhaleTx } from '../services/whaleAlertsService';
import { GlobalMarketHeader } from '../components/Dashboard/GlobalMarketHeader';
import TradingBotReport from '../components/Dashboard/TradingBotReport';
import { ApiStatusModal } from '../components/Settings/ApiStatusModal';
import { SettingsModal } from '../components/Settings/SettingsModal';
import { useAuth } from '../hooks/useAuth';
import { useVirtualPortfolio } from '../hooks/useVirtualPortfolio';
import { useTradingProvider } from '../hooks/useTradingProvider';
import { useScrollPreservation } from '../hooks/useScrollPreservation';
import { useWebSocket } from '../services/websocketService';
import { tradingProviderService } from '../services/tradingProviderService';
import { coinGeckoService } from '../services/coinGeckoService';
import { supabase } from '../lib/supabase';
import { Account, Position, Order, CryptoData } from '../types/trading';
import { TradingModeToggle } from '../components/TradingModeToggle';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [account, setAccount] = useState<Account | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [cryptoData, setCryptoData] = useState<CryptoData[]>([]);
  const [loading, setLoading] = useState(true);
  const [showApiStatus, setShowApiStatus] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showWhalePanel, setShowWhalePanel] = useState(false);
  const [whaleEvents, setWhaleEvents] = useState<WhaleTx[]>([]);
  const [whaleLoading, setWhaleLoading] = useState(false);
  const [whaleLastFetched, setWhaleLastFetched] = useState<number>(0);
  const [apiStatuses, setApiStatuses] = useState({
    broker: 'checking' as 'connected' | 'error' | 'checking',
    coingecko: 'checking' as 'connected' | 'error' | 'checking',
    supabase: 'checking' as 'connected' | 'error' | 'checking',
  });
  const [portfolioHistory, setPortfolioHistory] = useState<Array<{ date: string; value: number; pnl: number }>>([]);
  const [portfolioHistoryLoading, setPortfolioHistoryLoading] = useState(false);
  
  const { user, signOut } = useAuth();
  const { portfolio, stats: portfolioStats, getPerformanceHistory } = useVirtualPortfolio();
  const { preserveScroll } = useScrollPreservation(orders);
  const { lastMessage, isConnected: wsConnected } = useWebSocket('trade_executed');
  const { activeProvider } = useTradingProvider();


  // Handle WebSocket trade messages with scroll preservation
  useEffect(() => {
    if (lastMessage && lastMessage.type === 'trade_executed') {
      preserveScroll(() => {
        // Refresh orders when a trade is executed
        handleOrderPlaced();
      });
    }
  }, [lastMessage]);


  useEffect(() => {
    let cancelled = false;

    const loadHistory = async () => {
      setPortfolioHistoryLoading(true);
      try {
        const history = await getPerformanceHistory(30);
        if (!cancelled) {
          setPortfolioHistory(history ?? []);
        }
      } catch (error) {
        if (!cancelled) {
          setPortfolioHistory([]);
        }
      } finally {
        if (!cancelled) {
          setPortfolioHistoryLoading(false);
        }
      }
    };

    loadHistory();
    const historyInterval = setInterval(loadHistory, 60 * 1000);

    return () => {
      cancelled = true;
      clearInterval(historyInterval);
    };
  }, [getPerformanceHistory, portfolio?.id]);

  useEffect(() => {
    fetchData();

    // Live refresh intervals
    const cryptoInterval = setInterval(async () => {
      try {
        const updated = await coinGeckoService.getCryptoData(['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano']);
        setCryptoData(updated);
      } catch (err) {
        console.warn('Crypto refresh failed:', err);
      }
    }, 30 * 1000); // 30s

    const refreshBroker = async () => {
      if (!user) return;
      try {
        const [accountData, positionsData, ordersData] = await Promise.all([
          tradingProviderService.getAccount(),
          tradingProviderService.getPositions(),
          tradingProviderService.getOrders(),
        ]);
        setAccount(accountData);
        setPositions(positionsData);
        setOrders(ordersData);
        setApiStatuses((prev) => ({ ...prev, broker: 'connected' }));
      } catch (err) {
        console.warn('Broker refresh failed:', err);
        setApiStatuses((prev) => ({ ...prev, broker: 'error' }));
      }
    };

    const brokerInterval = setInterval(refreshBroker, 60 * 1000); // 60s

    // Subscribe to auto-trader
    const unsub = tradingAgent.subscribe((e) => {
      if (e.type === 'order_submitted') {
        refreshBroker();
      }
    });

    // Auto-refresh whale alerts every 2 minutes
    const whaleInterval = setInterval(() => {
      if (showWhalePanel) {
        fetchWhaleData();
      }
    }, 2 * 60 * 1000);

    return () => {
      clearInterval(cryptoInterval);
      clearInterval(brokerInterval);
      clearInterval(whaleInterval);
      unsub();
    };
  }, [user, showWhalePanel, activeProvider]);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      if (!user) {
        setLoading(false);
        return;
      }

      // Fetch crypto data first
      const cryptoData = await coinGeckoService.getCryptoData(['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano']);
      setCryptoData(cryptoData);

      // Then fetch other data
      const [accountData, positionsData, ordersData] = await Promise.all([
        tradingProviderService.getAccount(),
        tradingProviderService.getPositions(),
        tradingProviderService.getOrders(),
      ]);

      setAccount(accountData);
      setPositions(positionsData);
      setOrders(ordersData);
      
      // Update API statuses
      setApiStatuses({
        broker: 'connected',
        coingecko: 'connected',
        supabase: 'connected',
      });
    } catch (error) {
      console.error('Failed to fetch data:', error);
      
      // Check individual API statuses
      const newStatuses = { ...apiStatuses };
      
      try {
        await tradingProviderService.getAccount();
        newStatuses.broker = 'connected';
      } catch {
        newStatuses.broker = 'error';
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

  const fetchWhaleData = async () => {
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
  };

  const toggleWhalePanel = async () => {
    const now = Date.now();
    const needsRefresh = now - whaleLastFetched > 2 * 60 * 1000; // 2 minutes
    setShowWhalePanel(v => !v);
    if (!showWhalePanel && (whaleEvents.length === 0 || needsRefresh)) {
      await fetchWhaleData();
    }
  };

  const handleOrderPlaced = async () => {
    try {
      const [accountData, positionsData, ordersData] = await Promise.all([
        tradingProviderService.getAccount(),
        tradingProviderService.getPositions(),
        tradingProviderService.getOrders(),
      ]);
      setAccount(accountData);
      setPositions(positionsData);
      setOrders(ordersData);
    } catch (err) {
      console.warn('Post-order refresh failed:', err);
    }
  };

  const handleSignOut = async () => {
    await signOut();
    setAccount(null);
    setPositions([]);
    setOrders([]);
  };

  const getOverallApiStatus = () => {
    const statuses = Object.values(apiStatuses);
    const total = statuses.length;
    const connectedCount = statuses.filter((status) => status === 'connected').length;
    const errorCount = statuses.filter((status) => status === 'error').length;

    if (connectedCount === total) {
      return { icon: Wifi, color: 'text-green-400', text: 'All APIs Connected' };
    }
    if (errorCount === total) {
      return { icon: WifiOff, color: 'text-red-400', text: 'All APIs Disconnected' };
    }
    return { icon: AlertCircle, color: 'text-yellow-400', text: 'Partial API Issues' };
  };

  const formatHistoryDate = (value: string) => {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  const formatHistoryTooltipLabel = (value: string) => {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  };

  // Dashboard widgets configuration - Enhanced with new strategy widgets
  const dashboardWidgets = [
    {
      id: 'portfolio-value',
      title: 'Portfolio Value Chart ($50K Baseline)',
      component: (
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-2xl font-bold text-white">
                ${portfolioStats?.totalValue.toLocaleString() || '50,000'}
              </p>
              <p className={`text-sm ${(portfolioStats?.totalPnL || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {(portfolioStats?.totalPnL || 0) >= 0 ? '▲' : '▼'} ${Math.abs(portfolioStats?.totalPnL || 0).toFixed(2)} 
                ({portfolioStats?.totalPnLPercent.toFixed(2) || '0.00'}%) from $50,000
              </p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-400">Daily Change</p>
              <p className={`text-lg font-medium ${(portfolioStats?.dailyPnL || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${portfolioStats?.dailyPnL.toFixed(2) || '0.00'}
              </p>
            </div>
          </div>
          <div className="h-48 bg-gray-800 rounded p-2">
            {portfolioHistoryLoading ? (
              <div className="w-full h-full rounded bg-gray-900 animate-pulse" />
            ) : portfolioHistory.length === 0 ? (
              <div className="w-full h-full flex items-center justify-center text-sm text-gray-500">
                Portfolio history will appear once trades are recorded.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={portfolioHistory} margin={{ top: 8, right: 16, left: -12, bottom: 0 }}>
                  <defs>
                    <linearGradient id="portfolioValueGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#6366F1" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#6366F1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                  <XAxis
                    dataKey="date"
                    tickFormatter={formatHistoryDate}
                    tick={{ fill: '#9CA3AF', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    width={70}
                    tickFormatter={(value) => {
                      const numericValue = Number(value);
                      if (!Number.isFinite(numericValue)) return String(value);
                      return `$${numericValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
                    }}
                    tick={{ fill: '#9CA3AF', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    cursor={{ stroke: '#6366F1', strokeWidth: 1, strokeOpacity: 0.25 }}
                    formatter={(value: number | string) => {
                      const numericValue = typeof value === 'number' ? value : Number(value);
                      if (!Number.isFinite(numericValue)) {
                        return [String(value), 'Balance'];
                      }
                      return [`$${numericValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, 'Balance'];
                    }}
                    labelFormatter={formatHistoryTooltipLabel}
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1f2937', color: '#e5e7eb', borderRadius: 8 }}
                    itemStyle={{ color: '#e5e7eb' }}
                  />
                  <Area type="monotone" dataKey="value" stroke="#6366F1" strokeWidth={2} fill="url(#portfolioValueGradient)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      ),
    },
    {
      id: 'strategy-signals',
      title: 'Strategy Signals (4 Advanced Strategies)',
      component: (
        <WidgetErrorBoundary widgetName="Strategy Signals">
          <StrategySignals />
        </WidgetErrorBoundary>
      ),
    },
    {
      id: 'target-tracker',
      title: 'Weekly Target Tracker (3-5% Target)',
      component: (
        <WidgetErrorBoundary widgetName="Target Tracker">
          <TargetTracker />
        </WidgetErrorBoundary>
      ),
    },
    {
      id: 'on-chain-metrics',
      title: 'On-Chain Metrics & Whale Activity',
      component: (
        <WidgetErrorBoundary widgetName="On-Chain Metrics">
          <OnChainMetrics />
        </WidgetErrorBoundary>
      ),
    },
    {
      id: 'trading-signals',
      title: 'AI Trading Signals',
      component: <TradingSignals cryptoData={cryptoData} />,
    },
    {
      id: 'performance-calendar',
      title: 'Performance Calendar',
      component: <PerformanceCalendar />,
    },
    {
      id: 'strategy-control',
      title: 'Strategy Control Panel',
      component: (
        <WidgetErrorBoundary widgetName="Strategy Control">
          <StrategyControlPanel />
        </WidgetErrorBoundary>
      ),
    },
    {
      id: 'trade-management',
      title: 'Trade Management',
      component: (
        <WidgetErrorBoundary widgetName="Trade Management">
          <TradeManagement />
        </WidgetErrorBoundary>
      ),
    },
    {
      id: 'export-reporting',
      title: 'Export & Reports',
      component: (
        <WidgetErrorBoundary widgetName="Export Reporting">
          <ExportReporting />
        </WidgetErrorBoundary>
      ),
    },
    {
      id: 'fear-greed',
      title: 'Fear & Greed Index',
      component: <FearGreedIndexWidget />,
    },
    {
      id: 'portfolio-analytics',
      title: 'Portfolio Analytics',
      component: <PortfolioAnalyticsCompact />,
    },
    {
      id: 'recent-trades',
      title: 'Recent Trades (Automated Only)',
      component: <OrdersTable orders={orders.slice(0, 10)} />,
    },
    {
      id: 'market-insights',
      title: 'AI Market Insights',
      component: <MarketInsights cryptoData={cryptoData} />,
    },
    {
      id: 'whale-alerts',
      title: 'Whale Alerts Feed (Auto-refresh: 2 min)',
      component: (
        <div className="bg-gray-700 rounded-lg p-4">
          {whaleLoading ? (
            <div className="text-gray-400">Loading whale activity...</div>
          ) : whaleEvents.length === 0 ? (
            <div className="text-gray-400">No recent $1M+ transfers detected</div>
          ) : (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {whaleEvents.slice(0, 5).map(event => (
                <div key={event.id} className="p-2 bg-gray-800 rounded">
                  <div className="flex justify-between text-sm">
                    <span className="text-white font-medium">{event.symbol}</span>
                    <span className="text-gray-400">{new Date(event.timestamp).toLocaleTimeString()}</span>
                  </div>
                  <div className="text-sm text-gray-300">
                    {event.amount.toLocaleString()} {event.symbol}
                    {event.usdValue && ` ($${Math.round(event.usdValue).toLocaleString()})`}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ),
    },
  ];

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
            <div className="flex items-center space-x-4">
              <div className="flex items-center">
                <Bitcoin className="h-6 w-6 sm:h-8 sm:w-8 text-orange-400" />
                <h1 className="ml-2 sm:ml-3 text-lg sm:text-xl font-bold text-white hidden sm:block">AI Crypto Trading Agent</h1>
                <h1 className="ml-2 text-sm font-bold text-white sm:hidden">AI Crypto</h1>
              </div>
              <div className="px-2 sm:px-3 py-1 bg-green-900/30 text-green-400 rounded-full text-xs sm:text-sm">
                $50K PORTFOLIO
              </div>
              {/* Agent Status in Nav */}
              <div className={`flex items-center px-3 py-1 rounded-full text-xs sm:text-sm ${
                tradingAgent.isRunning() ? 'bg-green-900/30' : 'bg-red-900/30'
              }`}>
                <div className={`w-2 h-2 rounded-full mr-2 ${
                  tradingAgent.isRunning() ? 'bg-green-400 animate-pulse' : 'bg-red-400'
                }`} />
                <span className={tradingAgent.isRunning() ? 'text-green-400' : 'text-red-400'}>
                  Agent {tradingAgent.isRunning() ? 'Active' : 'Paused'}
                </span>
              </div>
              {/* Quick Stats */}
              <div className="hidden lg:flex items-center space-x-4 text-xs text-gray-400">
                <div>
                  Positions: <span className="text-white">{positions.length}/10</span>
                </div>
                <div>
                  Today: <span className="text-white">
                    {orders.filter(o => new Date(o.created_at).toDateString() === new Date().toDateString()).length} trades
                  </span>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-2 sm:space-x-4 relative">
              <div className="mr-2">
                <TradingModeToggle />
              </div>
              {/* Configure Agent Button - Primary CTA */}
              <button
                onClick={() => navigate('/agent-controls')}
                className="flex items-center px-3 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors font-medium"
              >
                <SlidersHorizontal className="h-4 w-4 mr-2" />
                <span className="hidden sm:inline">Configure Agent</span>
                <span className="sm:hidden">Agent</span>
              </button>

              {/* User Impact Metric */}
              {portfolioStats && (
                <div className="hidden lg:flex items-center text-gray-300 text-sm">
                  <span>User Impact: </span>
                  <span className={`ml-1 font-medium ${portfolioStats.userImpact >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {portfolioStats.userImpact >= 0 ? '+' : ''}{portfolioStats.userImpact.toFixed(1)}%
                  </span>
                </div>
              )}
             
              {/* API Status */}
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
                      <span className={`text-xs sm:text-sm ${status.color} hidden sm:inline`}>
                        APIs
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
                  title="Whale activity (auto-refresh every 2 min)"
                >
                  <Fish className="h-4 w-4 text-indigo-300" />
                </button>
                {showWhalePanel && (
                  <div className="absolute right-0 mt-2 w-[380px] max-h-[480px] overflow-y-auto bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-50">
                    <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
                      <div className="text-white font-semibold text-sm">Whale Activity (Auto-refresh: 2min)</div>
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
                <span>{user?.email}</span>
              </div>
              <button
                onClick={handleSignOut}
                className="flex items-center p-2 text-gray-400 hover:text-white transition-colors"
                title="Sign Out"
              >
                <LogOut className="h-5 w-5" />
              </button>

              {/* Settings */}
              <button
                onClick={() => setShowSettings(true)}
                className="flex items-center px-2 sm:px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                title="Settings"
              >
                <SettingsIcon className="h-4 w-4" />
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
              © 2024 AI Crypto Trading Agent. $50K Virtual Portfolio.
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
        userEmail={user?.email ?? null}
        onSignOut={handleSignOut}
      />

    </div>
  );
};
