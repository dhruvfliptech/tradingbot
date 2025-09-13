import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  ArrowLeft, 
  FileText, 
  Filter, 
  Download,
  RefreshCw,
  Search,
  Calendar,
  Activity,
  Settings,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  XCircle
} from 'lucide-react';
import { auditLogService, AuditLog, EventType } from '../services/persistence/auditLogService';

export const AuditLogs: React.FC = () => {
  const navigate = useNavigate();
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    eventType: '' as EventType | '',
    symbol: '',
    dateFrom: '',
    dateTo: '',
    searchTerm: ''
  });
  const [showFilters, setShowFilters] = useState(false);
  const [userImpact, setUserImpact] = useState<any>(null);

  useEffect(() => {
    loadLogs();
    loadUserImpact();
  }, [filters]);

  const loadLogs = async () => {
    try {
      setLoading(true);
      
      const logFilters: any = {};
      if (filters.eventType) logFilters.event_type = filters.eventType;
      if (filters.symbol) logFilters.symbol = filters.symbol;
      if (filters.dateFrom) logFilters.startDate = new Date(filters.dateFrom);
      if (filters.dateTo) logFilters.endDate = new Date(filters.dateTo);
      
      const auditLogs = await auditLogService.getAuditLogs(logFilters);
      
      // Apply search filter locally
      let filteredLogs = auditLogs;
      if (filters.searchTerm) {
        const term = filters.searchTerm.toLowerCase();
        filteredLogs = auditLogs.filter(log => 
          log.symbol?.toLowerCase().includes(term) ||
          log.event_type.toLowerCase().includes(term) ||
          log.user_reason?.toLowerCase().includes(term) ||
          JSON.stringify(log.ai_reasoning).toLowerCase().includes(term)
        );
      }
      
      setLogs(filteredLogs);
    } catch (error) {
      console.error('Failed to load audit logs:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadUserImpact = async () => {
    try {
      const endDate = new Date();
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - 30);
      
      const impact = await auditLogService.calculateUserImpact(startDate, endDate);
      setUserImpact(impact);
    } catch (error) {
      console.error('Failed to calculate user impact:', error);
    }
  };

  const handleExport = async () => {
    try {
      const csv = await auditLogService.exportToCSV();
      
      // Create blob and download
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `audit-logs-${new Date().toISOString().split('T')[0]}.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export logs:', error);
    }
  };

  const getEventIcon = (eventType: EventType) => {
    switch (eventType) {
      case 'ai_decision':
        return <Activity className="h-4 w-4 text-blue-400" />;
      case 'user_override':
        return <AlertCircle className="h-4 w-4 text-yellow-400" />;
      case 'setting_change':
        return <Settings className="h-4 w-4 text-purple-400" />;
      case 'trade_execution':
        return <TrendingUp className="h-4 w-4 text-green-400" />;
      case 'agent_pause':
      case 'agent_resume':
        return <AlertCircle className="h-4 w-4 text-orange-400" />;
      default:
        return <FileText className="h-4 w-4 text-gray-400" />;
    }
  };

  const getActionColor = (action?: string) => {
    if (!action) return 'text-gray-400';
    switch (action) {
      case 'buy':
        return 'text-green-400';
      case 'sell':
      case 'short':
        return 'text-red-400';
      case 'hold':
        return 'text-yellow-400';
      default:
        return 'text-gray-400';
    }
  };

  const formatTimestamp = (timestamp?: string) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleString();
  };

  const renderAIReasoning = (reasoning: any) => {
    if (!reasoning) return null;
    
    return (
      <div className="mt-2 p-2 bg-gray-900 rounded text-xs">
        {reasoning.confidence && (
          <div className="mb-1">
            <span className="text-gray-500">Confidence:</span>{' '}
            <span className="text-white font-medium">{reasoning.confidence}%</span>
          </div>
        )}
        {reasoning.signals && (
          <div className="grid grid-cols-3 gap-2 mt-2">
            {reasoning.signals.technical && (
              <div>
                <span className="text-gray-500">Technical:</span>
                <div className="text-gray-300">
                  RSI: {reasoning.signals.technical.rsi}<br/>
                  MACD: {reasoning.signals.technical.macd}
                </div>
              </div>
            )}
            {reasoning.signals.momentum && (
              <div>
                <span className="text-gray-500">Momentum:</span>
                <div className="text-gray-300">
                  24h: {reasoning.signals.momentum.change_24h}%<br/>
                  Vol: {(reasoning.signals.momentum.volume / 1e9).toFixed(1)}B
                </div>
              </div>
            )}
            {reasoning.signals.sentiment && (
              <div>
                <span className="text-gray-500">Sentiment:</span>
                <div className="text-gray-300">
                  F&G: {reasoning.signals.sentiment.fear_greed}<br/>
                  Whale: {reasoning.signals.sentiment.whale_activity}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <button
                onClick={() => navigate('/agent-controls')}
                className="flex items-center text-gray-400 hover:text-white transition-colors"
              >
                <ArrowLeft className="h-5 w-5 mr-2" />
                Back to Agent Controls
              </button>
              <div className="ml-8 flex items-center">
                <FileText className="h-6 w-6 text-indigo-400 mr-2" />
                <h1 className="text-xl font-bold text-white">Audit Logs</h1>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`flex items-center px-3 py-2 rounded-lg transition-colors ${
                  showFilters ? 'bg-indigo-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                <Filter className="h-4 w-4 mr-2" />
                Filters
              </button>
              
              <button
                onClick={handleExport}
                className="flex items-center px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
              >
                <Download className="h-4 w-4 mr-2" />
                Export CSV
              </button>
              
              <button
                onClick={loadLogs}
                className="flex items-center px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* User Impact Summary */}
      {userImpact && (
        <div className="bg-gray-800 border-b border-gray-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div>
                <p className="text-sm text-gray-400">Total Interventions</p>
                <p className="text-xl font-bold text-white">{userImpact.totalInterventions}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Overrides</p>
                <p className="text-xl font-bold text-yellow-400">{userImpact.overrideCount}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Pauses</p>
                <p className="text-xl font-bold text-orange-400">{userImpact.pauseCount}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Settings Changes</p>
                <p className="text-xl font-bold text-purple-400">{userImpact.settingChanges}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Impact on Returns</p>
                <p className={`text-xl font-bold ${userImpact.impactOnReturns >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {userImpact.impactOnReturns >= 0 ? '+' : ''}{userImpact.impactOnReturns.toFixed(2)}%
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      {showFilters && (
        <div className="bg-gray-800 border-b border-gray-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Event Type</label>
                <select
                  value={filters.eventType}
                  onChange={(e) => setFilters(prev => ({ ...prev, eventType: e.target.value as EventType | '' }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                >
                  <option value="">All Types</option>
                  <option value="ai_decision">AI Decision</option>
                  <option value="user_override">User Override</option>
                  <option value="setting_change">Setting Change</option>
                  <option value="trade_execution">Trade Execution</option>
                  <option value="agent_pause">Agent Pause</option>
                  <option value="agent_resume">Agent Resume</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Symbol</label>
                <input
                  type="text"
                  value={filters.symbol}
                  onChange={(e) => setFilters(prev => ({ ...prev, symbol: e.target.value }))}
                  placeholder="e.g., BTC"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">From Date</label>
                <input
                  type="date"
                  value={filters.dateFrom}
                  onChange={(e) => setFilters(prev => ({ ...prev, dateFrom: e.target.value }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">To Date</label>
                <input
                  type="date"
                  value={filters.dateTo}
                  onChange={(e) => setFilters(prev => ({ ...prev, dateTo: e.target.value }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Search</label>
                <div className="relative">
                  <input
                    type="text"
                    value={filters.searchTerm}
                    onChange={(e) => setFilters(prev => ({ ...prev, searchTerm: e.target.value }))}
                    placeholder="Search logs..."
                    className="w-full px-3 py-2 pl-9 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-500"
                  />
                  <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                </div>
              </div>
            </div>
            
            <div className="mt-4 flex justify-end">
              <button
                onClick={() => setFilters({
                  eventType: '',
                  symbol: '',
                  dateFrom: '',
                  dateTo: '',
                  searchTerm: ''
                })}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
              >
                Clear Filters
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
          </div>
        ) : logs.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="h-12 w-12 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400">No audit logs found matching your criteria</p>
          </div>
        ) : (
          <div className="space-y-4">
            {logs.map((log) => (
              <div key={log.id} className="bg-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    {getEventIcon(log.event_type)}
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className="text-white font-medium capitalize">
                          {log.event_type.replace('_', ' ')}
                        </span>
                        {log.symbol && (
                          <span className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-300">
                            {log.symbol}
                          </span>
                        )}
                        {log.action && (
                          <span className={`font-medium uppercase text-sm ${getActionColor(log.action)}`}>
                            {log.action}
                          </span>
                        )}
                        {log.confidence_score !== undefined && (
                          <span className="text-sm text-gray-400">
                            ({log.confidence_score}% confidence)
                          </span>
                        )}
                      </div>
                      
                      {log.user_reason && (
                        <div className="mt-2 text-sm text-gray-300">
                          <span className="text-gray-500">Reason:</span> {log.user_reason}
                        </div>
                      )}
                      
                      {log.ai_reasoning && renderAIReasoning(log.ai_reasoning)}
                      
                      {log.old_value && log.new_value && (
                        <div className="mt-2 text-sm">
                          <span className="text-gray-500">Changed from:</span>{' '}
                          <span className="text-gray-400">{JSON.stringify(log.old_value)}</span>
                          <span className="text-gray-500"> to </span>
                          <span className="text-white">{JSON.stringify(log.new_value)}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="text-xs text-gray-500">
                    {formatTimestamp(log.created_at)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
};