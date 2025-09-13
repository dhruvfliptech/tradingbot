import React, { useState } from 'react';
import { X, User as UserIcon, SlidersHorizontal, BarChart3, Key, RefreshCw, AlertTriangle } from 'lucide-react';
import { AutoTradeSettings } from '../Dashboard/AutoTradeSettings';
import { PerformanceAnalytics } from '../Dashboard/PerformanceAnalytics';
import { ApiKeysTab } from './ApiKeysTab';
import { virtualPortfolioService } from '../../services/persistence/virtualPortfolioService';
import { tradeHistoryService } from '../../services/persistence/tradeHistoryService';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  userEmail?: string | null;
  onSignOut?: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, userEmail, onSignOut }) => {
  const [tab, setTab] = useState<'profile' | 'trading' | 'analytics' | 'apikeys'>('profile');
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [resetting, setResetting] = useState(false);

  if (!isOpen) return null;

  const handlePortfolioReset = async () => {
    try {
      setResetting(true);
      
      // Initialize portfolio if not already done
      const portfolio = await virtualPortfolioService.initializePortfolio();
      
      if (portfolio) {
        // Reset the portfolio
        await virtualPortfolioService.resetPortfolio(true);
        
        // Reload the page to refresh all data
        window.location.reload();
      }
    } catch (error) {
      console.error('Failed to reset portfolio:', error);
      alert('Failed to reset portfolio. Please try again.');
    } finally {
      setResetting(false);
      setShowResetConfirm(false);
    }
  };

  const handleExportTrades = async () => {
    try {
      const csv = await tradeHistoryService.exportToCSV();
      
      if (!csv) {
        alert('No trade history to export');
        return;
      }
      
      // Create blob and download
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `trade-history-${new Date().toISOString().split('T')[0]}.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export trades:', error);
      alert('Failed to export trade history. Please try again.');
    }
  };

  const TabButton: React.FC<{ id: 'profile' | 'trading' | 'analytics' | 'apikeys'; label: string; icon: React.ReactNode }> = ({ id, label, icon }) => (
    <button
      onClick={() => setTab(id)}
      className={`flex items-center px-3 py-2 rounded-lg text-sm transition-colors ${
        tab === id ? 'bg-indigo-600 text-white' : 'bg-gray-700 text-gray-200 hover:bg-gray-600'
      }`}
    >
      <span className="mr-2">{icon}</span>
      {label}
    </button>
  );

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-4xl mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white">Settings</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white"><X className="h-5 w-5" /></button>
        </div>

        <div className="flex items-center space-x-2 mb-4">
          <TabButton id="profile" label="Profile" icon={<UserIcon className="h-4 w-4" />} />
          <TabButton id="trading" label="Trading" icon={<SlidersHorizontal className="h-4 w-4" />} />
          <TabButton id="analytics" label="Analytics" icon={<BarChart3 className="h-4 w-4" />} />
          <TabButton id="apikeys" label="API Keys" icon={<Key className="h-4 w-4" />} />
        </div>

        {tab === 'profile' && (
          <div className="space-y-4">
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="text-gray-300 text-sm">Signed in as</div>
              <div className="text-white text-lg font-semibold">{userEmail || 'Unknown user'}</div>
              <div className="mt-4 flex items-center space-x-2">
                <button
                  onClick={onSignOut}
                  className="px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded"
                >
                  Sign out
                </button>
              </div>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4">
              <h3 className="text-white font-semibold mb-3">Portfolio Management</h3>
              
              <div className="space-y-3">
                <button
                  onClick={handleExportTrades}
                  className="w-full flex items-center justify-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors"
                >
                  <svg className="h-4 w-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Export Trade History (CSV)
                </button>
                
                <button
                  onClick={() => setShowResetConfirm(true)}
                  className="w-full flex items-center justify-center px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Reset Portfolio to $50,000
                </button>
                
                <div className="p-3 bg-gray-800 rounded text-xs text-gray-400">
                  <AlertTriangle className="h-3 w-3 inline mr-1" />
                  Portfolio reset will archive all existing trades and start fresh with $50,000. This action cannot be undone.
                </div>
              </div>
            </div>
          </div>
        )}

        {tab === 'trading' && (
          <div className="bg-gray-700 rounded-lg p-4">
            <AutoTradeSettings />
          </div>
        )}

        {tab === 'analytics' && (
          <div className="bg-gray-700 rounded-lg p-4">
            <PerformanceAnalytics />
          </div>
        )}

        {tab === 'apikeys' && (
          <div className="bg-gray-700 rounded-lg p-4">
            <ApiKeysTab />
          </div>
        )}
      </div>
      
      {/* Portfolio Reset Confirmation Dialog */}
      {showResetConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-60">
          <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
            <div className="flex items-start space-x-3 mb-4">
              <AlertTriangle className="h-6 w-6 text-red-400 flex-shrink-0 mt-1" />
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Reset Portfolio to $50,000?</h3>
                <p className="text-gray-300 text-sm">
                  This will:
                </p>
                <ul className="list-disc list-inside text-gray-400 text-sm mt-2 space-y-1">
                  <li>Archive all existing trades</li>
                  <li>Reset your balance to $50,000</li>
                  <li>Clear all open positions</li>
                  <li>Reset performance metrics</li>
                </ul>
                <p className="text-red-400 text-sm mt-3 font-medium">
                  This action cannot be undone!
                </p>
              </div>
            </div>
            
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowResetConfirm(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handlePortfolioReset}
                disabled={resetting}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors flex items-center"
              >
                {resetting ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                    Resetting...
                  </>
                ) : (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Reset Portfolio
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};


