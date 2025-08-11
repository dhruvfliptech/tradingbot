import React, { useState } from 'react';
import { X, User as UserIcon, SlidersHorizontal, BarChart3 } from 'lucide-react';
import { AutoTradeSettings } from '../Dashboard/AutoTradeSettings';
import { PerformanceAnalytics } from '../Dashboard/PerformanceAnalytics';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  userEmail?: string | null;
  onSignOut?: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, userEmail, onSignOut }) => {
  const [tab, setTab] = useState<'profile' | 'trading' | 'analytics'>('profile');

  if (!isOpen) return null;

  const TabButton: React.FC<{ id: 'profile' | 'trading' | 'analytics'; label: string; icon: React.ReactNode }> = ({ id, label, icon }) => (
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
        </div>

        {tab === 'profile' && (
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
      </div>
    </div>
  );
};


