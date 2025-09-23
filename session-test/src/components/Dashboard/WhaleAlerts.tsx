import React, { useEffect, useState } from 'react';
import { fetchWhaleAlerts, WhaleTx } from '../../services/whaleAlertsService';

export const WhaleAlerts: React.FC = () => {
  const [events, setEvents] = useState<WhaleTx[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchAlerts = async () => {
    setLoading(true);
    try {
      const events = await fetchWhaleAlerts(['BTC','ETH','SOL'], 1_000_000);
      setEvents(events);
    } catch {
      setEvents([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
    const id = setInterval(fetchAlerts, 3 * 60 * 1000); // refresh every 3m
    return () => clearInterval(id);
  }, []);

  return (
    <div className="h-full min-h-[520px] overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-white">Whale Alerts</h2>
        <button onClick={fetchAlerts} disabled={loading} className="px-3 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 text-white rounded-lg text-sm">Refresh</button>
      </div>
      {loading ? (
        <div className="space-y-3">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-gray-700 rounded p-4 animate-pulse h-14" />
          ))}
        </div>
      ) : events.length === 0 ? (
        <div className="text-gray-400 text-sm">No recent large transfers found.</div>
      ) : (
        <div className="space-y-3">
          {events.map((e) => (
            <div key={e.id} className="bg-gray-700 rounded p-4">
              <div className="flex items-center justify-between">
                <div className="text-white font-semibold">{e.symbol}</div>
                <div className="text-gray-400 text-xs">{new Date(e.timestamp).toLocaleTimeString()}</div>
              </div>
              <div className="text-gray-300 text-sm">
                {e.amount.toLocaleString()} {e.symbol} {e.usdValue ? `(~$${Math.round(e.usdValue).toLocaleString()})` : ''}
              </div>
              {e.note && <div className="text-gray-400 text-xs mt-1">{e.note}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};


