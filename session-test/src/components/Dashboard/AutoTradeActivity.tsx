import React, { useEffect, useRef, useState } from 'react';
import { Zap, CheckCircle2, AlertTriangle, Brain, LineChart, BarChart3 } from 'lucide-react';
import { tradingAgentV2 as tradingAgent, AgentEvent } from '../../services/tradingAgentV2';

export const AutoTradeActivity: React.FC = () => {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const unsub = tradingAgent.subscribe((e) => {
      setEvents((prev) => [...prev.slice(-99), e]); // keep last 100
    });
    return () => unsub();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  const render = (e: AgentEvent, idx: number) => {
    if (e.type === 'status') {
      return (
        <div key={idx} className="flex items-center text-gray-300 text-sm">
          <Brain className={`h-4 w-4 mr-2 ${e.active ? 'text-green-400' : 'text-red-400'}`} />
          <span>
            Agent {e.active ? 'started' : 'stopped'} — {new Date(e.timestamp).toLocaleTimeString()}
          </span>
        </div>
      );
    }
    if (e.type === 'analysis') {
      return (
        <div key={idx} className="flex items-center text-gray-400 text-sm">
          <LineChart className="h-4 w-4 mr-2 text-blue-400" />
          <span>Analyzed {e.evaluated} symbols; top: {e.top.map(t => `${t.symbol} ${t.action} (${Math.round(t.confidence*100)}%)`).join(', ')}{e.note ? ` — ${e.note}` : ''}</span>
        </div>
      );
    }
    if (e.type === 'analysis_detail') {
      return <AnalysisRow key={idx} event={e} />;
    }
    if (e.type === 'market_sentiment') {
      return (
        <div key={idx} className="flex items-center text-gray-300 text-sm">
          <Brain className="h-4 w-4 mr-2 text-indigo-400" />
          <span>Market sentiment: {e.overall} (conf {Math.round(e.confidence*100)}%) — {e.reasoning}</span>
        </div>
      );
    }
    if (e.type === 'decision') {
      return (
        <div key={idx} className="flex items-center text-gray-200 text-sm">
          <Zap className="h-4 w-4 mr-2 text-yellow-400" />
          <span>
            Decision: {e.action} {e.symbol} @ ~${e.price.toLocaleString()} (conf {Math.round(e.confidence * 100)}%)
          </span>
        </div>
      );
    }
    if (e.type === 'no_trade') {
      return (
        <div key={idx} className="flex items-center text-gray-500 text-sm">
          <LineChart className="h-4 w-4 mr-2" />
          <span>No trade: {e.reason}</span>
        </div>
      );
    }
    if (e.type === 'order_submitted') {
      return (
        <div key={idx} className="flex items-center text-green-300 text-sm">
          <CheckCircle2 className="h-4 w-4 mr-2" />
          <span>
            Order: {e.side.toUpperCase()} {e.qty} {e.symbol} ({e.order_type}) — {new Date(e.timestamp).toLocaleTimeString()}
          </span>
        </div>
      );
    }
    return (
      <div key={idx} className="flex items-center text-red-300 text-sm">
        <AlertTriangle className="h-4 w-4 mr-2" />
        <span>
          Order error: {e.symbol} — {e.message}
        </span>
      </div>
    );
  };

  return (
    <div className="h-full min-h-[360px] overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
      <h2 className="text-xl font-bold text-white mb-4">Auto-Trade Activity</h2>
      <div className="space-y-2 max-h-full">
        {events.length === 0 ? (
          <div className="text-gray-400 text-sm">No activity yet. Start the agent to see live decisions and orders.</div>
        ) : (
          events.map(render)
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
};

export default AutoTradeActivity;

type AnalysisRowProps = { event: Extract<AgentEvent, { type: 'analysis_detail' }> };

const AnalysisRow: React.FC<AnalysisRowProps> = ({ event: e }) => {
  const [open, setOpen] = useState(false);
  const ind = e.indicators;
  const total = (e as any).breakdown ? (e as any).breakdown.reduce((s: number, b: any) => s + b.contribution, 0) : 0;
  return (
    <div className="text-gray-300 text-xs bg-gray-800/50 rounded-md p-2 border border-gray-700">
      <div className="flex items-center text-sm mb-1">
        <BarChart3 className="h-4 w-4 mr-2 text-purple-400" />
        <span className="font-medium">{e.symbol}: decision {e.decision} · conf {Math.round(e.confidence*100)}% @ ${e.price.toLocaleString()}</span>
        <button onClick={() => setOpen(o => !o)} className="ml-auto text-blue-400 hover:text-blue-300">{open ? 'Hide why' : 'Why?'}</button>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <div>RSI: {ind.rsi.toFixed(1)}</div>
          <div>MACD: {ind.macd.toFixed(3)}</div>
          <div>MA20/MA50: {ind.ma20.toFixed(2)} / {ind.ma50.toFixed(2)} ({ind.ma_trend})</div>
        </div>
        <div>
          <div>Rank: {e.fundamentals.market_cap_rank}</div>
          <div>ATH Δ: {e.fundamentals.ath_change_percentage.toFixed(1)}%</div>
          <div>24h: {e.momentum.changePercent.toFixed(2)}% · Vol: {Math.round(e.momentum.volume/1e9)}B</div>
        </div>
      </div>
      <div className="text-gray-400 mt-1">Reasoning: {e.reasoning}</div>
      <div className="text-gray-500 mt-1">Confidence model: combines momentum (|Δ|%), volume rank, MA trend agreement, RSI center-distance, and rank proximity to top-10; boosted by positive market/social sentiment and penalized by cooldown/near-ATH risk.</div>
      {open && (e as any).breakdown && (
        <div className="mt-2 space-y-1">
          {(e as any).breakdown.map((b: any, i: number) => (
            <div key={i} className="flex items-center">
              <div className="w-44 text-gray-300">{b.name}</div>
              <div className="flex-1 bg-gray-700 rounded h-2 mx-2 overflow-hidden">
                <div className="bg-blue-500 h-2" style={{ width: `${Math.round(b.contribution*100)}%` }} />
              </div>
              <div className="w-32 text-right text-gray-400">{(b.contribution*100).toFixed(0)}% · {b.detail}</div>
            </div>
          ))}
          <div className="flex items-center text-gray-400">
            <div className="w-44">Total contribution</div>
            <div className="flex-1" />
            <div className="w-32 text-right">{(total*100).toFixed(0)}%</div>
          </div>
        </div>
      )}
    </div>
  );
};
