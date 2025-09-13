import React, { useMemo, useRef, useState, useEffect } from 'react';
import { MessageSquare, X, Send, ChevronDown } from 'lucide-react';
import { Account, Position, Order, CryptoData } from '../../types/trading';
import { askDashboardAssistant } from '../../services/groqService';
import { alpacaService } from '../../services/alpacaService';
import { tradingAgentV2 as tradingAgent } from '../../services/tradingAgentV2';

interface DashboardChatProps {
  account: Account | null;
  positions: Position[];
  orders: Order[];
  cryptoData: CryptoData[];
  apiStatuses: Record<string, 'connected' | 'error' | 'checking'>;
  onPostTrade?: () => void;
  embedded?: boolean;
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export const DashboardChat: React.FC<DashboardChatProps> = ({
  account,
  positions,
  orders,
  cryptoData,
  apiStatuses,
  onPostTrade,
  embedded = false,
}) => {
  const [isOpen, setIsOpen] = useState(embedded);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content:
        'Hi! I am your trading copilot. Ask me about your account, watchlist, positions, recent orders, signals, or overall market sentiment.',
    },
  ]);

  const endRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen]);

  const signals = useMemo(() => {
    try {
      if (cryptoData && cryptoData.length > 0) {
        return tradingAgent.analyzeCryptoData(cryptoData).map(s => ({
          symbol: s.symbol,
          action: s.action,
          confidence: s.confidence,
        }));
      }
      return tradingAgent.getSignals().map(s => ({
        symbol: s.symbol,
        action: s.action,
        confidence: s.confidence,
      }));
    } catch {
      return [] as Array<{ symbol: string; action: 'BUY' | 'SELL' | 'HOLD'; confidence: number }>;
    }
  }, [cryptoData]);

  const normalizeSymbol = (raw: string): string => {
    const cleaned = raw.toUpperCase().replace(/[^A-Z]/g, '');
    if (!cleaned) return '';
    if (cleaned.endsWith('USD')) return cleaned;
    return `${cleaned}USD`;
  };

  const tryExecuteTradeFromText = async (text: string): Promise<boolean> => {
    const lower = text.toLowerCase();
    // Start/Stop auto-trader
    if (/(start|enable)\s+(auto|agent|auto[- ]?trade|auto[- ]?trading)/.test(lower) || /start\s+auto\s*trade/.test(lower)) {
      try {
        tradingAgent.start();
        setMessages(prev => [...prev, { role: 'assistant', content: 'Auto-trader started. It will analyze markets each cycle and place trades based on your settings. See Auto-Trade Activity for live decisions.' }]);
      } catch {}
      return true;
    }
    if (/(stop|disable)\s+(auto|agent|auto[- ]?trade|auto[- ]?trading)/.test(lower) || /stop\s+auto\s*trade/.test(lower)) {
      try {
        tradingAgent.stop();
        setMessages(prev => [...prev, { role: 'assistant', content: 'Auto-trader stopped.' }]);
      } catch {}
      return true;
    }
    // Sell all / close all
    if (/(sell|close)\s+(all|everything)|all of them/.test(lower)) {
      const livePositions = await alpacaService.getPositions();
      if (!livePositions || livePositions.length === 0) {
        setMessages(prev => [...prev, { role: 'assistant', content: 'No open positions to sell.' }]);
        return true;
      }
      for (const p of livePositions) {
        const symbol = normalizeSymbol((p as any).symbol || '');
        const qty = (p as any).qty || '0';
        if (!symbol || qty === '0') continue;
        await alpacaService.placeOrder({ symbol, qty, side: 'sell', order_type: 'market' });
      }
      onPostTrade?.();
      setMessages(prev => [...prev, { role: 'assistant', content: 'Submitted market sell orders for all current positions.' }]);
      return true;
    }
    // Buy/Sell N SYMBOL
    const m = lower.match(/\b(buy|sell)\s+([0-9]+(?:\.[0-9]+)?)\s*([a-z]{2,10})\b/);
    if (m) {
      const side = m[1] as 'buy' | 'sell';
      const qty = m[2];
      const base = m[3];
      const symbol = normalizeSymbol(base);
      if (!symbol) return false;
      await alpacaService.placeOrder({ symbol, qty, side, order_type: 'market' });
      onPostTrade?.();
      setMessages(prev => [...prev, { role: 'assistant', content: `Order placed: ${side.toUpperCase()} ${qty} ${symbol} (market).` }]);
      return true;
    }
    return false;
  };

  const ask = async () => {
    const question = input.trim();
    if (!question) return;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: question }]);
    setSending(true);
    try {
      // First, see if the user asked for a trade action
      const executed = await tryExecuteTradeFromText(question);
      if (executed) return;
      const answer = await askDashboardAssistant(question, {
        account: account ? {
          // Map our Account shape to the assistant context fields
          buying_power: Number((account as any).available_balance ?? 0),
          portfolio_value: Number((account as any).portfolio_value ?? 0),
          cash: Number((account as any).balance_usd ?? 0),
        } : null,
        positions: positions?.map(p => ({
          symbol: p.symbol,
          qty: (p as any).qty ?? (p as any).quantity ?? 0,
          avg_cost_basis: (p as any).avg_cost_basis ?? (p as any).avg_entry_price ?? undefined,
          side: (p as any).side,
          current_price: (p as any).current_price ?? undefined,
        })),
        orders: orders?.slice(0, 20).map(o => ({
          symbol: o.symbol,
          side: o.side,
          order_type: (o as any).order_type ?? o.type,
          status: o.status,
          quantity: Number((o as any).quantity ?? (o as any).qty ?? 0),
          limit_price: (o as any).limit_price,
          submitted_at: (o as any).submitted_at ?? (o as any).created_at,
        })),
        cryptoData,
        signals,
        apiStatuses,
      });
      setMessages(prev => [...prev, { role: 'assistant', content: answer }]);
    } catch (e: any) {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Sorry, I could not fetch an AI answer right now.' },
      ]);
    } finally {
      setSending(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!sending) ask();
    }
  };

  // Embedded mode: render directly without floating button
  if (embedded) {
    return (
      <div className="h-full bg-gray-900 border border-gray-700 rounded-lg flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.map((m, idx) => (
            <div key={idx} className={m.role === 'user' ? 'text-right' : 'text-left'}>
              <div
                className={
                  'inline-block rounded-lg px-3 py-2 whitespace-pre-wrap ' +
                  (m.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-100 border border-gray-700')
                }
              >
                {m.content}
              </div>
            </div>
          ))}
          <div ref={endRef} />
        </div>
        <div className="p-3 border-t border-gray-700 bg-gray-850">
          <div className="flex items-center space-x-2">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder={sending ? 'Thinking…' : 'Ask about trading strategies, market analysis, or agent decisions…'}
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-600"
              disabled={sending}
            />
            <button
              onClick={ask}
              disabled={sending || !input.trim()}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg flex items-center"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Floating mode: render with toggle button
  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(o => !o)}
        className="fixed bottom-5 right-5 z-50 rounded-full bg-blue-600 hover:bg-blue-700 text-white p-3 shadow-lg focus:outline-none"
        aria-label="Open AI Assistant"
      >
        {isOpen ? <ChevronDown className="h-5 w-5" /> : <MessageSquare className="h-5 w-5" />}
      </button>

      {/* Panel */}
      {isOpen && (
        <div className="fixed bottom-20 right-5 z-50 w-[92vw] sm:w-[420px] max-h-[70vh] bg-gray-900 border border-gray-700 rounded-xl shadow-2xl flex flex-col overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 bg-gray-800 border-b border-gray-700">
            <div className="text-white font-semibold">AI Trading Assistant</div>
            <button onClick={() => setIsOpen(false)} className="text-gray-300 hover:text-white">
              <X className="h-5 w-5" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.map((m, idx) => (
              <div key={idx} className={m.role === 'user' ? 'text-right' : 'text-left'}>
                <div
                  className={
                    'inline-block rounded-lg px-3 py-2 whitespace-pre-wrap ' +
                    (m.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-100 border border-gray-700')
                  }
                >
                  {m.content}
                </div>
              </div>
            ))}
            <div ref={endRef} />
          </div>
          <div className="p-3 border-t border-gray-700 bg-gray-850">
            <div className="flex items-center space-x-2">
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={sending ? 'Thinking…' : 'Ask about your dashboard…'}
                className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-600"
                disabled={sending}
              />
              <button
                onClick={ask}
                disabled={sending || !input.trim()}
                className="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg flex items-center"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default DashboardChat;


