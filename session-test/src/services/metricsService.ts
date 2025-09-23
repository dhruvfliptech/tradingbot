import { supabase } from '../lib/supabase';
import { Order } from '../types/trading';
import { tradingProviderService } from './tradingProviderService';

export interface ComputedMetrics {
  totalReturnPercent: number;
  dayReturnPercent: number;
  sharpe: number;
  totalTrades: number;
  winRate: number;        // 0..1
  avgTradeReturn: number; // percent
  bestTradeReturn: number;
  worstTradeReturn: number;
}

function computeSharpe(dailyReturns: number[], riskFree: number = 0): number {
  if (dailyReturns.length === 0) return 0;
  const mean = dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length;
  const excess = mean - riskFree / 252; // risk-free annualized to daily approx
  const variance = dailyReturns
    .map(r => (r - mean) ** 2)
    .reduce((a, b) => a + b, 0) / dailyReturns.length;
  const std = Math.sqrt(variance) || 1e-9;
  const sharpeDaily = excess / std;
  return sharpeDaily * Math.sqrt(252); // annualized Sharpe
}

interface ClosedTrade {
  symbol: string;
  closeDate: string; // ISO
  basisUsd: number;  // invested USD for the closed lot
  pnlUsd: number;    // realized PnL in USD
  retPct: number;    // percentage return for that lot (pnl/basis*100)
}

function computeClosedTrades(orders: Order[]): ClosedTrade[] {
  // Build FIFO inventory per symbol for long-only PnL pairing
  type Lot = { qty: number; costBasis: number }; // costBasis per unit (price)
  const inv: Record<string, Lot[]> = {};
  const events = orders
    .filter(o => (o.status || '').toLowerCase() === 'filled')
    .map(o => {
      const ts = o.filled_at || o.submitted_at || new Date().toISOString();
      const side = (o.side || '').toLowerCase();
      const qty = Number((o as any).filled_qty ?? (o as any).qty ?? 0);
      const price = Number(o.filled_avg_price ?? (o as any).limit_price ?? 0);
      return { ts, side, symbol: (o.symbol || '').toUpperCase(), qty, price };
    })
    .filter(e => e.qty > 0 && e.price > 0)
    .sort((a, b) => new Date(a.ts).getTime() - new Date(b.ts).getTime());

  const closed: ClosedTrade[] = [];

  for (const e of events) {
    inv[e.symbol] = inv[e.symbol] || [];
    if (e.side === 'buy') {
      // add lot
      inv[e.symbol].push({ qty: e.qty, costBasis: e.price });
    } else if (e.side === 'sell') {
      // close against FIFO
      let remaining = e.qty;
      while (remaining > 0 && inv[e.symbol].length > 0) {
        const lot = inv[e.symbol][0];
        const take = Math.min(remaining, lot.qty);
        const basisUsd = take * lot.costBasis;
        const pnlUsd = take * (e.price - lot.costBasis);
        const retPct = basisUsd > 0 ? (pnlUsd / basisUsd) * 100 : 0;
        closed.push({ symbol: e.symbol, closeDate: e.ts, basisUsd, pnlUsd, retPct });
        lot.qty -= take;
        remaining -= take;
        if (lot.qty <= 0.0000001) inv[e.symbol].shift();
      }
      // If inventory was empty, skip (shorts not handled in v1)
    }
  }

  return closed;
}

function buildDailyReturnSeries(closed: ClosedTrade[]): number[] {
  if (closed.length === 0) return [];
  // Aggregate by day: sum pnl / sum basis
  const map = new Map<string, { pnl: number; basis: number }>();
  for (const t of closed) {
    const day = new Date(t.closeDate);
    const key = `${day.getFullYear()}-${day.getMonth()+1}-${day.getDate()}`;
    const prev = map.get(key) || { pnl: 0, basis: 0 };
    prev.pnl += t.pnlUsd;
    prev.basis += t.basisUsd;
    map.set(key, prev);
  }
  return Array.from(map.values())
    .map(v => (v.basis > 0 ? v.pnl / v.basis : 0)); // daily return in decimal
}

export class MetricsService {
  async computeFromOrders(): Promise<ComputedMetrics> {
    const orders = await tradingProviderService.getOrders();
    const closed = computeClosedTrades(orders);

    const totalBasis = closed.reduce((s, t) => s + t.basisUsd, 0);
    const totalPnl = closed.reduce((s, t) => s + t.pnlUsd, 0);
    const totalReturnPercent = totalBasis > 0 ? (totalPnl / totalBasis) * 100 : 0;

    // Today return
    const todayKey = (() => { const d = new Date(); return `${d.getFullYear()}-${d.getMonth()+1}-${d.getDate()}`; })();
    let dayBasis = 0, dayPnl = 0;
    for (const t of closed) {
      const d = new Date(t.closeDate); const key = `${d.getFullYear()}-${d.getMonth()+1}-${d.getDate()}`;
      if (key === todayKey) { dayBasis += t.basisUsd; dayPnl += t.pnlUsd; }
    }
    const dayReturnPercent = dayBasis > 0 ? (dayPnl / dayBasis) * 100 : 0;

    // Per-trade stats
    const tradeReturns = closed.map(t => t.retPct);
    const totalTrades = tradeReturns.length;
    const wins = tradeReturns.filter(r => r > 0).length;
    const avgTradeReturn = totalTrades ? (tradeReturns.reduce((a,b)=>a+b,0) / totalTrades) : 0;
    const bestTradeReturn = totalTrades ? Math.max(...tradeReturns) : 0;
    const worstTradeReturn = totalTrades ? Math.min(...tradeReturns) : 0;

    // Sharpe on daily series
    const dailySeries = buildDailyReturnSeries(closed);
    const sharpe = computeSharpe(dailySeries, 0);

    return {
      totalReturnPercent,
      dayReturnPercent,
      sharpe,
      totalTrades,
      winRate: totalTrades ? wins / totalTrades : 0,
      avgTradeReturn,
      bestTradeReturn,
      worstTradeReturn,
    };
  }

  async upsertOverall(metrics: ComputedMetrics): Promise<void> {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;
    await supabase
      .from('performance_metrics')
      .upsert({
        user_id: user.id,
        window: 'overall',
        total_return_percent: metrics.totalReturnPercent,
        day_return_percent: metrics.dayReturnPercent,
        sharpe: metrics.sharpe,
        total_trades: metrics.totalTrades,
        win_rate: metrics.winRate,
        avg_trade_return: metrics.avgTradeReturn,
        best_trade_return: metrics.bestTradeReturn,
        worst_trade_return: metrics.worstTradeReturn,
        updated_at: new Date().toISOString(),
      }, { onConflict: 'user_id,window,window_date' });
  }

  async getOverall(): Promise<ComputedMetrics | null> {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return null;
    const { data } = await supabase
      .from('performance_metrics')
      .select('total_return_percent, day_return_percent, sharpe, total_trades, win_rate, avg_trade_return, best_trade_return, worst_trade_return')
      .eq('user_id', user.id)
      .eq('window', 'overall')
      .maybeSingle();
    if (!data) return null;
    return {
      totalReturnPercent: data.total_return_percent ?? 0,
      dayReturnPercent: data.day_return_percent ?? 0,
      sharpe: data.sharpe ?? 0,
      totalTrades: data.total_trades ?? 0,
      winRate: data.win_rate ?? 0,
      avgTradeReturn: data.avg_trade_return ?? 0,
      bestTradeReturn: data.best_trade_return ?? 0,
      worstTradeReturn: data.worst_trade_return ?? 0,
    };
  }
}

export const metricsService = new MetricsService();


