import { TradingSignal, CryptoData } from '../types/trading';
import { riskManager } from './riskManager';
import { coinGeckoService } from './coinGeckoService';
import { alpacaService } from './alpacaService';
import { groqService } from './groqService';

export type AgentEvent =
  | { type: 'status'; active: boolean; timestamp: string }
  | { type: 'analysis'; evaluated: number; top: Array<{ symbol: string; action: 'BUY' | 'SELL' | 'HOLD'; confidence: number }>; note?: string; timestamp: string }
  | { type: 'decision'; symbol: string; action: 'BUY' | 'SELL' | 'HOLD'; confidence: number; reason: string; price: number; timestamp: string }
  | { type: 'analysis_detail'; symbol: string; price: number; indicators: { rsi: number; macd: number; ma20: number; ma50: number; ma_trend: 'bullish' | 'bearish' | 'neutral' }; fundamentals: { market_cap_rank: number; ath_change_percentage: number }; momentum: { changePercent: number; volume: number }; decision: 'BUY' | 'SELL' | 'HOLD'; confidence: number; reasoning: string; breakdown: Array<{ name: string; score: number; weight: number; contribution: number; detail: string }>; timestamp: string }
  | { type: 'market_sentiment'; overall: 'bullish' | 'bearish' | 'neutral'; confidence: number; reasoning: string; timestamp: string }
  | { type: 'no_trade'; reason: string; timestamp: string }
  | { type: 'order_submitted'; symbol: string; side: 'buy' | 'sell'; qty: number; order_type: 'market' | 'limit'; limit_price?: number | null; timestamp: string }
  | { type: 'order_error'; symbol: string; message: string; timestamp: string };

class TradingAgent {
  private isActive: boolean = false;
  private signals: TradingSignal[] = [];
  private watchlist: string[] = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana'];
  private tickHandle: number | null = null;
  private subscribers: Array<(e: AgentEvent) => void> = [];
  private lastTradeAtBySymbol: Record<string, number> = {};
  private lastAccountSnapshot: { account: any; positions: any[] } | null = null;

  start(): void {
    if (this.isActive) return;
    this.isActive = true;
    this.emit({ type: 'status', active: true, timestamp: new Date().toISOString() });
    this.generateDefaultSignals();
    // kick off trading loop
    this.runCycle();
    this.tickHandle = window.setInterval(() => this.runCycle(), 45 * 1000);
  }

  stop(): void {
    this.isActive = false;
    if (this.tickHandle) {
      clearInterval(this.tickHandle);
      this.tickHandle = null;
    }
    this.emit({ type: 'status', active: false, timestamp: new Date().toISOString() });
  }

  subscribe(handler: (e: AgentEvent) => void): () => void {
    this.subscribers.push(handler);
    return () => {
      this.subscribers = this.subscribers.filter(h => h !== handler);
    };
  }

  private emit(event: AgentEvent) {
    try {
      for (const h of this.subscribers) h(event);
    } catch {}
  }

  private async runCycle(): Promise<void> {
    if (!this.isActive) return;
    try {
      // snapshot account/positions for risk manager
      try {
        const [accountSnap, positionsSnap] = await Promise.all([
          alpacaService.getAccount(),
          alpacaService.getPositions(),
        ]);
        this.lastAccountSnapshot = { account: accountSnap, positions: positionsSnap };
      } catch {}
      // 1) Fetch latest watchlist prices
      const data = await coinGeckoService.getCryptoData(this.watchlist);
      // 2) Generate signals
      const signals = this.analyzeCryptoData(data);
      this.signals = signals;

      // load settings once per cycle
      const { agentSettingsService } = await import('./agentSettingsService');
      const settings = await agentSettingsService.getSettings();

      // Emit analysis snapshot
      const top = [...signals]
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 3)
        .map(s => ({ symbol: s.symbol.toUpperCase(), action: s.action, confidence: s.confidence }));
      this.emit({ type: 'analysis', evaluated: signals.length, top, note: `threshold=${settings.confidenceThreshold}`, timestamp: new Date().toISOString() });

      // Emit per-coin detailed analysis (deep thinking)
      for (const c of data) {
        const rsi = this.calculateRSI(c);
        const macd = this.calculateMACD(c);
        const ma20 = this.calculateMA20(c);
        const ma50 = this.calculateMA50(c);
        const ma_trend: 'bullish' | 'bearish' | 'neutral' = ma20 > ma50 ? 'bullish' : ma50 > ma20 ? 'bearish' : 'neutral';
        const sig = signals.find(s => s.symbol === c.symbol)!

        // Factor scores and weighted breakdown (0..1)
        const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
        const momentumScore = clamp01(c.changePercent / 10);
        const volumeScore = clamp01(c.volume / 2_000_000_000);
        const maTrendScore = clamp01(((ma20 - ma50) / Math.max(1, c.price)) * 200);
        const rsiTarget = 60;
        const rsiScore = clamp01(1 - Math.abs(rsi - rsiTarget) / 40);
        const rankScore = clamp01((10 - Math.min(10, c.market_cap_rank || 50)) / 10);
        const athScore = clamp01(-c.ath_change_percentage / 60);
        const weights = { momentum: 0.30, volume: 0.15, maTrend: 0.20, rsi: 0.10, rank: 0.15, ath: 0.10 };
        const breakdown = [
          { name: 'Momentum (24h Δ%)', score: momentumScore, weight: weights.momentum, contribution: momentumScore * weights.momentum, detail: `${c.changePercent.toFixed(2)}%` },
          { name: 'Volume scale', score: volumeScore, weight: weights.volume, contribution: volumeScore * weights.volume, detail: `${Math.round(c.volume/1e9)}B` },
          { name: 'MA trend (20/50)', score: maTrendScore, weight: weights.maTrend, contribution: maTrendScore * weights.maTrend, detail: `${ma20.toFixed(2)} / ${ma50.toFixed(2)} (${ma_trend})` },
          { name: 'RSI proximity', score: rsiScore, weight: weights.rsi, contribution: rsiScore * weights.rsi, detail: rsi.toFixed(1) },
          { name: 'Market cap rank', score: rankScore, weight: weights.rank, contribution: rankScore * weights.rank, detail: `#${c.market_cap_rank}` },
          { name: 'Distance from ATH', score: athScore, weight: weights.ath, contribution: athScore * weights.ath, detail: `${c.ath_change_percentage.toFixed(1)}%` },
        ];
        this.emit({
          type: 'analysis_detail',
          symbol: c.symbol.toUpperCase(),
          price: c.price,
          indicators: { rsi, macd, ma20, ma50, ma_trend },
          fundamentals: { market_cap_rank: c.market_cap_rank, ath_change_percentage: c.ath_change_percentage },
          momentum: { changePercent: c.changePercent, volume: c.volume },
          decision: sig.action,
          confidence: sig.confidence,
          reasoning: sig.reason,
          breakdown,
          timestamp: new Date().toISOString(),
        });
      }

      // overall market sentiment via LLM (lightweight JSON)
      try {
        const [sentiment, social] = await Promise.all([
          groqService.getMarketSentiment(data),
          groqService.getSocialPulse(data),
        ]);
        this.emit({ type: 'market_sentiment', overall: sentiment.overall, confidence: sentiment.confidence, reasoning: `${sentiment.reasoning} • Social: ${social.overall} (${Math.round(social.confidence*100)}%) — ${social.summary}`, timestamp: new Date().toISOString() });
      } catch {}

      // 3) Decide and possibly trade (apply risk manager)
      let placedAny = false;
      for (const s of signals) {
        if (s.action === 'HOLD') continue;
        if (s.confidence < settings.confidenceThreshold) continue; // require configured confidence
        const now = Date.now();
        const sym = s.symbol.toUpperCase();
        const last = this.lastTradeAtBySymbol[sym] || 0;
        // Cooldown per settings
        if (now - last < settings.cooldownMinutes * 60 * 1000) continue;

        const alpacaSymbol = sym.endsWith('USD') ? sym : `${sym}USD`;
        const side = s.action === 'BUY' ? 'buy' : 'sell';
        // Risk manager sizing & leverage
        riskManager.updatePeak(this.lastAccountSnapshot?.account ?? null);
        const r = riskManager.evaluate({
          account: this.lastAccountSnapshot?.account ?? null,
          positions: this.lastAccountSnapshot?.positions ?? [],
          symbol: sym,
          price: s.current_price,
          indicators: { rsi: s.rsi, macd: s.macd, ma20: s.ma20, ma50: s.ma50 },
          momentum: { changePercent: (s.current_price && s.ma20 ? (s.current_price - (s.ma20)) / s.ma20 * 100 : 0) },
          confidence: s.confidence,
          market: data,
        });
        if (!r.proceed) {
          this.emit({ type: 'no_trade', reason: r.reason, timestamp: new Date().toISOString() });
          continue;
        }

        const price = s.current_price || 0;
        const notional = Math.max(0.0001, r.sizeUsd * r.leverage);
        const qty = price > 0 ? (notional / price).toString() : '0.001';

        this.emit({
          type: 'decision',
          symbol: sym,
          action: s.action,
          confidence: s.confidence,
          reason: `${s.reason} | Risk: ${r.reason} | Stop ${Math.round(r.stopPct*100)}% Target ${Math.round(r.targetPct*100)}% Lev x${r.leverage}`,
          price: s.current_price,
          timestamp: new Date().toISOString(),
        });

        try {
          await alpacaService.placeOrder({ symbol: alpacaSymbol, qty: Number(qty).toString(), side, order_type: 'market' });
          this.lastTradeAtBySymbol[sym] = now;
          this.emit({ type: 'order_submitted', symbol: sym, side, qty: Number(qty), order_type: 'market', timestamp: new Date().toISOString() });
          placedAny = true;
        } catch (err: any) {
          this.emit({ type: 'order_error', symbol: sym, message: err?.message || 'Order failed', timestamp: new Date().toISOString() });
        }
      }
      if (!placedAny) {
        this.emit({ type: 'no_trade', reason: 'No signals above threshold or all in cooldown', timestamp: new Date().toISOString() });
      }
    } catch (err) {
      // swallow; errors are visible in services
    }
  }

  getStatus(): { active: boolean; signalsCount: number; watchlistSize: number } {
    return {
      active: this.isActive,
      signalsCount: this.signals.length,
      watchlistSize: this.watchlist.length,
    };
  }

  getSignals(): TradingSignal[] {
    // If no signals generated yet, create default ones
    if (this.signals.length === 0) {
      this.generateDefaultSignals();
    }
    return this.signals;
  }

  private generateDefaultSignals(): void {
    // Mock signal generation for demo
    const signals: TradingSignal[] = [
      {
        symbol: 'BTC',
        name: 'Bitcoin',
        action: 'BUY',
        confidence: 0.85,
        reason: 'Technical breakout above resistance at $43,000',
        timestamp: new Date().toISOString(),
        price_target: 45000,
        stop_loss: 41000,
        current_price: 43250,
        rsi: 65.2,
        macd: 0.045,
        ma20: 42800,
        ma50: 41500,
        volume_indicator: 'High',
        trend: 'Bullish',
      },
      {
        symbol: 'ETH',
        name: 'Ethereum',
        action: 'SELL',
        confidence: 0.72,
        reason: 'Overbought RSI and resistance at $2,700',
        timestamp: new Date().toISOString(),
        price_target: 2500,
        stop_loss: 2750,
        current_price: 2650,
        rsi: 78.5,
        macd: -0.023,
        ma20: 2680,
        ma50: 2620,
        volume_indicator: 'Medium',
        trend: 'Bearish',
      },
      {
        symbol: 'SOL',
        name: 'Solana',
        action: 'HOLD',
        confidence: 0.68,
        reason: 'Consolidating in range, wait for breakout above $100',
        timestamp: new Date().toISOString(),
        current_price: 98.75,
        rsi: 52.1,
        macd: 0.008,
        ma20: 97.20,
        ma50: 95.80,
        volume_indicator: 'Low',
        trend: 'Neutral',
      },
    ];

    this.signals = signals;
  }

  analyzeCryptoData(data: CryptoData[]): TradingSignal[] {
    // Real-time momentum-based analysis using live data
    return data.map(crypto => ({
      symbol: crypto.symbol,
      name: crypto.name,
      action: this.generateAction(crypto),
      confidence: this.calculateConfidence(crypto),
      reason: this.generateReason(crypto),
      timestamp: new Date().toISOString(),
      price_target: this.calculatePriceTarget(crypto),
      stop_loss: this.calculateStopLoss(crypto),
      current_price: crypto.price,
      rsi: this.calculateRSI(crypto),
      macd: this.calculateMACD(crypto),
      ma20: this.calculateMA20(crypto),
      ma50: this.calculateMA50(crypto),
      volume_indicator: this.getVolumeIndicator(crypto),
      trend: this.getTrend(crypto),
    }));
  }

  private generateAction(crypto: CryptoData): 'BUY' | 'SELL' | 'HOLD' {
    const { changePercent, volume, market_cap_rank } = crypto;
    
    // Strong buy signals
    if (changePercent > 5 && volume > 1000000000 && market_cap_rank <= 10) return 'BUY';
    if (changePercent > 3 && market_cap_rank <= 5) return 'BUY';
    
    // Strong sell signals
    if (changePercent < -5 && volume > 1000000000) return 'SELL';
    if (changePercent < -3 && market_cap_rank > 20) return 'SELL';
    
    // Moderate signals
    if (changePercent > 2) return 'BUY';
    if (changePercent < -2) return 'SELL';
    
    return 'HOLD';
  }

  private calculateConfidence(crypto: CryptoData): number {
    const { changePercent, volume, market_cap_rank } = crypto;
    
    let confidence = Math.min(Math.abs(changePercent) / 10, 0.8);
    
    // Boost confidence for high volume
    if (volume > 2000000000) confidence += 0.1;
    
    // Boost confidence for top-ranked coins
    if (market_cap_rank <= 5) confidence += 0.1;
    if (market_cap_rank <= 10) confidence += 0.05;
    
    return Math.min(confidence, 0.95);
  }

  private generateReason(crypto: CryptoData): string {
    const { changePercent, volume, market_cap_rank, ath_change_percentage } = crypto;
    
    const reasons = [];
    
    if (Math.abs(changePercent) > 5) {
      reasons.push(`Strong ${changePercent > 0 ? 'bullish' : 'bearish'} momentum: ${changePercent.toFixed(2)}%`);
    } else if (Math.abs(changePercent) > 2) {
      reasons.push(`${changePercent > 0 ? 'Positive' : 'Negative'} price action: ${changePercent.toFixed(2)}%`);
    }
    
    if (volume > 2000000000) {
      reasons.push('High trading volume indicates strong interest');
    }
    
    if (market_cap_rank <= 5) {
      reasons.push('Top-tier cryptocurrency with strong fundamentals');
    }
    
    if (ath_change_percentage > -20) {
      reasons.push('Trading near all-time highs');
    } else if (ath_change_percentage < -50) {
      reasons.push('Significant discount from all-time high');
    }
    
    return reasons.length > 0 ? reasons.join('. ') : `Current trend: ${changePercent.toFixed(2)}%`;
  }

  private calculatePriceTarget(crypto: CryptoData): number {
    const { price, changePercent, ath } = crypto;
    
    if (changePercent > 0) {
      // For positive momentum, target 8-15% above current price
      const multiplier = 1 + (Math.min(changePercent, 10) / 100) + 0.08;
      return Math.min(price * multiplier, ath * 0.9); // Don't exceed 90% of ATH
    } else {
      // For negative momentum, target 5-10% below current price
      const multiplier = 1 + (Math.max(changePercent, -10) / 100) - 0.05;
      return price * multiplier;
    }
  }

  private calculateStopLoss(crypto: CryptoData): number {
    const { price, changePercent } = crypto;
    
    if (changePercent > 0) {
      // For bullish signals, stop loss 8-12% below current price
      return price * (1 - (0.08 + Math.min(changePercent / 100, 0.04)));
    } else {
      // For bearish signals, stop loss 5-8% above current price
      return price * (1 + (0.05 + Math.min(Math.abs(changePercent) / 100, 0.03)));
    }
  }

  private calculateRSI(crypto: CryptoData): number {
    // Mock RSI calculation based on price change
    const baseRSI = 50;
    const changeImpact = crypto.changePercent * 2;
    return Math.max(0, Math.min(100, baseRSI + changeImpact + (Math.random() * 20 - 10)));
  }

  private calculateMACD(crypto: CryptoData): number {
    // Mock MACD calculation
    const baseMACD = crypto.changePercent / 100;
    return baseMACD + (Math.random() * 0.02 - 0.01);
  }

  private calculateMA20(crypto: CryptoData): number {
    // Mock 20-day moving average (slightly below current price for uptrend)
    const variation = crypto.changePercent > 0 ? -0.02 : 0.02;
    return crypto.price * (1 + variation + (Math.random() * 0.01 - 0.005));
  }

  private calculateMA50(crypto: CryptoData): number {
    // Mock 50-day moving average (further from current price)
    const variation = crypto.changePercent > 0 ? -0.05 : 0.05;
    return crypto.price * (1 + variation + (Math.random() * 0.02 - 0.01));
  }

  private getVolumeIndicator(crypto: CryptoData): string {
    if (crypto.volume > 2000000000) return 'High';
    if (crypto.volume > 500000000) return 'Medium';
    return 'Low';
  }

  private getTrend(crypto: CryptoData): 'Bullish' | 'Bearish' | 'Neutral' {
    if (crypto.changePercent > 2) return 'Bullish';
    if (crypto.changePercent < -2) return 'Bearish';
    return 'Neutral';
  }
}

export const tradingAgent = new TradingAgent();