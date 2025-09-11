import { Account, Position, CryptoData } from '../types/trading';

export interface RiskDecision {
  proceed: boolean;
  reason: string;
  sizeUsd: number; // notional per trade before leverage
  leverage: number; // 1..5
  stopPct: number; // e.g., 0.01 => 1%
  targetPct: number; // e.g., 0.03 => 3%
}

interface EvaluateArgs {
  account: Account | null;
  positions: Position[];
  symbol: string;
  price: number;
  indicators: { rsi: number; macd: number; ma20: number; ma50: number; };
  momentum: { changePercent: number; };
  confidence: number; // 0..1
  market: CryptoData[];
}

// Stateful peak equity tracker for drawdown control
let peakEquity = 50000; // initial capital default

export const riskManager = {
  updatePeak(account: Account | null) {
    const pv = account?.portfolio_value ?? 0;
    if (pv > peakEquity) peakEquity = pv;
  },

  evaluate(args: EvaluateArgs): RiskDecision {
    const { account, positions, symbol, price, indicators, momentum, confidence, market } = args;
    const portfolioValue = account?.portfolio_value ?? 50000;
    const available = account?.available_balance ?? portfolioValue;

    // Drawdown/pause checks
    const currentDrawdown = peakEquity > 0 ? (peakEquity - portfolioValue) / peakEquity : 0;
    if (currentDrawdown > 0.10) {
      return { proceed: false, reason: `Paused: portfolio drawdown ${(currentDrawdown*100).toFixed(1)}% > 10%`, sizeUsd: 0, leverage: 1, stopPct: 0, targetPct: 0 };
    }

    // Hourly volatility approximation using watchlist dispersion (fallback)
    const dispersion = market.length ? market.reduce((s, c) => s + Math.abs(c.changePercent), 0) / market.length : 0;
    if (dispersion > 5) {
      return { proceed: false, reason: `Paused: market volatility proxy ${(dispersion).toFixed(1)}% > 5%`, sizeUsd: 0, leverage: 1, stopPct: 0, targetPct: 0 };
    }

    // Indicator cross validation (need 3+ positives for BUY; for SELL inverse but our agent currently does BUY focus)
    let positives = 0;
    if (indicators.ma20 > indicators.ma50) positives++;
    if (indicators.rsi > 55) positives++;
    if (indicators.macd > 0) positives++;
    if (momentum.changePercent > 0) positives++;
    if (confidence >= 0.7) positives++;
    if (positives < 3) {
      return { proceed: false, reason: `Rejected: only ${positives} confirmations (<3)`, sizeUsd: 0, leverage: 1, stopPct: 0, targetPct: 0 };
    }

    // Position sizing 5-10% of capital, clipped by available cash
    const baseSize = Math.min(portfolioValue * 0.10, Math.max(portfolioValue * 0.05, portfolioValue * (confidence - 0.5))); // scale with confidence
    const sizeUsd = Math.min(baseSize, available);

    // Leverage up to 5x for high confidence
    let leverage = 1;
    if (confidence > 0.9) leverage = 5; else if (confidence > 0.85) leverage = 4; else if (confidence > 0.8) leverage = 3; else if (confidence > 0.75) leverage = 2; else leverage = 1;

    // Correlation cap: sum notional for BTC/ETH-like symbols <=10% of capital
    const correlated = ['BTC', 'ETH'];
    const group = correlated.includes(symbol.toUpperCase()) ? correlated : [];
    if (group.length) {
      const existing = positions.filter(p => group.includes((p.symbol || '').toUpperCase()));
      const existingValue = existing.reduce((s, p) => s + (p.market_value || 0), 0);
      if ((existingValue + sizeUsd) > portfolioValue * 0.10) {
        return { proceed: false, reason: 'Rejected: correlation cap 10% exceeded for BTC/ETH group', sizeUsd: 0, leverage: 1, stopPct: 0, targetPct: 0 };
      }
    }

    // Risk/Reward parameters
    const stopPct = 0.01; // 1%
    const targetPct = 0.03; // 3% (≈3:1)

    return {
      proceed: true,
      reason: `Validated (${positives} confirmations), DD ${(currentDrawdown*100).toFixed(1)}%, size ${(sizeUsd).toFixed(0)} USD, lev x${leverage}`,
      sizeUsd,
      leverage,
      stopPct,
      targetPct,
    };
  },
};


