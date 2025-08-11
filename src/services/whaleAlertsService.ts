export type WhaleTx = {
  id: string;
  symbol: string;
  amount: number;
  usdValue?: number;
  timestamp: string;
  note?: string;
  txUrl?: string;
};

const WHALE_ALERT_API_KEY = import.meta.env.VITE_WHALE_ALERT_API_KEY as string | undefined;

export async function fetchWhaleAlerts(symbols: string[] = ['BTC','ETH','SOL'], minUsd = 1_000_000): Promise<WhaleTx[]> {
  if (!WHALE_ALERT_API_KEY) {
    // Fallback sample
    const iso = new Date().toISOString();
    return [
      { id: 'sample-1', symbol: 'BTC', amount: 2100, usdValue: 120_000_000, timestamp: iso, note: 'Exchange inflow' },
      { id: 'sample-2', symbol: 'ETH', amount: 30000, usdValue: 90_000_000, timestamp: iso, note: 'Cold wallet movement' },
      { id: 'sample-3', symbol: 'SOL', amount: 1500000, usdValue: 105_000_000, timestamp: iso, note: 'Exchange outflow' },
    ];
  }

  const now = Math.floor(Date.now() / 1000);
  const start = now - 60 * 60;
  const url = new URL('https://api.whale-alert.io/v1/transactions');
  url.searchParams.set('api_key', WHALE_ALERT_API_KEY);
  url.searchParams.set('start', String(start));
  url.searchParams.set('min_value', String(minUsd));
  url.searchParams.set('currency', 'usd');

  const resp = await fetch(url.toString());
  if (!resp.ok) return [];
  const data = await resp.json();
  const txs: any[] = data?.transactions || [];
  const events: WhaleTx[] = txs
    .filter(t => (t.amount_usd || 0) >= minUsd)
    .map((t, idx) => ({
      id: String(t.id || `${t.hash || idx}`),
      symbol: String((t.symbol || t.blockchain || '').toUpperCase()),
      amount: Number(t.amount || 0),
      usdValue: Number(t.amount_usd || 0),
      timestamp: new Date((t.timestamp || now) * 1000).toISOString(),
      txUrl: t.hash ? `https://www.blockchain.com/tx/${t.hash}` : undefined,
      note: t.transaction_type || undefined,
    }))
    .filter(e => symbols.includes(e.symbol))
    .slice(0, 20);
  return events;
}


