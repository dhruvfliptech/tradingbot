import { MarketDataAdapter } from '../../core/adapters';

interface CoinGeckoOptions {
  baseUrl?: string;
  apiKey?: string;
}

export class CoinGeckoMarketDataAdapter implements MarketDataAdapter {
  private baseUrl: string;
  private apiKey?: string;

  constructor(options: CoinGeckoOptions = {}) {
    this.baseUrl = options.baseUrl ?? 'https://api.coingecko.com/api/v3';
    this.apiKey = options.apiKey;
  }

  async fetchWatchlistPrices(symbols: string[]): Promise<Array<{ symbol: string; price: number }>> {
    if (symbols.length === 0) {
      return [];
    }

    const ids = symbols.join(',');
    const url = new URL(`${this.baseUrl}/simple/price`);
    url.searchParams.set('ids', ids);
    url.searchParams.set('vs_currencies', 'usd');

    const response = await fetch(url.toString(), {
      headers: this.apiKey ? { 'x-cg-pro-api-key': this.apiKey } : undefined,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`CoinGecko request failed: ${response.status} ${text}`);
    }

    const data = await response.json() as Record<string, { usd: number }>;

    return symbols.map((symbol) => ({
      symbol,
      price: data[symbol]?.usd ?? 0,
    }));
  }
}
