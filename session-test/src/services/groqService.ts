import { CryptoData } from '../types/trading';
import { apiKeysService } from './apiKeysService';

interface MarketInsight {
  id: string;
  title: string;
  summary: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  relevantCoins: string[];
  timestamp: string;
  confidence: number;
  sources?: Array<{ title?: string; url: string }>;
}

interface GroqResponse {
  choices: Array<{
    message: {
      content: string;
    };
  }>;
}

class GroqService {
  private apiKey: string;
  private baseUrl: string;
  private lastRequestTime: number = 0;
  private minRequestInterval: number = 3000; // 3 seconds between requests

  constructor() {
    this.apiKey = import.meta.env.VITE_GROQ_API_KEY || '';
    this.baseUrl = 'https://api.groq.com/openai/v1';
    this.initializeApiKey();
  }

  private async initializeApiKey() {
    try {
      const storedKey = await apiKeysService.getApiKeyWithFallback('groq', 'api_key');
      if (storedKey) {
        this.apiKey = storedKey;
        console.log('✅ Groq API key loaded from stored keys');
      } else if (this.apiKey) {
        console.log('📋 Using Groq API key from environment variables');
      } else {
        console.warn('⚠️ No Groq API key found in stored keys or environment');
      }
    } catch (error) {
      console.warn('⚠️ Failed to load Groq API key from stored keys, using environment fallback');
    }
  }

  private async makeRequest(messages: Array<{ role: string; content: string }>): Promise<string> {
    // Try to get the latest API key
    const apiKey = await apiKeysService.getApiKeyWithFallback('groq', 'api_key');
    
    if (!apiKey) {
      throw new Error('Groq API key not configured. Please add it in Settings > API Keys or set VITE_GROQ_API_KEY in your environment variables.');
    }

    // Rate limiting: ensure minimum time between requests
    const now = Date.now();
    const timeSinceLastRequest = now - this.lastRequestTime;
    if (timeSinceLastRequest < this.minRequestInterval) {
      await new Promise(resolve => setTimeout(resolve, this.minRequestInterval - timeSinceLastRequest));
    }
    this.lastRequestTime = Date.now();

    try {
      const response = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'meta-llama/llama-4-maverick-17b-128e-instruct',
          messages,
          temperature: 0.7,
          max_tokens: 2000,
        }),
      });

      if (!response.ok) {
        if (response.status === 429) {
          throw new Error('Rate limit exceeded. Please wait before making more requests.');
        }
        throw new Error(`Groq API error: ${response.status}`);
      }

      const data: GroqResponse = await response.json();
      return data.choices[0]?.message?.content || '';
    } catch (error) {
      console.error('Groq API request failed:', error);
      throw error;
    }
  }

  async generateMarketInsights(cryptoData: CryptoData[]): Promise<MarketInsight[]> {
    try {
      const marketSummary = this.createMarketSummary(cryptoData);
      
      const prompt = `As a crypto market analyst, analyze the following real-time market data and generate 5 insightful market analysis articles. Each article should be actionable and relevant to current market conditions.

Market Data:
${marketSummary}

Please respond with a JSON array of exactly 5 market insights in this format:
[
  {
    "id": "unique-id",
    "title": "Compelling article title",
    "summary": "2-3 sentence summary with actionable insights",
    "sentiment": "bullish|bearish|neutral",
    "relevantCoins": ["BTC", "ETH"],
    "confidence": 0.85
  }
]

Focus on:
- Current price movements and trends
- Market sentiment indicators
- Technical analysis insights
- Risk management advice
- Trading opportunities

Make each insight unique and valuable for crypto traders.`;

      const response = await this.makeRequest([
        { role: 'user', content: prompt }
      ]);

      // Parse the JSON response
      const jsonMatch = response.match(/\[[\s\S]*\]/);
      if (!jsonMatch) {
        throw new Error('Invalid response format from Groq API');
      }

      const insights: MarketInsight[] = JSON.parse(jsonMatch[0]);
      
      // Add timestamps and ensure proper format
      return insights.map((insight, index) => ({
        ...insight,
        id: insight.id || `insight-${Date.now()}-${index}`,
        timestamp: new Date().toISOString(),
        confidence: insight.confidence || 0.75,
      }));

    } catch (error) {
      return this.getFallbackInsights();
    }
  }

  private createMarketSummary(cryptoData: CryptoData[]): string {
    return cryptoData.map(crypto => 
      `${crypto.symbol}: $${crypto.price.toLocaleString()} (${crypto.changePercent >= 0 ? '+' : ''}${crypto.changePercent.toFixed(2)}%) - Vol: $${(crypto.volume / 1000000000).toFixed(1)}B - Cap: $${(crypto.market_cap / 1000000000).toFixed(1)}B`
    ).join('\n');
  }

  getFallbackInsights(): MarketInsight[] {
    return [
      {
        id: 'fallback-1',
        title: 'Bitcoin Consolidation Phase Presents Opportunity',
        summary: 'BTC is showing signs of consolidation around key support levels. This sideways movement often precedes significant breakouts, making it an ideal time for position building.',
        sentiment: 'neutral',
        relevantCoins: ['BTC'],
        timestamp: new Date().toISOString(),
        confidence: 0.75,
      },
      {
        id: 'fallback-2',
        title: 'Ethereum Layer 2 Adoption Driving Long-term Value',
        summary: 'Increased L2 activity and reduced gas fees are making Ethereum more accessible. This infrastructure improvement supports bullish long-term fundamentals.',
        sentiment: 'bullish',
        relevantCoins: ['ETH'],
        timestamp: new Date().toISOString(),
        confidence: 0.80,
      },
      {
        id: 'fallback-3',
        title: 'Altcoin Season Indicators Remain Mixed',
        summary: 'While some altcoins show strength, Bitcoin dominance remains elevated. Traders should be selective and focus on fundamentally strong projects.',
        sentiment: 'neutral',
        relevantCoins: ['BNB', 'SOL', 'ADA'],
        timestamp: new Date().toISOString(),
        confidence: 0.70,
      },
      {
        id: 'fallback-4',
        title: 'Risk Management Critical in Current Market',
        summary: 'Volatility remains elevated across crypto markets. Implementing proper stop-losses and position sizing is essential for capital preservation.',
        sentiment: 'bearish',
        relevantCoins: ['BTC', 'ETH'],
        timestamp: new Date().toISOString(),
        confidence: 0.85,
      },
      {
        id: 'fallback-5',
        title: 'DeFi Protocols Show Resilience Despite Market Uncertainty',
        summary: 'Total Value Locked in DeFi remains stable, indicating strong underlying demand. This suggests potential opportunities in quality DeFi tokens.',
        sentiment: 'bullish',
        relevantCoins: ['ETH', 'BNB'],
        timestamp: new Date().toISOString(),
        confidence: 0.75,
      },
    ];
  }

  async getMarketSentiment(
    cryptoData: CryptoData[],
    ctx?: { breadth?: number; avgAbs?: number; newsPolarity?: number }
  ): Promise<{
    overall: 'bullish' | 'bearish' | 'neutral';
    confidence: number;
    reasoning: string;
  }> {
    try {
      const marketSummary = this.createMarketSummary(cryptoData);
      
      const prompt = `Analyze the following crypto market data and provide an overall market sentiment assessment:

${marketSummary}

Respond with JSON in this exact format:
{
  "overall": "bullish|bearish|neutral",
  "confidence": 0.85,
  "reasoning": "Brief explanation of the sentiment analysis"
}`;

      const response = await this.makeRequest([
        { role: 'user', content: prompt }
      ]);

      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        // Blend confidence using price breadth/volatility + news tone + LLM output
        const avgAbs = typeof ctx?.avgAbs === 'number'
          ? ctx!.avgAbs
          : (cryptoData.length
            ? cryptoData.reduce((s, c) => s + Math.abs(c.changePercent || 0), 0) / cryptoData.length
            : 0);
        const breadth = typeof ctx?.breadth === 'number'
          ? Math.max(0, Math.min(1, ctx!.breadth))
          : (cryptoData.length
            ? cryptoData.filter(c => (c.changePercent || 0) > 0).length / cryptoData.length
            : 0.5);
        const newsPolarity = typeof ctx?.newsPolarity === 'number'
          ? Math.max(0, Math.min(1, ctx!.newsPolarity))
          : 0.5; // neutral if unknown

        // Normalize avgAbs: 0..1 where 8% avg abs move ~ 1.0
        const priceMoveScore = Math.max(0, Math.min(1, avgAbs / 8));

        // LLM confidence: ignore example constant 0.85
        let llmConf = Number(parsed.confidence);
        if (!isFinite(llmConf) || Math.abs(llmConf - 0.85) < 1e-6) llmConf = 0.5;

        // Weighted blend
        let conf = 0.4 * llmConf + 0.2 * priceMoveScore + 0.2 * breadth + 0.2 * newsPolarity;
        conf = Math.max(0.10, Math.min(0.95, conf));
        conf = Math.round(conf * 100) / 100;
        return { overall: parsed.overall, confidence: conf, reasoning: String(parsed.reasoning || '').trim() };
      }

      throw new Error('Invalid sentiment response format');
    } catch (error) {
      return {
        overall: 'neutral',
        confidence: 0.5,
        reasoning: 'Unable to analyze current market conditions due to API limitations.',
      };
    }
  }

  async getSocialPulse(cryptoData: CryptoData[]): Promise<{
    overall: 'positive' | 'negative' | 'mixed';
    confidence: number;
    summary: string;
  }> {
    try {
      const marketSummary = this.createMarketSummary(cryptoData);
      const prompt = `You are monitoring crypto social media (Twitter/X, Reddit, Telegram) in aggregate.
Based only on the following market snapshot, infer likely social sentiment and topics (do not fabricate metrics):

${marketSummary}

Respond as strict JSON:
{
  "overall": "positive|negative|mixed",
  "confidence": 0.0,
  "summary": "one or two sentences about what people likely discuss (momentum, ETFs, upgrades, hacks, etc.)"
}`;
      const response = await this.makeRequest([{ role: 'user', content: prompt }]);
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) return JSON.parse(jsonMatch[0]);
      throw new Error('Invalid social response');
    } catch {
      return { overall: 'mixed', confidence: 0.5, summary: 'Social sentiment unclear; chatter likely tracks the same leaders with attention spikes on volatility.' };
    }
  }
}

export const groqService = new GroqService();
export type { MarketInsight };
 
// --- AI Dashboard Q&A helper ---
export interface DashboardContext {
  account?: {
    buying_power?: number;
    portfolio_value?: number;
    cash?: number;
  } | null;
  positions?: Array<{
    symbol: string;
    qty: number | string;
    avg_cost_basis?: number;
    side?: string;
    current_price?: number;
  }>;
  orders?: Array<{
    symbol: string;
    side: string;
    order_type?: string;
    status?: string;
    quantity?: number;
    limit_price?: number;
    submitted_at?: string;
  }>;
  cryptoData?: CryptoData[];
  signals?: Array<{
    symbol: string;
    action: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
  }>;
  apiStatuses?: Record<string, 'connected' | 'error' | 'checking'>;
}

/**
 * Build a compact, model-friendly summary string of the current dashboard data.
 */
export function buildDashboardSummary(ctx: DashboardContext): string {
  const parts: string[] = [];

  if (ctx.account) {
    parts.push(
      `Account: portfolio_value=$${(ctx.account.portfolio_value ?? 0).toLocaleString()}, ` +
        `cash=$${(ctx.account.cash ?? 0).toLocaleString()}, ` +
        `buying_power=$${(ctx.account.buying_power ?? 0).toLocaleString()}`
    );
  }

  if (ctx.positions && ctx.positions.length > 0) {
    const top = ctx.positions.slice(0, 10)
      .map(p => `${p.symbol}:${typeof p.qty === 'string' ? p.qty : (p.qty ?? 0)}@${p.avg_cost_basis ?? '—'}`)
      .join(', ');
    parts.push(`Positions(${ctx.positions.length}): ${top}`);
  }

  if (ctx.orders && ctx.orders.length > 0) {
    const recent = ctx.orders.slice(0, 5)
      .map(o => `${o.side} ${o.symbol} ${o.quantity ?? ''} ${o.status ?? ''}`.trim())
      .join('; ');
    parts.push(`RecentOrders(${ctx.orders.length}): ${recent}`);
  }

  if (ctx.cryptoData && ctx.cryptoData.length > 0) {
    const top = ctx.cryptoData.slice(0, 10)
      .map(c => `${c.symbol.toUpperCase()}:$${c.price.toFixed(2)}(${c.changePercent.toFixed(2)}%)`)
      .join(', ');
    parts.push(`Watchlist(${ctx.cryptoData.length}): ${top}`);
  }

  if (ctx.signals && ctx.signals.length > 0) {
    const brief = ctx.signals.slice(0, 10)
      .map(s => `${s.symbol}:${s.action}(${Math.round(s.confidence * 100)}%)`)
      .join(', ');
    parts.push(`Signals(${ctx.signals.length}): ${brief}`);
  }

  if (ctx.apiStatuses) {
    const api = Object.entries(ctx.apiStatuses)
      .map(([k, v]) => `${k}:${v}`)
      .join(', ');
    parts.push(`API:${api}`);
  }

  return parts.join(' | ');
}

/**
 * High-level Q&A over the dashboard context. Returns a concise markdown answer.
 */
export async function askDashboardAssistant(
  question: string,
  ctx: DashboardContext
): Promise<string> {
  const groq = groqService as any as GroqService; // use same instance
  const summary = buildDashboardSummary(ctx);

  const system = `You are a concise crypto trading copilot embedded in a dashboard. 
Answer user questions strictly using the provided context. If data is missing, say so briefly.
Keep answers short, bullet where helpful, include concrete numbers from context. Avoid speculation.`;

  const user = `Context:\n${summary}\n\nQuestion: ${question}`;

  try {
    const content = await (groq as any).makeRequest([
      { role: 'system', content: system },
      { role: 'user', content: user },
    ]);
    return content?.trim() || 'No answer available.';
  } catch (err) {
    // Fallback: simple heuristic summary
    return (
      `AI assistant is temporarily unavailable. Here is a quick summary you can use:\n\n` +
      summary
    );
  }
}