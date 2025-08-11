export type SearchInsight = {
  id: string;
  title: string;
  summary: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  relevantCoins: string[];
  confidence: number; // 0..1
  timestamp: string;
  sources?: Array<{ title?: string; url: string }>;
};

// Call Netlify serverless proxy to avoid CORS for Brave/Tavily
export async function searchNewsInsights(symbols: string[], limit = 5): Promise<SearchInsight[]> {
  const payload = { symbols, limit };
  try {
    const resp = await fetch('/api/search-news', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) return [];
    const data = await resp.json().catch(() => ({ insights: [] }));
    const insights: SearchInsight[] = Array.isArray(data?.insights) ? data.insights : [];
    return insights;
  } catch {
    // Silent fallback; UI will use LLM-only insights
    return [];
  }
}


