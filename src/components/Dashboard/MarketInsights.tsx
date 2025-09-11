import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Minus, Brain, RefreshCw, Clock, Target } from 'lucide-react';
import { groqService, MarketInsight } from '../../services/groqService';
import { searchNewsInsights } from '../../services/newsSearchService';
import { CryptoData } from '../../types/trading';

interface MarketInsightsProps {
  cryptoData: CryptoData[];
}

export const MarketInsights: React.FC<MarketInsightsProps> = ({ cryptoData }) => {
  const [insights, setInsights] = useState<MarketInsight[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [overallSentiment, setOverallSentiment] = useState<{
    overall: 'bullish' | 'bearish' | 'neutral';
    confidence: number;
    reasoning: string;
  } | null>(null);

  const fetchInsights = async () => {
    if (cryptoData.length === 0) return;
    
    console.log('🔍 Fetching market insights with crypto data:', cryptoData.map(c => c.symbol));
    setLoading(true);
    try {
      // Try search-backed insights first (Tavily/Brave), fallback to Groq-only
      const symbols = cryptoData.map(c => c.symbol.toUpperCase());
      let marketInsights: MarketInsight[] = [];
      try {
        const news = await searchNewsInsights(symbols, 5);
        if (news && news.length) {
          marketInsights = news.map((i, index) => ({
            id: i.id || `news-${Date.now()}-${index}`,
            title: i.title,
            summary: i.summary,
            sentiment: i.sentiment || 'neutral',
            relevantCoins: i.relevantCoins?.length ? i.relevantCoins : symbols.slice(0, 3),
            confidence: typeof i.confidence === 'number' ? i.confidence : 0.65,
            timestamp: i.timestamp || new Date().toISOString(),
            sources: i.sources,
          }));
        }
      } catch {}

      if (!marketInsights.length) {
        marketInsights = await groqService.generateMarketInsights(cryptoData);
      }

      // Build lightweight signals for confidence blending
      const avgAbs = cryptoData.length
        ? cryptoData.reduce((s, c) => s + Math.abs(c.changePercent || 0), 0) / cryptoData.length
        : 0;
      const breadth = cryptoData.length
        ? cryptoData.filter(c => (c.changePercent || 0) > 0).length / cryptoData.length
        : 0.5;
      // Naive news polarity: +1 for 'up|bull|gain', -1 for 'down|bear|loss' keywords in summaries
      let newsPolarity = 0.5;
      if (marketInsights.length) {
        const text = marketInsights.map(i => `${i.title} ${i.summary}`.toLowerCase()).join(' ');
        const pos = (text.match(/\b(up|surge|bull|gain|rally|breakout)\b/g) || []).length;
        const neg = (text.match(/\b(down|drop|bear|loss|selloff|plunge)\b/g) || []).length;
        const score = pos + neg === 0 ? 0 : (pos - neg) / Math.max(1, pos + neg); // -1..1
        newsPolarity = Math.max(0, Math.min(1, 0.5 + score / 2));
      }

      const sentiment = await groqService.getMarketSentiment(cryptoData, {
        avgAbs,
        breadth,
        newsPolarity,
      });
      
      console.log('✅ Market insights fetched:', marketInsights.length, 'insights');
      console.log('✅ Market sentiment:', sentiment);
      setInsights(marketInsights);
      setOverallSentiment(sentiment);
      setLastUpdated(new Date());
    } catch (error) {
      console.warn('⚠️ Failed to fetch market insights, using fallback data:', error);
      // Use fallback data when API fails
      const fallbackInsights = groqService.getFallbackInsights();
      setInsights(fallbackInsights);
      setOverallSentiment({
        overall: 'neutral',
        confidence: 0.5,
        reasoning: 'Using fallback data due to API limitations.',
      });
      setLastUpdated(new Date());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    console.log('🔄 MarketInsights useEffect triggered, cryptoData length:', cryptoData.length);
    fetchInsights();
    // Only fetch on initial load or when cryptoData changes
    // No automatic refresh - user must click refresh button
  }, [cryptoData]);

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish':
        return <TrendingUp className="h-4 w-4" />;
      case 'bearish':
        return <TrendingDown className="h-4 w-4" />;
      default:
        return <Minus className="h-4 w-4" />;
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish':
        return 'text-green-400';
      case 'bearish':
        return 'text-red-400';
      default:
        return 'text-yellow-400';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-500';
    if (confidence >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  // For the overall sentiment bar, color by sentiment (not confidence) to avoid mixed signals
  const getOverallBarColor = (sentiment: 'bullish' | 'bearish' | 'neutral') => {
    if (sentiment === 'bullish') return 'bg-green-500';
    if (sentiment === 'bearish') return 'bg-red-500';
    return 'bg-yellow-500';
  };

  return (
    <div className="h-full min-h-[640px] overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <Brain className="h-6 w-6 text-purple-400 mr-3" />
          <h2 className="text-xl font-bold text-white">AI Market Insights</h2>
        </div>
        <div className="flex items-center space-x-3">
          {lastUpdated && (
            <div className="flex items-center text-gray-400 text-sm">
              <Clock className="h-4 w-4 mr-1" />
              {lastUpdated.toLocaleTimeString()}
            </div>
          )}
          <button
            onClick={fetchInsights}
            disabled={loading}
            className="flex items-center px-3 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg transition-colors text-sm"
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Overall Market Sentiment */}
      {overallSentiment && (() => {
        const mappedOverall: 'bullish' | 'bearish' | 'neutral' = overallSentiment.confidence < 0.5 ? 'neutral' : overallSentiment.overall;
        const reasoning = overallSentiment.confidence < 0.5
          ? `${overallSentiment.reasoning} (low confidence — treating as neutral)`
          : overallSentiment.reasoning;
        return (
        <div className="bg-gray-700 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-white font-semibold">Overall Market Sentiment</h3>
            <div className={`flex items-center ${getSentimentColor(mappedOverall)}`}>
              {getSentimentIcon(mappedOverall)}
              <span className="ml-2 font-medium capitalize">{mappedOverall}</span>
            </div>
          </div>
          <p className="text-gray-300 text-sm mb-3">{reasoning}</p>
          <div className="flex items-center">
            <span className="text-gray-400 text-sm mr-2">Confidence:</span>
            <div className="flex-1 bg-gray-600 rounded-full h-2 mr-2">
              <div
                className={`h-2 rounded-full ${getOverallBarColor(mappedOverall)}`}
                style={{ width: `${overallSentiment.confidence * 100}%` }}
              ></div>
            </div>
            <span className="text-white text-sm font-medium">
              {(overallSentiment.confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        );
      })()}

      {/* Market Insights */}
      <div className="space-y-4">
        {loading ? (
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="bg-gray-700 rounded-lg p-4 animate-pulse">
                <div className="h-4 bg-gray-600 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-600 rounded w-full mb-1"></div>
                <div className="h-3 bg-gray-600 rounded w-2/3"></div>
              </div>
            ))}
          </div>
        ) : (
          insights.map((insight, index) => (
            <div key={insight.id} className="bg-gray-700 rounded-lg p-4 hover:bg-gray-600/50 transition-colors">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center mb-2">
                    <span className="text-purple-400 font-bold text-sm mr-2">#{index + 1}</span>
                    <h3 className="text-white font-semibold text-sm sm:text-base">{insight.title}</h3>
                  </div>
                  <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-line">
                    {String(insight.summary || '')
                      .replace(/\s*={3,,}?\s*/g, ' ')
                      .replace(/\bImage\s+\d+:[^|\n]+/gi, '')
                      .replace(/\s{2,}/g, ' ')}
                  </p>
                  {insight.sources && insight.sources.length > 0 && (
                    <div className="mt-2">
                      <a
                        href={insight.sources[0].url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-indigo-400 text-xs hover:underline"
                        title={insight.sources[0].title || 'Open article'}
                      >
                        Read article →
                      </a>
                    </div>
                  )}
                </div>
                <div className={`ml-4 ${getSentimentColor(insight.sentiment)}`}>
                  {getSentimentIcon(insight.sentiment)}
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {/* Relevant Coins */}
                  <div className="flex items-center">
                    <Target className="h-3 w-3 text-gray-400 mr-1" />
                    <div className="flex space-x-1">
                      {insight.relevantCoins.slice(0, 3).map((coin) => (
                        <span
                          key={coin}
                          className="px-2 py-1 bg-gray-600 text-gray-300 rounded text-xs font-medium"
                        >
                          {coin}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  {/* Sentiment Badge */}
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    insight.sentiment === 'bullish' ? 'bg-green-900/30 text-green-400' :
                    insight.sentiment === 'bearish' ? 'bg-red-900/30 text-red-400' :
                    'bg-yellow-900/30 text-yellow-400'
                  }`}>
                    {insight.sentiment.toUpperCase()}
                  </span>
                </div>
                
                {/* Confidence Score */}
                <div className="flex items-center">
                  <span className="text-gray-400 text-xs mr-2">Confidence:</span>
                  <div className="w-12 bg-gray-600 rounded-full h-1.5">
                    <div
                      className={`h-1.5 rounded-full ${getConfidenceColor(insight.confidence)}`}
                      style={{ width: `${Math.max(0, Math.min(100, Math.round((insight.confidence || 0) * 100)))}%` }}
                    ></div>
                  </div>
                  <span className="text-gray-300 text-xs ml-2">
                    {Math.max(0, Math.min(100, Math.round((insight.confidence || 0) * 100)))}%
                  </span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Attribution removed per user request */}
    </div>
  );
};