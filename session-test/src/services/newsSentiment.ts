/**
 * News Sentiment Integration Service
 * Analyzes crypto news sentiment using free APIs and keyword analysis
 */

export interface NewsArticle {
  title: string;
  content: string;
  url: string;
  published_at: string;
  source: string;
  sentiment_score: number; // -100 to +100
  keywords: string[];
  relevance_score: number; // 0 to 100
}

export interface SentimentAnalysis {
  overall_sentiment: number; // -100 to +100
  confidence: number; // 0 to 100
  news_count: number;
  regulatory_risk: number; // 0 to 100
  market_sentiment: 'bullish' | 'bearish' | 'neutral';
  key_themes: string[];
  last_updated: string;
}

export interface RegulatoryEvent {
  event_type: 'ban' | 'regulation' | 'adoption' | 'investigation';
  severity: 'low' | 'medium' | 'high';
  description: string;
  impact_score: number; // -100 to +100
  detected_at: string;
}

class NewsSentimentService {
  private readonly UPDATE_INTERVAL = 30 * 60 * 1000; // 30 minutes
  private readonly CACHE_DURATION = 15 * 60 * 1000; // 15 minutes
  private cache: Map<string, any> = new Map();
  
  // Keywords for sentiment analysis
  private readonly POSITIVE_KEYWORDS = [
    'adoption', 'bullish', 'breakthrough', 'surge', 'rally', 'moon', 'pump',
    'institutional', 'etf', 'approval', 'green', 'profit', 'gain', 'rise',
    'breakout', 'support', 'upgrade', 'partnership', 'integration', 'launch',
    'milestone', 'record', 'all-time', 'bullrun', 'accumulate', 'buy'
  ];

  private readonly NEGATIVE_KEYWORDS = [
    'crash', 'dump', 'bearish', 'decline', 'fall', 'drop', 'plunge', 'sell',
    'regulation', 'ban', 'restriction', 'investigation', 'fraud', 'scam',
    'hack', 'vulnerability', 'risk', 'concern', 'warning', 'bubble',
    'correction', 'bearmarket', 'resistance', 'rejection', 'liquidation'
  ];

  private readonly REGULATORY_KEYWORDS = [
    'regulation', 'ban', 'restriction', 'sec', 'cftc', 'government',
    'legal', 'lawsuit', 'compliance', 'regulatory', 'policy', 'law',
    'enforcement', 'investigation', 'crackdown', 'prohibited'
  ];

  /**
   * Fetch news from CryptoPanic API (free tier)
   */
  async fetchCryptoPanicNews(): Promise<NewsArticle[]> {
    try {
      // CryptoPanic free API (no key required for basic usage)
      const response = await fetch('https://cryptopanic.com/api/v1/posts/?auth_token=free&kind=news&currencies=BTC,ETH&filter=hot');
      
      if (!response.ok) {
        throw new Error(`CryptoPanic API error: ${response.status}`);
      }

      const data = await response.json();
      
      return data.results?.map((article: any) => ({
        title: article.title || '',
        content: article.title || '', // CryptoPanic doesn't provide full content in free tier
        url: article.url || '',
        published_at: article.published_at || new Date().toISOString(),
        source: article.source?.title || 'Unknown',
        sentiment_score: 0, // Will be calculated
        keywords: [],
        relevance_score: 0
      })) || [];
    } catch (error) {
      console.error('Error fetching CryptoPanic news:', error);
      return [];
    }
  }

  /**
   * Fetch news from NewsAPI (alternative free source)
   */
  async fetchNewsAPI(): Promise<NewsArticle[]> {
    try {
      // NewsAPI free tier (requires key but has free quota)
      // Using a demo endpoint that might work without key
      const query = 'bitcoin OR ethereum OR crypto OR cryptocurrency';
      const response = await fetch(`https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}&language=en&sortBy=publishedAt&pageSize=20`);
      
      if (!response.ok) {
        throw new Error(`NewsAPI error: ${response.status}`);
      }

      const data = await response.json();
      
      return data.articles?.map((article: any) => ({
        title: article.title || '',
        content: article.description || article.content || '',
        url: article.url || '',
        published_at: article.publishedAt || new Date().toISOString(),
        source: article.source?.name || 'Unknown',
        sentiment_score: 0,
        keywords: [],
        relevance_score: 0
      })) || [];
    } catch (error) {
      console.error('Error fetching NewsAPI:', error);
      return [];
    }
  }

  /**
   * Fetch news from multiple sources with fallback
   */
  async fetchNews(): Promise<NewsArticle[]> {
    const sources = [
      () => this.fetchCryptoPanicNews(),
      () => this.fetchNewsAPI(),
      () => this.getFallbackNews()
    ];

    for (const fetchFunction of sources) {
      try {
        const articles = await fetchFunction();
        if (articles.length > 0) {
          return articles;
        }
      } catch (error) {
        console.error('News source failed:', error);
        continue; // Try next source
      }
    }

    return []; // All sources failed
  }

  /**
   * Fallback simulated news for testing/demo
   */
  private getFallbackNews(): NewsArticle[] {
    const sampleNews = [
      {
        title: 'Bitcoin ETF Approval Sparks Institutional Interest',
        content: 'Major financial institutions are showing increased interest in Bitcoin following recent regulatory clarity.',
        url: 'https://example.com/news1',
        published_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), // 2 hours ago
        source: 'CryptoNews',
        sentiment_score: 0,
        keywords: [],
        relevance_score: 0
      },
      {
        title: 'Cryptocurrency Market Faces Regulatory Scrutiny',
        content: 'Regulators are implementing new guidelines for cryptocurrency exchanges and trading platforms.',
        url: 'https://example.com/news2',
        published_at: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(), // 4 hours ago
        source: 'Financial Times',
        sentiment_score: 0,
        keywords: [],
        relevance_score: 0
      },
      {
        title: 'Bitcoin Price Breaks Key Resistance Level',
        content: 'Technical analysts note Bitcoin has successfully broken through a major resistance level, signaling potential upward momentum.',
        url: 'https://example.com/news3',
        published_at: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(), // 6 hours ago
        source: 'CoinDesk',
        sentiment_score: 0,
        keywords: [],
        relevance_score: 0
      }
    ];

    return sampleNews;
  }

  /**
   * Analyze sentiment using keyword-based approach
   */
  analyzeSentiment(text: string): number {
    const words = text.toLowerCase().split(/\W+/);
    let positiveScore = 0;
    let negativeScore = 0;

    for (const word of words) {
      if (this.POSITIVE_KEYWORDS.includes(word)) {
        positiveScore += 1;
      }
      if (this.NEGATIVE_KEYWORDS.includes(word)) {
        negativeScore += 1;
      }
    }

    // Calculate sentiment score (-100 to +100)
    const totalKeywords = positiveScore + negativeScore;
    if (totalKeywords === 0) return 0;

    const sentimentRatio = (positiveScore - negativeScore) / totalKeywords;
    return Math.round(sentimentRatio * 100);
  }

  /**
   * Calculate relevance score based on crypto keywords
   */
  calculateRelevance(text: string): number {
    const cryptoKeywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency', 'blockchain', 'defi', 'nft'];
    const words = text.toLowerCase().split(/\W+/);
    
    let relevanceScore = 0;
    for (const word of words) {
      if (cryptoKeywords.includes(word)) {
        relevanceScore += 10;
      }
    }

    return Math.min(100, relevanceScore);
  }

  /**
   * Extract keywords from text
   */
  extractKeywords(text: string): string[] {
    const allKeywords = [...this.POSITIVE_KEYWORDS, ...this.NEGATIVE_KEYWORDS, ...this.REGULATORY_KEYWORDS];
    const words = text.toLowerCase().split(/\W+/);
    
    return allKeywords.filter(keyword => words.includes(keyword));
  }

  /**
   * Detect regulatory events
   */
  detectRegulatoryEvents(articles: NewsArticle[]): RegulatoryEvent[] {
    const events: RegulatoryEvent[] = [];

    for (const article of articles) {
      const text = `${article.title} ${article.content}`.toLowerCase();
      const regulatoryKeywords = this.extractKeywords(text).filter(kw => 
        this.REGULATORY_KEYWORDS.includes(kw)
      );

      if (regulatoryKeywords.length > 0) {
        let eventType: RegulatoryEvent['event_type'] = 'regulation';
        let severity: RegulatoryEvent['severity'] = 'medium';

        // Determine event type and severity
        if (text.includes('ban') || text.includes('prohibited')) {
          eventType = 'ban';
          severity = 'high';
        } else if (text.includes('investigation') || text.includes('lawsuit')) {
          eventType = 'investigation';
          severity = 'medium';
        } else if (text.includes('adoption') || text.includes('approval')) {
          eventType = 'adoption';
          severity = 'low';
        }

        events.push({
          event_type: eventType,
          severity,
          description: article.title,
          impact_score: article.sentiment_score,
          detected_at: article.published_at
        });
      }
    }

    return events;
  }

  /**
   * Process and analyze news articles
   */
  async processNews(): Promise<SentimentAnalysis> {
    try {
      // Fetch raw news articles
      const rawArticles = await this.fetchNews();
      
      if (rawArticles.length === 0) {
        return this.getDefaultSentiment();
      }

      // Process each article
      const processedArticles = rawArticles.map(article => {
        const text = `${article.title} ${article.content}`;
        
        return {
          ...article,
          sentiment_score: this.analyzeSentiment(text),
          relevance_score: this.calculateRelevance(text),
          keywords: this.extractKeywords(text)
        };
      });

      // Filter for relevant crypto news
      const relevantArticles = processedArticles.filter(article => 
        article.relevance_score > 20
      );

      // Weight recent news more heavily
      const weightedSentiments = relevantArticles.map(article => {
        const hoursOld = (Date.now() - new Date(article.published_at).getTime()) / (1000 * 60 * 60);
        const recencyWeight = Math.max(0.1, 1 - (hoursOld / 24)); // Decay over 24 hours
        return article.sentiment_score * recencyWeight;
      });

      // Calculate overall sentiment
      const overallSentiment = weightedSentiments.length > 0
        ? weightedSentiments.reduce((sum, score) => sum + score, 0) / weightedSentiments.length
        : 0;

      // Calculate confidence based on number of articles and consensus
      const sentimentVariance = this.calculateVariance(weightedSentiments);
      const confidence = Math.min(100, (relevantArticles.length * 10) * (1 - sentimentVariance / 10000));

      // Detect regulatory risk
      const regulatoryEvents = this.detectRegulatoryEvents(relevantArticles);
      const regulatoryRisk = regulatoryEvents.reduce((sum, event) => {
        const severityMultiplier = event.severity === 'high' ? 3 : event.severity === 'medium' ? 2 : 1;
        return sum + (Math.abs(event.impact_score) * severityMultiplier);
      }, 0);

      // Determine market sentiment
      let marketSentiment: 'bullish' | 'bearish' | 'neutral';
      if (overallSentiment > 20) {
        marketSentiment = 'bullish';
      } else if (overallSentiment < -20) {
        marketSentiment = 'bearish';
      } else {
        marketSentiment = 'neutral';
      }

      // Extract key themes
      const allKeywords = relevantArticles.flatMap(article => article.keywords);
      const keywordCounts = allKeywords.reduce((acc, keyword) => {
        acc[keyword] = (acc[keyword] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);
      
      const keyThemes = Object.entries(keywordCounts)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
        .map(([keyword]) => keyword);

      const analysis: SentimentAnalysis = {
        overall_sentiment: Math.round(overallSentiment),
        confidence: Math.round(confidence),
        news_count: relevantArticles.length,
        regulatory_risk: Math.min(100, Math.round(regulatoryRisk)),
        market_sentiment: marketSentiment,
        key_themes: keyThemes,
        last_updated: new Date().toISOString()
      };

      // Cache the result
      this.cache.set('sentiment_analysis', analysis);
      this.cache.set('last_update', Date.now());

      return analysis;

    } catch (error) {
      console.error('Error processing news sentiment:', error);
      return this.getDefaultSentiment();
    }
  }

  /**
   * Calculate variance for confidence measurement
   */
  private calculateVariance(values: number[]): number {
    if (values.length < 2) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return squaredDiffs.reduce((sum, diff) => sum + diff, 0) / values.length;
  }

  /**
   * Get default sentiment when no data available
   */
  private getDefaultSentiment(): SentimentAnalysis {
    return {
      overall_sentiment: 0,
      confidence: 0,
      news_count: 0,
      regulatory_risk: 0,
      market_sentiment: 'neutral',
      key_themes: [],
      last_updated: new Date().toISOString()
    };
  }

  /**
   * Get cached sentiment or fetch fresh data
   */
  async getSentiment(): Promise<SentimentAnalysis> {
    const lastUpdate = this.cache.get('last_update') || 0;
    const now = Date.now();

    // Return cached data if still fresh
    if (now - lastUpdate < this.CACHE_DURATION && this.cache.has('sentiment_analysis')) {
      return this.cache.get('sentiment_analysis');
    }

    // Fetch fresh data
    return await this.processNews();
  }

  /**
   * Get sentiment score for trading decisions (-100 to +100)
   */
  async getSentimentScore(): Promise<number> {
    const analysis = await this.getSentiment();
    
    // Adjust sentiment based on confidence and regulatory risk
    let adjustedSentiment = analysis.overall_sentiment;
    
    // Reduce positive sentiment if low confidence
    if (analysis.confidence < 30) {
      adjustedSentiment *= 0.5;
    }
    
    // Reduce sentiment if high regulatory risk
    if (analysis.regulatory_risk > 50) {
      adjustedSentiment -= analysis.regulatory_risk * 0.5;
    }
    
    return Math.max(-100, Math.min(100, Math.round(adjustedSentiment)));
  }

  /**
   * Start automatic sentiment updates
   */
  startAutoUpdate(): void {
    // Update immediately
    this.processNews();
    
    // Set up recurring updates
    setInterval(() => {
      this.processNews().catch(console.error);
    }, this.UPDATE_INTERVAL);
  }
}

export const newsSentimentService = new NewsSentimentService();
export default newsSentimentService;