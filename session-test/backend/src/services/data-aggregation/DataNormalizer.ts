import logger from '../../utils/logger';
import { 
  OnChainData, 
  Transaction, 
  TokenHolding, 
  FundingRateData, 
  WhaleAlert, 
  SmartMoneyFlow, 
  LiquidationData 
} from './DataAggregatorService';

export interface NormalizationConfig {
  priceDecimals: number;
  volumeDecimals: number;
  ratePrecision: number;
  confidenceThreshold: number;
  minimumValueUsd: number;
}

export class DataNormalizer {
  private config: NormalizationConfig;
  private priceCache: Map<string, { price: number; timestamp: number }> = new Map();

  constructor(config?: Partial<NormalizationConfig>) {
    this.config = {
      priceDecimals: 8,
      volumeDecimals: 4,
      ratePrecision: 6,
      confidenceThreshold: 30,
      minimumValueUsd: 1000,
      ...config
    };
  }

  normalizeOnChainData(rawData: any, symbol: string): OnChainData {
    try {
      // Handle different data formats from Etherscan, Covalent, etc.
      const normalized: OnChainData = {
        address: this.normalizeAddress(rawData.address),
        balance: this.normalizeBalance(rawData.balance, rawData.decimals || 18),
        transactions: this.normalizeTransactions(rawData.transactions || []),
        tokenHoldings: this.normalizeTokenHoldings(rawData.tokenHoldings || []),
        lastActivity: this.normalizeDate(rawData.lastActivity),
        riskScore: this.calculateRiskScore(rawData)
      };

      return normalized;

    } catch (error) {
      logger.error('Error normalizing on-chain data:', error);
      throw error;
    }
  }

  normalizeFundingRate(rawData: any, source: string): FundingRateData {
    try {
      const rate = this.normalizeRate(rawData.rate || rawData.fundingRate || rawData.lastFundingRate);
      const symbol = this.normalizeSymbol(rawData.symbol);

      return {
        symbol,
        rate,
        nextFundingTime: this.normalizeDate(rawData.nextFundingTime || rawData.nextFundingRate),
        predictedRate: rawData.estimatedRate ? this.normalizeRate(rawData.estimatedRate) : undefined,
        historicalAverage: this.calculateHistoricalAverage(rate, symbol),
        source,
        timestamp: this.normalizeDate(rawData.timestamp || rawData.time || Date.now())
      };

    } catch (error) {
      logger.error('Error normalizing funding rate data:', error);
      throw error;
    }
  }

  normalizeWhaleAlert(rawData: any, source: string): WhaleAlert {
    try {
      const amount = this.normalizeAmount(rawData.amount || rawData.value);
      const amountUsd = this.normalizeAmountUsd(rawData.amountUsd || rawData.valueUsd, amount, rawData.symbol);

      return {
        id: this.generateWhaleAlertId(rawData),
        symbol: this.normalizeSymbol(rawData.symbol),
        amount,
        amountUsd,
        from: this.normalizeAddress(rawData.from || rawData.sender),
        to: this.normalizeAddress(rawData.to || rawData.receiver),
        txHash: this.normalizeHash(rawData.hash || rawData.txHash || rawData.transaction?.hash),
        timestamp: this.normalizeDate(rawData.timestamp || rawData.time),
        type: this.normalizeTransferType(rawData.type, rawData.from, rawData.to),
        exchange: this.extractExchangeName(rawData),
        confidence: this.calculateWhaleConfidence(rawData, amountUsd)
      };

    } catch (error) {
      logger.error('Error normalizing whale alert data:', error);
      throw error;
    }
  }

  normalizeSmartMoneyFlow(rawData: any): SmartMoneyFlow {
    try {
      const amount = this.normalizeAmount(rawData.amount || rawData.baseAmount);
      const amountUsd = this.normalizeAmountUsd(rawData.amountUsd || rawData.tradeAmountUsd, amount, rawData.symbol);

      return {
        address: this.normalizeAddress(rawData.address || rawData.taker || rawData.maker),
        label: this.normalizeLabel(rawData.label || rawData.annotation),
        action: this.normalizeAction(rawData.action || rawData.side),
        amount,
        amountUsd,
        symbol: this.normalizeSymbol(rawData.symbol || rawData.baseCurrency?.symbol),
        timestamp: this.normalizeDate(rawData.timestamp || rawData.time),
        confidence: this.normalizeConfidence(rawData.confidence),
        source: rawData.source || rawData.dex || rawData.exchange?.name || 'unknown'
      };

    } catch (error) {
      logger.error('Error normalizing smart money flow data:', error);
      throw error;
    }
  }

  normalizeLiquidation(rawData: any): LiquidationData {
    try {
      return {
        symbol: this.normalizeSymbol(rawData.symbol),
        side: this.normalizeSide(rawData.side),
        amount: this.normalizeAmount(rawData.amount || rawData.qty),
        amountUsd: this.normalizeAmountUsd(rawData.amountUsd || rawData.usd),
        price: this.normalizePrice(rawData.price),
        timestamp: this.normalizeDate(rawData.timestamp || rawData.time),
        exchange: rawData.exchange || 'unknown'
      };

    } catch (error) {
      logger.error('Error normalizing liquidation data:', error);
      throw error;
    }
  }

  // Core normalization methods

  private normalizeAddress(address: any): string {
    if (!address) return '';
    const addr = address.toString().toLowerCase();
    
    // Validate Ethereum address format
    if (addr.match(/^0x[a-f0-9]{40}$/)) {
      return addr;
    }
    
    // Handle other address formats or return as-is
    return addr;
  }

  private normalizeBalance(balance: any, decimals: number = 18): number {
    if (!balance) return 0;
    
    const balanceStr = balance.toString();
    const balanceNum = parseFloat(balanceStr);
    
    // If balance looks like it's in wei (very large number), convert it
    if (balanceNum > 1e15) {
      return balanceNum / Math.pow(10, decimals);
    }
    
    return Math.round(balanceNum * Math.pow(10, this.config.priceDecimals)) / Math.pow(10, this.config.priceDecimals);
  }

  private normalizeAmount(amount: any): number {
    if (!amount) return 0;
    
    const amountNum = parseFloat(amount.toString());
    return Math.round(amountNum * Math.pow(10, this.config.volumeDecimals)) / Math.pow(10, this.config.volumeDecimals);
  }

  private normalizeAmountUsd(amountUsd: any, amount?: number, symbol?: string): number {
    if (amountUsd) {
      return this.normalizeAmount(amountUsd);
    }
    
    // Try to calculate from amount and symbol
    if (amount && symbol) {
      const price = this.getPrice(symbol);
      return amount * price;
    }
    
    return 0;
  }

  private normalizePrice(price: any): number {
    if (!price) return 0;
    
    const priceNum = parseFloat(price.toString());
    return Math.round(priceNum * Math.pow(10, this.config.priceDecimals)) / Math.pow(10, this.config.priceDecimals);
  }

  private normalizeRate(rate: any): number {
    if (!rate) return 0;
    
    const rateNum = parseFloat(rate.toString());
    return Math.round(rateNum * Math.pow(10, this.config.ratePrecision)) / Math.pow(10, this.config.ratePrecision);
  }

  private normalizeSymbol(symbol: any): string {
    if (!symbol) return '';
    
    return symbol.toString()
      .toUpperCase()
      .replace(/[-_]/g, '')
      .replace('USDT', '')
      .replace('USD', '')
      .replace('PERP', '')
      .trim();
  }

  private normalizeDate(date: any): Date {
    if (!date) return new Date();
    
    if (date instanceof Date) return date;
    
    if (typeof date === 'number') {
      // Handle both seconds and milliseconds timestamps
      return new Date(date < 1e12 ? date * 1000 : date);
    }
    
    if (typeof date === 'string') {
      return new Date(date);
    }
    
    return new Date();
  }

  private normalizeHash(hash: any): string {
    if (!hash) return '';
    
    const hashStr = hash.toString();
    
    // Validate hash format (should be 0x followed by 64 hex chars for Ethereum)
    if (hashStr.match(/^0x[a-f0-9]{64}$/i)) {
      return hashStr.toLowerCase();
    }
    
    return hashStr;
  }

  private normalizeTransferType(type: any, from: any, to: any): 'transfer' | 'exchange_inflow' | 'exchange_outflow' {
    if (type) {
      const typeStr = type.toString().toLowerCase();
      if (typeStr.includes('inflow')) return 'exchange_inflow';
      if (typeStr.includes('outflow')) return 'exchange_outflow';
    }
    
    // Try to determine from addresses
    const fromStr = (from || '').toString().toLowerCase();
    const toStr = (to || '').toString().toLowerCase();
    
    if (this.isExchangeAddress(fromStr)) return 'exchange_outflow';
    if (this.isExchangeAddress(toStr)) return 'exchange_inflow';
    
    return 'transfer';
  }

  private normalizeAction(action: any): 'buy' | 'sell' | 'hold' {
    if (!action) return 'hold';
    
    const actionStr = action.toString().toLowerCase();
    
    if (actionStr.includes('buy') || actionStr.includes('long') || actionStr.includes('accumul')) {
      return 'buy';
    }
    
    if (actionStr.includes('sell') || actionStr.includes('short') || actionStr.includes('distribut')) {
      return 'sell';
    }
    
    return 'hold';
  }

  private normalizeSide(side: any): 'long' | 'short' {
    if (!side) return 'long';
    
    const sideStr = side.toString().toLowerCase();
    return sideStr.includes('short') ? 'short' : 'long';
  }

  private normalizeLabel(label: any): string {
    if (!label) return 'Unknown';
    
    const labelStr = label.toString().trim();
    
    // Clean up common label formats
    return labelStr
      .replace(/^(binance|coinbase|kraken|bitfinex)/i, (match) => this.capitalizeFirst(match))
      .replace(/\s+/g, ' ')
      .slice(0, 100); // Limit length
  }

  private normalizeConfidence(confidence: any): number {
    if (!confidence) return 50; // Default medium confidence
    
    const conf = parseFloat(confidence.toString());
    return Math.max(0, Math.min(100, conf)); // Clamp between 0-100
  }

  // Complex normalization methods

  private normalizeTransactions(transactions: any[]): Transaction[] {
    return transactions.map(tx => ({
      hash: this.normalizeHash(tx.hash || tx.tx_hash),
      from: this.normalizeAddress(tx.from || tx.from_address),
      to: this.normalizeAddress(tx.to || tx.to_address),
      value: this.normalizeAmount(tx.value),
      token: tx.token || tx.contractAddress,
      timestamp: this.normalizeDate(tx.timestamp || tx.timeStamp || tx.block_signed_at),
      gasUsed: parseInt(tx.gasUsed || tx.gas_used || '0'),
      gasPrice: parseInt(tx.gasPrice || tx.gas_price || '0'),
      type: this.determineTransactionType(tx)
    }));
  }

  private normalizeTokenHoldings(holdings: any[]): TokenHolding[] {
    return holdings
      .map(holding => ({
        token: holding.contractAddress || holding.contract_address || '',
        symbol: this.normalizeSymbol(holding.symbol || holding.contractTickerSymbol || holding.tokenSymbol),
        balance: this.normalizeBalance(holding.balance, holding.decimals || holding.contractDecimals || 18),
        valueUsd: this.normalizeAmountUsd(holding.quote || holding.valueUsd),
        percentage: parseFloat(holding.percentage || '0')
      }))
      .filter(holding => holding.balance > 0 && holding.valueUsd >= this.config.minimumValueUsd);
  }

  // Utility and calculation methods

  private calculateRiskScore(data: any): number {
    let riskScore = 50; // Base score
    
    // Increase risk for high transaction volume
    const txCount = (data.transactions || []).length;
    if (txCount > 100) riskScore += 20;
    else if (txCount > 50) riskScore += 10;
    
    // Increase risk for exchange interactions
    const hasExchangeInteractions = (data.transactions || []).some((tx: any) => 
      this.isExchangeAddress(tx.to) || this.isExchangeAddress(tx.from)
    );
    if (hasExchangeInteractions) riskScore += 15;
    
    // Decrease risk for older addresses
    const accountAge = Date.now() - new Date(data.lastActivity || 0).getTime();
    const daysSinceCreation = accountAge / (1000 * 60 * 60 * 24);
    if (daysSinceCreation > 365) riskScore -= 10;
    
    return Math.max(0, Math.min(100, riskScore));
  }

  private calculateWhaleConfidence(data: any, amountUsd: number): number {
    let confidence = this.config.confidenceThreshold;
    
    // Higher amounts get higher confidence
    if (amountUsd > 10000000) confidence += 40;      // $10M+
    else if (amountUsd > 1000000) confidence += 30;  // $1M+
    else if (amountUsd > 100000) confidence += 20;   // $100K+
    
    // Known addresses get higher confidence
    if (this.isKnownAddress(data.from) || this.isKnownAddress(data.to)) {
      confidence += 20;
    }
    
    // Exchange transactions get moderate confidence boost
    if (data.exchange) confidence += 10;
    
    return Math.min(100, confidence);
  }

  private calculateHistoricalAverage(currentRate: number, symbol: string): number {
    // In a real implementation, this would calculate from historical data
    // For now, return a reasonable estimate based on current rate
    return currentRate * 0.8; // Assume current is slightly above average
  }

  private determineTransactionType(tx: any): 'transfer' | 'swap' | 'deposit' | 'withdrawal' {
    // Analyze transaction to determine type
    if (tx.methodId) {
      const methodId = tx.methodId.toLowerCase();
      if (methodId.includes('swap')) return 'swap';
      if (methodId.includes('deposit')) return 'deposit';
      if (methodId.includes('withdraw')) return 'withdrawal';
    }
    
    // Check for DEX signatures in logs
    if (tx.logs && tx.logs.some((log: any) => 
      log.topics && log.topics[0] && log.topics[0].includes('swap')
    )) {
      return 'swap';
    }
    
    return 'transfer';
  }

  private generateWhaleAlertId(data: any): string {
    const hash = data.hash || data.txHash || '';
    const timestamp = this.normalizeDate(data.timestamp || data.time).getTime();
    const amount = this.normalizeAmount(data.amount || data.value);
    
    return `whale_${hash.slice(0, 10)}_${timestamp}_${amount}`.toLowerCase();
  }

  private extractExchangeName(data: any): string | undefined {
    // Extract exchange name from various fields
    if (data.exchange) return data.exchange;
    if (data.source && this.isExchangeName(data.source)) return data.source;
    
    // Check address annotations
    const annotations = [
      data.fromAnnotation,
      data.toAnnotation,
      data.sender?.annotation,
      data.receiver?.annotation
    ].filter(Boolean);
    
    for (const annotation of annotations) {
      const exchangeName = this.extractExchangeFromAnnotation(annotation);
      if (exchangeName) return exchangeName;
    }
    
    return undefined;
  }

  private isExchangeAddress(address: string): boolean {
    if (!address) return false;
    
    // This would typically check against a database of known exchange addresses
    // For now, use a simple heuristic
    const knownExchangePrefixes = [
      '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be', // Binance
      '0xd551234ae421e3bcba99a0da6d736074f22192ff', // Binance 2
      '0x503828976d22510aad0201ac7ec88293211d23da', // Coinbase
      '0xddfabcdc4d8ffc6d5beaf154f18b778f892a0740', // Coinbase 2
    ];
    
    return knownExchangePrefixes.some(prefix => 
      address.toLowerCase().startsWith(prefix.toLowerCase())
    );
  }

  private isKnownAddress(address: string): boolean {
    // Check if address is in our database of known addresses
    // For now, return false as this would require a database lookup
    return false;
  }

  private isExchangeName(name: string): boolean {
    const exchangeNames = [
      'binance', 'coinbase', 'kraken', 'bitfinex', 'huobi',
      'okex', 'kucoin', 'gate.io', 'bitstamp', 'gemini'
    ];
    
    return exchangeNames.some(exchange => 
      name.toLowerCase().includes(exchange)
    );
  }

  private extractExchangeFromAnnotation(annotation: string): string | undefined {
    if (!annotation) return undefined;
    
    const exchangeMap: Record<string, string> = {
      'binance': 'Binance',
      'coinbase': 'Coinbase',
      'kraken': 'Kraken',
      'bitfinex': 'Bitfinex',
      'huobi': 'Huobi',
      'okex': 'OKEx',
      'kucoin': 'KuCoin',
      'gate': 'Gate.io',
      'bitstamp': 'Bitstamp',
      'gemini': 'Gemini'
    };
    
    const lowerAnnotation = annotation.toLowerCase();
    for (const [keyword, exchange] of Object.entries(exchangeMap)) {
      if (lowerAnnotation.includes(keyword)) {
        return exchange;
      }
    }
    
    return undefined;
  }

  private getPrice(symbol: string): number {
    // Get price from cache or external source
    const cached = this.priceCache.get(symbol);
    if (cached && Date.now() - cached.timestamp < 60000) { // 1 minute cache
      return cached.price;
    }
    
    // Fallback prices (in a real implementation, this would fetch from an API)
    const fallbackPrices: Record<string, number> = {
      'BTC': 45000,
      'ETH': 3000,
      'USDC': 1,
      'USDT': 1,
      'BNB': 300,
      'ADA': 0.5,
      'SOL': 100,
      'MATIC': 1,
      'AVAX': 30,
      'DOT': 25
    };
    
    const price = fallbackPrices[symbol] || 0;
    this.priceCache.set(symbol, { price, timestamp: Date.now() });
    
    return price;
  }

  private capitalizeFirst(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
  }

  // Quality control methods

  validateNormalizedData(data: any, type: string): boolean {
    try {
      switch (type) {
        case 'onchain':
          return this.validateOnChainData(data);
        case 'funding':
          return this.validateFundingRateData(data);
        case 'whale':
          return this.validateWhaleAlert(data);
        case 'smartmoney':
          return this.validateSmartMoneyFlow(data);
        case 'liquidation':
          return this.validateLiquidationData(data);
        default:
          return false;
      }
    } catch (error) {
      logger.error(`Validation failed for ${type} data:`, error);
      return false;
    }
  }

  private validateOnChainData(data: OnChainData): boolean {
    return !!(
      data.address &&
      data.balance >= 0 &&
      data.lastActivity instanceof Date &&
      data.riskScore >= 0 && data.riskScore <= 100
    );
  }

  private validateFundingRateData(data: FundingRateData): boolean {
    return !!(
      data.symbol &&
      typeof data.rate === 'number' &&
      data.nextFundingTime instanceof Date &&
      data.source
    );
  }

  private validateWhaleAlert(data: WhaleAlert): boolean {
    return !!(
      data.id &&
      data.symbol &&
      data.amount > 0 &&
      data.amountUsd >= this.config.minimumValueUsd &&
      data.from &&
      data.to &&
      data.confidence >= 0 && data.confidence <= 100
    );
  }

  private validateSmartMoneyFlow(data: SmartMoneyFlow): boolean {
    return !!(
      data.address &&
      data.action &&
      data.amount > 0 &&
      data.symbol &&
      data.confidence >= 0 && data.confidence <= 100
    );
  }

  private validateLiquidationData(data: LiquidationData): boolean {
    return !!(
      data.symbol &&
      data.side &&
      data.amount > 0 &&
      data.amountUsd > 0 &&
      data.price > 0
    );
  }

  // Statistics and reporting

  getNormalizationStats(): {
    processedCount: number;
    errorCount: number;
    validationFailures: number;
    averageProcessingTime: number;
  } {
    // In a real implementation, this would track normalization statistics
    return {
      processedCount: 0,
      errorCount: 0,
      validationFailures: 0,
      averageProcessingTime: 0
    };
  }

  resetStats(): void {
    // Reset internal counters
  }
}