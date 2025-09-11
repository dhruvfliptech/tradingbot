import fetch from 'node-fetch';
import logger from '../../../utils/logger';

export interface BitqueryConfig {
  apiKey?: string;
  endpoint: string;
  rateLimit: number;
}

export interface BitqueryWhaleTransfer {
  hash: string;
  from: string;
  to: string;
  amount: number;
  amountUsd: number;
  symbol: string;
  timestamp: Date;
  exchange?: string;
  type: 'transfer' | 'exchange_inflow' | 'exchange_outflow';
}

export interface BitquerySmartMoneyFlow {
  address: string;
  label: string;
  action: 'buy' | 'sell';
  amount: number;
  amountUsd: number;
  symbol: string;
  timestamp: Date;
  confidence: number;
  dex: string;
}

export interface BitqueryDexTrade {
  hash: string;
  maker: string;
  taker: string;
  baseAmount: number;
  quoteAmount: number;
  baseSymbol: string;
  quoteSymbol: string;
  price: number;
  priceUsd: number;
  timestamp: Date;
  dex: string;
  side: 'buy' | 'sell';
}

export class BitqueryClient {
  private config: BitqueryConfig;
  private baseUrl: string;

  constructor(config: BitqueryConfig) {
    this.config = config;
    this.baseUrl = config.endpoint || 'https://graphql.bitquery.io';
  }

  async testConnection(): Promise<void> {
    try {
      // Simple test query to check connection
      const query = `
        query {
          ethereum(network: ethereum) {
            blocks(date: {since: "2023-01-01"} limit: 1) {
              height
              timestamp {
                time
              }
            }
          }
        }
      `;

      const response = await this.makeGraphQLRequest(query);
      
      if (response.errors) {
        throw new Error(`Bitquery API test failed: ${response.errors[0].message}`);
      }

      logger.debug('Bitquery API connection test successful');
    } catch (error) {
      logger.error('Bitquery API connection test failed:', error);
      throw error;
    }
  }

  async getWhaleTransfers(symbol: string, minAmountUsd: number = 100000): Promise<BitqueryWhaleTransfer[]> {
    try {
      logger.debug(`Fetching whale transfers for ${symbol} with min amount $${minAmountUsd}`);

      const query = `
        query GetWhaleTransfers($symbol: String!, $minAmount: Float!) {
          ethereum(network: ethereum) {
            transfers(
              currency: {symbol: {is: $symbol}}
              amount: {gt: $minAmount}
              date: {since: "2024-01-01"}
              options: {limit: 100, desc: "block.timestamp.time"}
            ) {
              transaction {
                hash
              }
              sender {
                address
                annotation
              }
              receiver {
                address
                annotation
              }
              amount(in: USD)
              amountInUSD: amount(in: USD)
              currency {
                symbol
                decimals
              }
              block {
                timestamp {
                  time
                }
              }
            }
          }
        }
      `;

      const variables = {
        symbol: symbol.toUpperCase(),
        minAmount: minAmountUsd
      };

      const response = await this.makeGraphQLRequest(query, variables);

      if (response.errors) {
        logger.warn(`Bitquery whale transfers query failed: ${response.errors[0].message}`);
        return [];
      }

      const transfers = response.data?.ethereum?.transfers || [];
      
      return transfers.map((transfer: any) => {
        const senderLabel = transfer.sender.annotation || '';
        const receiverLabel = transfer.receiver.annotation || '';
        
        let type: 'transfer' | 'exchange_inflow' | 'exchange_outflow' = 'transfer';
        let exchange: string | undefined;

        // Detect exchange flows
        if (this.isExchangeAddress(senderLabel) || this.isExchangeAddress(receiverLabel)) {
          if (this.isExchangeAddress(senderLabel)) {
            type = 'exchange_outflow';
            exchange = this.extractExchangeName(senderLabel);
          } else {
            type = 'exchange_inflow';
            exchange = this.extractExchangeName(receiverLabel);
          }
        }

        return {
          hash: transfer.transaction.hash,
          from: transfer.sender.address,
          to: transfer.receiver.address,
          amount: parseFloat(transfer.amount),
          amountUsd: parseFloat(transfer.amountInUSD),
          symbol: transfer.currency.symbol,
          timestamp: new Date(transfer.block.timestamp.time),
          exchange,
          type
        };
      });

    } catch (error) {
      logger.error(`Error fetching whale transfers for ${symbol}:`, error);
      return [];
    }
  }

  async getSmartMoneyFlows(symbol: string): Promise<BitquerySmartMoneyFlow[]> {
    try {
      logger.debug(`Fetching smart money flows for ${symbol}`);

      const query = `
        query GetSmartMoneyFlows($symbol: String!) {
          ethereum(network: ethereum) {
            dexTrades(
              baseCurrency: {symbol: {is: $symbol}}
              date: {since: "2024-01-01"}
              options: {limit: 100, desc: "block.timestamp.time"}
              tradeAmountUsd: {gt: 50000}
            ) {
              transaction {
                hash
              }
              maker {
                address
                annotation
              }
              taker {
                address
                annotation
              }
              baseAmount
              quoteAmount
              baseCurrency {
                symbol
              }
              quoteCurrency {
                symbol
              }
              tradeAmountUsd
              side
              block {
                timestamp {
                  time
                }
              }
              exchange {
                name
              }
              smartMoney {
                scoreDecile
              }
            }
          }
        }
      `;

      const variables = {
        symbol: symbol.toUpperCase()
      };

      const response = await this.makeGraphQLRequest(query, variables);

      if (response.errors) {
        logger.warn(`Bitquery smart money query failed: ${response.errors[0].message}`);
        return [];
      }

      const trades = response.data?.ethereum?.dexTrades || [];
      
      return trades
        .filter((trade: any) => this.isSmartMoney(trade))
        .map((trade: any) => {
          const address = trade.side === 'BUY' ? trade.taker.address : trade.maker.address;
          const label = trade.side === 'BUY' ? trade.taker.annotation : trade.maker.annotation;
          
          return {
            address,
            label: label || 'Unknown Smart Money',
            action: trade.side.toLowerCase() as 'buy' | 'sell',
            amount: parseFloat(trade.baseAmount),
            amountUsd: parseFloat(trade.tradeAmountUsd),
            symbol: trade.baseCurrency.symbol,
            timestamp: new Date(trade.block.timestamp.time),
            confidence: this.calculateConfidence(trade),
            dex: trade.exchange.name
          };
        });

    } catch (error) {
      logger.error(`Error fetching smart money flows for ${symbol}:`, error);
      return [];
    }
  }

  async getDexTrades(symbol: string, limit: number = 100): Promise<BitqueryDexTrade[]> {
    try {
      logger.debug(`Fetching DEX trades for ${symbol}`);

      const query = `
        query GetDexTrades($symbol: String!, $limit: Int!) {
          ethereum(network: ethereum) {
            dexTrades(
              baseCurrency: {symbol: {is: $symbol}}
              date: {since: "2024-01-01"}
              options: {limit: $limit, desc: "block.timestamp.time"}
            ) {
              transaction {
                hash
              }
              maker {
                address
              }
              taker {
                address
              }
              baseAmount
              quoteAmount
              baseCurrency {
                symbol
              }
              quoteCurrency {
                symbol
              }
              price
              priceInUSD
              side
              block {
                timestamp {
                  time
                }
              }
              exchange {
                name
              }
            }
          }
        }
      `;

      const variables = {
        symbol: symbol.toUpperCase(),
        limit
      };

      const response = await this.makeGraphQLRequest(query, variables);

      if (response.errors) {
        logger.warn(`Bitquery DEX trades query failed: ${response.errors[0].message}`);
        return [];
      }

      const trades = response.data?.ethereum?.dexTrades || [];
      
      return trades.map((trade: any) => ({
        hash: trade.transaction.hash,
        maker: trade.maker.address,
        taker: trade.taker.address,
        baseAmount: parseFloat(trade.baseAmount),
        quoteAmount: parseFloat(trade.quoteAmount),
        baseSymbol: trade.baseCurrency.symbol,
        quoteSymbol: trade.quoteCurrency.symbol,
        price: parseFloat(trade.price),
        priceUsd: parseFloat(trade.priceInUSD),
        timestamp: new Date(trade.block.timestamp.time),
        dex: trade.exchange.name,
        side: trade.side.toLowerCase() as 'buy' | 'sell'
      }));

    } catch (error) {
      logger.error(`Error fetching DEX trades for ${symbol}:`, error);
      return [];
    }
  }

  async getTokenMetrics(symbol: string): Promise<{
    holders: number;
    transfers24h: number;
    volume24h: number;
    uniqueAddresses24h: number;
  }> {
    try {
      const query = `
        query GetTokenMetrics($symbol: String!) {
          ethereum(network: ethereum) {
            transfers(
              currency: {symbol: {is: $symbol}}
              date: {since: "2024-01-01"}
            ) {
              count
              unique(of: sender)
              amount(in: USD)
            }
          }
        }
      `;

      const variables = {
        symbol: symbol.toUpperCase()
      };

      const response = await this.makeGraphQLRequest(query, variables);

      if (response.errors) {
        logger.warn(`Bitquery token metrics query failed: ${response.errors[0].message}`);
        return {
          holders: 0,
          transfers24h: 0,
          volume24h: 0,
          uniqueAddresses24h: 0
        };
      }

      const transfers = response.data?.ethereum?.transfers || [];
      const metrics = transfers[0] || {};

      return {
        holders: metrics.unique || 0,
        transfers24h: metrics.count || 0,
        volume24h: parseFloat(metrics.amount) || 0,
        uniqueAddresses24h: metrics.unique || 0
      };

    } catch (error) {
      logger.error(`Error fetching token metrics for ${symbol}:`, error);
      return {
        holders: 0,
        transfers24h: 0,
        volume24h: 0,
        uniqueAddresses24h: 0
      };
    }
  }

  async getTopHolders(symbol: string, limit: number = 50): Promise<Array<{
    address: string;
    balance: number;
    percentage: number;
    label?: string;
  }>> {
    try {
      const query = `
        query GetTopHolders($symbol: String!, $limit: Int!) {
          ethereum(network: ethereum) {
            balances(
              currency: {symbol: {is: $symbol}}
              options: {limit: $limit, desc: "value"}
            ) {
              address {
                address
                annotation
              }
              value
            }
          }
        }
      `;

      const variables = {
        symbol: symbol.toUpperCase(),
        limit
      };

      const response = await this.makeGraphQLRequest(query, variables);

      if (response.errors) {
        logger.warn(`Bitquery top holders query failed: ${response.errors[0].message}`);
        return [];
      }

      const balances = response.data?.ethereum?.balances || [];
      const totalValue = balances.reduce((sum: number, balance: any) => sum + parseFloat(balance.value), 0);

      return balances.map((balance: any) => ({
        address: balance.address.address,
        balance: parseFloat(balance.value),
        percentage: totalValue > 0 ? (parseFloat(balance.value) / totalValue) * 100 : 0,
        label: balance.address.annotation
      }));

    } catch (error) {
      logger.error(`Error fetching top holders for ${symbol}:`, error);
      return [];
    }
  }

  private async makeGraphQLRequest(query: string, variables: any = {}): Promise<any> {
    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      if (this.config.apiKey) {
        headers['X-API-KEY'] = this.config.apiKey;
      }

      const response = await fetch(this.baseUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          query,
          variables
        }),
        timeout: 15000
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return data;

    } catch (error) {
      logger.error('Bitquery GraphQL request failed:', error);
      throw error;
    }
  }

  private isExchangeAddress(annotation: string): boolean {
    if (!annotation) return false;
    
    const exchangeKeywords = [
      'binance', 'coinbase', 'kraken', 'bitfinex', 'huobi',
      'okex', 'kucoin', 'gate.io', 'bitstamp', 'gemini',
      'exchange', 'trading', 'wallet'
    ];

    return exchangeKeywords.some(keyword => 
      annotation.toLowerCase().includes(keyword)
    );
  }

  private extractExchangeName(annotation: string): string {
    if (!annotation) return 'Unknown Exchange';
    
    const exchangeNames: Record<string, string> = {
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

    for (const [keyword, name] of Object.entries(exchangeNames)) {
      if (annotation.toLowerCase().includes(keyword)) {
        return name;
      }
    }

    return 'Unknown Exchange';
  }

  private isSmartMoney(trade: any): boolean {
    // Criteria for identifying smart money:
    // 1. Large transaction size
    // 2. Known smart money addresses
    // 3. High score decile if available
    
    const isLargeTransaction = parseFloat(trade.tradeAmountUsd) > 50000;
    const hasSmartMoneyScore = trade.smartMoney && trade.smartMoney.scoreDecile >= 8;
    const isKnownSmartMoney = this.isKnownSmartMoneyAddress(trade.maker.address) || 
                             this.isKnownSmartMoneyAddress(trade.taker.address);

    return isLargeTransaction && (hasSmartMoneyScore || isKnownSmartMoney);
  }

  private isKnownSmartMoneyAddress(address: string): boolean {
    // This would typically be maintained in a database of known smart money addresses
    const knownAddresses = [
      // Add known smart money addresses here
      // These could include known funds, whales, or smart traders
    ];

    return knownAddresses.includes(address.toLowerCase());
  }

  private calculateConfidence(trade: any): number {
    let confidence = 50; // Base confidence

    // Increase confidence based on trade size
    const tradeAmount = parseFloat(trade.tradeAmountUsd);
    if (tradeAmount > 1000000) confidence += 30;
    else if (tradeAmount > 500000) confidence += 20;
    else if (tradeAmount > 100000) confidence += 10;

    // Increase confidence if we have smart money score
    if (trade.smartMoney && trade.smartMoney.scoreDecile) {
      confidence += trade.smartMoney.scoreDecile * 2;
    }

    // Increase confidence for known addresses
    if (this.isKnownSmartMoneyAddress(trade.maker.address) || 
        this.isKnownSmartMoneyAddress(trade.taker.address)) {
      confidence += 20;
    }

    return Math.min(100, confidence);
  }

  // Additional utility methods

  async getFlashLoanAttacks(limit: number = 10): Promise<Array<{
    hash: string;
    attacker: string;
    target: string;
    amountUsd: number;
    timestamp: Date;
    protocol: string;
  }>> {
    // This would implement a query to detect flash loan attacks
    // For now, return empty array
    return [];
  }

  async getMEVTransactions(symbol: string, limit: number = 50): Promise<Array<{
    hash: string;
    miner: string;
    extractedValue: number;
    gasUsed: number;
    timestamp: Date;
    type: 'arbitrage' | 'sandwich' | 'liquidation';
  }>> {
    // This would implement MEV detection queries
    // For now, return empty array
    return [];
  }

  async getArbitrageOpportunities(baseSymbol: string, quoteSymbol: string): Promise<Array<{
    exchange1: string;
    exchange2: string;
    price1: number;
    price2: number;
    spread: number;
    spreadPercent: number;
    timestamp: Date;
  }>> {
    // This would implement arbitrage opportunity detection
    // For now, return empty array
    return [];
  }
}