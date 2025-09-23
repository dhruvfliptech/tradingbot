import fetch from 'node-fetch';
import logger from '../../../utils/logger';

export interface EtherscanConfig {
  apiKey: string;
  endpoint: string;
  rateLimit: number;
}

export interface EtherscanAddressData {
  address: string;
  balance: number;
  transactions: EtherscanTransaction[];
  tokenHoldings: EtherscanTokenBalance[];
  lastActivity: Date;
}

export interface EtherscanTransaction {
  hash: string;
  from: string;
  to: string;
  value: string;
  token?: string;
  tokenSymbol?: string;
  timestamp: string;
  gasUsed: string;
  gasPrice: string;
  confirmations: string;
  isError: string;
}

export interface EtherscanTokenBalance {
  contractAddress: string;
  tokenName: string;
  tokenSymbol: string;
  tokenDecimal: string;
  balance: string;
}

export interface EtherscanLargeTransaction {
  hash: string;
  from: string;
  to: string;
  value: number;
  valueUsd: number;
  timestamp: Date;
  token: string;
  symbol: string;
}

export class EtherscanClient {
  private config: EtherscanConfig;
  private baseUrl: string;

  constructor(config: EtherscanConfig) {
    this.config = config;
    this.baseUrl = config.endpoint || 'https://api.etherscan.io/api';
  }

  async testConnection(): Promise<void> {
    try {
      const response = await this.makeRequest('account', 'balance', {
        address: '0x0000000000000000000000000000000000000000',
        tag: 'latest'
      });

      if (response.status !== '1' && response.message !== 'OK') {
        throw new Error(`Etherscan API test failed: ${response.message}`);
      }

      logger.debug('Etherscan API connection test successful');
    } catch (error) {
      logger.error('Etherscan API connection test failed:', error);
      throw error;
    }
  }

  async getAddressData(address: string): Promise<EtherscanAddressData | null> {
    try {
      logger.debug(`Fetching Etherscan data for address: ${address}`);

      // Get ETH balance
      const balanceResponse = await this.makeRequest('account', 'balance', {
        address,
        tag: 'latest'
      });

      if (balanceResponse.status !== '1') {
        logger.warn(`Failed to get balance for ${address}: ${balanceResponse.message}`);
        return null;
      }

      const balance = parseFloat(balanceResponse.result) / Math.pow(10, 18); // Convert wei to ETH

      // Get normal transactions
      const txResponse = await this.makeRequest('account', 'txlist', {
        address,
        startblock: '0',
        endblock: '99999999',
        page: '1',
        offset: '100',
        sort: 'desc'
      });

      const transactions = txResponse.status === '1' ? txResponse.result : [];

      // Get token transactions
      const tokenTxResponse = await this.makeRequest('account', 'tokentx', {
        address,
        startblock: '0',
        endblock: '99999999',
        page: '1',
        offset: '100',
        sort: 'desc'
      });

      const tokenTransactions = tokenTxResponse.status === '1' ? tokenTxResponse.result : [];

      // Get token balances
      const tokenBalances = await this.getTokenBalances(address);

      // Combine all transactions and find last activity
      const allTxs = [...transactions, ...tokenTransactions];
      const lastActivity = allTxs.length > 0 
        ? new Date(parseInt(allTxs[0].timeStamp) * 1000)
        : new Date(0);

      return {
        address,
        balance,
        transactions: allTxs,
        tokenHoldings: tokenBalances,
        lastActivity
      };

    } catch (error) {
      logger.error(`Error fetching Etherscan data for ${address}:`, error);
      return null;
    }
  }

  async getLargeTransactions(symbol: string, minValueUsd: number = 100000): Promise<EtherscanLargeTransaction[]> {
    try {
      // Get contract address for token
      const contractAddress = await this.getTokenContractAddress(symbol);
      if (!contractAddress) {
        logger.warn(`No contract address found for symbol: ${symbol}`);
        return [];
      }

      // Get recent token transfers
      const response = await this.makeRequest('account', 'tokentx', {
        contractaddress: contractAddress,
        page: '1',
        offset: '100',
        sort: 'desc'
      });

      if (response.status !== '1') {
        logger.warn(`Failed to get token transfers for ${symbol}: ${response.message}`);
        return [];
      }

      const largeTransactions: EtherscanLargeTransaction[] = [];

      for (const tx of response.result) {
        try {
          const value = parseFloat(tx.value) / Math.pow(10, parseInt(tx.tokenDecimal));
          
          // Get USD value (this would need to be enriched with price data)
          const valueUsd = await this.estimateUsdValue(symbol, value);
          
          if (valueUsd >= minValueUsd) {
            largeTransactions.push({
              hash: tx.hash,
              from: tx.from,
              to: tx.to,
              value,
              valueUsd,
              timestamp: new Date(parseInt(tx.timeStamp) * 1000),
              token: contractAddress,
              symbol: tx.tokenSymbol || symbol
            });
          }
        } catch (error) {
          logger.warn(`Error processing transaction ${tx.hash}:`, error);
        }
      }

      return largeTransactions.sort((a, b) => b.valueUsd - a.valueUsd);

    } catch (error) {
      logger.error(`Error fetching large transactions for ${symbol}:`, error);
      return [];
    }
  }

  async getTokenBalances(address: string): Promise<EtherscanTokenBalance[]> {
    try {
      // Get list of ERC-20 token transfers to determine which tokens the address holds
      const tokenTxResponse = await this.makeRequest('account', 'tokentx', {
        address,
        startblock: '0',
        endblock: '99999999',
        page: '1',
        offset: '100',
        sort: 'desc'
      });

      if (tokenTxResponse.status !== '1') {
        return [];
      }

      // Get unique contract addresses
      const contractAddresses = [...new Set(
        tokenTxResponse.result.map((tx: any) => tx.contractAddress)
      )];

      const balances: EtherscanTokenBalance[] = [];

      // Get balance for each token (limited to prevent rate limiting)
      for (const contractAddress of contractAddresses.slice(0, 10)) {
        try {
          const balanceResponse = await this.makeRequest('account', 'tokenbalance', {
            contractaddress: contractAddress,
            address,
            tag: 'latest'
          });

          if (balanceResponse.status === '1' && balanceResponse.result !== '0') {
            // Get token info
            const tokenInfo = tokenTxResponse.result.find(
              (tx: any) => tx.contractAddress === contractAddress
            );

            balances.push({
              contractAddress,
              tokenName: tokenInfo?.tokenName || 'Unknown',
              tokenSymbol: tokenInfo?.tokenSymbol || 'UNK',
              tokenDecimal: tokenInfo?.tokenDecimal || '18',
              balance: balanceResponse.result
            });
          }
        } catch (error) {
          logger.warn(`Error getting token balance for ${contractAddress}:`, error);
        }
      }

      return balances;

    } catch (error) {
      logger.error(`Error getting token balances for ${address}:`, error);
      return [];
    }
  }

  async getGasTracker(): Promise<{
    standard: number;
    fast: number;
    rapid: number;
  }> {
    try {
      const response = await this.makeRequest('gastracker', 'gasoracle');
      
      if (response.status !== '1') {
        throw new Error(`Gas tracker failed: ${response.message}`);
      }

      return {
        standard: parseInt(response.result.SafeGasPrice),
        fast: parseInt(response.result.StandardGasPrice),
        rapid: parseInt(response.result.FastGasPrice)
      };

    } catch (error) {
      logger.error('Error getting gas tracker data:', error);
      throw error;
    }
  }

  private async makeRequest(module: string, action: string, params: Record<string, any> = {}): Promise<any> {
    const url = new URL(this.baseUrl);
    url.searchParams.set('module', module);
    url.searchParams.set('action', action);
    url.searchParams.set('apikey', this.config.apiKey);

    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.set(key, value.toString());
    });

    try {
      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          'User-Agent': 'TradingBot/1.0'
        },
        timeout: 10000
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return data;

    } catch (error) {
      logger.error(`Etherscan API request failed: ${url.toString()}`, error);
      throw error;
    }
  }

  private async getTokenContractAddress(symbol: string): Promise<string | null> {
    // Common token contract addresses (this should be expanded or fetched from a database)
    const tokenContracts: Record<string, string> = {
      'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
      'USDC': '0xA0b86a33E6441c8C0F8aF8f6c38e5Cc9A6c98c18',
      'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
      'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
      'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
      'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
      'LINK': '0x514910771AF9Ca656af840dff83E8264EcF986CA',
      'AAVE': '0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9'
    };

    return tokenContracts[symbol.toUpperCase()] || null;
  }

  private async estimateUsdValue(symbol: string, amount: number): Promise<number> {
    // This is a placeholder - in a real implementation, you would:
    // 1. Fetch current price from a price feed
    // 2. Calculate USD value
    // 3. Cache the price to avoid excessive API calls

    // For now, return a rough estimate based on symbol
    const roughPrices: Record<string, number> = {
      'WBTC': 45000,
      'ETH': 3000,
      'USDC': 1,
      'USDT': 1,
      'DAI': 1,
      'WETH': 3000,
      'UNI': 25,
      'LINK': 15,
      'AAVE': 100
    };

    const price = roughPrices[symbol.toUpperCase()] || 0;
    return amount * price;
  }

  // Additional utility methods

  async getContractInfo(contractAddress: string): Promise<{
    name: string;
    symbol: string;
    decimals: number;
    totalSupply: string;
  } | null> {
    try {
      // This would require additional API calls to get contract details
      // For now, return basic info
      return {
        name: 'Unknown Token',
        symbol: 'UNK',
        decimals: 18,
        totalSupply: '0'
      };
    } catch (error) {
      logger.error(`Error getting contract info for ${contractAddress}:`, error);
      return null;
    }
  }

  async getBlockNumber(): Promise<number> {
    try {
      const response = await this.makeRequest('proxy', 'eth_blockNumber');
      return parseInt(response.result, 16);
    } catch (error) {
      logger.error('Error getting block number:', error);
      throw error;
    }
  }

  async getTransactionReceipt(txHash: string): Promise<any> {
    try {
      const response = await this.makeRequest('proxy', 'eth_getTransactionReceipt', {
        txhash: txHash
      });
      return response.result;
    } catch (error) {
      logger.error(`Error getting transaction receipt for ${txHash}:`, error);
      throw error;
    }
  }
}