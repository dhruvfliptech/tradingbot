import fetch from 'node-fetch';
import logger from '../../../utils/logger';

export interface CovalentConfig {
  apiKey: string;
  endpoint: string;
  rateLimit: number;
}

export interface CovalentAddressData {
  address: string;
  balance: number;
  transactions: CovalentTransaction[];
  tokenHoldings: CovalentTokenBalance[];
  lastActivity: Date;
  chainId: number;
}

export interface CovalentTransaction {
  hash: string;
  from: string;
  to: string;
  value: string;
  timestamp: string;
  gasUsed: number;
  gasPrice: number;
  logEvents: any[];
}

export interface CovalentTokenBalance {
  contractAddress: string;
  contractName: string;
  contractTickerSymbol: string;
  contractDecimals: number;
  balance: string;
  quote: number;
  quoteRate: number;
  logo: string;
}

export interface CovalentNftData {
  contractAddress: string;
  tokenId: string;
  tokenBalance: string;
  tokenUrl: string;
  metadata: any;
  originalOwner: string;
}

export class CovalentClient {
  private config: CovalentConfig;
  private baseUrl: string;

  constructor(config: CovalentConfig) {
    this.config = config;
    this.baseUrl = config.endpoint || 'https://api.covalenthq.com/v1';
  }

  async testConnection(): Promise<void> {
    try {
      // Test with a simple chains endpoint
      const response = await this.makeRequest('/chains/');
      
      if (!response.data || response.error) {
        throw new Error(`Covalent API test failed: ${response.error_message || 'Unknown error'}`);
      }

      logger.debug('Covalent API connection test successful');
    } catch (error) {
      logger.error('Covalent API connection test failed:', error);
      throw error;
    }
  }

  async getAddressData(address: string, chainId: number = 1): Promise<CovalentAddressData | null> {
    try {
      logger.debug(`Fetching Covalent data for address: ${address} on chain ${chainId}`);

      // Get token balances
      const balancesResponse = await this.makeRequest(`/${chainId}/address/${address}/balances_v2/`);
      
      if (balancesResponse.error) {
        logger.warn(`Failed to get balances for ${address}: ${balancesResponse.error_message}`);
        return null;
      }

      const items = balancesResponse.data?.items || [];
      
      // Get native token balance (ETH, MATIC, etc.)
      const nativeToken = items.find((item: any) => item.native_token);
      const balance = nativeToken ? parseFloat(nativeToken.balance) / Math.pow(10, nativeToken.contract_decimals) : 0;

      // Get transactions
      const transactionsResponse = await this.makeRequest(`/${chainId}/address/${address}/transactions_v2/`, {
        'page-size': 100
      });

      const transactions = transactionsResponse.data?.items || [];

      // Get token holdings (excluding native token)
      const tokenHoldings = items
        .filter((item: any) => !item.native_token && parseFloat(item.balance) > 0)
        .map((item: any) => ({
          contractAddress: item.contract_address,
          contractName: item.contract_name,
          contractTickerSymbol: item.contract_ticker_symbol,
          contractDecimals: item.contract_decimals,
          balance: item.balance,
          quote: item.quote || 0,
          quoteRate: item.quote_rate || 0,
          logo: item.logo_url
        }));

      // Find last activity
      const lastActivity = transactions.length > 0 
        ? new Date(transactions[0].block_signed_at)
        : new Date(0);

      return {
        address,
        balance,
        transactions: transactions.map((tx: any) => ({
          hash: tx.tx_hash,
          from: tx.from_address,
          to: tx.to_address,
          value: tx.value,
          timestamp: tx.block_signed_at,
          gasUsed: tx.gas_spent,
          gasPrice: tx.gas_price,
          logEvents: tx.log_events || []
        })),
        tokenHoldings,
        lastActivity,
        chainId
      };

    } catch (error) {
      logger.error(`Error fetching Covalent data for ${address}:`, error);
      return null;
    }
  }

  async getTokenTransfers(contractAddress: string, chainId: number = 1, limit: number = 100): Promise<Array<{
    hash: string;
    from: string;
    to: string;
    value: number;
    timestamp: Date;
    blockHeight: number;
  }>> {
    try {
      const response = await this.makeRequest(`/${chainId}/tokens/${contractAddress}/token_holders_changes/`, {
        'page-size': limit
      });

      if (response.error) {
        logger.warn(`Failed to get token transfers: ${response.error_message}`);
        return [];
      }

      const items = response.data?.items || [];
      
      return items.map((item: any) => ({
        hash: item.tx_hash,
        from: item.from_address,
        to: item.to_address,
        value: parseFloat(item.delta) / Math.pow(10, item.contract_decimals),
        timestamp: new Date(item.block_signed_at),
        blockHeight: item.block_height
      }));

    } catch (error) {
      logger.error(`Error fetching token transfers for ${contractAddress}:`, error);
      return [];
    }
  }

  async getTokenHolders(contractAddress: string, chainId: number = 1): Promise<Array<{
    address: string;
    balance: number;
    percentage: number;
  }>> {
    try {
      const response = await this.makeRequest(`/${chainId}/tokens/${contractAddress}/token_holders/`);

      if (response.error) {
        logger.warn(`Failed to get token holders: ${response.error_message}`);
        return [];
      }

      const items = response.data?.items || [];
      const totalSupply = response.data?.total_supply || 0;
      
      return items.map((item: any) => ({
        address: item.address,
        balance: parseFloat(item.balance) / Math.pow(10, item.contract_decimals),
        percentage: totalSupply > 0 ? (parseFloat(item.balance) / totalSupply) * 100 : 0
      }));

    } catch (error) {
      logger.error(`Error fetching token holders for ${contractAddress}:`, error);
      return [];
    }
  }

  async getDexTransactions(dexName: string, chainId: number = 1): Promise<Array<{
    hash: string;
    exchange: string;
    swapCount: number;
    totalVolumeUsd: number;
    timestamp: Date;
  }>> {
    try {
      const response = await this.makeRequest(`/${chainId}/xy=k/${dexName}/transactions/`);

      if (response.error) {
        logger.warn(`Failed to get DEX transactions: ${response.error_message}`);
        return [];
      }

      const items = response.data?.items || [];
      
      return items.map((item: any) => ({
        hash: item.tx_hash,
        exchange: item.exchange,
        swapCount: item.swap_count_24h,
        totalVolumeUsd: item.total_liquidity_quote,
        timestamp: new Date(item.block_signed_at)
      }));

    } catch (error) {
      logger.error(`Error fetching DEX transactions for ${dexName}:`, error);
      return [];
    }
  }

  async getNftTransactions(contractAddress: string, tokenId: string, chainId: number = 1): Promise<Array<{
    hash: string;
    from: string;
    to: string;
    timestamp: Date;
    value: number;
  }>> {
    try {
      const response = await this.makeRequest(`/${chainId}/tokens/${contractAddress}/nft_transactions/${tokenId}/`);

      if (response.error) {
        logger.warn(`Failed to get NFT transactions: ${response.error_message}`);
        return [];
      }

      const items = response.data?.items || [];
      
      return items.map((item: any) => ({
        hash: item.tx_hash,
        from: item.from_address,
        to: item.to_address,
        timestamp: new Date(item.block_signed_at),
        value: parseFloat(item.value) || 0
      }));

    } catch (error) {
      logger.error(`Error fetching NFT transactions for ${contractAddress}:${tokenId}:`, error);
      return [];
    }
  }

  async getPortfolioValue(address: string, chainId: number = 1): Promise<{
    totalValueUsd: number;
    holdings: Array<{
      symbol: string;
      balance: number;
      valueUsd: number;
      percentage: number;
    }>;
  }> {
    try {
      const response = await this.makeRequest(`/${chainId}/address/${address}/portfolio_v2/`);

      if (response.error) {
        logger.warn(`Failed to get portfolio value: ${response.error_message}`);
        return { totalValueUsd: 0, holdings: [] };
      }

      const items = response.data?.items || [];
      const totalValue = items.reduce((sum: number, item: any) => sum + (item.quote || 0), 0);
      
      const holdings = items
        .filter((item: any) => item.quote > 0)
        .map((item: any) => ({
          symbol: item.contract_ticker_symbol,
          balance: parseFloat(item.balance) / Math.pow(10, item.contract_decimals),
          valueUsd: item.quote,
          percentage: totalValue > 0 ? (item.quote / totalValue) * 100 : 0
        }))
        .sort((a, b) => b.valueUsd - a.valueUsd);

      return {
        totalValueUsd: totalValue,
        holdings
      };

    } catch (error) {
      logger.error(`Error fetching portfolio value for ${address}:`, error);
      return { totalValueUsd: 0, holdings: [] };
    }
  }

  async getGasPrices(chainId: number = 1): Promise<{
    slow: number;
    standard: number;
    fast: number;
    instant: number;
  }> {
    try {
      // This endpoint might not be available in all Covalent plans
      // Returning reasonable defaults based on chain
      const defaults: Record<number, any> = {
        1: { slow: 20, standard: 25, fast: 30, instant: 35 }, // Ethereum
        137: { slow: 1, standard: 2, fast: 3, instant: 5 },   // Polygon
        56: { slow: 3, standard: 5, fast: 7, instant: 10 },   // BSC
        43114: { slow: 25, standard: 30, fast: 35, instant: 40 } // Avalanche
      };

      return defaults[chainId] || defaults[1];

    } catch (error) {
      logger.error(`Error fetching gas prices for chain ${chainId}:`, error);
      return { slow: 20, standard: 25, fast: 30, instant: 35 };
    }
  }

  async getChainStatus(chainId: number): Promise<{
    name: string;
    isTestnet: boolean;
    blockHeight: number;
    gasTokenSymbol: string;
    logoUrl: string;
  }> {
    try {
      const response = await this.makeRequest(`/chains/status/`);

      if (response.error) {
        throw new Error(response.error_message);
      }

      const chain = response.data?.items?.find((item: any) => item.chain_id === chainId);
      
      if (!chain) {
        throw new Error(`Chain ${chainId} not found`);
      }

      return {
        name: chain.name,
        isTestnet: chain.is_testnet,
        blockHeight: chain.synced_block_height,
        gasTokenSymbol: chain.native_token.contract_ticker_symbol,
        logoUrl: chain.logo_url
      };

    } catch (error) {
      logger.error(`Error fetching chain status for ${chainId}:`, error);
      throw error;
    }
  }

  async getSupportedChains(): Promise<Array<{
    chainId: number;
    name: string;
    symbol: string;
    isTestnet: boolean;
  }>> {
    try {
      const response = await this.makeRequest('/chains/');

      if (response.error) {
        throw new Error(response.error_message);
      }

      const chains = response.data?.items || [];
      
      return chains.map((chain: any) => ({
        chainId: chain.chain_id,
        name: chain.name,
        symbol: chain.native_token.contract_ticker_symbol,
        isTestnet: chain.is_testnet
      }));

    } catch (error) {
      logger.error('Error fetching supported chains:', error);
      return [];
    }
  }

  private async makeRequest(endpoint: string, params: Record<string, any> = {}): Promise<any> {
    const url = new URL(`${this.baseUrl}${endpoint}`);
    
    // Add query parameters
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.set(key, value.toString());
    });

    try {
      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${this.config.apiKey}`,
          'User-Agent': 'TradingBot/1.0'
        },
        timeout: 15000
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return data;

    } catch (error) {
      logger.error(`Covalent API request failed: ${url.toString()}`, error);
      throw error;
    }
  }

  // Cross-chain analysis methods

  async getMultiChainBalance(address: string, chainIds: number[] = [1, 137, 56]): Promise<{
    totalValueUsd: number;
    balancesByChain: Array<{
      chainId: number;
      chainName: string;
      valueUsd: number;
      tokens: number;
    }>;
  }> {
    try {
      const balancePromises = chainIds.map(async (chainId) => {
        try {
          const portfolio = await this.getPortfolioValue(address, chainId);
          const chainStatus = await this.getChainStatus(chainId);
          
          return {
            chainId,
            chainName: chainStatus.name,
            valueUsd: portfolio.totalValueUsd,
            tokens: portfolio.holdings.length
          };
        } catch (error) {
          logger.warn(`Failed to get balance for chain ${chainId}:`, error);
          return {
            chainId,
            chainName: `Chain ${chainId}`,
            valueUsd: 0,
            tokens: 0
          };
        }
      });

      const balancesByChain = await Promise.all(balancePromises);
      const totalValueUsd = balancesByChain.reduce((sum, balance) => sum + balance.valueUsd, 0);

      return {
        totalValueUsd,
        balancesByChain: balancesByChain.filter(balance => balance.valueUsd > 0)
      };

    } catch (error) {
      logger.error(`Error fetching multi-chain balance for ${address}:`, error);
      return {
        totalValueUsd: 0,
        balancesByChain: []
      };
    }
  }

  async getBridgeTransactions(address: string, chainIds: number[] = [1, 137, 56]): Promise<Array<{
    hash: string;
    fromChain: number;
    toChain: number;
    amount: number;
    symbol: string;
    timestamp: Date;
    bridge: string;
  }>> {
    // This would require more complex analysis of cross-chain bridge contracts
    // For now, return empty array as this is an advanced feature
    return [];
  }

  async getYieldFarmingPositions(address: string, chainId: number = 1): Promise<Array<{
    protocol: string;
    pool: string;
    stakedAmount: number;
    stakedValueUsd: number;
    rewardsEarned: number;
    apr: number;
  }>> {
    // This would require integration with DeFi protocols
    // For now, return empty array as this is an advanced feature
    return [];
  }
}