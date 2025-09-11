import { supabase } from '../lib/supabase';
import CryptoJS from 'crypto-js';

export interface ApiKeyData {
  id?: string;
  provider: string;
  key_name: string;
  value: string; // Plain text value (never stored)
  is_active: boolean;
  last_validated_at?: string | null;
  validation_status: 'pending' | 'valid' | 'invalid' | 'error';
  validation_error?: string | null;
}

export interface StoredApiKey {
  id: string;
  provider: string;
  key_name: string;
  is_active: boolean;
  last_validated_at: string | null;
  validation_status: 'pending' | 'valid' | 'invalid' | 'error';
  validation_error: string | null;
  created_at: string;
  updated_at: string;
}

// API provider configurations
export const API_PROVIDERS = {
  coingecko: {
    name: 'CoinGecko',
    keys: [{ name: 'api_key', label: 'API Key', required: false }],
    description: 'CoinGecko Pro API for enhanced rate limits and features',
  },
  whalealert: {
    name: 'WhaleAlert',
    keys: [{ name: 'api_key', label: 'API Key', required: true }],
    description: 'Real-time whale transaction alerts',
  },
  etherscan: {
    name: 'Etherscan',
    keys: [{ name: 'api_key', label: 'API Key', required: true }],
    description: 'Ethereum blockchain data and analytics',
  },
  bitquery: {
    name: 'Bitquery',
    keys: [{ name: 'api_key', label: 'API Key', required: true }],
    description: 'Blockchain data APIs for multiple networks',
  },
  covalent: {
    name: 'Covalent',
    keys: [{ name: 'api_key', label: 'API Key', required: true }],
    description: 'Multi-blockchain data API',
  },
  coinglass: {
    name: 'Coinglass',
    keys: [{ name: 'api_key', label: 'API Key', required: true }],
    description: 'Crypto derivatives and futures data',
  },
  alpaca: {
    name: 'Alpaca',
    keys: [
      { name: 'api_key', label: 'API Key', required: true },
      { name: 'secret_key', label: 'Secret Key', required: true },
    ],
    description: 'Commission-free trading API',
  },
  groq: {
    name: 'Groq',
    keys: [{ name: 'api_key', label: 'API Key', required: true }],
    description: 'High-performance AI inference',
  },
  binance: {
    name: 'Binance',
    keys: [
      { name: 'api_key', label: 'API Key', required: false },
      { name: 'secret_key', label: 'Secret Key', required: false },
    ],
    description: 'Binance exchange API (future implementation)',
  },
} as const;

class ApiKeysService {
  private encryptionKey: string;

  constructor() {
    // Use a combination of environment variable and browser fingerprint for encryption
    // This provides basic security while keeping keys accessible to the user
    this.encryptionKey = this.generateEncryptionKey();
  }

  private generateEncryptionKey(): string {
    try {
      // Get base key from environment (fallback to default for development)
      const baseKey = import.meta.env.VITE_ENCRYPTION_KEY || 'trading-bot-default-key-2025';
      
      // Add browser fingerprint for additional security
      const fingerprint = navigator.userAgent + navigator.language + screen.width + screen.height;
      
      console.log(`🔑 Generating encryption key with base: ${baseKey ? 'present' : 'missing'}, fingerprint length: ${fingerprint.length}`);
      
      // Create a derived key using PBKDF2
      const derivedKey = CryptoJS.PBKDF2(baseKey + fingerprint, 'trading-bot-salt', {
        keySize: 256 / 32,
        iterations: 1000,
      }).toString();
      
      console.log(`🔑 Encryption key generated successfully, length: ${derivedKey.length}`);
      return derivedKey;
    } catch (error) {
      console.error('Error generating encryption key:', error);
      // Fallback to a simple key if PBKDF2 fails
      return 'fallback-encryption-key-2025';
    }
  }

  private encrypt(value: string): string {
    try {
      if (!value || typeof value !== 'string') {
        throw new Error('Invalid value for encryption');
      }
      
      if (!this.encryptionKey) {
        throw new Error('Encryption key not initialized');
      }
      
      console.log(`🔒 Encrypting value of length: ${value.length}`);
      const encrypted = CryptoJS.AES.encrypt(value, this.encryptionKey).toString();
      console.log(`🔒 Encryption successful, result length: ${encrypted.length}`);
      return encrypted;
    } catch (error) {
      console.error('Encryption error:', error);
      throw new Error(`Encryption failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private decrypt(encryptedValue: string): string {
    const bytes = CryptoJS.AES.decrypt(encryptedValue, this.encryptionKey);
    return bytes.toString(CryptoJS.enc.Utf8);
  }

  async getApiKeys(): Promise<StoredApiKey[]> {
    try {
      console.log('🔍 Fetching API keys...');
      
      // Test authentication first
      const { data: { session }, error: sessionError } = await supabase.auth.getSession();
      if (sessionError) {
        console.error('Session error in getApiKeys:', sessionError);
        throw new Error(`Session error: ${sessionError.message}`);
      }
      
      if (!session) {
        console.error('No active session found in getApiKeys');
        throw new Error('No active session. Please log in again.');
      }
      
      console.log(`✅ Session found in getApiKeys, user: ${session.user.id}`);
      
      const { data: user, error: authError } = await supabase.auth.getUser();
      if (authError) {
        console.error('Authentication error in getApiKeys:', authError);
        throw new Error(`Authentication failed: ${authError.message}`);
      }
      
      if (!user.user) {
        console.error('No user found in auth response in getApiKeys');
        throw new Error('User not authenticated');
      }

      console.log(`👤 User authenticated in getApiKeys: ${user.user.id}`);

      const { data, error } = await supabase
        .from('api_keys')
        .select('*')
        .eq('user_id', user.user.id)
        .order('provider', { ascending: true });

      if (error) {
        console.error('Database error in getApiKeys:', error);
        throw error;
      }

      console.log(`✅ Successfully fetched ${data?.length || 0} API keys`);
      return data || [];
    } catch (error) {
      console.error('Error fetching API keys:', error);
      throw error;
    }
  }

  async getApiKey(provider: string, keyName: string): Promise<string | null> {
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user.user) throw new Error('User not authenticated');

      const { data, error } = await supabase
        .from('api_keys')
        .select('encrypted_value')
        .eq('user_id', user.user.id)
        .eq('provider', provider)
        .eq('key_name', keyName)
        .eq('is_active', true)
        .single();

      if (error) {
        if (error.code === 'PGRST116') {
          // No rows found - return null instead of throwing
          return null;
        }
        throw error;
      }

      if (!data) return null;

      return this.decrypt(data.encrypted_value);
    } catch (error) {
      console.error(`Error fetching API key for ${provider}.${keyName}:`, error);
      return null;
    }
  }

  async saveApiKey(apiKeyData: ApiKeyData): Promise<void> {
    try {
      console.log(`🔐 Attempting to save API key for ${apiKeyData.provider}.${apiKeyData.key_name}`);
      
      // Test authentication first
      const { data: { session }, error: sessionError } = await supabase.auth.getSession();
      if (sessionError) {
        console.error('Session error:', sessionError);
        throw new Error(`Session error: ${sessionError.message}`);
      }
      
      if (!session) {
        console.error('No active session found');
        throw new Error('No active session. Please log in again.');
      }
      
      console.log(`✅ Session found, user: ${session.user.id}`);
      
      const { data: user, error: authError } = await supabase.auth.getUser();
      if (authError) {
        console.error('Authentication error:', authError);
        throw new Error(`Authentication failed: ${authError.message}`);
      }
      
      if (!user.user) {
        console.error('No user found in auth response');
        throw new Error('User not authenticated');
      }

      console.log(`👤 User authenticated: ${user.user.id}`);

      // Test database connection before attempting to save
      console.log('🧪 Testing database connection...');
      const { error: testError } = await supabase
        .from('api_keys')
        .select('count')
        .limit(1);
      
      if (testError) {
        console.error('Database connection test failed:', testError);
        throw new Error(`Database connection failed: ${testError.message}`);
      }
      
      console.log('✅ Database connection test passed');

      const encryptedValue = this.encrypt(apiKeyData.value);
      console.log(`🔒 Value encrypted successfully, length: ${encryptedValue.length}`);

      // Log the exact data being sent to Supabase
      const insertData = {
        user_id: user.user.id,
        provider: apiKeyData.provider,
        key_name: apiKeyData.key_name,
        encrypted_value: encryptedValue,
        is_active: apiKeyData.is_active,
        validation_status: 'pending',
        validation_error: null,
      };
      
      console.log('📤 Inserting data:', {
        ...insertData,
        encrypted_value: `${insertData.encrypted_value.substring(0, 20)}...` // Don't log full encrypted value
      });

      const { error } = await supabase
        .from('api_keys')
        .upsert(insertData, {
          onConflict: 'user_id,provider,key_name',
        });

      if (error) {
        console.error('Supabase upsert error:', error);
        console.error('Error details:', {
          code: error.code,
          message: error.message,
          details: error.details,
          hint: error.hint
        });
        throw error;
      }

      console.log(`✅ API key saved for ${apiKeyData.provider}.${apiKeyData.key_name}`);
    } catch (error) {
      console.error('Error saving API key:', error);
      
      // Provide more specific error messages
      if (error instanceof Error) {
        if (error.message.includes('permission denied')) {
          throw new Error('Permission denied. Please check your account status.');
        } else if (error.message.includes('duplicate key')) {
          throw new Error('This API key already exists for this provider.');
        } else if (error.message.includes('foreign key')) {
          throw new Error('Invalid user reference. Please log in again.');
        } else if (error.message.includes('not null')) {
          throw new Error('Missing required field. Please check your input.');
        } else {
          throw new Error(`Failed to save API key: ${error.message}`);
        }
      } else {
        throw new Error('Failed to save API key. Please try again.');
      }
    }
  }

  async deleteApiKey(provider: string, keyName: string): Promise<void> {
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user.user) throw new Error('User not authenticated');

      const { error } = await supabase
        .from('api_keys')
        .delete()
        .eq('user_id', user.user.id)
        .eq('provider', provider)
        .eq('key_name', keyName);

      if (error) throw error;

      console.log(`🗑️ API key deleted for ${provider}.${keyName}`);
    } catch (error) {
      console.error('Error deleting API key:', error);
      throw error;
    }
  }

  async updateValidationStatus(
    provider: string,
    keyName: string,
    status: 'valid' | 'invalid' | 'error',
    error?: string
  ): Promise<void> {
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user.user) throw new Error('User not authenticated');

      const { error: updateError } = await supabase
        .from('api_keys')
        .update({
          validation_status: status,
          validation_error: error || null,
          last_validated_at: new Date().toISOString(),
        })
        .eq('user_id', user.user.id)
        .eq('provider', provider)
        .eq('key_name', keyName);

      if (updateError) throw updateError;
    } catch (error) {
      console.error('Error updating validation status:', error);
      // Don't throw here as this is not critical
    }
  }

  // Test function to debug authentication and database access
  async testConnection(): Promise<{ auth: boolean; database: boolean; error?: string }> {
    try {
      console.log('🧪 Testing API Keys Service connection...');
      
      // Test authentication
      const { data: { session }, error: sessionError } = await supabase.auth.getSession();
      if (sessionError || !session) {
        return { auth: false, database: false, error: 'No active session' };
      }
      
      console.log('✅ Authentication test passed');
      
      // Test database access
      const { error: dbError } = await supabase
        .from('api_keys')
        .select('count')
        .limit(1);
      
      if (dbError) {
        return { auth: true, database: false, error: dbError.message };
      }
      
      console.log('✅ Database access test passed');
      return { auth: true, database: true };
    } catch (error) {
      return { 
        auth: false, 
        database: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }

  // Comprehensive test method for debugging
  async debugConnection(): Promise<{
    auth: { hasSession: boolean; userId?: string; error?: string };
    database: { canRead: boolean; canWrite: boolean; error?: string };
    encryption: { canEncrypt: boolean; canDecrypt: boolean; error?: string };
  }> {
    const result = {
      auth: { hasSession: false, userId: undefined as string | undefined, error: undefined as string | undefined },
      database: { canRead: false, canWrite: false, error: undefined as string | undefined },
      encryption: { canEncrypt: false, canDecrypt: false, error: undefined as string | undefined }
    };

    try {
      // Test authentication
      console.log('🔐 Testing authentication...');
      const { data: { session }, error: sessionError } = await supabase.auth.getSession();
      if (sessionError) {
        result.auth.error = sessionError.message;
      } else if (session) {
        result.auth.hasSession = true;
        result.auth.userId = session.user.id;
        console.log('✅ Authentication: OK');
      } else {
        result.auth.error = 'No active session';
      }

      // Test database read access
      if (result.auth.hasSession) {
        console.log('📖 Testing database read access...');
        const { error: readError } = await supabase
          .from('api_keys')
          .select('count')
          .limit(1);
        
        if (readError) {
          result.database.error = `Read failed: ${readError.message}`;
        } else {
          result.database.canRead = true;
          console.log('✅ Database read: OK');
        }

        // Test database write access with a temporary record
        if (result.database.canRead) {
          console.log('✍️ Testing database write access...');
          const testData = {
            user_id: session!.user.id,
            provider: 'test_provider',
            key_name: 'test_key',
            encrypted_value: 'test_encrypted_value',
            is_active: false,
            validation_status: 'pending' as const,
            validation_error: null,
          };

          const { error: writeError } = await supabase
            .from('api_keys')
            .upsert(testData, {
              onConflict: 'user_id,provider,key_name'
            });

          if (writeError) {
            result.database.error = `Write failed: ${writeError.message}`;
          } else {
            result.database.canWrite = true;
            console.log('✅ Database write: OK');

            // Clean up test record
            await supabase
              .from('api_keys')
              .delete()
              .eq('provider', 'test_provider')
              .eq('key_name', 'test_key');
          }
        }
      }

      // Test encryption
      console.log('🔒 Testing encryption...');
      try {
        const testValue = 'test_value_123';
        const encrypted = this.encrypt(testValue);
        const decrypted = this.decrypt(encrypted);
        
        if (decrypted === testValue) {
          result.encryption.canEncrypt = true;
          result.encryption.canDecrypt = true;
          console.log('✅ Encryption: OK');
        } else {
          result.encryption.error = 'Encryption/decryption mismatch';
        }
      } catch (encError) {
        result.encryption.error = encError instanceof Error ? encError.message : 'Unknown encryption error';
      }

    } catch (error) {
      console.error('Debug connection error:', error);
      if (!result.auth.error) {
        result.auth.error = error instanceof Error ? error.message : 'Unknown error';
      }
    }

    return result;
  }

  async validateApiKey(provider: string, keyName: string): Promise<boolean> {
    try {
      const apiKey = await this.getApiKey(provider, keyName);
      if (!apiKey) {
        await this.updateValidationStatus(provider, keyName, 'error', 'API key not found');
        return false;
      }

      let isValid = false;
      let validationError = '';

      // Validate based on provider
      switch (provider) {
        case 'coingecko':
          isValid = await this.validateCoinGeckoKey(apiKey);
          break;
        case 'alpaca':
          if (keyName === 'api_key') {
            const secretKey = await this.getApiKey(provider, 'secret_key');
            isValid = await this.validateAlpacaKeys(apiKey, secretKey || '');
          }
          break;
        case 'groq':
          isValid = await this.validateGroqKey(apiKey);
          break;
        default:
          // For other providers, just check if key is not empty
          isValid = apiKey.length > 0;
      }

      const status = isValid ? 'valid' : 'invalid';
      await this.updateValidationStatus(provider, keyName, status, validationError);

      return isValid;
    } catch (error) {
      console.error(`Error validating ${provider}.${keyName}:`, error);
      await this.updateValidationStatus(provider, keyName, 'error', String(error));
      return false;
    }
  }

  private async validateCoinGeckoKey(apiKey: string): Promise<boolean> {
    try {
      const response = await fetch('https://api.coingecko.com/api/v3/ping', {
        headers: {
          'x-cg-pro-api-key': apiKey,
        },
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  private async validateAlpacaKeys(apiKey: string, secretKey: string): Promise<boolean> {
    try {
      const response = await fetch('https://paper-api.alpaca.markets/v2/account', {
        headers: {
          'APCA-API-KEY-ID': apiKey,
          'APCA-API-SECRET-KEY': secretKey,
        },
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  private async validateGroqKey(apiKey: string): Promise<boolean> {
    try {
      const response = await fetch('https://api.groq.com/openai/v1/models', {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
        },
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  // Get API key with fallback to environment variables
  async getApiKeyWithFallback(provider: string, keyName: string): Promise<string | null> {
    // First try to get from stored keys
    const storedKey = await this.getApiKey(provider, keyName);
    if (storedKey) return storedKey;

    // Fallback to environment variables
    const envMap: Record<string, string> = {
      'coingecko.api_key': 'VITE_COINGECKO_API_KEY',
      'alpaca.api_key': 'VITE_ALPACA_API_KEY',
      'alpaca.secret_key': 'VITE_ALPACA_SECRET_KEY',
      'groq.api_key': 'VITE_GROQ_API_KEY',
      // Add more mappings as needed
    };

    const envKey = envMap[`${provider}.${keyName}`];
    if (envKey) {
      const envValue = import.meta.env[envKey];
      if (envValue) {
        console.log(`📋 Using environment variable ${envKey} for ${provider}.${keyName}`);
        return envValue;
      }
    }

    return null;
  }
}

export const apiKeysService = new ApiKeysService();