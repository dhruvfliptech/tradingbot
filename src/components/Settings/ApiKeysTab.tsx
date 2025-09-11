import React, { useState, useEffect } from 'react';
import { Eye, EyeOff, Save, Trash2, Check, X, AlertCircle, Key, RefreshCw, Info } from 'lucide-react';
import { apiKeysService, API_PROVIDERS, StoredApiKey, ApiKeyData } from '../../services/apiKeysService';
import { ApiConnectionStatus } from './ApiConnectionStatus';
import { ErrorBoundary } from './ErrorBoundary';

interface ApiKeyFormData {
  [provider: string]: {
    [keyName: string]: {
      value: string;
      visible: boolean;
    };
  };
}

export const ApiKeysTab: React.FC = () => {
  const [apiKeys, setApiKeys] = useState<StoredApiKey[]>([]);
  const [formData, setFormData] = useState<ApiKeyFormData>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);
  const [validating, setValidating] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Helper function to get environment variable for a provider and key
  const getEnvVarValue = (provider: string, keyName: string): string => {
    const envKey = `VITE_${provider.toUpperCase()}_${keyName.toUpperCase()}`;
    const envValue = import.meta.env[envKey];
    if (envValue) {
      console.log(`📋 Found environment variable ${envKey} for ${provider}.${keyName}`);
      return envValue;
    }
    return '';
  };

  // Function to initialize form data with environment variables
  const initializeFormDataWithEnvVars = () => {
    console.log('🔄 Initializing form data with environment variables...');
    const newFormData: ApiKeyFormData = {};
    
    Object.entries(API_PROVIDERS).forEach(([provider, config]) => {
      newFormData[provider] = {};
      config.keys.forEach(key => {
        const envValue = getEnvVarValue(provider, key.name);
        newFormData[provider][key.name] = {
          value: envValue,
          visible: false,
        };
      });
    });
    
    setFormData(newFormData);
    console.log('✅ Form data initialized with environment variables');
  };

  useEffect(() => {
    console.log('🔄 ApiKeysTab useEffect triggered');
    
    // Always initialize form data with environment variables first
    initializeFormDataWithEnvVars();
    
    // Then try to load from database
    loadApiKeys();
    
    // Fallback: ensure form data is populated even if database loading fails
    const fallbackTimer = setTimeout(() => {
      console.log('⏰ Fallback timer: ensuring form data is populated');
      if (Object.keys(formData).length === 0) {
        console.log('🔄 Form data still empty, re-initializing with environment variables');
        initializeFormDataWithEnvVars();
      }
    }, 3000); // 3 second fallback
    
    return () => clearTimeout(fallbackTimer);
  }, []);

  const loadApiKeys = async (retryCount = 0) => {
    try {
      setLoading(true);
      setError(null); // Clear any previous errors
      
      console.log(`🔄 Loading API keys... (attempt ${retryCount + 1})`);
      
      // Wait a bit for authentication to settle if this is a retry
      if (retryCount > 0) {
        await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
      }
      
      const keys = await apiKeysService.getApiKeys();
      console.log(`✅ Loaded ${keys.length} API keys`);
      
      setApiKeys(keys);
      
      // Initialize form data with environment variable fallbacks
      const newFormData: ApiKeyFormData = {};
      Object.entries(API_PROVIDERS).forEach(([provider, config]) => {
        newFormData[provider] = {};
        config.keys.forEach(key => {
          // Try to get value from stored keys first, then environment variables
          const storedKey = keys.find(k => k.provider === provider && k.key_name === key.name);
          let initialValue = '';
          
          if (!storedKey) {
            // No stored key, try environment variable
            initialValue = getEnvVarValue(provider, key.name);
          }
          
          newFormData[provider][key.name] = {
            value: initialValue,
            visible: false,
          };
        });
      });
      setFormData(newFormData);
      
      console.log('✅ Form data initialized successfully');
    } catch (err) {
      console.error('❌ Error in loadApiKeys:', err);
      
      // Check for specific error types
      if (err instanceof Error) {
        if (err.message.includes('Could not find the table')) {
          // Table doesn't exist yet, this is a setup issue
          setError('API keys table not found. Please contact support.');
        } else if (err.message.includes('No rows found')) {
          // No API keys stored yet, this is normal
          setApiKeys([]);
          // Still initialize form data with environment variable fallbacks
          const newFormData: ApiKeyFormData = {};
          Object.entries(API_PROVIDERS).forEach(([provider, config]) => {
            newFormData[provider] = {};
            config.keys.forEach(key => {
              // Try to get value from environment variables
              const initialValue = getEnvVarValue(provider, key.name);
              
              newFormData[provider][key.name] = {
                value: initialValue,
                visible: false,
              };
            });
          });
          setFormData(newFormData);
          console.log('✅ Initialized with environment variable fallbacks (normal state)');
        } else if (err.message.includes('No active session') || err.message.includes('User not authenticated')) {
          // Authentication issues - try to retry once
          if (retryCount === 0) {
            console.log('🔄 Authentication issue detected, retrying...');
            setTimeout(() => loadApiKeys(1), 2000);
            return;
          }
          setError('Authentication required. Please log in again.');
        } else if (err.message.includes('Session error')) {
          // Session issues - try to retry once
          if (retryCount === 0) {
            console.log('🔄 Session issue detected, retrying...');
            setTimeout(() => loadApiKeys(1), 2000);
            return;
          }
          setError('Session expired. Please refresh the page and log in again.');
        } else {
          // Other errors
          setError(`Failed to load API keys: ${err.message}`);
        }
      } else {
        // Unknown error type
        setError('Failed to load API keys. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (provider: string, keyName: string, value: string) => {
    try {
      console.log(`Input change for ${provider}.${keyName}:`, value.length, 'characters');
      
      setFormData(prev => {
        // Ensure the provider exists in the form data
        if (!prev[provider]) {
          console.warn(`Provider ${provider} not found in form data, initializing...`);
          prev[provider] = {};
        }
        // Ensure the keyName exists for this provider
        if (!prev[provider][keyName]) {
          console.warn(`Key ${keyName} not found for provider ${provider}, initializing...`);
          prev[provider][keyName] = {
            value: '',
            visible: false,
          };
        }
        
        return {
          ...prev,
          [provider]: {
            ...prev[provider],
            [keyName]: {
              ...prev[provider][keyName],
              value,
            },
          },
        };
      });
    } catch (error) {
      console.error('Error in handleInputChange:', error);
      setError('An error occurred while updating the input. Please try again.');
    }
  };

  const toggleVisibility = (provider: string, keyName: string) => {
    setFormData(prev => {
      // Ensure the provider and keyName exist in the form data
      if (!prev[provider]) {
        prev[provider] = {};
      }
      if (!prev[provider][keyName]) {
        prev[provider][keyName] = {
          value: '',
          visible: false,
        };
      }
      
      return {
        ...prev,
        [provider]: {
          ...prev[provider],
          [keyName]: {
            ...prev[provider][keyName],
            visible: !prev[provider][keyName].visible,
          },
        },
      };
    });
  };

  const saveApiKey = async (provider: string, keyName: string) => {
    const value = formData[provider]?.[keyName]?.value;
    if (!value) return;

    const saveKey = `${provider}.${keyName}`;
    try {
      setSaving(saveKey);
      setError(null);

      const apiKeyData: ApiKeyData = {
        provider,
        key_name: keyName,
        value,
        is_active: true,
        validation_status: 'pending',
      };

      await apiKeysService.saveApiKey(apiKeyData);
      
      // Clear the form field
      setFormData(prev => ({
        ...prev,
        [provider]: {
          ...prev[provider],
          [keyName]: {
            ...prev[provider][keyName],
            value: '',
          },
        },
      }));

      // Reload keys to show updated status
      await loadApiKeys();
    } catch (err) {
      setError(`Failed to save ${provider} ${keyName}`);
      console.error('Error saving API key:', err);
    } finally {
      setSaving(null);
    }
  };

  const deleteApiKey = async (provider: string, keyName: string) => {
    if (!confirm(`Delete ${provider} ${keyName}? This cannot be undone.`)) return;

    try {
      setError(null);
      await apiKeysService.deleteApiKey(provider, keyName);
      await loadApiKeys();
    } catch (err) {
      setError(`Failed to delete ${provider} ${keyName}`);
      console.error('Error deleting API key:', err);
    }
  };

  const validateApiKey = async (provider: string, keyName: string) => {
    const validateKey = `${provider}.${keyName}`;
    try {
      setValidating(validateKey);
      setError(null);
      
      const isValid = await apiKeysService.validateApiKey(provider, keyName);
      
      if (isValid) {
        // Reload to show updated validation status
        await loadApiKeys();
      }
    } catch (err) {
      setError(`Failed to validate ${provider} ${keyName}`);
      console.error('Error validating API key:', err);
    } finally {
      setValidating(null);
    }
  };

  const getStoredKey = (provider: string, keyName: string): StoredApiKey | undefined => {
    return apiKeys.find(key => key.provider === provider && key.key_name === keyName);
  };

  const getValidationIcon = (storedKey?: StoredApiKey) => {
    if (!storedKey) return null;

    switch (storedKey.validation_status) {
      case 'valid':
        return (
          <span title="Valid">
            <Check className="h-4 w-4 text-green-500" />
          </span>
        );
      case 'invalid':
        return (
          <span title="Invalid">
            <X className="h-4 w-4 text-red-500" />
          </span>
        );
      case 'error':
        return (
          <span title={storedKey.validation_error || 'Error'}>
            <AlertCircle className="h-4 w-4 text-yellow-500" />
          </span>
        );
      default:
        return (
          <span title="Pending validation">
            <AlertCircle className="h-4 w-4 text-gray-500" />
          </span>
        );
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <RefreshCw className="h-6 w-6 animate-spin text-gray-400" />
        <span className="ml-2 text-gray-400">Loading API keys...</span>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className="space-y-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Key className="h-5 w-5 text-indigo-400" />
          <h3 className="text-lg font-semibold text-white">API Keys Management</h3>
        </div>
        <button
          onClick={() => {
            console.log('🔄 Manual refresh triggered');
            initializeFormDataWithEnvVars();
            loadApiKeys();
          }}
          className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm flex items-center space-x-1"
        >
          <RefreshCw className="h-4 w-4" />
          <span>Refresh Keys</span>
        </button>
      </div>

      <div className="text-sm text-gray-400 bg-gray-800 p-3 rounded-lg">
        <p className="mb-2">
          Manage your API keys securely. Keys are encrypted before storage and only used for authenticated requests.
        </p>
        <p>
          Environment variables will be used as fallback if no stored keys are found.
        </p>
      </div>

      {/* Debug section - show available environment variables */}
      <div className="bg-blue-900/20 border border-blue-500 text-blue-300 p-3 rounded-lg">
        <details className="text-sm">
          <summary className="cursor-pointer font-medium">🔍 Debug: Available Environment Variables</summary>
          <div className="mt-2 space-y-1 text-xs">
            {Object.entries(API_PROVIDERS).map(([provider, config]) => 
              config.keys.map(key => {
                const envKey = `VITE_${provider.toUpperCase()}_${key.name.toUpperCase()}`;
                const envValue = import.meta.env[envKey];
                return (
                  <div key={`${provider}.${key.name}`} className="flex justify-between">
                    <span>{envKey}:</span>
                    <span className={envValue ? 'text-green-400' : 'text-red-400'}>
                      {envValue ? `${envValue.substring(0, 8)}...` : 'Not found'}
                    </span>
                  </div>
                );
              })
            )}
          </div>
        </details>
      </div>

      {apiKeys.length === 0 && !loading && (
        <div className="bg-blue-900/20 border border-blue-500 text-blue-300 p-4 rounded-lg mb-6">
          <div className="flex items-start space-x-2">
            <Info className="h-5 w-5 mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-medium mb-2">Getting Started with API Keys</p>
              <p className="mb-2">
                No API keys have been stored yet. You can add your API keys below to enhance the trading bot's capabilities:
              </p>
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li><strong>CoinGecko:</strong> Optional - provides higher rate limits for market data</li>
                <li><strong>Alpaca:</strong> Required - for live trading and portfolio management</li>
                <li><strong>Groq:</strong> Required - for AI-powered market insights and analysis</li>
                <li><strong>Other APIs:</strong> Optional - for additional data sources and features</li>
              </ul>
              <p className="mt-2 text-xs">
                Your app is currently running with environment variable fallbacks. Adding stored keys allows you to manage them securely through this interface.
              </p>
            </div>
          </div>
        </div>
      )}

      <ApiConnectionStatus />

      {error && (
        <div className="bg-red-900/20 border border-red-500 text-red-300 p-3 rounded-lg">
          <div className="flex items-center justify-between">
            <span>{error}</span>
            <button
              onClick={() => loadApiKeys()}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-sm"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      <div className="space-y-6">
        {Object.entries(API_PROVIDERS).map(([provider, config]) => (
          <div key={provider} className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-white font-medium">{config.name}</h4>
                <p className="text-sm text-gray-400">{config.description}</p>
              </div>
            </div>

            <div className="space-y-3">
              {config.keys.map(keyConfig => {
                const storedKey = getStoredKey(provider, keyConfig.name);
                const formValue = formData[provider]?.[keyConfig.name]?.value || '';
                const isVisible = formData[provider]?.[keyConfig.name]?.visible || false;
                const saveKey = `${provider}.${keyConfig.name}`;
                const isSaving = saving === saveKey;
                const isValidating = validating === saveKey;

                return (
                  <div key={keyConfig.name} className="space-y-2">
                    <label className="block text-sm font-medium text-gray-300">
                      {keyConfig.label}
                      {keyConfig.required && <span className="text-red-400 ml-1">*</span>}
                    </label>

                    <div className="flex items-center space-x-2">
                      <div className="relative flex-1">
                        <input
                          type={isVisible ? 'text' : 'password'}
                          value={formValue}
                          onChange={(e) => handleInputChange(provider, keyConfig.name, e.target.value)}
                          placeholder={storedKey ? 'Enter new key to update' : 'Enter API key'}
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent pr-10"
                        />
                        <button
                          type="button"
                          onClick={() => toggleVisibility(provider, keyConfig.name)}
                          className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-300"
                        >
                          {isVisible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                        </button>
                      </div>
                      {formValue && !storedKey && (
                        <div className="text-xs text-blue-400 bg-blue-900/20 px-2 py-1 rounded">
                          From .env
                        </div>
                      )}

                      <button
                        onClick={() => saveApiKey(provider, keyConfig.name)}
                        disabled={!formValue || isSaving}
                        className="px-3 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg flex items-center space-x-1"
                      >
                        {isSaving ? (
                          <RefreshCw className="h-4 w-4 animate-spin" />
                        ) : (
                          <Save className="h-4 w-4" />
                        )}
                        <span>{storedKey ? 'Update' : 'Save'}</span>
                      </button>

                      {storedKey && (
                        <>
                          <button
                            onClick={() => validateApiKey(provider, keyConfig.name)}
                            disabled={isValidating}
                            className="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg flex items-center space-x-1"
                          >
                            {isValidating ? (
                              <RefreshCw className="h-4 w-4 animate-spin" />
                            ) : (
                              <RefreshCw className="h-4 w-4" />
                            )}
                            <span>Test</span>
                          </button>

                          <button
                            onClick={() => deleteApiKey(provider, keyConfig.name)}
                            className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg flex items-center space-x-1"
                          >
                            <Trash2 className="h-4 w-4" />
                            <span>Delete</span>
                          </button>
                        </>
                      )}
                    </div>

                    {storedKey && (
                      <div className="flex items-center justify-between text-xs text-gray-400">
                        <div className="flex items-center space-x-2">
                          <span>Status:</span>
                          {getValidationIcon(storedKey)}
                          <span className="capitalize">{storedKey.validation_status}</span>
                        </div>
                        {storedKey.last_validated_at && (
                          <span>
                            Last validated: {new Date(storedKey.last_validated_at).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="bg-yellow-900/20 border border-yellow-500 text-yellow-300 p-3 rounded-lg">
        <div className="flex items-start space-x-2">
          <AlertCircle className="h-5 w-5 mt-0.5 flex-shrink-0" />
          <div className="text-sm">
            <p className="font-medium mb-1">Security Notice</p>
            <ul className="list-disc list-inside space-y-1">
              <li>API keys are encrypted before storage using AES encryption</li>
              <li>Keys are only decrypted when making authenticated API requests</li>
              <li>Never share your API keys or include them in screenshots</li>
              <li>Regularly rotate your keys for enhanced security</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
    </ErrorBoundary>
  );
};