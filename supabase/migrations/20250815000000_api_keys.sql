-- Create api_keys table for storing encrypted API keys
CREATE TABLE IF NOT EXISTS public.api_keys (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    key_name VARCHAR(100) NOT NULL,
    encrypted_value TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    last_validated_at TIMESTAMP WITH TIME ZONE,
    validation_status VARCHAR(20) DEFAULT 'pending', -- pending, valid, invalid, error
    validation_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique combination of user_id, provider, and key_name
    CONSTRAINT unique_user_provider_key UNIQUE (user_id, provider, key_name)
);

-- Create index for fast lookups
CREATE INDEX IF NOT EXISTS idx_api_keys_user_provider ON public.api_keys(user_id, provider);

-- Enable RLS (Row Level Security)
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;

-- Create policy for users to only see their own API keys
CREATE POLICY "Users can view their own API keys" ON public.api_keys
    FOR SELECT USING (auth.uid() = user_id);

-- Create policy for users to insert their own API keys
CREATE POLICY "Users can insert their own API keys" ON public.api_keys
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Create policy for users to update their own API keys
CREATE POLICY "Users can update their own API keys" ON public.api_keys
    FOR UPDATE USING (auth.uid() = user_id);

-- Create policy for users to delete their own API keys
CREATE POLICY "Users can delete their own API keys" ON public.api_keys
    FOR DELETE USING (auth.uid() = user_id);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_api_keys_updated_at 
    BEFORE UPDATE ON public.api_keys 
    FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

-- Insert some example providers (optional, for reference)
COMMENT ON TABLE public.api_keys IS 'Stores encrypted API keys for various providers like CoinGecko, WhaleAlert, Etherscan, etc.';
COMMENT ON COLUMN public.api_keys.provider IS 'API provider name (e.g., coingecko, whalealert, etherscan, bitquery, covalent, coinglass, alpaca, groq, binance)';
COMMENT ON COLUMN public.api_keys.key_name IS 'Key identifier (e.g., api_key, secret_key, access_token)';
COMMENT ON COLUMN public.api_keys.encrypted_value IS 'Encrypted API key value using AES encryption';
COMMENT ON COLUMN public.api_keys.validation_status IS 'Status of last validation attempt: pending, valid, invalid, error';