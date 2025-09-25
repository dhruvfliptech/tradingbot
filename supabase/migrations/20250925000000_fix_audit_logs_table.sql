-- Migration: Fix audit_logs table
-- Description: Ensure audit_logs table exists and has proper RLS policies

-- Check if audit_logs table exists, if not create it
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = 'audit_logs'
    ) THEN
        CREATE TABLE public.audit_logs (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,

            -- Event Details
            event_type VARCHAR(50) NOT NULL,
            event_category VARCHAR(50),

            -- Decision Context
            symbol VARCHAR(20),
            action VARCHAR(20),
            confidence_score DECIMAL(5, 2),

            -- AI Reasoning (JSONB for flexibility)
            ai_reasoning JSONB,

            -- User Context
            old_value JSONB,
            new_value JSONB,
            user_reason TEXT,

            -- System Context
            market_conditions JSONB,
            portfolio_state JSONB,

            -- Meta
            created_at TIMESTAMPTZ DEFAULT NOW(),
            session_id VARCHAR(100),
            ip_address INET,

            -- Indexing
            CONSTRAINT idx_audit_logs_user_created UNIQUE (user_id, created_at)
        );

        -- Create indexes
        CREATE INDEX idx_audit_logs_user_created ON public.audit_logs(user_id, created_at DESC);
        CREATE INDEX idx_audit_logs_event_type ON public.audit_logs(event_type);
        CREATE INDEX idx_audit_logs_symbol ON public.audit_logs(symbol);

        -- Enable RLS
        ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

        -- Create policies
        CREATE POLICY "Users can view own audit logs" ON public.audit_logs
            FOR SELECT USING (auth.uid() = user_id);

        CREATE POLICY "Users can insert own audit logs" ON public.audit_logs
            FOR INSERT WITH CHECK (auth.uid() = user_id);

        CREATE POLICY "Users can update own audit logs" ON public.audit_logs
            FOR UPDATE USING (auth.uid() = user_id);

        CREATE POLICY "Users can delete own audit logs" ON public.audit_logs
            FOR DELETE USING (auth.uid() = user_id);

        RAISE NOTICE 'Created audit_logs table with RLS policies';
    ELSE
        RAISE NOTICE 'audit_logs table already exists';
    END IF;
END
$$;

-- Update updated_at trigger if it doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add comment to table
COMMENT ON TABLE public.audit_logs IS 'Audit trail for all AI decisions and user interventions';
