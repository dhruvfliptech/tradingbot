import 'dotenv/config';
import { TradingAgentCore } from '../../core/agent/TradingAgentCore';
import { SupabasePersistenceAdapter } from '../adapters/SupabasePersistenceAdapter';
import { CoinGeckoMarketDataAdapter } from '../adapters/CoinGeckoMarketDataAdapter';
import { NoopBrokerAdapter } from '../adapters/NoopBrokerAdapter';
import { AgentContext, AgentEventEmitter, SettingsProvider } from '../../core/adapters';

const {
  SUPABASE_URL,
  SUPABASE_SERVICE_KEY,
  TRADING_USER_ID,
  COINGECKO_API_KEY,
  AGENT_INTERVAL_MS,
} = process.env;

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  throw new Error('SUPABASE_URL and SUPABASE_SERVICE_KEY must be set');
}

if (!TRADING_USER_ID) {
  throw new Error('TRADING_USER_ID must be specified (the bot user id)');
}

const persistence = new SupabasePersistenceAdapter({
  url: SUPABASE_URL,
  serviceRoleKey: SUPABASE_SERVICE_KEY,
  userId: TRADING_USER_ID,
});

const marketData = new CoinGeckoMarketDataAdapter({
  apiKey: COINGECKO_API_KEY,
});

const broker = new NoopBrokerAdapter();

const settingsProvider: SettingsProvider = {
  async loadSettings() {
    const state = await persistence.loadState<Record<string, any>>('agentSettings');
    return state ?? {
      watchlist: ['bitcoin', 'ethereum'],
      buyThresholdPct: 1.2,
      sellThresholdPct: -1.5,
      minConfidence: 70,
      orderQuantity: 0.0,
    };
  },
};

const eventEmitter: AgentEventEmitter = {
  emit(event, payload) {
    console.log(`[AgentEvent] ${event}`, payload);
  },
};

const context: AgentContext = {
  broker,
  persistence,
  marketData,
  settings: settingsProvider,
  events: eventEmitter,
  logger: console,
};

async function main() {
  const agent = new TradingAgentCore(context);
  agent.start();

  const intervalMs = Number(AGENT_INTERVAL_MS ?? 60_000);

  const run = async () => {
    try {
      await agent.runCycle();
    } catch (error) {
      console.error('[agent-runner] cycle failed', error);
      await persistence.appendAuditLog({
        event_type: 'system_alert',
        event_category: 'system',
        user_reason: 'Agent cycle failure',
        new_value: { error: (error as Error).message },
      });
    }
  };

  await run();
  setInterval(run, intervalMs);
}

main().catch((error) => {
  console.error('[agent-runner] fatal error', error);
  process.exit(1);
});
