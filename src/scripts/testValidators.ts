/**
 * Test script for the validator system
 * Run this to verify all validators are working correctly
 */

import { ValidatorSystem } from '../services/validatorSystem';
import { TechnicalIndicators, PriceData } from '../services/technicalIndicators';
import { CryptoData, TradingSignal } from '../types/trading';

// Generate sample price data
function generateSamplePriceData(trend: 'up' | 'down' | 'sideways' = 'up'): PriceData[] {
  const data: PriceData[] = [];
  let basePrice = 50000;
  
  for (let i = 0; i < 100; i++) {
    let change = 0;
    if (trend === 'up') {
      change = Math.random() * 1000 - 200; // Mostly positive
    } else if (trend === 'down') {
      change = Math.random() * -1000 + 200; // Mostly negative
    } else {
      change = Math.random() * 1000 - 500; // Random
    }
    
    basePrice += change;
    const high = basePrice + Math.random() * 500;
    const low = basePrice - Math.random() * 500;
    
    data.push({
      timestamp: Date.now() - (100 - i) * 3600000,
      open: basePrice - change / 2,
      high,
      low,
      close: basePrice,
      volume: 1000000 + Math.random() * 500000
    });
  }
  
  return data;
}

async function testValidatorSystem() {
  console.log('🧪 Testing Validator System...\n');
  
  const validator = new ValidatorSystem();
  
  // Test 1: Bullish scenario
  console.log('📈 Test 1: Bullish Market Scenario');
  const bullishPriceData = generateSamplePriceData('up');
  const bullishCrypto: CryptoData = {
    id: 'bitcoin',
    symbol: 'bitcoin',
    name: 'Bitcoin',
    image: '',
    price: bullishPriceData[bullishPriceData.length - 1].close,
    change: 2500,
    changePercent: 5.2,
    volume: 15000000000,
    volume24h: 15000000000,
    high: Math.max(...bullishPriceData.slice(-24).map(p => p.high)),
    low: Math.min(...bullishPriceData.slice(-24).map(p => p.low)),
    high24h: Math.max(...bullishPriceData.slice(-24).map(p => p.high)),
    low24h: Math.min(...bullishPriceData.slice(-24).map(p => p.low)),
    market_cap: 1000000000000,
    market_cap_rank: 1,
    price_change_24h: 2500,
    price_change_percentage_24h: 5.2,
    circulating_supply: 19000000,
    total_supply: 21000000,
    max_supply: 21000000,
    ath: 69000,
    ath_change_percentage: -20,
    last_updated: new Date().toISOString()
  };
  
  const bullishSignal: TradingSignal = {
    symbol: 'bitcoin',
    action: 'BUY',
    confidence: 75,
    price: bullishCrypto.price,
    timestamp: new Date().toISOString()
  };
  
  const bullishResult = await validator.validate(
    bullishSignal,
    bullishPriceData,
    bullishCrypto,
    {
      portfolioValue: 50000,
      existingPositions: 3,
      fearGreedIndex: 35, // Fear = good for buying
      settings: { volatilityTolerance: 'medium' }
    }
  );
  
  console.log('Result:', {
    passed: bullishResult.passed,
    finalScore: bullishResult.finalScore.toFixed(2),
    confidence: bullishResult.confidence.toFixed(2),
    recommendation: bullishResult.recommendation
  });
  console.log('Validators:');
  Object.entries(bullishResult.validators).forEach(([name, result]) => {
    console.log(`  ${result.passed ? '✅' : '❌'} ${name}: ${result.score.toFixed(0)}/100 - ${result.reason}`);
  });
  
  // Test 2: Bearish scenario
  console.log('\n📉 Test 2: Bearish Market Scenario');
  const bearishPriceData = generateSamplePriceData('down');
  const bearishCrypto: CryptoData = {
    ...bullishCrypto,
    symbol: 'ethereum',
    name: 'Ethereum',
    price: bearishPriceData[bearishPriceData.length - 1].close,
    change: -150,
    changePercent: -4.5,
    price_change_24h: -150,
    price_change_percentage_24h: -4.5
  };
  
  const bearishSignal: TradingSignal = {
    symbol: 'ethereum',
    action: 'SELL',
    confidence: 70,
    price: bearishCrypto.price,
    timestamp: new Date().toISOString()
  };
  
  const bearishResult = await validator.validate(
    bearishSignal,
    bearishPriceData,
    bearishCrypto,
    {
      portfolioValue: 50000,
      existingPositions: 5,
      fearGreedIndex: 75, // Greed = good for selling
      settings: { volatilityTolerance: 'high' }
    }
  );
  
  console.log('Result:', {
    passed: bearishResult.passed,
    finalScore: bearishResult.finalScore.toFixed(2),
    confidence: bearishResult.confidence.toFixed(2),
    recommendation: bearishResult.recommendation
  });
  console.log('Validators:');
  Object.entries(bearishResult.validators).forEach(([name, result]) => {
    console.log(`  ${result.passed ? '✅' : '❌'} ${name}: ${result.score.toFixed(0)}/100 - ${result.reason}`);
  });
  
  // Test 3: Technical Indicators
  console.log('\n📊 Test 3: Technical Indicators');
  const prices = bullishPriceData.map(p => p.close);
  
  const rsi = TechnicalIndicators.RSI(prices);
  const macd = TechnicalIndicators.MACD(prices);
  const bb = TechnicalIndicators.BollingerBands(prices);
  const composite = TechnicalIndicators.getCompositeScore(bullishPriceData);
  
  console.log('Technical Analysis:');
  console.log(`  RSI: ${rsi.value.toFixed(2)} - Signal: ${rsi.signal} (Strength: ${rsi.strength?.toFixed(0)})`);
  console.log(`  MACD: ${macd.macd.toFixed(2)} - Trend: ${macd.trend}`);
  console.log(`  Bollinger Bands: Upper=${bb.upper.toFixed(2)}, Middle=${bb.middle.toFixed(2)}, Lower=${bb.lower.toFixed(2)}`);
  console.log(`  Composite Score: ${composite.score.toFixed(2)} - Signal: ${composite.signal}`);
  
  console.log('\n✅ Validator System Test Complete!');
}

// Run the test
testValidatorSystem().catch(console.error);