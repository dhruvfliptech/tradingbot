/**
 * Strategy Services Index
 * Exports all available trading strategies for easy importing
 */

export { liquidityHuntingStrategy } from './liquidityHunting';
export { smartMoneyDivergenceStrategy } from './smartMoneyDivergence';
export { volumeProfileAnalysisStrategy } from './volumeProfileAnalysis';
export { microstructureAnalysisStrategy } from './microstructureAnalysis';

export type { LiquidityHuntingSignal, LiquidityAnalysisResult } from './liquidityHunting';
export type { SmartMoneySignal, DivergenceAnalysisResult } from './smartMoneyDivergence';
export type { VolumeProfileSignal, VolumeProfileAnalysisResult } from './volumeProfileAnalysis';
export type { MicrostructureSignal, MicrostructureAnalysisResult } from './microstructureAnalysis';

// Strategy registry for dynamic loading
export const strategies = {
  liquidityHunting: () => import('./liquidityHunting').then(m => m.liquidityHuntingStrategy),
  smartMoneyDivergence: () => import('./smartMoneyDivergence').then(m => m.smartMoneyDivergenceStrategy),
  volumeProfileAnalysis: () => import('./volumeProfileAnalysis').then(m => m.volumeProfileAnalysisStrategy),
  microstructureAnalysis: () => import('./microstructureAnalysis').then(m => m.microstructureAnalysisStrategy),
};

export type StrategyName = keyof typeof strategies;