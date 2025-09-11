"""
Gymnasium-Compatible Trading Environment for Reinforcement Learning
Main environment class that integrates all components for crypto trading RL
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import copy

# Import our custom components
from .state_processor import StateProcessor, MarketState
from .portfolio_manager import PortfolioManager, OrderSide, OrderType, PortfolioState
from .market_simulator import MarketSimulator, DataMode, MarketTick, MarketConditions
from rl_config import RLConfig, ActionType, RewardStrategy

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentState:
    """Complete environment state"""
    market_state: MarketState
    portfolio_state: PortfolioState
    market_conditions: MarketConditions
    step_number: int
    episode_number: int
    timestamp: datetime
    done: bool = False
    info: Dict[str, Any] = None


class TradingEnvironment(gym.Env):
    """
    Gymnasium-compatible trading environment for cryptocurrency trading
    
    This environment integrates:
    - Market simulation with realistic dynamics
    - Portfolio management with proper accounting
    - Feature engineering and state processing
    - Configurable reward functions
    - Support for multiple trading pairs
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 config: Optional[RLConfig] = None,
                 mode: str = 'train',
                 symbols: Optional[List[str]] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize trading environment
        
        Args:
            config: RL configuration object
            mode: Environment mode ('train', 'test', 'live')
            symbols: List of trading symbols
            render_mode: Rendering mode for visualization
        """
        super().__init__()
        
        # Configuration
        from rl_config import get_rl_config
        self.config = config or get_rl_config()
        self.mode = mode
        self.symbols = symbols or self.config.env.trading_pairs
        self.primary_symbol = self.config.env.primary_pair
        self.render_mode = render_mode
        
        # Validate primary symbol
        if self.primary_symbol not in self.symbols:
            self.primary_symbol = self.symbols[0]
            logger.warning(f"Primary symbol not in symbols list, using {self.primary_symbol}")
        
        # Initialize components
        self._initialize_components()
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Environment state
        self.current_step = 0
        self.episode_number = 0
        self.max_episode_steps = self.config.env.max_episode_steps
        self.episode_length = self.config.env.episode_length
        
        # Episode tracking
        self.episode_start_time: Optional[datetime] = None
        self.episode_returns: List[float] = []
        self.episode_rewards: List[float] = []
        self.episode_actions: List[int] = []
        
        # Performance metrics
        self.total_episodes = 0
        self.total_steps = 0
        self.best_episode_return = float('-inf')
        self.worst_episode_return = float('inf')
        
        # Data loaded flag
        self.data_loaded = False
        
        logger.info(f"TradingEnvironment initialized in {mode} mode with {len(self.symbols)} symbols")
        logger.info(f"Primary symbol: {self.primary_symbol}")
        logger.info(f"Episode length: {self.episode_length} steps")
        logger.info(f"Action space: {self.action_space}")
        logger.info(f"Observation space: {self.observation_space}")
    
    def _initialize_components(self):
        """Initialize environment components"""
        try:
            # Market simulator
            market_mode = DataMode.HISTORICAL if self.mode in ['train', 'test'] else DataMode.LIVE
            self.market_simulator = MarketSimulator(
                mode=market_mode,
                symbols=self.symbols,
                timeframe=self.config.data.data_frequency,
                enable_market_impact=self.config.env.enable_market_impact,
                volatility_factor=1.0
            )
            
            # Portfolio manager
            self.portfolio_manager = PortfolioManager(
                initial_balance=self.config.env.initial_balance,
                commission_rate=self.config.reward.transaction_cost_rate,
                slippage_rate=self.config.reward.slippage_rate,
                max_position_size=self.config.env.max_position_size
            )
            
            # State processor
            self.state_processor = StateProcessor(
                lookback_window=self.config.observation.lookback_window,
                normalization_method=self.config.observation.normalization_method,
                feature_selection=True
            )
            
            # Reward calculator
            self.reward_calculator = RewardCalculator(
                strategy=self.config.reward.strategy,
                config=self.config.reward
            )
            
            logger.info("Environment components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing environment components: {e}")
            raise
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        try:
            # Action space - discrete actions
            if self.config.action.action_type == 'discrete':
                n_actions = len(self.config.action.discrete_actions)
                self.action_space = spaces.Discrete(n_actions)
                logger.info(f"Discrete action space with {n_actions} actions")
            else:
                # Continuous action space
                low, high = self.config.action.continuous_action_bounds
                self.action_space = spaces.Box(
                    low=low, high=high, shape=(1,), dtype=np.float32
                )
                logger.info(f"Continuous action space: [{low}, {high}]")
            
            # Observation space - normalized features
            # Calculate total feature size
            total_features = (
                len(self.config.observation.price_features) +
                len(self.config.observation.technical_indicators) +
                len(self.config.observation.sentiment_features) +
                len(self.config.observation.portfolio_features) +
                len(self.config.observation.alternative_features)
            )
            
            # For lookback window, we flatten the feature matrix
            obs_size = total_features
            
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_size,),
                dtype=np.float32
            )
            
            logger.info(f"Observation space size: {obs_size}")
            
        except Exception as e:
            logger.error(f"Error setting up spaces: {e}")
            raise
    
    def load_data(self, 
                  start_date: Optional[datetime] = None, 
                  end_date: Optional[datetime] = None) -> bool:
        """
        Load market data for training/testing
        
        Args:
            start_date: Start date for data (uses config default if None)
            end_date: End date for data (uses config default if None)
            
        Returns:
            True if data loaded successfully
        """
        try:
            if start_date is None:
                start_date = datetime.strptime(self.config.data.start_date, '%Y-%m-%d')
            if end_date is None:
                end_date = datetime.strptime(self.config.data.end_date, '%Y-%m-%d')
            
            logger.info(f"Loading data from {start_date} to {end_date}")
            
            # Load market data
            success = self.market_simulator.load_data(start_date, end_date, self.symbols)
            
            if not success:
                logger.error("Failed to load market data")
                return False
            
            # Fit state processor on historical data
            self._fit_state_processor()
            
            self.data_loaded = True
            logger.info("Data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _fit_state_processor(self):
        """Fit state processor on historical data"""
        try:
            # Get market data for fitting
            primary_data = self.market_simulator.market_data.get(self.primary_symbol)
            
            if primary_data is None or primary_data.empty:
                logger.error("No primary market data available for fitting")
                return
            
            # Use a subset of data for fitting (e.g., first 80%)
            train_size = int(len(primary_data) * self.config.data.train_split)
            train_data = primary_data.iloc[:train_size]
            
            logger.info(f"Fitting state processor on {len(train_data)} data points")
            
            # Fit the processor
            self.state_processor.fit(train_data)
            
            logger.info("State processor fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting state processor: {e}")
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to start a new episode
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        try:
            super().reset(seed=seed)
            
            if not self.data_loaded:
                logger.warning("No data loaded, using default synthetic data")
                self._load_default_data()
            
            # Reset episode tracking
            self.current_step = 0
            self.episode_number += 1
            self.episode_start_time = datetime.now()
            self.episode_returns.clear()
            self.episode_rewards.clear()
            self.episode_actions.clear()
            
            # Reset components
            self.portfolio_manager.reset(self.config.env.initial_balance)
            
            # Set random start point for episodes (except first)
            if self.episode_number > 1 and self.mode == 'train':
                available_steps = self.market_simulator.get_available_steps()
                if available_steps > self.episode_length:
                    max_start = available_steps - self.episode_length
                    start_step = self.np_random.integers(0, max_start)
                    self.market_simulator.reset(start_step)
            else:
                self.market_simulator.reset(0)
            
            self.state_processor.reset()
            
            # Get initial state
            initial_state = self._get_current_state()
            observation = self._state_to_observation(initial_state)
            
            info = {
                'episode': self.episode_number,
                'step': self.current_step,
                'portfolio_value': self.portfolio_manager.get_total_equity(),
                'cash_balance': self.portfolio_manager.cash_balance,
                'timestamp': initial_state.timestamp.isoformat() if initial_state.timestamp else None
            }
            
            logger.info(f"Episode {self.episode_number} started")
            return observation.astype(np.float32), info
            
        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            # Return safe defaults
            obs_size = self.observation_space.shape[0]
            return np.zeros(obs_size, dtype=np.float32), {}
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        try:
            if not self.data_loaded:
                logger.error("No data loaded")
                return self._get_default_step_result()
            
            # Convert action if needed
            if isinstance(action, np.ndarray):
                action = action.item() if action.size == 1 else action[0]
            
            # Store action
            self.episode_actions.append(action)
            
            # Execute action
            order_executed = self._execute_action(action)
            
            # Step market simulator
            market_ticks, market_conditions = self.market_simulator.step()
            
            # Update portfolio with new market prices
            if market_ticks:
                prices = {symbol: tick.close for symbol, tick in market_ticks.items()}
                self.portfolio_manager.update_market_prices(prices)
            
            # Get new state
            current_state = self._get_current_state()
            
            # Calculate reward
            reward = self.reward_calculator.calculate_reward(
                current_state, self.portfolio_manager, market_conditions
            )
            
            self.episode_rewards.append(reward)
            
            # Check if episode is done
            self.current_step += 1
            self.total_steps += 1
            
            terminated = self._check_termination()
            truncated = self.current_step >= self.episode_length
            done = terminated or truncated
            
            # Get observation
            observation = self._state_to_observation(current_state)
            
            # Create info dict
            info = self._create_info_dict(current_state, order_executed, reward, done)
            
            # Update episode tracking
            if done:
                self._finalize_episode()
            
            return observation.astype(np.float32), float(reward), terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in environment step: {e}")
            return self._get_default_step_result()
    
    def _execute_action(self, action: int) -> bool:
        """
        Execute trading action
        
        Args:
            action: Action index
            
        Returns:
            True if order was executed
        """
        try:
            if self.config.action.action_type != 'discrete':
                logger.warning("Continuous actions not yet implemented")
                return False
            
            # Get action type
            action_types = list(ActionType)
            if action >= len(action_types):
                logger.warning(f"Invalid action: {action}")
                return False
            
            action_type = action_types[action]
            
            # Handle HOLD action
            if action_type == ActionType.HOLD:
                return True
            
            # Get current price
            current_price = self.market_simulator.get_current_prices().get(self.primary_symbol)
            if not current_price:
                logger.warning(f"No price available for {self.primary_symbol}")
                return False
            
            # Determine order side and size
            if action_type.name.startswith('BUY'):
                order_side = OrderSide.BUY
                size_key = f"BUY_{action_type.name.split('_')[1]}"
            elif action_type.name.startswith('SELL'):
                order_side = OrderSide.SELL
                size_key = f"SELL_{action_type.name.split('_')[1]}"
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
            
            # Get position size percentage
            size_pct = self.config.action.position_sizes.get(size_key, 0.2)
            
            # Calculate order quantity
            if order_side == OrderSide.BUY:
                # Buy based on available cash
                available_cash = self.portfolio_manager.cash_balance * size_pct
                quantity = available_cash / current_price
            else:
                # Sell based on current position
                current_position = self.portfolio_manager.get_position(self.primary_symbol)
                quantity = current_position.quantity * size_pct
            
            # Check minimum trade amount
            if quantity * current_price < self.config.env.min_trade_amount:
                logger.debug(f"Trade amount too small: {quantity * current_price}")
                return False
            
            # Place order
            order = self.portfolio_manager.place_order(
                symbol=self.primary_symbol,
                side=order_side,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            
            if not order:
                logger.debug("Order placement failed")
                return False
            
            # Execute order immediately (market order)
            execution_price = self.market_simulator.get_execution_price(
                self.primary_symbol, order_side == OrderSide.BUY, quantity
            )
            
            executed = self.portfolio_manager.execute_order(order, execution_price)
            
            if executed:
                logger.debug(f"Executed {order_side.value} {quantity:.6f} {self.primary_symbol} at ${execution_price:.2f}")
            
            return executed
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return False
    
    def _get_current_state(self) -> EnvironmentState:
        """Get current environment state"""
        try:
            # Get market data
            current_ticks, market_conditions = self.market_simulator.step()
            self.market_simulator.current_step -= 1  # Don't advance twice
            
            # Prepare data for state processor
            current_data = {
                'timestamp': market_conditions.timestamp,
                'price_data': {},
                'sentiment_data': {},
                'alternative_data': {},
                'portfolio_state': self.portfolio_manager.get_portfolio_state().to_dict()
            }
            
            # Add price data
            if current_ticks and self.primary_symbol in current_ticks:
                tick = current_ticks[self.primary_symbol]
                current_data['price_data'] = {
                    'open': tick.open,
                    'high': tick.high,
                    'low': tick.low,
                    'close': tick.close,
                    'volume': tick.volume
                }
            
            # Process state
            market_state = self.state_processor.transform(current_data)
            portfolio_state = self.portfolio_manager.get_portfolio_state()
            
            return EnvironmentState(
                market_state=market_state,
                portfolio_state=portfolio_state,
                market_conditions=market_conditions,
                step_number=self.current_step,
                episode_number=self.episode_number,
                timestamp=market_conditions.timestamp
            )
            
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            # Return default state
            from .state_processor import MarketState
            from .market_simulator import MarketConditions, MarketRegime
            
            default_market_state = MarketState(
                timestamp=datetime.now(),
                price_data={}, technical_indicators={}, sentiment_data={},
                portfolio_state={}, alternative_data={},
                raw_features=np.array([]), normalized_features=np.zeros(self.observation_space.shape[0])
            )
            
            default_portfolio_state = PortfolioState(
                cash_balance=self.config.env.initial_balance,
                total_equity=self.config.env.initial_balance,
                portfolio_value=self.config.env.initial_balance,
                unrealized_pnl=0.0, realized_pnl=0.0, total_pnl=0.0,
                positions={}, active_orders=[]
            )
            
            default_conditions = MarketConditions(
                regime=MarketRegime.SIDEWAYS, volatility=0.02, trend_strength=0.0,
                volume_profile=1.0, bid_ask_spread=0.001, market_impact_factor=0.001
            )
            
            return EnvironmentState(
                market_state=default_market_state,
                portfolio_state=default_portfolio_state,
                market_conditions=default_conditions,
                step_number=self.current_step,
                episode_number=self.episode_number,
                timestamp=datetime.now()
            )
    
    def _state_to_observation(self, state: EnvironmentState) -> np.ndarray:
        """Convert environment state to observation vector"""
        try:
            # Use the normalized features from market state
            if len(state.market_state.normalized_features) > 0:
                observation = state.market_state.normalized_features
            else:
                # Fallback: create observation from available data
                observation = np.zeros(self.observation_space.shape[0])
            
            # Ensure correct shape
            if len(observation) != self.observation_space.shape[0]:
                logger.warning(f"Observation shape mismatch: {len(observation)} vs {self.observation_space.shape[0]}")
                # Pad or truncate
                padded_obs = np.zeros(self.observation_space.shape[0])
                min_len = min(len(observation), self.observation_space.shape[0])
                padded_obs[:min_len] = observation[:min_len]
                observation = padded_obs
            
            return observation
            
        except Exception as e:
            logger.error(f"Error converting state to observation: {e}")
            return np.zeros(self.observation_space.shape[0])
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate early"""
        try:
            # Check portfolio value
            current_equity = self.portfolio_manager.get_total_equity()
            
            # Terminate if portfolio loses more than 50%
            if current_equity < self.config.env.initial_balance * 0.5:
                logger.info(f"Episode terminated: portfolio value too low (${current_equity:.2f})")
                return True
            
            # Terminate if maximum drawdown exceeded
            if self.portfolio_manager.max_drawdown > 0.3:  # 30% max drawdown
                logger.info(f"Episode terminated: max drawdown exceeded ({self.portfolio_manager.max_drawdown:.2%})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking termination: {e}")
            return False
    
    def _create_info_dict(self, 
                         state: EnvironmentState, 
                         order_executed: bool, 
                         reward: float, 
                         done: bool) -> Dict[str, Any]:
        """Create info dictionary for step result"""
        try:
            portfolio_metrics = self.portfolio_manager.get_performance_metrics()
            
            info = {
                'step': self.current_step,
                'episode': self.episode_number,
                'timestamp': state.timestamp.isoformat() if state.timestamp else None,
                'portfolio_value': state.portfolio_state.total_equity,
                'cash_balance': state.portfolio_state.cash_balance,
                'unrealized_pnl': state.portfolio_state.unrealized_pnl,
                'realized_pnl': state.portfolio_state.realized_pnl,
                'total_pnl': state.portfolio_state.total_pnl,
                'max_drawdown': state.portfolio_state.max_drawdown,
                'order_executed': order_executed,
                'reward': reward,
                'done': done,
                'market_regime': state.market_conditions.regime.value,
                'volatility': state.market_conditions.volatility,
                'current_prices': self.market_simulator.get_current_prices(),
                'performance_metrics': portfolio_metrics
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error creating info dict: {e}")
            return {'error': str(e)}
    
    def _finalize_episode(self):
        """Finalize episode and update tracking"""
        try:
            # Calculate episode return
            final_equity = self.portfolio_manager.get_total_equity()
            episode_return = (final_equity - self.config.env.initial_balance) / self.config.env.initial_balance
            self.episode_returns.append(episode_return)
            
            # Update best/worst returns
            if episode_return > self.best_episode_return:
                self.best_episode_return = episode_return
            if episode_return < self.worst_episode_return:
                self.worst_episode_return = episode_return
            
            self.total_episodes += 1
            
            # Log episode summary
            total_reward = sum(self.episode_rewards)
            avg_reward = total_reward / len(self.episode_rewards) if self.episode_rewards else 0
            
            logger.info(f"Episode {self.episode_number} completed:")
            logger.info(f"  Return: {episode_return:.2%}")
            logger.info(f"  Total reward: {total_reward:.2f}")
            logger.info(f"  Avg reward: {avg_reward:.2f}")
            logger.info(f"  Final portfolio: ${final_equity:,.2f}")
            logger.info(f"  Max drawdown: {self.portfolio_manager.max_drawdown:.2%}")
            
        except Exception as e:
            logger.error(f"Error finalizing episode: {e}")
    
    def _get_default_step_result(self) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Get default step result for error cases"""
        obs_size = self.observation_space.shape[0]
        observation = np.zeros(obs_size, dtype=np.float32)
        return observation, 0.0, True, False, {'error': 'Environment error'}
    
    def _load_default_data(self):
        """Load default synthetic data"""
        try:
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 6, 30)
            success = self.load_data(start_date, end_date)
            if not success:
                logger.error("Failed to load default data")
        except Exception as e:
            logger.error(f"Error loading default data: {e}")
    
    def render(self, mode: str = 'human'):
        """Render environment (placeholder for visualization)"""
        if mode == 'human':
            # Print current state
            current_prices = self.market_simulator.get_current_prices()
            portfolio_value = self.portfolio_manager.get_total_equity()
            
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${portfolio_value:,.2f}")
            print(f"Prices: {current_prices}")
            print("-" * 50)
    
    def close(self):
        """Clean up resources"""
        logger.info("Environment closed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive environment metrics"""
        try:
            portfolio_metrics = self.portfolio_manager.get_performance_metrics()
            
            metrics = {
                'total_episodes': self.total_episodes,
                'total_steps': self.total_steps,
                'current_episode': self.episode_number,
                'current_step': self.current_step,
                'best_episode_return': self.best_episode_return,
                'worst_episode_return': self.worst_episode_return,
                'avg_episode_return': np.mean(self.episode_returns) if self.episode_returns else 0.0,
                'episode_returns_std': np.std(self.episode_returns) if self.episode_returns else 0.0,
                'portfolio_metrics': portfolio_metrics,
                'current_portfolio_value': self.portfolio_manager.get_total_equity(),
                'data_loaded': self.data_loaded
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}


class RewardCalculator:
    """Calculates rewards for the trading environment"""
    
    def __init__(self, strategy: RewardStrategy, config: Any):
        self.strategy = strategy
        self.config = config
        self.previous_equity = None
        self.equity_history: List[float] = []
        
    def calculate_reward(self, 
                        state: EnvironmentState, 
                        portfolio_manager: PortfolioManager,
                        market_conditions: MarketConditions) -> float:
        """Calculate reward based on strategy"""
        try:
            current_equity = portfolio_manager.get_total_equity()
            
            if self.previous_equity is None:
                self.previous_equity = current_equity
                self.equity_history.append(current_equity)
                return 0.0
            
            self.equity_history.append(current_equity)
            
            if self.strategy == RewardStrategy.SIMPLE_RETURN:
                reward = self._simple_return_reward(current_equity)
            elif self.strategy == RewardStrategy.SHARPE_RATIO:
                reward = self._sharpe_ratio_reward()
            elif self.strategy == RewardStrategy.RISK_ADJUSTED:
                reward = self._risk_adjusted_reward(current_equity, portfolio_manager)
            else:  # COMBINED
                reward = self._combined_reward(current_equity, portfolio_manager)
            
            # Apply scaling and clipping
            reward *= self.config.reward_scaling
            reward = np.clip(reward, self.config.reward_clip_min, self.config.reward_clip_max)
            
            self.previous_equity = current_equity
            return float(reward)
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def _simple_return_reward(self, current_equity: float) -> float:
        """Simple return-based reward"""
        return (current_equity - self.previous_equity) / self.previous_equity
    
    def _sharpe_ratio_reward(self) -> float:
        """Sharpe ratio-based reward"""
        if len(self.equity_history) < self.config.sharpe_window:
            return 0.0
        
        recent_equity = self.equity_history[-self.config.sharpe_window:]
        returns = np.diff(recent_equity) / np.array(recent_equity[:-1])
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.config.risk_free_rate / 252
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        
        return sharpe
    
    def _risk_adjusted_reward(self, current_equity: float, portfolio_manager: PortfolioManager) -> float:
        """Risk-adjusted reward combining multiple factors"""
        # Return component
        return_component = (current_equity - self.previous_equity) / self.previous_equity
        
        # Drawdown penalty
        drawdown_penalty = -portfolio_manager.max_drawdown * self.config.max_drawdown_penalty
        
        # Transaction cost penalty
        recent_trades = portfolio_manager.completed_trades[-1:] if portfolio_manager.completed_trades else []
        transaction_penalty = -sum(trade.commission for trade in recent_trades) / current_equity
        
        return return_component + drawdown_penalty + transaction_penalty
    
    def _combined_reward(self, current_equity: float, portfolio_manager: PortfolioManager) -> float:
        """Combined reward using weighted components"""
        weights = self.config.reward_components
        
        # Return component
        return_comp = (current_equity - self.previous_equity) / self.previous_equity
        
        # Sharpe component
        sharpe_comp = self._sharpe_ratio_reward()
        
        # Drawdown penalty
        drawdown_comp = -portfolio_manager.max_drawdown
        
        # Transaction cost penalty
        recent_trades = portfolio_manager.completed_trades[-1:] if portfolio_manager.completed_trades else []
        transaction_comp = -sum(trade.commission for trade in recent_trades) / current_equity
        
        # Weighted combination
        total_reward = (
            weights.get('return', 0.4) * return_comp +
            weights.get('sharpe', 0.3) * sharpe_comp +
            weights.get('drawdown_penalty', -0.2) * drawdown_comp +
            weights.get('transaction_cost_penalty', -0.1) * transaction_comp
        )
        
        return total_reward
    
    def reset(self):
        """Reset reward calculator for new episode"""
        self.previous_equity = None
        self.equity_history.clear()


if __name__ == "__main__":
    # Example usage
    from rl_config import get_rl_config
    
    config = get_rl_config()
    env = TradingEnvironment(config=config, mode='train')
    
    # Load data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    
    if env.load_data(start_date, end_date):
        print("Environment ready for training")
        
        # Test environment
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i}: Action={action}, Reward={reward:.4f}, Portfolio=${info.get('portfolio_value', 0):,.2f}")
            
            if terminated or truncated:
                print("Episode ended")
                break
        
        env.close()
    else:
        print("Failed to load data")