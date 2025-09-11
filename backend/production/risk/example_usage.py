"""
Example usage of the comprehensive risk management system
Demonstrates integration of all risk components
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Import risk management components
from risk_manager import RiskManager, RiskLimits, Position
from position_sizer import PositionSizer, SizingMethod, SizingConfig, MarketContext
from portfolio_risk import PortfolioRiskAnalyzer
from circuit_breakers import CircuitBreakerSystem, BreakerConfig, KillSwitch
from risk_monitor import RiskMonitor, MonitorConfig
from stress_testing import StressTester


class IntegratedRiskSystem:
    """
    Complete integrated risk management system
    Combines all risk components for production trading
    """
    
    def __init__(self, account_balance: float = 1000000):
        """
        Initialize integrated risk system
        
        Args:
            account_balance: Starting account balance
        """
        self.account_balance = account_balance
        
        # Initialize risk limits (SOW compliant)
        risk_limits = RiskLimits(
            max_drawdown=0.15,  # 15% max drawdown
            daily_loss_limit=0.05,  # 5% daily loss limit
            max_leverage=5.0,  # 5x max leverage
            min_sharpe_ratio=1.5,  # Minimum Sharpe > 1.5
            weekly_return_target=0.03,  # 3% weekly target
            max_weekly_return=0.05  # 5% weekly max
        )
        
        # Initialize components
        self.risk_manager = RiskManager(
            account_balance=account_balance,
            risk_limits=risk_limits,
            enable_circuit_breakers=True
        )
        
        self.position_sizer = PositionSizer(
            account_balance=account_balance,
            config=SizingConfig(
                method=SizingMethod.DYNAMIC,
                base_risk_percent=0.02,
                max_risk_percent=0.05,
                kelly_fraction=0.25
            )
        )
        
        self.circuit_breakers = CircuitBreakerSystem(
            config=BreakerConfig(),
            action_callback=self.handle_breaker_action
        )
        
        self.risk_monitor = RiskMonitor(
            config=MonitorConfig(
                update_frequency=5,
                enable_webhook_alerts=True
            ),
            alert_callback=self.handle_alert
        )
        
        # Kill switch for emergencies
        self.kill_switch = KillSwitch(callback=self.emergency_shutdown)
        
        print(f"Integrated Risk System initialized with ${account_balance:,.2f}")
    
    async def evaluate_trade_opportunity(self, 
                                        symbol: str,
                                        signal_strength: float,
                                        entry_price: float,
                                        stop_loss: float,
                                        take_profit: float) -> Dict[str, Any]:
        """
        Evaluate a trading opportunity through all risk checks
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal strength (0-1)
            entry_price: Proposed entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Complete risk assessment
        """
        print(f"\n{'='*60}")
        print(f"Evaluating trade: {symbol}")
        print(f"Entry: ${entry_price:.2f}, Stop: ${stop_loss:.2f}, Target: ${take_profit:.2f}")
        
        # Step 1: Calculate position size
        stop_distance = abs(entry_price - stop_loss)
        
        market_context = MarketContext(
            volatility=0.02,
            trend_strength=0.7,
            regime="trending",
            correlation=0.3,
            liquidity=0.9,
            confidence=signal_strength,
            win_rate=0.55,
            expectancy=0.8,
            drawdown=self.risk_manager.risk_metrics.current_drawdown
        )
        
        sizing_result = self.position_sizer.calculate_position_size(
            signal_strength=signal_strength,
            stop_distance=stop_distance,
            entry_price=entry_price,
            market_context=market_context
        )
        
        print(f"\nPosition Sizing:")
        print(f"  Shares: {sizing_result['shares']:.0f}")
        print(f"  Position Value: ${sizing_result['position_value']:,.2f}")
        print(f"  Risk Amount: ${sizing_result['risk_amount']:,.2f}")
        print(f"  Risk %: {sizing_result['risk_percent']:.2%}")
        print(f"  Leverage: {sizing_result['leverage']:.2f}x")
        
        # Step 2: Risk manager evaluation
        risk_action, risk_assessment = self.risk_manager.evaluate_trade(
            symbol=symbol,
            position_size=sizing_result['shares'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        print(f"\nRisk Assessment:")
        print(f"  Action: {risk_action.value}")
        print(f"  Risk Level: {risk_assessment.get('risk_level', 'N/A')}")
        print(f"  Risk Score: {risk_assessment.get('risk_score', 0):.2f}")
        
        if risk_assessment.get('warnings'):
            print(f"  Warnings:")
            for warning in risk_assessment['warnings']:
                print(f"    - {warning}")
        
        # Step 3: Check circuit breakers
        metrics = {
            'drawdown': self.risk_manager.risk_metrics.current_drawdown,
            'daily_loss': abs(self.risk_manager.risk_metrics.daily_pnl) if self.risk_manager.risk_metrics.daily_pnl < 0 else 0,
            'volatility_ratio': 1.5,  # Example
            'liquidity_ratio': 0.8  # Example
        }
        
        breaker_actions = await self.circuit_breakers.check_all_breakers(metrics)
        
        if breaker_actions:
            print(f"\nCircuit Breaker Status:")
            for action in breaker_actions:
                print(f"  - {action.value}")
        
        # Final decision
        trade_allowed = (
            risk_action != risk_action.REJECT and
            not self.circuit_breakers.system_halted and
            self.circuit_breakers.trading_allowed
        )
        
        result = {
            'symbol': symbol,
            'trade_allowed': trade_allowed,
            'position_size': sizing_result['shares'] if trade_allowed else 0,
            'risk_assessment': risk_assessment,
            'sizing_result': sizing_result,
            'breaker_status': self.circuit_breakers.get_status()
        }
        
        print(f"\nFinal Decision: {'APPROVED' if trade_allowed else 'REJECTED'}")
        print(f"{'='*60}")
        
        return result
    
    async def handle_breaker_action(self, action: str):
        """Handle circuit breaker actions"""
        print(f"\n🚨 Circuit Breaker Action: {action}")
        
        if action == 'close_all':
            # Close all positions
            positions_closed = self.risk_manager.emergency_stop("CIRCUIT_BREAKER")
            print(f"Closed {len(positions_closed)} positions")
        
        elif action == 'close_losing':
            # Close losing positions
            for symbol, position in self.risk_manager.positions.items():
                if position.current_r_multiple < 0:
                    self.risk_manager.close_position(symbol, position.current_price, "CIRCUIT_BREAKER")
        
        elif action == 'system_halt':
            # Halt all trading
            await self.kill_switch.activate("CIRCUIT_BREAKER_HALT")
    
    async def handle_alert(self, alert):
        """Handle risk alerts"""
        print(f"\n⚠️ Risk Alert [{alert.level.value}]: {alert.message}")
        
        # Take action based on alert level
        if alert.level.value == "EMERGENCY":
            await self.kill_switch.activate(f"EMERGENCY_ALERT_{alert.metric_type.value}")
        
        elif alert.level.value == "CRITICAL":
            # Reduce exposure
            self.circuit_breakers.exposure_multiplier = 0.5
            print("Reducing exposure to 50%")
    
    async def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        print("\n🔴 EMERGENCY SHUTDOWN INITIATED")
        
        # 1. Close all positions
        positions = self.risk_manager.emergency_stop("EMERGENCY_SHUTDOWN")
        print(f"Closed {len(positions)} positions")
        
        # 2. Cancel all pending orders
        print("Cancelling all pending orders...")
        
        # 3. Disable trading
        self.circuit_breakers.system_halted = True
        self.circuit_breakers.trading_allowed = False
        
        # 4. Generate final report
        report = self.risk_manager.get_risk_report()
        print(f"\nFinal Account Balance: ${report['account']['balance']:,.2f}")
        print(f"Total P&L: ${report['account']['total_pnl']:,.2f}")
        
        # 5. Save state
        print("System state saved for recovery")
    
    async def run_daily_stress_test(self):
        """Run daily stress testing routine"""
        print("\n" + "="*60)
        print("DAILY STRESS TEST")
        print("="*60)
        
        # Prepare portfolio data
        portfolio_data = pd.DataFrame([
            {'symbol': symbol, 'value': pos.size * pos.current_price, 'weight': 0}
            for symbol, pos in self.risk_manager.positions.items()
        ])
        
        if not portfolio_data.empty:
            # Calculate weights
            total_value = portfolio_data['value'].sum()
            portfolio_data['weight'] = portfolio_data['value'] / total_value
            
            # Initialize stress tester
            stress_tester = StressTester(portfolio_data)
            
            # Run key scenarios
            scenarios_to_test = [
                "moderate_correction",
                "financial_crisis_2008",
                "covid_crash_2020"
            ]
            
            for scenario_name in scenarios_to_test:
                result = stress_tester.run_scenario_test(scenario_name)
                print(f"\nScenario: {result.scenario_name}")
                print(f"  Portfolio Loss: ${result.portfolio_loss:,.2f}")
                print(f"  Max Drawdown: {result.max_drawdown:.2%}")
                print(f"  Survival Probability: {result.survival_probability:.2%}")
                print(f"  Recovery Days: {result.recovery_days}")
            
            # Monte Carlo simulation
            mc_results = stress_tester.monte_carlo_simulation(n_simulations=1000)
            print(f"\nMonte Carlo Results (1000 simulations):")
            print(f"  Mean Return: {mc_results['mean_return']:.4f}")
            print(f"  Probability of Loss: {mc_results['probability_loss']:.2%}")
            print(f"  95% VaR: ${mc_results['var']['95%']:,.2f}")
            print(f"  95% CVaR: ${mc_results['cvar']['95%']:,.2f}")
            
            # Generate report
            full_report = stress_tester.generate_stress_report()
            print(f"\nOverall Risk Score: {full_report['risk_score']:.1f}/100")
            
            if full_report['recommendations']:
                print("\nRecommendations:")
                for rec in full_report['recommendations'][:5]:
                    print(f"  • {rec}")
        else:
            print("No positions to stress test")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        risk_report = self.risk_manager.get_risk_report()
        breaker_status = self.circuit_breakers.get_status()
        monitor_data = self.risk_monitor.get_dashboard_data()
        
        return {
            'account': risk_report['account'],
            'risk_metrics': risk_report['risk_metrics'],
            'positions': risk_report['positions'],
            'performance': risk_report['performance'],
            'circuit_breakers': breaker_status,
            'monitoring': monitor_data,
            'system_status': {
                'trading_allowed': self.circuit_breakers.trading_allowed,
                'kill_switch_active': self.kill_switch.activated,
                'risk_score': self.risk_monitor.get_risk_score()
            }
        }


async def main():
    """Main demonstration function"""
    print("="*60)
    print("COMPREHENSIVE RISK MANAGEMENT SYSTEM DEMO")
    print("="*60)
    
    # Initialize system
    risk_system = IntegratedRiskSystem(account_balance=1000000)
    
    # Start monitoring
    await risk_system.risk_monitor.start()
    
    # Example 1: Evaluate a trade
    print("\n1. EVALUATING TRADE OPPORTUNITY")
    trade_result = await risk_system.evaluate_trade_opportunity(
        symbol="AAPL",
        signal_strength=0.75,
        entry_price=150.00,
        stop_loss=145.00,
        take_profit=160.00
    )
    
    # Add position if approved
    if trade_result['trade_allowed']:
        position = Position(
            symbol="AAPL",
            size=trade_result['position_size'],
            entry_price=150.00,
            current_price=150.00,
            stop_loss=145.00,
            take_profit=160.00,
            correlation_group="tech",
            sector="Technology",
            entry_time=datetime.now(),
            risk_amount=trade_result['sizing_result']['risk_amount'],
            current_r_multiple=0
        )
        risk_system.risk_manager.add_position(position)
    
    # Example 2: Update position prices
    print("\n2. UPDATING POSITION PRICES")
    update_result = risk_system.risk_manager.update_position("AAPL", 152.00)
    print(f"Position update: {update_result}")
    
    # Example 3: Run stress test
    await risk_system.run_daily_stress_test()
    
    # Example 4: Get dashboard data
    print("\n3. DASHBOARD DATA")
    dashboard = risk_system.get_dashboard_data()
    print(f"Account Balance: ${dashboard['account']['balance']:,.2f}")
    print(f"Current Drawdown: {dashboard['risk_metrics']['current_drawdown']:.2%}")
    print(f"Risk Score: {dashboard['system_status']['risk_score']:.1f}/100")
    
    # Stop monitoring
    await risk_system.risk_monitor.stop()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())