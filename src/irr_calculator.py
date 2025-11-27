"""
IRR Calculator Module
Calculates Internal Rate of Return and NPV for portfolio valuations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import newton

class IRRCalculator:
    """
    Calculate IRR, NPV, and related metrics for credit portfolios
    """
    
    def __init__(self, discount_rate: float = 0.12):
        """
        Initialize IRR calculator
        
        Args:
            discount_rate: Annual discount rate for NPV (default 12%)
        """
        self.discount_rate = discount_rate
        self.monthly_discount_rate = (1 + discount_rate) ** (1/12) - 1
    
    def calculate_portfolio_metrics(self, cashflows: pd.DataFrame, 
                                    purchase_price: float = None,
                                    book_value: float = None) -> Dict:
        """
        Calculate comprehensive valuation metrics
        
        Args:
            cashflows: DataFrame with monthly cash flows
            purchase_price: Portfolio purchase price (optional)
            book_value: Book value of portfolio (used if no purchase price)
            
        Returns:
            Dictionary with NPV, IRR, MOIC, and other metrics
        """
        # Use book value if no purchase price provided
        # If neither is provided, use the beginning balance as a proxy
        if purchase_price is None:
            if book_value:
                purchase_price = book_value
            elif 'beginning_balance' in cashflows.columns:
                purchase_price = cashflows['beginning_balance'].iloc[0]
            else:
                # Fallback: use net cashflows sum as approximation
                purchase_price = max(cashflows['net_cashflow'].sum(), 1.0)
        
        # Extract net cash flows
        net_cashflows = cashflows['net_cashflow'].values
        
        # Calculate NPV
        npv = self.calculate_npv(net_cashflows, purchase_price)
        
        # Calculate IRR
        irr = self.calculate_irr(net_cashflows, purchase_price)
        
        # Calculate total collections and costs
        total_collections = cashflows['collections'].sum()
        total_losses = cashflows['losses'].sum() if 'losses' in cashflows.columns else 0
        
        # MOIC (Multiple on Invested Capital)
        moic = total_collections / purchase_price if purchase_price > 0 else 0
        
        # Payback period
        payback = self.calculate_payback_period(net_cashflows)
        
        # Yield metrics
        total_return = total_collections - purchase_price
        roi = (total_return / purchase_price * 100) if purchase_price > 0 else 0
        
        # Break-even analysis
        breakeven_loss_rate = self._calculate_breakeven_loss_rate(
            purchase_price, total_collections, total_losses
        )
        
        return {
            'purchase_price': purchase_price,
            'npv': npv,
            'irr_annual': irr * 100,  # Convert to percentage
            'irr_monthly': irr / 12 * 100,
            'moic': moic,
            'payback_period_months': payback,
            'total_collections': total_collections,
            'total_losses': total_losses,
            'total_return': total_return,
            'roi_percent': roi,
            'breakeven_loss_rate': breakeven_loss_rate,
            'discount_rate_used': self.discount_rate * 100
        }
    
    def calculate_npv(self, cashflows: np.ndarray, initial_investment: float) -> float:
        """
        Calculate Net Present Value
        
        Args:
            cashflows: Array of monthly cash flows
            initial_investment: Initial cash outflow (purchase price)
            
        Returns:
            NPV
        """
        # Discount each monthly cash flow
        periods = np.arange(1, len(cashflows) + 1)
        discount_factors = (1 + self.monthly_discount_rate) ** (-periods)
        
        pv_cashflows = np.sum(cashflows * discount_factors)
        npv = pv_cashflows - initial_investment
        
        return npv
    
    def calculate_irr(self, cashflows: np.ndarray, initial_investment: float) -> float:
        """
        Calculate Internal Rate of Return (annual)
        
        Args:
            cashflows: Array of monthly cash flows
            initial_investment: Initial cash outflow
            
        Returns:
            IRR as annual rate (e.g., 0.15 = 15%)
        """
        # Create full cash flow array with initial investment
        full_cf = np.concatenate([[-initial_investment], cashflows])
        
        try:
            # Use numpy_financial's IRR calculation (or manual calculation)
            try:
                import numpy_financial as npf
                monthly_irr = npf.irr(full_cf)
            except ImportError:
                # Fallback to manual IRR calculation using Newton's method
                monthly_irr = self._manual_irr(full_cf)
            
            # Convert monthly to annual
            annual_irr = (1 + monthly_irr) ** 12 - 1
            
            # Handle edge cases
            if np.isnan(annual_irr) or np.isinf(annual_irr):
                # Fallback to simple return
                total_return = np.sum(cashflows)
                annual_irr = (total_return / initial_investment - 1) if initial_investment > 0 else 0
                
        except:
            # Fallback calculation
            total_return = np.sum(cashflows)
            simple_return = (total_return / initial_investment - 1) if initial_investment > 0 else 0
            # Annualize assuming average holding period
            years = len(cashflows) / 12
            annual_irr = (1 + simple_return) ** (1 / years) - 1 if years > 0 else simple_return
        
        return annual_irr
    
    def _manual_irr(self, cashflows: np.ndarray, guess: float = 0.1) -> float:
        """
        Manual IRR calculation using Newton's method
        
        Args:
            cashflows: Array including initial investment as first element
            guess: Initial guess for IRR
            
        Returns:
            Monthly IRR
        """
        def npv_func(rate):
            return np.sum(cashflows / (1 + rate) ** np.arange(len(cashflows)))
        
        def npv_derivative(rate):
            return np.sum(-np.arange(len(cashflows)) * cashflows / (1 + rate) ** (np.arange(len(cashflows)) + 1))
        
        rate = guess
        for _ in range(100):  # Max 100 iterations
            try:
                npv_val = npv_func(rate)
                derivative = npv_derivative(rate)
                
                if abs(npv_val) < 1e-6:  # Close enough to zero
                    return rate
                
                if abs(derivative) < 1e-10:  # Avoid division by zero
                    break
                
                rate = rate - npv_val / derivative
                
                if rate < -0.99:  # Prevent extreme negative rates
                    rate = -0.99
                    
            except:
                break
        
        # If Newton's method fails, use simple approximation
        total_return = np.sum(cashflows[1:])  # Exclude initial investment
        if cashflows[0] != 0:
            simple_return = (total_return / abs(cashflows[0]) - 1)
            return simple_return / len(cashflows)
        return 0
    
    def calculate_payback_period(self, cashflows: np.ndarray) -> float:
        """
        Calculate payback period in months
        
        Args:
            cashflows: Array of monthly cash flows
            
        Returns:
            Number of months to recover initial investment
        """
        cumulative = np.cumsum(cashflows)
        
        # Find first month where cumulative is positive
        positive_months = np.where(cumulative > 0)[0]
        
        if len(positive_months) > 0:
            return positive_months[0] + 1
        else:
            return len(cashflows)  # Never paid back
    
    def sensitivity_analysis(self, cashflows: pd.DataFrame, 
                            purchase_price: float,
                            variable: str = 'discount_rate') -> pd.DataFrame:
        """
        Perform sensitivity analysis on key variables
        
        Args:
            cashflows: DataFrame with cash flows
            purchase_price: Portfolio purchase price
            variable: Variable to test ('discount_rate', 'default_rate', etc.)
            
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        if variable == 'discount_rate':
            rates = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
            
            for rate in rates:
                self.discount_rate = rate
                self.monthly_discount_rate = (1 + rate) ** (1/12) - 1
                
                npv = self.calculate_npv(
                    cashflows['net_cashflow'].values, 
                    purchase_price
                )
                
                results.append({
                    'Discount Rate': f"{rate*100:.0f}%",
                    'NPV': npv,
                    'Change from Base': npv - self.calculate_npv(cashflows['net_cashflow'].values, purchase_price)
                })
        
        elif variable == 'default_rate':
            # Vary default rate ±50%
            base_losses = cashflows['losses'].sum() if 'losses' in cashflows.columns else 0
            
            for multiplier in [0.5, 0.75, 1.0, 1.25, 1.5]:
                adjusted_cf = cashflows.copy()
                if 'losses' in adjusted_cf.columns:
                    adjusted_cf['losses'] = adjusted_cf['losses'] * multiplier
                    adjusted_cf['net_cashflow'] = (
                        adjusted_cf['collections'] - adjusted_cf['losses']
                    )
                
                npv = self.calculate_npv(
                    adjusted_cf['net_cashflow'].values,
                    purchase_price
                )
                
                irr = self.calculate_irr(
                    adjusted_cf['net_cashflow'].values,
                    purchase_price
                )
                
                results.append({
                    'Default Rate': f"{multiplier*100:.0f}% of base",
                    'NPV': npv,
                    'IRR': f"{irr*100:.2f}%"
                })
        
        return pd.DataFrame(results)
    
    def scenario_analysis(self, base_cashflows: pd.DataFrame,
                         purchase_price: float) -> Dict:
        """
        Perform scenario analysis (Base/Optimistic/Pessimistic/Stress)
        
        Args:
            base_cashflows: Base case cash flows
            purchase_price: Portfolio purchase price
            
        Returns:
            Dictionary with metrics for each scenario
        """
        scenarios = {}
        
        # Base Case
        scenarios['Base'] = self.calculate_portfolio_metrics(
            base_cashflows, purchase_price
        )
        
        # Optimistic: 20% better collections, 30% lower losses
        optimistic_cf = base_cashflows.copy()
        optimistic_cf['collections'] = optimistic_cf['collections'] * 1.20
        if 'losses' in optimistic_cf.columns:
            optimistic_cf['losses'] = optimistic_cf['losses'] * 0.70
        optimistic_cf['net_cashflow'] = (
            optimistic_cf['collections'] - 
            optimistic_cf.get('losses', 0)
        )
        scenarios['Optimistic'] = self.calculate_portfolio_metrics(
            optimistic_cf, purchase_price
        )
        
        # Pessimistic: 15% lower collections, 40% higher losses
        pessimistic_cf = base_cashflows.copy()
        pessimistic_cf['collections'] = pessimistic_cf['collections'] * 0.85
        if 'losses' in pessimistic_cf.columns:
            pessimistic_cf['losses'] = pessimistic_cf['losses'] * 1.40
        pessimistic_cf['net_cashflow'] = (
            pessimistic_cf['collections'] - 
            pessimistic_cf.get('losses', 0)
        )
        scenarios['Pessimistic'] = self.calculate_portfolio_metrics(
            pessimistic_cf, purchase_price
        )
        
        # Stress: 30% lower collections, 80% higher losses
        stress_cf = base_cashflows.copy()
        stress_cf['collections'] = stress_cf['collections'] * 0.70
        if 'losses' in stress_cf.columns:
            stress_cf['losses'] = stress_cf['losses'] * 1.80
        stress_cf['net_cashflow'] = (
            stress_cf['collections'] - 
            stress_cf.get('losses', 0)
        )
        scenarios['Stress'] = self.calculate_portfolio_metrics(
            stress_cf, purchase_price
        )
        
        return scenarios
    
    def _calculate_breakeven_loss_rate(self, purchase_price: float,
                                       total_collections: float,
                                       total_losses: float) -> float:
        """
        Calculate break-even loss rate
        
        Returns:
            Loss rate where NPV = 0
        """
        if total_collections == 0:
            return 0
        
        # Current loss rate
        current_loss_rate = total_losses / total_collections if total_collections > 0 else 0
        
        # Breakeven is when collections = purchase_price + losses
        # Collections * (1 - breakeven_rate) = purchase_price
        breakeven_rate = 1 - (purchase_price / total_collections) if total_collections > 0 else 0
        
        return max(0, breakeven_rate)
    
    def create_valuation_summary(self, method_a_metrics: Dict, 
                                method_b_metrics: Dict) -> pd.DataFrame:
        """
        Create comparison summary of both methods
        
        Args:
            method_a_metrics: Metrics from Static Pool Analysis
            method_b_metrics: Metrics from Dynamic CF Model
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = {
            'Metric': [
                'NPV',
                'IRR (Annual)',
                'MOIC',
                'Payback Period (months)',
                'Total Collections',
                'Total Losses',
                'ROI %'
            ],
            'Method A (Static Pool)': [
                f"${method_a_metrics['npv']:,.0f}",
                f"{method_a_metrics['irr_annual']:.2f}%",
                f"{method_a_metrics['moic']:.2f}x",
                f"{method_a_metrics['payback_period_months']:.1f}",
                f"${method_a_metrics['total_collections']:,.0f}",
                f"${method_a_metrics['total_losses']:,.0f}",
                f"{method_a_metrics['roi_percent']:.2f}%"
            ],
            'Method B (Dynamic CF)': [
                f"${method_b_metrics['npv']:,.0f}",
                f"{method_b_metrics['irr_annual']:.2f}%",
                f"{method_b_metrics['moic']:.2f}x",
                f"{method_b_metrics['payback_period_months']:.1f}",
                f"${method_b_metrics['total_collections']:,.0f}",
                f"${method_b_metrics['total_losses']:,.0f}",
                f"{method_b_metrics['roi_percent']:.2f}%"
            ],
            'Difference': [
                f"${method_a_metrics['npv'] - method_b_metrics['npv']:,.0f}",
                f"{method_a_metrics['irr_annual'] - method_b_metrics['irr_annual']:.2f}pp",
                f"{method_a_metrics['moic'] - method_b_metrics['moic']:.2f}x",
                f"{method_a_metrics['payback_period_months'] - method_b_metrics['payback_period_months']:.1f}",
                f"${method_a_metrics['total_collections'] - method_b_metrics['total_collections']:,.0f}",
                f"${method_a_metrics['total_losses'] - method_b_metrics['total_losses']:,.0f}",
                f"{method_a_metrics['roi_percent'] - method_b_metrics['roi_percent']:.2f}pp"
            ]
        }
        
        return pd.DataFrame(comparison_data)

