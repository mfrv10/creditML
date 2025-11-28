"""
Portfolio Financial Metrics Analyzer
Calculates comprehensive financial ratios, risk metrics, and valuations
"""

import pandas as pd
import numpy as np
from typing import Dict

class PortfolioAnalyzer:
    """
    Comprehensive financial analysis of credit portfolios
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with scored portfolio
        
        Args:
            df: DataFrame with risk scores already calculated
        """
        self.df = df
        self._prepare_calculations()
    
    def _prepare_calculations(self):
        """Pre-calculate common values"""
        # Ensure we have required columns
        if 'outstanding_balance' not in self.df.columns:
            if 'bill_amount_1' in self.df.columns:
                self.df['outstanding_balance'] = self.df['bill_amount_1']
            else:
                self.df['outstanding_balance'] = self.df['credit_limit'] * 0.3
        
        # Calculate utilization if not present
        if 'utilization_rate' not in self.df.columns:
            self.df['utilization_rate'] = (
                self.df['outstanding_balance'] / 
                (self.df['credit_limit'] + 1)
            ).clip(0, 2)
    
    def calculate_all_metrics(self) -> Dict:
        """
        Calculate all financial metrics for the portfolio
        
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Exposure metrics
        metrics['total_exposure'] = self.df['credit_limit'].sum()
        metrics['avg_credit_limit'] = self.df['credit_limit'].mean()
        metrics['total_outstanding'] = self.df['outstanding_balance'].sum()
        metrics['median_balance'] = self.df['outstanding_balance'].median()
        
        # Utilization metrics
        metrics['avg_utilization'] = self.df['utilization_rate'].mean()
        metrics['high_utilization_pct'] = (self.df['utilization_rate'] > 0.8).sum() / len(self.df)
        
        # Payment metrics
        pay_amt_cols = [c for c in self.df.columns if 'payment_amount' in c.lower()]
        bill_cols = [c for c in self.df.columns if 'bill_amount' in c.lower()]
        
        if pay_amt_cols and bill_cols:
            total_payments = self.df[pay_amt_cols].sum(axis=1).sum()
            total_bills = self.df[bill_cols].sum(axis=1).sum()
            metrics['payment_rate'] = total_payments / (total_bills + 1)
        else:
            metrics['payment_rate'] = 0.7
        
        # Risk metrics
        metrics['avg_default_probability'] = self.df['default_probability'].mean()
        metrics['high_risk_count'] = (self.df['risk_category'] == 'High Risk').sum()
        metrics['high_risk_pct'] = metrics['high_risk_count'] / len(self.df)
        
        # Expected Loss calculation
        metrics['expected_loss'] = (
            self.df['outstanding_balance'] * 
            self.df['default_probability']
        ).sum()
        
        # Value at Risk (VaR) - 95th percentile
        sorted_losses = (
            self.df['outstanding_balance'] * 
            self.df['default_probability']
        ).sort_values(ascending=False)
        var_95_idx = int(len(sorted_losses) * 0.05)
        metrics['var_95'] = sorted_losses.iloc[:var_95_idx].sum()
        
        # Loss Given Default (assume 45% recovery rate)
        metrics['lgd'] = 0.55
        
        # Exposure at Default
        metrics['ead'] = metrics['total_outstanding']
        
        # Risk-weighted assets (simplified Basel approach)
        risk_weights = self.df['default_probability'].clip(0.02, 1.0)
        metrics['rwa'] = (self.df['outstanding_balance'] * risk_weights).sum()
        
        # Detailed ratios
        metrics['ratios'] = self._calculate_detailed_ratios()
        
        return metrics
    
    def _calculate_detailed_ratios(self) -> Dict:
        """Calculate detailed financial ratios"""
        ratios = {}
        
        # Portfolio concentration metrics
        ratios['Top 10 Concentration'] = (
            self.df.nlargest(10, 'outstanding_balance')['outstanding_balance'].sum() /
            self.df['outstanding_balance'].sum()
        )
        
        ratios['HHI (Herfindahl Index)'] = (
            (self.df['outstanding_balance'] / self.df['outstanding_balance'].sum()) ** 2
        ).sum()
        
        # Credit quality metrics
        ratios['Avg Credit Score'] = self.df['risk_score'].mean()
        ratios['Score Std Dev'] = self.df['risk_score'].std()
        
        # Utilization metrics
        ratios['Weighted Avg Utilization'] = (
            (self.df['utilization_rate'] * self.df['credit_limit']).sum() /
            self.df['credit_limit'].sum()
        )
        
        ratios['Over-limit %'] = (self.df['utilization_rate'] > 1.0).sum() / len(self.df)
        
        # Delinquency metrics (if available) - only use numeric payment history columns
        pay_cols = []
        for c in self.df.columns:
            col_lower = c.lower()
            # Match payment history columns (pay_0, pay_1, pay_2, etc.) but not payment_status
            if (col_lower.startswith('pay_') and col_lower != 'pay_status' and 
                'payment_status' not in col_lower and
                pd.api.types.is_numeric_dtype(self.df[c])):
                pay_cols.append(c)
        
        # Also check for payment_history columns
        if not pay_cols:
            pay_cols = [c for c in self.df.columns if 'payment_history' in c.lower() and pd.api.types.is_numeric_dtype(self.df[c])]
        
        if pay_cols:
            # Use the first numeric payment column
            pay_col = pay_cols[0]
            if pd.api.types.is_numeric_dtype(self.df[pay_col]):
                ratios['30+ Days Past Due %'] = (
                    (self.df[pay_col] >= 1).sum() / len(self.df)
                )
                ratios['60+ Days Past Due %'] = (
                    (self.df[pay_col] >= 2).sum() / len(self.df)
                )
                ratios['90+ Days Past Due %'] = (
                    (self.df[pay_col] >= 3).sum() / len(self.df)
                )
        
        # Age distribution
        if 'age' in self.df.columns:
            ratios['Avg Account Age'] = self.df['age'].mean()
            ratios['Young Customer %'] = (self.df['age'] < 30).sum() / len(self.df)
        
        # Demographic breakdowns (if available)
        if 'education' in self.df.columns:
            ratios['Graduate School %'] = (self.df['education'] == 1).sum() / len(self.df)
            ratios['University %'] = (self.df['education'] == 2).sum() / len(self.df)
        
        # Loss coverage
        if 'expected_loss' in self.__dict__:
            ratios['Expected Loss Rate'] = (
                self.df['outstanding_balance'] * self.df['default_probability']
            ).sum() / self.df['outstanding_balance'].sum()
        
        return ratios
    
    def calculate_portfolio_value(self) -> Dict:
        """
        Calculate portfolio valuation metrics including cash flow-based methods
        
        Returns:
            Dictionary with valuation components
        """
        valuation = {}
        
        # Book value (total outstanding)
        valuation['book_value'] = self.df['outstanding_balance'].sum()
        
        # Expected loss
        valuation['expected_loss'] = (
            self.df['outstanding_balance'] * 
            self.df['default_probability'] * 
            0.55  # LGD assumption
        ).sum()
        
        # Risk-adjusted value
        valuation['risk_adjusted_value'] = (
            valuation['book_value'] - valuation['expected_loss']
        )
        
        # Expected revenue (assume 18% APR on outstanding balances)
        avg_interest_rate = 0.18
        valuation['expected_revenue'] = (
            self.df['outstanding_balance'] * 
            avg_interest_rate *
            (1 - self.df['default_probability'])  # Only from non-defaulters
        ).sum()
        
        # Operating costs (assume 5% of book value)
        valuation['operating_costs'] = valuation['book_value'] * 0.05
        
        # Net portfolio value
        valuation['net_value'] = (
            valuation['risk_adjusted_value'] + 
            valuation['expected_revenue'] - 
            valuation['operating_costs']
        )
        
        # Return on Assets
        valuation['roa'] = (
            valuation['expected_revenue'] / 
            (valuation['book_value'] + 1)
        )
        
        # Risk-adjusted return
        valuation['raroc'] = (
            (valuation['expected_revenue'] - valuation['expected_loss']) /
            (valuation['book_value'] * 0.1 + 1)  # Assume 10% capital requirement
        )
        
        # Break-even default rate
        valuation['breakeven_default_rate'] = (
            valuation['expected_revenue'] / 
            (valuation['book_value'] * 0.55 + 1)  # Divide by LGD
        )
        
        # Add cash flow-based valuations
        try:
            from src.cash_flow_projector import CashFlowProjector
            from src.irr_calculator import IRRCalculator
            
            # Project cash flows using both methods
            cf_projector = CashFlowProjector(self.df, projection_months=24)
            cf_results = cf_projector.project_both_methods()
            
            # Calculate IRR and NPV for both methods
            irr_calc = IRRCalculator(discount_rate=0.12)
            
            # Method A metrics
            method_a_cf = cf_results['method_a']['monthly_cashflows']
            method_a_metrics = irr_calc.calculate_portfolio_metrics(
                method_a_cf,
                book_value=valuation['book_value']
            )
            
            # Method B metrics
            method_b_cf = cf_results['method_b']['monthly_cashflows']
            method_b_metrics = irr_calc.calculate_portfolio_metrics(
                method_b_cf,
                book_value=valuation['book_value']
            )
            
            # Add to valuation dictionary
            valuation['cash_flow_analysis'] = {
                'method_a': method_a_metrics,
                'method_b': method_b_metrics,
                'comparison': cf_results['comparison'],
                'method_a_cashflows': cf_results['method_a'],
                'method_b_cashflows': cf_results['method_b']
            }
            
        except Exception as e:
            # If cash flow analysis fails, continue with basic valuation
            print(f"Cash flow analysis skipped: {str(e)}")
        
        return valuation
    
    def segment_portfolio(self) -> pd.DataFrame:
        """
        Segment portfolio by risk and characteristics
        
        Returns:
            Summary DataFrame by segment
        """
        segments = []
        
        # Segment by risk category
        for risk_cat in ['Low Risk', 'Medium Risk', 'High Risk']:
            seg_df = self.df[self.df['risk_category'] == risk_cat]
            if len(seg_df) > 0:
                segments.append({
                    'Segment': risk_cat,
                    'Count': len(seg_df),
                    'Avg Balance': seg_df['outstanding_balance'].mean(),
                    'Total Exposure': seg_df['credit_limit'].sum(),
                    'Avg Default Prob': seg_df['default_probability'].mean(),
                    'Expected Loss': (seg_df['outstanding_balance'] * seg_df['default_probability']).sum()
                })
        
        return pd.DataFrame(segments)
    
    def identify_high_risk_accounts(self, top_n: int = 20) -> pd.DataFrame:
        """
        Identify top high-risk accounts
        
        Args:
            top_n: Number of top accounts to return
            
        Returns:
            DataFrame with high-risk account details
        """
        high_risk = self.df.nlargest(top_n, 'default_probability').copy()
        
        # Calculate potential loss for each account
        high_risk['potential_loss'] = (
            high_risk['outstanding_balance'] * 
            high_risk['default_probability'] * 
            0.55
        )
        
        return high_risk
