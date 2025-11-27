"""
Cash Flow Projection Module
Implements both Static Pool Analysis and Dynamic Cash Flow methods
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime
from dateutil.relativedelta import relativedelta

class CashFlowProjector:
    """
    Projects future cash flows for credit portfolios using two methods:
    - Method A: Static Pool Analysis (for revolving credit like credit cards)
    - Method B: Dynamic Cash Flow Model (for term loans)
    """
    
    def __init__(self, df: pd.DataFrame, projection_months: int = 24):
        """
        Initialize cash flow projector
        
        Args:
            df: Portfolio DataFrame with risk scores
            projection_months: Number of months to project (default 24)
        """
        self.df = df
        self.projection_months = projection_months
        self.start_date = datetime.now()
        
    def project_both_methods(self) -> Dict:
        """
        Project cash flows using both methods and compare
        
        Returns:
            Dictionary with results from both methods and comparison
        """
        # Method A: Static Pool Analysis
        method_a_results = self.static_pool_analysis()
        
        # Method B: Dynamic Cash Flow Model
        method_b_results = self.dynamic_cash_flow_model()
        
        # Comparison
        comparison = self._compare_methods(method_a_results, method_b_results)
        
        return {
            'method_a': method_a_results,
            'method_b': method_b_results,
            'comparison': comparison
        }
    
    def static_pool_analysis(self) -> Dict:
        """
        METHOD A: Static Pool Analysis
        Standard approach for revolving credit (credit cards)
        
        Segments portfolio by risk cohorts and projects aggregate behavior
        """
        results = {
            'method': 'Static Pool Analysis (Revolving Credit)',
            'description': 'Projects aggregate cash flows by risk cohort without individual loan tracking'
        }
        
        # Segment by risk category
        cohorts = self._create_risk_cohorts()
        
        # Project cash flows by cohort
        all_cashflows = []
        cohort_summaries = []
        
        for cohort_name, cohort_df in cohorts.items():
            cf = self._project_cohort_cashflows(cohort_name, cohort_df)
            all_cashflows.append(cf)
            
            cohort_summaries.append({
                'Cohort': cohort_name,
                'Accounts': len(cohort_df),
                'Outstanding': cohort_df['outstanding_balance'].sum(),
                'Avg Default Prob': cohort_df['default_probability'].mean(),
                'Total Projected Collections': cf['collections'].sum(),
                'Total Projected Losses': cf['losses'].sum()
            })
        
        # Aggregate all cohorts
        total_cf = self._aggregate_cashflows(all_cashflows)
        
        results['monthly_cashflows'] = total_cf
        results['cohort_summary'] = pd.DataFrame(cohort_summaries)
        results['total_collections'] = total_cf['net_cashflow'].sum()
        results['total_losses'] = total_cf['losses'].sum()
        
        return results
    
    def dynamic_cash_flow_model(self) -> Dict:
        """
        METHOD B: Dynamic Cash Flow Model
        Detailed approach for term loans with individual loan tracking
        
        Models each loan's specific amortization and default timing
        """
        results = {
            'method': 'Dynamic Cash Flow Model (Term Loans)',
            'description': 'Projects individual loan cash flows with amortization schedules'
        }
        
        # For each account, create individual cash flow projection
        individual_cfs = []
        
        # Sample 1000 loans for detailed modeling (or all if less than 1000)
        sample_size = min(1000, len(self.df))
        sample_df = self.df.sample(n=sample_size, random_state=42) if len(self.df) > 1000 else self.df
        
        for idx, loan in sample_df.iterrows():
            loan_cf = self._project_individual_loan(loan)
            individual_cfs.append(loan_cf)
        
        # Aggregate individual projections
        total_cf = self._aggregate_individual_cashflows(individual_cfs, sample_size, len(self.df))
        
        # Create loan-level summary
        loan_summary = self._create_loan_summary(sample_df, individual_cfs)
        
        results['monthly_cashflows'] = total_cf
        results['loan_summary'] = loan_summary
        results['total_collections'] = total_cf['net_cashflow'].sum()
        results['total_losses'] = total_cf['losses'].sum()
        results['sample_size'] = sample_size
        results['total_portfolio_size'] = len(self.df)
        
        return results
    
    def _create_risk_cohorts(self) -> Dict[str, pd.DataFrame]:
        """Create risk-based cohorts for static pool analysis"""
        cohorts = {}
        
        for risk_cat in ['Low Risk', 'Medium Risk', 'High Risk']:
            cohort_df = self.df[self.df['risk_category'] == risk_cat].copy()
            if len(cohort_df) > 0:
                cohorts[risk_cat] = cohort_df
        
        return cohorts
    
    def _project_cohort_cashflows(self, cohort_name: str, cohort_df: pd.DataFrame) -> pd.DataFrame:
        """
        Project cash flows for a risk cohort (Static Pool method)
        """
        months = range(1, self.projection_months + 1)
        
        # Cohort characteristics
        total_outstanding = cohort_df['outstanding_balance'].sum()
        avg_default_prob = cohort_df['default_probability'].mean()
        avg_utilization = cohort_df['utilization_rate'].mean()
        
        # Assume monthly interest rate (18% APR / 12)
        monthly_rate = 0.18 / 12
        
        cashflows = []
        
        remaining_balance = total_outstanding
        cumulative_defaults = 0
        
        for month in months:
            # Monthly default rate increases over time (cumulative hazard)
            monthly_default_rate = avg_default_prob * (0.02 + 0.005 * month)  # Increasing default curve
            
            # Defaults this month
            defaults_this_month = remaining_balance * monthly_default_rate
            cumulative_defaults += defaults_this_month
            
            # Collections (payments on non-defaulted balance)
            # Assume 5% of outstanding is paid down monthly
            payment_rate = 0.05
            principal_payment = (remaining_balance - defaults_this_month) * payment_rate
            
            # Interest collected on remaining balance
            interest_payment = remaining_balance * monthly_rate * (1 - monthly_default_rate)
            
            # Recoveries on defaults (assume 45% recovery rate with 6-month lag)
            recovery = 0
            if month > 6:
                lagged_defaults = remaining_balance * avg_default_prob * (0.02 + 0.005 * (month - 6))
                recovery = lagged_defaults * 0.45
            
            # Total collections
            total_collections = principal_payment + interest_payment + recovery

            # Losses (defaults after recoveries)
            losses_this_month = defaults_this_month * 0.55  # 55% LGD (Loss Given Default)

            # Net cash flow (collections minus actual losses, not defaults)
            net_cf = total_collections - losses_this_month
            
            # Update remaining balance
            remaining_balance = remaining_balance - principal_payment - defaults_this_month
            
            if remaining_balance < 0:
                remaining_balance = 0
            
            cashflows.append({
                'month': month,
                'date': self.start_date + relativedelta(months=month),
                'cohort': cohort_name,
                'beginning_balance': remaining_balance + principal_payment + defaults_this_month,
                'principal_payment': principal_payment,
                'interest_payment': interest_payment,
                'recoveries': recovery,
                'collections': total_collections,
                'defaults': defaults_this_month,
                'losses': defaults_this_month * 0.55,  # 55% LGD
                'net_cashflow': net_cf,
                'ending_balance': remaining_balance
            })
        
        return pd.DataFrame(cashflows)
    
    def _aggregate_cashflows(self, cashflow_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Aggregate cash flows across all cohorts"""
        if not cashflow_list:
            return pd.DataFrame()
        
        # Concatenate all cohort cash flows
        all_cf = pd.concat(cashflow_list, ignore_index=True)
        
        # Group by month and sum
        aggregated = all_cf.groupby('month').agg({
            'beginning_balance': 'sum',
            'principal_payment': 'sum',
            'interest_payment': 'sum',
            'recoveries': 'sum',
            'collections': 'sum',
            'defaults': 'sum',
            'losses': 'sum',
            'net_cashflow': 'sum',
            'ending_balance': 'sum'
        }).reset_index()
        
        # Add date column
        aggregated['date'] = aggregated['month'].apply(
            lambda m: self.start_date + relativedelta(months=m)
        )
        
        return aggregated
    
    def _project_individual_loan(self, loan: pd.Series) -> Dict:
        """
        Project cash flows for an individual loan (Dynamic method)
        """
        # Loan characteristics
        outstanding = loan.get('outstanding_balance', 0)
        if outstanding == 0:
            # If no outstanding, assume 30% of credit limit
            outstanding = loan['credit_limit'] * 0.3
        
        credit_limit = loan['credit_limit']
        default_prob = loan['default_probability']
        
        # Assume this is a term loan being converted from revolving
        # Term: 36 months, fixed rate 18% APR
        term_months = 36
        monthly_rate = 0.18 / 12
        
        # Calculate monthly payment (amortization)
        if outstanding > 0:
            monthly_payment = outstanding * (monthly_rate * (1 + monthly_rate)**term_months) / \
                             ((1 + monthly_rate)**term_months - 1)
        else:
            monthly_payment = 0
        
        # Project when default might occur (if at all)
        # Higher default prob = earlier potential default
        default_month = None
        if default_prob > 0.15:  # High risk loans
            # Default randomly in first 12 months
            if np.random.random() < default_prob * 3:  # Increase likelihood
                default_month = np.random.randint(1, min(13, self.projection_months + 1))
        elif default_prob > 0.05:  # Medium risk
            if np.random.random() < default_prob * 2:
                default_month = np.random.randint(6, min(19, self.projection_months + 1))
        else:  # Low risk
            if np.random.random() < default_prob:
                default_month = np.random.randint(12, min(self.projection_months + 1, term_months + 1))
        
        cashflows = []
        remaining_balance = outstanding
        
        for month in range(1, min(self.projection_months + 1, term_months + 1)):
            if month == default_month:
                # Loan defaults this month
                cashflows.append({
                    'month': month,
                    'principal': 0,
                    'interest': 0,
                    'total_payment': 0,
                    'default': remaining_balance,
                    'recovery': remaining_balance * 0.45,  # 45% recovery
                    'loss': remaining_balance * 0.55,
                    'remaining_balance': 0,
                    'status': 'defaulted'
                })
                remaining_balance = 0
                break
            elif remaining_balance > 0:
                # Regular payment
                interest = remaining_balance * monthly_rate
                principal = min(monthly_payment - interest, remaining_balance)
                
                cashflows.append({
                    'month': month,
                    'principal': principal,
                    'interest': interest,
                    'total_payment': principal + interest,
                    'default': 0,
                    'recovery': 0,
                    'loss': 0,
                    'remaining_balance': remaining_balance - principal,
                    'status': 'performing'
                })
                
                remaining_balance -= principal
        
        return {
            'loan_id': loan.get('account_id', 'unknown'),
            'original_balance': outstanding,
            'default_probability': default_prob,
            'defaulted': default_month is not None,
            'default_month': default_month,
            'cashflows': cashflows
        }
    
    def _aggregate_individual_cashflows(self, individual_cfs: List[Dict], 
                                       sample_size: int, total_size: int) -> pd.DataFrame:
        """
        Aggregate individual loan cash flows and scale to full portfolio
        """
        # Initialize monthly totals
        monthly_totals = {month: {
            'principal': 0, 'interest': 0, 'total_payment': 0,
            'defaults': 0, 'recoveries': 0, 'losses': 0
        } for month in range(1, self.projection_months + 1)}
        
        # Aggregate from sample
        for loan_cf in individual_cfs:
            for cf in loan_cf['cashflows']:
                month = cf['month']
                if month <= self.projection_months:
                    monthly_totals[month]['principal'] += cf['principal']
                    monthly_totals[month]['interest'] += cf['interest']
                    monthly_totals[month]['total_payment'] += cf['total_payment']
                    monthly_totals[month]['defaults'] += cf['default']
                    monthly_totals[month]['recoveries'] += cf['recovery']
                    monthly_totals[month]['losses'] += cf['loss']
        
        # Scale to full portfolio
        scale_factor = total_size / sample_size
        
        df_rows = []
        for month in range(1, self.projection_months + 1):
            totals = monthly_totals[month]
            df_rows.append({
                'month': month,
                'date': self.start_date + relativedelta(months=month),
                'principal_payment': totals['principal'] * scale_factor,
                'interest_payment': totals['interest'] * scale_factor,
                'collections': totals['total_payment'] * scale_factor + totals['recoveries'] * scale_factor,
                'defaults': totals['defaults'] * scale_factor,
                'recoveries': totals['recoveries'] * scale_factor,
                'losses': totals['losses'] * scale_factor,
                'net_cashflow': (totals['total_payment'] + totals['recoveries'] - totals['losses']) * scale_factor
            })
        
        return pd.DataFrame(df_rows)
    
    def _create_loan_summary(self, sample_df: pd.DataFrame, individual_cfs: List[Dict]) -> pd.DataFrame:
        """Create summary of individual loan projections"""
        summaries = []
        
        for loan_cf in individual_cfs[:20]:  # Top 20 for display
            total_collections = sum(cf['total_payment'] + cf['recovery'] for cf in loan_cf['cashflows'])
            total_losses = sum(cf['loss'] for cf in loan_cf['cashflows'])
            
            summaries.append({
                'Loan ID': loan_cf['loan_id'],
                'Original Balance': loan_cf['original_balance'],
                'Default Prob': loan_cf['default_probability'],
                'Status': 'Defaulted' if loan_cf['defaulted'] else 'Performing',
                'Default Month': loan_cf['default_month'] if loan_cf['defaulted'] else '-',
                'Total Collections': total_collections,
                'Total Losses': total_losses
            })
        
        return pd.DataFrame(summaries)
    
    def _compare_methods(self, method_a: Dict, method_b: Dict) -> Dict:
        """
        Compare results from both methods
        """
        comparison = {}
        
        # Extract key metrics
        cf_a = method_a['monthly_cashflows']
        cf_b = method_b['monthly_cashflows']
        
        comparison['total_collections'] = {
            'Method A (Static Pool)': method_a['total_collections'],
            'Method B (Dynamic CF)': method_b['total_collections'],
            'Difference': method_a['total_collections'] - method_b['total_collections'],
            'Difference %': ((method_a['total_collections'] - method_b['total_collections']) / 
                           method_b['total_collections'] * 100) if method_b['total_collections'] > 0 else 0
        }
        
        comparison['total_losses'] = {
            'Method A (Static Pool)': method_a['total_losses'],
            'Method B (Dynamic CF)': method_b['total_losses'],
            'Difference': method_a['total_losses'] - method_b['total_losses'],
            'Difference %': ((method_a['total_losses'] - method_b['total_losses']) / 
                           method_b['total_losses'] * 100) if method_b['total_losses'] > 0 else 0
        }
        
        # Monthly comparison
        cf_comparison = pd.DataFrame({
            'Month': cf_a['month'],
            'Static Pool Collections': cf_a['collections'],
            'Dynamic CF Collections': cf_b['collections'],
            'Static Pool Losses': cf_a['losses'],
            'Dynamic CF Losses': cf_b['losses'],
            'Static Pool Net CF': cf_a['net_cashflow'],
            'Dynamic CF Net CF': cf_b['net_cashflow']
        })
        
        comparison['monthly_comparison'] = cf_comparison
        
        # Key differences explanation
        comparison['key_differences'] = {
            'Methodology': {
                'Method A': 'Aggregate cohort behavior - treats portfolio as pools',
                'Method B': 'Individual loan tracking - models each loan separately'
            },
            'Best For': {
                'Method A': 'Revolving credit (credit cards), quick analysis, regulatory reporting',
                'Method B': 'Term loans (auto, mortgage), M&A valuation, detailed portfolio analysis'
            },
            'Granularity': {
                'Method A': 'Cohort-level (Low/Med/High risk groups)',
                'Method B': 'Loan-level (individual amortization schedules)'
            },
            'Default Modeling': {
                'Method A': 'Aggregate default rates by cohort over time',
                'Method B': 'Probabilistic default timing for each loan'
            },
            'Computational Cost': {
                'Method A': 'Low - fast for any portfolio size',
                'Method B': 'Higher - may need sampling for large portfolios'
            }
        }
        
        return comparison

