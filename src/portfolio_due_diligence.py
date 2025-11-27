"""
Portfolio Due Diligence Module
Analyzes debt portfolios for red flags and composition issues
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class PortfolioDueDiligence:
    """
    Perform comprehensive due diligence on debt portfolios.
    Identifies red flags and calculates adjusted ERC.
    """

    def __init__(self, df: pd.DataFrame, base_recovery_rate: float = 0.30, statute_years: int = 5):
        """
        Initialize due diligence analyzer.

        Args:
            df: DataFrame with portfolio data
            base_recovery_rate: Initial recovery rate estimate
            statute_years: Statute of limitations period (varies by jurisdiction)
        """
        self.df = df.copy()
        self.base_recovery_rate = base_recovery_rate
        self.statute_years = statute_years
        self.statute_days = statute_years * 365

        # Ensure balance column exists
        if 'Current_Balance' not in self.df.columns and 'Balance' in self.df.columns:
            self.df['Current_Balance'] = self.df['Balance']
        elif 'current_balance' in self.df.columns:
            self.df['Current_Balance'] = self.df['current_balance']

    def analyze_composition(self) -> Dict:
        """
        Analyze portfolio composition by various segments.

        Returns:
            Dictionary with composition analysis results
        """
        results = {
            'total_face_value': self.df['Current_Balance'].sum(),
            'total_accounts': len(self.df)
        }

        # By loan type (if available)
        if 'Loan_Type' in self.df.columns or 'loan_type' in self.df.columns:
            type_col = 'Loan_Type' if 'Loan_Type' in self.df.columns else 'loan_type'
            results['by_loan_type'] = self._analyze_by_loan_type(type_col)

        # By balance size
        results['by_balance_size'] = self._analyze_by_balance_size()

        # By age of debt (if available)
        if self._has_date_info():
            results['by_age'] = self._analyze_by_age()

        return results

    def check_red_flags(self) -> Dict:
        """
        Check for critical red flags in the portfolio.

        Returns:
            Dictionary with red flag analysis
        """
        red_flags = {}

        # Red Flag #1: Statute of limitations
        red_flags['statute_issues'] = self._check_statute_issues()

        # Red Flag #2: Balance inflation
        red_flags['balance_inflation'] = self._check_balance_inflation()

        # Red Flag #3: Missing contact information
        red_flags['missing_contacts'] = self._check_missing_contacts()

        # Red Flag #4: Concentration risk
        red_flags['concentration'] = self._check_concentration()

        # Red Flag #5: Very small balances
        red_flags['small_balances'] = self._check_small_balances()

        return red_flags

    def calculate_adjusted_erc(self) -> Dict:
        """
        Calculate adjusted ERC based on due diligence findings.

        Returns:
            Dictionary with adjusted ERC calculation
        """
        total_face = self.df['Current_Balance'].sum()
        base_erc = total_face * self.base_recovery_rate

        adjustments = []
        affected_value = 0

        # Get red flags
        red_flags = self.check_red_flags()

        # Adjustment 1: Statute-barred debt (0% recovery)
        if red_flags['statute_issues']['past_statute_count'] > 0:
            statute_value = red_flags['statute_issues']['past_statute_value']
            adjustment = statute_value * (0 - self.base_recovery_rate)
            adjustments.append({
                'category': 'Statute-Barred Debt',
                'affected_value': statute_value,
                'adjustment': adjustment,
                'new_recovery_rate': 0.0
            })
            affected_value += statute_value

        # Adjustment 2: Missing contact info (5% recovery instead of base)
        if red_flags['missing_contacts']['count'] > 0:
            missing_value = red_flags['missing_contacts']['value']
            adjustment = missing_value * (0.05 - self.base_recovery_rate)
            adjustments.append({
                'category': 'Missing Contact Info',
                'affected_value': missing_value,
                'adjustment': adjustment,
                'new_recovery_rate': 0.05
            })
            affected_value += missing_value

        # Adjustment 3: Small balances (<€500) (50% of base rate)
        if red_flags['small_balances']['count'] > 0:
            small_value = red_flags['small_balances']['value']
            adjustment = small_value * (self.base_recovery_rate * 0.5 - self.base_recovery_rate)
            adjustments.append({
                'category': 'Small Balances (<€500)',
                'affected_value': small_value,
                'adjustment': adjustment,
                'new_recovery_rate': self.base_recovery_rate * 0.5
            })
            affected_value += small_value

        # Adjustment 4: Near statute (reduce recovery by 20%)
        if red_flags['statute_issues']['near_statute_count'] > 0:
            near_value = red_flags['statute_issues']['near_statute_value']
            adjustment = near_value * (self.base_recovery_rate * 0.8 - self.base_recovery_rate)
            adjustments.append({
                'category': 'Near Statute Limit',
                'affected_value': near_value,
                'adjustment': adjustment,
                'new_recovery_rate': self.base_recovery_rate * 0.8
            })
            affected_value += near_value

        # Adjustment 5: Concentration risk (apply discount to top 10)
        if red_flags['concentration']['top_10_pct'] > 20:
            top_10_value = red_flags['concentration']['top_10_value']
            concentration_discount = 0.10  # 10% discount
            adjustment = top_10_value * -concentration_discount
            adjustments.append({
                'category': 'Concentration Risk (Top 10)',
                'affected_value': top_10_value,
                'adjustment': adjustment,
                'new_recovery_rate': self.base_recovery_rate * (1 - concentration_discount)
            })

        # Calculate totals
        total_adjustment = sum(adj['adjustment'] for adj in adjustments)
        adjusted_erc = base_erc + total_adjustment
        adjusted_recovery_rate = adjusted_erc / total_face if total_face > 0 else 0

        return {
            'base_erc': base_erc,
            'base_recovery_rate': self.base_recovery_rate,
            'adjustments': adjustments,
            'total_adjustment': total_adjustment,
            'adjusted_erc': adjusted_erc,
            'adjusted_recovery_rate': adjusted_recovery_rate,
            'adjustment_summary': pd.DataFrame(adjustments) if adjustments else pd.DataFrame()
        }

    def _analyze_by_loan_type(self, type_col: str) -> pd.DataFrame:
        """Analyze portfolio by loan type."""
        composition = self.df.groupby(type_col).agg({
            'Current_Balance': ['sum', 'count', 'mean']
        })
        composition.columns = ['Total_Balance', 'Count', 'Avg_Balance']
        composition['Pct_of_Portfolio'] = (
            composition['Total_Balance'] / composition['Total_Balance'].sum() * 100
        )
        return composition.round(2)

    def _analyze_by_balance_size(self) -> pd.DataFrame:
        """Analyze portfolio by balance size segments."""
        self.df['Balance_Segment'] = pd.cut(
            self.df['Current_Balance'],
            bins=[0, 500, 1000, 5000, 10000, 50000, float('inf')],
            labels=['<€500', '€500-€1k', '€1k-€5k', '€5k-€10k', '€10k-€50k', '>€50k']
        )

        composition = self.df.groupby('Balance_Segment', observed=True).agg({
            'Current_Balance': ['sum', 'count']
        })
        composition.columns = ['Total_Balance', 'Count']
        composition['Pct'] = (
            composition['Total_Balance'] / composition['Total_Balance'].sum() * 100
        )
        return composition.round(2)

    def _analyze_by_age(self) -> pd.DataFrame:
        """Analyze portfolio by age of debt."""
        # Calculate days past due
        self._calculate_days_past_due()

        if 'Days_Past_Due' not in self.df.columns:
            return pd.DataFrame()

        self.df['Age_Segment'] = pd.cut(
            self.df['Days_Past_Due'],
            bins=[0, 365, 730, 1095, 1825, float('inf')],
            labels=['<1yr', '1-2yrs', '2-3yrs', '3-5yrs', '>5yrs']
        )

        composition = self.df.groupby('Age_Segment', observed=True).agg({
            'Current_Balance': ['sum', 'count']
        })
        composition.columns = ['Total_Balance', 'Count']
        composition['Pct'] = (
            composition['Total_Balance'] / composition['Total_Balance'].sum() * 100
        )
        return composition.round(2)

    def _check_statute_issues(self) -> Dict:
        """Check for statute of limitations issues."""
        self._calculate_days_past_due()

        if 'Days_Past_Due' not in self.df.columns:
            return {
                'past_statute_count': 0,
                'past_statute_value': 0,
                'near_statute_count': 0,
                'near_statute_value': 0,
                'status': 'unknown'
            }

        # Past statute
        statute_barred = self.df[self.df['Days_Past_Due'] > self.statute_days]

        # Near statute (80% of limit)
        near_statute = self.df[
            (self.df['Days_Past_Due'] > self.statute_days * 0.8) &
            (self.df['Days_Past_Due'] <= self.statute_days)
        ]

        past_value = statute_barred['Current_Balance'].sum()
        near_value = near_statute['Current_Balance'].sum()
        total_value = self.df['Current_Balance'].sum()

        past_pct = (past_value / total_value * 100) if total_value > 0 else 0
        near_pct = (near_value / total_value * 100) if total_value > 0 else 0

        # Determine status
        if past_pct > 5:
            status = 'critical'
        elif near_pct > 10:
            status = 'warning'
        else:
            status = 'ok'

        return {
            'past_statute_count': len(statute_barred),
            'past_statute_value': past_value,
            'past_statute_pct': past_pct,
            'near_statute_count': len(near_statute),
            'near_statute_value': near_value,
            'near_statute_pct': near_pct,
            'status': status,
            'statute_years': self.statute_years
        }

    def _check_balance_inflation(self) -> Dict:
        """Check for unrealistic balance inflation."""
        if 'Original_Loan_Amount' not in self.df.columns:
            return {'status': 'unknown', 'count': 0, 'value': 0}

        # Calculate growth
        self.df['Balance_Growth'] = (
            (self.df['Current_Balance'] - self.df['Original_Loan_Amount']) /
            self.df['Original_Loan_Amount']
        ) * 100

        # Flag balances that have more than doubled
        suspicious = self.df[self.df['Balance_Growth'] > 100]
        suspicious_pct = (len(suspicious) / len(self.df) * 100) if len(self.df) > 0 else 0

        status = 'warning' if suspicious_pct > 10 else 'ok'

        return {
            'count': len(suspicious),
            'value': suspicious['Current_Balance'].sum(),
            'pct': suspicious_pct,
            'status': status
        }

    def _check_missing_contacts(self) -> Dict:
        """Check for missing contact information."""
        # Look for common contact field names
        phone_cols = [col for col in self.df.columns if 'phone' in col.lower()]
        address_cols = [col for col in self.df.columns if 'address' in col.lower()]
        email_cols = [col for col in self.df.columns if 'email' in col.lower()]

        if not phone_cols and not address_cols and not email_cols:
            return {'status': 'unknown', 'count': 0, 'value': 0}

        # Check for missing all contact methods
        missing_mask = pd.Series(True, index=self.df.index)

        for col in phone_cols + address_cols + email_cols:
            missing_mask &= self.df[col].isnull()

        missing = self.df[missing_mask]
        missing_pct = (len(missing) / len(self.df) * 100) if len(self.df) > 0 else 0

        status = 'critical' if missing_pct > 20 else ('warning' if missing_pct > 10 else 'ok')

        return {
            'count': len(missing),
            'value': missing['Current_Balance'].sum(),
            'pct': missing_pct,
            'status': status
        }

    def _check_concentration(self) -> Dict:
        """Check for concentration risk."""
        total_value = self.df['Current_Balance'].sum()

        # Top 10 loans
        top_10 = self.df.nlargest(10, 'Current_Balance')
        top_10_value = top_10['Current_Balance'].sum()
        top_10_pct = (top_10_value / total_value * 100) if total_value > 0 else 0

        # Top 1 loan
        top_1_value = self.df['Current_Balance'].max()
        top_1_pct = (top_1_value / total_value * 100) if total_value > 0 else 0

        status = 'warning' if top_10_pct > 20 else 'ok'

        return {
            'top_10_value': top_10_value,
            'top_10_pct': top_10_pct,
            'top_1_value': top_1_value,
            'top_1_pct': top_1_pct,
            'status': status
        }

    def _check_small_balances(self) -> Dict:
        """Check for very small balances that may not be worth collecting."""
        small = self.df[self.df['Current_Balance'] < 500]
        total_value = self.df['Current_Balance'].sum()

        small_pct = (len(small) / len(self.df) * 100) if len(self.df) > 0 else 0
        small_value_pct = (small['Current_Balance'].sum() / total_value * 100) if total_value > 0 else 0

        status = 'warning' if small_pct > 15 else 'ok'

        return {
            'count': len(small),
            'value': small['Current_Balance'].sum(),
            'pct': small_pct,
            'value_pct': small_value_pct,
            'status': status
        }

    def _has_date_info(self) -> bool:
        """Check if portfolio has date information for aging analysis."""
        date_cols = [col for col in self.df.columns if any(x in col.lower() for x in ['date', 'default', 'charge'])]
        return len(date_cols) > 0

    def _calculate_days_past_due(self):
        """Calculate days past due if date information is available."""
        if 'Days_Past_Due' in self.df.columns:
            return  # Already calculated

        # Look for date columns
        date_cols = [col for col in self.df.columns if 'date' in col.lower() and 'default' in col.lower()]

        if not date_cols:
            return

        date_col = date_cols[0]

        try:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
            today = pd.Timestamp.now()
            self.df['Days_Past_Due'] = (today - self.df[date_col]).dt.days
        except:
            pass  # If conversion fails, skip

    def generate_due_diligence_report(self) -> Dict:
        """
        Generate comprehensive due diligence report.

        Returns:
            Dictionary with all due diligence findings
        """
        return {
            'composition': self.analyze_composition(),
            'red_flags': self.check_red_flags(),
            'adjusted_erc': self.calculate_adjusted_erc(),
            'summary': self._generate_summary()
        }

    def _generate_summary(self) -> Dict:
        """Generate executive summary of findings."""
        red_flags = self.check_red_flags()

        critical_issues = []
        warnings = []

        for flag_name, flag_data in red_flags.items():
            if isinstance(flag_data, dict) and 'status' in flag_data:
                if flag_data['status'] == 'critical':
                    critical_issues.append(flag_name)
                elif flag_data['status'] == 'warning':
                    warnings.append(flag_name)

        adjusted = self.calculate_adjusted_erc()
        erc_impact = ((adjusted['adjusted_erc'] - adjusted['base_erc']) / adjusted['base_erc'] * 100) if adjusted['base_erc'] > 0 else 0

        return {
            'critical_issues_count': len(critical_issues),
            'warnings_count': len(warnings),
            'critical_issues': critical_issues,
            'warnings': warnings,
            'erc_impact_pct': erc_impact,
            'recommendation': self._get_recommendation(critical_issues, warnings, erc_impact)
        }

    def _get_recommendation(self, critical: List, warnings: List, erc_impact: float) -> str:
        """Generate recommendation based on findings."""
        if len(critical) > 0:
            return "CAUTION: Critical issues found. Thorough review required before bidding."
        elif len(warnings) > 2:
            return "PROCEED WITH CAUTION: Multiple warnings detected. Adjust pricing accordingly."
        elif erc_impact < -20:
            return "SIGNIFICANT ADJUSTMENTS: ERC reduced by >20%. Re-evaluate pricing."
        else:
            return "PROCEED: No major issues detected. Standard pricing applicable."


# Example usage
if __name__ == "__main__":
    # Create sample portfolio
    sample_data = {
        'Current_Balance': np.random.lognormal(8, 1.5, 100) * 1000,
        'Original_Loan_Amount': np.random.lognormal(8, 1.2, 100) * 1000
    }

    df = pd.DataFrame(sample_data)

    # Run due diligence
    dd = PortfolioDueDiligence(df, base_recovery_rate=0.30)
    report = dd.generate_due_diligence_report()

    print("Due Diligence Report Generated")
    print(f"ERC Impact: {report['summary']['erc_impact_pct']:.1f}%")
    print(f"Recommendation: {report['summary']['recommendation']}")
