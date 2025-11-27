"""
Debt Collection Portfolio Pricing Module
Implements P/C ratio and DCF methodologies for pricing charged-off debt portfolios
"""

import pandas as pd
import numpy as np
import numpy_financial as npf
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DebtPortfolioPricer:
    """
    Price debt collection portfolios using industry-standard methodologies:
    - P/C Ratio (Price-to-Collections)
    - DCF (Discounted Cash Flow) with collection curves
    """

    # Standard collection curves by portfolio type
    STANDARD_CURVES = {
        'consumer_unsecured': {
            1: 0.40,  # 40% Year 1
            2: 0.30,  # 30% Year 2
            3: 0.20,  # 20% Year 3
            4: 0.10   # 10% Year 4
        },
        'consumer_secured': {
            1: 0.50,
            2: 0.30,
            3: 0.15,
            4: 0.05
        },
        'auto_loans': {
            1: 0.60,
            2: 0.25,
            3: 0.10,
            4: 0.05
        },
        'credit_cards': {
            1: 0.45,
            2: 0.30,
            3: 0.15,
            4: 0.10
        },
        'medical': {
            1: 0.35,
            2: 0.30,
            3: 0.20,
            4: 0.15
        }
    }

    # Standard P/C ratios by portfolio type
    STANDARD_PC_RATIOS = {
        'consumer_unsecured': 0.45,
        'consumer_secured': 0.55,
        'auto_loans': 0.60,
        'credit_cards': 0.50,
        'medical': 0.40
    }

    def __init__(
        self,
        face_value: float,
        recovery_rate: float,
        portfolio_type: str = 'consumer_unsecured',
        servicing_costs: float = 0.30,
        target_irr: float = 0.18
    ):
        """
        Initialize debt portfolio pricer.

        Args:
            face_value: Total face value of debt portfolio
            recovery_rate: Expected recovery rate (e.g., 0.30 for 30%)
            portfolio_type: Type of portfolio (for default curves/ratios)
            servicing_costs: Servicing costs as % of collections (default 30%)
            target_irr: Target annual IRR (default 18%)
        """
        self.face_value = face_value
        self.recovery_rate = recovery_rate
        self.portfolio_type = portfolio_type
        self.servicing_costs = servicing_costs
        self.target_irr = target_irr

        # Calculate ERC (Estimated Remaining Collections)
        self.erc = face_value * recovery_rate

    def calculate_both_methods(
        self,
        custom_curve: Optional[Dict[int, float]] = None,
        custom_pc_ratio: Optional[float] = None
    ) -> Dict:
        """
        Calculate price using both P/C and DCF methods.

        Args:
            custom_curve: Custom collection curve (overrides standard)
            custom_pc_ratio: Custom P/C ratio (overrides standard)

        Returns:
            Dictionary with pricing from both methods and recommendation
        """
        # Method 1: P/C Ratio
        pc_results = self.price_pc_ratio(custom_pc_ratio)

        # Method 2: DCF
        dcf_results = self.price_dcf(custom_curve)

        # Recommendation (average of both)
        recommended_bid = (pc_results['price'] + dcf_results['price']) / 2

        # Verify IRR at recommended bid
        irr_check = self.verify_irr(recommended_bid, custom_curve)

        return {
            'portfolio_info': {
                'face_value': self.face_value,
                'recovery_rate': self.recovery_rate,
                'erc': self.erc,
                'portfolio_type': self.portfolio_type,
                'servicing_costs': self.servicing_costs,
                'target_irr': self.target_irr
            },
            'pc_method': pc_results,
            'dcf_method': dcf_results,
            'recommendation': {
                'bid_amount': recommended_bid,
                'bid_as_pct_of_face': recommended_bid / self.face_value,
                'bid_as_pct_of_erc': recommended_bid / self.erc,
                'expected_irr': irr_check['annual_irr'],
                'exceeds_target': irr_check['exceeds_target'],
                'decision': 'BUY' if irr_check['exceeds_target'] else 'PASS',
                'margin_to_target': irr_check['margin']
            }
        }

    def price_pc_ratio(self, custom_pc_ratio: Optional[float] = None) -> Dict:
        """
        Price using P/C (Price-to-Collections) ratio method.

        Args:
            custom_pc_ratio: Custom P/C ratio (if not using standard)

        Returns:
            Dictionary with P/C pricing details
        """
        pc_ratio = custom_pc_ratio if custom_pc_ratio else \
                   self.STANDARD_PC_RATIOS.get(self.portfolio_type, 0.50)

        price = self.erc * pc_ratio

        return {
            'method': 'P/C Ratio',
            'pc_ratio': pc_ratio,
            'erc': self.erc,
            'price': price,
            'price_as_pct_of_face': price / self.face_value,
            'calculation': f'Price = ERC × P/C = €{self.erc:,.0f} × {pc_ratio:.0%}'
        }

    def price_dcf(self, custom_curve: Optional[Dict[int, float]] = None) -> Dict:
        """
        Price using DCF (Discounted Cash Flow) method.

        Args:
            custom_curve: Custom collection curve (if not using standard)

        Returns:
            Dictionary with DCF pricing details
        """
        # Use custom curve or standard curve for portfolio type
        curve = custom_curve if custom_curve else \
                self.STANDARD_CURVES.get(self.portfolio_type, self.STANDARD_CURVES['consumer_unsecured'])

        # Calculate present value of each year's collections
        yearly_cashflows = []
        total_pv = 0

        for year, pct in curve.items():
            # Gross collections this year
            gross = self.erc * pct

            # Net after servicing costs
            net = gross * (1 - self.servicing_costs)

            # Discount factor
            discount_factor = (1 + self.target_irr) ** year

            # Present value
            pv = net / discount_factor

            total_pv += pv

            yearly_cashflows.append({
                'year': year,
                'collection_pct': pct,
                'gross_collections': gross,
                'servicing_costs': gross * self.servicing_costs,
                'net_collections': net,
                'discount_factor': discount_factor,
                'present_value': pv
            })

        return {
            'method': 'DCF (Discounted Cash Flow)',
            'target_irr': self.target_irr,
            'collection_curve': curve,
            'yearly_cashflows': pd.DataFrame(yearly_cashflows),
            'total_pv': total_pv,
            'price': total_pv,
            'price_as_pct_of_face': total_pv / self.face_value
        }

    def verify_irr(
        self,
        bid_price: float,
        custom_curve: Optional[Dict[int, float]] = None
    ) -> Dict:
        """
        Verify IRR at a given bid price.

        Args:
            bid_price: Proposed bid price
            custom_curve: Custom collection curve (if not using standard)

        Returns:
            Dictionary with IRR verification results
        """
        # Use custom curve or standard curve
        curve = custom_curve if custom_curve else \
                self.STANDARD_CURVES.get(self.portfolio_type, self.STANDARD_CURVES['consumer_unsecured'])

        # Build monthly cash flows
        monthly_cashflows = []

        for year, pct in curve.items():
            # Yearly gross collections
            yearly_gross = self.erc * pct

            # Monthly gross
            monthly_gross = yearly_gross / 12

            # Monthly net (after servicing costs)
            monthly_net = monthly_gross * (1 - self.servicing_costs)

            # Add 12 months of collections for this year
            for month in range(12):
                monthly_cashflows.append(monthly_net)

        # Calculate IRR
        cash_flows = [-bid_price] + monthly_cashflows

        try:
            monthly_irr = npf.irr(cash_flows)
            annual_irr = (1 + monthly_irr) ** 12 - 1
        except:
            # If IRR calculation fails (e.g., no sign change), return None
            annual_irr = None
            monthly_irr = None

        # Check if exceeds target
        exceeds_target = annual_irr >= self.target_irr if annual_irr else False
        margin = annual_irr - self.target_irr if annual_irr else None

        return {
            'bid_price': bid_price,
            'monthly_irr': monthly_irr,
            'annual_irr': annual_irr,
            'target_irr': self.target_irr,
            'exceeds_target': exceeds_target,
            'margin': margin,
            'monthly_cashflows': monthly_cashflows,
            'total_collections': sum(monthly_cashflows)
        }

    def sensitivity_analysis(
        self,
        recovery_range: Tuple[float, float] = (0.20, 0.40),
        irr_range: Tuple[float, float] = (0.12, 0.24),
        steps: int = 5
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on recovery rate and target IRR.

        Args:
            recovery_range: (min, max) recovery rates to test
            irr_range: (min, max) IRRs to test
            steps: Number of steps in each dimension

        Returns:
            DataFrame with sensitivity analysis results
        """
        recovery_rates = np.linspace(recovery_range[0], recovery_range[1], steps)
        target_irrs = np.linspace(irr_range[0], irr_range[1], steps)

        results = []

        for recovery in recovery_rates:
            for irr in target_irrs:
                # Create temporary pricer with these parameters
                temp_pricer = DebtPortfolioPricer(
                    face_value=self.face_value,
                    recovery_rate=recovery,
                    portfolio_type=self.portfolio_type,
                    servicing_costs=self.servicing_costs,
                    target_irr=irr
                )

                # Calculate pricing
                pricing = temp_pricer.calculate_both_methods()

                results.append({
                    'recovery_rate': recovery,
                    'target_irr': irr,
                    'erc': temp_pricer.erc,
                    'pc_price': pricing['pc_method']['price'],
                    'dcf_price': pricing['dcf_method']['price'],
                    'recommended_bid': pricing['recommendation']['bid_amount'],
                    'expected_irr': pricing['recommendation']['expected_irr']
                })

        return pd.DataFrame(results)

    def create_monthly_schedule(
        self,
        custom_curve: Optional[Dict[int, float]] = None
    ) -> pd.DataFrame:
        """
        Create detailed monthly collection schedule.

        Args:
            custom_curve: Custom collection curve (if not using standard)

        Returns:
            DataFrame with monthly collection schedule
        """
        curve = custom_curve if custom_curve else \
                self.STANDARD_CURVES.get(self.portfolio_type, self.STANDARD_CURVES['consumer_unsecured'])

        schedule = []
        month_num = 0
        cumulative = 0

        for year, pct in curve.items():
            yearly_gross = self.erc * pct
            monthly_gross = yearly_gross / 12
            monthly_net = monthly_gross * (1 - self.servicing_costs)
            monthly_servicing = monthly_gross * self.servicing_costs

            for month in range(12):
                month_num += 1
                cumulative += monthly_net

                schedule.append({
                    'month': month_num,
                    'year': year,
                    'date': datetime.now() + relativedelta(months=month_num),
                    'gross_collections': monthly_gross,
                    'servicing_costs': monthly_servicing,
                    'net_collections': monthly_net,
                    'cumulative_net': cumulative,
                    'cumulative_pct_of_erc': cumulative / self.erc
                })

        return pd.DataFrame(schedule)


def compare_portfolios(portfolios: List[Dict]) -> pd.DataFrame:
    """
    Compare multiple debt portfolios side-by-side.

    Args:
        portfolios: List of portfolio dictionaries with pricing parameters

    Returns:
        DataFrame comparing all portfolios
    """
    comparisons = []

    for i, portfolio in enumerate(portfolios):
        pricer = DebtPortfolioPricer(
            face_value=portfolio['face_value'],
            recovery_rate=portfolio['recovery_rate'],
            portfolio_type=portfolio.get('portfolio_type', 'consumer_unsecured'),
            servicing_costs=portfolio.get('servicing_costs', 0.30),
            target_irr=portfolio.get('target_irr', 0.18)
        )

        results = pricer.calculate_both_methods(
            custom_curve=portfolio.get('custom_curve'),
            custom_pc_ratio=portfolio.get('custom_pc_ratio')
        )

        comparisons.append({
            'Portfolio': portfolio.get('name', f'Portfolio {i+1}'),
            'Face Value': results['portfolio_info']['face_value'],
            'Recovery Rate': results['portfolio_info']['recovery_rate'],
            'ERC': results['portfolio_info']['erc'],
            'P/C Price': results['pc_method']['price'],
            'DCF Price': results['dcf_method']['price'],
            'Recommended Bid': results['recommendation']['bid_amount'],
            'Expected IRR': results['recommendation']['expected_irr'],
            'Decision': results['recommendation']['decision']
        })

    return pd.DataFrame(comparisons)


# Example usage and testing
if __name__ == "__main__":
    # Example from user's code
    pricer = DebtPortfolioPricer(
        face_value=50_000_000,
        recovery_rate=0.30,
        portfolio_type='consumer_unsecured',
        servicing_costs=0.30,
        target_irr=0.18
    )

    results = pricer.calculate_both_methods()

    print("="*60)
    print(f"PORTFOLIO PRICING RESULTS")
    print("="*60)
    print(f"Face Value: €{results['portfolio_info']['face_value']:,.0f}")
    print(f"ERC: €{results['portfolio_info']['erc']:,.0f}")
    print()
    print(f"P/C Method Price: €{results['pc_method']['price']:,.0f}")
    print(f"DCF Method Price: €{results['dcf_method']['price']:,.0f}")
    print()
    print(f"RECOMMENDED BID: €{results['recommendation']['bid_amount']:,.0f}")
    print(f"Expected IRR: {results['recommendation']['expected_irr']:.1%}")
    print(f"Decision: {results['recommendation']['decision']}")
    print("="*60)
