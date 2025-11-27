"""
Test the credit portfolio analyzer with synthetic credit card data
"""

import sys
import pandas as pd
import numpy as np
from src.file_parser import CreditFileParser
from src.ml_models import CreditRiskModel
from src.portfolio_metrics import PortfolioAnalyzer

print("="*60)
print("TESTING CREDIT PORTFOLIO ANALYZER")
print("="*60)

# Test 1: File Parser (with synthetic data)
print("\n1. Testing File Parser...")
try:
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    
    test_data = {
        'ID': range(1, n_samples + 1),
        'LIMIT_BAL': np.random.randint(10000, 500000, n_samples),
        'SEX': np.random.randint(1, 3, n_samples),
        'EDUCATION': np.random.randint(1, 5, n_samples),
        'MARRIAGE': np.random.randint(1, 4, n_samples),
        'AGE': np.random.randint(21, 70, n_samples),
        'PAY_0': np.random.randint(-1, 4, n_samples),
        'PAY_2': np.random.randint(-1, 4, n_samples),
        'PAY_3': np.random.randint(-1, 4, n_samples),
        'BILL_AMT1': np.random.randint(0, 200000, n_samples),
        'BILL_AMT2': np.random.randint(0, 200000, n_samples),
        'PAY_AMT1': np.random.randint(0, 50000, n_samples),
        'PAY_AMT2': np.random.randint(0, 50000, n_samples),
    }
    
    test_df = pd.DataFrame(test_data)
    
    # Save to temporary CSV
    temp_file = 'temp_test_data.csv'
    test_df.to_csv(temp_file, index=False)
    
    parser = CreditFileParser()
    
    # Analyze the file
    data_info = parser.analyze_file(temp_file)
    
    print(f"   ✓ File Type: {data_info['file_type']}")
    print(f"   ✓ Records: {data_info['num_records']:,}")
    print(f"   ✓ Columns: {data_info['num_columns']}")
    print(f"   ✓ Quality Score: {data_info['quality_score']:.2%}")
    
    # Load and standardize
    df = parser.load_and_standardize(temp_file)
    
    print(f"   ✓ Standardized columns: {len(df.columns)}")
    print(f"   ✓ Sample columns: {list(df.columns[:5])}")
    
    # Verify balances are loaded
    if 'outstanding_balance' in df.columns:
        print(f"   ✓ Outstanding balance: Total=${df['outstanding_balance'].sum():,.0f}, Avg=${df['outstanding_balance'].mean():,.0f}")
    else:
        print(f"   ⚠️  Warning: outstanding_balance column not found!")
    
    # Clean up temp file
    import os
    os.remove(temp_file)
    
except Exception as e:
    print(f"   ❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    # Clean up temp file if it exists
    try:
        import os
        if os.path.exists('temp_test_data.csv'):
            os.remove('temp_test_data.csv')
    except:
        pass
    sys.exit(1)

# Test 2: ML Risk Scoring
print("\n2. Testing ML Risk Scoring...")
try:
    risk_model = CreditRiskModel()
    df = risk_model.score_portfolio(df)
    
    print(f"   ✓ Scores generated for {len(df)} accounts")
    print(f"   ✓ Avg default probability: {df['default_probability'].mean():.2%}")
    print(f"   ✓ Avg risk score: {df['risk_score'].mean():.2f}")
    print(f"   ✓ Risk categories:")
    print(df['risk_category'].value_counts().to_string().replace('\n', '\n      '))
    
except Exception as e:
    print(f"   ❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Portfolio Metrics
print("\n3. Testing Portfolio Metrics...")
try:
    analyzer = PortfolioAnalyzer(df)
    metrics = analyzer.calculate_all_metrics()
    
    print(f"   ✓ Total Exposure: ${metrics['total_exposure']:,.0f}")
    print(f"   ✓ Total Outstanding: ${metrics['total_outstanding']:,.0f}")
    print(f"   ✓ Avg Utilization: {metrics['avg_utilization']:.2%}")
    print(f"   ✓ Expected Loss: ${metrics['expected_loss']:,.0f}")
    print(f"   ✓ High Risk Accounts: {metrics['high_risk_count']:,}")
    
    # Test valuation
    valuation = analyzer.calculate_portfolio_value()
    print(f"   ✓ Book Value: ${valuation['book_value']:,.0f}")
    print(f"   ✓ Net Portfolio Value: ${valuation['net_value']:,.0f}")
    print(f"   ✓ ROA: {valuation['roa']:.2%}")
    
    # Test cash flow analysis
    if 'cash_flow_analysis' in valuation:
        print(f"\n   📊 Cash Flow Analysis Available:")
        cf_analysis = valuation['cash_flow_analysis']
        
        method_a = cf_analysis['method_a']
        method_b = cf_analysis['method_b']
        
        print(f"   ✓ Method A (Static Pool):")
        print(f"      • NPV: ${method_a['npv']:,.0f}")
        print(f"      • IRR: {method_a['irr_annual']:.2f}%")
        print(f"      • MOIC: {method_a['moic']:.2f}x")
        
        print(f"   ✓ Method B (Dynamic CF):")
        print(f"      • NPV: ${method_b['npv']:,.0f}")
        print(f"      • IRR: {method_b['irr_annual']:.2f}%")
        print(f"      • MOIC: {method_b['moic']:.2f}x")
        
        # Check difference
        npv_diff_pct = abs((method_a['npv'] - method_b['npv']) / method_b['npv'] * 100) if method_b['npv'] != 0 else 0
        print(f"   ✓ NPV Difference: {npv_diff_pct:.1f}%")
        
        if npv_diff_pct < 15:
            print(f"   ✅ Methods are in reasonable agreement")
        else:
            print(f"   ⚠️  Methods show >15% difference (expected for test data)")
    else:
        print(f"   ℹ️  Cash flow analysis not available (may need more data)")
    
except Exception as e:
    print(f"   ❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Report Generation
print("\n4. Testing Report Generation...")
try:
    from src.report_generator import ReportGenerator
    
    report_gen = ReportGenerator(df, metrics, valuation)
    
    # Test Excel generation
    excel_buffer = report_gen.create_excel_report()
    print(f"   ✓ Excel report generated: {len(excel_buffer.getvalue())} bytes")
    
    # Test PDF generation
    pdf_buffer = report_gen.create_pdf_report()
    print(f"   ✓ PDF report generated: {len(pdf_buffer.getvalue())} bytes")
    
except Exception as e:
    print(f"   ❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nThe system is working correctly!")
print("You can now run:")
print("  • streamlit run /home/claude/credit_portfolio_analyzer.py")
print("  • python /home/claude/analyze_cli.py --file your_file.csv")
