"""
Command-line interface for Credit Portfolio Analyzer
For users who prefer CLI over web interface
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.file_parser import CreditFileParser
from src.ml_models import CreditRiskModel
from src.portfolio_metrics import PortfolioAnalyzer
from src.report_generator import ReportGenerator

def main():
    parser = argparse.ArgumentParser(
        description='Analyze credit portfolio files and generate reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_cli.py --file portfolio.csv
  python analyze_cli.py --file loans.xlsx --output my_report.pdf
  python analyze_cli.py --file credit_cards.csv --format excel --output analysis.xlsx
        """
    )
    
    parser.add_argument(
        '--file',
        required=True,
        help='Path to credit portfolio file (CSV or Excel)'
    )
    
    parser.add_argument(
        '--output',
        default=None,
        help='Output file path (default: auto-generated)'
    )
    
    parser.add_argument(
        '--format',
        choices=['excel', 'pdf', 'both', 'csv'],
        default='both',
        help='Output format (default: both)'
    )
    
    parser.add_argument(
        '--model',
        default=None,
        help='Path to pre-trained ML model (optional)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_file = Path(args.file)
    if not input_file.exists():
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)
    
    if not args.quiet:
        print("=" * 60)
        print("🏦 CREDIT PORTFOLIO ANALYZER")
        print("=" * 60)
        print(f"\n📁 Loading file: {input_file.name}")
    
    # Step 1: Parse file
    try:
        parser_obj = CreditFileParser()
        
        if not args.quiet:
            print("🔍 Analyzing file structure...")
        
        with open(input_file, 'rb') as f:
            data_info = parser_obj.analyze_file(f)
        
        if not args.quiet:
            print(f"   ✓ Detected: {data_info['file_type']}")
            print(f"   ✓ Records: {data_info['num_records']:,}")
            print(f"   ✓ Data Quality: {data_info['quality_score']:.1%}")
        
        # Load and standardize
        with open(input_file, 'rb') as f:
            df = parser_obj.load_and_standardize(f)
        
    except Exception as e:
        print(f"❌ Error parsing file: {str(e)}")
        sys.exit(1)
    
    # Step 2: ML Risk Scoring
    try:
        if not args.quiet:
            print("\n🤖 Running ML risk model...")
        
        risk_model = CreditRiskModel(model_path=args.model)
        df = risk_model.score_portfolio(df)
        
        if not args.quiet:
            avg_risk = df['default_probability'].mean()
            high_risk = (df['risk_category'] == 'High Risk').sum()
            print(f"   ✓ Average default probability: {avg_risk:.2%}")
            print(f"   ✓ High-risk accounts: {high_risk:,}")
        
    except Exception as e:
        print(f"❌ Error in risk scoring: {str(e)}")
        sys.exit(1)
    
    # Step 3: Calculate Metrics
    try:
        if not args.quiet:
            print("\n📊 Calculating portfolio metrics...")
        
        analyzer = PortfolioAnalyzer(df)
        metrics = analyzer.calculate_all_metrics()
        valuation = analyzer.calculate_portfolio_value()
        
        if not args.quiet:
            print(f"   ✓ Total exposure: ${metrics['total_exposure']:,.0f}")
            print(f"   ✓ Expected loss: ${metrics['expected_loss']:,.0f}")
            print(f"   ✓ Portfolio value: ${valuation['net_value']:,.0f}")
        
    except Exception as e:
        print(f"❌ Error calculating metrics: {str(e)}")
        sys.exit(1)
    
    # Step 4: Generate Reports
    try:
        if not args.quiet:
            print("\n📥 Generating reports...")
        
        report_gen = ReportGenerator(df, metrics, valuation)
        
        # Determine output paths
        if args.output:
            base_name = Path(args.output).stem
            output_dir = Path(args.output).parent
        else:
            base_name = input_file.stem + '_analysis'
            output_dir = input_file.parent
        
        generated_files = []
        
        # Generate Excel report
        if args.format in ['excel', 'both']:
            excel_path = output_dir / f"{base_name}.xlsx"
            excel_buffer = report_gen.create_excel_report()
            
            with open(excel_path, 'wb') as f:
                f.write(excel_buffer.getvalue())
            
            generated_files.append(excel_path)
            if not args.quiet:
                print(f"   ✓ Excel report: {excel_path}")
        
        # Generate PDF report
        if args.format in ['pdf', 'both']:
            pdf_path = output_dir / f"{base_name}.pdf"
            pdf_buffer = report_gen.create_pdf_report()
            
            with open(pdf_path, 'wb') as f:
                f.write(pdf_buffer.getvalue())
            
            generated_files.append(pdf_path)
            if not args.quiet:
                print(f"   ✓ PDF report: {pdf_path}")
        
        # Generate scored CSV
        if args.format == 'csv':
            csv_path = output_dir / f"{base_name}_scored.csv"
            df.to_csv(csv_path, index=False)
            
            generated_files.append(csv_path)
            if not args.quiet:
                print(f"   ✓ Scored portfolio: {csv_path}")
        
        # Summary
        if not args.quiet:
            print("\n" + "=" * 60)
            print("✅ ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"\n📊 Key Findings:")
            print(f"   • Total Accounts: {len(df):,}")
            print(f"   • High Risk: {metrics['high_risk_count']:,} ({metrics['high_risk_pct']:.1%})")
            print(f"   • Expected Loss: ${metrics['expected_loss']:,.0f}")
            print(f"   • Portfolio Value: ${valuation['net_value']:,.0f}")
            print(f"\n📁 Generated {len(generated_files)} file(s)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating reports: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    sys.exit(main())
