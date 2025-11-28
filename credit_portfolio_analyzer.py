"""
Automated Credit Portfolio Analyzer
Upload any credit file and get instant analysis, risk scoring, and valuation
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Credit Portfolio Analyzer",
    page_icon="💳",
    layout="wide"
)

# Import custom modules
from src.file_parser import CreditFileParser
from src.ml_models import CreditRiskModel
from src.portfolio_metrics import PortfolioAnalyzer
from src.report_generator import ReportGenerator
from src.column_mapper import ColumnMapper
from src.debt_pricing import DebtPortfolioPricer, compare_portfolios
from src.portfolio_due_diligence import PortfolioDueDiligence
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    st.title("🏦 Automated Credit Portfolio Analyzer")

    # Mode selector
    mode = st.radio(
        "Select Analysis Mode:",
        ["📊 Active Portfolio Analysis", "💰 Debt Collection Pricing"],
        horizontal=True
    )

    if mode == "💰 Debt Collection Pricing":
        debt_collection_pricing_app()
        return

    # Original app continues for Active Portfolio Analysis
    st.markdown("Upload your credit portfolio file and get instant analysis, risk scoring, and valuation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Gemini API configuration
        st.markdown("### 🤖 AI Document Parsing")
        use_gemini = st.checkbox("Enable Gemini AI (PDFs/Images)", value=True)
        if use_gemini:
            gemini_key = os.getenv('GEMINI_API_KEY')
            if gemini_key:
                st.success("✅ Gemini API key detected")
            else:
                st.warning("⚠️ No Gemini API key found in .env")

        st.markdown("---")

        analysis_type = st.selectbox(
            "Analysis Type",
            ["Quick Analysis", "Deep Analysis", "Custom"]
        )

        # Manual mapping option
        enable_manual_mapping = st.checkbox(
            "Enable Manual Column Mapping",
            value=False,
            help="Manually map columns if auto-detection fails"
        )

        st.markdown("---")
        st.markdown("### 📊 What This Tool Does:")
        st.markdown("""
        1. **Auto-detects** file structure
        2. **AI-powered** PDF/Image parsing (Gemini)
        3. **ML Risk Scoring** for each account
        4. **Financial Ratios** calculation
        5. **Portfolio Valuation**
        6. **Risk Metrics** (VaR, Expected Loss)
        7. **Downloadable Reports**
        """)
    
    # File upload
    file_types = ['csv', 'xlsx', 'xls']
    if use_gemini:
        file_types.extend(['pdf', 'png', 'jpg', 'jpeg'])

    uploaded_file = st.file_uploader(
        "Upload Credit Portfolio File",
        type=file_types,
        help="Upload Excel/CSV file or PDF/Image (with Gemini AI enabled)"
    )
    
    # Large file sampling option
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 100:
            st.warning(f"⚠️ Large file detected ({file_size_mb:.1f}MB). Consider using sampling for faster processing.")
            use_sampling = st.checkbox("Use sampling (analyze subset of data)", value=True)
            if use_sampling:
                sample_size = st.slider("Sample size (rows)", 1000, 100000, 10000, step=1000)
        else:
            use_sampling = False
            sample_size = None
    else:
        use_sampling = False
        sample_size = None
    
    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing your file..."):
            # Step 1: Parse and understand the file
            parser = CreditFileParser(use_gemini=use_gemini)
            data_info = parser.analyze_file(uploaded_file)

            # Check if Gemini was used
            is_gemini_parsed = data_info.get('parser') == 'gemini'

            st.success(f"✅ File understood: **{data_info['file_type']}** with **{data_info['num_records']:,}** records")

            # Display file understanding
            with st.expander("📋 File Structure Detected", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Records", f"{data_info['num_records']:,}")
                    st.metric("Total Columns", data_info['num_columns'])
                with col2:
                    st.metric("File Type", data_info['file_type'])
                    st.metric("Data Quality", f"{data_info['quality_score']:.1%}")

                if is_gemini_parsed:
                    st.info("🤖 Parsed using Gemini AI")
                    if 'gemini_metadata' in data_info:
                        st.json(data_info['gemini_metadata'])
                else:
                    st.markdown("**Detected Columns:**")
                    st.dataframe(data_info['column_mapping'], use_container_width=True)

            # Initialize column mapper
            column_mapper = ColumnMapper()

            # Step 2: Manual column mapping (if enabled and not Gemini)
            manual_mapping = None
            if enable_manual_mapping and not is_gemini_parsed:
                st.markdown("---")

                # Load data for preview
                if hasattr(uploaded_file, 'seek'):
                    uploaded_file.seek(0)

                # Quick load for column mapping
                if uploaded_file.name.endswith('.csv'):
                    df_preview = pd.read_csv(uploaded_file, nrows=5)
                else:
                    df_preview = pd.read_excel(uploaded_file, nrows=5)

                if hasattr(uploaded_file, 'seek'):
                    uploaded_file.seek(0)

                # Get automatic mapping for defaults
                auto_mapping_data = data_info.get('column_mapping', pd.DataFrame())
                auto_mapping = {}
                if not auto_mapping_data.empty:
                    for _, row in auto_mapping_data.iterrows():
                        if row['Confidence'] > 60:
                            auto_mapping[row['Mapped To']] = row['Original Column']

                # Render mapping UI
                manual_mapping, mapping_ready = column_mapper.render_mapping_ui(
                    df_preview,
                    auto_mapping=auto_mapping
                )

                if not mapping_ready:
                    st.stop()

                st.markdown("---")

            # Step 3: Load and process data
            df = parser.load_and_standardize(uploaded_file, manual_mapping=manual_mapping)
            
            # Apply sampling if requested
            if use_sampling and sample_size and len(df) > sample_size:
                st.info(f"📊 Sampling {sample_size:,} rows from {len(df):,} total records")
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            # Step 3: Run ML Risk Model
            st.markdown("---")
            st.subheader("🤖 ML Risk Scoring")
            
            with st.spinner("Running credit risk model..."):
                risk_model = CreditRiskModel()
                df = risk_model.score_portfolio(df)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Portfolio Risk Score",
                    f"{df['risk_score'].mean():.2f}",
                    delta=f"{df['risk_score'].std():.2f} std"
                )
            with col2:
                default_rate = df['default_probability'].mean()
                st.metric("Avg Default Probability", f"{default_rate:.2%}")
            with col3:
                high_risk = (df['risk_category'] == 'High Risk').sum()
                st.metric("High Risk Accounts", f"{high_risk:,}")
            with col4:
                model_auc = risk_model.model_performance.get('auc', 0.75)
                st.metric("Model AUC", f"{model_auc:.3f}")
            
            # Risk Distribution Chart
            fig_risk = px.histogram(
                df, 
                x='default_probability',
                color='risk_category',
                nbins=50,
                title="Risk Distribution Across Portfolio",
                color_discrete_map={
                    'Low Risk': '#2ecc71',
                    'Medium Risk': '#f39c12',
                    'High Risk': '#e74c3c'
                }
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Step 4: Calculate Portfolio Metrics
            st.markdown("---")
            st.subheader("📊 Portfolio Financial Metrics")
            
            analyzer = PortfolioAnalyzer(df)
            metrics = analyzer.calculate_all_metrics()
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 💰 Exposure Metrics")
                st.metric("Total Exposure", f"${metrics['total_exposure']:,.0f}")
                st.metric("Avg Credit Limit", f"${metrics['avg_credit_limit']:,.0f}")
                st.metric("Total Outstanding", f"${metrics['total_outstanding']:,.0f}")
            
            with col2:
                st.markdown("#### 📈 Utilization Metrics")
                st.metric("Avg Utilization Rate", f"{metrics['avg_utilization']:.2%}")
                st.metric("High Utilization %", f"{metrics['high_utilization_pct']:.1%}")
                st.metric("Payment Rate", f"{metrics['payment_rate']:.2%}")
            
            with col3:
                st.markdown("#### ⚠️ Risk Metrics")
                st.metric("Expected Loss", f"${metrics['expected_loss']:,.0f}")
                st.metric("Value at Risk (95%)", f"${metrics['var_95']:,.0f}")
                st.metric("Loss Given Default", f"{metrics['lgd']:.2%}")
            
            # Financial Ratios Table
            with st.expander("📋 Detailed Financial Ratios", expanded=False):
                ratios_df = pd.DataFrame({
                    'Ratio': list(metrics['ratios'].keys()),
                    'Value': list(metrics['ratios'].values())
                })
                st.dataframe(ratios_df, use_container_width=True)
            
            # Step 5: Portfolio Valuation
            st.markdown("---")
            st.subheader("💎 Portfolio Valuation")
            
            valuation = analyzer.calculate_portfolio_value()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Book Value", f"${valuation['book_value']:,.0f}")
            with col2:
                st.metric("Risk-Adjusted Value", f"${valuation['risk_adjusted_value']:,.0f}")
            with col3:
                st.metric("Expected Revenue", f"${valuation['expected_revenue']:,.0f}")
            with col4:
                st.metric("Net Portfolio Value", f"${valuation['net_value']:,.0f}")
            
            # Valuation breakdown
            fig_val = go.Figure(data=[
                go.Bar(
                    x=['Book Value', 'Expected Loss', 'Risk-Adjusted Value'],
                    y=[valuation['book_value'], -valuation['expected_loss'], valuation['risk_adjusted_value']],
                    marker_color=['#3498db', '#e74c3c', '#2ecc71']
                )
            ])
            fig_val.update_layout(title="Portfolio Valuation Breakdown", yaxis_title="Value ($)")
            st.plotly_chart(fig_val, use_container_width=True)
            
            # NEW: Cash Flow-Based Valuation Comparison
            if 'cash_flow_analysis' in valuation:
                st.markdown("---")
                st.subheader("📊 Cash Flow-Based Valuation: Method Comparison")
                
                st.info("""
                **Two valuation approaches calculated:**
                - **Method A (Static Pool Analysis)**: Best for revolving credit like credit cards
                - **Method B (Dynamic Cash Flow Model)**: Best for term loans with fixed payments
                """)
                
                cf_analysis = valuation['cash_flow_analysis']
                method_a = cf_analysis['method_a']
                method_b = cf_analysis['method_b']
                
                # Side-by-side comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Method A: Static Pool Analysis")
                    st.markdown("*Revolving Credit Approach*")
                    st.metric("NPV", f"${method_a['npv']:,.0f}")
                    st.metric("IRR (Annual)", f"{method_a['irr_annual']:.2f}%")
                    st.metric("MOIC", f"{method_a['moic']:.2f}x")
                    st.metric("Payback Period", f"{method_a['payback_period_months']:.1f} months")
                    st.metric("Total Collections (24mo)", f"${method_a['total_collections']:,.0f}")
                    st.metric("Total Losses (24mo)", f"${method_a['total_losses']:,.0f}")
                    
                    with st.expander("ℹ️ About Method A"):
                        st.markdown("""
                        **Static Pool Analysis:**
                        - Segments portfolio by risk cohorts
                        - Projects aggregate behavior per cohort
                        - Standard for credit card portfolios
                        - Fast computation
                        - Good for regulatory reporting
                        """)
                
                with col2:
                    st.markdown("### 📉 Method B: Dynamic Cash Flow Model")
                    st.markdown("*Term Loan Approach*")
                    st.metric("NPV", f"${method_b['npv']:,.0f}")
                    st.metric("IRR (Annual)", f"{method_b['irr_annual']:.2f}%")
                    st.metric("MOIC", f"{method_b['moic']:.2f}x")
                    st.metric("Payback Period", f"{method_b['payback_period_months']:.1f} months")
                    st.metric("Total Collections (24mo)", f"${method_b['total_collections']:,.0f}")
                    st.metric("Total Losses (24mo)", f"${method_b['total_losses']:,.0f}")
                    
                    with st.expander("ℹ️ About Method B"):
                        st.markdown("""
                        **Dynamic Cash Flow Model:**
                        - Models each loan individually
                        - Specific amortization schedules
                        - Standard for term loans
                        - Higher computational cost
                        - Best for M&A valuation
                        """)
                
                # Difference highlights
                st.markdown("### 🔍 Key Differences")
                col1, col2, col3 = st.columns(3)
                with col1:
                    npv_diff = method_a['npv'] - method_b['npv']
                    st.metric(
                        "NPV Difference", 
                        f"${abs(npv_diff):,.0f}",
                        delta=f"{npv_diff/method_b['npv']*100:.1f}% vs Method B" if method_b['npv'] != 0 else "N/A"
                    )
                with col2:
                    irr_diff = method_a['irr_annual'] - method_b['irr_annual']
                    st.metric(
                        "IRR Difference",
                        f"{abs(irr_diff):.2f}pp",
                        delta=f"Method A {'higher' if irr_diff > 0 else 'lower'}"
                    )
                with col3:
                    collections_diff = method_a['total_collections'] - method_b['total_collections']
                    st.metric(
                        "Collections Difference",
                        f"${abs(collections_diff):,.0f}",
                        delta=f"{collections_diff/method_b['total_collections']*100:.1f}% vs Method B" if method_b['total_collections'] != 0 else "N/A"
                    )
                
                # Monthly cash flow comparison chart
                st.markdown("### 📅 Monthly Cash Flow Projection Comparison")
                
                comparison = cf_analysis['comparison']
                monthly_comp = comparison['monthly_comparison']
                
                fig_cf = go.Figure()
                fig_cf.add_trace(go.Scatter(
                    x=monthly_comp['Month'],
                    y=monthly_comp['Static Pool Net CF'],
                    name='Method A (Static Pool)',
                    mode='lines+markers',
                    line=dict(color='#3498db', width=2)
                ))
                fig_cf.add_trace(go.Scatter(
                    x=monthly_comp['Month'],
                    y=monthly_comp['Dynamic CF Net CF'],
                    name='Method B (Dynamic CF)',
                    mode='lines+markers',
                    line=dict(color='#e74c3c', width=2, dash='dash')
                ))
                fig_cf.update_layout(
                    title="Net Cash Flow Projection: Method A vs Method B",
                    xaxis_title="Month",
                    yaxis_title="Net Cash Flow ($)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_cf, use_container_width=True)
                
                # Collections vs Losses comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_coll = go.Figure()
                    fig_coll.add_trace(go.Bar(
                        x=['Method A', 'Method B'],
                        y=[method_a['total_collections'], method_b['total_collections']],
                        marker_color=['#3498db', '#e74c3c']
                    ))
                    fig_coll.update_layout(
                        title="Total Collections (24 months)",
                        yaxis_title="Collections ($)"
                    )
                    st.plotly_chart(fig_coll, use_container_width=True)
                
                with col2:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Bar(
                        x=['Method A', 'Method B'],
                        y=[method_a['total_losses'], method_b['total_losses']],
                        marker_color=['#e74c3c', '#c0392b']
                    ))
                    fig_loss.update_layout(
                        title="Total Losses (24 months)",
                        yaxis_title="Losses ($)"
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                
                # Recommendation
                st.markdown("### 💡 Which Method Should You Use?")
                st.info("""
                **Recommendation based on your portfolio type:**
                - **Credit Cards / Revolving Credit**: Use **Method A (Static Pool)**
                  - Faster computation
                  - Industry standard for credit cards
                  - Better for large portfolios
                  
                - **Term Loans (Auto, Mortgage, Personal)**: Use **Method B (Dynamic CF)**
                  - More precise for individual loans
                  - Better for M&A valuation
                  - Shows loan-specific behavior
                  
                - **Mixed Portfolio**: Review both methods and use the average or conservative estimate
                """)
            
            # Step 6: Risk Segmentation
            st.markdown("---")
            st.subheader("🎯 Risk Segmentation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk category breakdown
                risk_counts = df['risk_category'].value_counts()
                fig_pie = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Portfolio Risk Composition",
                    color=risk_counts.index,
                    color_discrete_map={
                        'Low Risk': '#2ecc71',
                        'Medium Risk': '#f39c12',
                        'High Risk': '#e74c3c'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Risk by credit limit bands
                df['limit_band'] = pd.cut(
                    df['credit_limit'], 
                    bins=[0, 50000, 100000, 200000, 1000000],
                    labels=['<50K', '50-100K', '100-200K', '>200K']
                )
                risk_by_limit = df.groupby('limit_band')['default_probability'].mean().reset_index()
                fig_bar = px.bar(
                    risk_by_limit,
                    x='limit_band',
                    y='default_probability',
                    title="Default Probability by Credit Limit Band",
                    labels={'default_probability': 'Avg Default Prob', 'limit_band': 'Credit Limit'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Step 7: Download Reports
            st.markdown("---")
            st.subheader("📥 Download Reports")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Generate Excel report
                report_gen = ReportGenerator(df, metrics, valuation)
                excel_buffer = report_gen.create_excel_report()
                
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_buffer,
                    file_name=f"credit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # Generate PDF summary
                pdf_buffer = report_gen.create_pdf_report()
                
                st.download_button(
                    label="📄 Download PDF Summary",
                    data=pdf_buffer,
                    file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            
            with col3:
                # Generate scored portfolio CSV
                csv_buffer = BytesIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.download_button(
                    label="📋 Download Scored Portfolio",
                    data=csv_buffer,
                    file_name=f"scored_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # High-risk accounts table
            st.markdown("---")
            st.subheader("⚠️ High Risk Accounts (Top 20)")
            high_risk_df = df.nlargest(20, 'default_probability')[
                ['account_id', 'credit_limit', 'outstanding_balance', 'utilization_rate', 
                 'default_probability', 'risk_category', 'risk_score']
            ].round(4)
            st.dataframe(high_risk_df, use_container_width=True)
            
    else:
        # Show example/instructions when no file uploaded
        st.info("👆 Upload a credit portfolio file to begin analysis")
        
        st.markdown("---")
        st.markdown("### 📝 Supported File Types:")
        st.markdown("""
        - **Credit Card Portfolios**: Account-level data with payment history
        - **Loan Portfolios**: Mortgage, personal, auto loans
        - **Trade Credit**: B2B credit accounts
        - **Mixed Portfolios**: Any credit exposure data
        """)
        
        st.markdown("### 💡 What You'll Get:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Automated Analysis:**
            - Smart file structure detection
            - Data quality assessment
            - Automatic column mapping
            - Missing data handling
            """)
        with col2:
            st.markdown("""
            **Comprehensive Outputs:**
            - ML-based risk scores
            - 20+ financial ratios
            - Portfolio valuation
            - Executive PDF report
            """)


def debt_collection_pricing_app():
    """
    Debt Collection Portfolio Pricing Mode
    Price charged-off debt portfolios using P/C ratio and DCF methodologies
    """
    st.markdown("## 💰 Debt Collection Portfolio Pricing")
    st.markdown("Price charged-off debt portfolios using industry-standard P/C ratio and DCF methodologies")

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Pricing Parameters")

        # Portfolio type
        portfolio_type = st.selectbox(
            "Portfolio Type",
            ['consumer_unsecured', 'consumer_secured', 'auto_loans', 'credit_cards', 'medical'],
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Servicing costs
        servicing_costs = st.slider(
            "Servicing Costs (%)",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
            help="Operating costs as % of gross collections"
        ) / 100

        # Target IRR
        target_irr = st.slider(
            "Target IRR (%)",
            min_value=10,
            max_value=30,
            value=18,
            step=1,
            help="Minimum required annual return"
        ) / 100

        st.markdown("---")
        st.markdown("### 📖 About This Tool")
        st.markdown("""
        **Pricing Methods:**
        - **P/C Ratio**: Market-based pricing
        - **DCF**: Cash flow-based pricing

        **Collection Curves:**
        Standard industry curves by portfolio type

        **IRR Verification:**
        Confirms bid meets target returns
        """)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Single Portfolio",
        "📊 Portfolio Comparison",
        "🔬 Sensitivity Analysis",
        "🚨 Due Diligence"
    ])

    with tab1:
        st.markdown("### Portfolio Pricing")

        # Option to upload portfolio file or enter manually
        input_method = st.radio(
            "Input Method:",
            ["📁 Upload Portfolio File", "✍️ Manual Entry"],
            horizontal=True
        )

        if input_method == "📁 Upload Portfolio File":
            # File upload for automatic calculation
            st.markdown("#### Upload Portfolio Data")
            uploaded_portfolio = st.file_uploader(
                "Upload Portfolio File (CSV/Excel)",
                type=['csv', 'xlsx', 'xls'],
                key="pricing_upload"
            )

            if uploaded_portfolio is not None:
                try:
                    # Load portfolio data
                    if uploaded_portfolio.name.endswith('.csv'):
                        df_portfolio = pd.read_csv(uploaded_portfolio)
                    else:
                        df_portfolio = pd.read_excel(uploaded_portfolio)

                    st.success(f"✅ Loaded {len(df_portfolio):,} accounts")

                    # Auto-detect balance column
                    balance_cols = [col for col in df_portfolio.columns
                                  if any(term in col.lower() for term in ['balance', 'amount', 'principal', 'outstanding'])]

                    if balance_cols:
                        balance_col = st.selectbox("Select Balance Column:", balance_cols, index=0)

                        # Calculate face value from data
                        auto_face_value = df_portfolio[balance_col].sum()

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Calculated Face Value", f"€{auto_face_value:,.0f}")
                            st.caption(f"From {len(df_portfolio):,} accounts")

                        with col2:
                            # Optional: Auto-calculate recovery rate from defaults (if available)
                            default_cols = [col for col in df_portfolio.columns
                                          if 'default' in col.lower() or 'status' in col.lower()]

                            if default_cols:
                                st.write("Default data detected - can estimate recovery rate")

                        # Use auto-calculated value
                        face_value = auto_face_value
                        portfolio_name = uploaded_portfolio.name.replace('.csv', '').replace('.xlsx', '').replace('.xls', '')

                        # Allow user to adjust recovery rate
                        recovery_rate = st.slider(
                            "Recovery Rate (%)",
                            min_value=10,
                            max_value=60,
                            value=30,
                            step=5,
                            help="Expected % of face value to collect",
                            key="file_recovery"
                        ) / 100

                    else:
                        st.error("Could not find balance/amount column in file. Please select manual entry.")
                        face_value = None

                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    face_value = None
            else:
                st.info("👆 Upload a portfolio file to automatically calculate face value")
                face_value = None

        else:
            # Manual entry (original implementation)
            st.markdown("#### Manual Parameters")

            col1, col2, col3 = st.columns(3)

            with col1:
                portfolio_name = st.text_input("Portfolio Name", value="Spain Consumer Portfolio Nov 2025")
                face_value = st.number_input(
                    "Face Value (€)",
                    min_value=100_000,
                    max_value=1_000_000_000,
                    value=50_000_000,
                    step=1_000_000,
                    format="%d"
                )

            with col2:
                recovery_rate = st.slider(
                    "Recovery Rate (%)",
                    min_value=10,
                    max_value=60,
                    value=30,
                    step=5,
                    help="Expected % of face value to collect"
                ) / 100

                # Option to use custom P/C ratio
                use_custom_pc = st.checkbox("Use Custom P/C Ratio")
                if use_custom_pc:
                    custom_pc_ratio = st.slider("Custom P/C Ratio (%)", 20, 80, 50, 5) / 100
                else:
                    custom_pc_ratio = None

            with col3:
                # Option to use custom collection curve
                use_custom_curve = st.checkbox("Use Custom Collection Curve")
                if use_custom_curve:
                    st.markdown("**Collection % by Year:**")
                    year1 = st.slider("Year 1 (%)", 20, 70, 40, 5) / 100
                    year2 = st.slider("Year 2 (%)", 10, 50, 30, 5) / 100
                    year3 = st.slider("Year 3 (%)", 5, 40, 20, 5) / 100
                    year4 = st.slider("Year 4 (%)", 0, 30, 10, 5) / 100

                    total_pct = (year1 + year2 + year3 + year4) * 100
                    if abs(total_pct - 100) > 0.01:
                        st.warning(f"⚠️ Total: {total_pct:.0f}% (should be 100%)")

                    custom_curve = {1: year1, 2: year2, 3: year3, 4: year4}
                else:
                    custom_curve = None

        # Common section for both file upload and manual entry
        # Set defaults for optional parameters if not set by file upload
        if input_method == "📁 Upload Portfolio File":
            custom_pc_ratio = None
            custom_curve = None

        # Calculate button (only show if we have data)
        if face_value is not None and st.button("💰 Calculate Pricing", type="primary", use_container_width=True):
            with st.spinner("Calculating pricing..."):
                # Initialize pricer
                pricer = DebtPortfolioPricer(
                    face_value=face_value,
                    recovery_rate=recovery_rate,
                    portfolio_type=portfolio_type,
                    servicing_costs=servicing_costs,
                    target_irr=target_irr
                )

                # Calculate both methods
                results = pricer.calculate_both_methods(
                    custom_curve=custom_curve,
                    custom_pc_ratio=custom_pc_ratio
                )

                # Display results
                st.markdown("---")
                st.markdown(f"## 📊 Pricing Results: {portfolio_name}")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Face Value", f"€{face_value:,.0f}")
                    st.metric("ERC", f"€{results['portfolio_info']['erc']:,.0f}")

                with col2:
                    st.metric("P/C Method", f"€{results['pc_method']['price']:,.0f}")
                    st.caption(f"{results['pc_method']['price_as_pct_of_face']:.1%} of Face")

                with col3:
                    st.metric("DCF Method", f"€{results['dcf_method']['price']:,.0f}")
                    st.caption(f"{results['dcf_method']['price_as_pct_of_face']:.1%} of Face")

                with col4:
                    margin_value = results['recommendation'].get('margin')
                    st.metric(
                        "RECOMMENDED BID",
                        f"€{results['recommendation']['bid_amount']:,.0f}",
                        delta=f"{margin_value:.1%} vs Target IRR" if margin_value is not None else None
                    )
                    decision_color = "green" if results['recommendation']['decision'] == 'BUY' else "red"
                    st.markdown(f"<h3 style='color: {decision_color};'>{'✓ BUY' if results['recommendation']['decision'] == 'BUY' else '✗ PASS'}</h3>", unsafe_allow_html=True)

                # Detailed results
                st.markdown("---")
                st.markdown("### 📋 Detailed Analysis")

                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:
                    st.markdown("#### P/C Ratio Method")
                    st.write(f"**P/C Ratio:** {results['pc_method']['pc_ratio']:.1%}")
                    st.write(f"**Formula:** Price = ERC × P/C Ratio")
                    st.write(f"**Calculation:** €{results['portfolio_info']['erc']:,.0f} × {results['pc_method']['pc_ratio']:.1%}")
                    st.write(f"**Price:** €{results['pc_method']['price']:,.0f}")

                with detail_col2:
                    st.markdown("#### DCF Method")
                    st.write(f"**Target IRR:** {results['dcf_method']['target_irr']:.1%}")
                    st.write(f"**Collection Curve:** {len(results['dcf_method']['collection_curve'])} years")
                    st.write(f"**Net Present Value:** €{results['dcf_method']['total_pv']:,.0f}")

                # DCF Cash Flow Table
                st.markdown("#### Yearly Cash Flow Analysis")
                st.dataframe(
                    results['dcf_method']['yearly_cashflows'].style.format({
                        'gross_collections': '€{:,.0f}',
                        'servicing_costs': '€{:,.0f}',
                        'net_collections': '€{:,.0f}',
                        'present_value': '€{:,.0f}',
                        'collection_pct': '{:.1%}',
                        'discount_factor': '{:.3f}'
                    }),
                    use_container_width=True
                )

                # IRR Verification
                st.markdown("---")
                st.markdown("### ✅ IRR Verification")
                irr_col1, irr_col2, irr_col3 = st.columns(3)

                with irr_col1:
                    st.metric("Expected IRR", f"{results['recommendation']['expected_irr']:.1%}")

                with irr_col2:
                    st.metric("Target IRR", f"{target_irr:.1%}")

                with irr_col3:
                    margin_value = results['recommendation'].get('margin')
                    st.metric(
                        "Margin",
                        f"{margin_value:+.1%}" if margin_value is not None else "N/A",
                        delta="Above Target" if margin_value and margin_value > 0 else "Below Target"
                    )

                # Monthly Collection Schedule
                st.markdown("---")
                st.markdown("### 📅 Monthly Collection Schedule")

                monthly_schedule = pricer.create_monthly_schedule(custom_curve)

                # Plot cumulative collections
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=monthly_schedule['month'],
                    y=monthly_schedule['cumulative_net'],
                    mode='lines',
                    name='Cumulative Collections',
                    fill='tozeroy'
                ))
                fig.update_layout(
                    title='Cumulative Net Collections Over Time',
                    xaxis_title='Month',
                    yaxis_title='Cumulative Collections (€)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show first 12 months
                st.markdown("**First 12 Months:**")
                st.dataframe(
                    monthly_schedule.head(12).style.format({
                        'gross_collections': '€{:,.0f}',
                        'servicing_costs': '€{:,.0f}',
                        'net_collections': '€{:,.0f}',
                        'cumulative_net': '€{:,.0f}',
                        'cumulative_pct_of_erc': '{:.1%}'
                    }),
                    use_container_width=True
                )

    with tab2:
        st.markdown("### Compare Multiple Portfolios")
        st.markdown("Add multiple portfolios to compare pricing side-by-side")

        num_portfolios = st.number_input("Number of Portfolios", 2, 5, 2)

        portfolios = []
        for i in range(num_portfolios):
            st.markdown(f"#### Portfolio {i+1}")
            col1, col2, col3 = st.columns(3)

            with col1:
                name = st.text_input(f"Name", value=f"Portfolio {i+1}", key=f"name_{i}")
                face = st.number_input(f"Face Value (€)", min_value=100_000, value=10_000_000 * (i+1), key=f"face_{i}", format="%d")

            with col2:
                recovery = st.slider(f"Recovery %", 10, 60, 30-i*5, 5, key=f"recovery_{i}") / 100

            with col3:
                ptype = st.selectbox(
                    f"Type",
                    ['consumer_unsecured', 'consumer_secured', 'credit_cards'],
                    key=f"type_{i}",
                    format_func=lambda x: x.replace('_', ' ').title()
                )

            portfolios.append({
                'name': name,
                'face_value': face,
                'recovery_rate': recovery,
                'portfolio_type': ptype,
                'servicing_costs': servicing_costs,
                'target_irr': target_irr
            })

        if st.button("Compare Portfolios", type="primary"):
            comparison_df = compare_portfolios(portfolios)

            st.markdown("### Comparison Results")
            st.dataframe(
                comparison_df.style.format({
                    'Face Value': '€{:,.0f}',
                    'Recovery Rate': '{:.1%}',
                    'ERC': '€{:,.0f}',
                    'P/C Price': '€{:,.0f}',
                    'DCF Price': '€{:,.0f}',
                    'Recommended Bid': '€{:,.0f}',
                    'Expected IRR': '{:.1%}'
                }),
                use_container_width=True
            )

            # Bar chart comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(name='P/C Price', x=comparison_df['Portfolio'], y=comparison_df['P/C Price']))
            fig.add_trace(go.Bar(name='DCF Price', x=comparison_df['Portfolio'], y=comparison_df['DCF Price']))
            fig.add_trace(go.Bar(name='Recommended', x=comparison_df['Portfolio'], y=comparison_df['Recommended Bid']))
            fig.update_layout(title='Pricing Comparison', barmode='group')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Sensitivity Analysis")
        st.markdown("Test how pricing changes with different recovery rates and target IRRs")

        sens_face = st.number_input("Face Value (€)", min_value=1_000_000, value=50_000_000, key="sens_face", format="%d")

        col1, col2 = st.columns(2)
        with col1:
            recovery_min = st.slider("Min Recovery %", 10, 50, 20) / 100
            recovery_max = st.slider("Max Recovery %", 20, 60, 40) / 100

        with col2:
            irr_min = st.slider("Min Target IRR %", 5, 25, 12) / 100
            irr_max = st.slider("Max Target IRR %", 10, 30, 24) / 100

        steps = st.slider("Analysis Steps", 3, 7, 5)

        if st.button("Run Sensitivity Analysis", type="primary"):
            pricer = DebtPortfolioPricer(
                face_value=sens_face,
                recovery_rate=0.30,  # Base case
                portfolio_type=portfolio_type,
                servicing_costs=servicing_costs,
                target_irr=target_irr
            )

            sens_df = pricer.sensitivity_analysis(
                recovery_range=(recovery_min, recovery_max),
                irr_range=(irr_min, irr_max),
                steps=steps
            )

            # Pivot for heatmap
            pivot_data = sens_df.pivot(
                index='target_irr',
                columns='recovery_rate',
                values='recommended_bid'
            )

            # Heatmap
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=[f"{r:.1%}" for r in pivot_data.columns],
                y=[f"{i:.1%}" for i in pivot_data.index],
                colorscale='RdYlGn',
                text=pivot_data.values,
                texttemplate='€%{text:,.0f}',
                textfont={"size": 10},
                colorbar=dict(title="Bid Amount (€)")
            ))
            fig.update_layout(
                title='Recommended Bid Sensitivity',
                xaxis_title='Recovery Rate',
                yaxis_title='Target IRR',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Data table
            st.markdown("### Detailed Results")
            st.dataframe(
                sens_df.style.format({
                    'recovery_rate': '{:.1%}',
                    'target_irr': '{:.1%}',
                    'erc': '€{:,.0f}',
                    'pc_price': '€{:,.0f}',
                    'dcf_price': '€{:,.0f}',
                    'recommended_bid': '€{:,.0f}',
                    'expected_irr': '{:.1%}'
                }),
                use_container_width=True
            )

    with tab4:
        st.markdown("### Portfolio Due Diligence")
        st.markdown("Upload a portfolio file to analyze composition and detect red flags")

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Portfolio Data (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            key="dd_upload"
        )

        if uploaded_file is not None:
            # Load data
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_portfolio = pd.read_csv(uploaded_file)
                else:
                    df_portfolio = pd.read_excel(uploaded_file)

                st.success(f"✅ Loaded {len(df_portfolio):,} accounts")

                # Show preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df_portfolio.head(10), use_container_width=True)

                # Configuration
                col1, col2, col3 = st.columns(3)

                with col1:
                    base_recovery = st.slider(
                        "Base Recovery Rate (%)",
                        min_value=10,
                        max_value=60,
                        value=30,
                        step=5,
                        key="dd_recovery"
                    ) / 100

                with col2:
                    statute_yrs = st.number_input(
                        "Statute of Limitations (years)",
                        min_value=3,
                        max_value=15,
                        value=5,
                        key="dd_statute"
                    )

                with col3:
                    st.write("")  # Spacing
                    analyze_btn = st.button(
                        "🔍 Run Due Diligence",
                        type="primary",
                        use_container_width=True
                    )

                if analyze_btn:
                    with st.spinner("Analyzing portfolio..."):
                        # Run due diligence
                        dd = PortfolioDueDiligence(
                            df_portfolio,
                            base_recovery_rate=base_recovery,
                            statute_years=statute_yrs
                        )

                        report = dd.generate_due_diligence_report()

                        # Executive Summary
                        st.markdown("---")
                        st.markdown("## 📊 Executive Summary")

                        summary = report['summary']

                        # Alert boxes
                        if summary['critical_issues_count'] > 0:
                            st.error(f"🚨 {summary['critical_issues_count']} CRITICAL ISSUES FOUND")
                            for issue in summary['critical_issues']:
                                st.write(f"- {issue.replace('_', ' ').title()}")

                        if summary['warnings_count'] > 0:
                            st.warning(f"⚠️ {summary['warnings_count']} WARNINGS")
                            for warning in summary['warnings']:
                                st.write(f"- {warning.replace('_', ' ').title()}")

                        if summary['critical_issues_count'] == 0 and summary['warnings_count'] == 0:
                            st.success("✅ No major issues detected")

                        # Recommendation
                        st.info(f"**Recommendation:** {summary['recommendation']}")

                        # ERC Impact
                        adjusted = report['adjusted_erc']

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Base ERC",
                                f"€{adjusted['base_erc']:,.0f}",
                                delta=f"{adjusted['base_recovery_rate']:.1%} recovery"
                            )

                        with col2:
                            st.metric(
                                "Total Adjustments",
                                f"€{adjusted['total_adjustment']:,.0f}",
                                delta=f"{summary['erc_impact_pct']:.1f}%" if summary['erc_impact_pct'] != 0 else None
                            )

                        with col3:
                            st.metric(
                                "Adjusted ERC",
                                f"€{adjusted['adjusted_erc']:,.0f}",
                                delta=f"{adjusted['adjusted_recovery_rate']:.1%} recovery"
                            )

                        # Detailed Adjustments
                        if not adjusted['adjustment_summary'].empty:
                            st.markdown("---")
                            st.markdown("### 💰 ERC Adjustments")

                            st.dataframe(
                                adjusted['adjustment_summary'].style.format({
                                    'affected_value': '€{:,.0f}',
                                    'adjustment': '€{:,.0f}',
                                    'new_recovery_rate': '{:.1%}'
                                }),
                                use_container_width=True
                            )

                        # Portfolio Composition
                        st.markdown("---")
                        st.markdown("### 📊 Portfolio Composition")

                        composition = report['composition']

                        # By balance size
                        if 'by_balance_size' in composition and not composition['by_balance_size'].empty:
                            st.markdown("#### By Balance Size")
                            st.dataframe(
                                composition['by_balance_size'].style.format({
                                    'Total_Balance': '€{:,.0f}',
                                    'Count': '{:,.0f}',
                                    'Pct': '{:.1f}%'
                                }),
                                use_container_width=True
                            )

                            # Pie chart
                            fig = px.pie(
                                composition['by_balance_size'].reset_index(),
                                values='Total_Balance',
                                names='Balance_Segment',
                                title='Portfolio Value Distribution'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # By age (if available)
                        if 'by_age' in composition and not composition['by_age'].empty:
                            st.markdown("#### By Age of Debt")
                            st.dataframe(
                                composition['by_age'].style.format({
                                    'Total_Balance': '€{:,.0f}',
                                    'Count': '{:,.0f}',
                                    'Pct': '{:.1f}%'
                                }),
                                use_container_width=True
                            )

                        # Red Flags Detail
                        st.markdown("---")
                        st.markdown("### 🚨 Red Flags Analysis")

                        red_flags = report['red_flags']

                        # Statute Issues
                        statute = red_flags['statute_issues']
                        st.markdown("#### ⚖️ Statute of Limitations")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Past Statute (Unrecoverable)",
                                f"{statute['past_statute_count']:,} accounts",
                                delta=f"€{statute['past_statute_value']:,.0f}"
                            )

                        with col2:
                            st.metric(
                                "Near Statute (Risky)",
                                f"{statute['near_statute_count']:,} accounts",
                                delta=f"€{statute['near_statute_value']:,.0f}"
                            )

                        if statute['status'] == 'critical':
                            st.error(f"🚨 CRITICAL: {statute['past_statute_pct']:.1f}% of portfolio is past statute!")
                        elif statute['status'] == 'warning':
                            st.warning(f"⚠️ WARNING: {statute['near_statute_pct']:.1f}% near statute limit")
                        else:
                            st.success("✅ Statute of limitations not a major issue")

                        # Missing Contacts
                        contacts = red_flags['missing_contacts']
                        st.markdown("#### 📞 Contact Information")

                        if contacts['status'] != 'unknown':
                            st.metric(
                                "Missing Contact Info",
                                f"{contacts['count']:,} accounts ({contacts['pct']:.1f}%)",
                                delta=f"€{contacts['value']:,.0f}"
                            )

                            if contacts['status'] == 'critical':
                                st.error("🚨 CRITICAL: >20% missing contact information!")
                            elif contacts['status'] == 'warning':
                                st.warning("⚠️ WARNING: Significant missing contact data")
                            else:
                                st.success("✅ Contact information mostly complete")

                        # Concentration Risk
                        concentration = red_flags['concentration']
                        st.markdown("#### 🎯 Concentration Risk")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Top 10 Accounts",
                                f"{concentration['top_10_pct']:.1f}% of portfolio",
                                delta=f"€{concentration['top_10_value']:,.0f}"
                            )

                        with col2:
                            st.metric(
                                "Largest Account",
                                f"{concentration['top_1_pct']:.1f}% of portfolio",
                                delta=f"€{concentration['top_1_value']:,.0f}"
                            )

                        if concentration['status'] == 'warning':
                            st.warning("⚠️ WARNING: High concentration risk detected")
                        else:
                            st.success("✅ Good diversification")

                        # Small Balances
                        small = red_flags['small_balances']
                        st.markdown("#### 💸 Small Balances (<€500)")

                        st.metric(
                            "Small Balance Accounts",
                            f"{small['count']:,} accounts ({small['pct']:.1f}%)",
                            delta=f"€{small['value']:,.0f} ({small['value_pct']:.1f}% of value)"
                        )

                        if small['status'] == 'warning':
                            st.warning("⚠️ WARNING: High proportion of small balances (collection costs may exceed recovery)")
                        else:
                            st.success("✅ Small balances not a major concern")

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.write("Please ensure your file has a 'Current_Balance' or 'Balance' column")


if __name__ == "__main__":
    main()
