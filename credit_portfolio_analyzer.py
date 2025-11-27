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

def main():
    st.title("🏦 Automated Credit Portfolio Analyzer")
    st.markdown("Upload your credit portfolio file and get instant analysis, risk scoring, and valuation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Quick Analysis", "Deep Analysis", "Custom"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 What This Tool Does:")
        st.markdown("""
        1. **Auto-detects** file structure
        2. **ML Risk Scoring** for each account
        3. **Financial Ratios** calculation
        4. **Portfolio Valuation**
        5. **Risk Metrics** (VaR, Expected Loss)
        6. **Downloadable Reports**
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Credit Portfolio File",
        type=['csv', 'xlsx', 'xls'],
        help="Upload Excel or CSV file containing credit accounts (up to 1GB supported)"
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
            parser = CreditFileParser()
            data_info = parser.analyze_file(uploaded_file)
            
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
                
                st.markdown("**Detected Columns:**")
                st.dataframe(data_info['column_mapping'], width='stretch')
            
            # Step 2: Load and process data
            df = parser.load_and_standardize(uploaded_file)
            
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

if __name__ == "__main__":
    main()
