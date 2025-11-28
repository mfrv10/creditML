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


def create_debt_pricing_pdf(
    portfolio_name: str,
    face_value: float,
    recovery_rate: float,
    results: dict,
    monthly_schedule: pd.DataFrame,
    portfolio_type: str,
    servicing_costs: float,
    target_irr: float,
    composition: dict = None,
    red_flags: dict = None,
    erc_analysis: dict = None
) -> BytesIO:
    """
    Generate comprehensive PDF report for debt portfolio pricing.

    Includes: Portfolio summary, risk analysis, pricing results, cash flow projections, and recommendations.
    """
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                     Spacer, PageBreak, Image)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.pdfgen import canvas
    import plotly.graph_objects as go
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)

    # Container for PDF elements
    elements = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2ca02c'),
        spaceAfter=12,
        spaceBefore=12
    )

    # Title Page
    elements.append(Paragraph("DEBT PORTFOLIO PRICING REPORT", title_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(f"<b>Portfolio:</b> {portfolio_name}", styles['Normal']))
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    elements.append(Spacer(1, 0.5*inch))

    # Executive Summary
    elements.append(Paragraph("EXECUTIVE SUMMARY", heading_style))

    decision = results['recommendation']['decision']
    decision_color = 'green' if decision == 'BUY' else 'red'
    elements.append(Paragraph(
        f"<b>RECOMMENDATION: <font color='{decision_color}'>{decision}</font></b>",
        styles['Normal']
    ))
    elements.append(Spacer(1, 0.2*inch))

    # Summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Face Value', f"€{face_value:,.0f}"],
        ['Recovery Rate', f"{recovery_rate:.1%}"],
        ['Expected Collections (ERC)', f"€{results['portfolio_info']['erc']:,.0f}"],
        ['', ''],
        ['P/C Method Price', f"€{results['pc_method']['price']:,.0f}"],
        ['DCF Method Price', f"€{results['dcf_method']['price']:,.0f}"],
        ['RECOMMENDED BID', f"€{results['recommendation']['bid_amount']:,.0f}"],
        ['Expected IRR', f"{results['recommendation']['expected_irr']:.1%}"],
        ['Target IRR', f"{target_irr:.1%}"],
        ['Margin vs Target', f"{results['recommendation'].get('margin', 0):.1%}"]
    ]

    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 7), (-1, 7), 'Helvetica-Bold'),  # Highlight recommended bid
        ('BACKGROUND', (0, 7), (-1, 7), colors.HexColor('#90EE90')),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 0.3*inch))

    # Risk Analysis (if available)
    if composition and red_flags and erc_analysis:
        elements.append(PageBreak())
        elements.append(Paragraph("RISK ANALYSIS", heading_style))

        # Portfolio composition
        elements.append(Paragraph("<b>Portfolio Composition:</b>", styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))

        comp_data = [
            ['Metric', 'Value'],
            ['Total Accounts', f"{composition['total_accounts']:,}"],
            ['Total Face Value', f"€{composition['total_face_value']:,.0f}"],
            ['Average Balance', f"€{composition['total_face_value']/composition['total_accounts']:,.2f}"]
        ]

        comp_table = Table(comp_data, colWidths=[3*inch, 2*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(comp_table)
        elements.append(Spacer(1, 0.2*inch))

        # Red flags
        elements.append(Paragraph("<b>Red Flags Detected:</b>", styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))

        red_flag_text = []
        if red_flags['statute_issues']['past_statute_count'] > 0:
            red_flag_text.append(f"• Statute-Barred Debt: {red_flags['statute_issues']['past_statute_count']:,} accounts (€{red_flags['statute_issues']['past_statute_value']:,.0f})")
        if red_flags['missing_contacts']['count'] > 0:
            red_flag_text.append(f"• Missing Contacts: {red_flags['missing_contacts']['count']:,} accounts ({red_flags['missing_contacts']['pct']:.1f}%)")
        if red_flags['small_balances']['count'] > 0:
            red_flag_text.append(f"• Small Balances: {red_flags['small_balances']['count']:,} accounts (<€500)")
        if red_flags['concentration']['top_10_pct'] > 20:
            red_flag_text.append(f"• Concentration Risk: Top 10 accounts = {red_flags['concentration']['top_10_pct']:.1f}%")

        if red_flag_text:
            for flag in red_flag_text:
                elements.append(Paragraph(flag, styles['Normal']))
        else:
            elements.append(Paragraph("✓ No major red flags detected", styles['Normal']))

        elements.append(Spacer(1, 0.2*inch))

        # ERC adjustment
        elements.append(Paragraph(f"<b>ERC Adjustment:</b> {erc_analysis['base_recovery_rate']:.1%} → {erc_analysis['adjusted_recovery_rate']:.1%} (Impact: €{erc_analysis['total_adjustment']:,.0f})", styles['Normal']))

    # Pricing Methods Detail
    elements.append(PageBreak())
    elements.append(Paragraph("PRICING METHODOLOGY", heading_style))

    # P/C Ratio Method
    elements.append(Paragraph("<b>1. P/C Ratio Method:</b>", styles['Normal']))
    elements.append(Paragraph(f"   • P/C Ratio: {results['pc_method']['pc_ratio']:.1%}", styles['Normal']))
    elements.append(Paragraph(f"   • Formula: Price = ERC × P/C Ratio", styles['Normal']))
    elements.append(Paragraph(f"   • Calculation: €{results['portfolio_info']['erc']:,.0f} × {results['pc_method']['pc_ratio']:.1%} = €{results['pc_method']['price']:,.0f}", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    # DCF Method
    elements.append(Paragraph("<b>2. DCF Method:</b>", styles['Normal']))
    elements.append(Paragraph(f"   • Target IRR: {results['dcf_method']['target_irr']:.1%}", styles['Normal']))
    elements.append(Paragraph(f"   • Collection Period: {len(results['dcf_method']['collection_curve'])} years", styles['Normal']))
    elements.append(Paragraph(f"   • Net Present Value: €{results['dcf_method']['total_pv']:,.0f}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    # Cash Flow Projections
    elements.append(Paragraph("CASH FLOW PROJECTIONS", heading_style))

    # Yearly cashflow table
    yearly_cf = results['dcf_method']['yearly_cashflows']
    cf_data = [['Year', 'Collections', 'Costs', 'Net CF', 'PV']]
    for _, row in yearly_cf.iterrows():
        cf_data.append([
            f"Year {int(row['year'])}",
            f"€{row['gross_collections']:,.0f}",
            f"€{row['servicing_costs']:,.0f}",
            f"€{row['net_collections']:,.0f}",
            f"€{row['present_value']:,.0f}"
        ])

    cf_table = Table(cf_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    cf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
    ]))

    elements.append(cf_table)
    elements.append(Spacer(1, 0.3*inch))

    # Monthly summary (first 12 months)
    elements.append(Paragraph("<b>First 12 Months Detail:</b>", styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))

    monthly_data = [['Month', 'Collections', 'Cumulative']]
    for _, row in monthly_schedule.head(12).iterrows():
        monthly_data.append([
            f"Month {int(row['month'])}",
            f"€{row['net_collections']:,.0f}",
            f"€{row['cumulative_net']:,.0f}"
        ])

    monthly_table = Table(monthly_data, colWidths=[1.5*inch, 2*inch, 2*inch])
    monthly_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))

    elements.append(monthly_table)

    # Final Recommendation
    elements.append(PageBreak())
    elements.append(Paragraph("FINAL RECOMMENDATION", heading_style))

    recommendation_text = f"""
    Based on the comprehensive analysis of the {portfolio_name} portfolio:

    • <b>Face Value:</b> €{face_value:,.0f}
    • <b>Expected Collections:</b> €{results['portfolio_info']['erc']:,.0f} ({recovery_rate:.1%} recovery rate)
    • <b>Portfolio Type:</b> {portfolio_type.replace('_', ' ').title()}
    • <b>Servicing Costs:</b> {servicing_costs:.0%} of collections

    <b>Pricing Analysis:</b>
    • P/C Method suggests: €{results['pc_method']['price']:,.0f}
    • DCF Method suggests: €{results['dcf_method']['price']:,.0f}
    • Recommended Bid: €{results['recommendation']['bid_amount']:,.0f}

    <b>Return Analysis:</b>
    • Expected IRR: {results['recommendation']['expected_irr']:.1%}
    • Target IRR: {target_irr:.1%}
    • Margin: {results['recommendation'].get('margin', 0):+.1%}

    <b>Decision: <font color='{decision_color}'>{decision}</font></b>

    {f"This portfolio MEETS the target IRR requirements and is recommended for acquisition at the suggested bid price."
     if decision == 'BUY' else
     "This portfolio DOES NOT meet the target IRR requirements at the suggested pricing. Consider passing or renegotiating terms."}
    """

    elements.append(Paragraph(recommendation_text, styles['Normal']))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


def main():
    # Page configuration
    st.set_page_config(page_title="Credit Portfolio Analyzer", layout="wide", initial_sidebar_state="expanded")

    # Header
    st.title("🏦 Credit Portfolio Analyzer")
    st.markdown("### ML-Driven Portfolio Analysis & Debt Pricing")
    st.markdown("---")

    # Initialize session state for persistent data
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None
    if 'portfolio_name' not in st.session_state:
        st.session_state.portfolio_name = None
    if 'data_info' not in st.session_state:
        st.session_state.data_info = None
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    # Progress indicator
    if st.session_state.uploaded_df is None:
        st.info("👉 **Step 1:** Upload your portfolio file to begin")
    elif not st.session_state.analysis_complete:
        st.info("👉 **Step 2:** ML analysis in progress...")
    else:
        st.success("✅ **Ready:** Portfolio analyzed - Choose your analysis path below")

    st.markdown("---")

    # Sidebar settings (always visible)
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

        # Manual mapping option
        enable_manual_mapping = st.checkbox(
            "Enable Manual Column Mapping",
            value=False,
            help="Manually map columns if auto-detection fails"
        )

        st.markdown("---")
        st.markdown("### 📊 ML-Driven Analysis:")
        st.markdown("""
        **Step 1:** Upload & Parse
        **Step 2:** ML Risk Scoring 🎯
        **Step 3:** Choose Analysis Path

        **Features:**
        - AI Document Parsing (Gemini)
        - ML Default Prediction
        - Risk-Adjusted Pricing
        - Due Diligence Analysis
        - Professional Reports
        """)

    # STEP 1: File Upload
    st.markdown("## 📁 STEP 1: Upload Portfolio")

    file_types = ['csv', 'xlsx', 'xls']
    if use_gemini:
        file_types.extend(['pdf', 'png', 'jpg', 'jpeg'])

    uploaded_file = st.file_uploader(
        "Upload your credit portfolio file",
        type=file_types,
        help="Supports CSV, Excel, PDF, and images (with Gemini AI)",
        key="main_file_upload"
    )

    # Process uploaded file once
    if uploaded_file is not None:
        # Check if this is a new file
        file_changed = (st.session_state.portfolio_name != uploaded_file.name)

        if file_changed or st.session_state.uploaded_df is None:
            with st.spinner("🔍 Processing your file..."):
                # Parse and understand the file
                parser = CreditFileParser(use_gemini=use_gemini)
                data_info = parser.analyze_file(uploaded_file)

                is_gemini_parsed = data_info.get('parser') == 'gemini'

                st.success(f"✅ File loaded: **{data_info['file_type']}** with **{data_info['num_records']:,}** records")

                # Manual column mapping if needed
                manual_mapping = None
                if enable_manual_mapping and not is_gemini_parsed:
                    column_mapper = ColumnMapper()

                    if hasattr(uploaded_file, 'seek'):
                        uploaded_file.seek(0)

                    if uploaded_file.name.endswith('.csv'):
                        df_preview = pd.read_csv(uploaded_file, nrows=5)
                    else:
                        df_preview = pd.read_excel(uploaded_file, nrows=5)

                    if hasattr(uploaded_file, 'seek'):
                        uploaded_file.seek(0)

                    auto_mapping_data = data_info.get('column_mapping', pd.DataFrame())
                    auto_mapping = {}
                    if not auto_mapping_data.empty:
                        for _, row in auto_mapping_data.iterrows():
                            if row['Confidence'] > 60:
                                auto_mapping[row['Mapped To']] = row['Original Column']

                    manual_mapping, mapping_ready = column_mapper.render_mapping_ui(
                        df_preview,
                        auto_mapping=auto_mapping
                    )

                    if not mapping_ready:
                        st.warning("⚠️ Please complete column mapping before proceeding")
                        st.stop()

                # Load and process data
                df = parser.load_and_standardize(uploaded_file, manual_mapping=manual_mapping)

                # Store in session state
                st.session_state.uploaded_df = df
                st.session_state.portfolio_name = uploaded_file.name
                st.session_state.data_info = data_info

                st.success("✅ Portfolio data ready - Select analysis mode below")

    # Show current file status
    if st.session_state.uploaded_df is not None:
        st.markdown("## ✅ Portfolio Loaded")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 File", st.session_state.portfolio_name[:20] + "..." if len(st.session_state.portfolio_name) > 20 else st.session_state.portfolio_name)
        with col2:
            st.metric("📊 Accounts", f"{len(st.session_state.uploaded_df):,}")
        with col3:
            st.metric("🔧 Fields", f"{len(st.session_state.uploaded_df.columns)}")
        with col4:
            total_balance = st.session_state.uploaded_df.get('balance', st.session_state.uploaded_df.get('credit_limit', pd.Series([0]))).sum()
            st.metric("💰 Total", f"${total_balance:,.0f}")

        st.markdown("---")

        # STEP 2: Choose Analysis Path
        st.markdown("## 🎯 STEP 2: Choose Analysis Path")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📊 Active Portfolio Analysis
            **For managing your current portfolio**

            ✓ ML Risk Scoring for each account
            ✓ Default Probability Predictions
            ✓ Expected Loss Analysis
            ✓ Portfolio Health Metrics
            ✓ Risk Dashboard & Reports

            *Best for: Active loan portfolio management*
            """)
            if st.button("📊 Analyze Active Portfolio", type="primary", use_container_width=True):
                mode = "📊 Active Portfolio Analysis"

        with col2:
            st.markdown("""
            ### 💰 Debt Collection Pricing
            **For pricing charged-off debt portfolios**

            ✓ ML-Driven Due Diligence
            ✓ Risk-Adjusted Pricing
            ✓ P/C Ratio & DCF Methods
            ✓ Cash Flow Projections
            ✓ Professional PDF Reports

            *Best for: Purchasing NPL portfolios*
            """)
            if st.button("💰 Price Debt Portfolio", type="primary", use_container_width=True):
                mode = "💰 Debt Collection Pricing"

        # Traditional mode selector (for navigation)
        st.markdown("---")
        mode = st.radio(
            "Or use quick selector:",
            ["📊 Active Portfolio Analysis", "💰 Debt Collection Pricing"],
            horizontal=True,
            label_visibility="collapsed"
        )

    else:
        st.info("👆 Upload a credit portfolio file to begin")
        st.stop()

    if mode == "💰 Debt Collection Pricing":
        debt_collection_pricing_app(
            df=st.session_state.uploaded_df,
            portfolio_name=st.session_state.portfolio_name
        )
        return

    # Active Portfolio Analysis
    st.markdown("---")
    st.markdown("## 📊 Active Portfolio Analysis")
    st.markdown("### 🎯 ML-Driven Risk Assessment & Portfolio Management")

    # Use data from session state
    df = st.session_state.uploaded_df
    data_info = st.session_state.data_info

    # STEP 3: ML Risk Scoring (PROMINENT SECTION)
    st.markdown("---")
    st.markdown("## 🤖 STEP 3: ML Risk Analysis")

    with st.spinner("🔄 Running ML credit risk model..."):
        risk_model = CreditRiskModel()
        df = risk_model.score_portfolio(df)

    st.success("✅ ML Risk Scoring Complete")

    # Key ML Metrics - Large and Prominent
    st.markdown("### 📊 ML Risk Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Portfolio Risk Score",
            f"{df['risk_score'].mean():.2f}",
            delta=f"±{df['risk_score'].std():.2f} std",
            help="Average ML-predicted risk score across all accounts"
        )
    with col2:
        default_rate = df['default_probability'].mean()
        st.metric(
            "Avg Default Probability",
            f"{default_rate:.2%}",
            help="ML-predicted likelihood of default"
        )
    with col3:
        high_risk = (df['risk_category'] == 'High Risk').sum()
        high_risk_pct = (high_risk / len(df)) * 100
        st.metric(
            "High Risk Accounts",
            f"{high_risk:,}",
            delta=f"{high_risk_pct:.1f}%",
            help="Accounts flagged as high default risk by ML model"
        )
    with col4:
        model_auc = risk_model.model_performance.get('auc', 0.75)
        st.metric(
            "Model Accuracy (AUC)",
            f"{model_auc:.3f}",
            help="Model performance metric (0.5=random, 1.0=perfect)"
        )

    # Risk Distribution Visualization - Prominent
    st.markdown("#### 📈 Risk Distribution Across Portfolio")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Histogram of default probabilities
        fig_risk = px.histogram(
            df,
            x='default_probability',
            color='risk_category',
            nbins=50,
            title="Default Probability Distribution",
            color_discrete_map={
                'Low Risk': '#2ecc71',
                'Medium Risk': '#f39c12',
                'High Risk': '#e74c3c'
            },
            labels={'default_probability': 'Default Probability', 'count': 'Number of Accounts'}
        )
        fig_risk.update_layout(height=400)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_chart2:
        # Pie chart of risk categories
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
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # STEP 4: Portfolio Health & Financial Metrics
    st.markdown("---")
    st.markdown("## 💰 STEP 4: Portfolio Health & Financial Metrics")

    with st.spinner("📊 Calculating portfolio metrics..."):
        analyzer = PortfolioAnalyzer(df)
        metrics = analyzer.calculate_all_metrics()

    # Portfolio Health Score (Prominent)
    st.markdown("### 🏥 Portfolio Health Score")

    # Calculate custom health metrics
    expected_loss = metrics['expected_loss']
    total_outstanding = metrics['total_outstanding']
    loss_rate = (expected_loss / total_outstanding) if total_outstanding > 0 else 0
    recovery_rate = 1 - loss_rate

    # Overall health score (0-100)
    health_score = (
        (recovery_rate * 40) +  # Recovery rate worth 40 points
        ((1 - default_rate) * 30) +  # Low default probability worth 30 points
        (min(metrics['payment_rate'], 1.0) * 20) +  # Payment rate worth 20 points
        ((1 - min(loss_rate, 1.0)) * 10)  # Low loss rate worth 10 points
    ) * 100

    health_color = "🟢" if health_score >= 70 else "🟡" if health_score >= 50 else "🔴"
    health_status = "Healthy" if health_score >= 70 else "Moderate" if health_score >= 50 else "At Risk"

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            f"{health_color} Health Score",
            f"{health_score:.1f}/100",
            delta=health_status,
            help="Overall portfolio health based on multiple risk factors"
        )

    with col2:
        st.metric(
            "Expected Recovery",
            f"{recovery_rate:.2%}",
            delta=f"${total_outstanding * recovery_rate:,.0f}",
            help="Estimated recovery rate based on ML predictions"
        )

    with col3:
        st.metric(
            "Expected Loss Rate",
            f"{loss_rate:.2%}",
            delta=f"-${expected_loss:,.0f}" if expected_loss > 0 else "$0",
            delta_color="inverse",
            help="Percentage of portfolio expected to default"
        )

    with col4:
        expected_collections = total_outstanding * recovery_rate
        st.metric(
            "Expected Collections",
            f"${expected_collections:,.0f}",
            delta=f"{(expected_collections/total_outstanding)*100:.1f}% of outstanding",
            help="ML-estimated total collections"
        )

    with col5:
        healthy_accounts = len(df[df['risk_category'] == 'Low Risk'])
        healthy_pct = (healthy_accounts / len(df)) * 100
        st.metric(
            "Healthy Accounts",
            f"{healthy_accounts:,}",
            delta=f"{healthy_pct:.1f}%",
            help="Low risk accounts with high recovery probability"
        )

    # Detailed Metrics in Expanders
    with st.expander("💰 Exposure & Balance Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Exposure", f"${metrics['total_exposure']:,.0f}")
            st.metric("Total Outstanding", f"${metrics['total_outstanding']:,.0f}")

        with col2:
            st.metric("Avg Credit Limit", f"${metrics['avg_credit_limit']:,.0f}")
            st.metric("Avg Utilization Rate", f"{metrics['avg_utilization']:.2%}")

        with col3:
            st.metric("High Utilization %", f"{metrics['high_utilization_pct']:.1%}")
            st.metric("Payment Rate", f"{metrics['payment_rate']:.2%}")

    with st.expander("⚠️ Risk Metrics & Loss Estimates", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Expected Loss", f"${metrics['expected_loss']:,.0f}")
            st.metric("Loss Given Default", f"{metrics['lgd']:.2%}")

        with col2:
            st.metric("Value at Risk (95%)", f"${metrics['var_95']:,.0f}")
            st.metric("Default Probability", f"{default_rate:.2%}")

        with col3:
            st.metric("Risk-Adjusted Value", f"${metrics['total_outstanding'] - expected_loss:,.0f}")

    with st.expander("📋 Detailed Financial Ratios", expanded=False):
        ratios_df = pd.DataFrame({
            'Ratio': list(metrics['ratios'].keys()),
            'Value': list(metrics['ratios'].values())
        })
        st.dataframe(ratios_df, use_container_width=True)

    # STEP 5: ML-Based Risk Segmentation & Analysis
    st.markdown("---")
    st.markdown("## 🎯 STEP 5: ML-Based Risk Segmentation")

    with st.expander("💎 Portfolio Valuation Summary", expanded=False):
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

        # Valuation breakdown chart
        fig_val = go.Figure(data=[
            go.Bar(
                x=['Book Value', 'Expected Loss', 'Risk-Adjusted Value'],
                y=[valuation['book_value'], -valuation['expected_loss'], valuation['risk_adjusted_value']],
                marker_color=['#3498db', '#e74c3c', '#2ecc71']
            )
        ])
        fig_val.update_layout(title="Portfolio Valuation Breakdown", yaxis_title="Value ($)")
        st.plotly_chart(fig_val, use_container_width=True)

    # Risk Segmentation - Prominent Section
    st.markdown("### 📊 Risk Segmentation Analysis")

    col1, col2 = st.columns(2)

    with col1:
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
            title="ML Default Probability by Credit Limit",
            labels={'default_probability': 'Avg Default Probability', 'limit_band': 'Credit Limit Band'},
            color='default_probability',
            color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c']
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Risk score distribution by category
        fig_box = px.box(
            df,
            x='risk_category',
            y='risk_score',
            title="Risk Score Distribution by Category",
            color='risk_category',
            color_discrete_map={
                'Low Risk': '#2ecc71',
                'Medium Risk': '#f39c12',
                'High Risk': '#e74c3c'
            },
            labels={'risk_score': 'ML Risk Score', 'risk_category': 'Risk Category'}
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Detailed risk breakdown table
    with st.expander("📋 Detailed Risk Breakdown by Category", expanded=False):
        risk_summary = df.groupby('risk_category').agg({
            'account_id': 'count',
            'credit_limit': ['sum', 'mean'],
            'outstanding_balance': ['sum', 'mean'],
            'default_probability': 'mean',
            'risk_score': 'mean'
        }).round(2)

        risk_summary.columns = ['Count', 'Total Exposure', 'Avg Exposure', 'Total Outstanding', 'Avg Outstanding', 'Avg Default Prob', 'Avg Risk Score']
        st.dataframe(risk_summary.style.format({
            'Total Exposure': '${:,.0f}',
            'Avg Exposure': '${:,.0f}',
            'Total Outstanding': '${:,.0f}',
            'Avg Outstanding': '${:,.0f}',
            'Avg Default Prob': '{:.2%}',
            'Avg Risk Score': '{:.2f}'
        }), use_container_width=True)

    # High-risk accounts detail
    st.markdown("---")
    st.markdown("### ⚠️ High Risk Accounts - Action Required")

    high_risk_df = df.nlargest(20, 'default_probability')[
        ['account_id', 'credit_limit', 'outstanding_balance', 'utilization_rate',
         'default_probability', 'risk_category', 'risk_score']
    ].round(4)

    st.dataframe(
        high_risk_df.style.format({
            'credit_limit': '${:,.0f}',
            'outstanding_balance': '${:,.0f}',
            'utilization_rate': '{:.2%}',
            'default_probability': '{:.2%}',
            'risk_score': '{:.2f}'
        }).background_gradient(subset=['default_probability'], cmap='Reds'),
        use_container_width=True
    )

    # STEP 6: Reports & Data Export
    st.markdown("---")
    st.markdown("## 📥 STEP 6: Download Reports & Scored Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Generate Excel report
        report_gen = ReportGenerator(df, metrics, valuation)
        excel_buffer = report_gen.create_excel_report()

        st.download_button(
            label="📊 Download Excel Report",
            data=excel_buffer,
            file_name=f"credit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col2:
        # Generate PDF summary
        pdf_buffer = report_gen.create_pdf_report()

        st.download_button(
            label="📄 Download PDF Summary",
            data=pdf_buffer,
            file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
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
            mime="text/csv",
            use_container_width=True
        )

    # Optional: File structure details
    with st.expander("📋 File Structure & Column Mapping Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", f"{data_info['num_records']:,}")
            st.metric("Total Columns", data_info['num_columns'])
        with col2:
            st.metric("File Type", data_info['file_type'])
            st.metric("Data Quality", f"{data_info['quality_score']:.1%}")

        if data_info.get('parser') == 'gemini':
            st.info("🤖 Parsed using Gemini AI")
            if 'gemini_metadata' in data_info:
                st.json(data_info['gemini_metadata'])
        else:
            st.markdown("**Detected Columns:**")
            st.dataframe(data_info.get('column_mapping', pd.DataFrame()), use_container_width=True)


def debt_collection_pricing_app(df=None, portfolio_name=None):
    """
    Debt Collection Portfolio Pricing Mode
    Price charged-off debt portfolios using P/C ratio and DCF methodologies

    Args:
        df: Pre-loaded DataFrame from session state (optional)
        portfolio_name: Name of the uploaded file (optional)
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

        # Check if we have uploaded data from session state
        has_uploaded_data = (df is not None and portfolio_name is not None)

        # Option to use uploaded file or enter manually
        if has_uploaded_data:
            input_method = st.radio(
                "Input Method:",
                ["📁 Use Uploaded File", "✍️ Manual Entry"],
                horizontal=True
            )
        else:
            input_method = "✍️ Manual Entry"
            st.info("💡 Upload a file in the main section above to enable automatic pricing from your portfolio data")

        if input_method == "📁 Use Uploaded File" and has_uploaded_data:
            # Use the already uploaded and processed file
            st.markdown("#### Using Uploaded Portfolio Data")
            st.success(f"✅ Using: **{portfolio_name}** ({len(df):,} accounts)")

            try:
                # Use the dataframe from session state
                df_portfolio = df.copy()

                # Auto-detect balance column
                balance_cols = [col for col in df_portfolio.columns
                              if any(term in col.lower() for term in ['balance', 'amount', 'principal', 'outstanding'])]

                if balance_cols:
                    balance_col = st.selectbox("Select Balance Column:", balance_cols, index=0)

                    # Calculate face value from data
                    auto_face_value = df_portfolio[balance_col].sum()

                    # Standardize column name for due diligence
                    df_analysis = df_portfolio.copy()
                    df_analysis['Current_Balance'] = df_portfolio[balance_col]

                    st.markdown("---")
                    st.markdown("### 🤖 ML Risk Analysis & Due Diligence")

                    # Run ML risk scoring first
                    with st.spinner("🔄 Running ML risk model on portfolio..."):
                        from src.ml_models import CreditRiskModel

                        risk_model = CreditRiskModel()
                        df_analysis = risk_model.score_portfolio(df_analysis)

                        # Calculate ML-based recovery adjustment
                        ml_avg_default_prob = df_analysis['default_probability'].mean()
                        ml_recovery_rate = 1 - ml_avg_default_prob  # Simple approach: recovery = 1 - default prob

                        # Show ML results
                        st.success("✅ ML Risk Scoring Complete")

                        ml_col1, ml_col2, ml_col3 = st.columns(3)
                        with ml_col1:
                            st.metric(
                                "ML Avg Default Probability",
                                f"{ml_avg_default_prob:.2%}",
                                help="Average ML-predicted default probability"
                            )
                        with ml_col2:
                            high_risk_pct = (df_analysis['risk_category'] == 'High Risk').sum() / len(df_analysis)
                            st.metric(
                                "High Risk Accounts",
                                f"{high_risk_pct:.1%}",
                                help="Percentage of accounts flagged as high risk by ML"
                            )
                        with ml_col3:
                            st.metric(
                                "ML-Based Recovery Estimate",
                                f"{ml_recovery_rate:.2%}",
                                help="Estimated recovery based on ML default predictions"
                            )

                    # Run traditional due diligence analysis
                    with st.spinner("📊 Running traditional due diligence..."):
                        base_recovery_estimate = 0.30  # Default 30%
                        dd_analyzer = PortfolioDueDiligence(
                            df=df_analysis,
                            base_recovery_rate=base_recovery_estimate,
                            statute_years=5
                        )

                        # Get composition and red flags
                        composition = dd_analyzer.analyze_composition()
                        red_flags = dd_analyzer.check_red_flags()
                        erc_analysis = dd_analyzer.calculate_adjusted_erc()

                    # Display risk summary
                    st.markdown("#### 📊 Portfolio Composition")
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        st.metric("Total Face Value", f"€{composition['total_face_value']:,.0f}")
                        st.metric("Total Accounts", f"{composition['total_accounts']:,}")
                    with comp_col2:
                        st.metric("Avg Balance per Account", f"€{composition['total_face_value']/composition['total_accounts']:,.2f}")

                    # Show balance distribution
                    if 'by_balance_size' in composition:
                        with st.expander("📈 Balance Size Distribution", expanded=True):
                            st.dataframe(composition['by_balance_size'], use_container_width=True)

                    # Show red flags
                    st.markdown("#### 🚨 Red Flags Detected")

                    red_flag_count = 0
                    if red_flags['statute_issues']['past_statute_count'] > 0:
                        red_flag_count += 1
                    if red_flags['missing_contacts']['count'] > 0:
                        red_flag_count += 1
                    if red_flags['small_balances']['count'] > 0:
                        red_flag_count += 1
                    if red_flags['concentration']['top_10_pct'] > 20:
                        red_flag_count += 1

                    if red_flag_count > 0:
                        st.warning(f"⚠️ Found {red_flag_count} risk categories")
                    else:
                        st.success("✅ No major red flags detected")

                    flag_col1, flag_col2 = st.columns(2)

                    with flag_col1:
                            # Statute issues
                            if red_flags['statute_issues']['past_statute_count'] > 0:
                                st.error(f"⚠️ **Statute-Barred Debt**: {red_flags['statute_issues']['past_statute_count']:,} accounts (€{red_flags['statute_issues']['past_statute_value']:,.0f})")
                            if red_flags['statute_issues']['near_statute_count'] > 0:
                                st.warning(f"⏰ **Near Statute Limit**: {red_flags['statute_issues']['near_statute_count']:,} accounts (€{red_flags['statute_issues']['near_statute_value']:,.0f})")

                            # Missing contacts
                            if red_flags['missing_contacts']['count'] > 0:
                                st.warning(f"📞 **Missing Contacts**: {red_flags['missing_contacts']['count']:,} accounts ({red_flags['missing_contacts']['pct']:.1f}%)")

                    with flag_col2:
                            # Small balances
                            if red_flags['small_balances']['count'] > 0:
                                st.info(f"💰 **Small Balances**: {red_flags['small_balances']['count']:,} accounts (<€500)")

                            # Concentration risk
                            if red_flags['concentration']['top_10_pct'] > 20:
                                st.warning(f"🎯 **Concentration Risk**: Top 10 = {red_flags['concentration']['top_10_pct']:.1f}%")

                    # Show ERC adjustments
                    st.markdown("---")
                    st.markdown("#### 💡 Risk-Adjusted Recovery Rate")

                    erc_col1, erc_col2, erc_col3 = st.columns(3)
                    with erc_col1:
                            st.metric("Base Recovery Rate", f"{erc_analysis['base_recovery_rate']:.1%}")
                    with erc_col2:
                            adjustment_pct = (erc_analysis['adjusted_recovery_rate'] - erc_analysis['base_recovery_rate'])
                            st.metric(
                                "Risk-Adjusted Rate",
                                f"{erc_analysis['adjusted_recovery_rate']:.1%}",
                                delta=f"{adjustment_pct:+.1%}"
                            )
                    with erc_col3:
                            st.metric("ERC Impact", f"€{erc_analysis['total_adjustment']:,.0f}")

                    if len(erc_analysis['adjustments']) > 0:
                            with st.expander("📋 View Detailed Adjustments"):
                                st.dataframe(erc_analysis['adjustment_summary'], use_container_width=True)

                    # Use values for pricing
                    face_value = auto_face_value
                    # portfolio_name is already provided as parameter

                    # Store both recovery rates for comparison
                    base_recovery_rate = base_recovery_estimate
                    risk_adjusted_recovery_rate = erc_analysis['adjusted_recovery_rate']

                    st.markdown("---")
                    st.markdown("### 💰 Pricing Calculation")
                    st.info("💡 Choose your pricing approach: ML-based, Traditional Risk-Adjusted, or Custom")

                    # Allow user to choose recovery method with ML option
                    recovery_method = st.radio(
                            "Recovery Rate Method:",
                            ["🤖 Use ML-Based Rate", "🎯 Use Traditional Risk-Adjusted", "📊 Use Custom Rate", "📈 Compare All Methods"],
                            horizontal=True
                    )

                    if recovery_method == "🤖 Use ML-Based Rate":
                        recovery_rate = ml_recovery_rate
                        use_comparison = False
                        st.success(f"✅ Using ML-based recovery rate: {ml_recovery_rate:.2%}")
                    elif recovery_method == "🎯 Use Traditional Risk-Adjusted":
                        recovery_rate = risk_adjusted_recovery_rate
                        use_comparison = False
                        st.success(f"✅ Using traditional risk-adjusted rate: {risk_adjusted_recovery_rate:.2%}")
                    elif recovery_method == "📊 Use Custom Rate":
                        recovery_rate = st.slider(
                            "Custom Recovery Rate (%)",
                            min_value=10,
                            max_value=60,
                            value=int(base_recovery_estimate * 100),
                            step=5,
                            help="Override with your own estimate",
                            key="custom_recovery"
                        ) / 100
                        use_comparison = False
                    else:  # Compare All Methods
                        recovery_rate = base_recovery_estimate
                        use_comparison = True
                        st.info("📊 Will compare: Base, ML-Based, and Traditional Risk-Adjusted pricing")

                    # File upload mode doesn't use custom curves/ratios
                    custom_curve = None
                    custom_pc_ratio = None

                else:
                    st.error("Could not find balance/amount column in file. Please select manual entry.")
                    face_value = None
                    use_comparison = False
                    custom_curve = None
                    custom_pc_ratio = None

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                face_value = None
                use_comparison = False
                custom_curve = None
                custom_pc_ratio = None

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

            # Manual entry doesn't use comparison mode
            use_comparison = False

        # Common section for both file upload and manual entry
        # Set defaults for optional parameters if not set by file upload
        if input_method == "📁 Upload Portfolio File":
            custom_pc_ratio = None
            custom_curve = None

        # Calculate button (only show if we have data)
        if face_value is not None and st.button("💰 Calculate Pricing", type="primary", use_container_width=True):
            # Initialize variables for PDF generation
            composition = None
            red_flags = None
            erc_analysis = None

            # If in comparison mode (file upload with risk analysis), recalculate for PDF
            if use_comparison and df is not None:
                # Prepare dataframe with Current_Balance column for analysis
                df_for_analysis = df.copy()

                # Auto-detect balance column
                balance_cols = [col for col in df_for_analysis.columns
                              if any(term in col.lower() for term in ['balance', 'amount', 'principal', 'outstanding'])]

                if balance_cols:
                    df_for_analysis['Current_Balance'] = df_for_analysis[balance_cols[0]]

                    # Run due diligence for PDF data
                    dd_analyzer = PortfolioDueDiligence(
                        df=df_for_analysis,
                        base_recovery_rate=base_recovery_rate,
                        statute_years=5
                    )
                    composition = dd_analyzer.analyze_composition()
                    red_flags = dd_analyzer.check_red_flags()
                    erc_analysis = dd_analyzer.calculate_adjusted_erc()

            with st.spinner("Calculating pricing..."):
                # Check if we need to calculate comparison
                if use_comparison:
                    # Calculate with base rate
                    pricer_base = DebtPortfolioPricer(
                        face_value=face_value,
                        recovery_rate=base_recovery_rate,
                        portfolio_type=portfolio_type,
                        servicing_costs=servicing_costs,
                        target_irr=target_irr
                    )
                    results_base = pricer_base.calculate_both_methods(
                        custom_curve=custom_curve,
                        custom_pc_ratio=custom_pc_ratio
                    )

                    # Calculate with ML-based rate (if available)
                    results_ml = None
                    pricer_ml = None
                    if 'ml_recovery_rate' in locals() and ml_recovery_rate is not None:
                        pricer_ml = DebtPortfolioPricer(
                            face_value=face_value,
                            recovery_rate=ml_recovery_rate,
                            portfolio_type=portfolio_type,
                            servicing_costs=servicing_costs,
                            target_irr=target_irr
                        )
                        results_ml = pricer_ml.calculate_both_methods(
                            custom_curve=custom_curve,
                            custom_pc_ratio=custom_pc_ratio
                        )

                    # Calculate with traditional risk-adjusted rate
                    pricer_adjusted = DebtPortfolioPricer(
                        face_value=face_value,
                        recovery_rate=risk_adjusted_recovery_rate,
                        portfolio_type=portfolio_type,
                        servicing_costs=servicing_costs,
                        target_irr=target_irr
                    )
                    results_adjusted = pricer_adjusted.calculate_both_methods(
                        custom_curve=custom_curve,
                        custom_pc_ratio=custom_pc_ratio
                    )

                    # Display comparison
                    st.markdown("---")
                    st.markdown(f"## 📊 Pricing Comparison: {portfolio_name}")

                    if results_ml is not None:
                        st.markdown("#### Three-Way Comparison: Base vs ML-Based vs Traditional Risk-Adjusted")
                    else:
                        st.markdown("#### Standard (Base Rate) vs Traditional Risk-Adjusted Pricing")

                    # Comparison metrics
                    if results_ml is not None:
                        comp_col1, comp_col2, comp_col3, comp_col4, comp_col5 = st.columns(5)
                    else:
                        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)

                    with comp_col1:
                        st.metric("Face Value", f"€{face_value:,.0f}")
                        st.caption("Same for all")

                    with comp_col2:
                        st.metric("Base Recovery", f"{base_recovery_rate:.1%}")
                        st.caption(f"ERC: €{results_base['portfolio_info']['erc']:,.0f}")

                    if results_ml is not None:
                        with comp_col3:
                            erc_diff_ml = results_ml['portfolio_info']['erc'] - results_base['portfolio_info']['erc']
                            st.metric(
                                "🤖 ML Recovery",
                                f"{ml_recovery_rate:.1%}",
                                delta=f"{erc_diff_ml:+,.0f} ERC"
                            )
                            st.caption(f"ERC: €{results_ml['portfolio_info']['erc']:,.0f}")

                    with comp_col4 if results_ml is not None else comp_col3:
                        erc_diff = results_adjusted['portfolio_info']['erc'] - results_base['portfolio_info']['erc']
                        st.metric(
                            "Traditional Risk-Adj",
                            f"{risk_adjusted_recovery_rate:.1%}",
                            delta=f"{erc_diff:+,.0f} ERC"
                        )
                        st.caption(f"ERC: €{results_adjusted['portfolio_info']['erc']:,.0f}")

                    with comp_col5 if results_ml is not None else comp_col4:
                        # Show largest difference
                        if results_ml is not None:
                            max_bid = max(results_base['recommendation']['bid_amount'],
                                        results_ml['recommendation']['bid_amount'],
                                        results_adjusted['recommendation']['bid_amount'])
                            min_bid = min(results_base['recommendation']['bid_amount'],
                                        results_ml['recommendation']['bid_amount'],
                                        results_adjusted['recommendation']['bid_amount'])
                            bid_range = max_bid - min_bid
                            st.metric(
                                "Price Range",
                                f"€{bid_range:,.0f}",
                                delta=f"{(bid_range/min_bid)*100:.1f}% spread"
                            )
                        else:
                            bid_diff = results_adjusted['recommendation']['bid_amount'] - results_base['recommendation']['bid_amount']
                            bid_diff_pct = bid_diff / results_base['recommendation']['bid_amount'] if results_base['recommendation']['bid_amount'] > 0 else 0
                            st.metric(
                                "Price Difference",
                                f"€{abs(bid_diff):,.0f}",
                                delta=f"{bid_diff_pct:+.1%}"
                            )

                    # Side-by-side comparison
                    st.markdown("---")

                    if results_ml is not None:
                        # Three-way comparison
                        compare_col1, compare_col2, compare_col3 = st.columns(3)

                        with compare_col1:
                            st.markdown("### 📊 Base Rate")
                            st.markdown(f"**Recovery:** {base_recovery_rate:.1%}")
                            st.metric("Recommended Bid", f"€{results_base['recommendation']['bid_amount']:,.0f}")
                            decision_color = "green" if results_base['recommendation']['decision'] == 'BUY' else "red"
                            st.markdown(f"<h4 style='color: {decision_color};'>{'✓ BUY' if results_base['recommendation']['decision'] == 'BUY' else '✗ PASS'}</h4>", unsafe_allow_html=True)

                        with compare_col2:
                            st.markdown("### 🤖 ML-Based")
                            st.markdown(f"**Recovery:** {ml_recovery_rate:.1%}")
                            ml_bid_diff = results_ml['recommendation']['bid_amount'] - results_base['recommendation']['bid_amount']
                            st.metric(
                                "Recommended Bid",
                                f"€{results_ml['recommendation']['bid_amount']:,.0f}",
                                delta=f"{ml_bid_diff:+,.0f}"
                            )
                            decision_color = "green" if results_ml['recommendation']['decision'] == 'BUY' else "red"
                            st.markdown(f"<h4 style='color: {decision_color};'>{'✓ BUY' if results_ml['recommendation']['decision'] == 'BUY' else '✗ PASS'}</h4>", unsafe_allow_html=True)

                        with compare_col3:
                            st.markdown("### 🎯 Traditional Risk-Adj")
                            st.markdown(f"**Recovery:** {risk_adjusted_recovery_rate:.1%}")
                            trad_bid_diff = results_adjusted['recommendation']['bid_amount'] - results_base['recommendation']['bid_amount']
                            st.metric(
                                "Recommended Bid",
                                f"€{results_adjusted['recommendation']['bid_amount']:,.0f}",
                                delta=f"{trad_bid_diff:+,.0f}"
                            )
                            decision_color = "green" if results_adjusted['recommendation']['decision'] == 'BUY' else "red"
                            st.markdown(f"<h4 style='color: {decision_color};'>{'✓ BUY' if results_adjusted['recommendation']['decision'] == 'BUY' else '✗ PASS'}</h4>", unsafe_allow_html=True)

                        # Recommendation for three-way
                        st.markdown("---")
                        st.markdown("### 💡 Recommendation")

                        # Determine which method to recommend
                        all_bids = [
                            ("Base", results_base['recommendation']['bid_amount']),
                            ("ML-Based", results_ml['recommendation']['bid_amount']),
                            ("Traditional", results_adjusted['recommendation']['bid_amount'])
                        ]
                        all_bids_sorted = sorted(all_bids, key=lambda x: x[1])

                        st.info(f"""
                        **Three Pricing Approaches Compared:**

                        🤖 **ML-Based Pricing (€{results_ml['recommendation']['bid_amount']:,.0f}):**
                        - Uses machine learning default predictions
                        - Data-driven recovery estimate: {ml_recovery_rate:.2%}
                        - Best for: Portfolios with rich historical data

                        🎯 **Traditional Risk-Adjusted (€{results_adjusted['recommendation']['bid_amount']:,.0f}):**
                        - Uses rule-based red flags detection
                        - Accounts for statute issues, missing contacts, concentration risk
                        - Best for: Conservative, compliance-focused buyers

                        📊 **Base Rate (€{results_base['recommendation']['bid_amount']:,.0f}):**
                        - Industry standard recovery rate
                        - No portfolio-specific adjustments
                        - Best for: Quick estimates and benchmarking

                        **Price Range:** €{all_bids_sorted[0][1]:,.0f} to €{all_bids_sorted[2][1]:,.0f} (€{all_bids_sorted[2][1] - all_bids_sorted[0][1]:,.0f} spread)

                        **Recommended Strategy:** Use the **ML-Based** or **Traditional Risk-Adjusted** approach, as they incorporate portfolio-specific risk factors.
                        Consider the average: **€{(results_ml['recommendation']['bid_amount'] + results_adjusted['recommendation']['bid_amount']) / 2:,.0f}**
                        """)

                    else:
                        # Two-way comparison (original)
                        compare_col1, compare_col2 = st.columns(2)

                        with compare_col1:
                            st.markdown("### 📊 Standard Pricing (Base Rate)")
                            st.markdown(f"**Recovery Rate:** {base_recovery_rate:.1%}")

                            st.metric("P/C Method", f"€{results_base['pc_method']['price']:,.0f}")
                            st.caption(f"{results_base['pc_method']['price_as_pct_of_face']:.1%} of Face")

                            st.metric("DCF Method", f"€{results_base['dcf_method']['price']:,.0f}")
                            st.caption(f"{results_base['dcf_method']['price_as_pct_of_face']:.1%} of Face")

                            margin_base = results_base['recommendation'].get('margin')
                            st.metric(
                                "Recommended Bid",
                                f"€{results_base['recommendation']['bid_amount']:,.0f}",
                                delta=f"{margin_base:.1%} margin" if margin_base is not None else None
                            )
                            decision_color = "green" if results_base['recommendation']['decision'] == 'BUY' else "red"
                            st.markdown(f"<h3 style='color: {decision_color};'>{'✓ BUY' if results_base['recommendation']['decision'] == 'BUY' else '✗ PASS'}</h3>", unsafe_allow_html=True)

                        with compare_col2:
                            st.markdown("### 🎯 Traditional Risk-Adjusted Pricing")
                            st.markdown(f"**Recovery Rate:** {risk_adjusted_recovery_rate:.1%}")

                            pc_diff = results_adjusted['pc_method']['price'] - results_base['pc_method']['price']
                            st.metric(
                                "P/C Method",
                                f"€{results_adjusted['pc_method']['price']:,.0f}",
                                delta=f"{pc_diff:+,.0f}"
                            )
                            st.caption(f"{results_adjusted['pc_method']['price_as_pct_of_face']:.1%} of Face")

                            dcf_diff = results_adjusted['dcf_method']['price'] - results_base['dcf_method']['price']
                            st.metric(
                                "DCF Method",
                                f"€{results_adjusted['dcf_method']['price']:,.0f}",
                                delta=f"{dcf_diff:+,.0f}"
                            )
                            st.caption(f"{results_adjusted['dcf_method']['price_as_pct_of_face']:.1%} of Face")

                            margin_adjusted = results_adjusted['recommendation'].get('margin')
                            st.metric(
                                "Recommended Bid",
                                f"€{results_adjusted['recommendation']['bid_amount']:,.0f}",
                                delta=f"{margin_adjusted:.1%} margin" if margin_adjusted is not None else None
                            )
                            decision_color = "green" if results_adjusted['recommendation']['decision'] == 'BUY' else "red"
                            st.markdown(f"<h3 style='color: {decision_color};'>{'✓ BUY' if results_adjusted['recommendation']['decision'] == 'BUY' else '✗ PASS'}</h3>", unsafe_allow_html=True)

                        # Recommendation
                        st.markdown("---")
                        st.markdown("### 💡 Recommendation")
                        if results_adjusted['recommendation']['decision'] != results_base['recommendation']['decision']:
                            st.warning("⚠️ **Risk adjustment changes the recommendation!**")

                        bid_diff = results_adjusted['recommendation']['bid_amount'] - results_base['recommendation']['bid_amount']
                        st.info(f"""
                        **Suggested Approach:** Use the **Traditional Risk-Adjusted Pricing** (€{results_adjusted['recommendation']['bid_amount']:,.0f}) as it accounts for:
                        - Portfolio-specific red flags
                        - Statute of limitations issues
                        - Contact information quality
                        - Balance size distribution
                        - Concentration risk

                        The risk-adjusted bid is **€{abs(bid_diff):,.0f} {('lower' if bid_diff < 0 else 'higher')}** than the standard pricing.
                        """)

                    # Use risk-adjusted results for detailed display
                    results = results_adjusted
                    pricer = pricer_adjusted

                else:
                    # Single calculation (manual entry or single rate selected)
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

                # PDF Report Generation
                st.markdown("---")
                st.markdown("### 📄 Download Comprehensive Report")

                # Initialize PDF buffer in session state
                if 'pdf_buffer' not in st.session_state:
                    st.session_state.pdf_buffer = None
                if 'pdf_filename' not in st.session_state:
                    st.session_state.pdf_filename = None

                if st.button("📥 Generate PDF Report", type="primary", use_container_width=True, key="gen_pdf_btn"):
                    with st.spinner("Generating comprehensive PDF report..."):
                        try:
                            # Generate PDF
                            pdf_buffer = create_debt_pricing_pdf(
                                portfolio_name=portfolio_name,
                                face_value=face_value,
                                recovery_rate=recovery_rate,
                                results=results,
                                monthly_schedule=monthly_schedule,
                                portfolio_type=portfolio_type,
                                servicing_costs=servicing_costs,
                                target_irr=target_irr,
                                composition=composition,
                                red_flags=red_flags,
                                erc_analysis=erc_analysis
                            )

                            # Store in session state
                            st.session_state.pdf_buffer = pdf_buffer.getvalue()
                            st.session_state.pdf_filename = f"Debt_Pricing_Report_{portfolio_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
                            st.success("✅ PDF report generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                            st.session_state.pdf_buffer = None

                # Show download button if PDF is ready
                if st.session_state.pdf_buffer is not None:
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=st.session_state.pdf_buffer,
                        file_name=st.session_state.pdf_filename,
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_pdf_btn"
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

        # Check if we have uploaded data from session state
        if df is not None and portfolio_name is not None:
            st.success(f"✅ Using uploaded file: **{portfolio_name}** ({len(df):,} accounts)")

            # Use the dataframe from session state
            df_portfolio = df.copy()

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
                    # Prepare dataframe with Current_Balance column
                    df_for_dd = df_portfolio.copy()

                    # Auto-detect balance column and standardize
                    balance_cols = [col for col in df_for_dd.columns
                                  if any(term in col.lower() for term in ['balance', 'amount', 'principal', 'outstanding'])]

                    if balance_cols:
                        df_for_dd['Current_Balance'] = df_for_dd[balance_cols[0]]
                    elif 'Current_Balance' not in df_for_dd.columns:
                        st.error("Could not find a balance column. Please ensure your file has a column with 'balance', 'amount', 'principal', or 'outstanding' in the name.")
                        st.stop()

                    # Run due diligence
                    dd = PortfolioDueDiligence(
                        df_for_dd,
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

        else:
            st.info("👆 Upload a portfolio file in the main section above to run due diligence analysis")
            st.markdown("""
            **Due Diligence Analysis includes:**
            - Portfolio composition by balance size and age
            - Red flags detection (statute issues, missing contacts, concentration risk)
            - Risk-adjusted ERC calculation
            - Comprehensive risk assessment report
            """)


if __name__ == "__main__":
    main()
