# 🏦 Automated Credit Portfolio Analyzer

**Upload any credit file → Get instant risk scoring, financial analysis, and valuation**

An intelligent system that automatically understands credit portfolio files, applies machine learning risk scoring, calculates comprehensive financial metrics, and generates professional reports.

## ✨ Features

### 🔍 **Smart File Understanding**
- Automatically detects credit file structure (credit cards, loans, trade credit)
- Fuzzy column matching - works with various naming conventions
- Data quality assessment and validation
- Handles missing data intelligently

### 🤖 **ML-Powered Risk Scoring**
- XGBoost-based default probability prediction
- Individual account risk scores (0-1000 scale)
- Risk categorization (Low/Medium/High)
- Portfolio-level risk metrics

### 📊 **Comprehensive Financial Analysis**
- **20+ Financial Ratios**: Utilization, concentration, quality metrics
- **Portfolio Valuation**: Book value, risk-adjusted value, NPV
- **Risk Metrics**: Expected Loss, VaR, Loss Given Default
- **Exposure Analysis**: Total exposure, outstanding balances, limits

### 📈 **Interactive Visualizations**
- Risk distribution histograms
- Portfolio composition pie charts
- Utilization analysis
- High-risk account identification

### 📥 **Professional Reports**
- **Excel Report**: Multi-sheet workbook with all analysis
- **PDF Summary**: Executive-level summary report
- **Scored Portfolio CSV**: Download enriched data with risk scores

## 🚀 Quick Start

### Installation

**Create a virtual environment**

On Windows:
```bash
python -m venv venv
venv\Scripts\activate

```bash
# Clone or download this project
cd credit-portfolio-analyzer

# Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
streamlit run credit_portfolio_analyzer.py
```

This will open the analyzer in your web browser at `http://localhost:8501`

### Command Line Usage

```bash
python analyze_cli.py --file portfolio.csv --output report.pdf
```

## 📁 Supported File Formats

### Credit Card Portfolios
```
Columns: ID, Credit_Limit, Payment_Status, Bill_Amounts, Payment_Amounts, Demographics
```

### Loan Portfolios
```
Columns: Account_ID, Loan_Amount, Interest_Rate, Payment_History, Balance, Status
```

### Trade Credit
```
Columns: Customer_ID, Credit_Line, Outstanding, Days_Past_Due, Payment_Terms
```

**The system automatically detects and adapts to your file structure!**

## 📊 What You Get

### 1. **Instant File Analysis**
- File type detection
- Data quality score
- Column mapping confirmation

### 2. **ML Risk Scoring**
- Default probability for each account
- Risk score (0-1000)
- Risk category (Low/Medium/High)
- Model performance metrics (AUC)

### 3. **Portfolio Metrics Dashboard**

**Exposure Metrics:**
- Total exposure across portfolio
- Average credit limits
- Total outstanding balances
- Utilization rates

**Risk Metrics:**
- Expected Loss calculation
- Value at Risk (VaR 95%)
- Loss Given Default
- High-risk account count

**Valuation:**
- Book value
- Risk-adjusted value
- Expected revenue
- Net portfolio value
- ROA and RAROC

### 4. **Downloadable Reports**
- **Excel**: Detailed multi-sheet analysis
- **PDF**: Executive summary
- **CSV**: Scored portfolio data

## 📖 How It Works

```
1. Upload File
   ↓
2. Auto-Detect Structure → Smart column mapping
   ↓
3. Data Validation → Clean and standardize
   ↓
4. Feature Engineering → Create predictive features
   ↓
5. ML Scoring → XGBoost risk model
   ↓
6. Calculate Metrics → 20+ ratios and metrics
   ↓
7. Portfolio Valuation → Risk-adjusted value
   ↓
8. Generate Reports → Excel, PDF, CSV
```

## 🎯 Use Cases

### **Credit Risk Management**
- Monitor portfolio health
- Identify high-risk accounts
- Calculate capital requirements
- Track credit quality trends

### **Portfolio Valuation**
- M&A due diligence
- Portfolio sale/purchase
- Mark-to-market valuation
- Investor reporting

### **Strategic Planning**
- Risk appetite analysis
- Capital allocation
- Collection prioritization
- Underwriting policy review

## 📊 Sample Output

### Executive Summary
```
Total Accounts:          30,000
Total Exposure:          $4.5B
Average Utilization:     47%
Expected Loss:           $180M
High Risk Accounts:      3,200 (10.7%)
Portfolio ROA:           12.3%
```

### Risk Distribution
```
Low Risk:     18,500 accounts (61.7%)
Medium Risk:   8,300 accounts (27.7%)
High Risk:     3,200 accounts (10.7%)
```

## 🔧 Customization

### Using Your Own ML Model

```python
from src.ml_models import CreditRiskModel

# Load your pre-trained model
risk_model = CreditRiskModel(model_path='your_model.pkl')
df = risk_model.score_portfolio(df)
```

### Custom Financial Ratios

Edit `src/portfolio_metrics.py`:

```python
def _calculate_detailed_ratios(self):
    ratios = {}
    # Add your custom ratios here
    ratios['Custom Metric'] = your_calculation
    return ratios
```

## 📁 Project Structure

```
credit-portfolio-analyzer/
├── credit_portfolio_analyzer.py   # Main Streamlit app
├── analyze_cli.py                 # Command-line interface
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── src/
    ├── file_parser.py            # Smart file detection
    ├── ml_models.py              # ML risk scoring
    ├── portfolio_metrics.py      # Financial calculations
    └── report_generator.py       # Report creation
```

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

## 📝 Technical Details

### Machine Learning
- **Algorithm**: XGBoost Classifier
- **Features**: Payment history, utilization, demographics, behavior patterns
- **Performance**: AUC ~0.77 (calibrated on UCI Credit Card dataset)
- **Threshold**: Optimized at 0.30 for balanced precision/recall

### Financial Calculations
- **Expected Loss**: EL = EAD × PD × LGD
- **Value at Risk**: 95th percentile of loss distribution
- **Risk-Adjusted Value**: Book Value - Expected Loss
- **RAROC**: (Revenue - Expected Loss) / Economic Capital

### Data Requirements
**Minimum required:**
- Account identifier
- Credit limit or loan amount
- Outstanding balance

**Optimal for ML scoring:**
- Payment history (6+ months)
- Bill/statement history
- Demographics (age, education)
- Payment amounts

## 🔐 Data Privacy

- **All processing is local** - data never leaves your machine
- No data is stored or transmitted
- Reports contain only aggregated statistics
- Individual account data stays in your downloads

## 📞 Support

For questions or issues:
1. Check the examples in the `examples/` folder
2. Review the code documentation
3. Open an issue on GitHub

## 🎓 Learn More

- [Credit Risk Modeling Best Practices](https://example.com)
- [Basel II/III Framework](https://example.com)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📜 License

MIT License - Feel free to use and modify!

---

**Built with ❤️ for credit risk professionals**

*Last updated: November 2025*
