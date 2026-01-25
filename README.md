# 💄 Beauty E-Commerce Intelligence Dashboard

## 📋 Overview

A comprehensive business intelligence dashboard built with Streamlit for analyzing competitor data in the beauty/haircare e-commerce space. This dashboard provides executives with actionable insights from 374 days of market data, covering sales velocity, pricing strategies, inventory management, and strategic decision-making.

**Key Capabilities:**
- Competitive sales intelligence based on stock movement analysis
- Price tracking and elasticity analysis
- Inventory risk management and reorder recommendations
- Strategic concentration risk identification
- Predictive analytics and demand forecasting
- Actionable recommendations with ROI projections

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Required data file: `data/processed/cleaned_data.parquet`

### Installation

1. **Install dependencies:**
```bash
pip install streamlit pandas numpy plotly pygwalker pyarrow
```

2. **Run the dashboard:**
```bash
streamlit run app/main.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 🧭 Navigation Guide

The dashboard consists of 8 specialized modules, each providing unique insights for executive decision-making:

### 🏠 Home (main.py)
**What it provides:**
- Strategic alerts highlighting immediate risks (brand concentration, revenue trends, supply chain gaps)
- Executive summary KPIs (total revenue, daily revenue, units sold, stockout rate)
- Top brands and products by revenue
- Quick navigation overview

**CEO Value:** Immediate visibility into critical business metrics and risks requiring attention.

---

### 📈 Page 1: Sales Intelligence
**What it provides:**
- Estimated sales velocity and revenue trends over time
- Product and brand performance rankings
- Declining product identification
- Revenue contribution analysis

**CEO Value:** Understand which products and brands drive revenue, identify growth opportunities, and spot underperforming SKUs for potential delisting.

**Key Insights:**
- Top 15 products by revenue and velocity
- Weekly revenue trends by brand
- Revenue distribution treemaps
- Declining products requiring attention

---

### 💰 Page 2: Pricing Intelligence
**What it provides:**
- Price change detection and tracking
- Price distribution by brand and segment
- Elasticity estimation for pricing optimization
- Competitive positioning analysis

**CEO Value:** Identify pricing opportunities, understand competitor pricing strategies, and optimize margins without losing volume.

**Key Insights:**
- Recent price increases and decreases
- Price segment performance (€0-10, €10-20, etc.)
- Products with pricing power (inelastic demand)
- Margin improvement opportunities

---

### 📦 Page 3: Inventory Risk
**What it provides:**
- Real-time stockout dashboard
- Days of inventory calculations
- Restock pattern analysis
- ROI calculator for inventory investments

**CEO Value:** Prevent lost sales from stockouts, optimize working capital allocation, and understand the true cost of inventory gaps.

**Key Insights:**
- Products requiring immediate reorder (CRITICAL/STOCKOUT)
- Revenue at risk from supply chain gaps
- Working capital investment requirements
- Holding cost vs. opportunity cost analysis

---

### 🔍 Page 4: Exploration
**What it provides:**
- Interactive data exploration with PyGWalker
- Multiple dataset views (Daily, Products, Brands, Weekly, Raw)
- Drag-and-drop visual analysis
- Custom filtering and aggregations

**CEO Value:** Self-service BI for ad-hoc analysis without waiting for data team. Answer business questions in real-time.

**Key Features:**
- Tableau-style interface
- Save and persist custom charts
- Export capabilities for presentations
- No coding required

---

### 🎯 Page 5: Strategic Intelligence
**What it provides:**
- Hidden risks and blind spots (brand concentration, dependencies)
- Executive KPI scorecard with trend analysis
- Anomaly detection
- Efficiency opportunity identification

**CEO Value:** Discover what you don't know. Strategic risks that aren't obvious in standard reports, including concentration vulnerabilities and pricing misalignments.

**Key Insights:**
- Brand concentration risk (HHI index)
- Single-product brand vulnerabilities
- Underpriced high-velocity products
- Week-over-week performance anomalies

---

### 🔮 Page 6: Predictive Analytics
**What it provides:**
- Demand forecasting (30/60/90 day projections)
- Seasonality pattern analysis (weekly, monthly)
- Trend-based projections with confidence intervals
- Scenario planning tools

**CEO Value:** Anticipate future demand, plan inventory buildups before peak periods, and make proactive decisions based on forecasts, not just historical data.

**Key Insights:**
- Best/worst days and months for sales
- Growth rate trends
- Forecast accuracy metrics (MAPE)
- Seasonal adjustment factors

---

### 🎬 Page 7: Action Center
**What it provides:**
- Prioritized action recommendations
- Investment cases with ROI projections
- Quick wins identification
- Success metrics and rollback triggers

**CEO Value:** Bridge from insight to execution. Know exactly what to do, how much it will cost, what the return will be, and how to measure success.

**Key Features:**
- Priority matrix (impact vs. investment)
- Top 10 actionable recommendations
- Inventory, pricing, and portfolio actions
- Downloadable reorder reports

---

### 📊 Page 8: Strategic Decisions
**What it provides:**
- Business cases with Options A/B/C analysis
- Decision trees for major strategic choices
- Cost, timeline, and ROI for each option
- Success metrics for tracking

**CEO Value:** Make informed strategic decisions with full context. Compare multiple approaches, understand trade-offs, and choose the path with the best risk-adjusted return.

**Key Decisions:**
1. **Brand Concentration Risk**: How to reduce dependency on top brands
2. **Stockout Strategy**: Whether to invest in inventory or accept losses
3. **Pricing Optimization**: How aggressive to be with price increases

---

## 📊 Data Pipeline

### Data Flow
```
Raw Data (data/raw/all_data.csv)
    ↓
Cleaning & Processing (scripts/04_data_cleaning.py)
    ↓
Cleaned Data (data/processed/cleaned_data.parquet)
    ↓
Data Loader (app/data/loader.py)
    ↓
Dashboard Pages (app/main.py, app/pages/*.py)
```

### Key Metrics Computed
- **Stock Changes**: Inventory movement between timestamps
- **Estimated Sales**: Negative stock changes (excluding restocks)
- **Estimated Revenue**: Sales × Price
- **Daily Velocity**: Average units sold per day
- **Stockout Rate**: Percentage of days out of stock
- **Days of Inventory**: Current stock ÷ daily velocity
- **Price Elasticity**: Volume change in response to price changes

---

## 🏗️ Architecture

### Project Structure
```
final_pandas_x_sqlbi/
├── app/
│   ├── main.py                          # Home page with executive summary
│   ├── data/
│   │   ├── loader.py                    # Data loading & metric computation
│   │   └── __init__.py
│   └── pages/
│       ├── 1_📈_Sales_Intelligence.py
│       ├── 2_💰_Pricing_Intelligence.py
│       ├── 3_📦_Inventory_Risk.py
│       ├── 4_🔍_Exploration.py
│       ├── 5_🎯_Strategic_Intelligence.py
│       ├── 6_🔮_Predictive_Analytics.py
│       ├── 7_🎬_Action_Center.py
│       └── 8_📊_Strategic_Decisions.py
├── data/
│   ├── raw/                             # Original scraped data
│   └── processed/                       # Cleaned parquet files
├── notebooks/
│   ├── exploratory_data_analysis.ipynb  # EDA notebook
│   └── README.md                        # Notebook documentation
└── README.md                            # This file
```

### Technology Stack
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly
- **Interactive Analysis**: PyGWalker
- **Data Storage**: Parquet (via PyArrow)

---

## 💡 CEO-Specific Value Propositions

### 1. **Stop Leaving Money on the Table**
- Identifies €50,000+ in annual revenue opportunity from underpriced products
- Detects products with pricing power that can support 5-10% increases

### 2. **Prevent Revenue Loss**
- Quantifies compound impact of stockouts (immediate loss × customer churn multiplier)
- Provides reorder recommendations to capture at-risk revenue

### 3. **Reduce Strategic Risk**
- Calculates brand concentration exposure using HHI index
- Identifies single-product brand dependencies that create vulnerabilities

### 4. **Optimize Working Capital**
- Shows exact ROI for inventory investments
- Balances holding costs against opportunity costs

### 5. **Make Data-Driven Decisions**
- Provides Options A/B/C analysis for major strategic choices
- Includes success metrics and rollback triggers for each decision

---

## 📈 Key Metrics & KPIs

### Sales Metrics
- **Total Estimated Revenue**: €XXX,XXX (based on stock decrements)
- **Average Daily Revenue**: €X,XXX
- **Total Units Sold**: XX,XXX units
- **Top Brand Share**: XX% (concentration risk indicator)

### Inventory Health
- **Average Stockout Rate**: ~6%
- **Products Currently Out of Stock**: XX
- **Critical Stock Products** (<7 days): XX
- **Revenue at Risk**: €XX,XXX (30-day exposure)

### Strategic Indicators
- **HHI Concentration Index**: X,XXX (1,500 = competitive, 2,500+ = high risk)
- **Underpriced Products**: XX opportunities
- **Pricing Opportunity**: €XX,XXX annual upside

---

## 🔧 Troubleshooting

### Issue: "No such file or directory: cleaned_data.parquet"
**Solution:** Ensure the data cleaning script has been run:
```bash
python scripts/04_data_cleaning.py
```

### Issue: "ModuleNotFoundError: No module named 'pygwalker'"
**Solution:** Install PyGWalker:
```bash
pip install pygwalker
```

### Issue: Dashboard loads slowly
**Solution:** Data caching is enabled with 1-hour TTL. First load may be slow (~30s for 1.7M rows), subsequent loads are instant.

---

## 📝 Notes

- **Data Privacy**: All data is competitor data scraped from public e-commerce sites. No customer PII.
- **Estimation Method**: Sales are estimated from stock decrements, not actual transaction data.
- **Update Frequency**: Dashboard reflects static historical data. For real-time monitoring, implement automated data refresh pipeline.
- **Recommended Review Cadence**: 
  - Daily: Strategic Alerts (main page)
  - Weekly: Sales Intelligence, Inventory Risk
  - Monthly: Strategic Intelligence, Pricing opportunities
  - Quarterly: Strategic Decisions

---

## 🚀 Next Steps

1. **First-Time Users**: Start with the Home page to understand strategic alerts
2. **Deep Dive**: Visit Sales Intelligence → Inventory Risk → Action Center
3. **Strategic Planning**: Review Strategic Intelligence → Predictive Analytics → Strategic Decisions
4. **Ad-Hoc Analysis**: Use Exploration page for custom questions

---

## 📞 Support

For questions about the dashboard, data sources, or methodology, refer to:
- Notebook documentation: `notebooks/README.md`
- Data cleaning scripts: `scripts/04_data_cleaning.py`
- Metrics computation: `app/data/loader.py`

---

**Built with ❤️ for executive decision-making**
