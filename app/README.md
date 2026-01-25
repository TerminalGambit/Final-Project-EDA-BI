# App Structure

Streamlit multi-page dashboard for Beauty E-Commerce Competitive Intelligence.

## Running the App

```bash
streamlit run app/main.py
```

## Directory Structure

```
app/
├── main.py              # Entry point & Executive Summary page
├── config.py            # Centralized path configuration
├── data/
│   └── loader.py        # Data loading & preprocessing functions
└── pages/               # Dashboard modules (auto-discovered by Streamlit)
```

## Core Modules

### `config.py`
Centralized path configuration for data files and outputs. All paths are relative to project root for portability.

### `data/loader.py`
Data pipeline that loads cleaned data and computes derived metrics:
- **Stock changes**: Inventory movement between snapshots
- **Estimated sales**: Units sold (negative stock changes)
- **Estimated revenue**: Sales × price
- **Aggregations**: Daily, weekly, by product, by brand

Uses `@st.cache_data` for performance optimization.

## Dashboard Pages

| Page | Purpose |
|------|---------|
| **main.py** | Executive summary with KPIs, strategic alerts, and navigation |
| **1_📈_Sales_Intelligence** | Sales velocity, revenue rankings, trend analysis |
| **2_💰_Pricing_Intelligence** | Price tracking, competitor pricing, elasticity analysis |
| **3_📦_Inventory_Risk** | Stock levels, stockout tracking, reorder recommendations |
| **4_🔍_Exploration** | Interactive data exploration and filtering |
| **5_🎯_Strategic_Intelligence** | Market positioning, competitive insights |
| **6_🔮_Predictive_Analytics** | Demand forecasting, seasonality patterns |
| **7_🎬_Action_Center** | Prioritized recommendations and action items |
| **8_📊_Strategic_Decisions** | Decision support with scenario analysis |

## Data Flow

```
cleaned_data.parquet
        ↓
    loader.py (compute metrics)
        ↓
    Cached DataFrames
        ↓
    Dashboard Pages
        ↓
    metrics.json (exported)
```
