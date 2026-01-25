"""
Executive Intelligence Dashboard
================================
Beauty/Haircare E-Commerce Competitive Intelligence

Main entry point for the Streamlit dashboard.
Run with: streamlit run app/main.py
"""

import streamlit as st
import pandas as pd

from data.loader import load_all_data, compute_all_metrics, export_metrics_json

# Page configuration
st.set_page_config(
    page_title="Beauty E-Commerce Intelligence",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Main page content
st.markdown('<p class="main-header">💄 Beauty E-Commerce Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Competitive Intelligence Dashboard • Powered by 374 Days of Market Data</p>', unsafe_allow_html=True)

# Load data
with st.spinner("Loading data..."):
    df, daily, products, brands = load_all_data()
    metrics = compute_all_metrics()

# Export metrics JSON on load
export_metrics_json()

# =============================================================================
# STRATEGIC ALERTS - What the CEO doesn't know
# =============================================================================
st.subheader("🚨 Strategic Alerts")

# Calculate key risks
top_brand = brands.iloc[0]
top_brand_pct = (top_brand['total_revenue'] / brands['total_revenue'].sum()) * 100
single_product_brands = brands[brands['n_products'] <= 2]
single_product_revenue = single_product_brands['total_revenue'].sum()

# Weekly trend (last 4 weeks vs previous 4 weeks)
daily_copy = daily.copy()
daily_copy['date_day'] = pd.to_datetime(daily_copy['date_day'])
recent_4w = daily_copy[daily_copy['date_day'] > daily_copy['date_day'].max() - pd.Timedelta(days=28)]['estimated_revenue'].sum()
prev_4w = daily_copy[(daily_copy['date_day'] > daily_copy['date_day'].max() - pd.Timedelta(days=56)) & 
                     (daily_copy['date_day'] <= daily_copy['date_day'].max() - pd.Timedelta(days=28))]['estimated_revenue'].sum()
revenue_trend = ((recent_4w - prev_4w) / prev_4w * 100) if prev_4w > 0 else 0

col1, col2, col3 = st.columns(3)

with col1:
    if top_brand_pct > 15:
        st.error(f"""
        **⚠️ Brand Concentration Risk**
        
        {top_brand['brand']} = {top_brand_pct:.1f}% of revenue
        
        *Supplier disruption risk: €{top_brand['total_revenue']/12:,.0f}/month*
        """)
    else:
        st.success("✅ Brand portfolio well diversified")

with col2:
    if revenue_trend < -5:
        st.warning(f"""
        **📉 Revenue Declining**
        
        Last 4 weeks: {revenue_trend:+.1f}% vs prior
        
        *Investigate: seasonality, competition, or stockouts?*
        """)
    elif revenue_trend > 10:
        st.success(f"""
        **📈 Strong Growth**
        
        Last 4 weeks: {revenue_trend:+.1f}% vs prior
        
        *Ensure inventory can support demand*
        """)
    else:
        st.info(f"📊 Revenue trend: {revenue_trend:+.1f}% (stable)")

with col3:
    high_stockout = products[products['stockout_rate'] > 0.15]
    if len(high_stockout) > 0:
        lost_rev = (high_stockout['daily_velocity'] * high_stockout['avg_price'] * high_stockout['stockout_days']).sum()
        st.warning(f"""
        **📦 Supply Chain Gaps**
        
        {len(high_stockout)} products with >15% stockout
        
        *Est. lost revenue: €{lost_rev:,.0f}*
        """)
    else:
        st.success("✅ Inventory health good")

st.divider()

# Executive Summary KPIs
st.subheader("📊 Executive Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💰 Estimated Total Revenue",
        f"€{metrics['sales_estimates']['total_estimated_revenue']:,.0f}",
        help="Estimated competitor revenue based on stock decrements × prices"
    )

with col2:
    st.metric(
        "📅 Avg Daily Revenue",
        f"€{metrics['sales_estimates']['avg_daily_revenue']:,.0f}",
        help="Average estimated daily revenue"
    )

with col3:
    st.metric(
        "📦 Total Units Sold",
        f"{metrics['sales_estimates']['total_estimated_sales']:,}",
        help="Estimated total units sold based on stock decrements"
    )

with col4:
    st.metric(
        "⚠️ Avg Stockout Rate",
        f"{metrics['inventory_health']['avg_stockout_rate']:.1f}%",
        help="Average percentage of days products were out of stock"
    )

st.divider()

# Data overview
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏢 Top Brands By Revenue")
    top_brands = brands.head(8)[['brand', 'total_revenue', 'n_products', 'daily_velocity']].copy()
    top_brands['total_revenue'] = top_brands['total_revenue'].apply(lambda x: f"€{x:,.0f}")
    top_brands['daily_velocity'] = top_brands['daily_velocity'].apply(lambda x: f"{x:.1f} units/day")
    top_brands.columns = ['Brand', 'Est. Revenue', 'Products', 'Daily Velocity']
    st.dataframe(top_brands, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🏆 Top Products By Revenue")
    top_products = products.head(8)[['title', 'brand', 'total_revenue', 'daily_velocity']].copy()
    top_products['total_revenue'] = top_products['total_revenue'].apply(lambda x: f"€{x:,.0f}")
    top_products['daily_velocity'] = top_products['daily_velocity'].apply(lambda x: f"{x:.1f} units/day")
    top_products.columns = ['Product', 'Brand', 'Est. Revenue', 'Daily Velocity']
    st.dataframe(top_products, use_container_width=True, hide_index=True)

st.divider()

# Navigation guide
st.subheader("🧭 Dashboard Modules")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    ### 📈 Sales Intelligence
    - Estimated sales velocity
    - Revenue rankings
    - Trend analysis
    - Product performance
    """)

with col2:
    st.markdown("""
    ### 💰 Pricing Intelligence
    - Price change detection
    - Elasticity analysis
    - Competitive positioning
    - Margin opportunities
    """)

with col3:
    st.markdown("""
    ### 📦 Inventory Risk
    - Stockout analysis
    - Days of inventory
    - Restock patterns
    - ROI calculations
    """)

with col4:
    st.markdown("""
    ### 🔍 Exploration
    - Interactive data exploration
    - Multiple dataset views
    - Visual analysis tools
    - Self-service BI
    """)

st.info("👈 Use the sidebar to navigate between modules")

# Data freshness footer
st.divider()
st.caption(f"Data range: {metrics['data_summary']['date_range']['start'][:10]} → {metrics['data_summary']['date_range']['end'][:10]} | "
           f"{metrics['data_summary']['n_brands']} brands | {metrics['data_summary']['n_products']} products | "
           f"{metrics['data_summary']['total_rows']:,} data points")
