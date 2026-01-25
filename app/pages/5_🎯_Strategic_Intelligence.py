"""
Strategic Intelligence Module
=============================
Executive KPI scorecards, anomaly detection, and strategic risk analysis.
Focuses on what the CEO DOESN'T already know.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.loader import load_all_data, get_weekly_trends

st.set_page_config(page_title="Strategic Intelligence", page_icon="🎯", layout="wide")

st.title("🎯 Strategic Intelligence")
st.markdown("*Hidden insights, concentration risks, and strategic opportunities the CEO needs to know*")

# Load data
with st.spinner("Loading data..."):
    df, daily, products, brands = load_all_data()
    weekly = get_weekly_trends(daily)

st.divider()

# =============================================================================
# SECTION 1: WHAT YOU DON'T KNOW (Hidden Risks & Opportunities)
# =============================================================================
st.header("🔮 What You Don't Know")
st.markdown("*Strategic blind spots and hidden patterns in your competitive data*")

col1, col2, col3 = st.columns(3)

# 1. Brand Concentration Risk
with col1:
    st.subheader("⚠️ Brand Concentration Risk")
    
    top_brand = brands.iloc[0]
    top_brand_pct = (top_brand['total_revenue'] / brands['total_revenue'].sum()) * 100
    top_3_brands = brands.head(3)
    top_3_pct = (top_3_brands['total_revenue'].sum() / brands['total_revenue'].sum()) * 100
    
    # Herfindahl-Hirschman Index (HHI) for concentration
    revenue_shares = brands['total_revenue'] / brands['total_revenue'].sum()
    hhi = (revenue_shares ** 2).sum() * 10000
    
    if hhi > 2500:
        risk_level = "HIGH"
        risk_color = "red"
    elif hhi > 1500:
        risk_level = "MODERATE"
        risk_color = "orange"
    else:
        risk_level = "LOW"
        risk_color = "green"
    
    st.metric("Concentration Risk", risk_level)
    st.metric("Top Brand Share", f"{top_brand_pct:.1f}%", help=f"{top_brand['brand']}")
    st.metric("Top 3 Brands Share", f"{top_3_pct:.1f}%")
    
    st.warning(f"""
    **Risk Alert:** {top_brand['brand']} accounts for {top_brand_pct:.1f}% of estimated revenue.
    
    If this brand has supply issues or exits the market, you lose €{top_brand['total_revenue']:,.0f} in annual revenue.
    """)
    
    with st.expander("📝 Next Steps & Success Metrics"):
        st.markdown(f"""
        **Immediate Actions:**
        1. Identify 2-3 alternative brands in same category
        2. Request backup supplier quotes for top {top_brand['brand']} SKUs
        3. Document contingency plan for supply disruption
        
        **Success Metrics (track weekly):**
        - Top brand share (target: <15%, current: {top_brand_pct:.1f}%)
        - HHI concentration index (target: <1500, current: {hhi:.0f})
        - Alternative supplier readiness score
        """)

# 2. Hidden Dependencies
with col2:
    st.subheader("🔗 Hidden Dependencies")
    
    # Single-product brands (high risk)
    single_product_brands = brands[brands['n_products'] <= 2]
    single_product_revenue = single_product_brands['total_revenue'].sum()
    single_product_pct = (single_product_revenue / brands['total_revenue'].sum()) * 100
    
    st.metric("Single-Product Brand Risk", f"€{single_product_revenue:,.0f}")
    st.metric("% of Revenue at Risk", f"{single_product_pct:.1f}%")
    
    if not single_product_brands.empty:
        st.info(f"""
        **{len(single_product_brands)} brands** have only 1-2 products:
        
        {', '.join(single_product_brands['brand'].tolist())}
        
        *These represent concentrated risk - if one SKU fails, the entire brand contribution is lost.*
        """)
        
        with st.expander("📝 Next Steps & Success Metrics"):
            st.markdown(f"""
            **Immediate Actions:**
            1. For each single-product brand, identify expansion opportunities
            2. Negotiate minimum product range commitments with suppliers
            3. Consider delisting brands that can't expand within 6 months
            
            **Success Metrics (track monthly):**
            - Single-product brand count (target: 0, current: {len(single_product_brands)})
            - Revenue at risk from single-product brands (target: <5%, current: {single_product_pct:.1f}%)
            """)

# 3. Velocity vs Revenue Mismatch
with col3:
    st.subheader("📊 Efficiency Anomalies")
    
    # Products with high velocity but low revenue (pricing opportunity)
    products_copy = products.copy()
    products_copy['velocity_rank'] = products_copy['daily_velocity'].rank(ascending=False)
    products_copy['revenue_rank'] = products_copy['total_revenue'].rank(ascending=False)
    products_copy['rank_gap'] = products_copy['velocity_rank'] - products_copy['revenue_rank']
    
    # High velocity, low revenue = underpriced
    underpriced = products_copy[products_copy['rank_gap'] < -50].head(5)
    # Low velocity, high revenue = overpriced or luxury
    overpriced = products_copy[products_copy['rank_gap'] > 50].head(5)
    
    n_underpriced = len(products_copy[products_copy['rank_gap'] < -30])
    n_overpriced = len(products_copy[products_copy['rank_gap'] > 30])
    
    st.metric("Underpriced Products", n_underpriced, help="High sales velocity but low revenue rank")
    st.metric("Potentially Overpriced", n_overpriced, help="Low velocity but high revenue contribution")
    
    if n_underpriced > 0:
        potential_revenue = n_underpriced * 500  # Conservative estimate
        st.success(f"""
        **Pricing Opportunity:** {n_underpriced} products are selling fast but priced below their value.
        
        Potential additional annual revenue: **€{potential_revenue:,.0f}+**
        """)
        
        with st.expander("📝 Next Steps & Success Metrics"):
            st.markdown(f"""
            **Immediate Actions:**
            1. Go to 🎬 Action Center for detailed pricing recommendations
            2. Review top 10 underpriced products with merchandising team
            3. Implement Phase 1 price adjustments within 2 weeks
            
            **Success Metrics (track weekly):**
            - Underpriced product count (target: <10, current: {n_underpriced})
            - Revenue per unit (should increase)
            - Margin improvement vs baseline
            """)

st.divider()

# =============================================================================
# SECTION 2: KPI SCORECARD WITH TRENDS
# =============================================================================
st.header("📈 Executive KPI Scorecard")
st.markdown("*Performance tracking with trend analysis and anomaly detection*")

# Calculate weekly KPIs
weekly_totals = daily.groupby('date_day').agg({
    'estimated_revenue': 'sum',
    'estimated_sales': 'sum',
    'is_stockout': 'sum',
    'product_id': 'nunique'
}).reset_index()
weekly_totals['date_day'] = pd.to_datetime(weekly_totals['date_day'])
weekly_totals['week'] = weekly_totals['date_day'].dt.to_period('W').dt.start_time
weekly_totals['stockout_rate'] = weekly_totals['is_stockout'] / weekly_totals['product_id']

weekly_kpis = weekly_totals.groupby('week').agg({
    'estimated_revenue': 'sum',
    'estimated_sales': 'sum',
    'stockout_rate': 'mean'
}).reset_index()

# Calculate WoW changes
weekly_kpis['revenue_wow'] = weekly_kpis['estimated_revenue'].pct_change() * 100
weekly_kpis['sales_wow'] = weekly_kpis['estimated_sales'].pct_change() * 100
weekly_kpis['stockout_wow'] = weekly_kpis['stockout_rate'].diff() * 100

# Detect anomalies (Z-score > 2)
def detect_anomalies(series, threshold=2):
    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    return abs(z_scores) > threshold

weekly_kpis['revenue_anomaly'] = detect_anomalies(weekly_kpis['estimated_revenue'])
weekly_kpis['sales_anomaly'] = detect_anomalies(weekly_kpis['estimated_sales'])

# Latest metrics
latest = weekly_kpis.iloc[-1]
prev = weekly_kpis.iloc[-2] if len(weekly_kpis) > 1 else latest

col1, col2, col3, col4 = st.columns(4)

with col1:
    delta = f"{latest['revenue_wow']:+.1f}%" if pd.notna(latest['revenue_wow']) else None
    st.metric("Weekly Revenue", f"€{latest['estimated_revenue']:,.0f}", delta=delta)
    
with col2:
    delta = f"{latest['sales_wow']:+.1f}%" if pd.notna(latest['sales_wow']) else None
    st.metric("Weekly Units", f"{latest['estimated_sales']:,.0f}", delta=delta)

with col3:
    delta = f"{latest['stockout_wow']:+.1f}pp" if pd.notna(latest['stockout_wow']) else None
    st.metric("Stockout Rate", f"{latest['stockout_rate']*100:.1f}%", delta=delta, delta_color="inverse")

with col4:
    # Health score (composite)
    revenue_health = min(100, (latest['estimated_revenue'] / weekly_kpis['estimated_revenue'].mean()) * 50)
    stockout_health = max(0, 50 - (latest['stockout_rate'] * 500))
    health_score = revenue_health + stockout_health
    
    st.metric("Health Score", f"{health_score:.0f}/100")

# Trend charts
st.subheader("📊 KPI Trends (Last 52 Weeks)")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Weekly Revenue", "Weekly Units Sold", "Stockout Rate", "Revenue Growth %"),
    vertical_spacing=0.12
)

# Revenue trend
fig.add_trace(
    go.Scatter(x=weekly_kpis['week'], y=weekly_kpis['estimated_revenue'], 
               mode='lines', name='Revenue', line=dict(color='blue')),
    row=1, col=1
)
# Add anomaly markers
anomalies = weekly_kpis[weekly_kpis['revenue_anomaly']]
fig.add_trace(
    go.Scatter(x=anomalies['week'], y=anomalies['estimated_revenue'],
               mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')),
    row=1, col=1
)

# Units trend
fig.add_trace(
    go.Scatter(x=weekly_kpis['week'], y=weekly_kpis['estimated_sales'],
               mode='lines', name='Units', line=dict(color='green')),
    row=1, col=2
)

# Stockout trend
fig.add_trace(
    go.Scatter(x=weekly_kpis['week'], y=weekly_kpis['stockout_rate']*100,
               mode='lines', name='Stockout %', line=dict(color='red')),
    row=2, col=1
)

# Growth rate
fig.add_trace(
    go.Bar(x=weekly_kpis['week'], y=weekly_kpis['revenue_wow'],
           name='WoW Growth', marker_color=weekly_kpis['revenue_wow'].apply(lambda x: 'green' if x > 0 else 'red')),
    row=2, col=2
)

fig.update_layout(height=500, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Anomaly alerts
if weekly_kpis['revenue_anomaly'].any() or weekly_kpis['sales_anomaly'].any():
    st.warning(f"""
    **⚠️ Anomalies Detected:** 
    - {weekly_kpis['revenue_anomaly'].sum()} unusual revenue weeks
    - {weekly_kpis['sales_anomaly'].sum()} unusual sales weeks
    
    *Red X markers on charts indicate statistical anomalies (>2 standard deviations)*
    """)

st.divider()

# =============================================================================
# SECTION 3: COMPETITIVE POSITION ANALYSIS
# =============================================================================
st.header("🏆 Competitive Position Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Brand Portfolio Matrix")
    
    # BCG-style matrix: Revenue vs Growth
    # Calculate brand growth (first half vs second half)
    daily_copy = daily.copy()
    daily_copy['period'] = daily_copy['date_day'].apply(
        lambda x: 'H1' if x < daily_copy['date_day'].median() else 'H2'
    )
    
    brand_periods = daily_copy.groupby(['brand', 'period'])['estimated_revenue'].sum().unstack()
    brand_periods['growth'] = ((brand_periods['H2'] - brand_periods['H1']) / brand_periods['H1']) * 100
    brand_periods['total'] = brand_periods['H1'] + brand_periods['H2']
    brand_periods = brand_periods.reset_index()
    
    # Drop rows with NaN values that would break the scatter plot
    brand_periods = brand_periods.dropna(subset=['growth', 'total'])
    
    # Classify into quadrants
    avg_growth = brand_periods['growth'].median()
    avg_revenue = brand_periods['total'].median()
    
    brand_periods['quadrant'] = brand_periods.apply(
        lambda r: '⭐ Star' if r['growth'] > avg_growth and r['total'] > avg_revenue
        else '❓ Question Mark' if r['growth'] > avg_growth and r['total'] <= avg_revenue
        else '🐄 Cash Cow' if r['growth'] <= avg_growth and r['total'] > avg_revenue
        else '🐕 Dog',
        axis=1
    )
    
    fig = px.scatter(
        brand_periods,
        x='total',
        y='growth',
        color='quadrant',
        size='total',
        text='brand',
        title="Brand Portfolio Matrix (Revenue vs Growth)",
        labels={'total': 'Total Revenue (€)', 'growth': 'Growth Rate (%)'}
    )
    fig.add_hline(y=avg_growth, line_dash="dash", line_color="gray")
    fig.add_vline(x=avg_revenue, line_dash="dash", line_color="gray")
    fig.update_traces(textposition='top center', textfont_size=8)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary counts
    quadrant_counts = brand_periods['quadrant'].value_counts()
    st.markdown(f"""
    **Portfolio Summary:**
    - ⭐ Stars (high growth, high revenue): {quadrant_counts.get('⭐ Star', 0)}
    - ❓ Question Marks (high growth, low revenue): {quadrant_counts.get('❓ Question Mark', 0)}
    - 🐄 Cash Cows (low growth, high revenue): {quadrant_counts.get('🐄 Cash Cow', 0)}
    - 🐕 Dogs (low growth, low revenue): {quadrant_counts.get('🐕 Dog', 0)}
    """)

with col2:
    st.subheader("Revenue Velocity Efficiency")
    
    # Revenue per unit (efficiency)
    brands_eff = brands.copy()
    brands_eff['revenue_per_unit'] = brands_eff['total_revenue'] / brands_eff['total_sales']
    brands_eff = brands_eff.sort_values('revenue_per_unit', ascending=False)
    
    fig = px.bar(
        brands_eff.head(15),
        x='brand',
        y='revenue_per_unit',
        color='total_revenue',
        title="Revenue per Unit Sold (Pricing Power)",
        labels={'revenue_per_unit': '€ per Unit', 'brand': ''},
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top/bottom efficiency
    top_eff = brands_eff.iloc[0]
    bottom_eff = brands_eff.iloc[-1]
    
    st.info(f"""
    **Efficiency Gap:**
    - Highest: {top_eff['brand']} at €{top_eff['revenue_per_unit']:.2f}/unit
    - Lowest: {bottom_eff['brand']} at €{bottom_eff['revenue_per_unit']:.2f}/unit
    
    *Gap represents pricing power difference of {(top_eff['revenue_per_unit']/bottom_eff['revenue_per_unit']):.1f}x*
    """)

st.divider()

# =============================================================================
# SECTION 4: STRATEGIC RECOMMENDATIONS
# =============================================================================
st.header("💡 Strategic Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Immediate Actions")
    
    # Get dogs for discontinuation
    dogs = brand_periods[brand_periods['quadrant'] == '🐕 Dog']['brand'].tolist()
    # Get stars for expansion
    stars = brand_periods[brand_periods['quadrant'] == '⭐ Star']['brand'].tolist()
    
    st.markdown(f"""
    **1. Portfolio Rebalancing**
    - Review/exit: {', '.join(dogs[:3]) if dogs else 'None identified'}
    - Expand: {', '.join(stars[:3]) if stars else 'Top performers'}
    
    **2. Concentration Risk Mitigation**
    - Diversify from {top_brand['brand']} dependency
    - Target: Reduce to <15% single-brand share
    
    **3. Pricing Optimization**
    - {n_underpriced} products identified for price increase
    - Estimated opportunity: €{n_underpriced * 500:,}+/year
    """)

with col2:
    st.markdown("### 📅 30-Day Priorities")
    
    # High-impact, quick-win items
    high_stockout = products[products['stockout_rate'] > 0.15].head(5)
    
    st.markdown(f"""
    **1. Fix Supply Chain Gaps**
    - {len(high_stockout)} products with >15% stockout
    - Est. lost revenue: €{(high_stockout['daily_velocity'] * high_stockout['avg_price'] * high_stockout['stockout_days']).sum():,.0f}
    
    **2. Implement Price Tests**
    - Start with top 5 underpriced products
    - A/B test 5-10% increases
    
    **3. Set Up KPI Monitoring**
    - Weekly revenue threshold: €{weekly_kpis['estimated_revenue'].quantile(0.25):,.0f}
    - Alert if below threshold
    """)

with col3:
    st.markdown("### 🔮 90-Day Strategy")
    
    st.markdown(f"""
    **1. Brand Partnership Review**
    - Negotiate better terms with Stars
    - Performance review for Dogs
    
    **2. New Brand Evaluation**
    - Target gap: Premium segment (€50+)
    - Current share: {(products[products['avg_price'] > 50]['total_revenue'].sum() / products['total_revenue'].sum() * 100):.1f}%
    
    **3. Predictive Inventory System**
    - Implement demand forecasting
    - Target: 50% stockout reduction
    - Expected ROI: 3-5x holding cost
    """)

# Final insight box
st.divider()
st.success(f"""
### 🎓 Key Strategic Insight

**Your biggest blind spot:** {top_brand['brand']} concentration.

At {top_brand_pct:.1f}% of revenue, a single supplier disruption could cost €{top_brand['total_revenue']/12:,.0f}/month.
Combined with {len(single_product_brands)} single-product brands representing €{single_product_revenue:,.0f}, 
your **portfolio risk exposure is €{top_brand['total_revenue'] + single_product_revenue:,.0f}** ({((top_brand['total_revenue'] + single_product_revenue)/brands['total_revenue'].sum()*100):.1f}% of total revenue).

**Recommended action:** Diversify brand portfolio and develop contingency plans for top 3 brands.

👉 **Go to 🎬 Action Center** for detailed, executable recommendations with investment cases.
""")
