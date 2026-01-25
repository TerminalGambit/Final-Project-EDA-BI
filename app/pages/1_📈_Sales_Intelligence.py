"""
Sales Intelligence Module
=========================
Estimated sales analysis based on stock decrements.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data.loader import load_all_data, get_weekly_trends, compute_causal_analysis

st.set_page_config(page_title="Sales Intelligence", page_icon="📈", layout="wide")

st.title("📈 Sales Intelligence")
st.markdown("*Estimated competitor sales based on stock decrements*")

# Load data
with st.spinner("Loading data..."):
    df, daily, products, brands = load_all_data()
    weekly = get_weekly_trends(daily)

st.divider()

# Key Metrics
st.subheader("🎯 Key Sales Metrics")

col1, col2, col3, col4 = st.columns(4)

total_revenue = daily['estimated_revenue'].sum()
total_sales = daily['estimated_sales'].sum()
avg_daily_revenue = daily.groupby('date_day')['estimated_revenue'].sum().mean()
top_brand = brands.iloc[0]['brand']

with col1:
    st.metric("Total Est. Revenue", f"€{total_revenue:,.0f}")
with col2:
    st.metric("Total Units Sold", f"{total_sales:,.0f}")
with col3:
    st.metric("Avg Daily Revenue", f"€{avg_daily_revenue:,.0f}")
with col4:
    st.metric("Top Brand", top_brand)

st.divider()

# Tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Revenue Trends", "🏆 Product Rankings", "🏢 Brand Analysis", "📉 Declining Products", "💵 Profit Contribution", "🔍 Why Did This Happen?"])

with tab1:
    st.subheader("📈 Revenue Trends Over Time")
    
    # Daily revenue trend
    daily_totals = daily.groupby('date_day').agg({
        'estimated_revenue': 'sum',
        'estimated_sales': 'sum'
    }).reset_index()
    
    # Rolling average
    daily_totals['revenue_7d_avg'] = daily_totals['estimated_revenue'].rolling(7).mean()
    daily_totals['revenue_30d_avg'] = daily_totals['estimated_revenue'].rolling(30).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_totals['date_day'],
        y=daily_totals['estimated_revenue'],
        name='Daily Revenue',
        line=dict(color='lightblue', width=1),
        opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=daily_totals['date_day'],
        y=daily_totals['revenue_7d_avg'],
        name='7-Day Avg',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=daily_totals['date_day'],
        y=daily_totals['revenue_30d_avg'],
        name='30-Day Avg',
        line=dict(color='darkblue', width=2)
    ))
    fig.update_layout(
        title="Estimated Daily Revenue Trend",
        xaxis_title="Date",
        yaxis_title="Revenue (€)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekly by brand
    st.subheader("📊 Weekly Revenue By Brand")
    
    # Top 5 brands
    top_5_brands = brands.head(5)['brand'].tolist()
    weekly_top = weekly[weekly['brand'].isin(top_5_brands)]
    
    fig2 = px.area(
        weekly_top,
        x='week',
        y='estimated_revenue',
        color='brand',
        title="Weekly Revenue - Top 5 Brands"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("🏆 Product Performance Rankings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 15 by revenue
        st.markdown("#### 🥇 Top 15 Products by Revenue")
        top_15 = products.head(15)[['title', 'brand', 'total_revenue', 'daily_velocity', 'stockout_rate']].copy()
        top_15['total_revenue'] = top_15['total_revenue'].apply(lambda x: f"€{x:,.0f}")
        top_15['daily_velocity'] = top_15['daily_velocity'].apply(lambda x: f"{x:.2f}")
        top_15['stockout_rate'] = top_15['stockout_rate'].apply(lambda x: f"{x*100:.1f}%")
        top_15.columns = ['Product', 'Brand', 'Revenue', 'Units/Day', 'Stockout %']
        st.dataframe(top_15, use_container_width=True, hide_index=True)
    
    with col2:
        # Top by velocity
        st.markdown("#### ⚡ Top 15 Products by Sales Velocity")
        by_velocity = products.sort_values('daily_velocity', ascending=False).head(15)
        by_velocity = by_velocity[['title', 'brand', 'daily_velocity', 'total_revenue', 'avg_price']].copy()
        by_velocity['daily_velocity'] = by_velocity['daily_velocity'].apply(lambda x: f"{x:.2f}")
        by_velocity['total_revenue'] = by_velocity['total_revenue'].apply(lambda x: f"€{x:,.0f}")
        by_velocity['avg_price'] = by_velocity['avg_price'].apply(lambda x: f"€{x:.2f}")
        by_velocity.columns = ['Product', 'Brand', 'Units/Day', 'Revenue', 'Avg Price']
        st.dataframe(by_velocity, use_container_width=True, hide_index=True)
    
    # Revenue distribution chart
    st.markdown("#### 📊 Revenue Distribution By Product")
    fig = px.treemap(
        products.head(50),
        path=['brand', 'title'],
        values='total_revenue',
        color='daily_velocity',
        color_continuous_scale='Blues',
        title="Revenue Treemap - Top 50 Products (color = velocity)"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🏢 Brand Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand revenue bar chart
        fig = px.bar(
            brands.head(15),
            x='total_revenue',
            y='brand',
            orientation='h',
            title="Total Estimated Revenue by Brand",
            color='total_revenue',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue per product scatter
        fig = px.scatter(
            brands,
            x='n_products',
            y='revenue_per_product',
            size='total_revenue',
            color='brand',
            title="Revenue per Product vs Portfolio Size",
            hover_data=['total_revenue', 'daily_velocity']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Brand metrics table
    st.markdown("#### 📊 Brand Metrics Summary")
    brand_table = brands.copy()
    brand_table['total_revenue'] = brand_table['total_revenue'].apply(lambda x: f"€{x:,.0f}")
    brand_table['revenue_per_product'] = brand_table['revenue_per_product'].apply(lambda x: f"€{x:,.0f}")
    brand_table['daily_velocity'] = brand_table['daily_velocity'].apply(lambda x: f"{x:.1f}")
    brand_table = brand_table[['brand', 'total_revenue', 'total_sales', 'n_products', 'revenue_per_product', 'daily_velocity']]
    brand_table.columns = ['Brand', 'Total Revenue', 'Units Sold', 'Products', 'Rev/Product', 'Daily Velocity']
    st.dataframe(brand_table, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("⚠️ Products Requiring Attention")
    
    st.markdown("""
    **Declining Products** - These products show low sales velocity and may be candidates for:
    - Discontinuation to free up capital
    - Promotional pricing to clear inventory
    - Investigation into root cause (poor placement, competition, seasonality)
    """)
    
    # Bottom performers
    bottom_20 = products.sort_values('daily_velocity').head(20)
    bottom_20 = bottom_20[bottom_20['days_tracked'] > 30]  # Only products tracked for at least 30 days
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Lowest Velocity Products (>30 days tracked)")
        bottom_display = bottom_20[['title', 'brand', 'daily_velocity', 'total_revenue', 'stockout_rate', 'avg_price']].copy()
        bottom_display['daily_velocity'] = bottom_display['daily_velocity'].apply(lambda x: f"{x:.3f}")
        bottom_display['total_revenue'] = bottom_display['total_revenue'].apply(lambda x: f"€{x:,.0f}")
        bottom_display['stockout_rate'] = bottom_display['stockout_rate'].apply(lambda x: f"{x*100:.1f}%")
        bottom_display['avg_price'] = bottom_display['avg_price'].apply(lambda x: f"€{x:.2f}")
        bottom_display.columns = ['Product', 'Brand', 'Units/Day', 'Total Rev', 'Stockout %', 'Price']
        st.dataframe(bottom_display, use_container_width=True, hide_index=True)
    
    with col2:
        # Capital tied up in slow movers
        slow_movers_revenue = bottom_20['total_revenue'].sum()
        pct_of_total = (slow_movers_revenue / total_revenue) * 100
        
        st.metric("Revenue from Bottom 20", f"€{slow_movers_revenue:,.0f}")
        st.metric("% of Total Revenue", f"{pct_of_total:.1f}%")
        
        st.info(f"""
        **💡 Recommendation:**
        The bottom 20 products contribute only {pct_of_total:.1f}% of revenue.
        Consider reallocating this inventory capital to top performers.
        """)

with tab5:
    st.subheader("💰 Profit Contribution Analysis")
    
    st.markdown("""
    **Methodology:** We estimate profit contribution using assumed industry margins.
    Actual margins may vary - adjust the margin assumption below.
    """)
    
    # Margin assumption slider
    col1, col2 = st.columns([1, 3])
    with col1:
        assumed_margin = st.slider("Assumed Gross Margin %", 20, 60, 35)
    
    # Calculate profit contribution
    products_profit = products.copy()
    products_profit['estimated_profit'] = products_profit['total_revenue'] * (assumed_margin / 100)
    products_profit['profit_per_unit'] = products_profit['avg_price'] * (assumed_margin / 100)
    
    # Brand-level profit
    brand_profit = products_profit.groupby('brand').agg({
        'total_revenue': 'sum',
        'estimated_profit': 'sum',
        'total_sales': 'sum'
    }).reset_index()
    brand_profit['profit_margin'] = (brand_profit['estimated_profit'] / brand_profit['total_revenue']) * 100
    brand_profit = brand_profit.sort_values('estimated_profit', ascending=False)
    
    with col2:
        total_profit = products_profit['estimated_profit'].sum()
        st.metric("Total Estimated Profit", f"€{total_profit:,.0f}", help=f"At {assumed_margin}% margin")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Brand Profit Contribution")
        fig = px.bar(
            brand_profit.head(12),
            x='brand',
            y='estimated_profit',
            color='estimated_profit',
            title="Estimated Profit by Brand",
            color_continuous_scale='Greens'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Profit per Unit (Pricing Power)")
        
        # Products with highest profit per unit
        top_profit_products = products_profit.nlargest(15, 'profit_per_unit')
        
        fig = px.bar(
            top_profit_products,
            x='title',
            y='profit_per_unit',
            color='brand',
            title="Top 15 Products by Profit/Unit"
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Pareto analysis (80/20 rule)
    st.markdown("#### 🎯 Pareto Analysis (80/20 Rule)")
    
    products_sorted = products_profit.sort_values('estimated_profit', ascending=False)
    products_sorted['cumulative_profit'] = products_sorted['estimated_profit'].cumsum()
    products_sorted['cumulative_pct'] = (products_sorted['cumulative_profit'] / total_profit) * 100
    products_sorted['product_rank'] = range(1, len(products_sorted) + 1)
    products_sorted['product_pct'] = (products_sorted['product_rank'] / len(products_sorted)) * 100
    
    # Find 80% threshold
    products_80 = products_sorted[products_sorted['cumulative_pct'] <= 80]
    n_products_80 = len(products_80)
    pct_products_80 = (n_products_80 / len(products_sorted)) * 100
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=products_sorted['product_pct'],
            y=products_sorted['cumulative_pct'],
            mode='lines',
            name='Cumulative Profit %',
            line=dict(color='blue', width=2)
        ))
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% profit")
        fig.add_vline(x=pct_products_80, line_dash="dash", line_color="red", annotation_text=f"{pct_products_80:.0f}% products")
        fig.update_layout(
            title="Profit Concentration Curve",
            xaxis_title="% of Products",
            yaxis_title="% of Cumulative Profit"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Products for 80% Profit", f"{n_products_80} products")
        st.metric("% of Portfolio", f"{pct_products_80:.1f}%")
        
        # Bottom 20% contribution
        bottom_20_pct = 100 - products_sorted[products_sorted['product_pct'] <= 80]['cumulative_pct'].max()
        st.metric("Bottom 20% Contribution", f"{bottom_20_pct:.1f}%")
        
        st.info(f"""
        **Key Insight:**
        
        {pct_products_80:.0f}% of products generate 80% of profit.
        
        Consider:
        - Focus inventory on top {n_products_80} products
        - Review bottom {len(products_sorted) - n_products_80} for discontinuation
        - Potential capital reallocation: €{products_sorted.tail(int(len(products_sorted)*0.2))['estimated_profit'].sum():,.0f}
        """)

with tab6:
    st.subheader("🔍 Why Did This Happen? - Causal Analysis")
    st.markdown("*Understanding the WHY behind revenue anomalies, not just the WHAT*")
    
    # Load causal analysis
    causal = compute_causal_analysis()
    
    # Weekly pattern insight
    weekly_pattern = causal.get('weekly_pattern', {})
    if weekly_pattern:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Day", weekly_pattern.get('best_day', 'N/A'))
        with col2:
            st.metric("Worst Day", weekly_pattern.get('worst_day', 'N/A'))
        with col3:
            st.metric("Day-to-Day Variance", f"{weekly_pattern.get('variance_pct', 0):.0f}%")
        
        st.info(f"💡 **Pattern Insight:** {weekly_pattern.get('explanation', '')}")
    
    st.divider()
    
    # Anomaly narratives
    st.markdown("### 📊 Recent Revenue Anomalies Explained")
    st.markdown(f"*{causal.get('total_anomalies', 0)} significant anomalies detected (>30% deviation from 7-day average)*")
    
    anomalies = causal.get('anomalies', [])
    if anomalies:
        for anomaly in anomalies:
            direction = anomaly.get('direction', 'change')
            deviation = anomaly.get('deviation_pct', 0)
            
            if direction == 'spike':
                st.success(f"""
                **{anomaly.get('narrative', '')}**
                
                Revenue: €{anomaly.get('revenue', 0):,.0f} | Day: {anomaly.get('day_of_week', '')} | Date: {anomaly.get('date', '')}
                """)
            else:
                st.warning(f"""
                **{anomaly.get('narrative', '')}**
                
                Revenue: €{anomaly.get('revenue', 0):,.0f} | Day: {anomaly.get('day_of_week', '')} | Date: {anomaly.get('date', '')}
                """)
    else:
        st.info("No significant anomalies detected in recent data.")
    
    # Investigation guidance
    with st.expander("🔬 How to Investigate Further"):
        st.markdown("""
        ### When You See Unexplained Anomalies:
        
        **For Revenue Spikes:**
        1. Check if marketing campaigns were running
        2. Look for competitor stockouts (you may have captured their demand)
        3. Verify if any products were featured externally (press, influencers)
        4. Check for seasonal events (holidays, pay days)
        
        **For Revenue Drops:**
        1. Check stockout levels on that day
        2. Look for competitor promotions
        3. Verify no website/platform issues occurred
        4. Check if any major products had price increases
        
        **Data Limitations:**
        - This analysis correlates internal data only
        - External factors (marketing, competitor moves) require separate data sources
        - Consider setting up tracking for marketing calendar events
        """)

# Actionable insights
st.divider()
st.subheader("💡 Actionable Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(f"""
    **Top Performer:** {products.iloc[0]['title']}
    - Brand: {products.iloc[0]['brand']}
    - Est. Revenue: €{products.iloc[0]['total_revenue']:,.0f}
    - Velocity: {products.iloc[0]['daily_velocity']:.2f} units/day
    
    *Consider expanding this product line or negotiating better terms.*
    """)

with col2:
    high_velocity_low_stock = products[
        (products['daily_velocity'] > products['daily_velocity'].median()) & 
        (products['stockout_rate'] > 0.05)
    ].head(3)
    
    if not high_velocity_low_stock.empty:
        st.warning(f"""
        **High-Velocity Products with Stockout Issues:**
        
        {', '.join(high_velocity_low_stock['title'].tolist()[:3])}
        
        *These products have strong demand but supply issues. 
        Estimated lost revenue opportunity exists.*
        """)
