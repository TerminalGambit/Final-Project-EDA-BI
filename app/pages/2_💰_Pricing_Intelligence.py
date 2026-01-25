"""
Pricing Intelligence Module
===========================
Price tracking, elasticity analysis, and competitive positioning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data.loader import load_all_data, get_price_changes

st.set_page_config(page_title="Pricing Intelligence", page_icon="💰", layout="wide")

st.title("💰 Pricing Intelligence")
st.markdown("*Competitor pricing strategy analysis and opportunities*")

# Load data
with st.spinner("Loading data..."):
    df, daily, products, brands = load_all_data()
    price_changes = get_price_changes(df)

st.divider()

# Key Metrics
st.subheader("🎯 Pricing Overview")

col1, col2, col3, col4 = st.columns(4)

avg_price = df['offer_price'].mean()
price_range = f"€{df['offer_price'].min():.0f} - €{df['offer_price'].max():.0f}"
n_price_changes = len(price_changes)
products_with_changes = price_changes['product_id'].nunique()

with col1:
    st.metric("Average Price", f"€{avg_price:.2f}")
with col2:
    st.metric("Price Range", price_range)
with col3:
    st.metric("Price Change Events", f"{n_price_changes:,}")
with col4:
    st.metric("Products with Price Changes", products_with_changes)

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Distribution", "🔄 Price Changes", "📈 Elasticity Analysis", "💡 Opportunities"])

with tab1:
    st.subheader("Price Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution histogram
        fig = px.histogram(
            products,
            x='avg_price',
            nbins=30,
            title="Price Distribution Across Products",
            labels={'avg_price': 'Average Price (€)'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price by brand box plot
        fig = px.box(
            df.drop_duplicates(['product_id', 'brand']).merge(products[['product_id', 'avg_price']], on='product_id'),
            x='brand',
            y='avg_price',
            title="Price Distribution by Brand"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Price segments analysis
    st.subheader("Price Segment Analysis")
    
    # Create price segments
    products_copy = products.copy()
    products_copy['price_segment'] = pd.cut(
        products_copy['avg_price'],
        bins=[0, 10, 20, 30, 50, 200],
        labels=['€0-10', '€10-20', '€20-30', '€30-50', '€50+']
    )
    
    segment_analysis = products_copy.groupby('price_segment', observed=True).agg({
        'product_id': 'count',
        'total_revenue': 'sum',
        'daily_velocity': 'mean'
    }).reset_index()
    segment_analysis.columns = ['Price Segment', 'Products', 'Total Revenue', 'Avg Velocity']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            segment_analysis,
            values='Total Revenue',
            names='Price Segment',
            title="Revenue Share by Price Segment"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            segment_analysis,
            x='Price Segment',
            y='Avg Velocity',
            title="Average Sales Velocity by Price Segment",
            color='Avg Velocity',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Price Change Detection")
    
    if len(price_changes) > 0:
        # Timeline of price changes
        price_changes_daily = price_changes.copy()
        price_changes_daily['date_day'] = price_changes_daily['date'].dt.date
        changes_by_day = price_changes_daily.groupby('date_day').size().reset_index(name='n_changes')
        changes_by_day['date_day'] = pd.to_datetime(changes_by_day['date_day'])
        
        fig = px.line(
            changes_by_day,
            x='date_day',
            y='n_changes',
            title="Price Change Events Over Time"
        )
        fig.update_traces(fill='tozeroy')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Recent price increases
            st.markdown("#### 📈 Recent Price Increases")
            increases = price_changes[price_changes['price_change'] > 0].sort_values('date', ascending=False).head(15)
            if not increases.empty:
                increases_display = increases[['date', 'title', 'brand', 'offer_price', 'price_change', 'price_change_pct']].copy()
                increases_display['date'] = increases_display['date'].dt.strftime('%Y-%m-%d')
                increases_display['offer_price'] = increases_display['offer_price'].apply(lambda x: f"€{x:.2f}")
                increases_display['price_change'] = increases_display['price_change'].apply(lambda x: f"+€{x:.2f}")
                increases_display['price_change_pct'] = increases_display['price_change_pct'].apply(lambda x: f"+{x:.1f}%")
                increases_display.columns = ['Date', 'Product', 'Brand', 'New Price', 'Change', '% Change']
                st.dataframe(increases_display, use_container_width=True, hide_index=True)
            else:
                st.info("No recent price increases detected")
        
        with col2:
            # Recent price decreases
            st.markdown("#### 📉 Recent Price Decreases")
            decreases = price_changes[price_changes['price_change'] < 0].sort_values('date', ascending=False).head(15)
            if not decreases.empty:
                decreases_display = decreases[['date', 'title', 'brand', 'offer_price', 'price_change', 'price_change_pct']].copy()
                decreases_display['date'] = decreases_display['date'].dt.strftime('%Y-%m-%d')
                decreases_display['offer_price'] = decreases_display['offer_price'].apply(lambda x: f"€{x:.2f}")
                decreases_display['price_change'] = decreases_display['price_change'].apply(lambda x: f"€{x:.2f}")
                decreases_display['price_change_pct'] = decreases_display['price_change_pct'].apply(lambda x: f"{x:.1f}%")
                decreases_display.columns = ['Date', 'Product', 'Brand', 'New Price', 'Change', '% Change']
                st.dataframe(decreases_display, use_container_width=True, hide_index=True)
            else:
                st.info("No recent price decreases detected")
        
        # Price change magnitude distribution
        st.markdown("#### Price Change Magnitude Distribution")
        fig = px.histogram(
            price_changes,
            x='price_change_pct',
            nbins=50,
            title="Distribution of Price Change Percentages",
            labels={'price_change_pct': 'Price Change (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No significant price changes detected in the data")

with tab3:
    st.subheader("Price Elasticity Estimation")
    
    st.markdown("""
    **Methodology:** We estimate price elasticity by comparing sales velocity before and after price changes.
    Products with minimal velocity change after price increase are considered **inelastic** (safe to raise prices).
    """)
    
    if len(price_changes) > 0:
        # For each product with price changes, compute before/after velocity
        # This is a simplified estimation
        
        products_with_changes = price_changes['product_id'].unique()
        
        elasticity_data = []
        for pid in products_with_changes[:50]:  # Limit to 50 for performance
            product_daily = daily[daily['product_id'] == pid].sort_values('date_day')
            if len(product_daily) < 30:
                continue
            
            # Get price change dates for this product
            pc_dates = price_changes[price_changes['product_id'] == pid]['date'].dt.date.tolist()
            
            for pc_date in pc_dates[:1]:  # First price change only
                before = product_daily[product_daily['date_day'] < pd.Timestamp(pc_date)]
                after = product_daily[product_daily['date_day'] >= pd.Timestamp(pc_date)]
                
                if len(before) >= 7 and len(after) >= 7:
                    before_velocity = before['estimated_sales'].mean()
                    after_velocity = after['estimated_sales'].mean()
                    
                    if before_velocity > 0:
                        velocity_change_pct = ((after_velocity - before_velocity) / before_velocity) * 100
                        
                        price_info = price_changes[
                            (price_changes['product_id'] == pid) & 
                            (price_changes['date'].dt.date == pc_date)
                        ].iloc[0]
                        
                        elasticity_data.append({
                            'product_id': pid,
                            'title': price_info['title'],
                            'brand': price_info['brand'],
                            'price_change_pct': price_info['price_change_pct'],
                            'velocity_change_pct': velocity_change_pct,
                            'before_velocity': before_velocity,
                            'after_velocity': after_velocity
                        })
        
        if elasticity_data:
            elasticity_df = pd.DataFrame(elasticity_data)
            
            # Scatter plot: price change vs velocity change
            fig = px.scatter(
                elasticity_df,
                x='price_change_pct',
                y='velocity_change_pct',
                color='brand',
                hover_data=['title'],
                title="Price Change vs Sales Velocity Change",
                labels={
                    'price_change_pct': 'Price Change (%)',
                    'velocity_change_pct': 'Velocity Change (%)'
                }
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Inelastic products (price up, velocity stable/up)
                st.markdown("#### 🟢 Inelastic Products (Safe to Raise Prices)")
                inelastic = elasticity_df[
                    (elasticity_df['price_change_pct'] > 0) & 
                    (elasticity_df['velocity_change_pct'] > -10)
                ].sort_values('price_change_pct', ascending=False)
                
                if not inelastic.empty:
                    inelastic_display = inelastic[['title', 'brand', 'price_change_pct', 'velocity_change_pct']].head(10).copy()
                    inelastic_display['price_change_pct'] = inelastic_display['price_change_pct'].apply(lambda x: f"+{x:.1f}%")
                    inelastic_display['velocity_change_pct'] = inelastic_display['velocity_change_pct'].apply(lambda x: f"{x:+.1f}%")
                    inelastic_display.columns = ['Product', 'Brand', 'Price Δ', 'Velocity Δ']
                    st.dataframe(inelastic_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No clear inelastic products identified")
            
            with col2:
                # Elastic products (price up, velocity down significantly)
                st.markdown("#### 🔴 Elastic Products (Price Sensitive)")
                elastic = elasticity_df[
                    (elasticity_df['price_change_pct'] > 0) & 
                    (elasticity_df['velocity_change_pct'] < -20)
                ].sort_values('velocity_change_pct')
                
                if not elastic.empty:
                    elastic_display = elastic[['title', 'brand', 'price_change_pct', 'velocity_change_pct']].head(10).copy()
                    elastic_display['price_change_pct'] = elastic_display['price_change_pct'].apply(lambda x: f"+{x:.1f}%")
                    elastic_display['velocity_change_pct'] = elastic_display['velocity_change_pct'].apply(lambda x: f"{x:.1f}%")
                    elastic_display.columns = ['Product', 'Brand', 'Price Δ', 'Velocity Δ']
                    st.dataframe(elastic_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No highly elastic products identified")
        else:
            st.warning("Insufficient data to compute elasticity estimates")
    else:
        st.info("No price changes available for elasticity analysis")

with tab4:
    st.subheader("💡 Pricing Opportunities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Underpriced High-Performers")
        st.markdown("*Products with high velocity but below-average prices for their brand*")
        
        # Find products with velocity above median but price below brand average
        brand_avg_price = products.groupby('brand')['avg_price'].mean().to_dict()
        products_copy = products.copy()
        products_copy['brand_avg_price'] = products_copy['brand'].map(brand_avg_price)
        products_copy['price_vs_brand'] = products_copy['avg_price'] - products_copy['brand_avg_price']
        
        underpriced = products_copy[
            (products_copy['daily_velocity'] > products_copy['daily_velocity'].median()) &
            (products_copy['price_vs_brand'] < -3)  # At least €3 below brand average
        ].sort_values('daily_velocity', ascending=False)
        
        if not underpriced.empty:
            underpriced_display = underpriced[['title', 'brand', 'avg_price', 'brand_avg_price', 'daily_velocity']].head(10).copy()
            underpriced_display['avg_price'] = underpriced_display['avg_price'].apply(lambda x: f"€{x:.2f}")
            underpriced_display['brand_avg_price'] = underpriced_display['brand_avg_price'].apply(lambda x: f"€{x:.2f}")
            underpriced_display['daily_velocity'] = underpriced_display['daily_velocity'].apply(lambda x: f"{x:.2f}")
            underpriced_display.columns = ['Product', 'Brand', 'Current Price', 'Brand Avg', 'Velocity']
            st.dataframe(underpriced_display, use_container_width=True, hide_index=True)
            
            potential_increase = len(underpriced) * 2  # Assume €2 potential increase
            st.success(f"**Potential margin opportunity:** €{potential_increase:.0f}/day additional revenue if prices aligned with brand averages")
        else:
            st.info("No significant underpricing opportunities detected")
    
    with col2:
        st.markdown("#### 🏷️ Overpriced Low-Performers")
        st.markdown("*Products with low velocity but above-average prices - consider discounting*")
        
        overpriced = products_copy[
            (products_copy['daily_velocity'] < products_copy['daily_velocity'].quantile(0.25)) &
            (products_copy['price_vs_brand'] > 5)  # At least €5 above brand average
        ].sort_values('daily_velocity')
        
        if not overpriced.empty:
            overpriced_display = overpriced[['title', 'brand', 'avg_price', 'brand_avg_price', 'daily_velocity']].head(10).copy()
            overpriced_display['avg_price'] = overpriced_display['avg_price'].apply(lambda x: f"€{x:.2f}")
            overpriced_display['brand_avg_price'] = overpriced_display['brand_avg_price'].apply(lambda x: f"€{x:.2f}")
            overpriced_display['daily_velocity'] = overpriced_display['daily_velocity'].apply(lambda x: f"{x:.3f}")
            overpriced_display.columns = ['Product', 'Brand', 'Current Price', 'Brand Avg', 'Velocity']
            st.dataframe(overpriced_display, use_container_width=True, hide_index=True)
            
            st.warning("**Recommendation:** Consider promotional pricing or bundle offers to move these products")
        else:
            st.info("No significant overpricing issues detected")
    
    # ROI calculation
    st.divider()
    st.markdown("### 💵 Pricing ROI Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_increase = st.slider("Price increase %", 1, 20, 5)
    with col2:
        affected_products = st.slider("% of products affected", 10, 100, 30)
    with col3:
        demand_impact = st.slider("Assumed demand drop %", 0, 30, 10)
    
    # Calculate impact
    current_revenue = daily['estimated_revenue'].sum()
    daily_avg = daily.groupby('date_day')['estimated_revenue'].sum().mean()
    
    revenue_increase = (price_increase / 100) * (affected_products / 100) * current_revenue
    demand_loss = (demand_impact / 100) * (affected_products / 100) * current_revenue
    net_impact = revenue_increase - demand_loss
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Revenue from Price Increase", f"€{revenue_increase:,.0f}")
    with col2:
        st.metric("Lost Revenue from Demand Drop", f"-€{demand_loss:,.0f}")
    with col3:
        delta_color = "normal" if net_impact > 0 else "inverse"
        st.metric("Net Impact", f"€{net_impact:,.0f}", delta=f"{(net_impact/current_revenue)*100:.1f}%", delta_color=delta_color)
