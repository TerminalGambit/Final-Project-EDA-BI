"""
Inventory Risk Management Module
================================
Stockout analysis, days of inventory, restock patterns, and ROI calculations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from data.loader import load_all_data, compute_inventory_metrics, compute_reorder_recommendations, compute_ltv_impact

st.set_page_config(page_title="Inventory Risk", page_icon="📦", layout="wide")

st.title("📦 Inventory Risk Management")
st.markdown("*Stockout analysis, supply chain intelligence, and inventory optimization*")

# Load data
with st.spinner("Loading data..."):
    df, daily, products, brands = load_all_data()
    inventory = compute_inventory_metrics(df, daily)

st.divider()

# Key Metrics
st.subheader("🎯 Inventory Health Overview")

col1, col2, col3, col4 = st.columns(4)

# Calculate metrics
current_stockouts = inventory[inventory['risk_level'] == 'STOCKOUT'].shape[0]
critical_products = inventory[inventory['risk_level'] == 'CRITICAL'].shape[0]
avg_stockout_rate = products['stockout_rate'].mean() * 100
total_stockout_days = products['stockout_days'].sum()

with col1:
    st.metric("Currently Out of Stock", current_stockouts, help="Products with no stock right now")
with col2:
    st.metric("Critical Stock (<7 days)", critical_products, help="Products that will stockout within 7 days")
with col3:
    st.metric("Avg Stockout Rate", f"{avg_stockout_rate:.1f}%", help="Average % of days products were out of stock")
with col4:
    st.metric("Total Stockout Days", f"{total_stockout_days:,.0f}", help="Sum of stockout days across all products")

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🚨 Risk Dashboard", "📊 Stockout Analysis", "🔄 Restock Patterns", "💰 ROI Calculator"])

with tab1:
    st.subheader("Inventory Risk Dashboard")
    
    # Risk level distribution
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = inventory['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        
        # Custom color mapping
        color_map = {
            'HEALTHY': '#28a745',
            'MEDIUM': '#ffc107', 
            'LOW': '#fd7e14',
            'CRITICAL': '#dc3545',
            'STOCKOUT': '#6c757d'
        }
        
        fig = px.pie(
            risk_counts,
            values='Count',
            names='Risk Level',
            title="Products by Risk Level",
            color='Risk Level',
            color_discrete_map=color_map
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk by brand
        brand_risk = inventory.groupby(['brand', 'risk_level']).size().reset_index(name='count')
        
        fig = px.bar(
            brand_risk,
            x='brand',
            y='count',
            color='risk_level',
            title="Risk Distribution by Brand",
            color_discrete_map=color_map,
            barmode='stack'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Critical items table
    st.markdown("### 🚨 Products Requiring Immediate Attention")
    
    critical_items = inventory[inventory['risk_level'].isin(['STOCKOUT', 'CRITICAL', 'LOW'])].copy()
    critical_items = critical_items.sort_values('days_of_inventory')
    
    if not critical_items.empty:
        critical_display = critical_items[['title', 'brand', 'stock', 'avg_daily_sales', 'days_of_inventory', 'risk_level']].head(20).copy()
        critical_display['avg_daily_sales'] = critical_display['avg_daily_sales'].apply(lambda x: f"{x:.2f}")
        critical_display['days_of_inventory'] = critical_display['days_of_inventory'].apply(
            lambda x: f"{x:.0f} days" if x < 1000 else "∞"
        )
        critical_display.columns = ['Product', 'Brand', 'Current Stock', 'Daily Sales', 'Days Left', 'Risk']
        
        st.dataframe(
            critical_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk": st.column_config.TextColumn(
                    "Risk",
                    help="Inventory risk level"
                )
            }
        )
    else:
        st.success("✅ No critical inventory issues detected!")
    
    # Quick Reorder Report Section
    st.divider()
    st.subheader("📥 Quick Reorder Report")
    st.markdown("*Generate a procurement-ready report with customizable parameters*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        lead_time_input = st.slider("Supplier Lead Time (days)", 7, 30, 14, key="inv_lead_time")
    with col2:
        safety_stock_input = st.slider("Safety Stock Buffer (days)", 3, 14, 7, key="inv_safety")
    with col3:
        holding_cost_input = st.slider("Annual Holding Cost (%)", 15, 35, 20, key="inv_holding")
    
    # Compute reorder recommendations with user parameters
    reorder_recs = compute_reorder_recommendations(lead_time_days=lead_time_input, safety_stock_days=safety_stock_input)
    
    if not reorder_recs.empty:
        # Working capital impact
        total_investment = reorder_recs['investment_required'].sum()
        total_risk = reorder_recs['revenue_at_risk'].sum()
        annual_holding = total_investment * (holding_cost_input / 100)
        expected_benefit = total_risk * 0.7  # 70% capture rate
        net_benefit = expected_benefit - annual_holding
        
        st.markdown("#### 💰 Working Capital Impact")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Products to Reorder", len(reorder_recs))
        with col2:
            st.metric("Investment Required", f"€{total_investment:,.0f}")
        with col3:
            st.metric("Annual Holding Cost", f"€{annual_holding:,.0f}")
        with col4:
            st.metric("Net Annual Benefit", f"€{net_benefit:,.0f}")
        
        # Download button
        csv_buffer = BytesIO()
        reorder_recs.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                label="📥 Download Reorder Report",
                data=csv_buffer,
                file_name="reorder_report.csv",
                mime="text/csv"
            )
        with col2:
            st.info("👉 For detailed implementation guidance, visit **🎬 Action Center**")
    else:
        st.success("✅ No products require immediate reorder at current parameters!")

with tab2:
    st.subheader("Stockout Pattern Analysis")
    
    # Stockout timeline
    stockout_daily = daily.groupby('date_day').agg({
        'is_stockout': 'sum',
        'product_id': 'nunique'
    }).reset_index()
    stockout_daily['stockout_rate'] = stockout_daily['is_stockout'] / stockout_daily['product_id'] * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stockout_daily['date_day'],
        y=stockout_daily['is_stockout'],
        name='Products Out of Stock',
        fill='tozeroy',
        line=dict(color='red')
    ))
    fig.update_layout(
        title="Daily Stockout Count Over Time",
        xaxis_title="Date",
        yaxis_title="Products Out of Stock",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Worst stockout offenders
        st.markdown("#### 📉 Products with Highest Stockout Rates")
        worst_stockout = products.sort_values('stockout_rate', ascending=False).head(15)
        worst_stockout = worst_stockout[worst_stockout['stockout_rate'] > 0]
        
        if not worst_stockout.empty:
            worst_display = worst_stockout[['title', 'brand', 'stockout_rate', 'stockout_days', 'daily_velocity']].copy()
            worst_display['stockout_rate'] = worst_display['stockout_rate'].apply(lambda x: f"{x*100:.1f}%")
            worst_display['daily_velocity'] = worst_display['daily_velocity'].apply(lambda x: f"{x:.2f}")
            worst_display.columns = ['Product', 'Brand', 'Stockout Rate', 'Days OOS', 'Velocity']
            st.dataframe(worst_display, use_container_width=True, hide_index=True)
        else:
            st.info("No products with stockouts")
    
    with col2:
        # Stockout by brand
        st.markdown("#### 🏢 Stockout Rate by Brand")
        brand_stockout = daily.groupby('brand').agg({
            'is_stockout': ['sum', 'count']
        }).reset_index()
        brand_stockout.columns = ['brand', 'stockout_count', 'total_records']
        brand_stockout['stockout_rate'] = brand_stockout['stockout_count'] / brand_stockout['total_records'] * 100
        brand_stockout = brand_stockout.sort_values('stockout_rate', ascending=True)
        
        fig = px.bar(
            brand_stockout,
            x='stockout_rate',
            y='brand',
            orientation='h',
            title="Stockout Rate by Brand (%)",
            color='stockout_rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Lost revenue estimation
    st.markdown("### 💸 Estimated Revenue Lost to Stockouts")
    
    # Calculate lost revenue
    products_with_stockout = products[products['stockout_rate'] > 0].copy()
    products_with_stockout['estimated_lost_sales'] = (
        products_with_stockout['daily_velocity'] * products_with_stockout['stockout_days']
    )
    products_with_stockout['estimated_lost_revenue'] = (
        products_with_stockout['estimated_lost_sales'] * products_with_stockout['avg_price']
    )
    
    total_lost_revenue = products_with_stockout['estimated_lost_revenue'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Est. Lost Units", f"{products_with_stockout['estimated_lost_sales'].sum():,.0f}")
    with col2:
        st.metric("Est. Lost Revenue", f"€{total_lost_revenue:,.0f}")
    with col3:
        total_revenue = daily['estimated_revenue'].sum()
        lost_pct = (total_lost_revenue / (total_revenue + total_lost_revenue)) * 100
        st.metric("% of Potential Revenue Lost", f"{lost_pct:.1f}%")
    
    # LTV Impact Section - Compound Loss Analysis
    st.divider()
    st.markdown("### 📈 Compound Impact: Customer Lifetime Value Loss")
    st.markdown("*Stockouts don't just lose immediate sales—they cause customer churn, multiplying the true cost*")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        churn_rate_input = st.slider("Est. Churn Rate (%)", 1, 15, 5, 
                                     help="% of customers who don't return after stockout experience")
        avg_ltv_input = st.slider("Avg Customer LTV (€)", 50, 500, 150,
                                  help="Average lifetime value per customer")
    
    # Compute LTV impact with user parameters
    ltv_data = compute_ltv_impact(churn_rate=churn_rate_input/100, avg_ltv=avg_ltv_input)
    
    with col2:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Immediate Lost Revenue", f"€{ltv_data.get('immediate_lost_revenue', 0):,.0f}")
        with col_b:
            st.metric("LTV Loss (Churn)", f"€{ltv_data.get('ltv_loss', 0):,.0f}")
        with col_c:
            st.metric("Total Compound Loss", f"€{ltv_data.get('total_compound_loss', 0):,.0f}")
    
    # Impact multiplier insight
    multiplier = ltv_data.get('multiplier', 1)
    st.error(f"""
    **💡 Key Insight:** {ltv_data.get('explanation', '')}
    
    Your €{ltv_data.get('immediate_lost_revenue', 0):,.0f} in lost revenue actually represents **€{ltv_data.get('total_compound_loss', 0):,.0f}** 
    when accounting for customer churn (**{multiplier}x multiplier**).
    """)
    
    # Show formula
    with st.expander("📊 Methodology & Assumptions"):
        st.markdown(f"""
        ### LTV Impact Calculation
        
        **Formula:**
        ```
        Compound Loss = Immediate Lost Revenue + (Affected Customers × Churn Rate × Avg LTV)
        ```
        
        **Current Parameters:**
        - Churn Rate: {churn_rate_input}% of stockout-affected customers don't return
        - Average LTV: €{avg_ltv_input} per customer
        - Multiplier: {multiplier}x
        
        **Interpretation:**
        - For every €1 lost to stockouts, you're actually losing €{multiplier:.2f}
        - This makes inventory investment ROI {multiplier:.1f}x higher than simple calculations suggest
        
        **Adjust the sliders** to see how different churn/LTV assumptions affect the total impact.
        """)

with tab3:
    st.subheader("Restock Pattern Analysis")
    
    st.markdown("""
    **Restock Detection:** We identify restocks as large positive stock increases (>10 units).
    This reveals the competitor's supply chain patterns and ordering cycles.
    """)
    
    # Restock events over time
    restock_daily = daily.groupby('date_day')['is_restock'].sum().reset_index()
    restock_daily.columns = ['date', 'restock_count']
    
    fig = px.bar(
        restock_daily,
        x='date',
        y='restock_count',
        title="Restock Events Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Restock frequency by day of week
        daily_copy = daily.copy()
        daily_copy['day_of_week'] = daily_copy['date_day'].dt.day_name()
        dow_restocks = daily_copy.groupby('day_of_week')['is_restock'].sum().reset_index()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_restocks['day_of_week'] = pd.Categorical(dow_restocks['day_of_week'], categories=dow_order, ordered=True)
        dow_restocks = dow_restocks.sort_values('day_of_week')
        
        fig = px.bar(
            dow_restocks,
            x='day_of_week',
            y='is_restock',
            title="Restocks by Day of Week",
            color='is_restock',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Products with most restocks
        st.markdown("#### 🔄 Most Frequently Restocked Products")
        most_restocked = products.sort_values('restock_count', ascending=False).head(15)
        most_restocked = most_restocked[most_restocked['restock_count'] > 0]
        
        if not most_restocked.empty:
            restock_display = most_restocked[['title', 'brand', 'restock_count', 'daily_velocity']].copy()
            restock_display['daily_velocity'] = restock_display['daily_velocity'].apply(lambda x: f"{x:.2f}")
            restock_display.columns = ['Product', 'Brand', 'Restocks', 'Velocity']
            st.dataframe(restock_display, use_container_width=True, hide_index=True)
            
            st.info("💡 High-frequency restocks indicate strong demand and tight inventory management")
        else:
            st.info("No significant restock events detected")

with tab4:
    st.subheader("💰 Safety Stock ROI Calculator")
    
    st.markdown("""
    **Calculate the ROI of holding additional safety stock** to prevent stockouts.
    Compare the cost of holding inventory vs. the revenue lost from stockouts.
    """)
    
    # Select a product or use aggregate
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_type = st.radio(
            "Analysis Type",
            ["Single Product", "Portfolio Average"],
            horizontal=True
        )
    
    if analysis_type == "Single Product":
        # Product selector
        high_stockout_products = products[products['stockout_rate'] > 0.05].sort_values('total_revenue', ascending=False)
        product_options = high_stockout_products['title'].tolist()[:30]
        
        if product_options:
            selected_product = st.selectbox("Select Product", product_options)
            product_data = products[products['title'] == selected_product].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Daily Velocity", f"{product_data['daily_velocity']:.2f} units")
            with col2:
                st.metric("Current Stockout Rate", f"{product_data['stockout_rate']*100:.1f}%")
            with col3:
                st.metric("Avg Price", f"€{product_data['avg_price']:.2f}")
            
            avg_price = product_data['avg_price']
            daily_velocity = product_data['daily_velocity']
            stockout_rate = product_data['stockout_rate']
        else:
            st.warning("No products with significant stockout rates to analyze")
            st.stop()
    else:
        # Portfolio average
        avg_price = products['avg_price'].mean()
        daily_velocity = products['daily_velocity'].mean()
        stockout_rate = products['stockout_rate'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Daily Velocity", f"{daily_velocity:.2f} units")
        with col2:
            st.metric("Avg Stockout Rate", f"{stockout_rate*100:.1f}%")
        with col3:
            st.metric("Avg Product Price", f"€{avg_price:.2f}")
    
    st.divider()
    
    # ROI Calculator inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        safety_stock_days = st.slider("Safety Stock (days of inventory)", 3, 30, 7)
    with col2:
        holding_cost_pct = st.slider("Annual Holding Cost (%)", 10, 40, 20)
    with col3:
        stockout_capture_rate = st.slider("Stockout Capture Rate (%)", 50, 100, 80)
    
    # Calculations
    safety_stock_units = daily_velocity * safety_stock_days
    inventory_value = safety_stock_units * avg_price
    annual_holding_cost = inventory_value * (holding_cost_pct / 100)
    daily_holding_cost = annual_holding_cost / 365
    
    # Revenue impact
    daily_lost_revenue = daily_velocity * avg_price * stockout_rate
    captured_revenue = daily_lost_revenue * (stockout_capture_rate / 100)
    net_daily_benefit = captured_revenue - daily_holding_cost
    annual_benefit = net_daily_benefit * 365
    
    roi = (annual_benefit / annual_holding_cost) * 100 if annual_holding_cost > 0 else 0
    payback_days = inventory_value / net_daily_benefit if net_daily_benefit > 0 else float('inf')
    
    st.divider()
    
    # Results
    st.markdown("### 📊 ROI Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Safety Stock Investment", f"€{inventory_value:,.0f}")
    with col2:
        st.metric("Annual Holding Cost", f"€{annual_holding_cost:,.0f}")
    with col3:
        st.metric("Annual Revenue Captured", f"€{captured_revenue * 365:,.0f}")
    with col4:
        roi_color = "normal" if roi > 0 else "inverse"
        st.metric("Annual ROI", f"{roi:.0f}%", delta_color=roi_color)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Net Annual Benefit", f"€{annual_benefit:,.0f}")
    with col2:
        payback_text = f"{payback_days:.0f} days" if payback_days < 1000 else "N/A"
        st.metric("Payback Period", payback_text)
    
    # Recommendation
    if roi > 100:
        st.success(f"""
        **Strong Recommendation:** Increase safety stock to {safety_stock_days} days.
        
        - Investment: €{inventory_value:,.0f}
        - Expected annual return: €{annual_benefit:,.0f}
        - ROI: {roi:.0f}%
        
        This represents a high-confidence opportunity to capture lost revenue.
        """)
    elif roi > 0:
        st.info(f"""
        **Moderate Recommendation:** Consider safety stock of {safety_stock_days} days.
        
        - Positive ROI of {roi:.0f}%
        - Payback in {payback_days:.0f} days
        
        Evaluate based on capital availability and risk tolerance.
        """)
    else:
        st.warning(f"""
        **Not Recommended** at current parameters.
        
        - Holding costs exceed captured revenue
        - Consider reducing safety stock days or targeting higher-velocity products
        """)
