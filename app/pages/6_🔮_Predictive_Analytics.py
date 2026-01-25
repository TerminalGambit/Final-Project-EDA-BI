"""
Predictive Analytics Module
===========================
Demand forecasting, seasonal patterns, and scenario planning.
Answers "What will happen?" not just "What happened?"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.loader import load_all_data, compute_forecast_accuracy

st.set_page_config(page_title="Predictive Analytics", page_icon="🔮", layout="wide")

st.title("🔮 Predictive Analytics")
st.markdown("*Demand forecasting, seasonal patterns, and scenario planning*")

# Load data
with st.spinner("Loading data..."):
    df, daily, products, brands = load_all_data()

st.divider()

# =============================================================================
# SECTION 1: SEASONALITY ANALYSIS
# =============================================================================
st.header("📅 Seasonality & Pattern Analysis")
st.markdown("*Understand cyclical patterns to anticipate demand*")

# Prepare time features
daily_agg = daily.groupby('date_day').agg({
    'estimated_revenue': 'sum',
    'estimated_sales': 'sum'
}).reset_index()
daily_agg['date_day'] = pd.to_datetime(daily_agg['date_day'])
daily_agg['day_of_week'] = daily_agg['date_day'].dt.dayofweek
daily_agg['day_name'] = daily_agg['date_day'].dt.day_name()
daily_agg['month'] = daily_agg['date_day'].dt.month
daily_agg['month_name'] = daily_agg['date_day'].dt.month_name()
daily_agg['week_of_year'] = daily_agg['date_day'].dt.isocalendar().week

col1, col2 = st.columns(2)

with col1:
    st.subheader("Weekly Pattern")
    
    # Day of week pattern
    dow_pattern = daily_agg.groupby(['day_of_week', 'day_name'])['estimated_revenue'].mean().reset_index()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_pattern['day_name'] = pd.Categorical(dow_pattern['day_name'], categories=dow_order, ordered=True)
    dow_pattern = dow_pattern.sort_values('day_name')
    
    avg_revenue = dow_pattern['estimated_revenue'].mean()
    dow_pattern['vs_avg'] = ((dow_pattern['estimated_revenue'] - avg_revenue) / avg_revenue) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dow_pattern['day_name'],
        y=dow_pattern['estimated_revenue'],
        marker_color=dow_pattern['vs_avg'].apply(lambda x: 'green' if x > 0 else 'red'),
        text=dow_pattern['vs_avg'].apply(lambda x: f"{x:+.1f}%"),
        textposition='outside'
    ))
    fig.add_hline(y=avg_revenue, line_dash="dash", annotation_text="Average")
    fig.update_layout(title="Average Daily Revenue by Day of Week", yaxis_title="Revenue (€)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insight
    best_day = dow_pattern.loc[dow_pattern['estimated_revenue'].idxmax()]
    worst_day = dow_pattern.loc[dow_pattern['estimated_revenue'].idxmin()]
    
    st.info(f"""
    **Weekly Pattern Insight:**
    - Best day: **{best_day['day_name']}** ({best_day['vs_avg']:+.1f}% vs avg)
    - Worst day: **{worst_day['day_name']}** ({worst_day['vs_avg']:+.1f}% vs avg)
    - Weekly variance: **€{(best_day['estimated_revenue'] - worst_day['estimated_revenue']):,.0f}** daily revenue swing
    """)

with col2:
    st.subheader("Monthly Pattern")
    
    # Monthly pattern
    monthly_pattern = daily_agg.groupby(['month', 'month_name'])['estimated_revenue'].mean().reset_index()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_pattern['month_name'] = pd.Categorical(monthly_pattern['month_name'], categories=month_order, ordered=True)
    monthly_pattern = monthly_pattern.sort_values('month')
    
    avg_monthly = monthly_pattern['estimated_revenue'].mean()
    monthly_pattern['vs_avg'] = ((monthly_pattern['estimated_revenue'] - avg_monthly) / avg_monthly) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_pattern['month_name'],
        y=monthly_pattern['estimated_revenue'],
        marker_color=monthly_pattern['vs_avg'].apply(lambda x: 'green' if x > 0 else 'red'),
        text=monthly_pattern['vs_avg'].apply(lambda x: f"{x:+.1f}%"),
        textposition='outside'
    ))
    fig.add_hline(y=avg_monthly, line_dash="dash", annotation_text="Average")
    fig.update_layout(title="Average Daily Revenue by Month", yaxis_title="Revenue (€)", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal insights
    peak_months = monthly_pattern.nlargest(3, 'estimated_revenue')['month_name'].tolist()
    low_months = monthly_pattern.nsmallest(3, 'estimated_revenue')['month_name'].tolist()
    
    st.info(f"""
    **Seasonal Pattern Insight:**
    - Peak months: **{', '.join(peak_months)}**
    - Low months: **{', '.join(low_months)}**
    - Plan inventory buildup 4-6 weeks before peak periods
    """)

st.divider()

# =============================================================================
# SECTION 2: DEMAND FORECASTING
# =============================================================================
st.header("📈 Demand Forecasting")
st.markdown("*Simple trend-based projections for the next 30/60/90 days*")

# Simple linear trend forecast
daily_agg_sorted = daily_agg.sort_values('date_day')
daily_agg_sorted['day_num'] = range(len(daily_agg_sorted))

# Calculate trend using last 90 days
last_90 = daily_agg_sorted.tail(90)
x = last_90['day_num'].values
y = last_90['estimated_revenue'].values

# Linear regression
slope, intercept = np.polyfit(x, y, 1)
trend_direction = "📈 Growing" if slope > 0 else "📉 Declining"
daily_growth_rate = (slope / y.mean()) * 100

# Generate forecast
last_day_num = daily_agg_sorted['day_num'].max()
forecast_days = 90
forecast_x = np.arange(last_day_num + 1, last_day_num + forecast_days + 1)
forecast_y = slope * forecast_x + intercept

# Add seasonality adjustment (simple weekly pattern)
forecast_dates = pd.date_range(start=daily_agg_sorted['date_day'].max() + pd.Timedelta(days=1), periods=forecast_days)
forecast_df = pd.DataFrame({
    'date': forecast_dates,
    'forecast': forecast_y,
    'day_of_week': forecast_dates.dayofweek
})

# Apply weekly seasonality
dow_factors = dow_pattern.set_index('day_of_week')['estimated_revenue'] / dow_pattern['estimated_revenue'].mean()
forecast_df['seasonal_factor'] = forecast_df['day_of_week'].map(dow_factors)
forecast_df['adjusted_forecast'] = forecast_df['forecast'] * forecast_df['seasonal_factor']

# Calculate confidence intervals (±15%)
forecast_df['lower'] = forecast_df['adjusted_forecast'] * 0.85
forecast_df['upper'] = forecast_df['adjusted_forecast'] * 1.15

# Compute forecast accuracy metrics
forecast_accuracy = compute_forecast_accuracy()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Trend Direction", trend_direction)
with col2:
    st.metric("Daily Growth Rate", f"{daily_growth_rate:+.2f}%")
with col3:
    forecast_30d = forecast_df.head(30)['adjusted_forecast'].sum()
    st.metric("30-Day Forecast", f"€{forecast_30d:,.0f}")
with col4:
    forecast_90d = forecast_df['adjusted_forecast'].sum()
    st.metric("90-Day Forecast", f"€{forecast_90d:,.0f}")

# Forecast Accuracy & Confidence Section
st.subheader("🎯 Forecast Accuracy & Confidence")

if 'error' not in forecast_accuracy:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy_color = {'Excellent': '🟢', 'Good': '🟢', 'Moderate': '🟡', 'Low': '🔴'}.get(
            forecast_accuracy.get('accuracy_rating', 'Unknown'), '⚪'
        )
        st.metric("Model Accuracy", f"{accuracy_color} {forecast_accuracy.get('accuracy_rating', 'N/A')}")
    with col2:
        st.metric("MAPE", f"{forecast_accuracy.get('mape', 0):.1f}%", 
                 help="Mean Absolute Percentage Error - lower is better")
    with col3:
        st.metric("Confidence Level", forecast_accuracy.get('confidence_level', 'N/A'))
    with col4:
        st.metric("Weekly Variance", f"±{forecast_accuracy.get('weekly_variance_pct', 0):.0f}%",
                 help="Inherent variability in daily revenue")
    
    # Confidence explanation
    st.info(f"💡 **Confidence Explanation:** {forecast_accuracy.get('confidence_explanation', '')}")
    
    # Model details expander
    with st.expander("📊 Model Details & Assumptions"):
        st.markdown(f"""
        ### Forecast Methodology
        **Model:** {forecast_accuracy.get('model_description', 'Linear trend + seasonality')}
        
        **Backtesting Results:**
        - Test period: {forecast_accuracy.get('test_period_days', 0)} days
        - MAPE: {forecast_accuracy.get('mape', 0):.1f}% (Mean Absolute Percentage Error)
        - RMSE: €{forecast_accuracy.get('rmse', 0):,.0f} (Root Mean Square Error)
        
        **Key Assumptions:**
        """)
        for assumption in forecast_accuracy.get('assumptions', []):
            st.markdown(f"- {assumption}")
        
        st.markdown("""
        ### What Could Make This Forecast Wrong?
        - **External shocks:** New competitor, economic changes, viral trends
        - **Marketing campaigns:** Unplanned promotions or influencer features
        - **Supply disruptions:** Stockouts not accounted for in forecast
        - **Seasonality shifts:** Holiday timing, weather patterns
        
        **Recommendation:** Use this forecast as a baseline, not a guarantee. 
        Monitor weekly and adjust as new data arrives.
        """)
else:
    st.warning("Insufficient data to compute forecast accuracy metrics.")

# Forecast chart
fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(
    x=daily_agg_sorted['date_day'],
    y=daily_agg_sorted['estimated_revenue'],
    mode='lines',
    name='Historical',
    line=dict(color='blue')
))

# Trend line for historical
trend_line = slope * daily_agg_sorted['day_num'] + intercept
fig.add_trace(go.Scatter(
    x=daily_agg_sorted['date_day'],
    y=trend_line,
    mode='lines',
    name='Trend',
    line=dict(color='gray', dash='dash')
))

# Forecast
fig.add_trace(go.Scatter(
    x=forecast_df['date'],
    y=forecast_df['adjusted_forecast'],
    mode='lines',
    name='Forecast',
    line=dict(color='green')
))

# Confidence interval
fig.add_trace(go.Scatter(
    x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
    y=pd.concat([forecast_df['upper'], forecast_df['lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(0,255,0,0.1)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Confidence Interval'
))

fig.update_layout(
    title="Revenue Trend & 90-Day Forecast",
    xaxis_title="Date",
    yaxis_title="Daily Revenue (€)",
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# Forecast by period
st.subheader("📊 Forecast Summary by Period")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Next 30 Days")
    f30 = forecast_df.head(30)
    st.metric("Projected Revenue", f"€{f30['adjusted_forecast'].sum():,.0f}")
    st.metric("Projected Units", f"{f30['adjusted_forecast'].sum() / products['avg_price'].mean():,.0f}")
    st.metric("Confidence Range", f"€{f30['lower'].sum():,.0f} - €{f30['upper'].sum():,.0f}")

with col2:
    st.markdown("### Next 60 Days")
    f60 = forecast_df.head(60)
    st.metric("Projected Revenue", f"€{f60['adjusted_forecast'].sum():,.0f}")
    st.metric("Projected Units", f"{f60['adjusted_forecast'].sum() / products['avg_price'].mean():,.0f}")
    st.metric("Confidence Range", f"€{f60['lower'].sum():,.0f} - €{f60['upper'].sum():,.0f}")

with col3:
    st.markdown("### Next 90 Days")
    f90 = forecast_df
    st.metric("Projected Revenue", f"€{f90['adjusted_forecast'].sum():,.0f}")
    st.metric("Projected Units", f"{f90['adjusted_forecast'].sum() / products['avg_price'].mean():,.0f}")
    st.metric("Confidence Range", f"€{f90['lower'].sum():,.0f} - €{f90['upper'].sum():,.0f}")

st.divider()

# =============================================================================
# SECTION 3: SCENARIO PLANNING
# =============================================================================
st.header("🎲 Scenario Planning")
st.markdown("*Model different business scenarios and their financial impact*")

tab1, tab2, tab3 = st.tabs(["💰 Pricing Scenarios", "📦 Inventory Scenarios", "📈 Growth Scenarios"])

with tab1:
    st.subheader("Pricing Impact Simulator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        price_change = st.slider("Price Change (%)", -20, 20, 5, key="price_scenario")
        elasticity = st.slider("Demand Elasticity", 0.5, 2.0, 1.2, 0.1, 
                               help="How much demand changes for each 1% price change. 1.2 means 1.2% demand drop for 1% price increase")
        affected_pct = st.slider("Products Affected (%)", 10, 100, 50)
    
    with col2:
        # Calculate impact
        current_revenue = daily['estimated_revenue'].sum()
        current_units = daily['estimated_sales'].sum()
        
        affected_revenue = current_revenue * (affected_pct / 100)
        unaffected_revenue = current_revenue - affected_revenue
        
        # Price effect on affected products
        demand_change = -price_change * elasticity  # % change in demand
        new_price_factor = 1 + (price_change / 100)
        new_demand_factor = 1 + (demand_change / 100)
        
        new_affected_revenue = affected_revenue * new_price_factor * new_demand_factor
        new_total_revenue = unaffected_revenue + new_affected_revenue
        
        revenue_impact = new_total_revenue - current_revenue
        margin_impact = affected_revenue * (price_change / 100) * new_demand_factor  # Simplified margin
        
        # Results
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Revenue Impact", f"€{revenue_impact:+,.0f}", 
                     delta=f"{(revenue_impact/current_revenue)*100:+.1f}%")
        with col_b:
            st.metric("Margin Impact", f"€{margin_impact:+,.0f}")
        with col_c:
            st.metric("Unit Volume Change", f"{demand_change:+.1f}%")
        
        # Recommendation
        if revenue_impact > 0 and margin_impact > 0:
            st.success(f"""
            **Recommended:** This pricing strategy increases both revenue and margin.
            - Annual revenue gain: €{revenue_impact:,.0f}
            - Consider implementing in phases
            """)
        elif margin_impact > 0:
            st.info(f"""
            **Margin Positive:** Revenue decreases but margins improve.
            - Best for premium positioning
            - Monitor customer retention
            """)
        else:
            st.warning(f"""
            **Caution:** This scenario shows negative outcomes.
            - Consider smaller price changes
            - Test on limited product set first
            """)

with tab2:
    st.subheader("Inventory Investment Simulator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        safety_days = st.slider("Safety Stock (Days)", 0, 30, 7, key="inv_scenario")
        holding_cost_pct = st.slider("Holding Cost (%/year)", 10, 40, 20, key="inv_holding")
        stockout_capture = st.slider("Stockout Capture Rate (%)", 50, 100, 75,
                                     help="What % of lost sales could be captured with better inventory")
    
    with col2:
        # Calculate current stockout cost
        avg_velocity = products['daily_velocity'].mean()
        avg_price = products['avg_price'].mean()
        avg_stockout_rate = products['stockout_rate'].mean()
        
        # Annual lost revenue from stockouts
        annual_lost_revenue = avg_velocity * avg_price * avg_stockout_rate * 365 * len(products)
        
        # Investment calculation
        safety_stock_units = avg_velocity * safety_days * len(products)
        inventory_investment = safety_stock_units * avg_price
        annual_holding_cost = inventory_investment * (holding_cost_pct / 100)
        
        # Revenue capture
        captured_revenue = annual_lost_revenue * (stockout_capture / 100)
        net_benefit = captured_revenue - annual_holding_cost
        roi = (net_benefit / inventory_investment) * 100 if inventory_investment > 0 else 0
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Investment Required", f"€{inventory_investment:,.0f}")
        with col_b:
            st.metric("Annual Net Benefit", f"€{net_benefit:,.0f}")
        with col_c:
            st.metric("ROI", f"{roi:.0f}%")
        
        # Scenario table
        scenarios = []
        for days in [3, 7, 14, 21]:
            inv = avg_velocity * days * len(products) * avg_price
            hold = inv * (holding_cost_pct / 100)
            cap = annual_lost_revenue * (stockout_capture / 100) * (days / 30)  # Scaled capture
            net = cap - hold
            scenarios.append({
                'Days': days,
                'Investment': f"€{inv:,.0f}",
                'Holding Cost': f"€{hold:,.0f}",
                'Revenue Captured': f"€{cap:,.0f}",
                'Net Benefit': f"€{net:,.0f}",
                'ROI': f"{(net/inv)*100:.0f}%"
            })
        
        st.markdown("**Scenario Comparison:**")
        st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Growth Scenario Planner")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        growth_rate = st.slider("Annual Growth Target (%)", 0, 50, 15, key="growth_rate")
        new_brands = st.slider("New Brands Added", 0, 10, 2)
        marketing_increase = st.slider("Marketing Budget Increase (%)", 0, 100, 25)
    
    with col2:
        current_annual = daily['estimated_revenue'].sum()
        target_annual = current_annual * (1 + growth_rate / 100)
        growth_needed = target_annual - current_annual
        
        # Estimate sources of growth
        existing_growth = current_annual * 0.05  # Assume 5% organic
        new_brand_contribution = new_brands * (brands['total_revenue'].mean() * 0.5)  # New brands at 50% of avg
        marketing_lift = current_annual * (marketing_increase / 100) * 0.1  # 10% marketing efficiency
        
        total_projected = current_annual + existing_growth + new_brand_contribution + marketing_lift
        gap = target_annual - total_projected
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Current Annual Revenue", f"€{current_annual:,.0f}")
        with col_b:
            st.metric("Target Revenue", f"€{target_annual:,.0f}")
        with col3:
            on_track = "✅ On Track" if total_projected >= target_annual else "⚠️ Gap Exists"
            st.metric("Status", on_track)
        
        # Growth waterfall
        st.markdown("**Growth Sources:**")
        growth_sources = pd.DataFrame({
            'Source': ['Current Revenue', 'Organic Growth', f'{new_brands} New Brands', 'Marketing Lift', 'Gap to Target'],
            'Amount': [current_annual, existing_growth, new_brand_contribution, marketing_lift, max(0, gap)],
            'Cumulative': [current_annual, 
                          current_annual + existing_growth,
                          current_annual + existing_growth + new_brand_contribution,
                          total_projected,
                          target_annual]
        })
        
        fig = go.Figure(go.Waterfall(
            name="Growth",
            orientation="v",
            x=growth_sources['Source'],
            y=[current_annual, existing_growth, new_brand_contribution, marketing_lift, max(0, gap)],
            connector={"line": {"color": "gray"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}}
        ))
        fig.update_layout(title="Revenue Growth Waterfall", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        if gap > 0:
            st.warning(f"""
            **Gap Analysis:** €{gap:,.0f} additional revenue needed to hit target.
            
            Options to close gap:
            - Add {int(gap / (brands['total_revenue'].mean() * 0.5))} more brands
            - Increase marketing budget by additional {int((gap / current_annual) * 1000)}%
            - Implement pricing optimization (see Pricing Scenarios)
            """)
        else:
            st.success(f"""
            **On Track:** Projected revenue exceeds target by €{-gap:,.0f}.
            
            Consider:
            - Accelerating timeline
            - Investing excess in market share
            - Building reserves for volatility
            """)

st.divider()

# =============================================================================
# SECTION 4: STOCKOUT PREDICTIONS
# =============================================================================
st.header("⚠️ Stockout Risk Predictions")
st.markdown("*Products likely to stockout in the next 30 days based on velocity and current stock*")

# Get latest stock and velocity
latest_stock = df.sort_values('date').groupby('product_id').last()[['stock', 'title', 'brand', 'offer_price']].reset_index()
velocity = products[['product_id', 'daily_velocity']].copy()

stockout_risk = latest_stock.merge(velocity, on='product_id')
stockout_risk['days_until_stockout'] = np.where(
    stockout_risk['daily_velocity'] > 0,
    stockout_risk['stock'] / stockout_risk['daily_velocity'],
    np.inf
)
stockout_risk['risk_level'] = pd.cut(
    stockout_risk['days_until_stockout'],
    bins=[-np.inf, 7, 14, 30, np.inf],
    labels=['🔴 Critical (<7d)', '🟠 High (7-14d)', '🟡 Medium (14-30d)', '🟢 Low (>30d)']
)

# Summary
risk_summary = stockout_risk['risk_level'].value_counts()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🔴 Critical", risk_summary.get('🔴 Critical (<7d)', 0))
with col2:
    st.metric("🟠 High Risk", risk_summary.get('🟠 High (7-14d)', 0))
with col3:
    st.metric("🟡 Medium Risk", risk_summary.get('🟡 Medium (14-30d)', 0))
with col4:
    st.metric("🟢 Low Risk", risk_summary.get('🟢 Low (>30d)', 0))

# Critical products table
critical = stockout_risk[stockout_risk['risk_level'].isin(['🔴 Critical (<7d)', '🟠 High (7-14d)'])].sort_values('days_until_stockout')

if not critical.empty:
    st.markdown("### Products Requiring Immediate Reorder")
    critical_display = critical[['title', 'brand', 'stock', 'daily_velocity', 'days_until_stockout', 'risk_level']].head(15).copy()
    critical_display['daily_velocity'] = critical_display['daily_velocity'].apply(lambda x: f"{x:.2f}")
    critical_display['days_until_stockout'] = critical_display['days_until_stockout'].apply(lambda x: f"{x:.0f} days")
    critical_display.columns = ['Product', 'Brand', 'Stock', 'Daily Velocity', 'Days Left', 'Risk']
    st.dataframe(critical_display, use_container_width=True, hide_index=True)
    
    # Revenue at risk
    critical['revenue_at_risk'] = critical['daily_velocity'] * critical['offer_price'] * 30  # 30-day risk
    total_risk = critical['revenue_at_risk'].sum()
    
    st.error(f"""
    **Revenue at Risk:** €{total_risk:,.0f} over the next 30 days if these products stockout.
    
    **Action Required:** Place orders for {len(critical)} products immediately.
    """)
else:
    st.success("✅ No critical stockout risks detected in the next 14 days.")

st.divider()

# =============================================================================
# SECTION 5: INVENTORY PLAN FOR FORECAST
# =============================================================================
st.header("📦 Inventory Plan for Forecast")
st.markdown("*Stock requirements to support forecasted demand - bridges prediction to action*")

# Calculate inventory requirements based on forecast
col1, col2, col3 = st.columns(3)
with col1:
    planning_horizon = st.selectbox("Planning Horizon", [30, 60, 90], index=0, format_func=lambda x: f"{x} days")
with col2:
    safety_buffer_pct = st.slider("Safety Buffer (%)", 10, 50, 25, help="Extra stock above forecast")
with col3:
    reorder_lead_time = st.slider("Reorder Lead Time (days)", 7, 30, 14)

# Get forecast for selected horizon
forecast_period = forecast_df.head(planning_horizon)
forecast_revenue = forecast_period['adjusted_forecast'].sum()
forecast_units = forecast_revenue / products['avg_price'].mean()

# Calculate per-product requirements
product_forecast = products.copy()
product_forecast['forecasted_demand'] = product_forecast['daily_velocity'] * planning_horizon
product_forecast['safety_stock_needed'] = product_forecast['forecasted_demand'] * (safety_buffer_pct / 100)
product_forecast['total_stock_needed'] = product_forecast['forecasted_demand'] + product_forecast['safety_stock_needed']

# Merge with current stock
current_stock = df.sort_values('date').groupby('product_id').last()[['stock']].reset_index()
product_forecast = product_forecast.merge(current_stock, on='product_id', how='left')
product_forecast['stock'] = product_forecast['stock'].fillna(0)

# Calculate gap
product_forecast['stock_gap'] = product_forecast['total_stock_needed'] - product_forecast['stock']
product_forecast['needs_reorder'] = product_forecast['stock_gap'] > 0

# Summary metrics
products_needing_reorder = product_forecast[product_forecast['needs_reorder']]
total_units_gap = products_needing_reorder['stock_gap'].sum()
total_investment = (products_needing_reorder['stock_gap'] * products_needing_reorder['avg_price'] * 0.5).sum()  # 50% margin

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(f"Forecast Revenue ({planning_horizon}d)", f"€{forecast_revenue:,.0f}")
with col2:
    st.metric("Products Need Reorder", len(products_needing_reorder))
with col3:
    st.metric("Units Gap", f"{total_units_gap:,.0f}")
with col4:
    st.metric("Investment Required", f"€{total_investment:,.0f}")

# Detailed breakdown
st.subheader("📋 Inventory Requirements by Priority")

priority_df = products_needing_reorder.sort_values('stock_gap', ascending=False).head(20)
if not priority_df.empty:
    display_df = priority_df[['title', 'brand', 'stock', 'forecasted_demand', 'safety_stock_needed', 'stock_gap', 'avg_price']].copy()
    display_df['stock'] = display_df['stock'].astype(int)
    display_df['forecasted_demand'] = display_df['forecasted_demand'].apply(lambda x: f"{x:.0f}")
    display_df['safety_stock_needed'] = display_df['safety_stock_needed'].apply(lambda x: f"{x:.0f}")
    display_df['stock_gap'] = display_df['stock_gap'].apply(lambda x: f"{x:.0f}")
    display_df['investment'] = priority_df['stock_gap'] * priority_df['avg_price'] * 0.5
    display_df['investment'] = display_df['investment'].apply(lambda x: f"€{x:,.0f}")
    display_df['avg_price'] = display_df['avg_price'].apply(lambda x: f"€{x:.2f}")
    
    display_df.columns = ['Product', 'Brand', 'Current Stock', f'{planning_horizon}d Demand', 'Safety Stock', 'Gap', 'Price', 'Investment']
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.success(f"✅ Current inventory is sufficient to support {planning_horizon}-day forecast!")

# Timeline guidance
st.subheader("📅 Reorder Timeline")

order_by_date = pd.Timestamp.now() + pd.Timedelta(days=max(0, 7 - reorder_lead_time))
arrival_date = order_by_date + pd.Timedelta(days=reorder_lead_time)

st.info(f"""
**To support the {planning_horizon}-day forecast:**

1. 📝 **Order by:** {order_by_date.strftime('%Y-%m-%d')} (to allow for lead time)
2. 📦 **Expected arrival:** {arrival_date.strftime('%Y-%m-%d')}
3. 💰 **Budget required:** €{total_investment:,.0f}

👉 **For detailed reorder recommendations with prioritization, visit 🎬 Action Center**
""")
