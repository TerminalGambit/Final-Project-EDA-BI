"""
CEO Action Center
=================
Executable business recommendations with prioritized actions,
investment cases, and ROI projections.

Bridges the gap from insight to execution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from data.loader import (
    load_all_data, 
    compute_reorder_recommendations,
    compute_pricing_recommendations,
    compute_portfolio_recommendations,
    compute_priority_matrix,
    compute_financial_summary
)

st.set_page_config(page_title="Action Center", page_icon="🎬", layout="wide")

st.title("🎬 CEO Action Center")
st.markdown("*Prioritized, executable recommendations with investment cases and ROI projections*")

# Load data
with st.spinner("Computing recommendations..."):
    df, daily, products, brands = load_all_data()
    financial_summary = compute_financial_summary()
    priority_matrix = compute_priority_matrix()

st.divider()

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
st.header("📊 Executive Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Revenue at Risk",
        f"€{financial_summary['total_revenue_at_risk']:,.0f}",
        help="30-day revenue at risk from potential stockouts"
    )

with col2:
    st.metric(
        "Investment Required",
        f"€{financial_summary['total_investment_required']:,.0f}",
        help="Total working capital needed to address inventory gaps"
    )

with col3:
    st.metric(
        "Revenue Opportunity",
        f"€{financial_summary['total_revenue_opportunity']:,.0f}",
        help="Annual revenue gain from pricing optimization"
    )

with col4:
    roi_display = f"{financial_summary['expected_roi_pct']:.0f}%" if financial_summary['expected_roi_pct'] < 999 else "∞"
    st.metric(
        "Expected ROI",
        roi_display,
        help="Return on investment for recommended actions"
    )

# Action counts
action_counts = financial_summary['action_counts']
st.info(f"""
**Ready to Execute:** {action_counts['inventory']} inventory actions • {action_counts['pricing']} pricing actions • {action_counts['portfolio']} portfolio actions
""")

st.divider()

# =============================================================================
# SECTION 1: PRIORITY MATRIX
# =============================================================================
st.header("🎯 Priority Matrix")
st.markdown("*Top actions ranked by impact and ROI - focus here first*")

if not priority_matrix.empty:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Scatter plot: Impact vs Investment (bubble = priority)
        fig = px.scatter(
            priority_matrix,
            x='investment_eur',
            y='impact_eur',
            color='category',
            size='priority',
            hover_data=['action', 'payback_days'],
            title="Action Priority Matrix (Impact vs Investment)",
            labels={
                'investment_eur': 'Investment Required (€)',
                'impact_eur': 'Revenue Impact (€)',
                'category': 'Category'
            },
            color_discrete_map={
                'Inventory': '#FF6B6B',
                'Pricing': '#4ECDC4',
                'Portfolio': '#45B7D1'
            }
        )
        # Add quadrant lines
        fig.add_hline(y=priority_matrix['impact_eur'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=priority_matrix['investment_eur'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Quick Wins")
        st.markdown("*High impact, low/no investment*")
        
        quick_wins = priority_matrix[
            (priority_matrix['investment_eur'] < priority_matrix['investment_eur'].median()) &
            (priority_matrix['impact_eur'] > priority_matrix['impact_eur'].median())
        ].head(5)
        
        if not quick_wins.empty:
            for _, row in quick_wins.iterrows():
                st.success(f"**{row['category']}:** {row['action'][:50]}... → €{row['impact_eur']:,.0f}")
        else:
            st.info("No quick wins identified in current data")
    
    # Top 10 Actions Table
    st.subheader("📋 Top 10 Prioritized Actions")
    
    top_actions = priority_matrix.head(10).copy()
    top_actions['investment_eur'] = top_actions['investment_eur'].apply(lambda x: f"€{x:,.0f}")
    top_actions['impact_eur'] = top_actions['impact_eur'].apply(lambda x: f"€{x:,.0f}")
    top_actions['payback_days'] = top_actions['payback_days'].apply(lambda x: f"{x} days" if x < 999 else "Immediate")
    top_actions['priority'] = top_actions['priority'].apply(lambda x: "⭐" * int(x))
    
    display_cols = ['priority', 'category', 'action', 'impact_eur', 'investment_eur', 'effort', 'payback_days']
    top_actions_display = top_actions[display_cols].copy()
    top_actions_display.columns = ['Priority', 'Category', 'Action', 'Impact', 'Investment', 'Effort', 'Payback']
    
    st.dataframe(top_actions_display, use_container_width=True, hide_index=True)
    
    # Success Metrics & Rollback Triggers
    st.subheader("🎯 Success Metrics & Rollback Triggers")
    st.markdown("*How to measure success and when to change course*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📦 Inventory Actions")
        st.markdown("""
        **Success Metric:** Stockout Rate
        - **Target:** <5% (from current ~6%)
        - **Timeline:** 30 days post-restock
        - **Measure:** Products OOS / Total Products
        
        **Rollback Trigger:**
        - ⚠️ If velocity drops >20% after restock
        - ⚠️ If holding costs exceed 25% of revenue
        - ⚠️ If inventory turns drop below 4x annually
        
        **Action if triggered:** Reduce safety stock, investigate demand shift
        """)
    
    with col2:
        st.markdown("### 💰 Pricing Actions")
        st.markdown("""
        **Success Metric:** Revenue per Product
        - **Target:** +5-10% (net of volume loss)
        - **Timeline:** 14 days per phase
        - **Measure:** Daily revenue / product
        
        **Rollback Trigger:**
        - ⚠️ If volume drops >15% (price too high)
        - ⚠️ If cart abandonment increases >2pp
        - ⚠️ If competitor undercuts significantly
        
        **Action if triggered:** Revert to original price, analyze competitor response
        """)
    
    with col3:
        st.markdown("### 🏢 Portfolio Actions")
        st.markdown("""
        **Success Metric:** Concentration Index
        - **Target:** Top brand <15% share
        - **Timeline:** 6 months
        - **Measure:** HHI index, top brand %
        
        **Rollback Trigger:**
        - ⚠️ If new brands underperform 50%+ below projections
        - ⚠️ If top brand relationship deteriorates
        - ⚠️ If diversification costs exceed budget by >30%
        
        **Action if triggered:** Pause expansion, renegotiate existing relationships
        """)
    
    # Weekly review checklist
    with st.expander("📋 Weekly Review Checklist"):
        st.markdown("""
        ### Every Week, Review:
        
        **Inventory:**
        - [ ] Current stockout count vs. target
        - [ ] Reorder status (placed? in transit? received?)
        - [ ] Any new stockout risks emerging?
        
        **Pricing:**
        - [ ] Phase progress (which products repriced?)
        - [ ] Revenue per product trend
        - [ ] Customer feedback/complaints
        
        **Portfolio:**
        - [ ] New brand pipeline status
        - [ ] Top brand relationship health
        - [ ] Concentration metrics trending
        
        **Overall:**
        - [ ] Total revenue vs. forecast
        - [ ] Any rollback triggers hit?
        - [ ] Actions for next week
        """)
else:
    st.warning("No actions computed. Check data availability.")

st.divider()

# =============================================================================
# SECTION 2: INVENTORY ACTION PLAN
# =============================================================================
st.header("📦 Inventory Action Plan")
st.markdown("*Products requiring immediate reorder with investment case*")

# User inputs for customization
col1, col2, col3 = st.columns(3)
with col1:
    lead_time = st.slider("Supplier Lead Time (days)", 7, 30, 14, help="Days until order arrives")
with col2:
    safety_days = st.slider("Safety Stock (days)", 3, 14, 7, help="Buffer inventory to maintain")
with col3:
    margin_estimate = st.slider("Estimated Margin (%)", 30, 70, 50, help="Product margin for cost calculation")

# Recompute with user parameters
inventory_actions = compute_reorder_recommendations(lead_time_days=lead_time, safety_stock_days=safety_days)

if not inventory_actions.empty:
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Products to Reorder", len(inventory_actions))
    with col2:
        total_investment = inventory_actions['investment_required'].sum()
        st.metric("Total Investment", f"€{total_investment:,.0f}")
    with col3:
        total_risk = inventory_actions['revenue_at_risk'].sum()
        st.metric("Revenue Protected", f"€{total_risk:,.0f}")
    with col4:
        roi = (total_risk * 0.7) / total_investment * 100 if total_investment > 0 else 0
        st.metric("Expected ROI", f"{roi:.0f}%")
    
    # Detailed table
    st.subheader("📋 Reorder List")
    
    inventory_display = inventory_actions.copy()
    inventory_display['stock'] = inventory_display['stock'].astype(int)
    inventory_display['avg_daily_sales'] = inventory_display['avg_daily_sales'].apply(lambda x: f"{x:.1f}")
    inventory_display['days_until_stockout'] = inventory_display['days_until_stockout'].apply(lambda x: f"{x:.0f}" if x < 1000 else "∞")
    inventory_display['recommended_qty'] = inventory_display['recommended_qty'].astype(int)
    inventory_display['investment_required'] = inventory_display['investment_required'].apply(lambda x: f"€{x:,.0f}")
    inventory_display['revenue_at_risk'] = inventory_display['revenue_at_risk'].apply(lambda x: f"€{x:,.0f}")
    inventory_display['priority_score'] = inventory_display['priority_score'].apply(lambda x: "🔴" if x >= 4 else "🟡" if x >= 3 else "🟢")
    
    display_cols = ['priority_score', 'title', 'brand', 'stock', 'avg_daily_sales', 
                   'days_until_stockout', 'recommended_qty', 'investment_required', 'revenue_at_risk']
    inventory_display = inventory_display[display_cols]
    inventory_display.columns = ['Priority', 'Product', 'Brand', 'Stock', 'Daily Sales', 
                                 'Days Left', 'Order Qty', 'Investment', 'Revenue at Risk']
    
    st.dataframe(inventory_display, use_container_width=True, hide_index=True)
    
    # Download button
    csv_buffer = BytesIO()
    inventory_actions.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    st.download_button(
        label="📥 Download Reorder Report (CSV)",
        data=csv_buffer,
        file_name="reorder_recommendations.csv",
        mime="text/csv"
    )
    
    # Implementation guidance
    with st.expander("📘 Implementation Guide"):
        st.markdown(f"""
        ### How to Execute This Plan
        
        **Phase 1: Critical Items (Priority 🔴)**
        1. Contact suppliers immediately for {len(inventory_actions[inventory_actions['priority_score'] >= 4])} critical SKUs
        2. Request expedited shipping where available
        3. Expected investment: €{inventory_actions[inventory_actions['priority_score'] >= 4]['investment_required'].sum():,.0f}
        
        **Phase 2: High Priority (Priority 🟡)**
        1. Place orders within 48 hours for remaining high-priority items
        2. Bundle orders by supplier to optimize shipping costs
        
        **Working Capital Impact:**
        - Total inventory investment: €{total_investment:,.0f}
        - Estimated holding cost (20% annually): €{total_investment * 0.2:,.0f}/year
        - Revenue protected: €{total_risk:,.0f}
        - Net benefit: €{total_risk * 0.7 - total_investment * 0.2:,.0f}/year
        
        **Success Metrics to Track:**
        - Stockout rate (target: <5%)
        - Inventory turns (maintain or improve)
        - Lost sales recovery rate
        """)
else:
    st.success("✅ No critical inventory actions needed at current parameters!")

st.divider()

# =============================================================================
# SECTION 3: PRICING ACTION PLAN
# =============================================================================
st.header("💰 Pricing Action Plan")
st.markdown("*Underpriced products with revenue optimization opportunity*")

pricing_actions = compute_pricing_recommendations()

if not pricing_actions.empty:
    # Summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Products to Reprice", len(pricing_actions))
    with col2:
        total_opportunity = pricing_actions['annual_revenue_change'].sum()
        st.metric("Annual Revenue Gain", f"€{total_opportunity:,.0f}")
    with col3:
        avg_increase = pricing_actions['price_increase_pct'].mean()
        st.metric("Avg Price Increase", f"{avg_increase:.1f}%")
    
    # Phased rollout
    st.subheader("📅 Phased Implementation")
    
    col1, col2, col3 = st.columns(3)
    
    phase1 = pricing_actions.head(10)
    phase2 = pricing_actions.iloc[10:30]
    phase3 = pricing_actions.iloc[30:]
    
    with col1:
        st.markdown("### Phase 1 (Week 1-2)")
        st.markdown(f"**{len(phase1)} products**")
        st.metric("Revenue Impact", f"€{phase1['annual_revenue_change'].sum():,.0f}/year")
        if not phase1.empty:
            for _, row in phase1.head(5).iterrows():
                st.caption(f"• {row['title'][:30]}... €{row['avg_price']:.0f}→€{row['recommended_price']:.0f}")
    
    with col2:
        st.markdown("### Phase 2 (Week 3-4)")
        st.markdown(f"**{len(phase2)} products**")
        st.metric("Revenue Impact", f"€{phase2['annual_revenue_change'].sum():,.0f}/year")
    
    with col3:
        st.markdown("### Phase 3 (Month 2)")
        st.markdown(f"**{len(phase3)} products**")
        st.metric("Revenue Impact", f"€{phase3['annual_revenue_change'].sum():,.0f}/year")
    
    # Detailed table
    st.subheader("📋 Pricing Recommendations")
    
    pricing_display = pricing_actions.copy()
    pricing_display['avg_price'] = pricing_display['avg_price'].apply(lambda x: f"€{x:.2f}")
    pricing_display['recommended_price'] = pricing_display['recommended_price'].apply(lambda x: f"€{x:.2f}")
    pricing_display['price_increase_pct'] = pricing_display['price_increase_pct'].apply(lambda x: f"+{x:.1f}%")
    pricing_display['volume_impact_pct'] = pricing_display['volume_impact_pct'].apply(lambda x: f"{x:.1f}%")
    pricing_display['annual_revenue_change'] = pricing_display['annual_revenue_change'].apply(lambda x: f"€{x:,.0f}")
    pricing_display['daily_velocity'] = pricing_display['daily_velocity'].apply(lambda x: f"{x:.1f}")
    
    display_cols = ['title', 'brand', 'avg_price', 'recommended_price', 'price_increase_pct',
                   'daily_velocity', 'volume_impact_pct', 'annual_revenue_change']
    pricing_display = pricing_display[display_cols]
    pricing_display.columns = ['Product', 'Brand', 'Current Price', 'New Price', 'Increase',
                              'Daily Velocity', 'Volume Impact', 'Annual Gain']
    
    st.dataframe(pricing_display, use_container_width=True, hide_index=True)
    
    # Download
    csv_buffer = BytesIO()
    pricing_actions.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    st.download_button(
        label="📥 Download Pricing Report (CSV)",
        data=csv_buffer,
        file_name="pricing_recommendations.csv",
        mime="text/csv"
    )
    
    # Implementation guidance
    with st.expander("📘 Implementation Guide"):
        st.markdown(f"""
        ### How to Execute This Plan
        
        **Pre-Implementation:**
        1. Review each product for competitive positioning
        2. Validate elasticity assumptions with historical data
        3. Prepare customer communication if needed
        
        **Implementation Steps:**
        1. Update prices in e-commerce platform
        2. Phase 1 products first (highest confidence)
        3. Monitor daily sales velocity for 2 weeks
        4. Adjust Phase 2/3 based on Phase 1 results
        
        **Risk Mitigation:**
        - Start with conservative increases (50% of gap to brand avg)
        - A/B test where possible
        - Prepare rollback plan if velocity drops >20%
        
        **Success Metrics:**
        - Revenue per product (should increase despite volume drop)
        - Overall category revenue
        - Customer retention rate
        - Cart abandonment rate
        """)
else:
    st.info("No pricing optimization opportunities identified with current parameters.")

st.divider()

# =============================================================================
# SECTION 4: PORTFOLIO ACTION PLAN
# =============================================================================
st.header("🏢 Portfolio Action Plan")
st.markdown("*Brand strategy recommendations for growth and risk mitigation*")

portfolio_recs = compute_portfolio_recommendations()

if portfolio_recs:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⭐ Brands to Expand")
        st.markdown("*High growth and/or high revenue brands*")
        
        expand_brands = portfolio_recs.get('expand', [])
        if expand_brands:
            for brand_info in expand_brands:
                st.success(f"""
                **{brand_info['brand']}** ({brand_info['quadrant']})
                
                - Revenue: €{brand_info['revenue']:,.0f} ({brand_info['revenue_share']:.1f}% share)
                - Growth: {brand_info['growth_rate']:+.1f}%
                - Products: {brand_info['n_products']}
                - **Action:** {brand_info['action']}
                """)
        else:
            st.info("No brands flagged for expansion")
    
    with col2:
        st.subheader("⚠️ Brands to Review")
        st.markdown("*Low growth and/or low revenue brands*")
        
        review_brands = portfolio_recs.get('review', [])
        if review_brands:
            for brand_info in review_brands:
                st.warning(f"""
                **{brand_info['brand']}** ({brand_info['quadrant']})
                
                - Revenue: €{brand_info['revenue']:,.0f} ({brand_info['revenue_share']:.1f}% share)
                - Growth: {brand_info['growth_rate']:+.1f}%
                - Products: {brand_info['n_products']}
                - **Action:** {brand_info['action']}
                """)
        else:
            st.info("No brands flagged for review")
    
    # Concentration Risk
    st.subheader("⚠️ Concentration Risk Mitigation")
    
    concentration = portfolio_recs.get('concentration_risk', {})
    if concentration:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Top Brand", concentration.get('top_brand', 'N/A'))
            st.metric("Top Brand Share", f"{concentration.get('top_brand_share', 0):.1f}%")
        
        with col2:
            st.metric("Top 3 Share", f"{concentration.get('top_3_share', 0):.1f}%")
            st.metric("Monthly Risk Exposure", f"€{concentration.get('monthly_risk_exposure', 0):,.0f}")
        
        with col3:
            st.markdown("**Mitigation Target:**")
            st.markdown(concentration.get('mitigation_target', 'N/A'))
        
        with st.expander("📘 Concentration Risk Mitigation Plan"):
            st.markdown(f"""
            ### 12-Month Mitigation Roadmap
            
            **Current State:**
            - {concentration.get('top_brand', 'Top brand')} accounts for {concentration.get('top_brand_share', 0):.1f}% of revenue
            - Top 3 brands account for {concentration.get('top_3_share', 0):.1f}%
            - Monthly exposure: €{concentration.get('monthly_risk_exposure', 0):,.0f}
            
            **Quarter 1: Diversification Planning**
            - Identify 2-3 new brands in adjacent categories
            - Negotiate trial terms with alternative suppliers for top brand's products
            - Document contingency plans for supply disruption
            
            **Quarter 2: Execution**
            - Onboard 1-2 new brands
            - Test alternative suppliers for select SKUs
            - Target: Reduce top brand share to <18%
            
            **Quarter 3-4: Optimization**
            - Scale successful new brands
            - Review performance and adjust portfolio
            - Target: Top brand <15%, Top 3 <40%
            
            **Success Metrics:**
            - Herfindahl-Hirschman Index (HHI) reduction
            - Revenue stability during disruptions
            - New brand contribution to growth
            """)
    
    # Maintain brands (Cash Cows)
    maintain_brands = portfolio_recs.get('maintain', [])
    if maintain_brands:
        with st.expander(f"🐄 Cash Cows - Maintain ({len(maintain_brands)} brands)"):
            for brand_info in maintain_brands:
                st.info(f"""
                **{brand_info['brand']}**: €{brand_info['revenue']:,.0f} revenue, {brand_info['growth_rate']:+.1f}% growth
                
                *Action: {brand_info['action']}*
                """)

st.divider()

# =============================================================================
# FINAL SUMMARY
# =============================================================================
st.header("✅ Action Summary")

total_actions = action_counts['inventory'] + action_counts['pricing'] + action_counts['portfolio']
total_benefit = financial_summary['total_revenue_at_risk'] * 0.7 + financial_summary['total_revenue_opportunity']

st.success(f"""
### Total Impact Summary

**{total_actions} actions identified** across inventory, pricing, and portfolio optimization.

| Category | Actions | Investment | Annual Benefit |
|----------|---------|------------|----------------|
| Inventory | {action_counts['inventory']} | €{financial_summary['total_investment_required']:,.0f} | €{financial_summary['total_revenue_at_risk'] * 0.7:,.0f} (protected) |
| Pricing | {action_counts['pricing']} | €0 | €{financial_summary['total_revenue_opportunity']:,.0f} (new revenue) |
| Portfolio | {action_counts['portfolio']} | TBD | Risk mitigation |

**Total Expected Annual Benefit: €{total_benefit:,.0f}**

**Recommended Next Steps:**
1. ✅ Download reorder report and share with procurement
2. ✅ Review pricing recommendations with merchandising team  
3. ✅ Schedule portfolio strategy session with leadership
""")
