"""
Strategic Decisions Module
==========================
Deep-dive business cases with Options A/B/C decision trees.
Provides CEO-level strategic guidance with costs, timelines, and ROI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data.loader import (
    load_all_data,
    compute_portfolio_recommendations,
    compute_ltv_impact,
    compute_reorder_recommendations,
    compute_pricing_recommendations,
    generate_decision_tree
)

st.set_page_config(page_title="Strategic Decisions", page_icon="📊", layout="wide")

st.title("📊 Strategic Decisions")
st.markdown("*Business cases with Options A/B/C — costs, timelines, ROI, and recommendations*")

# Load data
with st.spinner("Computing strategic analysis..."):
    df, daily, products, brands = load_all_data()
    portfolio_recs = compute_portfolio_recommendations()
    ltv_impact = compute_ltv_impact()

st.divider()

# =============================================================================
# DECISION 1: BRAND CONCENTRATION RISK
# =============================================================================
st.header("🏢 Decision 1: Brand Concentration Risk")

concentration = portfolio_recs.get('concentration_risk', {})
top_brand = brands.iloc[0] if not brands.empty else None

if concentration and top_brand is not None:
    # Build context for decision tree
    context = {
        'top_brand': concentration.get('top_brand', 'Unknown'),
        'top_brand_share': concentration.get('top_brand_share', 0),
        'monthly_exposure': concentration.get('monthly_risk_exposure', 0),
        'n_products': int(top_brand['n_products']) if 'n_products' in top_brand else 0
    }
    
    decision = generate_decision_tree('brand_concentration', context)
    
    # Issue Statement
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(f"**{decision['issue']}**")
    with col2:
        st.warning(f"**Risk:** {decision['risk_quantified']}")
    with col3:
        st.info(f"**Expected Loss:** {decision['expected_loss']}")
    
    st.markdown(f"*{decision['probability']}*")
    
    # Options
    st.subheader("📋 Strategic Options")
    
    cols = st.columns(3)
    for i, option in enumerate(decision['options']):
        with cols[i]:
            effort_color = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴', 'Very High': '🔴'}.get(option['effort'], '⚪')
            st.markdown(f"### {option['name']}")
            st.markdown(f"**Action:** {option['action']}")
            st.markdown(f"**Cost:** {option['cost']}")
            st.markdown(f"**Timeline:** {option['timeline']}")
            st.markdown(f"**Impact:** {option['impact']}")
            st.markdown(f"**ROI:** {option['roi']}")
            st.markdown(f"**Effort:** {effort_color} {option['effort']}")
    
    # Recommendation
    st.success(f"**✅ RECOMMENDATION:** {decision['recommendation']}")
    
    # Success Metrics
    with st.expander("📈 Success Metrics & Tracking"):
        for metric in decision['success_metrics']:
            st.markdown(f"- **{metric['metric']}**: Target {metric['target']} within {metric['timeline']}")

st.divider()

# =============================================================================
# DECISION 2: STOCKOUT STRATEGY
# =============================================================================
st.header("📦 Decision 2: Stockout Strategy")

# Calculate stockout context
high_stockout = products[products['stockout_rate'] > 0.15]
n_stockout_products = len(high_stockout)

if n_stockout_products > 0:
    immediate_loss = ltv_impact.get('immediate_lost_revenue', 0)
    ltv_loss = ltv_impact.get('ltv_loss', 0)
    
    # Get investment required from reorder recommendations
    try:
        reorder_recs = compute_reorder_recommendations()
        investment_required = reorder_recs['investment_required'].sum() if not reorder_recs.empty else 0
    except:
        investment_required = 50000  # Default estimate
    
    context = {
        'n_products': n_stockout_products,
        'threshold': 15,
        'immediate_loss': immediate_loss,
        'ltv_loss': ltv_loss,
        'investment_required': investment_required
    }
    
    decision = generate_decision_tree('stockout', context)
    
    # Issue Statement with compound impact
    col1, col2 = st.columns(2)
    with col1:
        st.error(f"**{decision['issue']}**")
        st.markdown(f"**Immediate Loss:** {decision['immediate_loss']}")
    with col2:
        st.warning(f"**Compound LTV Impact:** {decision['compound_loss']}")
        st.markdown(f"**Total Exposure:** {decision['total_at_risk']}")
    
    # Visual: Impact multiplier
    multiplier = ltv_impact.get('multiplier', 1)
    st.info(f"💡 **Key Insight:** Stockout losses multiply **{multiplier}x** when accounting for customer churn. {ltv_impact.get('explanation', '')}")
    
    # Options
    st.subheader("📋 Strategic Options")
    
    cols = st.columns(3)
    for i, option in enumerate(decision['options']):
        with cols[i]:
            effort_color = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}.get(option['effort'], '⚪')
            st.markdown(f"### {option['name']}")
            st.markdown(f"**Action:** {option['action']}")
            st.markdown(f"**Cost:** {option['cost']}")
            st.markdown(f"**Timeline:** {option['timeline']}")
            st.markdown(f"**Impact:** {option['impact']}")
            st.markdown(f"**ROI:** {option['roi']}")
            st.markdown(f"**Effort:** {effort_color} {option['effort']}")
    
    # Recommendation
    st.success(f"**✅ RECOMMENDATION:** {decision['recommendation']}")
    
    # Success Metrics
    with st.expander("📈 Success Metrics & Tracking"):
        for metric in decision['success_metrics']:
            st.markdown(f"- **{metric['metric']}**: Target {metric['target']} within {metric['timeline']}")
    
    # Top impacted products
    with st.expander("🔍 Top 10 Impacted Products (by compound loss)"):
        top_products = ltv_impact.get('top_impacted_products', [])
        if top_products:
            top_df = pd.DataFrame(top_products)
            top_df['stockout_rate'] = top_df['stockout_rate'].apply(lambda x: f"{x*100:.1f}%")
            top_df['immediate_lost_revenue'] = top_df['immediate_lost_revenue'].apply(lambda x: f"€{x:,.0f}")
            top_df['ltv_loss'] = top_df['ltv_loss'].apply(lambda x: f"€{x:,.0f}")
            top_df['compound_90d_loss'] = top_df['compound_90d_loss'].apply(lambda x: f"€{x:,.0f}")
            top_df = top_df[['title', 'brand', 'stockout_rate', 'immediate_lost_revenue', 'ltv_loss', 'compound_90d_loss']]
            top_df.columns = ['Product', 'Brand', 'Stockout Rate', 'Immediate Loss', 'LTV Loss', 'Total Impact']
            st.dataframe(top_df, use_container_width=True, hide_index=True)
else:
    st.success("✅ No significant stockout issues (no products with >15% stockout rate)")

st.divider()

# =============================================================================
# DECISION 3: PRICING OPTIMIZATION
# =============================================================================
st.header("💰 Decision 3: Pricing Optimization")

try:
    pricing_recs = compute_pricing_recommendations()
    n_underpriced = len(pricing_recs) if not pricing_recs.empty else 0
    annual_opportunity = pricing_recs['annual_revenue_change'].sum() if not pricing_recs.empty else 0
    avg_gap_pct = pricing_recs['price_increase_pct'].mean() if not pricing_recs.empty else 0
except:
    n_underpriced = 0
    annual_opportunity = 0
    avg_gap_pct = 0

if n_underpriced > 0:
    context = {
        'n_underpriced': n_underpriced,
        'annual_opportunity': annual_opportunity,
        'avg_gap_pct': avg_gap_pct
    }
    
    decision = generate_decision_tree('pricing', context)
    
    # Issue Statement
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**{decision['issue']}**")
    with col2:
        st.success(f"**Opportunity:** {decision['opportunity']}")
    with col3:
        st.warning(f"**Current State:** {decision['current_state']}")
    
    # Options
    st.subheader("📋 Strategic Options")
    
    cols = st.columns(3)
    for i, option in enumerate(decision['options']):
        with cols[i]:
            effort_color = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}.get(option['effort'], '⚪')
            st.markdown(f"### {option['name']}")
            st.markdown(f"**Action:** {option['action']}")
            st.markdown(f"**Cost:** {option['cost']}")
            st.markdown(f"**Timeline:** {option['timeline']}")
            st.markdown(f"**Impact:** {option['impact']}")
            st.markdown(f"**Risk:** {option.get('risk', 'N/A')}")
            st.markdown(f"**Effort:** {effort_color} {option['effort']}")
    
    # Recommendation
    st.success(f"**✅ RECOMMENDATION:** {decision['recommendation']}")
    
    # Success Metrics
    with st.expander("📈 Success Metrics & Tracking"):
        for metric in decision['success_metrics']:
            st.markdown(f"- **{metric['metric']}**: Target {metric['target']} within {metric['timeline']}")
else:
    st.info("No significant pricing opportunities identified with current parameters.")

st.divider()

# =============================================================================
# DECISION SUMMARY
# =============================================================================
st.header("📝 Decision Summary")

# Calculate total opportunity
total_risk = ltv_impact.get('total_compound_loss', 0)
total_opportunity = annual_opportunity
total_addressable = total_risk + total_opportunity

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Risk to Mitigate", f"€{total_risk:,.0f}")
with col2:
    st.metric("Total Opportunity", f"€{total_opportunity:,.0f}")
with col3:
    st.metric("Total Addressable Value", f"€{total_addressable:,.0f}")

# Priority ranking
st.subheader("🎯 Recommended Priority Order")

priorities = []
if total_risk > 0:
    priorities.append(("1️⃣", "Stockout Strategy", "Highest urgency - revenue actively being lost", "This week"))
if n_underpriced > 0:
    priorities.append(("2️⃣", "Pricing Optimization", "Zero-cost opportunity", "Next 2 weeks"))
if concentration:
    priorities.append(("3️⃣", "Brand Concentration", "Strategic risk mitigation", "This quarter"))

for emoji, title, reason, timeline in priorities:
    st.markdown(f"{emoji} **{title}** — {reason} (Timeline: {timeline})")

# Implementation sequencing
with st.expander("📅 Implementation Sequencing"):
    st.markdown("""
    ### Week 1-2: Quick Wins
    1. **Stockout:** Place reorders for top 10 critical products (Option A for high-value)
    2. **Pricing:** Implement Phase 1 price adjustments on top 10 products
    
    ### Week 3-4: Momentum Building  
    3. **Stockout:** Implement demand management (Option B) for remaining products
    4. **Pricing:** Monitor Phase 1 results, proceed to Phase 2 if successful
    
    ### Month 2-3: Strategic Foundation
    5. **Brand Concentration:** Begin Option B (expand within top brand)
    6. **Brand Concentration:** Start Option A planning (new brand sourcing)
    
    ### Ongoing: Monitoring
    - Track all success metrics weekly
    - Adjust strategy based on results
    - Report progress to leadership monthly
    """)

st.divider()

# Final CTA
st.info("""
**Next Steps:**
1. Review each decision with relevant team leads
2. Validate cost estimates with finance/procurement  
3. Set calendar reminders for success metric reviews
4. Begin Week 1 actions immediately

👉 Visit **🎬 Action Center** for detailed execution lists and downloadable reports.
""")
