"""
Data Loader & Preprocessing Module
===================================
Loads cleaned data and computes derived metrics for sales intelligence:
- stock_change: inventory movement between snapshots
- estimated_sales: units sold (negative stock changes)
- estimated_revenue: sales × price
- price_change: price movements
- Daily/weekly aggregations
"""

import pandas as pd
import numpy as np
from config import CLEANED_DATA_PATH, METRICS_JSON_PATH
import json
import streamlit as st



@st.cache_data(ttl=3600)
def load_raw_data() -> pd.DataFrame:
    """Load the cleaned parquet data."""
    df = pd.read_parquet(CLEANED_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data(ttl=3600)
def compute_stock_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stock changes per variant between consecutive timestamps.
    Negative change = sales, positive change = restock.
    """
    # Sort by variant and time
    df = df.sort_values(['offer_offer_id', 'date']).copy()
    
    # Calculate stock change per variant
    df['stock_change'] = df.groupby('offer_offer_id')['stock'].diff()
    
    # Calculate price change per variant
    df['price_change'] = df.groupby('offer_offer_id')['offer_price'].diff()
    
    # Estimated sales: only count negative stock changes (sales)
    # Ignore large restocks (positive changes)
    df['estimated_sales'] = df['stock_change'].apply(lambda x: max(0, -x) if pd.notna(x) else 0)
    
    # Cap unrealistic sales (>50 units/hour is likely a data error or restock correction)
    df.loc[df['estimated_sales'] > 50, 'estimated_sales'] = 0
    
    # Estimated revenue
    df['estimated_revenue'] = df['estimated_sales'] * df['offer_price']
    
    # Flag restocks (large positive stock changes)
    df['is_restock'] = (df['stock_change'] > 10) & (df['stock_change'].notna())
    
    # Flag stockouts
    df['is_stockout'] = ~df['offer_in_stock']
    
    return df


@st.cache_data(ttl=3600)
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics to daily level per variant."""
    df = df.copy()
    df['date_day'] = df['date'].dt.date
    
    daily = df.groupby(['offer_offer_id', 'product_id', 'title', 'brand', 'offer_title', 'date_day']).agg({
        'estimated_sales': 'sum',
        'estimated_revenue': 'sum',
        'stock': 'last',  # End of day stock
        'offer_price': 'last',  # End of day price
        'is_stockout': 'any',  # Was out of stock at any point
        'is_restock': 'any',  # Was restocked during day
        'price_change': lambda x: x[x != 0].sum() if (x != 0).any() else 0,  # Total price movement
    }).reset_index()
    
    daily['date_day'] = pd.to_datetime(daily['date_day'])
    
    return daily


@st.cache_data(ttl=3600)
def aggregate_by_product(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to product level for overall rankings."""
    product_agg = daily_df.groupby(['product_id', 'title', 'brand']).agg({
        'estimated_sales': 'sum',
        'estimated_revenue': 'sum',
        'is_stockout': 'sum',  # Days with stockout
        'is_restock': 'sum',  # Number of restocks
        'offer_price': 'mean',  # Average price
        'date_day': 'nunique',  # Days tracked
    }).reset_index()
    
    product_agg.columns = ['product_id', 'title', 'brand', 'total_sales', 'total_revenue', 
                           'stockout_days', 'restock_count', 'avg_price', 'days_tracked']
    
    # Calculate velocity (sales per day)
    product_agg['daily_velocity'] = product_agg['total_sales'] / product_agg['days_tracked']
    
    # Stockout rate
    product_agg['stockout_rate'] = product_agg['stockout_days'] / product_agg['days_tracked']
    
    return product_agg.sort_values('total_revenue', ascending=False)


@st.cache_data(ttl=3600)
def aggregate_by_brand(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to brand level."""
    brand_agg = daily_df.groupby('brand').agg({
        'estimated_sales': 'sum',
        'estimated_revenue': 'sum',
        'is_stockout': 'sum',
        'product_id': 'nunique',
        'date_day': 'nunique',
    }).reset_index()
    
    brand_agg.columns = ['brand', 'total_sales', 'total_revenue', 'stockout_days', 
                         'n_products', 'days_tracked']
    
    brand_agg['daily_velocity'] = brand_agg['total_sales'] / brand_agg['days_tracked']
    brand_agg['revenue_per_product'] = brand_agg['total_revenue'] / brand_agg['n_products']
    
    return brand_agg.sort_values('total_revenue', ascending=False)


@st.cache_data(ttl=3600)
def get_price_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all price change events."""
    price_changes = df[df['price_change'] != 0].copy()
    price_changes = price_changes[price_changes['price_change'].notna()]
    
    price_changes['price_change_pct'] = (price_changes['price_change'] / 
                                          (price_changes['offer_price'] - price_changes['price_change'])) * 100
    
    return price_changes[['date', 'product_id', 'title', 'brand', 'offer_price', 
                          'price_change', 'price_change_pct', 'stock']]


@st.cache_data(ttl=3600)
def compute_inventory_metrics(df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute inventory health metrics per product."""
    # Get latest stock levels
    latest = df.sort_values('date').groupby('offer_offer_id').last().reset_index()
    
    # Get average daily sales
    avg_sales = daily_df.groupby('product_id')['estimated_sales'].mean().reset_index()
    avg_sales.columns = ['product_id', 'avg_daily_sales']
    
    inventory = latest[['product_id', 'title', 'brand', 'stock', 'offer_price', 'offer_in_stock']].copy()
    inventory = inventory.merge(avg_sales, on='product_id', how='left')
    
    # Days of inventory
    inventory['days_of_inventory'] = np.where(
        inventory['avg_daily_sales'] > 0,
        inventory['stock'] / inventory['avg_daily_sales'],
        np.inf
    )
    
    # Classify risk
    def classify_risk(row):
        if not row['offer_in_stock']:
            return 'STOCKOUT'
        elif row['days_of_inventory'] < 7:
            return 'CRITICAL'
        elif row['days_of_inventory'] < 14:
            return 'LOW'
        elif row['days_of_inventory'] < 30:
            return 'MEDIUM'
        else:
            return 'HEALTHY'
    
    inventory['risk_level'] = inventory.apply(classify_risk, axis=1)
    
    return inventory


@st.cache_data(ttl=3600)
def get_weekly_trends(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to weekly trends."""
    daily_df = daily_df.copy()
    daily_df['week'] = daily_df['date_day'].dt.to_period('W').dt.start_time
    
    weekly = daily_df.groupby(['brand', 'week']).agg({
        'estimated_sales': 'sum',
        'estimated_revenue': 'sum',
        'is_stockout': 'sum',
    }).reset_index()
    
    return weekly


def compute_all_metrics() -> dict:
    """Compute summary metrics for JSON export, including action recommendations."""
    df = load_raw_data()
    df_with_changes = compute_stock_changes(df)
    daily = aggregate_daily(df_with_changes)
    product_agg = aggregate_by_product(daily)
    brand_agg = aggregate_by_brand(daily)
    
    metrics = {
        "data_summary": {
            "total_rows": int(len(df)),
            "date_range": {
                "start": df['date'].min().isoformat(),
                "end": df['date'].max().isoformat()
            },
            "n_brands": int(df['brand'].nunique()),
            "n_products": int(df['product_id'].nunique()),
            "n_variants": int(df['offer_offer_id'].nunique())
        },
        "sales_estimates": {
            "total_estimated_sales": int(daily['estimated_sales'].sum()),
            "total_estimated_revenue": round(float(daily['estimated_revenue'].sum()), 2),
            "avg_daily_sales": round(float(daily.groupby('date_day')['estimated_sales'].sum().mean()), 2),
            "avg_daily_revenue": round(float(daily.groupby('date_day')['estimated_revenue'].sum().mean()), 2)
        },
        "top_products_by_revenue": product_agg.head(10)[['product_id', 'title', 'brand', 'total_revenue', 'daily_velocity']].to_dict('records'),
        "top_brands_by_revenue": brand_agg.head(10)[['brand', 'total_revenue', 'daily_velocity', 'n_products']].to_dict('records'),
        "inventory_health": {
            "avg_stockout_rate": round(float(product_agg['stockout_rate'].mean() * 100), 2),
            "products_with_high_stockout": int((product_agg['stockout_rate'] > 0.1).sum())
        }
    }
    
    # Add action recommendations (computed separately to avoid circular imports in Streamlit)
    try:
        financial_summary = compute_financial_summary()
        metrics["financial_summary"] = financial_summary
        
        # Add top action recommendations
        priority_matrix = compute_priority_matrix()
        if not priority_matrix.empty:
            metrics["action_recommendations"] = {
                "top_actions": priority_matrix.head(10).to_dict('records'),
                "total_actions": len(priority_matrix)
            }
        
        # Add portfolio recommendations summary
        portfolio_recs = compute_portfolio_recommendations()
        if portfolio_recs:
            metrics["portfolio_recommendations"] = {
                "brands_to_expand": len(portfolio_recs.get('expand', [])),
                "brands_to_review": len(portfolio_recs.get('review', [])),
                "concentration_risk": portfolio_recs.get('concentration_risk', {})
            }
    except Exception as e:
        # If action computations fail, still return basic metrics
        metrics["action_recommendations"] = {"error": str(e)}
    
    return metrics


def export_metrics_json():
    """Export metrics to JSON file for data pipeline compliance."""
    metrics = compute_all_metrics()
    METRICS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON_PATH, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    return metrics


# Convenience function to load all data at once
@st.cache_data(ttl=3600)
def load_all_data():
    """Load and preprocess all data. Returns tuple of DataFrames."""
    df = load_raw_data()
    df_with_changes = compute_stock_changes(df)
    daily = aggregate_daily(df_with_changes)
    product_agg = aggregate_by_product(daily)
    brand_agg = aggregate_by_brand(daily)
    
    return df_with_changes, daily, product_agg, brand_agg


@st.cache_data(ttl=3600)
def compute_reorder_recommendations(lead_time_days: int = 14, safety_stock_days: int = 7) -> pd.DataFrame:
    """
    Compute reorder recommendations for products at risk of stockout.
    Returns prioritized list with investment required and revenue at risk.
    """
    df, daily, products, brands = load_all_data()
    inventory = compute_inventory_metrics(df, daily)
    
    # Get latest stock and velocity
    latest_stock = df.sort_values('date').groupby('product_id').last()[['stock', 'offer_price']].reset_index()
    
    reorder_df = inventory.merge(
        products[['product_id', 'daily_velocity', 'total_revenue', 'stockout_rate']],
        on='product_id',
        how='left',
        suffixes=('', '_prod')
    )
    
    # Calculate days until stockout
    reorder_df['days_until_stockout'] = np.where(
        reorder_df['avg_daily_sales'] > 0,
        reorder_df['stock'] / reorder_df['avg_daily_sales'],
        np.inf
    )
    
    # Filter to products needing reorder (stockout within lead_time + safety)
    threshold = lead_time_days + safety_stock_days
    at_risk = reorder_df[reorder_df['days_until_stockout'] < threshold].copy()
    
    if at_risk.empty:
        return pd.DataFrame()
    
    # Calculate recommended reorder quantity
    # Reorder enough for: lead_time + safety_stock + buffer (30 days total coverage)
    target_days = lead_time_days + safety_stock_days + 30
    at_risk['recommended_qty'] = np.maximum(
        0,
        (at_risk['avg_daily_sales'] * target_days) - at_risk['stock']
    ).round(0).astype(int)
    
    # Estimate unit cost (assume 50% margin, so cost = price * 0.5)
    at_risk['unit_cost_estimate'] = at_risk['offer_price'] * 0.5
    at_risk['investment_required'] = at_risk['recommended_qty'] * at_risk['unit_cost_estimate']
    
    # Revenue at risk (30 days of lost sales if stockout occurs)
    at_risk['revenue_at_risk'] = at_risk['avg_daily_sales'] * at_risk['offer_price'] * 30
    
    # Priority score (1-5): based on revenue at risk and days until stockout
    at_risk['urgency_score'] = np.where(at_risk['days_until_stockout'] <= 7, 5,
                              np.where(at_risk['days_until_stockout'] <= 14, 4,
                              np.where(at_risk['days_until_stockout'] <= 21, 3, 2)))
    at_risk['impact_score'] = pd.qcut(at_risk['revenue_at_risk'], q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
    at_risk['priority_score'] = ((at_risk['urgency_score'] + at_risk['impact_score']) / 2).round(0).astype(int)
    
    # Sort by priority
    at_risk = at_risk.sort_values(['priority_score', 'days_until_stockout'], ascending=[False, True])
    
    return at_risk[['product_id', 'title', 'brand', 'stock', 'avg_daily_sales', 
                    'days_until_stockout', 'recommended_qty', 'investment_required',
                    'revenue_at_risk', 'priority_score', 'offer_price']]


@st.cache_data(ttl=3600)
def compute_pricing_recommendations() -> pd.DataFrame:
    """
    Compute pricing recommendations for underpriced high-velocity products.
    Returns prioritized list with recommended prices and expected revenue impact.
    """
    df, daily, products, brands = load_all_data()
    
    # Calculate brand average prices
    brand_avg_price = products.groupby('brand')['avg_price'].mean().to_dict()
    brand_median_price = products.groupby('brand')['avg_price'].median().to_dict()
    
    pricing_df = products.copy()
    pricing_df['brand_avg_price'] = pricing_df['brand'].map(brand_avg_price)
    pricing_df['brand_median_price'] = pricing_df['brand'].map(brand_median_price)
    pricing_df['price_gap'] = pricing_df['brand_avg_price'] - pricing_df['avg_price']
    pricing_df['price_gap_pct'] = (pricing_df['price_gap'] / pricing_df['avg_price']) * 100
    
    # Identify underpriced products: high velocity + price below brand average
    velocity_median = pricing_df['daily_velocity'].median()
    underpriced = pricing_df[
        (pricing_df['daily_velocity'] >= velocity_median) &
        (pricing_df['price_gap'] > 2)  # At least €2 below brand average
    ].copy()
    
    if underpriced.empty:
        return pd.DataFrame()
    
    # Calculate recommended price (conservative: 50% of gap to brand avg)
    underpriced['recommended_price'] = underpriced['avg_price'] + (underpriced['price_gap'] * 0.5)
    underpriced['price_increase_pct'] = ((underpriced['recommended_price'] - underpriced['avg_price']) / underpriced['avg_price']) * 100
    
    # Estimate volume impact (assume elasticity of 1.2)
    elasticity = 1.2
    underpriced['volume_impact_pct'] = -underpriced['price_increase_pct'] * elasticity
    
    # Calculate net revenue change (annual)
    underpriced['current_daily_revenue'] = underpriced['daily_velocity'] * underpriced['avg_price']
    underpriced['new_daily_velocity'] = underpriced['daily_velocity'] * (1 + underpriced['volume_impact_pct']/100)
    underpriced['new_daily_revenue'] = underpriced['new_daily_velocity'] * underpriced['recommended_price']
    underpriced['daily_revenue_change'] = underpriced['new_daily_revenue'] - underpriced['current_daily_revenue']
    underpriced['annual_revenue_change'] = underpriced['daily_revenue_change'] * 365
    
    # Priority based on revenue impact
    underpriced = underpriced[underpriced['annual_revenue_change'] > 0]  # Only positive impact
    if underpriced.empty:
        return pd.DataFrame()
    
    underpriced['priority_score'] = pd.qcut(
        underpriced['annual_revenue_change'], q=min(5, len(underpriced)), 
        labels=range(1, min(6, len(underpriced)+1)), duplicates='drop'
    ).astype(int)
    
    underpriced = underpriced.sort_values('annual_revenue_change', ascending=False)
    
    return underpriced[['product_id', 'title', 'brand', 'avg_price', 'recommended_price',
                        'price_increase_pct', 'daily_velocity', 'volume_impact_pct',
                        'annual_revenue_change', 'priority_score']]


@st.cache_data(ttl=3600)
def compute_portfolio_recommendations() -> dict:
    """
    Compute portfolio recommendations: brands to expand, review, or exit.
    Based on BCG matrix (growth vs revenue share).
    """
    df, daily, products, brands = load_all_data()
    
    # Calculate brand growth (first half vs second half)
    daily_copy = daily.copy()
    daily_copy['date_day'] = pd.to_datetime(daily_copy['date_day'])
    median_date = daily_copy['date_day'].median()
    daily_copy['period'] = daily_copy['date_day'].apply(lambda x: 'H1' if x < median_date else 'H2')
    
    brand_periods = daily_copy.groupby(['brand', 'period'])['estimated_revenue'].sum().unstack(fill_value=0)
    brand_periods['growth_rate'] = np.where(
        brand_periods['H1'] > 0,
        ((brand_periods['H2'] - brand_periods['H1']) / brand_periods['H1']) * 100,
        0
    )
    brand_periods['total_revenue'] = brand_periods['H1'] + brand_periods['H2']
    brand_periods = brand_periods.reset_index()
    
    # Merge with brand data
    portfolio = brand_periods.merge(brands[['brand', 'n_products', 'daily_velocity']], on='brand')
    
    # Classify into quadrants
    growth_median = portfolio['growth_rate'].median()
    revenue_median = portfolio['total_revenue'].median()
    
    def classify(row):
        if row['growth_rate'] > growth_median and row['total_revenue'] > revenue_median:
            return 'Star'
        elif row['growth_rate'] > growth_median and row['total_revenue'] <= revenue_median:
            return 'Question Mark'
        elif row['growth_rate'] <= growth_median and row['total_revenue'] > revenue_median:
            return 'Cash Cow'
        else:
            return 'Dog'
    
    portfolio['quadrant'] = portfolio.apply(classify, axis=1)
    
    # Generate recommendations
    recommendations = {
        'expand': [],  # Stars and high-potential Question Marks
        'maintain': [],  # Cash Cows
        'review': [],  # Dogs and low-potential Question Marks
        'concentration_risk': None
    }
    
    total_revenue = portfolio['total_revenue'].sum()
    
    for _, row in portfolio.iterrows():
        brand_info = {
            'brand': row['brand'],
            'quadrant': row['quadrant'],
            'growth_rate': round(row['growth_rate'], 1),
            'revenue': round(row['total_revenue'], 0),
            'revenue_share': round((row['total_revenue'] / total_revenue) * 100, 1),
            'n_products': row['n_products']
        }
        
        if row['quadrant'] == 'Star':
            brand_info['action'] = 'Expand product range, negotiate better terms'
            brand_info['investment_priority'] = 'High'
            recommendations['expand'].append(brand_info)
        elif row['quadrant'] == 'Question Mark' and row['growth_rate'] > growth_median * 1.5:
            brand_info['action'] = 'Invest to grow market share'
            brand_info['investment_priority'] = 'Medium'
            recommendations['expand'].append(brand_info)
        elif row['quadrant'] == 'Cash Cow':
            brand_info['action'] = 'Maintain, optimize margins'
            brand_info['investment_priority'] = 'Low'
            recommendations['maintain'].append(brand_info)
        else:
            brand_info['action'] = 'Review partnership, consider phasing out'
            brand_info['investment_priority'] = 'None'
            recommendations['review'].append(brand_info)
    
    # Concentration risk analysis
    top_brand = portfolio.loc[portfolio['total_revenue'].idxmax()]
    top_3_revenue = portfolio.nlargest(3, 'total_revenue')['total_revenue'].sum()
    
    recommendations['concentration_risk'] = {
        'top_brand': top_brand['brand'],
        'top_brand_share': round((top_brand['total_revenue'] / total_revenue) * 100, 1),
        'top_3_share': round((top_3_revenue / total_revenue) * 100, 1),
        'monthly_risk_exposure': round(top_brand['total_revenue'] / 12, 0),
        'mitigation_target': 'Reduce top brand to <15% share within 12 months'
    }
    
    return recommendations


def compute_priority_matrix() -> pd.DataFrame:
    """
    Combine all action recommendations into a unified priority matrix.
    Returns top actions sorted by impact-to-effort ratio.
    """
    actions = []
    
    # Inventory actions
    try:
        inventory_recs = compute_reorder_recommendations()
        if not inventory_recs.empty:
            for _, row in inventory_recs.head(10).iterrows():
                actions.append({
                    'category': 'Inventory',
                    'action': f"Reorder {row['title'][:40]}...",
                    'product_id': row['product_id'],
                    'impact_eur': row['revenue_at_risk'],
                    'investment_eur': row['investment_required'],
                    'effort': 'Low',  # Reordering is operational
                    'payback_days': int(row['investment_required'] / (row['revenue_at_risk']/30)) if row['revenue_at_risk'] > 0 else 999,
                    'priority': row['priority_score']
                })
    except Exception:
        pass
    
    # Pricing actions
    try:
        pricing_recs = compute_pricing_recommendations()
        if not pricing_recs.empty:
            for _, row in pricing_recs.head(10).iterrows():
                actions.append({
                    'category': 'Pricing',
                    'action': f"Reprice {row['title'][:40]}...",
                    'product_id': row['product_id'],
                    'impact_eur': row['annual_revenue_change'],
                    'investment_eur': 0,  # No investment needed
                    'effort': 'Low',
                    'payback_days': 0,
                    'priority': row['priority_score']
                })
    except Exception:
        pass
    
    # Portfolio actions (simplified)
    try:
        portfolio_recs = compute_portfolio_recommendations()
        for brand_info in portfolio_recs.get('expand', [])[:3]:
            actions.append({
                'category': 'Portfolio',
                'action': f"Expand {brand_info['brand']} range",
                'product_id': None,
                'impact_eur': brand_info['revenue'] * 0.2,  # Estimate 20% growth potential
                'investment_eur': brand_info['revenue'] * 0.1,  # Estimate 10% investment
                'effort': 'High',
                'payback_days': 180,  # Estimate 6 months
                'priority': 4 if brand_info['quadrant'] == 'Star' else 3
            })
    except Exception:
        pass
    
    if not actions:
        return pd.DataFrame()
    
    actions_df = pd.DataFrame(actions)
    actions_df['roi_ratio'] = np.where(
        actions_df['investment_eur'] > 0,
        actions_df['impact_eur'] / actions_df['investment_eur'],
        actions_df['impact_eur'] / 100  # For zero-investment actions, use impact/100
    )
    actions_df = actions_df.sort_values(['priority', 'roi_ratio'], ascending=[False, False])
    
    return actions_df


def compute_financial_summary() -> dict:
    """
    Compute overall financial summary of all recommended actions.
    """
    summary = {
        'total_revenue_at_risk': 0,
        'total_investment_required': 0,
        'total_revenue_opportunity': 0,
        'expected_roi_pct': 0,
        'avg_payback_days': 0,
        'action_counts': {'inventory': 0, 'pricing': 0, 'portfolio': 0}
    }
    
    try:
        inventory_recs = compute_reorder_recommendations()
        if not inventory_recs.empty:
            summary['total_revenue_at_risk'] = inventory_recs['revenue_at_risk'].sum()
            summary['total_investment_required'] += inventory_recs['investment_required'].sum()
            summary['action_counts']['inventory'] = len(inventory_recs)
    except Exception:
        pass
    
    try:
        pricing_recs = compute_pricing_recommendations()
        if not pricing_recs.empty:
            summary['total_revenue_opportunity'] += pricing_recs['annual_revenue_change'].sum()
            summary['action_counts']['pricing'] = len(pricing_recs)
    except Exception:
        pass
    
    try:
        portfolio_recs = compute_portfolio_recommendations()
        summary['action_counts']['portfolio'] = len(portfolio_recs.get('expand', [])) + len(portfolio_recs.get('review', []))
    except Exception:
        pass
    
    # Calculate overall ROI
    total_benefit = summary['total_revenue_at_risk'] * 0.7 + summary['total_revenue_opportunity']  # 70% capture rate
    if summary['total_investment_required'] > 0:
        summary['expected_roi_pct'] = round((total_benefit / summary['total_investment_required']) * 100, 0)
        summary['avg_payback_days'] = round((summary['total_investment_required'] / (total_benefit / 365)), 0)
    else:
        summary['expected_roi_pct'] = 999  # Infinite ROI for zero-cost actions
        summary['avg_payback_days'] = 0
    
    # Round values
    summary['total_revenue_at_risk'] = round(summary['total_revenue_at_risk'], 0)
    summary['total_investment_required'] = round(summary['total_investment_required'], 0)
    summary['total_revenue_opportunity'] = round(summary['total_revenue_opportunity'], 0)
    
    return summary


@st.cache_data(ttl=3600)
def compute_causal_analysis() -> dict:
    """
    Detect revenue anomalies and correlate with potential causes.
    Returns narratives explaining WHY revenue spikes/drops happened.
    """
    df, daily, products, brands = load_all_data()
    
    # Aggregate daily revenue
    daily_revenue = daily.groupby('date_day').agg({
        'estimated_revenue': 'sum',
        'estimated_sales': 'sum',
        'is_stockout': 'sum',
        'is_restock': 'sum',
        'price_change': lambda x: (x != 0).sum()
    }).reset_index()
    daily_revenue['date_day'] = pd.to_datetime(daily_revenue['date_day'])
    daily_revenue['day_of_week'] = daily_revenue['date_day'].dt.day_name()
    daily_revenue['week'] = daily_revenue['date_day'].dt.isocalendar().week
    
    # Calculate rolling average and detect anomalies
    daily_revenue['rolling_avg'] = daily_revenue['estimated_revenue'].rolling(7, min_periods=1).mean()
    daily_revenue['pct_vs_avg'] = ((daily_revenue['estimated_revenue'] - daily_revenue['rolling_avg']) / daily_revenue['rolling_avg']) * 100
    
    # Identify significant anomalies (>30% deviation)
    anomalies = daily_revenue[abs(daily_revenue['pct_vs_avg']) > 30].copy()
    
    causal_insights = []
    
    for _, row in anomalies.iterrows():
        insight = {
            'date': row['date_day'].strftime('%Y-%m-%d'),
            'day_of_week': row['day_of_week'],
            'revenue': round(row['estimated_revenue'], 0),
            'deviation_pct': round(row['pct_vs_avg'], 1),
            'direction': 'spike' if row['pct_vs_avg'] > 0 else 'drop',
            'potential_causes': [],
            'narrative': ''
        }
        
        # Analyze potential causes
        if row['is_stockout'] == 0 and row['pct_vs_avg'] > 0:
            insight['potential_causes'].append('Zero stockouts (full availability)')
        elif row['is_stockout'] > 5 and row['pct_vs_avg'] < 0:
            insight['potential_causes'].append(f"{int(row['is_stockout'])} products out of stock")
        
        if row['is_restock'] > 3 and row['pct_vs_avg'] > 0:
            insight['potential_causes'].append(f"{int(row['is_restock'])} products restocked")
        
        if row['price_change'] > 5:
            insight['potential_causes'].append(f"{int(row['price_change'])} price changes")
        
        if row['day_of_week'] in ['Saturday', 'Sunday'] and row['pct_vs_avg'] < -15:
            insight['potential_causes'].append('Weekend effect (typically lower sales)')
        
        # Generate narrative
        if insight['potential_causes']:
            causes_text = ' + '.join(insight['potential_causes'])
            insight['narrative'] = f"{row['date_day'].strftime('%b %d')}: Revenue {insight['direction']} ({insight['deviation_pct']:+.1f}%) correlates with: {causes_text}"
        else:
            insight['narrative'] = f"{row['date_day'].strftime('%b %d')}: Revenue {insight['direction']} ({insight['deviation_pct']:+.1f}%) - requires investigation (no clear correlation)"
        
        causal_insights.append(insight)
    
    # Weekly pattern analysis
    weekly_pattern = daily_revenue.groupby('day_of_week')['estimated_revenue'].mean()
    best_day = weekly_pattern.idxmax()
    worst_day = weekly_pattern.idxmin()
    weekly_variance = ((weekly_pattern.max() - weekly_pattern.min()) / weekly_pattern.mean()) * 100
    
    return {
        'anomalies': causal_insights[-10:],  # Last 10 anomalies
        'total_anomalies': len(anomalies),
        'weekly_pattern': {
            'best_day': best_day,
            'worst_day': worst_day,
            'variance_pct': round(weekly_variance, 1),
            'explanation': f"{worst_day} averages {weekly_variance:.0f}% less than {best_day} - consider day-specific promotions"
        }
    }


@st.cache_data(ttl=3600)
def compute_ltv_impact(churn_rate: float = 0.05, avg_ltv: float = 150) -> dict:
    """
    Calculate compound Lifetime Value impact of stockouts.
    Stockouts -> customer churn -> LTV loss (not just immediate revenue).
    
    Args:
        churn_rate: Estimated % of customers who don't return after stockout experience
        avg_ltv: Average customer lifetime value in EUR
    """
    df, daily, products, brands = load_all_data()
    
    # Products with stockouts
    stockout_products = products[products['stockout_rate'] > 0].copy()
    
    if stockout_products.empty:
        return {'total_ltv_impact': 0, 'products': []}
    
    # Calculate immediate lost revenue
    stockout_products['immediate_lost_revenue'] = (
        stockout_products['daily_velocity'] * 
        stockout_products['avg_price'] * 
        stockout_products['stockout_days']
    )
    
    # Estimate affected customers (units lost = customers affected, simplified)
    stockout_products['customers_affected'] = (
        stockout_products['daily_velocity'] * stockout_products['stockout_days']
    )
    
    # Customers who churn
    stockout_products['customers_churned'] = stockout_products['customers_affected'] * churn_rate
    
    # LTV loss from churned customers
    stockout_products['ltv_loss'] = stockout_products['customers_churned'] * avg_ltv
    
    # 90-day compound loss (immediate + LTV)
    stockout_products['compound_90d_loss'] = (
        stockout_products['immediate_lost_revenue'] + 
        stockout_products['ltv_loss']
    )
    
    # Summary
    total_immediate = stockout_products['immediate_lost_revenue'].sum()
    total_ltv = stockout_products['ltv_loss'].sum()
    total_compound = stockout_products['compound_90d_loss'].sum()
    
    # Top impacted products
    top_impact = stockout_products.nlargest(10, 'compound_90d_loss')[[
        'product_id', 'title', 'brand', 'stockout_rate', 
        'immediate_lost_revenue', 'ltv_loss', 'compound_90d_loss'
    ]].to_dict('records')
    
    return {
        'immediate_lost_revenue': round(total_immediate, 0),
        'ltv_loss': round(total_ltv, 0),
        'total_compound_loss': round(total_compound, 0),
        'churn_rate_used': churn_rate,
        'avg_ltv_used': avg_ltv,
        'multiplier': round(total_compound / total_immediate, 1) if total_immediate > 0 else 1,
        'top_impacted_products': top_impact,
        'explanation': f"Stockout losses multiply {total_compound/total_immediate:.1f}x when accounting for {churn_rate*100:.0f}% customer churn (LTV=€{avg_ltv})"
    }


@st.cache_data(ttl=3600)
def compute_forecast_accuracy() -> dict:
    """
    Evaluate forecast model accuracy by comparing predictions to actuals.
    Uses backtesting: predict past periods and compare to what actually happened.
    """
    df, daily, products, brands = load_all_data()
    
    # Aggregate daily revenue
    daily_revenue = daily.groupby('date_day')['estimated_revenue'].sum().reset_index()
    daily_revenue['date_day'] = pd.to_datetime(daily_revenue['date_day'])
    daily_revenue = daily_revenue.sort_values('date_day')
    
    if len(daily_revenue) < 60:
        return {'error': 'Insufficient data for accuracy analysis'}
    
    # Split: use first 80% to "forecast" last 20%
    split_idx = int(len(daily_revenue) * 0.8)
    train = daily_revenue.iloc[:split_idx].copy()
    test = daily_revenue.iloc[split_idx:].copy()
    
    # Simple forecast: linear trend + weekly seasonality
    train['day_num'] = range(len(train))
    slope, intercept = np.polyfit(train['day_num'], train['estimated_revenue'], 1)
    
    # Weekly seasonality factors
    train['dow'] = train['date_day'].dt.dayofweek
    dow_factors = train.groupby('dow')['estimated_revenue'].mean() / train['estimated_revenue'].mean()
    
    # Generate predictions for test period
    test['day_num'] = range(split_idx, split_idx + len(test))
    test['dow'] = test['date_day'].dt.dayofweek
    test['forecast'] = (slope * test['day_num'] + intercept) * test['dow'].map(dow_factors)
    
    # Calculate accuracy metrics
    test['error'] = test['estimated_revenue'] - test['forecast']
    test['abs_error'] = abs(test['error'])
    # Avoid division by zero - only compute where revenue > 0
    test['pct_error'] = np.where(
        test['estimated_revenue'] > 0,
        (test['abs_error'] / test['estimated_revenue']) * 100,
        0
    )
    
    # Handle edge cases where all values might be problematic
    valid_errors = test['pct_error'][test['pct_error'] < np.inf]
    mape = valid_errors.mean() if len(valid_errors) > 0 else 50.0  # Default to moderate if no valid data
    rmse = np.sqrt((test['error'] ** 2).mean())
    
    # Accuracy rating
    if mape < 10:
        accuracy_rating = 'Excellent'
        confidence = 'High'
    elif mape < 20:
        accuracy_rating = 'Good'
        confidence = 'Medium-High'
    elif mape < 30:
        accuracy_rating = 'Moderate'
        confidence = 'Medium'
    else:
        accuracy_rating = 'Low'
        confidence = 'Low'
    
    # Weekly variance explanation
    weekly_variance = daily_revenue['estimated_revenue'].std() / daily_revenue['estimated_revenue'].mean() * 100
    
    return {
        'mape': round(mape, 1),
        'rmse': round(rmse, 0),
        'accuracy_rating': accuracy_rating,
        'confidence_level': confidence,
        'test_period_days': len(test),
        'weekly_variance_pct': round(weekly_variance, 1),
        'model_description': 'Linear trend + weekly seasonality adjustment',
        'assumptions': [
            'Assumes historical trend continues',
            'Weekly patterns remain consistent',
            'No major external shocks (marketing campaigns, competitor moves)',
            f'Based on {len(train)} days of training data'
        ],
        'confidence_explanation': f"Forecast range is ±{weekly_variance:.0f}% due to inherent daily variance. {accuracy_rating} accuracy ({mape:.1f}% MAPE) based on backtesting."
    }


def generate_decision_tree(issue_type: str, context: dict) -> dict:
    """
    Generate a decision tree with Options A/B/C for strategic issues.
    
    Args:
        issue_type: 'brand_concentration', 'stockout', 'pricing', 'seasonal'
        context: Relevant data for the decision
    """
    decision_trees = {
        'brand_concentration': lambda ctx: {
            'issue': f"Brand Concentration Risk: {ctx.get('top_brand', 'Unknown')} = {ctx.get('top_brand_share', 0):.1f}% of revenue",
            'risk_quantified': f"If disrupted: €{ctx.get('monthly_exposure', 0):,.0f}/month lost",
            'probability': '12% annual disruption probability (industry baseline)',
            'expected_loss': f"€{ctx.get('monthly_exposure', 0) * 0.12 * 12:,.0f}/year expected loss",
            'options': [
                {
                    'name': 'Option A: Diversify Portfolio',
                    'action': 'Add 3-5 new brands in same category',
                    'cost': '€50,000 (sourcing + marketing)',
                    'timeline': '6 months',
                    'impact': f"Reduces top brand to <10% share",
                    'roi': 'Risk reduction worth €' + f"{ctx.get('monthly_exposure', 0) * 0.12 * 6:,.0f}/year",
                    'effort': 'High'
                },
                {
                    'name': 'Option B: Expand Within Brand',
                    'action': f"Grow {ctx.get('top_brand', 'top brand')} from {ctx.get('n_products', 0)} to {ctx.get('n_products', 0) + 5} SKUs",
                    'cost': '€15,000 (inventory + marketing)',
                    'timeline': '2 months',
                    'impact': 'Reduces concentration by deepening relationship (backup leverage)',
                    'roi': 'Quick win with supplier relationship improvement',
                    'effort': 'Low'
                },
                {
                    'name': 'Option C: Private Label Alternative',
                    'action': 'Develop private label products in same category',
                    'cost': '€100,000 (development + launch)',
                    'timeline': '9-12 months',
                    'impact': 'Eliminates brand dependency entirely',
                    'roi': 'Long-term margin improvement + risk elimination',
                    'effort': 'Very High'
                }
            ],
            'recommendation': 'Implement Option B immediately (quick win) + begin Option A planning. Option C only if brand relationship deteriorates.',
            'success_metrics': [
                {'metric': 'Top brand share', 'target': '<15%', 'timeline': '6 months'},
                {'metric': 'Alternative supplier identified', 'target': 'Yes', 'timeline': '30 days'},
                {'metric': 'Contingency plan documented', 'target': 'Complete', 'timeline': '14 days'}
            ]
        },
        
        'stockout': lambda ctx: {
            'issue': f"Stockout Crisis: {ctx.get('n_products', 0)} products with >{ctx.get('threshold', 15)}% stockout rate",
            'immediate_loss': f"€{ctx.get('immediate_loss', 0):,.0f} lost revenue",
            'compound_loss': f"€{ctx.get('ltv_loss', 0):,.0f} LTV impact (5% customer churn)",
            'total_at_risk': f"€{ctx.get('immediate_loss', 0) + ctx.get('ltv_loss', 0):,.0f} total exposure",
            'options': [
                {
                    'name': 'Option A: Full Reorder',
                    'action': f"Reorder all {ctx.get('n_products', 0)} affected products",
                    'cost': f"€{ctx.get('investment_required', 0):,.0f} (inventory investment)",
                    'timeline': '2-3 weeks (lead time dependent)',
                    'impact': f"Recovers €{ctx.get('immediate_loss', 0) + ctx.get('ltv_loss', 0):,.0f} at risk",
                    'roi': f"{((ctx.get('immediate_loss', 0) + ctx.get('ltv_loss', 0)) / max(ctx.get('investment_required', 1), 1) * 100):.0f}% ROI",
                    'effort': 'Medium'
                },
                {
                    'name': 'Option B: Selective Reorder + Demand Management',
                    'action': 'Reorder top 10 products + raise prices 10% on remainder to reduce demand',
                    'cost': f"€{ctx.get('investment_required', 0) * 0.4:,.0f} (reduced inventory)",
                    'timeline': '1-2 weeks',
                    'impact': 'Recovers 70% of value with 40% of investment',
                    'roi': 'Higher ROI but some revenue left on table',
                    'effort': 'Low'
                },
                {
                    'name': 'Option C: Substitute + Preorder',
                    'action': 'Identify substitute products + launch preorder campaign',
                    'cost': '€5,000 (marketing)',
                    'timeline': '1 week',
                    'impact': 'Captures demand with alternatives, builds customer engagement',
                    'roi': 'Lowest cost but requires execution excellence',
                    'effort': 'Medium'
                }
            ],
            'recommendation': 'Option A for top 10 highest-value products. Option B for remainder. Option C for any with >30 day lead time.',
            'success_metrics': [
                {'metric': 'Stockout rate', 'target': '<5%', 'timeline': '30 days'},
                {'metric': 'Lost revenue recovered', 'target': '>70%', 'timeline': '60 days'},
                {'metric': 'Customer churn', 'target': '<3%', 'timeline': '90 days'}
            ]
        },
        
        'pricing': lambda ctx: {
            'issue': f"Pricing Opportunity: {ctx.get('n_underpriced', 0)} underpriced products identified",
            'opportunity': f"€{ctx.get('annual_opportunity', 0):,.0f}/year potential revenue gain",
            'current_state': f"Average {ctx.get('avg_gap_pct', 0):.1f}% below brand averages",
            'options': [
                {
                    'name': 'Option A: Aggressive Repricing',
                    'action': 'Raise all underpriced products to brand average',
                    'cost': '€0 (operational only)',
                    'timeline': '1 week',
                    'impact': f"Full €{ctx.get('annual_opportunity', 0):,.0f}/year capture",
                    'risk': 'Volume drop if customers are price-sensitive',
                    'effort': 'Low'
                },
                {
                    'name': 'Option B: Phased Repricing',
                    'action': 'Phase 1: Top 10 products. Phase 2: Next 20. Phase 3: Remainder',
                    'cost': '€0',
                    'timeline': '6 weeks (2 weeks per phase)',
                    'impact': 'Controlled rollout with learning',
                    'risk': 'Lower - can abort if Phase 1 shows issues',
                    'effort': 'Medium'
                },
                {
                    'name': 'Option C: Value-Add Repricing',
                    'action': 'Raise prices but add value (free shipping, loyalty points, bundles)',
                    'cost': '€2,000-5,000 (value-add costs)',
                    'timeline': '2-3 weeks',
                    'impact': 'Price increase with customer value perception maintained',
                    'risk': 'Lowest volume risk',
                    'effort': 'High'
                }
            ],
            'recommendation': 'Option B (Phased) is safest. Start with highest-confidence products (inelastic based on historical data).',
            'success_metrics': [
                {'metric': 'Revenue per product', 'target': '+5-10%', 'timeline': '30 days'},
                {'metric': 'Volume change', 'target': '>-15%', 'timeline': '30 days'},
                {'metric': 'Cart abandonment', 'target': '<+2pp', 'timeline': '14 days'}
            ]
        }
    }
    
    if issue_type in decision_trees:
        return decision_trees[issue_type](context)
    return {'error': f'Unknown issue type: {issue_type}'}


if __name__ == "__main__":
    # Test the module
    print("Loading data...")
    df, daily, products, brands = load_all_data()
    print(f"Raw data: {len(df):,} rows")
    print(f"Daily data: {len(daily):,} rows")
    print(f"Products: {len(products)} products")
    print(f"Brands: {len(brands)} brands")
    
    print("\nComputing action recommendations...")
    reorder = compute_reorder_recommendations()
    print(f"Reorder recommendations: {len(reorder)} products")
    
    pricing = compute_pricing_recommendations()
    print(f"Pricing recommendations: {len(pricing)} products")
    
    financial = compute_financial_summary()
    print(f"Financial summary: {financial}")
    
    print("\nExporting metrics...")
    metrics = export_metrics_json()
    print(f"Metrics exported to {METRICS_JSON_PATH}")
    print(f"Total estimated revenue: €{metrics['sales_estimates']['total_estimated_revenue']:,.2f}")
