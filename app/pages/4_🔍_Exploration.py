"""
Exploration Module
==================
Self-service data exploration with interactive tables.
"""

import streamlit as st
import pandas as pd

from data.loader import load_all_data

st.set_page_config(page_title="Data Exploration", page_icon="🔍", layout="wide")

st.title("🔍 Data Exploration")
st.markdown("*Browse and explore all available datasets*")

st.divider()

# Load data
with st.spinner("Loading data..."):
    df, daily, products, brands = load_all_data()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily Aggregates", "🏆 Product Summary", "🏢 Brand Summary", "🔢 Raw Data"])

with tab1:
    st.subheader("Daily Aggregates")
    st.info(f"{len(daily):,} rows | Aggregated sales and inventory metrics per product per day")
    st.dataframe(daily, use_container_width=True, height=500)

with tab2:
    st.subheader("Product Summary")
    st.info(f"{len(products):,} rows | Overall metrics per product")
    st.dataframe(products, use_container_width=True, height=500)

with tab3:
    st.subheader("Brand Summary")
    st.info(f"{len(brands):,} rows | Aggregated metrics per brand")
    st.dataframe(brands, use_container_width=True, height=500)

with tab4:
    st.subheader("Raw Data Sample")
    st.info(f"Showing first 1,000 rows (full dataset: {len(df):,} rows)")
    st.dataframe(df.head(1000), use_container_width=True, height=500)

# Footer
st.divider()
st.caption("""
**Tips:** Use the search box in each table to filter data. Click column headers to sort.
""")
