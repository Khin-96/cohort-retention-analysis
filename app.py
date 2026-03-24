import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from datetime import datetime

# Page config
st.set_page_config(page_title="Glovo-style Cohort Retention & LTV Dashboard", layout="wide")

st.title("🚀 Glovo-style Cohort Retention & LTV Dashboard")
st.markdown("""
This dashboard provides a comprehensive analysis of user retention and Lifetime Value (LTV) using synthetic Glovo-style data.
- **Cohort Analysis:** Tracks user retention across monthly signup cohorts.
- **LTV Analysis:** Uses BG/NBD and Gamma-Gamma models to predict future customer value.
""")

# Load Data
@st.cache_data
def load_data():
    try:
        users = pd.read_csv('users.csv', parse_dates=['signup_date'])
        orders = pd.read_csv('orders.csv', parse_dates=['order_date'])
        payments = pd.read_csv('payments.csv')
        return users, orders, payments
    except FileNotFoundError:
        st.error("Data files not found. Please run 'py generate_data.py' first.")
        return None, None, None

users, orders, payments = load_data()

if users is not None:
    # --- COHORT ANALYSIS ---
    st.header("1. Cohort Retention Matrix")
    
    # Preprocessing
    orders['order_month'] = orders['order_date'].dt.to_period('M').astype(str)
    
    # Get first order month per user
    first_order = orders.groupby('user_id')['order_date'].min().reset_index()
    first_order.columns = ['user_id', 'cohort_month']
    first_order['cohort_month'] = first_order['cohort_month'].dt.to_period('M').astype(str)
    
    # Join orders with first order month
    cohort_data = pd.merge(orders, first_order, on='user_id')
    
    # Calculate order period (number of months after first order)
    def calculate_period(row):
        start = datetime.strptime(row['cohort_month'], '%Y-%m')
        current = datetime.strptime(row['order_month'], '%Y-%m')
        return (current.year - start.year) * 12 + (current.month - start.month)

    cohort_data['period'] = cohort_data.apply(calculate_period, axis=1)
    
    # Pivot for retention matrix
    cohort_pivot = cohort_data.pivot_table(index='cohort_month', columns='period', values='user_id', aggfunc='nunique')
    
    # Calculate retention percentage
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
    
    # Plotly Heatmap
    fig_cohort = px.imshow(
        retention_matrix,
        text_auto=".0%",
        color_continuous_scale='Viridis',
        labels=dict(x="Months Since First Order", y="Cohort Month", color="Retention %"),
        title="Monthly Cohort Retention Matrix (%)"
    )
    st.plotly_chart(fig_cohort, use_container_width=True)

    # --- LTV SEGMENTATION ---
    st.header("2. LTV Predictions (BG/NBD Model)")
    
    # Prepare data for lifetimes
    lf_data = summary_data_from_transaction_data(
        orders,
        'user_id',
        'order_date',
        monetary_value_col='order_value',
        observation_period_end=orders['order_date'].max()
    )
    
    # BG/NBD model to predict frequency
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(lf_data['frequency'], lf_data['recency'], lf_data['T'])
    
    # Predict next 3 months purchases
    t = 90 # days
    lf_data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, lf_data['frequency'], lf_data['recency'], lf_data['T'])
    
    # Gamma-Gamma model for monetary value
    # Filter customers with at least one repeat purchase
    returning_customers_summary = lf_data[lf_data['frequency'] > 0]
    
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(returning_customers_summary['frequency'], returning_customers_summary['monetary_value'])
    
    lf_data['predicted_ltv'] = ggf.customer_lifetime_value(
        bgf,
        lf_data['frequency'],
        lf_data['recency'],
        lf_data['T'],
        lf_data['monetary_value'],
        time=3, # months
        discount_rate=0.01
    )
    
    # Segmentation
    lf_data['segment'] = pd.qcut(lf_data['predicted_ltv'], q=4, labels=['Low Value', 'Medium Value', 'High Value', 'VIP'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Segments (Predicted 3-Month LTV)")
        segment_counts = lf_data['segment'].value_counts().reset_index()
        fig_segments = px.pie(segment_counts, values='count', names='segment', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_segments, use_container_width=True)
        
    with col2:
        st.subheader("Top 10 Customers by Predicted LTV")
        st.dataframe(lf_data.sort_values(by='predicted_ltv', ascending=False).head(10)[['predicted_ltv', 'segment']])

    st.info("💡 LTV calculation uses the lifetimes library implementing BG/NBD (Beta-Geometric/Negative Binomial Distribution) and Gamma-Gamma model.")
