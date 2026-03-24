# Cohort Retention & LTV Dashboard (Glovo-style)

This project provides a comprehensive analysis of user retention and Lifetime Value (LTV) using synthetic Glovo-style data. It is designed to demonstrate key skills required for a data analyst position at high-growth tech companies.

## Key Features
- **SQL Schema Design**: Models `users`, `orders`, and `payments` tables common in food delivery systems.
- **Cohort Retention Matrix**: Visualizes user retention trends across monthly cohorts.
- **LTV Modeling**: Implements BG/NBD and Gamma-Gamma models using the `lifetimes` library to predict future customer value and segment users.
- **Streamlit Dashboard**: A beautiful, interactive dashboard with Plotly heatmap and segmentation charts.

## Crucial SQL: Monthly Cohort Retention
The core logic for cohort retention implemented using window functions. This is the exact type of query used in technical interviews for analyst roles.

```sql
-- Cohort retention matrix using window functions
WITH first_order AS (
  SELECT 
    user_id, 
    DATE_TRUNC('month', MIN(order_date)) AS cohort_month
  FROM orders 
  GROUP BY user_id
)
SELECT 
    cohort_month,
    DATEDIFF('month', cohort_month, DATE_TRUNC('month', o.order_date)) AS period,
    COUNT(DISTINCT o.user_id) AS retained_users
FROM orders o 
JOIN first_order f USING (user_id)
GROUP BY 1, 2
ORDER BY 1, 2;
```

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate synthetic data:
   ```bash
   py generate_data.py
   ```
3. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## Technologies Used
- **Python**: `pandas`, `faker`, `lifetimes`, `plotly`
- **Dashboard**: `streamlit`
- **Analytics**: BG/NBD modeling, Gamma-Gamma monetary model, Cohort Analysis
