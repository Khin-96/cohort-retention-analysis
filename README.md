# Advanced Retention Modeling & Stochastic LTV Analysis (MIT/Harvard Tier)

This research-grade analytics platform transcends basic reporting by implementing **Bayesian Non-Parametric Survival Analysis** and **State-Space Markov Chains** to quantify customer lifecycle dynamics. It is designed for high-scale tech firms where retention is the primary lever for valuation.

## Core Methodologies
- **Survival Analysis (Kaplan-Meier)**: Models the continuous probability of user persistence, moving beyond discrete cohort buckets to identify exact hazard rates.
- **Markov Chain State Transitions**: Tracks the stochastic movement of users between 'New', 'Active', 'At-Risk', and 'Churned' states, calculating the equilibrium (steady-state) distribution of the user base.
- **Bayesian LTV Modeling**: Uses Hierarchical BG/NBD and Gamma-Gamma models to project future frequency and monetary value distributions.

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
- **Python**: `pandas`, `faker`, `lifetimes`, `lifelines`, `plotly`
- **Dashboard**: `streamlit`
- **Research-grade Analytics**: Kaplan-Meier Survival, Markov State Transitions, Bayesian BG/NBD.
