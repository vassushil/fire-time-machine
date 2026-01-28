import numpy as np
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from fire_engine import FireEngine

# --- Configuration ---
st.set_page_config(page_title="FIRE Time Machine", layout="wide")
BACKEND_URL = "http://localhost:8000/simulate"

@st.cache_resource
def load_engine():
    """Loads the engine once and caches it to speed up UI interactions."""
    try:
        # If you have the pickled engine:
        return FireEngine.load("models/fire_engine.pkl")
    except:
        # Fallback to creating a new instance if pickle is missing
        return FireEngine()
    
engine = load_engine()

# --- Title & Header ---
st.title("🏹 Multi-Asset FIRE Time Machine")
st.markdown("""
Predict your financial independence in the Indian market by simulating 1,000+ versions of the future.
*Uses Monte Carlo Simulations based on historical Nifty 50 volatility.*
""")

# --- Sidebar: Personal & Macro Data ---
with st.sidebar:
    st.header("1. Personal Details")
    current_age = st.number_input("Current Age", min_value=18, max_value=80, value=30)
    total_corpus = st.number_input("Current Total Corpus (₹)", min_value=0, value=2000000, step=100000)
    monthly_invest = st.number_input("Monthly SIP / Investment (₹)", min_value=0, value=50000, step=5000)
    annual_expense = st.number_input("Current Annual Expenses (₹)", min_value=0, value=800000, step=10000)
    
    st.header("2. Macro Parameters")
    inflation_rate = st.slider("Expected Annual Inflation (CPI) %", 3.0, 10.0, 6.0) / 100
    sim_years = st.slider("Years to Simulate", 10, 50, 40)

# --- Main Page: Asset Allocation ---
st.header("3. Asset Allocation & Expected Returns")
st.info("Define how your corpus is split and the annualized return you expect from each bucket.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("📈 Equity")
    e_alloc = st.number_input("Allocation %", value=50, key="e_a") / 100
    e_ret = st.number_input("Exp. Return %", value=12.0, key="e_r") / 100

with col2:
    st.subheader("💰 Debt/Bonds")
    d_alloc = st.number_input("Allocation %", value=20, key="d_a") / 100
    d_ret = st.number_input("Exp. Return %", value=7.0, key="d_r") / 100

with col3:
    st.subheader("🟡 Gold")
    g_alloc = st.number_input("Allocation %", value=10, key="g_a") / 100
    g_ret = st.number_input("Exp. Return %", value=8.0, key="g_r") / 100

with col4:
    st.subheader("🏠 Real Estate")
    r_alloc = st.number_input("Allocation %", value=20, key="r_a") / 100
    r_ret = st.number_input("Exp. Return %", value=5.0, key="r_r") / 100

# --- Validation & Simulation ---
total_alloc = round(e_alloc + d_alloc + g_alloc + r_alloc, 2)

if total_alloc != 1.0:
    st.error(f"🚨 Total allocation must equal 100%. Current total: {total_alloc*100}%")
else:
    if st.button("🚀 Run Time Machine Simulation"):
        
        try:
            with st.spinner("Calculating 1,000+ versions of your future..."):
                # DIRECT CALL: No requests.post needed!
                yearly_results = engine.run_simulation(
                    current_age=current_age,
                    initial_corpus=total_corpus,
                    monthly_invest=monthly_invest,
                    annual_expense=annual_expense,
                    inflation_rate=inflation_rate,
                    years=sim_years,
                    equity_ret=e_ret, equity_alloc=e_alloc,
                    debt_ret=d_ret, debt_alloc=d_alloc,
                    gold_ret=g_ret, gold_alloc=g_alloc,
                    re_ret=r_ret, re_alloc=r_alloc
                )

                # --- Process Results ---
                final_wealths = yearly_results[:, -1]
                success_rate = (np.sum(final_wealths > 0) / len(final_wealths)) * 100
                median_wealth = np.median(final_wealths)

                # --- Results Display ---
                st.divider()
                m_col1, m_col2 = st.columns(2)
                
                m_col1.metric("Probability of Success", f"{success_rate:.1f}%")
                m_col2.metric(f"Median Net Worth (Age {current_age + sim_years})", f"₹{median_wealth:,.0f}")

                # --- Visualizations ---
                # 1. Wealth Trajectories
                num_years = yearly_results.shape[1]
                age_axis = list(range(current_age + 1, current_age + num_years + 1))
                sample_indices = np.random.choice(range(len(yearly_results)), size=10, replace=False)
                fig_lines = go.Figure()
                for i, idx in enumerate(sample_indices):
                    fig_lines.add_trace(go.Scatter(
                        x=age_axis, y=yearly_results[idx, :], mode='lines', 
                        name=f"Scenario {i+1}", opacity=0.5,
                        hovertemplate="Age %{x}: ₹%{y:,.0f}<extra></extra>"
                    ))

                # Add FIRE Target Line (25x Expenses)
                fire_target = annual_expense * 25
                fig_lines.add_hline(y=fire_target, line_dash="dash", line_color="green", 
                                    annotation_text="FIRE Target (Nominal)")

                fig_lines.update_layout(
                    title="Yearly Wealth Projections (Inflation Adjusted)",
                    xaxis_title="Age",
                    yaxis_title="Net Worth (₹)",
                    hovermode="x unified",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_lines, use_container_width=True)

                # 2. Asset Mix Pie Chart
                pie_df = pd.DataFrame({
                    "Asset": ["Equity", "Debt", "Gold", "Real Estate"],
                    "Allocation": [e_alloc, d_alloc, g_alloc, r_alloc]
                })
                fig_pie = px.pie(pie_df, values='Allocation', names='Asset', 
                                 title="Target Portfolio Mix", hole=0.4)
                st.plotly_chart(fig_pie)

        except Exception as e:
            st.error(f"Could not connect to the Backend. Is FastAPI running? Error: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Built as a Portfolio Project by an AI Freelancer. Data sources: Nifty 50 Historical Returns via yfinance.")