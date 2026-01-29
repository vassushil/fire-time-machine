import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import babel.numbers
import os
from fire_engine import FireEngine

# --- UI Configuration ---
st.set_page_config(page_title="FIRE Time Machine", layout="wide", initial_sidebar_state="expanded")

# Helper for Indian Currency Formatting
def format_inr(number):
    return babel.numbers.format_currency(number, 'INR', locale='en_IN', format=u'¤ #,##,##0')

def human_format_indian(num):
    if num >= 10_00_00_00:
        return f"₹{num / 10_00_00_00:.2f} Cr"
    elif num >= 1_00_000:
        return f"₹{num / 1_00_000:.2f} L"
    else:
        return f"₹{num:,.0f}"

# --- Logic: Load Engine ---
@st.cache_resource
def get_engine():
    try:
        return FireEngine.load("models/fire_engine.pkl")
    except:
        return FireEngine()

engine = get_engine()

# --- HEADER SECTION ---
st.title("🏹 FIRE Time Machine")
st.markdown("""
<p style='font-size: 1.2em; color: #808495;'>
    Calculate your financial independence using <b>Probabilistic Monte Carlo Modeling</b> tailored for the Indian Market.
</p>
""", unsafe_allow_html=True)

# --- SIDEBAR: PRIMARY INPUTS ---
with st.sidebar:
    st.header("👤 Age Factor")
    current_age = st.number_input("How old are you today?", min_value=18, max_value=70, value=30)
    sim_years = st.slider("Years to simulate", 10, 60, 40, help="Total timeline including your work and retirement years.")
    
    st.divider()
    
    st.header("📉 Macro Adjustments")
    inflation_rate = st.slider("Expected Annual Inflation (CPI) %", 3.0, 12.0, 6.0, 
                               help="Historical avg in India is ~6%. This erodes your purchasing power.") / 100
    
    st.divider()
    st.caption("v2.0 | Built with Python & NumPy")

# --- MAIN LAYOUT ---
col_inputs, col_viz = st.columns([1, 1.2], gap="large")

with col_inputs:
    st.subheader("💰 Financial Foundation")
    
    # Using columns for the "Big Three" to save vertical space
    c1, c2 = st.columns(2)
    with c1:
        total_corpus = st.number_input("Current Savings (₹)", value=2000000, step=100000, 
                                       help="Total current value of all your investments.")
        annual_expense = st.number_input("Current Annual Spend (₹)", value=800000, step=50000,
                                         help="How much you spend in a year today.")
    with c2:
        monthly_invest = st.number_input("Monthly Investment / SIP (₹)", value=50000, step=5000,
                                         help="How much you add to your portfolio every month.")

    st.markdown("### 🏦 Asset Allocation")
    st.info("How is your money split? (Total must equal 100%)")
    
    # UX Improvement: Tabs for Asset Buckets
    tab_eq, tab_dt, tab_gd, tab_re = st.tabs(["📈 Equity", "💰 Debt", "🟡 Gold", "🏠 Real Estate"])
    
    with tab_eq:
        e_alloc = st.slider("Equity Allocation %", 0, 100, 50) / 100
        e_ret = st.number_input("Exp. Equity Return %", value=12.0) / 100
    with tab_dt:
        d_alloc = st.slider("Debt Allocation %", 0, 100, 20) / 100
        d_ret = st.number_input("Exp. Debt Return %", value=7.0) / 100
    with tab_gd:
        g_alloc = st.slider("Gold Allocation %", 0, 100, 10) / 100
        g_ret = st.number_input("Exp. Gold Return %", value=8.0) / 100
    with tab_re:
        r_alloc = st.slider("Real Estate Allocation %", 0, 100, 20) / 100
        r_ret = st.number_input("Exp. RE Return %", value=5.0) / 100

    total_alloc = round(e_alloc + d_alloc + g_alloc + r_alloc, 2)
    
    if total_alloc != 1.0:
        st.error(f"⚠️ Total allocation is {int(total_alloc*100)}%. Please adjust to 100%.")
        run_disabled = True
    else:
        run_disabled = False

    st.divider()
    run_btn = st.button("🚀 Run Time Machine", use_container_width=True, disabled=run_disabled, type="primary")

with col_viz:
    if not run_btn:
        # Show a summary pie chart before simulation
        st.subheader("Current Portfolio Structure")
        pie_data = pd.DataFrame({
            "Asset": ["Equity", "Debt", "Gold", "Real Estate"],
            "Amount": [e_alloc, d_alloc, g_alloc, r_alloc]
        })
        fig_pie = px.pie(pie_data, values='Amount', names='Asset', hole=0.5, 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    else:
        with st.spinner("Simulating 1,000 market futures..."):
            results = engine.run_simulation(
                current_age=current_age, initial_corpus=total_corpus,
                monthly_invest=monthly_invest, annual_expense=annual_expense,
                inflation_rate=inflation_rate, years=sim_years,
                equity_ret=e_ret, equity_alloc=e_alloc,
                debt_ret=d_ret, debt_alloc=d_alloc,
                gold_ret=g_ret, gold_alloc=g_alloc,
                re_ret=r_ret, re_alloc=r_alloc
            )
            
            # --- Analysis ---
            final_wealths = results[:, -1]
            success_rate = (np.sum(final_wealths > 0) / len(final_wealths)) * 100
            median_wealth = np.median(final_wealths)

            # 1. Success Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = success_rate,
                domain = {'x': [0, 1], 'y': [0, 1]},
                number = {'suffix': "%", 'font': {'size': 40}},
                title = {'text': "Confidence Level", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 50], 'color': "#FF6B6B"},    # Unsafe
                        {'range': [50, 85], 'color': "#FFD93D"},   # Risky
                        {'range': [85, 100], 'color': "#6BCB77"}   # Robust
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            fig_gauge.add_annotation(x=0.15, y=0.25, text="UNSAFE", showarrow=False, font=dict(color="red"))
            fig_gauge.add_annotation(x=0.5, y=0.5, text="MODERATE", showarrow=False, font=dict(color="orange"))
            fig_gauge.add_annotation(x=0.85, y=0.25, text="ROBUST", showarrow=False, font=dict(color="green"))
            fig_gauge.update_layout(height=350, margin=dict(t=0, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # 2. Key Metrics
            m1, m2 = st.columns(2)
            m1.metric("Final Median Wealth", human_format_indian(median_wealth))
            m2.metric("Projected Age", f"{current_age + sim_years} Years")

# --- FULL WIDTH VISUALIZATION ---
if run_btn:
    st.divider()
    st.subheader("Wealth Trajectories Over Time")
    
    num_years = results.shape[1]
    age_axis = list(range(current_age + 1, current_age + num_years + 1))
    sample_indices = np.random.choice(range(len(results)), size=15, replace=False)
    
    fig_line = go.Figure()
    for idx in sample_indices:
        # Color code: Green if success, Red if wealth hits zero
        path = results[idx, :]
        color = "rgba(0, 204, 150, 0.4)" if path[-1] > 0 else "rgba(255, 75, 75, 0.4)"
        
        fig_line.add_trace(go.Scatter(
            x=age_axis, y=path, mode='lines', line=dict(color=color, width=1.5),
            hovertemplate=f"Age: %{{x}}<br>Wealth: %{{y:,.0f}}<extra></extra>"
        ))
    
    # Add FIRE Target Line
    fire_target = annual_expense * 25
    fig_line.add_hline(y=fire_target, line_dash="dash", line_color="white", 
                       annotation_text=f"FIRE Target: {format_inr(fire_target)}")

    fig_line.update_layout(
        xaxis_title="Your Age",
        yaxis_title="Corpus Value (₹)",
        template="plotly_dark",
        height=500,
        hovermode="x unified"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.success(f"Simulation complete. In the Indian context, a success rate above 85% is considered 'Robust'.")