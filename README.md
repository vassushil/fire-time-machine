# 🏹 FIRE Time Machine

A high-performance **Multi-Asset Monte Carlo Simulator** tailored for the Indian market. This project demonstrates the transition from traditional software engineering to **AI/ML Engineering** by combining stochastic modeling with modern machine learning.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fire-time-machine.streamlit.app/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

---

## 🚀 Overview
Most retirement calculators use static "average" returns (e.g., a flat 12%). This is dangerous because it ignores **Sequence of Returns Risk**. 

The **FIRE Time Machine** uses **Monte Carlo Simulations** to run 1,000+ parallel "lives," accounting for market volatility and inflation shocks. It also features an **XGBoost Regressor** trained on these simulations to provide instant outcome predictions.