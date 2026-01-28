import numpy as np
import json
import os
import pickle

class FireEngine:
    def __init__(self, params_path='data/market_params.json'):
        with open(params_path, 'r') as f:
            self.market = json.load(f)

    def run_simulation_monthly(self, initial_savings, monthly_invest, annual_expense, 
                       inflation_rate, risk_free_rate, equity_allocation=0.7, 
                       years=40, sims=1000):
        
        # Market Parameters (Historical Equity Stats)
        mu = self.market['annual_mean_return']
        sigma = self.market['annual_volatility']
        
        n_steps = years * 12
        dt = 1/12
        
        # Convert Annual Rates to Monthly
        monthly_inflation = (1 + inflation_rate)**(1/12) - 1
        monthly_rf = (1 + risk_free_rate)**(1/12) - 1
        
        # Equity Component (Stochastic)
        monthly_mu_eq = mu / 12
        monthly_sigma_eq = sigma / np.sqrt(12)

        results = np.zeros((sims, n_steps))
        results[:, 0] = initial_savings

        for t in range(1, n_steps):
            # 1. Simulate Equity Return (Geometric Brownian Motion)
            shocks = np.random.normal(0, 1, sims)
            equity_growth = np.exp((monthly_mu_eq - 0.5 * monthly_sigma_eq**2) * dt + 
                                   monthly_sigma_eq * np.sqrt(dt) * shocks)
            
            # 2. Portfolio Return (Weighted average of Equity and Risk-Free)
            # We assume a fixed rebalancing to the chosen equity_allocation
            portfolio_growth = (equity_allocation * equity_growth) + ((1 - equity_allocation) * (1 + monthly_rf))
            
            # 3. Inflate Expenses
            current_monthly_expense = (annual_expense / 12) * ((1 + monthly_inflation) ** t)
            
            # 4. Calculate New Wealth
            results[:, t] = (results[:, t-1] * portfolio_growth) + monthly_invest - current_monthly_expense
            
            # Floor at zero
            results[:, t] = np.maximum(results[:, t], 0)

        return results
    
    def run_simulation_old(self, initial_savings, monthly_invest, annual_expense, 
                       inflation_rate, risk_free_rate, equity_allocation=0.7, 
                       years=40, sims=1000):
        
        mu = self.market['annual_mean_return']
        sigma = self.market['annual_volatility']
        
        # We still calculate monthly to keep the "Sequence of Returns" accuracy
        n_months = years * 12
        dt = 1/12
        
        monthly_inflation = (1 + inflation_rate)**(1/12) - 1
        monthly_rf = (1 + risk_free_rate)**(1/12) - 1
        monthly_mu_eq = mu / 12
        monthly_sigma_eq = sigma / np.sqrt(12)

        # Full monthly matrix for internal calculation
        monthly_results = np.zeros((sims, n_months))
        monthly_results[:, 0] = initial_savings

        for t in range(1, n_months):
            shocks = np.random.normal(0, 1, sims)
            equity_growth = np.exp((monthly_mu_eq - 0.5 * monthly_sigma_eq**2) * dt + 
                                   monthly_sigma_eq * np.sqrt(dt) * shocks)
            
            portfolio_growth = (equity_allocation * equity_growth) + ((1 - equity_allocation) * (1 + monthly_rf))
            current_monthly_expense = (annual_expense / 12) * ((1 + monthly_inflation) ** t)
            
            monthly_results[:, t] = (monthly_results[:, t-1] * portfolio_growth) + monthly_invest - current_monthly_expense
            monthly_results[:, t] = np.maximum(monthly_results[:, t], 0)

        # --- NEW: Downsample to Yearly ---
        # We pick every 12th month (index 11, 23, 35...)
        yearly_results = monthly_results[:, 11::12] 
        return yearly_results
    
    def run_simulation(self, current_age, initial_corpus, monthly_invest, annual_expense, 
                       inflation_rate, years=40, sims=1000, 
                       # New Bucket Inputs (Returns and Allocations)
                       equity_ret=0.12, equity_alloc=0.5,
                       debt_ret=0.07, debt_alloc=0.2,
                       gold_ret=0.08, gold_alloc=0.1,
                       re_ret=0.05, re_alloc=0.2):
        
        # Market volatility (Sigma) from historical Nifty 50 data
        sigma_eq = self.market['annual_volatility']
        
        n_months = years * 12
        dt = 1/12
        
        # Convert all annual inputs to monthly
        m_infl = (1 + inflation_rate)**dt - 1
        m_equity = (1 + equity_ret)**dt - 1
        m_debt = (1 + debt_ret)**dt - 1
        m_gold = (1 + gold_ret)**dt - 1
        m_re = (1 + re_ret)**dt - 1

        monthly_results = np.zeros((sims, n_months))
        monthly_results[:, 0] = initial_corpus

        for t in range(1, n_months):
            # Equity remains stochastic (GBM)
            shocks = np.random.normal(0, 1, sims)
            eq_growth = np.exp((m_equity - 0.5 * (sigma_eq**2 * dt)) + (sigma_eq * np.sqrt(dt) * shocks))
            
            # Others are modeled as stable monthly accrual
            # You can also add 'shocks' to Gold/RE if you have that data
            portfolio_growth = (
                (equity_alloc * eq_growth) + 
                (debt_alloc * (1 + m_debt)) + 
                (gold_alloc * (1 + m_gold)) + 
                (re_alloc * (1 + m_re))
            )
            
            # Expense grows with inflation
            cur_monthly_exp = (annual_expense / 12) * ((1 + m_infl) ** t)
            
            # Net Wealth Calculation
            monthly_results[:, t] = (monthly_results[:, t-1] * portfolio_growth) + monthly_invest - cur_monthly_exp
            monthly_results[:, t] = np.maximum(monthly_results[:, t], 0)

        # Downsample to yearly
        return monthly_results[:, 11::12]

    def save(self, path='models/fire_engine.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path='models/fire_engine.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)