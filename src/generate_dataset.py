import pandas as pd
import numpy as np
from fire_engine import FireEngine
import os

def generate_synthetic_data(n_samples=5000):
    engine = FireEngine()
    data_records = []

    print(f"Generating {n_samples} synthetic financial lives...")

    for i in range(n_samples):
        # 1. Randomize User Inputs (Features)
        age = np.random.randint(22, 50)
        corpus = np.random.uniform(100000, 10000000) # 1L to 1Cr
        monthly_sip = np.random.uniform(5000, 200000) # 5k to 2L
        expense = np.random.uniform(300000, 2400000) # 3L to 24L
        inflation = np.random.uniform(0.04, 0.09)     # 4% to 9%
        
        # 2. Randomize Asset Returns
        e_ret = np.random.uniform(0.08, 0.16)
        d_ret = np.random.uniform(0.05, 0.08)
        g_ret = np.random.uniform(0.04, 0.12)
        r_ret = np.random.uniform(0.03, 0.10)

        # 3. Run a single simulation path (sims=1)
        # We want the "outcome" of one specific random market path
        result = engine.run_simulation(
            current_age=age,
            initial_corpus=corpus,
            monthly_invest=monthly_sip,
            annual_expense=expense,
            inflation_rate=inflation,
            years=30, # Standardize to a 30-year horizon for the ML model
            sims=1,   
            equity_ret=e_ret, equity_alloc=0.5,
            debt_ret=d_ret, debt_alloc=0.2,
            gold_ret=g_ret, gold_alloc=0.1,
            re_ret=r_ret, re_alloc=0.2
        )

        # 4. Capture the Final Wealth (The Label/Target)
        final_wealth = result[0, -1] 

        data_records.append({
            "current_age": age,
            "initial_corpus": corpus,
            "monthly_invest": monthly_sip,
            "annual_expense": expense,
            "inflation_rate": inflation,
            "equity_return_expected": e_ret,
            "final_wealth": final_wealth # This is our 'y' variable
        })

        if i % 1000 == 0:
            print(f"Progress: {i}/{n_samples} records created.")

    # Create DataFrame and Save
    df = pd.DataFrame(data_records)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_fire_data.csv', index=False)
    print("Success! Dataset saved to data/synthetic_fire_data.csv")

