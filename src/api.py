from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import uvicorn
from fire_engine import FireEngine

# Initialize the FastAPI app
app = FastAPI(title="FIRE Time Machine API")

# Load the pickled engine once on startup
try:
    engine = FireEngine.load("models/fire_engine.pkl")
except FileNotFoundError:
    # Fallback if pickle doesn't exist yet: initialize a new one
    from data_fetcher import fetch_and_analyze_market
    fetch_and_analyze_market()
    engine = FireEngine()
    engine.save("models/fire_engine.pkl")

# --- Request Schema ---
class FireRequest(BaseModel):
    current_age: int = Field(..., ge=18, le=80)
    initial_corpus: float = Field(..., ge=0)
    monthly_invest: float = Field(..., ge=0)
    annual_expense: float = Field(..., ge=0)
    inflation_rate: float = Field(..., ge=0, le=0.20)
    years: int = Field(40, ge=1, le=60)
    
    # Bucket Parameters
    equity_ret: float
    equity_alloc: float
    debt_ret: float
    debt_alloc: float
    gold_ret: float
    gold_alloc: float
    re_ret: float
    re_alloc: float

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "online", "model": "Monte Carlo FIRE Engine v2.0"}

@app.post("/simulate")
async def simulate_fire(data: FireRequest):
    try:
        # 1. Run the Multi-Asset Simulation
        # The engine returns a matrix of [sims, years]
        yearly_results = engine.run_simulation(
            current_age=data.current_age,
            initial_corpus=data.initial_corpus,
            monthly_invest=data.monthly_invest,
            annual_expense=data.annual_expense,
            inflation_rate=data.inflation_rate,
            years=data.years,
            equity_ret=data.equity_ret,
            equity_alloc=data.equity_alloc,
            debt_ret=data.debt_ret,
            debt_alloc=data.debt_alloc,
            gold_ret=data.gold_ret,
            gold_alloc=data.gold_alloc,
            re_ret=data.re_ret,
            re_alloc=data.re_alloc
        )

        # 2. Calculate Statistics
        # Success = Wealth > 0 at the final year of simulation
        final_wealths = yearly_results[:, -1]
        success_count = np.sum(final_wealths > 0)
        success_probability = (success_count / len(final_wealths)) * 100
        
        median_final_wealth = np.median(final_wealths)

        # 3. Prepare Trajectories for Visualization
        # We send back 10 random paths to show variability in the UI
        sample_indices = np.random.choice(range(len(yearly_results)), size=10, replace=False)
        sample_paths = yearly_results[sample_indices, :].tolist()

        return {
            "success_probability": round(float(success_probability), 2),
            "median_final_wealth": round(float(median_final_wealth), 2),
            "trajectories": sample_paths
        }

    except Exception as e:
        print(f"Error during simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))