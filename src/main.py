from data_fetcher import fetch_and_analyze_market
from fire_engine import FireEngine
from generate_dataset import generate_synthetic_data

def main():
    #fetch_and_analyze_market()
    #engine = FireEngine()
    # User has 10L, invests 50k/mo, spends 6L/year
    #sim_data = engine.run_simulation(1000000, 50000, 600000)
    #print(f"Simulation Matrix Shape: {sim_data.shape}")
    #engine.save()
    generate_synthetic_data()

if __name__ == "__main__":
    main()
