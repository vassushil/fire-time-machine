import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

def train_fire_model():
    # 1. Load the synthetic data we generated
    data_path = 'data/synthetic_fire_data.csv'
    if not os.path.exists(data_path):
        print("Error: Dataset not found. Run generate_dataset.py first.")
        return

    df = pd.read_csv(data_path)

    # 2. Define Features (X) and Target (y)
    # We include the bucket returns and macro params
    features = [
        "current_age", "initial_corpus", "monthly_invest", 
        "annual_expense", "inflation_rate", "equity_return_expected"
    ]
    X = df[features]
    y = df['final_wealth']

    # 3. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training XGBoost model on {len(X_train)} samples...")

    # 4. Initialize and Train XGBoost Regressor
    # Using 'hist' tree method for faster training on larger datasets
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    # 5. Evaluate the Model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model Evaluation:")
    print(f"- Mean Absolute Error: ₹{mae:,.2f}")
    print(f"- R² Score: {r2:.4f} (Higher is better)")

    # 6. Save the Model using Pickle
    os.makedirs('models', exist_ok=True)
    with open('models/fire_xgboost.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("XGBoost model saved to models/fire_xgboost.pkl")

    # 7. Optional: Print Feature Importance
    
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature Importance (What drives wealth most?):")
    print(importances)


train_fire_model()
