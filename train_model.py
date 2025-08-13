# train_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_and_evaluate_models():
    """
    Loads preprocessed data, trains and evaluates several regression models,
    and saves the best-performing model to disk.
    """
    print("--- Starting Model Training and Evaluation ---")
    try:
        # Step 1: Load the preprocessed data
        print("  - Loading preprocessed data...")
        X_train = pd.read_csv('X_train.csv')
        X_test = pd.read_csv('X_test.csv')
        y_train = pd.read_csv('y_train.csv')
        y_test = pd.read_csv('y_test.csv')
        print("  - Data loaded successfully.")

        # Step 2: Initialize the models
        print("\n  - Initializing models...")
        linear_reg_model = LinearRegression()
        decision_tree_model = DecisionTreeRegressor(random_state=42)
        random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

        models = {
            'Linear Regression': linear_reg_model,
            'Decision Tree': decision_tree_model,
            'Random Forest': random_forest_model
        }
        
        # Step 3: Train and evaluate each model
        for name, model in models.items():
            print(f"\n--- Training and Evaluating {name} ---")
            
            # Train the model
            model.fit(X_train, y_train.values.ravel())
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate the model
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  - Mean Absolute Error (MAE): {mae:,.2f}")
            print(f"  - Mean Squared Error (MSE): {mse:,.2f}")
            print(f"  - R-squared (R2) Score: {r2:.4f}")

        # Step 4: Save the best model (Random Forest is typically the best here)
        print("\n--- Saving the Best Model ---")
        joblib.dump(random_forest_model, 'best_house_price_model.pkl')
        print("  - Best model (Random Forest) saved as 'best_house_price_model.pkl'.")
        print("--- Process complete ---")

    except FileNotFoundError:
        print("\nFATAL ERROR: One or more data files not found. Please run `preprocess.py` first.")
    except Exception as e:
        print(f"\nFATAL ERROR: An unexpected error occurred: {e}")

if __name__ == '__main__':
    train_and_evaluate_models()
