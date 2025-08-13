# preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_and_save_data():
    """
    Loads the dataset, preprocesses it by dropping irrelevant features, 
    engineering a new feature, splitting the data, and scaling the numerical columns.
    Saves the final training and testing data, the scaler, and feature names.
    """
    print("--- Starting Data Preprocessing and Feature Engineering ---")
    try:
        # Step 1: Load the dataset
        df = pd.read_csv('kc_house_data.csv')
        print("  - Dataset loaded successfully.")

        # Step 2: Drop irrelevant features
        # We drop columns that are not useful for our model or that cause data leakage.
        irrelevant_features = ['PropertyID', 'DocumentDate', 'ym', 'zhvi_px', 'zhvi_idx', 'AdjSalePrice', 'NewConstruction']
        df.drop(columns=irrelevant_features, inplace=True)
        print("  - Irrelevant features dropped.")
        
        # Step 3: Feature Engineering - Create a new 'HouseAge' feature
        # We'll calculate the age of the house from the year it was built.
        current_year = 2025 # A consistent reference year for our calculation
        df['HouseAge'] = current_year - df['YrBuilt']
        df.drop(columns=['YrBuilt'], inplace=True)
        print("  - Engineered 'HouseAge' feature and dropped 'YrBuilt'.")
        
        # Step 4: Handle categorical features with one-hot encoding
        categorical_cols = ['PropertyType']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        print("  - Categorical features have been one-hot encoded.")
        print("  - Columns after encoding:", df.columns.tolist())

        # Step 5: Define features (X) and target (y)
        X = df.drop(columns=['SalePrice'])
        y = df['SalePrice']
        
        # Get the final list of feature names in the correct order
        feature_names = X.columns.tolist()
        print("\n  - Features (X) and target (y) defined.")
        print("  - Final feature names:", feature_names)

        # Step 6: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"\n  - Data split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")

        # Step 7: Feature Scaling
        # We only fit the scaler on the training data to avoid data leakage
        numerical_cols_to_scale = X_train.select_dtypes(include=np.number).columns.tolist()
        
        # We will create and save a single scaler for all numerical columns
        scaler = StandardScaler()
        X_train[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
        X_test[numerical_cols_to_scale] = scaler.transform(X_test[numerical_cols_to_scale])
        print("  - Numerical features scaled using StandardScaler.")

        # Step 8: Save the processed data, scaler, and feature names
        # We save the data for model training and the scaler/feature names for the app.
        X_train.to_csv('X_train.csv', index=False)
        X_test.to_csv('X_test.csv', index=False)
        y_train.to_csv('y_train.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(feature_names, 'feature_names.pkl')
        joblib.dump(numerical_cols_to_scale, 'numerical_cols_to_scale.pkl')
        print("\n  - All processed files saved successfully.")
        
    except FileNotFoundError:
        print("\nFATAL ERROR: 'kc_house_data.csv' not found. Please ensure the dataset file is in the same directory.")
    except Exception as e:
        print(f"\nFATAL ERROR: An unexpected error occurred: {e}")

if __name__ == '__main__':
    preprocess_and_save_data()
