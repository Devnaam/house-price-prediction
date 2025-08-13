# eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Acquisition ---
# Load the dataset
# Ensure 'kc_house_data.csv' is in the same directory as this script.
try:
    df = pd.read_csv('kc_house_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'kc_house_data.csv' not found. Please check the file path.")
    # Exit if the file is not found
    exit()

# --- Initial Data Inspection ---
print("\n--- First 5 rows of the dataset ---")
print(df.head())

print("\n--- DataFrame Information ---")
df.info()

print("\n--- Statistical Summary ---")
print(df.describe())

# --- Exploratory Data Analysis (EDA) ---

# 1. Visualize the Distribution of the Target Variable ('SalePrice')
print("\n--- Visualizing the Distribution of House Prices ---")
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of House Prices', fontsize=16)
plt.xlabel('Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# 2. Visualize Key Relationships: Living Space vs. Price
print("\n--- Visualizing the Relationship between Living Space and Price ---")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SqFtTotLiving', y='SalePrice', data=df, alpha=0.6, color='coral')
plt.title('Living Space (sqft) vs. Price', fontsize=16)
plt.xlabel('Square Footage of Living Area', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()

# 3. Visualize a Correlation Heatmap
print("\n--- Generating a Correlation Heatmap ---")
# We'll use the .select_dtypes() method to ensure we only include numerical columns
numerical_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 10))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Features', fontsize=18)
plt.show()

print("\n--- EDA Complete ---")
