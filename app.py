# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Custom CSS for a professional look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f2f6;
        color: #333;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-18ni7ap {
        background-color: transparent;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #0c1c30;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 400;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stForm {
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        background-color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        color: white !important;
        background-color: #2c3e50;
        border: none;
        padding: 0.75rem 0;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #34495e;
        transform: translateY(-2px);
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        background-color: #e8f8f5;
        color: #1a5276;
        border: 2px solid #5cb85c;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        color: #a0a0a0;
        font-size: 0.9rem;
    }
    .streamlit-expanderHeader {
        font-size: 1.25rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- Set up the Streamlit App UI ---
st.title("King County House Price Predictor", anchor=False)
st.markdown("<p class='sub-header'>Predict a house's value using machine learning.</p>", unsafe_allow_html=True)

# --- Function to load the saved artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('best_house_price_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        numerical_cols_to_scale = joblib.load('numerical_cols_to_scale.pkl')
        st.success("Model, scaler, and feature names loaded successfully!")
        return model, scaler, feature_names, numerical_cols_to_scale
    except FileNotFoundError:
        st.error("Error: One or more model files not found. Please run `preprocess.py` and `train_model.py` first.")
        st.stop()

# Load the model and scalers
model, scaler, feature_names, numerical_cols_to_scale = load_artifacts()

# --- User Input Section ---
st.subheader("Enter House Details", anchor=False)

with st.form(key='prediction_form'):
    # Create input widgets for the features in a two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        bathrooms = st.slider("Bathrooms", min_value=0.0, max_value=8.0, value=2.25, step=0.25)
        bedrooms = st.slider("Bedrooms", min_value=0, max_value=10, value=3, step=1)
        sqft_living = st.number_input("SqFtTotLiving (Living Area)", min_value=370, max_value=13540, value=2000, step=10)
        sqft_lot = st.number_input("SqFtLot (Lot Area)", min_value=520, max_value=1651359, value=15000, step=100)
        grade = st.slider("BldgGrade (Quality of Construction)", min_value=1, max_value=13, value=7, step=1)
        
    with col2:
        zipcode = st.number_input("ZipCode", min_value=98001, max_value=98199, value=98001, step=1)
        traffic_noise = st.slider("TrafficNoise Level", min_value=0, max_value=3, value=0, step=1)
        house_age = st.number_input("House Age", min_value=0, max_value=119, value=25, step=1)
        sqft_basement = st.number_input("SqFtFinBasement (Finished Basement Area)", min_value=0, max_value=4820, value=0, step=10)
        
        # We handle the categorical feature separately
        property_type_list = ['Single Family', 'Townhouse', 'Multiplex']
        property_type = st.radio("Property Type", property_type_list, index=0)
        
    st.markdown("---")
    submit_button = st.form_submit_button(label='Predict Price')

if submit_button:
    # Preprocess the user input to match the model's training data
    input_data = pd.DataFrame(columns=feature_names)
    
    # Populate the DataFrame with user inputs
    input_data['Bathrooms'] = [bathrooms]
    input_data['Bedrooms'] = [bedrooms]
    input_data['SqFtTotLiving'] = [sqft_living]
    input_data['SqFtLot'] = [sqft_lot]
    input_data['BldgGrade'] = [grade]
    input_data['ZipCode'] = [zipcode]
    input_data['TrafficNoise'] = [traffic_noise]
    input_data['HouseAge'] = [house_age]
    input_data['SqFtFinBasement'] = [sqft_basement]

    # Handle the one-hot encoded features
    input_data['PropertyType_Single Family'] = [1 if property_type == 'Single Family' else 0]
    input_data['PropertyType_Townhouse'] = [1 if property_type == 'Townhouse' else 0]
    input_data['PropertyType_Multiplex'] = [1 if property_type == 'Multiplex' else 0]

    # Fill in the rest of the features with placeholder values if not used in the form
    input_data['NbrLivingUnits'] = [1]
    input_data['YrRenovated'] = [0]
    input_data['LandVal'] = [220321]
    input_data['ImpsVal'] = [300471]
    
    # Ensure the columns are in the exact same order as the training data
    input_data = input_data[feature_names]

    # Scale only the numerical features that were scaled during training
    input_data[numerical_cols_to_scale] = scaler.transform(input_data[numerical_cols_to_scale])
    
    # Make the prediction
    predicted_price = model.predict(input_data)
    
    st.markdown("---")
    st.subheader("Predicted House Price", anchor=False)
    st.markdown(f"**Based on the features you entered, the estimated house price is:**")
    st.success(f"**${predicted_price[0]:,.2f}**")

# --- Footer ---
st.markdown("<div class='footer'>Developed by Devnaam Priyadershi</div>", unsafe_allow_html=True)
