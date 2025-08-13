# 🏠 King County House Price Prediction

This is a final year B.Tech data science project that demonstrates a complete machine learning pipeline for predicting house prices in King County, USA. The project uses a regression model to estimate house values based on a variety of features.

## 🚀 Project Overview

The goal of this project is to build a predictive model that can estimate the `SalePrice` of a house. The project implements a full data science workflow:

1. **Exploratory Data Analysis (EDA)**: Investigating the dataset to understand data distributions and feature relationships.

2. **Data Preprocessing**: Cleaning and transforming the raw data, including feature engineering and scaling.

3. **Model Training**: Training and evaluating several regression models (Linear Regression, Decision Tree, Random Forest).

4. **Deployment**: Creating an interactive web application using Streamlit to showcase the model's predictions.

<img width="1280" height="768" alt="image" src="https://github.com/user-attachments/assets/db3c2ba0-972f-4710-9178-1bcfbce4a9d7" />

<img width="1280" height="768" alt="image" src="https://github.com/user-attachments/assets/659da9ff-f698-419e-b955-92dca7bfb765" />



## 📁 File Structure

The project is organized into the following files:

* `app.py`: The Streamlit web application that serves as the user interface for making predictions.

* `preprocess.py`: A Python script to perform all data cleaning, feature engineering, and scaling. It saves the processed data and the `StandardScaler` to disk.

* `train_model.py`: A Python script to load the preprocessed data, train the machine learning model, and save the best-performing model to disk.

* `kc_house_data.csv`: The dataset used for training and evaluation. This file must be placed in the same directory.

* `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`: The preprocessed and split data files.

* `best_house_price_model.pkl`: The saved machine learning model.

* `scaler.pkl`: The saved `StandardScaler` used to transform numerical features.

* `feature_names.pkl`: A file containing the names of the features in the correct order.

* `numerical_cols_to_scale.pkl`: A list of the numerical columns that were scaled.

## ⚙️ Installation and Setup

### Prerequisites

Make sure you have Python installed. The project requires the following libraries. You can install them using `pip`:

### Running the Project

1. **Download the Dataset**: Download the `kc_house_data.csv` file from [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) and place it in the project directory.

2. **Preprocess the Data**: Open your terminal or command prompt, navigate to the project directory, and run the preprocessing script. This will generate the necessary data and scaler files.


3. **Train the Model**: After preprocessing is complete, run the training script. This will create and save the final model file.

4. **Launch the App**: Once the model is trained, start the Streamlit application.


Your web browser should automatically open the app at `http://localhost:8501`.

## 💡 How to Use the App

The Streamlit app provides a simple form to input house details. You can:

* Adjust features like **Bathrooms**, **Bedrooms**, **Living Area**, and **Lot Area** using sliders and number inputs.

* Select the **Property Type**.

* Click the **Predict Price** button to get an estimated house price.

## 👩‍💻 Future Improvements

* **Model Comparison**: Experiment with more advanced models like Gradient Boosting or Neural Networks.

* **Hyperparameter Tuning**: Use `GridSearchCV` or `RandomizedSearchCV` to fine-tune the best model for even better performance.

* **Deployment**: Deploy the Streamlit app to a cloud platform like Streamlit Cloud for public access.

* **Feature Importance**: Analyze and visualize which features have the greatest impact on the predicted price.

## 🤝 Contribution

Feel free to fork the repository and contribute to this project. If you find a bug or have a suggestion, please open an issue.

---

*Developed by Devnaam Priyadershi*
