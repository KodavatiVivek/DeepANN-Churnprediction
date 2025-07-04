import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# --- Load all necessary files ---

# Load Keras model
model = load_model('churn_model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the one-hot encoder
with open('ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)

# Load the label encoder for Gender
with open('label_encoder.pkl', 'rb') as f:
    label = pickle.load(f)

# --- Streamlit UI ---
st.title("🧠 Customer Churn Prediction App")
st.write("Enter the customer details below to predict churn probability:")

# User input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
geography = st.selectbox("Geography", [o.split('_')[1] for o in ohe.get_feature_names_out(['Geography'])])
gender = st.selectbox("Gender", label.classes_.tolist())
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years with bank)", 0, 10, 5)
balance = st.number_input("Account Balance", min_value=0.0, value=90000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# When Predict button is clicked
if st.button("Predict"):

    # --- Step 1: Create input DataFrame ---
    input_data = pd.DataFrame([{
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': salary
    }])

    # --- Step 2: Encode Categorical Features ---
    # One-hot encode Geography
    geo = ohe.transform(input_data[['Geography']]).toarray()
    geo_df = pd.DataFrame(geo, columns=ohe.get_feature_names_out(['Geography']))
    # Concatenate the one-hot encoded Geography with the input data
    input_data = pd.concat([input_data, geo_df], axis=1)
    input_data.drop(columns=['Geography'], inplace=True)

    # Label encode Gender
    input_data['Gender'] = label.transform(input_data['Gender'])

    # --- Step 3: Scale the data ---
    input_scaled = scaler.transform(input_data)

    # --- Step 4: Make prediction ---
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]

    # --- Step 5: Display result ---
    st.markdown("---")
    if prediction_proba > 0.5:
        st.error(f"🚨 The customer is **likely to churn**.\n\n**Probability:** {prediction_proba:.2%}")
    else:
        st.success(f"✅ The customer is **not likely to churn**.\n\n**Probability:** {prediction_proba:.2%}")