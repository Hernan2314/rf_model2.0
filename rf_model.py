import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    with open('rf_modelp.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_model_and_scaler()

# Function to make the prediction
def make_prediction(input_data):
    input_data_scaled = scaler.transform([input_data])  # Scale input data
    prediction = model.predict(input_data_scaled)       # Predict with model
    return "Fraud" if prediction[0] == 1 else "Not Fraud"

# Title and description
st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to predict the likelihood of fraud.")

# Create input fields for the transaction features
time = st.number_input("Transaction Time (in seconds)", min_value=0, step=1)
amount = st.number_input("Transaction Amount (in dollars)", min_value=0.0, step=0.01)

# Create sliders for additional features (assuming they are named V1 to V28)
v_features = []
for i in range(1, 29):  # V1 to V28
    v_feature = st.slider(f"V{i}", min_value=-30.0, max_value=30.0, step=0.1)  # Adjust ranges as necessary
    v_features.append(v_feature)

# Combine all input data into a single list
input_data = [time, amount] + v_features

# Prediction button
if st.button("Predict"):
    result = make_prediction(input_data)
    st.success(f'The prediction is: {result}')
