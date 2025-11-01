import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# App title
st.title("ML Prediction App")

# Example: suppose your model takes 4 input features
st.header("Enter Input Features")
f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)

# Predict button
if st.button("Predict"):
    features = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(features)
    st.success(f"Prediction: {prediction[0]}")

