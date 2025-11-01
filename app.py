import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="ML Predictor",
    page_icon="🤖",
    layout="centered",
)

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #F9FAFB;
        color: #111827;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("IRIS CLASSIFIER")
st.markdown("### Enter the Flower Features")

# Input fields in columns
col1, col2 = st.columns(2)
with col1:
    f1 = st.number_input("Sepal length in cms", value=0.0)
    f2 = st.number_input("Sepal width in cms", value=0.0)
with col2:
   f3 = st.number_input("Petal length in cms", value=0.0)
    f4 = st.number_input("Petal width in cms", value=0.0)

 Prediction button

if st.button(" Predict"):
    features = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(features)
    st.success(f" Predicted Class: {prediction[0]}")


