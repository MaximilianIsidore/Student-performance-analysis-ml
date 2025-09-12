import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    reg = pickle.load(f)

# Min and Max values from your dataset
min_values = {
    "Hours Studied": 1,
    "Previous Scores": 40,
    "Extracurricular Activities": 0,   # stays 0/1
    "Sleep Hours": 4,
    "Sample Question Papers Practiced": 0
}

max_values = {
    "Hours Studied": 9,
    "Previous Scores": 99,
    "Extracurricular Activities": 1,   # stays 0/1
    "Sleep Hours": 9,
    "Sample Question Papers Practiced": 9
}

# Normalization function
def min_max_normalize(x, feature, min_vals, max_vals):
    return (x - min_vals[feature]) / (max_vals[feature] - min_vals[feature])

# Streamlit UI
st.title("📘 Student Performance Predictor")

hours_studied = st.number_input("Hours Studied", min_value=1, max_value=9, value=5)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"])
extracurricular = 1 if extracurricular == "Yes" else 0
sleep_hours = st.slider("Sleep Hours", min_value=4, max_value=9, value=7)
sample_papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=9, value=5)

if st.button("Predict Performance"):
    # Prepare input
    X_normalized = np.array([[
        min_max_normalize(hours_studied, "Hours Studied", min_values, max_values),
        min_max_normalize(previous_scores, "Previous Scores", min_values, max_values),
        extracurricular,
        min_max_normalize(sleep_hours, "Sleep Hours", min_values, max_values),
        min_max_normalize(sample_papers, "Sample Question Papers Practiced", min_values, max_values)
    ]])

    # Prediction
    prediction = reg.predict(X_normalized)
    st.success(f"🎯 Predicted Performance Index: {prediction[0]:.2f}")
