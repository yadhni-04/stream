
import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit app UI
st.title("🚩 Fake Job Posting Predictor")
st.write("Enter job details below and click **Predict** to detect if it's a real or fake job posting.")

# Input fields
description = st.text_area("Job Description", height=200)
title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile", height=150)
requirements = st.text_area("Requirements", height=150)

# Combine inputs for prediction
full_text = f"{title} {company_profile} {description} {requirements}"

# Prediction button
if st.button("🔍 Predict"):
    if full_text.strip() == "":
        st.warning("Please enter some job-related information.")
    else:
        # Preprocess and predict
        X = vectorizer.transform([full_text])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        # Output
        st.subheader("🧠 Model Prediction:")
        st.write("**Fake Job Posting**" if prediction == 1 else "**Legitimate Job Posting**")
        st.write(f"**Confidence:** {probability:.2f}")
