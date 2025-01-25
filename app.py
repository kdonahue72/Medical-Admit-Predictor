import streamlit as st
import numpy as np
from xgboost import XGBClassifier
import pickle

# Load the trained model
# Save your trained model to a file if not already done
# Uncomment the following line to save the model:
# pickle.dump(final_model, open("xgb_model.pkl", "wb"))

# Load the model from file
model = pickle.load(open("xgb_model.pkl", "rb"))

# Define the app title
st.title("Medical Readmission Prediction")

# User inputs
st.sidebar.header("Enter Patient Details")
age = st.sidebar.slider("Age", 18, 90, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
chronic_conditions = st.sidebar.slider("Number of Chronic Conditions", 0, 10, 2)
medications = st.sidebar.slider("Number of Medications", 0, 20, 5)
past_hospitalizations = st.sidebar.slider("Number of Past Hospitalizations", 0, 5, 1)

# Preprocess user inputs
gender_encoded = 0 if gender == "Male" else 1
chronic_load = chronic_conditions * medications
bmi_age_interaction = bmi * age

# Combine inputs into an array
input_data = np.array([[age, gender_encoded, bmi, chronic_conditions, medications,
                        past_hospitalizations, chronic_load, bmi_age_interaction]])

# Make prediction
prediction = model.predict(input_data)[0]
prediction_prob = model.predict_proba(input_data)[0][1]

# Display results
st.header("Prediction Results")
if prediction == 1:
    st.subheader("This patient is likely to be readmitted.")
else:
    st.subheader("This patient is not likely to be readmitted.")

st.write(f"Probability of readmission: {prediction_prob:.2f}")
