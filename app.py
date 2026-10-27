import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/model.pkl")

st.title("Customer Churn Predictor")
st.write("Enter customer details to estimate the chance they will leave.")

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months with company)", 0, 72, 12)
    monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0)
    senior = st.selectbox("Senior citizen?", [0, 1],
                          format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Has partner?", ["Yes", "No"])
    dependents = st.selectbox("Has dependents?", ["Yes", "No"])
    paperless = st.selectbox("Paperless billing?", ["Yes", "No"])
with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
    tech = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])

if st.button("Predict"):
    row = pd.DataFrame([{
        "tenure": tenure, "MonthlyCharges": monthly, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents, "Contract": contract,
        "InternetService": internet, "OnlineSecurity": security,
        "TechSupport": tech, "PaperlessBilling": paperless,
        "PaymentMethod": payment,
    }])
    prob = model.predict_proba(row)[0][1]
    if prob >= 0.5:
        st.error(f"High churn risk: {prob:.0%}")
        st.info("Suggestion: offer a discount or a longer-term contract.")
    else:
        st.success(f"Low churn risk: {prob:.0%}")