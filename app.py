import streamlit as st
import joblib
import pandas as pd

# ---- Load saved objects (must be in the same folder as this file) ----
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')  # dict: {column_name: fitted LabelEncoder}

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn Predictor")
st.caption("Enter a customer's details to estimate their likelihood of churning.")

with st.form("churn_form"):

    st.subheader("Customer profile")
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Female", "Male"])
    senior_citizen = c2.selectbox("Senior citizen?", ["No", "Yes"])
    partner = c3.selectbox("Has partner?", ["Yes", "No"])
    dependents = c1.selectbox("Has dependents?", ["Yes", "No"])
    tenure = c2.slider("Tenure (months)", 0, 72, 12)
    contract = c3.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])

    st.subheader("Account & billing")
    b1, b2, b3 = st.columns(3)
    paperless_billing = b1.selectbox("Paperless billing?", ["Yes", "No"])
    payment_method = b2.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = b3.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=1.0)
    total_charges = st.number_input("Total charges ($)", 0.0, 10000.0, 1000.0, step=10.0)

    st.subheader("Services")
    s1, s2, s3 = st.columns(3)
    phone_service = s1.selectbox("Phone service?", ["Yes", "No"])
    multiple_lines = s2.selectbox("Multiple lines?", ["Yes", "No", "No phone service"])
    internet_service = s3.selectbox("Internet service", ["DSL", "Fiber optic", "No"])

    with st.expander("Add-on services (security, backup, streaming...)"):
        e1, e2, e3 = st.columns(3)
        online_security = e1.selectbox("Online security", ["Yes", "No", "No internet service"])
        online_backup = e2.selectbox("Online backup", ["Yes", "No", "No internet service"])
        device_protection = e3.selectbox("Device protection", ["Yes", "No", "No internet service"])
        tech_support = e1.selectbox("Tech support", ["Yes", "No", "No internet service"])
        streaming_tv = e2.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = e3.selectbox("Streaming movies", ["Yes", "No", "No internet service"])

    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    row = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    input_df = pd.DataFrame([row])

    # Apply the SAME label encoders used at training time to each categorical column.
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # This model's scaler was fit on ALL feature columns, not just the numeric ones.
    input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        if prediction == 1:
            st.metric("Prediction", "Churn risk")
        else:
            st.metric("Prediction", "Likely to stay")
    with r2:
        st.write("**Churn probability**")
        st.progress(min(max(probability, 0.0), 1.0))
        st.write(f"{probability:.0%}")

    if prediction == 1:
        st.error("This customer shows a high risk of churning. Consider proactive retention outreach.")
    else:
        st.success("This customer looks likely to stay, based on the current profile.")
