import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Encoding maps (used during training)
workclass_map = {
    'Federal-gov': 0, 'Local-gov': 1, 'Private': 2, 'Self-emp-inc': 3,
    'Self-emp-not-inc': 4, 'State-gov': 5, 'Without-pay': 6, 'Notlisted': 7
}
marital_status_map = {
    'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2,
    'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6
}
occupation_map = {
    'Adm-clerical': 0, 'Craft-repair': 1, 'Exec-managerial': 2, 'Farming-fishing': 3,
    'Handlers-cleaners': 4, 'Machine-op-inspct': 5, 'Other-service': 6, 'Priv-house-serv': 7,
    'Prof-specialty': 8, 'Protective-serv': 9, 'Sales': 10, 'Tech-support': 11,
    'Transport-moving': 12, 'Armed-Forces': 13, 'Others': 14
}
relationship_map = {'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2, 'Own-child': 3, 'Unmarried': 4, 'Wife': 5}
race_map = {'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3, 'White': 4}
gender_map = {'Male': 1, 'Female': 0}
country_map = {'United-States': 38, 'India': 17, 'Philippines': 27, 'Germany': 10, 'Others': 1}

# Streamlit Page Config
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼")

# Title
st.title("💼 Employee Salary Prediction App")
st.markdown("This app predicts whether an employee earns `>50K` or `<=50K` using demographic and job-related info.")

# Sidebar input
st.sidebar.header("📋 Input Employee Details")

age = st.sidebar.slider("Age", 18, 75, 30)
workclass = st.sidebar.selectbox("Workclass", list(workclass_map.keys()))
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=1000, max_value=1000000, value=50000)
educational_num = st.sidebar.slider("Education Number (1-16)", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
occupation = st.sidebar.selectbox("Occupation", list(occupation_map.keys()))
relationship = st.sidebar.selectbox("Relationship", list(relationship_map.keys()))
race = st.sidebar.selectbox("Race", list(race_map.keys()))
gender = st.sidebar.radio("Gender", list(gender_map.keys()))
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
native_country = st.sidebar.selectbox("Native Country", list(country_map.keys()))

# Prediction on form input
if st.button("🔍 Predict Salary"):
    input_data = np.array([[
        age,
        workclass_map[workclass],
        fnlwgt,
        educational_num,
        marital_status_map[marital_status],
        occupation_map[occupation],
        relationship_map[relationship],
        race_map[race],
        gender_map[gender],
        capital_gain,
        capital_loss,
        hours_per_week,
        country_map[native_country]
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = ">50K" if prediction == 1 else "<=50K"

    st.success(f"✅ Predicted Salary Category: **{result}**")

# --- Batch CSV Prediction ---
st.markdown("---")
st.subheader("📂 Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload a CSV file with matching columns", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("🔎 Preview of Uploaded Data:")
    st.dataframe(df.head())

    try:
        scaled = scaler.transform(df)
        preds = model.predict(scaled)
        df["Predicted Salary"] = [">50K" if p == 1 else "<=50K" for p in preds]

        st.write("📈 Results:")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results CSV", csv, "predicted_salaries.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
