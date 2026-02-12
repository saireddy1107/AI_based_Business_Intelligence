import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# Load Pickle Files
# -------------------------------
model = pickle.load(open("sales_prediction_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI-Based Business Intelligence System",
    layout="centered"
)

# -------------------------------
# Title & Description
# -------------------------------
st.title("📊 AI-Based Business Intelligence System")
st.markdown("""
This system uses **Machine Learning and Data Analytics** to predict  
**E-commerce Sales (Units Sold)** and support intelligent business decisions.
""")

st.divider()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🔍 Input Business Data")

price = st.sidebar.number_input("Product Price", min_value=1.0, value=500.0)
discount = st.sidebar.slider("Discount (%)", 0, 90, 10)
marketing_spend = st.sidebar.number_input(
    "Marketing Spend", min_value=0.0, value=2000.0
)

product_category = st.sidebar.selectbox(
    "Product Category",
    ["Electronics", "Clothing", "Home", "Beauty", "Sports"]
)

customer_segment = st.sidebar.selectbox(
    "Customer Segment",
    ["Regular", "Premium", "New"]
)

# -------------------------------
# Encode Inputs
# -------------------------------
product_category_encoded = encoder.fit_transform([product_category])[0]
customer_segment_encoded = encoder.fit_transform([customer_segment])[0]

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("📈 Predict Units Sold"):

    input_data = np.array([[
        price,
        discount,
        marketing_spend,
        product_category_encoded,
        customer_segment_encoded
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"✅ Predicted Units Sold: **{int(prediction[0])}**")

# -------------------------------
# Optional Dataset Preview
# -------------------------------
st.divider()
st.subheader("📂 Dataset Preview")

try:
    df = pd.read_csv("Ecommerce_Sales_Prediction_Dataset.csv")
    st.dataframe(df.head())
except:
    st.info("Dataset file not found (optional)")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "🎓 **AI-Based Business Intelligence System using Machine Learning & Data Science**"
)
