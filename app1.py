import streamlit as st
import pandas as pd
import joblib

# -------------------------------------
# Page Configuration
# -------------------------------------
st.set_page_config(
    page_title="Bankruptcy Prevention Predictor",
    page_icon="🏢",
    layout="wide"
)

# -------------------------------------
# Load Model
# -------------------------------------
model = joblib.load("best_model.pkl")

# -------------------------------------
# Custom CSS
# -------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.big-title{
    font-size:40px;
    font-weight:bold;
    color:#1f3c88;
}
.subtitle{
    font-size:18px;
    color:gray;
}
.result{
    padding:20px;
    border-radius:12px;
    font-size:22px;
    font-weight:bold;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------
# Header
# -------------------------------------
st.markdown("<p class='big-title'>🏢 Bankruptcy Prevention Predictor</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict a company's bankruptcy risk based on six qualitative factors.</p>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------
# Sidebar
# -------------------------------------
st.sidebar.header("📋 Company Risk Profile")

st.sidebar.subheader("⚠ Risk Factors")

industrial_risk = st.sidebar.slider(
    "Industrial Risk",
    0.0,2.0,1.0,0.5
)

management_risk = st.sidebar.slider(
    "Management Risk",
    0.0,2.0,1.0,0.5
)

operating_risk = st.sidebar.slider(
    "Operating Risk",
    0.0,2.0,1.0,0.5
)

st.sidebar.subheader("💪 Strength Factors")

financial_flexibility = st.sidebar.slider(
    "Financial Flexibility",
    0.0,2.0,1.0,0.5
)

credibility = st.sidebar.slider(
    "Credibility",
    0.0,2.0,1.0,0.5
)

competitiveness = st.sidebar.slider(
    "Competitiveness",
    0.0,2.0,1.0,0.5
)

# -------------------------------------
# Input Data
# -------------------------------------
input_df = pd.DataFrame({
    "industrial_risk":[industrial_risk],
    "management_risk":[management_risk],
    "financial_flexibility":[financial_flexibility],
    "credibility":[credibility],
    "competitiveness":[competitiveness],
    "operating_risk":[operating_risk]
})

# -------------------------------------
# Display Input
# -------------------------------------
st.subheader("📋 Selected Company Profile")
st.dataframe(input_df, use_container_width=True)

st.markdown("---")

# -------------------------------------
# Prediction
# -------------------------------------
if st.button("🔍 Predict", use_container_width=True):

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    pred = str(prediction).lower()

    if pred in ["bankruptcy","bankrupt"]:
        st.error("⚠ Company is likely to become **Bankrupt**")

    elif pred in ["non-bankruptcy","non_bankruptcy","non-bankrupt","non bankrupt"]:
        st.success("✅ Company is **Non-Bankrupt**")

    else:
        st.info(f"Prediction : {prediction}")

    # ---------------------------------
    # Prediction Probability
    # ---------------------------------
    if hasattr(model,"predict_proba"):

        st.markdown("---")
        st.subheader("📊 Prediction Probability")

        probability = model.predict_proba(input_df)[0]

        for cls,prob in zip(model.classes_,probability):

            st.write(f"**{cls}**")

            st.progress(float(prob))

            st.write(f"{prob*100:.2f}%")

st.markdown("---")

st.caption("Developed using Streamlit | Bankruptcy Prevention Project")