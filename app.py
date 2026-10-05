import os
import joblib
import pandas as pd
import streamlit as st

from src.feature_extraction import extract_features

st.set_page_config(page_title="Phishing & Fraudulent Website Detector", layout="centered")

st.title("🛡️ Fraudulent Website Detection")
st.markdown(
    "Analyze web domains in real time using structural URL feature extraction "
    "and a trained machine learning classification model."
)

MODEL_PATH = os.path.join("models", "rf_phishing_model.joblib")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

if model is None:
    st.error("Model file not found. Run `python train.py` first to generate the model artifact.")
else:
    url_input = st.text_input(
        "Enter URL to inspect:",
        placeholder="e.g., [http://secure-login-update.account-verification.xyz](http://secure-login-update.account-verification.xyz)",
    )

    if st.button("Scan URL", type="primary"):
        if not url_input.strip():
            st.warning("Please enter a valid URL.")
        else:
            features = extract_features(url_input)
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]

            st.divider()

            if prediction == 1:
                st.error(f"⚠️ **High Risk Detected (Phishing / Fraudulent Website)**")
                st.metric("Fraud Probability Score", f"{probability * 100:.1f}%")
            else:
                st.success(f"✅ **Low Risk (Likely Legitimate Website)**")
                st.metric("Safety Confidence", f"{(1 - probability) * 100:.1f}%")

            st.subheader("Extracted Feature Analysis")
            feature_display = pd.DataFrame(
                {
                    "Feature": features.columns,
                    "Extracted Value": features.iloc[0].values,
                }
            )
            st.dataframe(feature_display, use_container_width=True)