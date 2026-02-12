import streamlit as st
import pandas as pd
import joblib
import io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shap

# Load trained model
model = joblib.load("xgb_fraud_model.pkl")

st.title("Credit Card Fraud Detection System with SHAP & Confusion Matrix")
st.write("You can either upload a CSV file or paste input data manually, then click Predict.")

# Option 1: Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# Option 2: Paste input manually
manual_input = st.text_area(
    "Or paste your input data here (comma separated, include header):",
    height=150
)

# Predict button
if st.button("Predict"):

    # Load data
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    elif manual_input:
        try:
            data = pd.read_csv(io.StringIO(manual_input))
        except Exception as e:
            st.error(f"Error reading input: {e}")
            st.stop()
    else:
        st.error("Please upload a CSV or paste input data.")
        st.stop()

    # Drop target column if exists
    if "Class" in data.columns:
        target_present = True
        y_true = data["Class"]
        data = data.drop("Class", axis=1)
    else:
        target_present = False

    # Check columns match model
    expected_features = model.get_booster().feature_names
    if list(data.columns) != expected_features:
        st.error("Feature columns do not match the model. Please check input columns.")
        st.write("Expected columns:", expected_features)
        st.stop()

    # Make predictions
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]
    result_labels = ["Not Fraud" if p == 0 else "Fraud" for p in predictions]

    data["Prediction"] = result_labels
    data["Fraud Probability"] = probabilities

    st.subheader("Prediction Results")
    st.write(data)

    # Confusion Matrix (if target exists)
    if target_present:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, predictions)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
        disp.plot(ax=ax)
        st.pyplot(fig)

    # SHAP Summary Plot
    st.subheader("SHAP Feature Importance")

    # Use only feature columns for SHAP
    feature_data = data[expected_features]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_data)

    # Plot summary
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_data, plot_type="bar", show=False)
    st.pyplot(fig2)
