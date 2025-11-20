# app.py — Streamlit app for your IDS ensemble model
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# ---------------------------
# Helper: Preprocessing utils
# ---------------------------
@st.cache_data(show_spinner=False)
def load_raw_dataset(path="datasets/UNSW_NB15.csv", nrows=None):
    """Load original dataset to derive encoders & scaler (used to preprocess incoming data)."""
    df = pd.read_csv(path, nrows=nrows)
    # replace service '-' with NaN and drop missing (same as training pipeline)
    if "service" in df.columns:
        df["service"] = df["service"].replace("-", np.nan)
    df.dropna(inplace=True)
    return df

@st.cache_data(show_spinner=False)
def fit_preprocessors(df):
    """Fit LabelEncoders for categorical columns and a MinMaxScaler for numeric columns."""
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)  # ensure string
        le.fit(df[col])
        encoders[col] = le

    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    # remove target 'label' if present
    if "label" in numeric_cols:
        numeric_cols.remove("label")

    scaler = MinMaxScaler()
    scaler.fit(df[numeric_cols])

    return encoders, scaler, categorical_cols, numeric_cols

def preprocess_input(df_in, encoders, scaler, categorical_cols, numeric_cols):
    """Apply encoders & scaler to new dataframe. Return processed df ready for model."""
    df = df_in.copy()

    # Ensure all columns exist
    # For categorical cols: if missing in df, create col with default 'undefined'
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "undefined"
        df[col] = df[col].astype(str)
        # map unseen labels to a placeholder if possible (LabelEncoder will raise if unseen);
        # we will map unseen labels to the most common label in encoder.classes_
        le = encoders[col]
        mapped = []
        for v in df[col].values:
            if v in le.classes_:
                mapped.append(v)
            else:
                mapped.append(le.classes_[0])  # fallback to first seen class
        df[col] = le.transform(mapped)

    # numeric cols: if missing, fill with 0
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0

    # scale numeric columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # remove any extra columns that model wasn't trained on (we will assume model trained on full dataset columns w/o 'label')
    # The model expects the same order of features as training X; we'll let that be handled by user uploading same columns.
    return df

# ---------------------------
# Load / prepare model + preprocessors
# ---------------------------
st.set_page_config(page_title="IDS Ensemble Demo", layout="wide")

st.title("Network Intrusion Detection — Ensemble IDS (UNSW-NB15)")
st.markdown(
    "Demo app for your project. Upload network records (CSV) or pick a sample row and run inference with the saved ensemble."
)

# Load raw dataset to derive preprocessing mappings
DATA_PATH = "datasets/UNSW_NB15.csv"
MODEL_PATH = "models/ensemble_ids.pkl"

if not Path(DATA_PATH).exists():
    st.error(f"Dataset not found at `{DATA_PATH}`. Place the UNSW_NB15 CSV in this path.")
    st.stop()

raw_df = load_raw_dataset(DATA_PATH)

# Fit encoders and scaler (cached)
encoders, scaler, categorical_cols, numeric_cols = fit_preprocessors(raw_df)

# Load model
if not Path(MODEL_PATH).exists():
    st.warning(f"Trained model not found at `{MODEL_PATH}`. Make sure you saved the ensemble there.")
    model = None
else:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Sidebar options
st.sidebar.header("Options")
mode = st.sidebar.radio("Select mode", ("Sample from dataset", "Upload CSV for batch inference"))

# ---------------------------
# Mode 1: Sample from dataset
# ---------------------------
if mode == "Sample from dataset":
    st.subheader("Pick a sample row from the original dataset")
    sample_idx = st.slider("Row index (from cleaned dataset)", 0, int(len(raw_df) - 1), 0)
    sample_row = raw_df.iloc[[sample_idx]].copy()  # keep as DataFrame
    st.write("Original row (raw values):")
    st.dataframe(sample_row.T, height=250)

    st.markdown("**Preprocessed features used for prediction**")
    X_sample = preprocess_input(sample_row.drop(columns=["label"], errors="ignore"), encoders, scaler, categorical_cols, numeric_cols)
    st.dataframe(X_sample.T, height=200)

    if st.button("Predict sample"):
        if model is None:
            st.error("Model not loaded — cannot predict.")
        else:
            pred = model.predict(X_sample)[0]
            label_str = "Normal (0)" if int(pred) == 0 else "Attack (1)"
            st.success(f"Prediction: **{label_str}**")
            # If you have saved label encodings for multi-class you can show attack_cat mapping; here it's binary.

# ---------------------------
# Mode 2: CSV upload + batch inference
# ---------------------------
else:
    st.subheader("Upload a CSV file for batch prediction")
    st.markdown(
        "CSV should contain the same feature columns as the UNSW dataset (excluding 'label' if unknown). "
        "If a column is missing, the app will attempt a safe fallback."
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_user = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error("Failed to read CSV — make sure the file is valid.")
            st.stop()

        st.write("Preview of uploaded data")
        st.dataframe(df_user.head())

        # Preprocess
        try:
            X_user = preprocess_input(df_user.drop(columns=["label"], errors="ignore"), encoders, scaler, categorical_cols, numeric_cols)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

        st.write("Data after preprocessing (first rows):")
        st.dataframe(X_user.head())

        if st.button("Run predictions on uploaded CSV"):
            if model is None:
                st.error("Model not loaded — cannot predict.")
            else:
                preds = model.predict(X_user)
                df_out = df_user.copy()
                df_out["prediction_label"] = preds
                df_out["prediction_text"] = df_out["prediction_label"].apply(lambda x: "Normal" if int(x) == 0 else "Attack")

                st.success("Predictions completed.")
                st.write("Prediction sample:")
                st.dataframe(df_out.head())

                # Summary counts
                st.markdown("### Prediction counts")
                st.write(df_out["prediction_text"].value_counts())

                # Classification report if 'label' column present in upload
                if "label" in df_user.columns:
                    st.markdown("### Metrics (since ground-truth 'label' was provided in the CSV)")
                    report = classification_report(df_user["label"].astype(int), preds, output_dict=True)
                    st.json(report)

                # Provide download button
                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

# ---------------------------
# Footer / notes
# ---------------------------
st.markdown("---")
st.markdown(
    "Notes:\n"
    "- This demo fits LabelEncoders and MinMaxScaler on the ORIGINAL dataset at startup so uploaded rows are preprocessed the same way as training.\n"
    "- For production, save preprocessors (encoders + scaler) during training and load them here to ensure exact mappings."
)
