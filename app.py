import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Naive Bayes Project")

st.title("📘 Naive Bayes Classification (Browse Dataset)")

# -----------------------------------
# Upload Dataset
# -----------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------
    # Select Target Column
    # -----------------------------------
    target_col = st.selectbox("Select Target Column", df.columns)

    # -----------------------------------
    # Handle Missing Values
    # -----------------------------------
    df = df.dropna()

    # -----------------------------------
    # Encode Categorical Columns
    # -----------------------------------
    df_encoded = df.copy()
    encoders = {}

    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le

    # -----------------------------------
    # Split Features & Target
    # -----------------------------------
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    # -----------------------------------
    # Train-Test Split
    # -----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------------
    # Train Naive Bayes
    # -----------------------------------
    model = GaussianNB()
    model.fit(X_train, y_train)

    # -----------------------------------
    # Prediction
    # -----------------------------------
    y_pred = model.predict(X_test)

    # -----------------------------------
    # Results
    # -----------------------------------
    st.subheader("Model Performance")
    st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.4f}**")

    st.subheader("Sample Predictions")
    result_df = X_test.copy()
    result_df["Actual"] = y_test.values
    result_df["Predicted"] = y_pred

    st.dataframe(result_df.head())
