import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

import gdown
import os

# Function to download CSV from Google Drive
def download_csv_from_drive(file_id, destination_path="train_data.csv"):
    if not os.path.exists(destination_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination_path, quiet=False)

# Download CSV
download_csv_from_drive("1JjATBsMg1_vsM0pjPoo95p-vElIl_yqz")

from new_code_a import load_data, process_data, create_features, predict_sales
from new_code_a import stacked_model, scaler, features


# Load Model
@st.cache_resource
def load_model():
    model_script = "new_code_a.py"
    
    if not os.path.exists(model_script):
        st.error(f"Model script '{model_script}' not found!")
        return None, None
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", model_script)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    return model_module.scaler, model_module.stacked_model

scaler, model = load_model()

# Streamlit UI
st.title("📈 E-Commerce Sales Forecasting Web App")
st.write("Upload your sales data in CSV format and get sales predictions for the next 3 months.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file,encoding="ISO-8859-1")
    st.write("📌 Uploaded Data Preview:")
    st.write(df.head())

    if st.button("Run Prediction"):
        if model is None:
            st.error("Model failed to load. Please check your model script.")
        else:
            try:
                df = df.copy()

                # Convert InvoiceDate to datetime
                df["week"] = pd.to_datetime(df["InvoiceDate"], format="%d-%m-%Y %H:%M", errors="coerce")
                df = df.dropna(subset=["week"])  # Drop invalid dates
                df["week"] = df["week"].dt.to_period('W').apply(lambda x: x.start_time)

                # Find last available date
                last_date = df["week"].max()

                # Generate future weeks (for 3 months)
                future_weeks = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=12, freq="W")

                # Create a new dataframe for future predictions
                stock_codes = df["StockCode"].unique()
                future_df = pd.DataFrame({
                    "StockCode": np.tile(stock_codes, len(future_weeks)),
                    "week": np.repeat(future_weeks, len(stock_codes))
                })

                # Feature Engineering
                df["year"] = df["week"].dt.year
                df["month"] = df["week"].dt.month
                df["weekofyear"] = df["week"].dt.isocalendar().week
                df["dayofweek"] = df["week"].dt.dayofweek
                df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

                # Compute lag features
                df = df.sort_values(by=["StockCode", "week"])
                df["lag_1"] = df.groupby("StockCode")["Quantity"].shift(1)
                df["rolling_mean_2"] = df.groupby("StockCode")["Quantity"].transform(lambda x: x.rolling(2, min_periods=1).mean())

                # Fill NaNs
                df.fillna(method="bfill", inplace=True)

                # Apply same feature engineering to future_df
                future_df["year"] = future_df["week"].dt.year
                future_df["month"] = future_df["week"].dt.month
                future_df["weekofyear"] = future_df["week"].dt.isocalendar().week
                future_df["dayofweek"] = future_df["week"].dt.dayofweek
                future_df["is_weekend"] = (future_df["dayofweek"] >= 5).astype(int)

                # Merge with last available sales data to estimate lag values
                last_sales = df.groupby("StockCode")["Quantity"].last().reset_index()
                future_df = future_df.merge(last_sales, on="StockCode", how="left")
                future_df.rename(columns={"Quantity": "lag_1"}, inplace=True)

                # Fill missing values in lag_1 with median sales
                future_df["lag_1"].fillna(df["lag_1"].median(), inplace=True)

                # Rolling mean approximation
                rolling_means = df.groupby("StockCode")["rolling_mean_2"].last().reset_index()
                future_df = future_df.merge(rolling_means, on="StockCode", how="left")

                # Fill missing rolling_mean_2
                future_df["rolling_mean_2"].fillna(df["rolling_mean_2"].median(), inplace=True)

                # Normalize using trained scaler
                features = ["year", "month", "weekofyear", "dayofweek", "is_weekend", "lag_1", "rolling_mean_2"]
                future_df[features] = scaler.transform(future_df[features])

                # Predict future sales
                future_df["Predicted Sales"] = model.predict(future_df[features])

                # Keep only required columns
                future_df = future_df[["StockCode", "week", "Predicted Sales"]]

                # Save predictions
                output_file = "predictions.csv"
                future_df.to_csv(output_file, index=False)

                # Download button
                with open(output_file, "rb") as f:
                    st.download_button("Download Predictions", f, file_name="predictions.csv", mime="text/csv")

                st.success("✅ Predictions Generated Successfully!")

            except Exception as e:
                st.error(f"Error processing data: {e}")

