import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load Pre-trained Model and Scaler
@st.cache_resource
def load_model():
    try:
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("stacked_model.pkl")
        return scaler, model
    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {e}")
        return None, None

scaler, model = load_model()

# Streamlit UI
st.title("📈 E-Commerce Sales Forecasting Web App")
st.write("Upload your sales data in CSV format and get sales predictions for the next 3 months.")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

st.write("Tip: Make sure the file has columns in the same order as - 'InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country'.")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    st.write("📌 Uploaded Data Preview:")
    st.write(df.head())

    if st.button("▶️ Run Prediction"):
        if model is None:
            st.error("Model failed to load. Please check your model and scaler files.")
        else:
            try:
                df = df.copy()

                df["week"] = pd.to_datetime(df["InvoiceDate"], format="%d-%m-%Y %H:%M", errors="coerce")
                df = df.dropna(subset=["week"])
                df["week"] = df["week"].dt.to_period('W').apply(lambda x: x.start_time)

                last_date = df["week"].max()
                future_weeks = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=12, freq="W")
                stock_codes = df["StockCode"].unique()

                future_df = pd.DataFrame({
                    "StockCode": np.tile(stock_codes, len(future_weeks)),
                    "week": np.repeat(future_weeks, len(stock_codes))
                })

                df["year"] = df["week"].dt.year
                df["month"] = df["week"].dt.month
                df["weekofyear"] = df["week"].dt.isocalendar().week
                df["dayofweek"] = df["week"].dt.dayofweek
                df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

                df = df.sort_values(by=["StockCode", "week"])
                df["lag_1"] = df.groupby("StockCode")["Quantity"].shift(1)
                df["rolling_mean_2"] = df.groupby("StockCode")["Quantity"].transform(lambda x: x.rolling(2, min_periods=1).mean())
                df.fillna(method="bfill", inplace=True)

                future_df["year"] = future_df["week"].dt.year
                future_df["month"] = future_df["week"].dt.month
                future_df["weekofyear"] = future_df["week"].dt.isocalendar().week
                future_df["dayofweek"] = future_df["week"].dt.dayofweek
                future_df["is_weekend"] = (future_df["dayofweek"] >= 5).astype(int)

                last_sales = df.groupby("StockCode")["Quantity"].last().reset_index()
                future_df = future_df.merge(last_sales, on="StockCode", how="left")
                future_df.rename(columns={"Quantity": "lag_1"}, inplace=True)
                future_df["lag_1"].fillna(df["lag_1"].median(), inplace=True)

                rolling_means = df.groupby("StockCode")["rolling_mean_2"].last().reset_index()
                future_df = future_df.merge(rolling_means, on="StockCode", how="left")
                future_df["rolling_mean_2"].fillna(df["rolling_mean_2"].median(), inplace=True)

                features = ["year", "month", "weekofyear", "dayofweek", "is_weekend", "lag_1", "rolling_mean_2"]
                future_df[features] = scaler.transform(future_df[features])

                future_df["Predicted Sales"] = model.predict(future_df[features])

                result = future_df[["StockCode", "week", "Predicted Sales"]]
                result.to_csv("predictions.csv", index=False)

                with open("predictions.csv", "rb") as f:
                    st.download_button("⬇️ Download Predictions", f, file_name="sales_predictions.csv", mime="text/csv")

                st.success("✅ Predictions Generated Successfully!")

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
