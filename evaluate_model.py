# %%
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Load Test Data ===
def load_test_data(filepath):
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed", errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df.rename(columns={"InvoiceDate": "date", "Quantity": "sales"}, inplace=True)
    df["date"] = df["date"].dt.date
    return df.sort_values(by="date")

# === Preprocess Test Data ===
def preprocess(df):
    daily_sales = df.groupby(["date", "StockCode"]).agg({"sales": "sum"}).reset_index()
    daily_sales["week"] = pd.to_datetime(daily_sales["date"]).dt.to_period("W")
    weekly = daily_sales.groupby(["week", "StockCode"]).agg({"sales": "sum"}).reset_index()
    weekly["week"] = weekly["week"].apply(lambda x: x.start_time)

    weekly["year"] = weekly["week"].dt.year
    weekly["month"] = weekly["week"].dt.month
    weekly["weekofyear"] = weekly["week"].dt.isocalendar().week
    weekly["dayofweek"] = weekly["week"].dt.dayofweek
    weekly["is_weekend"] = (weekly["dayofweek"] >= 5).astype(int)

    weekly["lag_1"] = weekly.groupby("StockCode")["sales"].shift(1)
    weekly["rolling_mean_2"] = weekly.groupby("StockCode")["sales"].transform(lambda x: x.rolling(2, min_periods=1).mean())
    weekly["rolling_std_2"] = weekly.groupby("StockCode")["sales"].transform(lambda x: x.rolling(2, min_periods=1).std())

    weekly["rolling_std_2"].fillna(0, inplace=True)
    weekly.bfill(inplace=True)
    return weekly

# === Evaluation ===
def evaluate(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    accuracy = max(0, (1 - (mae / true.mean())) * 100)
    print(f"\n📊 Evaluation Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Accuracy (1 - MAE/mean): {accuracy:.2f}%")

# === Main ===
if __name__ == "__main__":
    # Load test data
    df = load_test_data("test_data.csv")
    test_df = preprocess(df)

    # Load model and scaler
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("stacked_model.pkl")

    features = ["year", "month", "weekofyear", "dayofweek", "is_weekend", "lag_1", "rolling_mean_2", "rolling_std_2"]
    X_test = test_df[features]
    y_test = test_df["sales"]

    # Scale features and predict
    X_scaled = scaler.transform(X_test)
    y_pred = np.expm1(model.predict(X_scaled))

    # Evaluate
    evaluate(y_test, y_pred)


import matplotlib.pyplot as plt
# Optional: reset index if working from CSV
test_df = test_df.reset_index(drop=True)
plt.figure(figsize=(14, 6))
plt.plot(y_test.values, label='Actual Sales', linewidth=2, color='blue', alpha=0.7, linestyle='-')
plt.plot(y_pred, label='Predicted Sales', linewidth=2, color='orange', alpha=0.7, linestyle='--', marker='o', markersize=3)

plt.title('📊 Actual vs Predicted Weekly Sales')
plt.xlabel('Sample Index')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('🔎 Predicted vs Actual Sales (Scatter Plot)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
