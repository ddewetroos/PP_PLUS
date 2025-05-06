import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime

# Dataset URLs
dataset_urls = {
    "LLDPE": "https://raw.githubusercontent.com/ddewetroos/PP_PLUS/main/Linear%20Low%20Density%20Polyethylene%20Futures%20Historical%20Data.csv",
    "Polypropylene": "https://raw.githubusercontent.com/ddewetroos/PP_PLUS/main/Polypropylene%20Futures%20Historical%20Data.csv",
    "PVC": "https://raw.githubusercontent.com/ddewetroos/PP_PLUS/main/Polyvinyl%20Chloride%20Futures%20Historical%20Data.csv"
}

# Title
st.title("📈 Commodity Plastics Forecasting App")

# Dataset selection
choice = st.selectbox("Select a plastic type:", list(dataset_urls.keys()))
url = dataset_urls[choice]

# Load and clean data
df = pd.read_csv(url)
df.columns = [col.strip() for col in df.columns]

# Parse and clean
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
price_col = [col for col in df.columns if col != 'Date'][0]
df[price_col] = df[price_col].astype(str).str.replace(',', '').str.replace('$', '').str.strip()
df['Price'] = pd.to_numeric(df[price_col], errors='coerce')
df = df.dropna(subset=['Date', 'Price'])
df = df.sort_values("Date")

# Preview chart
st.subheader("📊 Historical Price Chart")
st.line_chart(df.set_index("Date")["Price"])

# Summary stats
six_months_ago = df["Date"].max() - pd.DateOffset(months=6)
recent_df = df[df["Date"] >= six_months_ago]
st.subheader("🔍 Summary Stats (Last 6 Months)")
summary_stats = recent_df["Price"].describe()[["min", "max", "mean", "std"]].round(2)
st.write(summary_stats)

# Forecast with Prophet
st.subheader("🔮 6-Month Forecast (Prophet)")
df_prophet = df.rename(columns={"Date": "ds", "Price": "y"})
model = Prophet(daily_seasonality=False)
model.fit(df_prophet)
future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)
fig = model.plot(forecast)
st.pyplot(fig)

# True back-prediction using last 12 months as test set
st.subheader("⏪ Model Fit on Last 12 Months (Holdout Evaluation)")

# 12-month holdout split
cutoff_date = df_prophet["ds"].max() - pd.DateOffset(months=12)
train_df = df_prophet[df_prophet["ds"] <= cutoff_date]
test_df = df_prophet[df_prophet["ds"] > cutoff_date]

# Train model on data up to 12 months ago
model_bt = Prophet(daily_seasonality=False)
model_bt.fit(train_df)

# Forecast only the next N days (equal to test set length)
future_bt = model_bt.make_future_dataframe(periods=len(test_df), freq='D')
forecast_bt = model_bt.predict(future_bt)

# Extract only forecast overlapping with test data
forecast_trimmed = forecast_bt[forecast_bt["ds"].isin(test_df["ds"])]

# Merge actual and predicted
compare_df = test_df.merge(forecast_trimmed[["ds", "yhat"]], on="ds", how="left")
compare_df["error"] = compare_df["y"] - compare_df["yhat"]
mae = round(compare_df["error"].abs().mean(), 2)

# Plot prediction vs actual
fig_bt, ax = plt.subplots(figsize=(10, 4))
ax.plot(compare_df["ds"], compare_df["y"], label="Actual", linewidth=2)
ax.plot(compare_df["ds"], compare_df["yhat"], label="Predicted", linestyle="--")
ax.set_title("Model Back-Test: Last 12 Months")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig_bt)

# Show MAE
st.write(f"📉 Mean Absolute Error over last 12 months: {mae}")
