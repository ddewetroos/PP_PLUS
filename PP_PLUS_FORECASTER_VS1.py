import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime

#st.set_option('deprecation.showPyplotGlobalUse', False)

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

# Back-prediction
st.subheader("⏪ Back-Prediction Accuracy (Last 6 Months)")
cutoff = df_prophet["ds"].max() - pd.DateOffset(months=6)
train_df = df_prophet[df_prophet["ds"] <= cutoff]
test_df = df_prophet[df_prophet["ds"] > cutoff]

model_back = Prophet(daily_seasonality=False)
model_back.fit(train_df)
future_back = model_back.make_future_dataframe(periods=len(test_df))
forecast_back = model_back.predict(future_back)

# Join forecast with actual
comparison = test_df.merge(forecast_back[["ds", "yhat"]], on="ds", how="left")
comparison["error"] = comparison["y"] - comparison["yhat"]
mae = round(comparison["error"].abs().mean(), 2)
st.write(f"📉 Mean Absolute Error: {mae}")
