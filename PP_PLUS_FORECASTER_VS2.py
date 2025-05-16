import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Dataset URLs
dataset_urls = {
    "LLDPE": "https://raw.githubusercontent.com/ddewetroos/PP_PLUS/main/Linear%20Low%20Density%20Polyethylene%20Futures%20Historical%20Data.csv",
    "Polypropylene": "https://raw.githubusercontent.com/ddewetroos/PP_PLUS/main/Polypropylene%20Futures%20Historical%20Data.csv",
    "PVC": "https://raw.githubusercontent.com/ddewetroos/PP_PLUS/main/Polyvinyl%20Chloride%20Futures%20Historical%20Data.csv"
}

st.title("📈 Commodity Plastics Forecasting App")
choice = st.selectbox("Select a plastic type:", list(dataset_urls.keys()))
url = dataset_urls[choice]

# Load and clean data
df = pd.read_csv(url)
df.columns = [col.strip() for col in df.columns]
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
price_col = [col for col in df.columns if col != 'Date'][0]
df[price_col] = df[price_col].astype(str).str.replace(',', '', regex=False).str.replace('$', '', regex=False).str.strip()
df['Price'] = pd.to_numeric(df[price_col], errors='coerce')
df = df.dropna(subset=['Date', 'Price'])
df = df.sort_values("Date")
df = df.rename(columns={"Date": "ds", "Price": "y"})

# Historical price chart
st.subheader("📊 Historical Price Chart")
st.line_chart(df.set_index("ds")["y"])

# Summary stats
six_months_ago = df["ds"].max() - pd.DateOffset(months=6)
recent_df = df[df["ds"] >= six_months_ago]
st.subheader("🔍 Summary Stats (Last 6 Months)")
summary_stats = recent_df["y"].describe()[["min", "max", "mean", "std"]].round(2)
st.write(summary_stats)

# Forecasting methods
def forecast_prophet(train_df, future_periods):
    model = Prophet(daily_seasonality=False)
    model.fit(train_df)
    future = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future)
    return model, forecast

def forecast_arima(train_df, test_df, order=(5,1,0)):
    ts = train_df.set_index("ds")["y"]
    model = ARIMA(ts, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test_df))
    return pd.DataFrame({"ds": test_df["ds"].values, "yhat": forecast})

def forecast_sarima(train_df, test_df, order=(1,1,1), seasonal_order=(1,1,1,12)):
    ts = train_df.set_index("ds")["y"]
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(test_df))
    return pd.DataFrame({"ds": test_df["ds"].values, "yhat": forecast})

def forecast_rf(train_df, test_df, lags=10):
    df_all = pd.concat([train_df, test_df]).copy()
    for i in range(1, lags + 1):
        df_all[f"lag_{i}"] = df_all["y"].shift(i)
    df_all.dropna(inplace=True)

    features = [f"lag_{i}" for i in range(1, lags + 1)]
    split_idx = train_df.index[-1]

    X_train = df_all[df_all.index <= split_idx][features]
    y_train = df_all[df_all.index <= split_idx]["y"]
    X_test = df_all[df_all.index > split_idx][features]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    return pd.DataFrame({"ds": test_df["ds"].values[:len(yhat)], "yhat": yhat})

def forecast_lstm(train_df, test_df, n_input=30):
    data = pd.concat([train_df, test_df])["y"].values
    mu, sigma = np.mean(data[:len(train_df)]), np.std(data[:len(train_df)])
    scaled = (data - mu) / sigma

    X, y = [], []
    for i in range(n_input, len(train_df)):
        X.append(scaled[i - n_input:i])
        y.append(scaled[i])
    X = np.array(X).reshape(-1, n_input, 1)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, verbose=0)

    preds = []
    input_seq = scaled[len(train_df)-n_input:len(train_df)]
    for _ in range(len(test_df)):
        x_input = input_seq[-n_input:].reshape(1, n_input, 1)
        yhat = model.predict(x_input, verbose=0)
        preds.append(yhat[0][0])
        input_seq = np.append(input_seq, yhat[0][0])

    yhat_rescaled = np.array(preds) * sigma + mu
    return pd.DataFrame({"ds": test_df["ds"].values[:len(yhat_rescaled)], "yhat": yhat_rescaled})

# Consensus Forecast
st.subheader("🤝 Consensus Forecast (Average of All Models)")

future_dates = pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=180)

prophet_fc = forecast_prophet(df, 180)[1][["ds", "yhat"]]
arima_fc = forecast_arima(df, pd.DataFrame({"ds": future_dates}))
sarima_fc = forecast_sarima(df, pd.DataFrame({"ds": future_dates}))
rf_fc = forecast_rf(df, pd.DataFrame({"ds": future_dates}))
lstm_fc = forecast_lstm(df, pd.DataFrame({"ds": future_dates}))

forecasts_merged = prophet_fc.rename(columns={"yhat": "Prophet"}).merge(
    arima_fc.rename(columns={"yhat": "ARIMA"}), on="ds", how="outer"
).merge(
    sarima_fc.rename(columns={"yhat": "SARIMA"}), on="ds", how="outer"
).merge(
    rf_fc.rename(columns={"yhat": "RandomForest"}), on="ds", how="outer"
).merge(
    lstm_fc.rename(columns={"yhat": "LSTM"}), on="ds", how="outer"
)

forecasts_merged["Consensus"] = forecasts_merged[["Prophet", "ARIMA", "SARIMA", "RandomForest", "LSTM"]].mean(axis=1)

# Plot Consensus Forecast
fig_consensus, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["ds"], df["y"], label="Historical", linewidth=2)
ax.plot(forecasts_merged["ds"], forecasts_merged["Consensus"], label="Consensus Forecast", linestyle="--")
ax.set_title("6-Month Consensus Forecast (Average of All Models)")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig_consensus)

# Download Button
def convert_df(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')

st.download_button("📥 Download Consensus Forecast", convert_df(forecasts_merged[["ds", "Consensus"]]), f"{choice}_consensus_forecast.csv", "text/csv")
