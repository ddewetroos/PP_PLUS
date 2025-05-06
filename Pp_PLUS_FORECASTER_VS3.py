import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
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
model_choice = st.selectbox("Select forecasting model:", ["Prophet", "ARIMA", "SARIMA", "Random Forest", "LSTM"])
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

# Split data for back-prediction
cutoff_date = df["ds"].max() - pd.DateOffset(months=12)
train_df = df[df["ds"] <= cutoff_date]
test_df = df[df["ds"] > cutoff_date]

# Forecasting methods
def forecast_prophet(train_df, future_periods):
    model = Prophet(daily_seasonality=False)
    model.fit(train_df)
    future = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future)
    return model, forecast

def forecast_arima(train_df, steps):
    ts = train_df.set_index("ds")["y"]
    model = ARIMA(ts, order=(5,1,0)).fit()
    forecast_values = model.forecast(steps=steps)
    last_date = train_df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    return pd.DataFrame({"ds": future_dates, "yhat": forecast_values})

def forecast_sarima(train_df, steps):
    ts = train_df.set_index("ds")["y"]
    model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    forecast_values = model.forecast(steps=steps)
    last_date = train_df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    return pd.DataFrame({"ds": future_dates, "yhat": forecast_values})

def forecast_rf_future(train_df, steps=180, lags=10):
    df_all = train_df.copy()
    for i in range(1, lags + 1):
        df_all[f"lag_{i}"] = df_all["y"].shift(i)
    df_all.dropna(inplace=True)

    X_train = df_all[[f"lag_{i}" for i in range(1, lags + 1)]]
    y_train = df_all["y"]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    last_known = df_all.iloc[-1][[f"lag_{i}" for i in range(1, lags + 1)]].values
    preds, dates = [], []
    current_input = list(last_known)

    for i in range(steps):
        pred = model.predict([current_input[-lags:]])[0]
        preds.append(pred)
        current_input.append(pred)
        dates.append(train_df["ds"].max() + pd.Timedelta(days=i+1))

    return pd.DataFrame({"ds": dates, "yhat": preds})

def forecast_lstm_future(train_df, steps=180, n_input=30):
    data = train_df["y"].values
    mu, sigma = np.mean(data), np.std(data)
    scaled = (data - mu) / sigma

    X, y = [], []
    for i in range(n_input, len(scaled)):
        X.append(scaled[i - n_input:i])
        y.append(scaled[i])
    X = np.array(X).reshape(-1, n_input, 1)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, verbose=0)

    forecast_scaled = []
    input_seq = scaled[-n_input:]
    for _ in range(steps):
        x_input = input_seq[-n_input:].reshape(1, n_input, 1)
        yhat = model.predict(x_input, verbose=0)
        forecast_scaled.append(yhat[0][0])
        input_seq = np.append(input_seq, yhat[0][0])

    yhat_rescaled = np.array(forecast_scaled) * sigma + mu
    dates = [train_df["ds"].max() + pd.Timedelta(days=i+1) for i in range(steps)]
    return pd.DataFrame({"ds": dates, "yhat": yhat_rescaled})

# 🔮 Forecast the next 6 months
st.subheader(f"📈 Forecast for Next 6 Months ({model_choice})")
forecast_6mo = None

if model_choice == "Prophet":
    model_future, forecast = forecast_prophet(df, 180)
    forecast_6mo = forecast.tail(180)
elif model_choice == "ARIMA":
    forecast_6mo = forecast_arima(df, 180)
elif model_choice == "SARIMA":
    forecast_6mo = forecast_sarima(df, 180)
elif model_choice == "Random Forest":
    forecast_6mo = forecast_rf_future(df, 180)
elif model_choice == "LSTM":
    forecast_6mo = forecast_lstm_future(df, 180)

if forecast_6mo is not None:
    fig_fcast, ax_fcast = plt.subplots(figsize=(10, 4))
    ax_fcast.plot(df["ds"], df["y"], label="Historical")
    ax_fcast.plot(forecast_6mo["ds"], forecast_6mo["yhat"], label="Forecast", linestyle="--")
    ax_fcast.set_title(f"6-Month Forward Forecast - {model_choice}")
    ax_fcast.set_xlabel("Date")
    ax_fcast.set_ylabel("Price")
    ax_fcast.legend()
    st.pyplot(fig_fcast)

# ⏪ Back-prediction
st.subheader("⏪ Model Fit on Last 12 Months (Holdout Evaluation)")
if model_choice == "Prophet":
    _, backcast = forecast_prophet(train_df, len(test_df))
    backcast = backcast[backcast["ds"].isin(test_df["ds"])][["ds", "yhat"]]
elif model_choice == "ARIMA":
    backcast = forecast_arima(train_df, len(test_df))
elif model_choice == "SARIMA":
    backcast = forecast_sarima(train_df, len(test_df))
elif model_choice == "Random Forest":
    backcast = forecast_rf_future(train_df, len(test_df))
elif model_choice == "LSTM":
    backcast = forecast_lstm_future(train_df, len(test_df))
else:
    st.error("Unsupported model selected.")
    st.stop()

# Compare
compare_df = test_df.merge(backcast, on="ds", how="left")
compare_df["error"] = compare_df["y"] - compare_df["yhat"]
mae = round(compare_df["error"].abs().mean(), 2)

fig_bt, ax = plt.subplots(figsize=(10, 4))
ax.plot(compare_df["ds"], compare_df["y"], label="Actual", linewidth=2)
ax.plot(compare_df["ds"], compare_df["yhat"], label="Predicted", linestyle="--")
ax.set_title(f"Model Back-Test: Last 12 Months ({model_choice})")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig_bt)

st.write(f"📉 Mean Absolute Error over last 12 months: {mae}")
