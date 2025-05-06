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
from datetime import datetime

st.title("📈 Multi-Model Forecasting for Commodity Plastics")

# Dataset URLs
dataset_urls = {
    "LLDPE": "https://raw.githubusercontent.com/ddewetroos/PP_PLUS/main/Linear%20Low%20Density%20Polyethylene%20Futures%20Historical%20Data.csv",
    "Polypropylene": "https://raw.githubusercontent.com/ddewetroos/PP_PLUS/main/Polypropylene%20Futures%20Historical%20Data.csv",
    "PVC": "https://raw.githubusercontent.com/ddewetroos/PP_PLUS/main/Polyvinyl%20Chloride%20Futures%20Historical%20Data.csv"
}

# Model choices
model_options = ["Prophet", "ARIMA", "SARIMA", "Random Forest", "LSTM"]

# UI: select dataset and model
dataset_choice = st.selectbox("Choose Dataset", list(dataset_urls.keys()))
model_choice = st.selectbox("Choose Forecasting Model", model_options)
url = dataset_urls[dataset_choice]

# Load and preprocess
df = pd.read_csv(url)
df.columns = [col.strip() for col in df.columns]
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
price_col = [col for col in df.columns if col != 'Date'][0]
df[price_col] = df[price_col].astype(str).str.replace(',', '').str.replace('$', '').str.strip()
df['Price'] = pd.to_numeric(df[price_col], errors='coerce')
df = df.dropna(subset=['Date', 'Price']).sort_values("Date")
df = df[["Date", "Price"]].rename(columns={"Date": "ds", "Price": "y"})

# Split
cutoff = df["ds"].max() - pd.DateOffset(months=12)
train_df = df[df["ds"] <= cutoff].copy()
test_df = df[df["ds"] > cutoff].copy()

# Forecast function
def forecast_prophet(train_df, test_df):
    model = Prophet(daily_seasonality=False)
    model.fit(train_df)
    future = model.make_future_dataframe(periods=len(test_df), freq='D')
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat"]].tail(len(test_df))
    return forecast

def forecast_arima(train_df, test_df, order=(5,1,0)):
    ts = train_df.set_index("ds")["y"]
    model = ARIMA(ts, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test_df))
    return pd.DataFrame({"ds": test_df["ds"].values, "yhat": forecast.values})

def forecast_sarima(train_df, test_df, order=(1,1,1), seasonal_order=(1,1,1,12)):
    ts = train_df.set_index("ds")["y"]
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(test_df))
    return pd.DataFrame({"ds": test_df["ds"].values, "yhat": forecast.values})

def forecast_rf(train_df, test_df, lags=10):
    df_all = pd.concat([train_df, test_df]).copy()
    for lag in range(1, lags+1):
        df_all[f"lag_{lag}"] = df_all["y"].shift(lag)
    df_all.dropna(inplace=True)

    X = df_all[[f"lag_{lag}" for lag in range(1, lags+1)]]
    y = df_all["y"]

    X_train = X.loc[train_df.index[-(len(X)):]]
    y_train = y.loc[train_df.index[-(len(X)):]]
    X_test = X.loc[test_df.index.intersection(X.index)]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    return pd.DataFrame({"ds": test_df["ds"].values[:len(yhat)], "yhat": yhat})

def forecast_lstm(train_df, test_df, n_input=30):
    data = pd.concat([train_df, test_df]).set_index("ds")["y"].values
    scaler = lambda x: (x - np.mean(x)) / np.std(x)
    inv_scaler = lambda x, mu, sigma: x * sigma + mu
    mu, sigma = np.mean(data[:len(train_df)]), np.std(data[:len(train_df)])
    scaled_data = scaler(data)

    X, y = [], []
    for i in range(n_input, len(train_df)):
        X.append(scaled_data[i - n_input:i])
        y.append(scaled_data[i])
    X = np.array(X).reshape(-1, n_input, 1)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, verbose=0)

    # Predict last 12 months
    preds = []
    input_seq = scaled_data[len(train_df)-n_input:len(train_df)]
    for _ in range(len(test_df)):
        x_input = input_seq[-n_input:].reshape(1, n_input, 1)
        yhat = model.predict(x_input, verbose=0)
        preds.append(inv_scaler(yhat[0][0], mu, sigma))
        input_seq = np.append(input_seq, yhat[0][0])

    return pd.DataFrame({"ds": test_df["ds"].values[:len(preds)], "yhat": preds})

# Run forecast
st.subheader(f"🔮 Forecast using {model_choice}")
if model_choice == "Prophet":
    forecast = forecast_prophet(train_df, test_df)
elif model_choice == "ARIMA":
    forecast = forecast_arima(train_df, test_df)
elif model_choice == "SARIMA":
    forecast = forecast_sarima(train_df, test_df)
elif model_choice == "Random Forest":
    forecast = forecast_rf(train_df, test_df)
elif model_choice == "LSTM":
    forecast = forecast_lstm(train_df, test_df)
else:
    st.error("Invalid model selected.")
    st.stop()

# Merge and evaluate
result = test_df.merge(forecast, on="ds", how="left")
result.dropna(inplace=True)
mae = mean_absolute_error(result["y"], result["yhat"])

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(result["ds"], result["y"], label="Actual", linewidth=2)
ax.plot(result["ds"], result["yhat"], label="Predicted", linestyle="--")
ax.set_title(f"Back-Test (Last 12 Months) - {model_choice}")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Show error
st.write(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
