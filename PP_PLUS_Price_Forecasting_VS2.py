import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="LLDPE Price Forecast", layout="wide")

st.title("📈 LLDPE Price Forecast Tool")
st.markdown("Upload a CSV file containing LLDPE pricing data with columns **Date** and **Price**.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Check required columns
        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.error("❌ The uploaded file must contain 'Date' and 'Price' columns.")
        else:
            # Parse and clean
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Price'] = df['Price'].astype(str).str.replace(",", "").astype(float)
            df = df.dropna(subset=['Date', 'Price'])
            df = df.sort_values('Date')
            df_prophet = df.rename(columns={'Date': 'ds', 'Price': 'y'})

            # Show preview
            st.success("✅ Data successfully uploaded and validated.")
            st.dataframe(df.head())

            # Forecast
            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)

            # Future to 2025
            periods = (pd.Timestamp("2025-12-31") - df_prophet['ds'].max()).days
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

            # Plot forecast
            fig = model.plot(forecast)
            plt.title("Forecasted LLDPE Prices")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            st.pyplot(fig)

            # Download option
            forecast_out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            csv = forecast_out.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Forecast CSV", csv, "LLDPE_Forecast.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

else:
    st.info("📤 Please upload a CSV file to begin.")
